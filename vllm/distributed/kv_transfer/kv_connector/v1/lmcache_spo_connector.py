# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

_WRITE_THROUGH = "write_through"
_SPILL_OVER = "spill_over"


@dataclass
class _PendingSpec:
    """KV state snapshot captured at wait_for_save time.

    Updated (overwritten) every forward step so it always reflects the latest
    computed tokens.
    """

    token_ids: list[int]
    slot_mapping: torch.Tensor       # CPU tensor; moved to device at flush time
    is_last_prefill: bool
    skip_leading_tokens: int         # from save_spec; avoids re-storing LMCache-hit prefix
    request_configs: Optional[dict]  # for tag-based key generation (lmcache.tag.*)


class LMCacheSPOConnector(LMCacheConnectorV1):
    """LMCache KV connector with spill-over (SPO) write-mode support.

    Two modes, selected via kv_transfer_config extra_config ``write_mode``:

    ``write_through`` (default)
        Identical to :class:`LMCacheConnectorV1`.

    ``spill_over``
        KV is saved to SSD *only* when a prefix-cache block holding the KV
        is about to be evicted from HBM and reused.

    The spill-over path:

    * ``wait_for_save`` — snapshots each running request into ``_pending_store``.
      Snapshots persist until the request's blocks are evicted.  Finished
      requests are NOT eagerly removed; their entries survive until
      ``_flush_evicted`` handles the actual block eviction.

    * ``handle_block_evictions`` (called BEFORE the forward pass that would
      overwrite evicted blocks) —
      Scans ``_pending_store`` for any request whose blocks overlap with
      the evicted set and calls ``lmcache_engine.store()`` while HBM data
      is still valid.  Cleanup happens inside ``_flush_evicted`` only after
      a successful store, not eagerly on ``finished_req_ids``.

    Configuration example::

        kv_transfer_config:
          kv_connector: LMCacheSPOConnector
          kv_connector_extra_config:
            write_mode: spill_over
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        write_mode = vllm_config.kv_transfer_config.get_from_extra_config(
            "write_mode", _WRITE_THROUGH
        )
        if write_mode not in (_WRITE_THROUGH, _SPILL_OVER):
            raise ValueError(
                f"Invalid write_mode={write_mode!r}. "
                f"Choose '{_WRITE_THROUGH}' or '{_SPILL_OVER}'."
            )

        self._is_spo = write_mode == _SPILL_OVER

        if self._is_spo and role == KVConnectorRole.WORKER:
            if getattr(self._lmcache_engine, "use_layerwise", False):
                logger.warning(
                    "SPO mode is incompatible with use_layerwise=True; "
                    "falling back to write-through mode."
                )
                self._is_spo = False

        logger.info(
            "LMCacheSPOConnector initialised in %s mode",
            "spill-over" if self._is_spo else "write-through",
        )

        # All requests (running, preempted, finished) linger here until their
        # blocks are evicted or explicitly cleaned up via finished_req_ids.
        # Overwritten each step via _capture_pending for running requests.
        self._pending_store: dict[str, _PendingSpec] = {}
        # Hash of the first stored chunk per successful flush; prevents storing
        # the same prefix content twice (e.g., concurrent requests sharing a
        # system prompt that are each evicted independently).
        self._stored_chunk_sigs: set[int] = set()

    # ──────────────────────────────────────────────────────────────
    # Worker-side overrides
    # ──────────────────────────────────────────────────────────────

    def handle_block_evictions(
        self, evicted_block_ids: set[int], finished_req_ids: set[str]
    ) -> None:
        """Flush KV to SSD for blocks about to be overwritten.

        Called BEFORE the forward pass so HBM data is still valid.

        ``_pending_store`` entries are removed only inside ``_flush_evicted``
        after the KV has been stored.  We do NOT eagerly clean up
        ``finished_req_ids`` here because a request can finish in the same
        scheduling step that evicts *other* requests' blocks; removing the
        entry at that point would prevent us from storing KV when the
        finished request's own blocks are evicted in a later step.
        """
        if not self._is_spo:
            return
        if evicted_block_ids:
            self._flush_evicted(evicted_block_ids)

    def wait_for_save(self) -> None:
        """SPO: snapshot running requests without writing to SSD.
        Write-through: delegate to parent.
        """
        if not self._is_spo:
            super().wait_for_save()
            return
        self._capture_pending()

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        # Keep _pending_store entry alive until the request's blocks are
        # evicted and stored by _flush_evicted.
        return super().request_finished(request, block_ids)

    # ──────────────────────────────────────────────────────────────
    # SPO internals
    # ──────────────────────────────────────────────────────────────

    def _capture_pending(self) -> None:
        """Snapshot every running request and release lookup pins."""
        from lmcache.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorMetadata,
        )

        impl = self._lmcache_engine
        engine = impl.lmcache_engine
        if engine is None:
            return

        meta = self._get_connector_metadata()
        if not isinstance(meta, LMCacheConnectorMetadata):
            return

        for req in meta.requests:
            engine.lookup_unpin(req.req_id)

            save_spec = req.save_spec
            if save_spec is None or not save_spec.can_save:
                continue

            token_ids = req.token_ids
            if save_spec.skip_leading_tokens >= len(token_ids):
                continue

            # Preserve the minimum skip seen across steps so that multi-chunk
            # prefill is stored in full at eviction time.
            # tracker.num_saved_tokens advances optimistically each step
            # (assuming write-through), which would cause save_spec.skip to
            # grow and lose earlier chunks. Taking the min keeps the initial
            # lookup-hit-based skip (tokens already in LMCache) while ignoring
            # the optimistic advancement.
            existing = self._pending_store.get(req.req_id)
            skip = save_spec.skip_leading_tokens
            if existing is not None:
                skip = min(existing.skip_leading_tokens, skip)

            self._pending_store[req.req_id] = _PendingSpec(
                token_ids=token_ids,
                slot_mapping=req.slot_mapping,
                is_last_prefill=req.is_last_prefill,
                skip_leading_tokens=skip,
                request_configs=getattr(req, "request_configs", None),
            )
            logger.debug("SPO capture: req=%s tokens=%d skip=%d", req.req_id, len(token_ids), skip)

    def _flush_evicted(self, evicted_block_ids: set[int]) -> None:
        """Store KV for any request whose blocks overlap with evicted_block_ids.

        Running requests' blocks have ref_cnt > 0 and cannot be evicted, so
        they produce no false positives in the scan.

        The skip/mask logic mirrors LMCacheConnectorV1Impl.wait_for_save():
        - skip_leading_tokens (chunk-aligned) covers tokens already in LMCache
          from a prior hit, avoiding redundant re-stores.
        - request_configs is forwarded so tag-based cache keys match lookups.

        Safety: engine.store() is synchronous with respect to GPU memory for
        CPU-target (SSD offloading) connectors — LMCache calls
        store_stream.synchronize() inside gpu_connector.from_gpu() before
        returning. By the time this method returns, GPU KV data has been copied
        to CPU and the forward pass may safely overwrite the evicted blocks.
        """
        impl = self._lmcache_engine
        engine = impl.lmcache_engine
        if engine is None:
            return
        if not impl.kv_caches:
            logger.warning("SPO eviction: kv_caches not yet registered — skipping")
            return

        kvcaches = list(impl.kv_caches.values())
        chunk_size = impl._lmcache_chunk_size
        block_size = self._vllm_config.cache_config.block_size

        triggered: dict[str, _PendingSpec] = {}
        for req_id, spec in list(self._pending_store.items()):
            spec_blocks = {
                int(b)
                for b in torch.unique(spec.slot_mapping // block_size).tolist()
            }
            if spec_blocks & evicted_block_ids:
                triggered[req_id] = spec

        for req_id, spec in triggered.items():
            token_ids = spec.token_ids
            slot_mapping = spec.slot_mapping.to(impl.device)

            # Chunk-align skip (mirrors wait_for_save lines 1139-1142).
            skip = spec.skip_leading_tokens // chunk_size * chunk_size

            if skip >= len(token_ids):
                self._cleanup_req(req_id)
                logger.debug(
                    "SPO eviction: req=%s fully covered by prior cache hit — skip",
                    req_id,
                )
                continue

            # Chunk-align token sequence for intermediate prefill steps
            # (mirrors wait_for_save lines 1162-1169).
            if not spec.is_last_prefill and not getattr(
                impl, "enable_blending", False
            ):
                aligned_len = len(token_ids) // chunk_size * chunk_size
                token_ids = token_ids[:aligned_len]
                slot_mapping = slot_mapping[:aligned_len]

            if skip >= len(token_ids):
                self._cleanup_req(req_id)
                continue

            # Content-level dedup: skip if the first stored chunk was already
            # flushed this session (handles concurrent requests sharing the
            # same prefix and cross-step re-eviction of the same content).
            chunk_sig = hash(tuple(token_ids[skip:skip + chunk_size]))
            if chunk_sig in self._stored_chunk_sigs:
                logger.debug(
                    "SPO dedup: req=%s first-chunk already stored — skip", req_id
                )
                self._cleanup_req(req_id)
                continue

            store_mask = torch.ones(len(token_ids), dtype=torch.bool)
            store_mask[:skip] = False

            logger.info(
                "SPO eviction: req=%s storing %d tokens (skip=%d) → SSD",
                req_id,
                len(token_ids) - skip,
                skip,
            )
            try:
                engine.store(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    offset=skip,
                    request_configs=spec.request_configs,
                    req_id=req_id,
                )
                self._stored_chunk_sigs.add(chunk_sig)
            except Exception:
                logger.warning(
                    "SPO eviction: store raised for req=%s", req_id, exc_info=True
                )
            finally:
                self._cleanup_req(req_id)

    def _cleanup_req(self, req_id: str) -> None:
        """Remove tracking state for a request."""
        self._pending_store.pop(req_id, None)
