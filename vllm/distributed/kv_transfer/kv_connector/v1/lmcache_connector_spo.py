# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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

    Stored per-request and flushed to SSD only when the request is preempted.
    Updated (overwritten) every forward step so it always reflects the latest
    computed tokens.

    Note: ``skip_leading_tokens`` is intentionally NOT stored here.  The
    scheduler's ``from_request_tracker`` updates ``num_saved_tokens`` as if
    write-through saves had occurred, so its value is wrong for SPO mode
    (SPO never saves until preemption).  The actual number of tokens stored
    by SPO is tracked separately in ``LMCacheSPOConnector._spo_stored``.
    """

    token_ids: list[int]
    slot_mapping: torch.Tensor  # CPU tensor; moved to device at flush time
    is_last_prefill: bool


class LMCacheSPOConnector(LMCacheConnectorV1):
    """LMCache KV connector with spill-over (SPO) write-mode support.

    Two modes, selected via kv_transfer_config extra_config ``write_mode``:

    ``write_through`` (default)
        Identical to :class:`LMCacheConnectorV1` — KV is saved to SSD after
        every forward step.

    ``spill_over``
        KV is saved to SSD *only* when HBM pressure causes a request to be
        preempted.  Running requests that complete normally are never written
        to SSD, which avoids unnecessary disk I/O for short-lived requests.

    The spill-over path works as follows:

    * ``wait_for_save`` (called after every forward pass) — captures each
      request's token IDs and slot-mapping into ``_pending_store`` without
      touching the SSD.  The entry is overwritten each step so it always
      contains the full up-to-date KV location.

    * ``handle_preemptions`` (called at the *start* of the next step, before
      any blocks are overwritten) — for each preempted request, reads the
      pending entry and calls ``lmcache_engine.store()`` while the HBM data
      is still valid.

    * ``request_finished`` — discards the pending entry without storing
      (request completed normally; nothing to save).

    Load / lookup behaviour is inherited unchanged from
    :class:`LMCacheConnectorV1`.

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

        # store_layer() is a generator that must run inside the forward-pass
        # context, so layerwise mode cannot be deferred to handle_preemptions.
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

        # req_id → latest KV snapshot; overwritten each step, popped on
        # preemption or request completion.
        self._pending_store: dict[str, _PendingSpec] = {}

        # req_id → number of tokens SPO has *actually* stored to SSD so far.
        # This is separate from the scheduler's num_saved_tokens, which
        # incorrectly assumes write-through semantics and would cause SPO to
        # skip prefix chunks that were never actually stored.
        self._spo_stored: dict[str, int] = {}

    # ──────────────────────────────────────────────────────────────
    # Worker-side overrides
    # ──────────────────────────────────────────────────────────────

    def handle_preemptions(self, preempted_req_ids: set[str]) -> None:
        """Flush pending KV to SSD for every preempted request.

        Called *before* ``_update_states``, so the blocks still contain the
        correct KV data written during the previous forward pass.
        """
        if not self._is_spo or not preempted_req_ids:
            return
        self._flush_preempted(preempted_req_ids)

    def wait_for_save(self) -> None:
        """SPO mode: snapshot each request's KV location without writing to SSD.
        Write-through mode: delegate to the parent implementation.
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
        # Request finished normally — discard snapshots without storing.
        if self._is_spo:
            self._pending_store.pop(request.request_id, None)
            self._spo_stored.pop(request.request_id, None)
        return super().request_finished(request, block_ids)

    # ──────────────────────────────────────────────────────────────
    # SPO internals
    # ──────────────────────────────────────────────────────────────

    def _capture_pending(self) -> None:
        """Snapshot every saveable request into ``_pending_store``.

        Also calls ``lookup_unpin`` to release entries pinned during lookup,
        mirroring what the write-through ``wait_for_save`` does.
        """
        # Lazy import: lmcache may not be installed in all environments.
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
            # Always unpin — mirrors write-through wait_for_save behaviour.
            engine.lookup_unpin(req.req_id)

            save_spec = req.save_spec
            if save_spec is None or not save_spec.can_save:
                continue

            token_ids = req.token_ids
            if save_spec.skip_leading_tokens >= len(token_ids):
                continue

            # Overwrite any prior entry — we always want the latest snapshot
            # (more tokens than the previous step).
            self._pending_store[req.req_id] = _PendingSpec(
                token_ids=token_ids,
                slot_mapping=req.slot_mapping,  # CPU tensor, fresh each step
                is_last_prefill=req.is_last_prefill,
            )
            logger.debug(
                "SPO capture: req=%s tokens=%d",
                req.req_id,
                len(token_ids),
            )

    def _flush_preempted(self, preempted_req_ids: set[str]) -> None:
        """Call ``lmcache_engine.store()`` for each preempted request.

        Must be called while the paged KV buffers still hold the data written
        during the previous forward pass (i.e. before ``_update_states``).
        """
        impl = self._lmcache_engine
        engine = impl.lmcache_engine
        if engine is None:
            return

        if not impl.kv_caches:
            logger.warning("SPO flush: kv_caches not yet registered — skipping")
            return

        kvcaches = list(impl.kv_caches.values())
        chunk_size = impl._lmcache_chunk_size

        for req_id in preempted_req_ids:
            spec = self._pending_store.pop(req_id, None)
            if spec is None:
                continue

            token_ids = spec.token_ids
            slot_mapping = spec.slot_mapping.to(impl.device)

            # For decode steps (not last prefill), align to chunk boundary —
            # same logic as the write-through wait_for_save path.
            if not spec.is_last_prefill and not getattr(
                impl, "enable_blending", False
            ):
                aligned_len = len(token_ids) // chunk_size * chunk_size
                token_ids = token_ids[:aligned_len]
                slot_mapping = slot_mapping[:aligned_len]

            # Determine skip based on what SPO has *actually* stored, NOT the
            # scheduler's num_saved_tokens.  The scheduler assumes write-through
            # semantics and advances its counter every step, so using
            # save_spec.skip_leading_tokens would cause SPO to skip prefix
            # chunks that were never actually written to SSD, producing a
            # suffix-only store that LMCache's prefix-based lookup cannot find.
            actually_stored = self._spo_stored.get(req_id, 0)
            skip = actually_stored // chunk_size * chunk_size

            if skip >= len(token_ids):
                logger.debug(
                    "SPO flush: req=%s already fully saved, skipping", req_id
                )
                continue

            store_mask = torch.ones(len(token_ids), dtype=torch.bool)
            store_mask[:skip] = False

            logger.info(
                "SPO flush: req=%s storing %d tokens (skip=%d) → SSD",
                req_id,
                len(token_ids) - skip,
                skip,
            )
            engine.store(
                token_ids,
                mask=store_mask,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
                req_id=req_id,
            )
            self._spo_stored[req_id] = len(token_ids)
