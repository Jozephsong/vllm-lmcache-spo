# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data-integrity test for LMCacheSPOConnector.

Approach:
  1. Populate fake HBM (kv_caches) with deterministic tensor values.
  2. Capture a request snapshot via _capture_pending.
  3. Trigger _flush_evicted by calling handle_block_evictions with the
     request's block IDs.
  4. FakeLMCacheEngine intercepts engine.store() and copies the actual
     KV tensor values from kvcaches at the masked slot positions.
  5. Compare the captured values against the original HBM data.
  6. For the round-trip test, call FakeLMCacheEngine.retrieve() into a
     fresh kv_cache and verify the values are identical to the original.

No GPU, no real LMCache installation, no SSD required.
"""
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_spo_connector import (
    LMCacheSPOConnector,
)

# ---------------------------------------------------------------------------
# Fake lmcache adapter module
# ---------------------------------------------------------------------------

class _FakeSaveSpec:
    def __init__(self, can_save: bool = True, skip_leading_tokens: int = 0):
        self.can_save = can_save
        self.skip_leading_tokens = skip_leading_tokens


class _FakeReqMeta:
    def __init__(self, req_id, token_ids, slot_mapping,
                 save_spec=None, is_last_prefill=True):
        self.req_id = req_id
        self.token_ids = token_ids
        self.slot_mapping = slot_mapping
        self.save_spec = save_spec or _FakeSaveSpec()
        self.is_last_prefill = is_last_prefill
        self.request_configs = None


class _FakeLMCacheConnectorMetadata:
    def __init__(self, requests=None):
        self.requests = requests or []


_fake_adapter_module = types.ModuleType(
    "lmcache.integration.vllm.vllm_v1_adapter"
)
_fake_adapter_module.LMCacheConnectorMetadata = _FakeLMCacheConnectorMetadata  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# In-memory "SSD" engine
# ---------------------------------------------------------------------------

class FakeLMCacheEngine:
    """Intercepts store() calls, copies KV values from kvcaches in-memory,
    and can write them back via retrieve() to simulate a load from SSD."""

    def __init__(self):
        # req_id → {"token_ids": list[int], "layer_data": list[Tensor]}
        self._storage: dict[str, dict] = {}
        self.store_call_count = 0
        self.retrieve_call_count = 0

    def store(self, token_ids, *, mask, kvcaches, slot_mapping,
              req_id=None, **kwargs):
        self.store_call_count += 1
        active = mask.bool()
        active_slots = slot_mapping[active]
        # Copy the actual KV values from each layer at the active positions.
        self._storage[req_id] = {
            "token_ids": [t for t, m in zip(token_ids, active.tolist()) if m],
            "layer_data": [cache[active_slots].clone() for cache in kvcaches],
        }

    def retrieve(self, token_ids, *, kvcaches, slot_mapping,
                 req_id=None, **kwargs):
        """Write stored KV data back into kvcaches at the given slot positions.
        Returns the number of tokens written."""
        self.retrieve_call_count += 1
        if req_id not in self._storage:
            return 0
        stored = self._storage[req_id]
        n = len(stored["token_ids"])
        for cache, layer_data in zip(kvcaches, stored["layer_data"]):
            cache[slot_mapping[:n]] = layer_data
        return n

    def lookup_unpin(self, req_id):
        pass

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

BLOCK_SIZE = 4   # tokens per vLLM block
CHUNK_SIZE = 8   # LMCache chunk size (2 blocks per chunk)
N_BLOCKS  = 16   # total blocks in the fake HBM pool
N_LAYERS  = 2
HEAD_DIM  = 8    # KV vector dimension per token

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kv_caches() -> dict[str, torch.Tensor]:
    """Return per-layer kv_caches with unique, slot-indexed float values.

    cache["layerL"][slot] == (slot + 1) * (L + 1) so each (layer, slot)
    pair has a distinct value that is easy to verify."""
    total_slots = N_BLOCKS * BLOCK_SIZE
    return {
        f"layer{l}": (
            torch.arange(1, total_slots + 1, dtype=torch.float32)
            .mul(l + 1)
            .unsqueeze(1)
            .expand(total_slots, HEAD_DIM)
            .contiguous()
        )
        for l in range(N_LAYERS)
    }


def _slot_mapping(block_ids: list[int]) -> torch.Tensor:
    """Flat slot indices for a list of block IDs (contiguous within each block)."""
    slots = []
    for bid in block_ids:
        slots.extend(range(bid * BLOCK_SIZE, (bid + 1) * BLOCK_SIZE))
    return torch.tensor(slots, dtype=torch.long)


def _make_spo(fake_engine: FakeLMCacheEngine,
              kv_caches: dict[str, torch.Tensor]) -> LMCacheSPOConnector:
    """Return an SPO connector with real method bindings and injected fixtures.

    Uses MagicMock as a base so we can bind only the methods under test
    without running __init__ or importing GPU/LMCache dependencies."""
    connector = MagicMock(spec=LMCacheSPOConnector)

    impl = MagicMock()
    impl.lmcache_engine = fake_engine
    impl.kv_caches = kv_caches
    impl._lmcache_chunk_size = CHUNK_SIZE
    impl.device = "cpu"
    impl.enable_blending = False

    vllm_cfg = MagicMock()
    vllm_cfg.cache_config.block_size = BLOCK_SIZE

    connector._lmcache_engine = impl
    connector._vllm_config = vllm_cfg
    connector._pending_store = {}
    connector._is_spo = True

    for name in ("_capture_pending", "_flush_evicted", "_cleanup_req",
                 "handle_block_evictions", "wait_for_save"):
        fn = getattr(LMCacheSPOConnector, name)
        setattr(connector, name, fn.__get__(connector, LMCacheSPOConnector))

    return connector


def _inject_metadata(spo, req_id: str, block_ids: list[int],
                     skip: int = 0, is_last_prefill: bool = True):
    """Inject one fake request into the connector and run _capture_pending."""
    n_tokens = len(block_ids) * BLOCK_SIZE
    slot_map = _slot_mapping(block_ids)
    req = _FakeReqMeta(
        req_id=req_id,
        token_ids=list(range(n_tokens)),
        slot_mapping=slot_map,
        save_spec=_FakeSaveSpec(can_save=True, skip_leading_tokens=skip),
        is_last_prefill=is_last_prefill,
    )
    spo._get_connector_metadata = MagicMock(
        return_value=_FakeLMCacheConnectorMetadata(requests=[req])
    )
    spo._capture_pending()
    return slot_map, n_tokens

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_lmcache_modules():
    with patch.dict("sys.modules", {
        "lmcache": MagicMock(),
        "lmcache.integration": MagicMock(),
        "lmcache.integration.vllm": MagicMock(),
        "lmcache.integration.vllm.vllm_v1_adapter": _fake_adapter_module,
    }):
        yield


@pytest.fixture()
def kv_caches():
    return _make_kv_caches()


@pytest.fixture()
def fake_engine():
    return FakeLMCacheEngine()


@pytest.fixture()
def spo(fake_engine, kv_caches):
    return _make_spo(fake_engine, kv_caches)

# ---------------------------------------------------------------------------
# Data-integrity tests
# ---------------------------------------------------------------------------

class TestSPOWriteIntegrity:
    """Verify that the KV values written to 'SSD' exactly match the HBM data
    captured at eviction time."""

    def test_stored_data_matches_hbm_all_tokens(self, spo, fake_engine, kv_caches):
        """When skip=0, every token's KV must be stored.  The values recorded
        by FakeLMCacheEngine must equal kv_caches[layer][slot] for each slot."""
        block_ids = [0, 1]
        slot_map, n = _inject_metadata(spo, "req1", block_ids)

        spo.handle_block_evictions(set(block_ids), set())

        assert fake_engine.store_call_count == 1
        stored = fake_engine._storage["req1"]
        assert len(stored["token_ids"]) == n

        kvcache_list = list(kv_caches.values())
        for layer_idx, layer_key in enumerate(kv_caches):
            expected = kv_caches[layer_key][slot_map]          # [n, HEAD_DIM]
            actual   = stored["layer_data"][layer_idx]         # same shape
            assert torch.allclose(actual, expected), (
                f"{layer_key}: stored values differ from HBM (no skip)"
            )

    def test_stored_data_matches_hbm_with_skip(self, spo, fake_engine, kv_caches):
        """When the first CHUNK_SIZE tokens were already in LMCache (skip),
        only the remaining tokens should be stored, and their values must
        match the HBM data at the corresponding slot positions."""
        block_ids = [2, 3, 4, 5]          # 4 blocks = 16 tokens
        skip = CHUNK_SIZE                  # first 8 already cached
        slot_map, n = _inject_metadata(spo, "req1", block_ids, skip=skip)

        spo.handle_block_evictions(set(block_ids), set())

        stored = fake_engine._storage["req1"]
        assert len(stored["token_ids"]) == n - skip

        active_slots = slot_map[skip:]     # slots corresponding to stored tokens
        for layer_idx, layer_key in enumerate(kv_caches):
            expected = kv_caches[layer_key][active_slots]
            actual   = stored["layer_data"][layer_idx]
            assert torch.allclose(actual, expected), (
                f"{layer_key}: stored values differ from HBM (with skip)"
            )

    def test_round_trip_store_then_load(self, spo, fake_engine, kv_caches):
        """Full round-trip: store via SPO eviction, retrieve into a fresh
        kv_cache, then compare every slot against the original HBM data."""
        block_ids = [6, 7]
        slot_map, n = _inject_metadata(spo, "req1", block_ids)

        # --- store phase ---
        spo.handle_block_evictions(set(block_ids), set())
        assert fake_engine.store_call_count == 1

        # --- load phase: write stored data into a blank kv_cache ---
        total_slots = N_BLOCKS * BLOCK_SIZE
        fresh = {k: torch.zeros(total_slots, HEAD_DIM) for k in kv_caches}
        n_loaded = fake_engine.retrieve(
            list(range(n)),
            kvcaches=list(fresh.values()),
            slot_mapping=slot_map,
            req_id="req1",
        )
        assert n_loaded == n

        # --- comparison ---
        for layer_key in kv_caches:
            original  = kv_caches[layer_key][slot_map]
            retrieved = fresh[layer_key][slot_map]
            assert torch.allclose(original, retrieved), (
                f"{layer_key}: round-trip mismatch (store → SSD → load)"
            )

    def test_no_store_when_no_block_overlap(self, spo, fake_engine, kv_caches):
        """Evicting blocks that don't belong to the request must not trigger
        a store, and the pending entry must remain intact."""
        _inject_metadata(spo, "req1", [0, 1])          # request on blocks 0, 1

        spo.handle_block_evictions({8, 9}, set())       # evict unrelated blocks

        assert fake_engine.store_call_count == 0
        assert "req1" in spo._pending_store

    def test_partial_overlap_stores_full_sequence(self, spo, fake_engine, kv_caches):
        """If only one of the request's blocks is evicted, the connector must
        store the FULL captured token sequence (LMCache stores by key prefix,
        not by individual block)."""
        block_ids = [2, 3, 4, 5]           # 4 blocks = 16 tokens
        slot_map, n = _inject_metadata(spo, "req1", block_ids)

        spo.handle_block_evictions({3}, set())          # evict only block 3

        assert fake_engine.store_call_count == 1
        assert len(fake_engine._storage["req1"]["token_ids"]) == n

    def test_pending_entry_removed_after_store(self, spo, fake_engine, kv_caches):
        """_pending_store must be cleaned up after the eviction-triggered store."""
        _inject_metadata(spo, "req1", [0, 1])
        spo.handle_block_evictions({0, 1}, set())
        assert "req1" not in spo._pending_store

    def test_finished_req_not_prematurely_cleaned_up(
        self, spo, fake_engine, kv_caches
    ):
        """Regression for premature-cleanup bug.

        When request X finishes in the same scheduling step that evicts
        *other* blocks, the old code removed X from _pending_store immediately.
        When X's own blocks were evicted later, nothing was stored.

        With the fix, X's entry must survive until its own blocks are evicted.
        """
        # Request X occupies blocks 4 and 5.
        block_ids_x = [4, 5]
        _inject_metadata(spo, "user_x", block_ids_x)

        # Step T: blocks 0, 1 evicted (not X's); X is in finished_req_ids.
        spo.handle_block_evictions({0, 1}, {"user_x"})

        # X must still be in _pending_store — its blocks haven't been evicted.
        assert "user_x" in spo._pending_store, (
            "user_x was prematurely removed before its own blocks were evicted"
        )
        assert fake_engine.store_call_count == 0

        # Step T+N: X's own blocks are finally evicted.
        spo.handle_block_evictions(set(block_ids_x), set())

        assert fake_engine.store_call_count == 1
        assert "user_x" in fake_engine._storage
        assert "user_x" not in spo._pending_store

    def test_multi_round_qa_two_rounds(self, spo, fake_engine, kv_caches):
        """Simulate a two-round conversation matching the multi-round-qa
        benchmark pattern.

        Round 1:
          - User sends a long prompt.  SPO captures the KV snapshot.
          - Request finishes; blocks go to the free pool.
          - Eviction triggered → KV stored to 'SSD'.

        Round 2:
          - LMCache lookup hits → retrieve() loads stored KV into fresh HBM.
          - Verify that the loaded values are identical to the round-1 HBM data.
        """
        r1_blocks = [0, 1, 2, 3]          # system_prompt + user_info
        slot_map_r1, n_r1 = _inject_metadata(spo, "round1", r1_blocks)

        # Round 1 finishes; its blocks get evicted in the same step.
        spo.handle_block_evictions(set(r1_blocks), {"round1"})
        assert fake_engine.store_call_count == 1, "Round 1 KV not stored"

        # Round 2: allocate fresh HBM and load from SSD.
        total_slots = N_BLOCKS * BLOCK_SIZE
        fresh = {k: torch.zeros(total_slots, HEAD_DIM) for k in kv_caches}
        n_loaded = fake_engine.retrieve(
            list(range(n_r1)),
            kvcaches=list(fresh.values()),
            slot_mapping=slot_map_r1,
            req_id="round1",
        )
        assert n_loaded == n_r1, "Partial or no retrieve from SSD"

        for layer_key in kv_caches:
            original  = kv_caches[layer_key][slot_map_r1]
            recovered = fresh[layer_key][slot_map_r1]
            assert torch.allclose(original, recovered), (
                f"{layer_key}: round-2 loaded data does not match round-1 HBM"
            )

    def test_chunk_align_for_intermediate_prefill(
        self, spo, fake_engine, kv_caches
    ):
        """For non-final prefill chunks (is_last_prefill=False), the stored
        sequence must be truncated to the nearest chunk boundary and the KV
        values at those truncated slots must still match HBM."""
        block_ids = [0, 1, 2]             # 3 blocks = 12 tokens
        n_tokens  = len(block_ids) * BLOCK_SIZE  # 12
        aligned   = n_tokens // CHUNK_SIZE * CHUNK_SIZE  # 8

        slot_map, _ = _inject_metadata(
            spo, "req1", block_ids, is_last_prefill=False
        )
        spo.handle_block_evictions(set(block_ids), set())

        assert fake_engine.store_call_count == 1
        stored = fake_engine._storage["req1"]
        assert len(stored["token_ids"]) == aligned, (
            f"Expected {aligned} tokens after chunk-align, "
            f"got {len(stored['token_ids'])}"
        )

        active_slots = slot_map[:aligned]
        for layer_idx, layer_key in enumerate(kv_caches):
            expected = kv_caches[layer_key][active_slots]
            actual   = stored["layer_data"][layer_idx]
            assert torch.allclose(actual, expected), (
                f"{layer_key}: chunk-aligned stored data differs from HBM"
            )

    def test_store_exception_does_not_leave_stale_entry(
        self, spo, fake_engine, kv_caches
    ):
        """If engine.store() raises, the _pending_store entry must still be
        removed (the finally block in _flush_evicted) to avoid a stale snapshot
        that would be incorrectly replayed on future evictions."""
        fake_engine.store = MagicMock(side_effect=RuntimeError("disk full"))

        _inject_metadata(spo, "req1", [0, 1])
        spo.handle_block_evictions({0, 1}, set())

        assert "req1" not in spo._pending_store, (
            "Stale _pending_store entry left after store() exception"
        )
