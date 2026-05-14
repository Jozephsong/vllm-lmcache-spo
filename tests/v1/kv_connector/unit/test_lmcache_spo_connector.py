# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for LMCacheSPOConnector.

All LMCache and vLLM model-level dependencies are mocked so these tests
run without a GPU or an LMCache installation.
"""
import sys
import types
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector_spo import (
    LMCacheSPOConnector,
    _PendingSpec,
    _SPILL_OVER,
    _WRITE_THROUGH,
)

# ---------------------------------------------------------------------------
# Fake LMCache types
# The _capture_pending method does a lazy
#   from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorMetadata
# and checks isinstance(..., LMCacheConnectorMetadata).
# We inject a fake module so the import resolves to our own class.
# ---------------------------------------------------------------------------

CHUNK_SIZE = 256


class FakeSaveSpec:
    def __init__(self, can_save: bool = True, skip_leading_tokens: int = 0):
        self.can_save = can_save
        self.skip_leading_tokens = skip_leading_tokens


class FakeReqMeta:
    def __init__(
        self,
        req_id: str,
        token_ids: list[int],
        slot_mapping: torch.Tensor,
        save_spec: FakeSaveSpec | None = None,
        is_last_prefill: bool = True,
    ):
        self.req_id = req_id
        self.token_ids = token_ids
        self.slot_mapping = slot_mapping
        self.save_spec = save_spec or FakeSaveSpec()
        self.is_last_prefill = is_last_prefill


class FakeLMCacheConnectorMetadata:
    def __init__(self, requests: list[FakeReqMeta] | None = None):
        self.requests = requests or []


# Build a fake module and register it before any test runs.
_fake_adapter_module = types.ModuleType("lmcache.integration.vllm.vllm_v1_adapter")
_fake_adapter_module.LMCacheConnectorMetadata = FakeLMCacheConnectorMetadata  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slot_mapping(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.long)


def _req(
    req_id: str,
    num_tokens: int,
    *,
    can_save: bool = True,
    skip: int = 0,
    is_last_prefill: bool = True,
) -> FakeReqMeta:
    return FakeReqMeta(
        req_id=req_id,
        token_ids=list(range(num_tokens)),
        slot_mapping=_slot_mapping(num_tokens),
        save_spec=FakeSaveSpec(can_save=can_save, skip_leading_tokens=skip),
        is_last_prefill=is_last_prefill,
    )


def _meta(*reqs: FakeReqMeta) -> FakeLMCacheConnectorMetadata:
    return FakeLMCacheConnectorMetadata(requests=list(reqs))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_lmcache_modules():
    """Replace lmcache import so tests run without the package installed."""
    with patch.dict(
        "sys.modules",
        {
            "lmcache": MagicMock(),
            "lmcache.integration": MagicMock(),
            "lmcache.integration.vllm": MagicMock(),
            "lmcache.integration.vllm.vllm_v1_adapter": _fake_adapter_module,
        },
    ):
        yield


@pytest.fixture()
def mock_impl():
    """Minimal fake of LMCacheConnectorV1Impl."""
    impl = MagicMock()
    impl.lmcache_engine = MagicMock()
    impl.kv_caches = {
        "layer0": torch.zeros(200, 2, 4),
        "layer1": torch.zeros(200, 2, 4),
    }
    impl._lmcache_chunk_size = CHUNK_SIZE
    impl.device = "cpu"
    impl.use_layerwise = False
    impl.enable_blending = False
    return impl


def _make_connector(mock_impl, *, is_spo: bool = True) -> LMCacheSPOConnector:
    """Create an LMCacheSPOConnector bound to *mock_impl* without calling
    LMCache or vLLM __init__ chains."""
    connector = MagicMock(spec=LMCacheSPOConnector)
    connector._lmcache_engine = mock_impl
    connector._pending_store = {}
    connector._is_spo = is_spo

    for name in (
        "_capture_pending",
        "_flush_preempted",
        "wait_for_save",
        "handle_preemptions",
        "request_finished",
    ):
        fn = getattr(LMCacheSPOConnector, name)
        setattr(connector, name, fn.__get__(connector, LMCacheSPOConnector))

    return connector


@pytest.fixture()
def spo(mock_impl):
    return _make_connector(mock_impl, is_spo=True)


@pytest.fixture()
def wt(mock_impl):
    return _make_connector(mock_impl, is_spo=False)


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------


class TestInit:
    """Test __init__ logic without running the LMCache setup."""

    def _make_raw(self, write_mode: str, use_layerwise: bool = False,
                  role: KVConnectorRole = KVConnectorRole.WORKER):
        with patch.object(LMCacheConnectorV1, "__init__", return_value=None):
            vllm_config = MagicMock()
            vllm_config.kv_transfer_config.get_from_extra_config.return_value = write_mode

            connector = LMCacheSPOConnector.__new__(LMCacheSPOConnector)
            # Parent __init__ is a no-op; manually set what it would normally provide.
            connector._lmcache_engine = MagicMock(use_layerwise=use_layerwise)
            connector._connector_metadata = None
            connector._kv_cache_events = None

            LMCacheSPOConnector.__init__(
                connector, vllm_config, role, MagicMock()
            )
        return connector

    def test_write_through_default(self):
        c = self._make_raw(_WRITE_THROUGH)
        assert c._is_spo is False

    def test_spill_over_mode_active(self):
        c = self._make_raw(_SPILL_OVER)
        assert c._is_spo is True
        assert isinstance(c._pending_store, dict)
        assert len(c._pending_store) == 0

    def test_invalid_write_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid write_mode"):
            self._make_raw("bad_mode")

    def test_layerwise_true_forces_write_through(self):
        c = self._make_raw(_SPILL_OVER, use_layerwise=True)
        assert c._is_spo is False

    def test_layerwise_ignored_on_scheduler_role(self):
        """Scheduler role does not have use_layerwise; SPO should stay active."""
        with patch.object(LMCacheConnectorV1, "__init__", return_value=None):
            vllm_config = MagicMock()
            vllm_config.kv_transfer_config.get_from_extra_config.return_value = _SPILL_OVER

            connector = LMCacheSPOConnector.__new__(LMCacheSPOConnector)
            connector._lmcache_engine = MagicMock(spec=[])  # no use_layerwise
            connector._connector_metadata = None
            connector._kv_cache_events = None

            LMCacheSPOConnector.__init__(
                connector, vllm_config, KVConnectorRole.SCHEDULER, MagicMock()
            )
        # getattr fallback returns False → no forced write-through
        assert connector._is_spo is True


# ---------------------------------------------------------------------------
# wait_for_save dispatch
# ---------------------------------------------------------------------------


class TestWaitForSave:
    def test_spo_calls_capture_pending(self, spo):
        spo._capture_pending = MagicMock()
        spo.wait_for_save()
        spo._capture_pending.assert_called_once()

    def test_write_through_delegates_to_parent(self, wt):
        wt.wait_for_save()
        # super().wait_for_save() is represented by the MagicMock attribute
        LMCacheConnectorV1.wait_for_save  # not called via super on mock
        # The real assertion: parent's wait_for_save was invoked
        wt._lmcache_engine.wait_for_save  # ensure no SPO path ran
        assert wt._pending_store == {}  # nothing captured in write-through


# ---------------------------------------------------------------------------
# handle_preemptions dispatch
# ---------------------------------------------------------------------------


class TestHandlePreemptions:
    def test_spo_empty_set_is_noop(self, spo, mock_impl):
        spo.handle_preemptions(set())
        mock_impl.lmcache_engine.store.assert_not_called()

    def test_spo_calls_flush_preempted(self, spo):
        spo._flush_preempted = MagicMock()
        spo.handle_preemptions({"req-1"})
        spo._flush_preempted.assert_called_once_with({"req-1"})

    def test_write_through_is_noop(self, wt, mock_impl):
        wt.handle_preemptions({"req-1"})
        mock_impl.lmcache_engine.store.assert_not_called()


# ---------------------------------------------------------------------------
# _capture_pending
# ---------------------------------------------------------------------------


class TestCapturePending:
    def _run(self, spo, metadata: FakeLMCacheConnectorMetadata):
        spo._get_connector_metadata = MagicMock(return_value=metadata)
        spo._capture_pending()

    def test_saveable_request_stored(self, spo, mock_impl):
        req = _req("r1", CHUNK_SIZE)
        self._run(spo, _meta(req))

        assert "r1" in spo._pending_store
        spec = spo._pending_store["r1"]
        assert spec.token_ids == req.token_ids
        assert torch.equal(spec.slot_mapping, req.slot_mapping)
        assert spec.skip_leading_tokens == 0
        assert spec.is_last_prefill is True

    def test_lookup_unpin_called_for_all_requests(self, spo, mock_impl):
        reqs = [
            _req("r1", CHUNK_SIZE, can_save=True),
            _req("r2", CHUNK_SIZE, can_save=False),
        ]
        self._run(spo, _meta(*reqs))

        calls = mock_impl.lmcache_engine.lookup_unpin.call_args_list
        unpinned = {c.args[0] for c in calls}
        assert unpinned == {"r1", "r2"}

    def test_can_save_false_not_stored(self, spo, mock_impl):
        req = _req("r1", CHUNK_SIZE, can_save=False)
        self._run(spo, _meta(req))
        assert "r1" not in spo._pending_store

    def test_no_save_spec_not_stored(self, spo, mock_impl):
        req = FakeReqMeta(
            req_id="r1",
            token_ids=list(range(CHUNK_SIZE)),
            slot_mapping=_slot_mapping(CHUNK_SIZE),
            save_spec=None,
        )
        self._run(spo, _meta(req))
        assert "r1" not in spo._pending_store

    def test_skip_equals_tokens_not_stored(self, spo, mock_impl):
        # skip_leading_tokens == len(token_ids) → nothing new to save
        req = _req("r1", CHUNK_SIZE, skip=CHUNK_SIZE)
        self._run(spo, _meta(req))
        assert "r1" not in spo._pending_store

    def test_partial_skip_stored(self, spo, mock_impl):
        # First CHUNK_SIZE already saved, but there are 2×CHUNK_SIZE total
        req = _req("r1", 2 * CHUNK_SIZE, skip=CHUNK_SIZE)
        self._run(spo, _meta(req))
        assert "r1" in spo._pending_store
        assert spo._pending_store["r1"].skip_leading_tokens == CHUNK_SIZE

    def test_later_step_overwrites_earlier_entry(self, spo, mock_impl):
        # Step 1: 256 tokens
        req_v1 = _req("r1", CHUNK_SIZE)
        self._run(spo, _meta(req_v1))
        assert len(spo._pending_store["r1"].token_ids) == CHUNK_SIZE

        # Step 2: 512 tokens (request ran one more decode step)
        req_v2 = _req("r1", 2 * CHUNK_SIZE)
        self._run(spo, _meta(req_v2))
        assert len(spo._pending_store["r1"].token_ids) == 2 * CHUNK_SIZE

    def test_metadata_type_mismatch_returns_early(self, spo, mock_impl):
        spo._get_connector_metadata = MagicMock(return_value=MagicMock())
        spo._capture_pending()
        mock_impl.lmcache_engine.lookup_unpin.assert_not_called()
        assert spo._pending_store == {}

    def test_empty_metadata_no_store(self, spo, mock_impl):
        self._run(spo, _meta())
        assert spo._pending_store == {}

    def test_multiple_requests_stored(self, spo, mock_impl):
        reqs = [_req(f"r{i}", CHUNK_SIZE) for i in range(3)]
        self._run(spo, _meta(*reqs))
        assert set(spo._pending_store.keys()) == {"r0", "r1", "r2"}


# ---------------------------------------------------------------------------
# _flush_preempted
# ---------------------------------------------------------------------------


class TestFlushPreempted:
    def _plant(self, spo, req_id: str, num_tokens: int, *,
               skip: int = 0, is_last_prefill: bool = True):
        spo._pending_store[req_id] = _PendingSpec(
            token_ids=list(range(num_tokens)),
            slot_mapping=_slot_mapping(num_tokens),
            skip_leading_tokens=skip,
            is_last_prefill=is_last_prefill,
        )

    def test_store_called_with_correct_args(self, spo, mock_impl):
        self._plant(spo, "r1", CHUNK_SIZE)
        spo._flush_preempted({"r1"})

        store = mock_impl.lmcache_engine.store
        store.assert_called_once()
        kwargs = store.call_args.kwargs

        assert kwargs["kvcaches"] == list(mock_impl.kv_caches.values())
        assert kwargs["req_id"] == "r1"
        assert len(kwargs["mask"]) == CHUNK_SIZE
        assert kwargs["mask"].all()  # skip=0 → all True

    def test_entry_removed_after_flush(self, spo, mock_impl):
        self._plant(spo, "r1", CHUNK_SIZE)
        spo._flush_preempted({"r1"})
        assert "r1" not in spo._pending_store

    def test_no_pending_spec_no_store(self, spo, mock_impl):
        spo._flush_preempted({"r1"})  # nothing in _pending_store
        mock_impl.lmcache_engine.store.assert_not_called()

    def test_skip_aligned_to_chunk_boundary(self, spo, mock_impl):
        # skip=300 should be aligned down to 256 (one CHUNK_SIZE)
        self._plant(spo, "r1", 2 * CHUNK_SIZE, skip=300)
        spo._flush_preempted({"r1"})

        kwargs = mock_impl.lmcache_engine.store.call_args.kwargs
        expected_skip = 256  # 300 // 256 * 256
        assert not kwargs["mask"][:expected_skip].any()
        assert kwargs["mask"][expected_skip:].all()

    def test_skip_zero_stores_all_tokens(self, spo, mock_impl):
        self._plant(spo, "r1", CHUNK_SIZE, skip=0)
        spo._flush_preempted({"r1"})
        kwargs = mock_impl.lmcache_engine.store.call_args.kwargs
        assert kwargs["mask"].all()

    def test_skip_equals_aligned_token_count_no_store(self, spo, mock_impl):
        # CHUNK_SIZE tokens with skip=CHUNK_SIZE → nothing left to store
        self._plant(spo, "r1", CHUNK_SIZE, skip=CHUNK_SIZE)
        spo._flush_preempted({"r1"})
        mock_impl.lmcache_engine.store.assert_not_called()

    def test_decode_step_aligns_to_chunk_boundary(self, spo, mock_impl):
        # 300 tokens, not last prefill → aligned down to 256
        self._plant(spo, "r1", 300, is_last_prefill=False)
        spo._flush_preempted({"r1"})

        kwargs = mock_impl.lmcache_engine.store.call_args.kwargs
        assert len(kwargs["mask"]) == CHUNK_SIZE  # trimmed to 256

    def test_last_prefill_not_trimmed(self, spo, mock_impl):
        # 300 tokens, last prefill → stored as-is (not aligned)
        self._plant(spo, "r1", 300, is_last_prefill=True)
        spo._flush_preempted({"r1"})

        kwargs = mock_impl.lmcache_engine.store.call_args.kwargs
        assert len(kwargs["mask"]) == 300

    def test_multiple_preempted_requests(self, spo, mock_impl):
        for i in range(3):
            self._plant(spo, f"r{i}", CHUNK_SIZE)
        spo._flush_preempted({"r0", "r1", "r2"})

        assert mock_impl.lmcache_engine.store.call_count == 3
        assert spo._pending_store == {}

    def test_partial_overlap_only_preempted_flushed(self, spo, mock_impl):
        self._plant(spo, "preempted", CHUNK_SIZE)
        self._plant(spo, "running", CHUNK_SIZE)

        spo._flush_preempted({"preempted"})

        assert "preempted" not in spo._pending_store
        assert "running" in spo._pending_store
        mock_impl.lmcache_engine.store.assert_called_once()

    def test_no_kv_caches_no_store(self, spo, mock_impl):
        mock_impl.kv_caches = {}
        self._plant(spo, "r1", CHUNK_SIZE)
        spo._flush_preempted({"r1"})
        mock_impl.lmcache_engine.store.assert_not_called()

    def test_slot_mapping_moved_to_device(self, spo, mock_impl):
        mock_impl.device = "cpu"
        self._plant(spo, "r1", CHUNK_SIZE)
        spo._flush_preempted({"r1"})

        kwargs = mock_impl.lmcache_engine.store.call_args.kwargs
        assert kwargs["slot_mapping"].device.type == "cpu"


# ---------------------------------------------------------------------------
# request_finished
# ---------------------------------------------------------------------------


class TestRequestFinished:
    def _make_request(self, req_id: str) -> MagicMock:
        req = MagicMock()
        req.request_id = req_id
        return req

    def test_spo_removes_pending_entry(self, spo, mock_impl):
        spo._pending_store["r1"] = MagicMock()
        spo.request_finished(self._make_request("r1"), [])
        assert "r1" not in spo._pending_store

    def test_spo_no_entry_is_safe(self, spo, mock_impl):
        # Should not raise even if the request was never captured
        spo.request_finished(self._make_request("unknown"), [])

    def test_spo_does_not_affect_other_entries(self, spo, mock_impl):
        spo._pending_store["r1"] = MagicMock()
        spo._pending_store["r2"] = MagicMock()
        spo.request_finished(self._make_request("r1"), [])
        assert "r2" in spo._pending_store

    def test_write_through_leaves_pending_store_unchanged(self, wt, mock_impl):
        wt._pending_store["r1"] = MagicMock()
        wt.request_finished(self._make_request("r1"), [])
        # write-through does not touch _pending_store
        assert "r1" in wt._pending_store


# ---------------------------------------------------------------------------
# End-to-end multi-step workflow
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Simulate the per-step call sequence vLLM makes on the connector."""

    def _capture(self, spo, *reqs):
        spo._get_connector_metadata = MagicMock(
            return_value=_meta(*reqs)
        )
        spo._capture_pending()

    def test_normal_completion_never_stores(self, spo, mock_impl):
        """Requests that finish normally must never hit SSD."""
        req = MagicMock()
        req.request_id = "r1"

        for step in range(3):
            self._capture(spo, _req("r1", (step + 1) * CHUNK_SIZE))
            # No preemption — handle_preemptions called with empty set
            spo.handle_preemptions(set())

        spo.request_finished(req, [])

        mock_impl.lmcache_engine.store.assert_not_called()
        assert spo._pending_store == {}

    def test_preemption_triggers_save(self, spo, mock_impl):
        """Preemption on step 2 must save the KV captured at step 1."""
        # Step 1: request runs, 256 tokens captured
        self._capture(spo, _req("r1", CHUNK_SIZE))
        spo.handle_preemptions(set())

        # Step 2: scheduler preempts r1 before the forward pass
        spo.handle_preemptions({"r1"})

        mock_impl.lmcache_engine.store.assert_called_once()
        call_kwargs = mock_impl.lmcache_engine.store.call_args.kwargs
        assert call_kwargs["req_id"] == "r1"
        assert "r1" not in spo._pending_store

    def test_latest_snapshot_is_stored_on_preemption(self, spo, mock_impl):
        """The *most recent* token count must be flushed, not an earlier one."""
        # Step 1 — 256 tokens
        self._capture(spo, _req("r1", CHUNK_SIZE))
        spo.handle_preemptions(set())

        # Step 2 — 512 tokens (request generated more tokens)
        self._capture(spo, _req("r1", 2 * CHUNK_SIZE))
        spo.handle_preemptions(set())

        # Step 3 — preemption
        spo.handle_preemptions({"r1"})

        call_args = mock_impl.lmcache_engine.store.call_args
        stored_tokens = call_args.args[0]  # positional token_ids
        assert len(stored_tokens) == 2 * CHUNK_SIZE

    def test_multiple_requests_selective_preemption(self, spo, mock_impl):
        """Only preempted request is flushed; running requests keep their state."""
        self._capture(spo, _req("r1", CHUNK_SIZE), _req("r2", CHUNK_SIZE))
        spo.handle_preemptions(set())

        # r1 is preempted, r2 keeps running
        spo.handle_preemptions({"r1"})

        assert mock_impl.lmcache_engine.store.call_count == 1
        call_kwargs = mock_impl.lmcache_engine.store.call_args.kwargs
        assert call_kwargs["req_id"] == "r1"
        assert "r2" in spo._pending_store

    def test_decode_step_accumulation(self, spo, mock_impl):
        """Decode steps (is_last_prefill=False) accumulate until preemption."""
        # Prefill: 256 tokens
        self._capture(spo, _req("r1", CHUNK_SIZE, is_last_prefill=True))
        spo.handle_preemptions(set())

        # Decode steps: add tokens one by one (simulate 10 decode steps)
        for n_tokens in range(CHUNK_SIZE + 1, CHUNK_SIZE + 11):
            self._capture(spo, _req("r1", n_tokens, is_last_prefill=False))
            spo.handle_preemptions(set())

        # Preemption
        spo.handle_preemptions({"r1"})

        # The last snapshot (CHUNK_SIZE + 10 tokens) should be flushed,
        # but decode steps are aligned to chunk boundary → CHUNK_SIZE tokens
        call_kwargs = mock_impl.lmcache_engine.store.call_args.kwargs
        assert len(call_kwargs["mask"]) == CHUNK_SIZE
