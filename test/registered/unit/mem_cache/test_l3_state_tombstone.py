"""CPU checks of the v6 writer ownership and saved-token admission rules."""

import threading
import unittest
from collections import defaultdict
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer, PoolTransferResult
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import HybridCacheController
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache, _OngoingPrefetch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _step(freed=0, pending=None):
    return SimpleNamespace(
        tracker={ComponentType.MAMBA: freed},
        mamba_storage_backup_node_id=pending,
        device_frees={}, host_frees={}, cache_actions=[],
    )


def _writer_fixture(*, state_ok=True, kv_ok=True, queued=True):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache._l3_write_state_on_tombstone = True
    cache._l3_trace = None
    cache._l3_lifetime_ledger = None
    cache._write_behind_clean_tokens = 0
    cache.tree_core = SimpleNamespace(
        page_size=64,
        enable_storage=True,
        begin_mamba_host_eviction=Mock(return_value=_step(pending=7)),
        finish_mamba_host_eviction=Mock(return_value=_step(freed=1)),
        drive_host_eviction=Mock(return_value=_step(freed=1)),
    )
    state = PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([3]), keys=["h"])
    spec = SimpleNamespace(
        host_value=torch.arange(64), token_ids=list(range(64)), hash_value=["h"],
        prefix_keys=None, comp_xfers={ComponentType.MAMBA: [state]},
    )
    cache.tree_core.build_storage_backup_spec = Mock(return_value=spec)
    cache.hicache_storage_pass_prefix_keys = False
    cache._storage_backup_is_covered = Mock(return_value=True)
    cache._build_sidecar_transfers = Mock(return_value=[])
    cache.ongoing_backup = {}
    events = []
    cache.inc_host_lock_ref = Mock(side_effect=lambda _: (
        events.append("lock") or SimpleNamespace(to_dec_params=lambda: "pin")
    ))
    cache.dec_host_lock_ref = Mock(side_effect=lambda *_: events.append("unlock"))
    cache._apply_eviction_result = Mock(side_effect=lambda result: (
        events.append("free") if result.tracker.get(ComponentType.MAMBA, 0) else None
    ))
    op = SimpleNamespace(
        id=11, hash_value=["h"], completed_tokens=64 if kv_ok else 0,
        pool_transfers=[state],
        pool_storage_result=PoolTransferResult(0, {PoolName.MAMBA: int(state_ok)}),
    )
    cc = SimpleNamespace(
        ack_backup_queue=Queue(), storage_stop_event=threading.Event(),
        backup_thread=SimpleNamespace(is_alive=lambda: False),
        should_backup=lambda _: True,
    )
    def write(*args, **kwargs):
        events.append("write")
        if queued:
            cc.ack_backup_queue.put(op)
        return op.id
    cc.write_storage = Mock(side_effect=write)
    cache.cache_controller = cc
    return cache, op, events


class TestMambaTombstoneWrite(unittest.TestCase):
    def test_actual_victim_is_pinned_written_then_freed_even_if_believed_stored(self):
        cache, operation, events = _writer_fixture()
        self.assertEqual(cache._drive_host_eviction(1, ComponentType.MAMBA), 1)
        self.assertEqual(events, ["lock", "write", "unlock", "free"])
        cache._storage_backup_is_covered.assert_not_called()
        cache.tree_core.begin_mamba_host_eviction.assert_called_once_with(1, min_depth=8192)
        cache.tree_core.finish_mamba_host_eviction.assert_called_once_with(7)
        self.assertEqual(cache.ongoing_backup, {})
        # The local-only operation never enters the ordinary rank-synced drain.
        self.assertTrue(cache.cache_controller.ack_backup_queue.empty())
        self.assertEqual(cache.cache_controller.write_storage.call_args.kwargs["ledger_reason"], "state_tombstone")

    def test_local_ack_removal_preserves_earlier_ordinary_fifo_and_count(self):
        cache, operation, _ = _writer_fixture()
        queue = cache.cache_controller.ack_backup_queue
        ordinary = [SimpleNamespace(id=4), SimpleNamespace(id=9)]
        for ack in ordinary:
            queue.put(ack)
        agreed_count = queue.qsize()
        cache._drive_host_eviction(1, ComponentType.MAMBA)
        self.assertEqual(queue.qsize(), agreed_count)
        self.assertEqual([queue.get_nowait() for _ in range(agreed_count)], ordinary)
        self.assertTrue(queue.empty())

    def test_state_or_kv_write_failure_never_frees_the_victim(self):
        for state_ok, kv_ok in [(False, True), (True, False)]:
            with self.subTest(state_ok=state_ok, kv_ok=kv_ok):
                cache, operation, events = _writer_fixture(state_ok=state_ok, kv_ok=kv_ok)
                with self.assertRaisesRegex(RuntimeError, "backup failed"):
                    cache._drive_host_eviction(1, ComponentType.MAMBA)
                self.assertEqual(events, ["lock", "write"])
                self.assertIn(operation.id, cache.ongoing_backup)
                self.assertTrue(operation.l3_tombstone_quarantined)
                cache.tree_core.finish_mamba_host_eviction.assert_not_called()
                # Even an attempted later drain cannot unlock this source.
                cache.host_memory_mode = "cache"
                cc = cache.cache_controller
                cc.prefetch_hit_queue = Queue()
                cc.ack_prefetch_queue = Queue()
                with self.assertRaisesRegex(RuntimeError, "quarantined Mamba"):
                    cache._drain_storage_control_queues_impl(0, 0, 1, 0, {}, False)
                cache.dec_host_lock_ref.assert_not_called()

    def test_missing_ack_retains_source_ownership(self):
        cache, operation, events = _writer_fixture(queued=False)
        with self.assertRaisesRegex(RuntimeError, "before Mamba tombstone ACK"):
            cache._drive_host_eviction(1, ComponentType.MAMBA)
        self.assertEqual(events, ["lock", "write"])
        self.assertIn(11, cache.ongoing_backup)
        cache.tree_core.finish_mamba_host_eviction.assert_not_called()
        cache.host_memory_mode = "cache"
        cc = cache.cache_controller
        cc.prefetch_hit_queue = Queue()
        cc.ack_prefetch_queue = Queue()
        cc.ack_backup_queue.put(operation)
        with self.assertRaisesRegex(RuntimeError, "quarantined Mamba"):
            cache._drain_storage_control_queues_impl(0, 0, 1, 0, {}, False)
        cache.dec_host_lock_ref.assert_not_called()

    def test_ack_arrives_from_worker_without_scheduler_polling(self):
        cache, operation, events = _writer_fixture(queued=False)
        entered = threading.Event()
        finish = threading.Event()
        original_write = cache.cache_controller.write_storage.side_effect
        def write(*args, **kwargs):
            result = original_write(*args, **kwargs)
            entered.set()
            return result
        cache.cache_controller.write_storage.side_effect = write
        def worker():
            entered.wait(2)
            self.assertEqual(events, ["lock", "write"])
            self.assertFalse(cache.tree_core.finish_mamba_host_eviction.called)
            cache.cache_controller.ack_backup_queue.put(operation)
            finish.wait(2)
        thread = threading.Thread(target=worker)
        cache.cache_controller.backup_thread = thread
        thread.start()
        try:
            self.assertEqual(cache._drive_host_eviction(1, ComponentType.MAMBA), 1)
            self.assertEqual(events, ["lock", "write", "unlock", "free"])
        finally:
            finish.set()
            thread.join(2)

    def test_flag_off_uses_original_walk(self):
        cache, _, events = _writer_fixture()
        cache._l3_write_state_on_tombstone = False
        self.assertEqual(cache._drive_host_eviction(1, ComponentType.MAMBA), 1)
        self.assertEqual(events, ["free"])
        cache.tree_core.begin_mamba_host_eviction.assert_not_called()
        cache.cache_controller.write_storage.assert_not_called()

    def test_resume_failure_does_not_repeat_the_same_victim(self):
        cache, _, _ = _writer_fixture()
        cache.tree_core.finish_mamba_host_eviction.return_value = _step()
        with self.assertRaisesRegex(RuntimeError, "victim changed"):
            cache._drive_host_eviction(1, ComponentType.MAMBA)
        self.assertEqual(cache.tree_core.begin_mamba_host_eviction.call_count, 1)


def _async_fixture(*, available=lambda events: 8, **kwargs):
    """SGLANG_HICACHE_L3_TOMBSTONE_ASYNC on top of _writer_fixture; the host Mamba
    pool's free count is a function of the events so far."""
    cache, op, events = _writer_fixture(**kwargs)
    cache._l3_tombstone_pending = {}
    cache._l3_tombstone_reserve = 0
    cache._l3_ack_lag_steps = 0
    cache.tree_core.begin_mamba_host_eviction = Mock(side_effect=[_step(pending=7), _step()])
    pool = SimpleNamespace(available_size=lambda: available(events))
    cache.host_pool_group = SimpleNamespace(get_pool=lambda name: pool)
    cache.host_memory_mode = "cache"
    cc = cache.cache_controller
    cc.prefetch_hit_queue = Queue()
    cc.ack_prefetch_queue = Queue()
    cc.host_mem_release_queue = Queue()
    return cache, op, events


class TestMambaTombstoneAsyncWrite(unittest.TestCase):
    def test_write_does_not_wait_and_the_synced_drain_frees_the_victim(self):
        cache, operation, events = _async_fixture()
        self.assertEqual(cache._drive_host_eviction(1, ComponentType.MAMBA), 0)
        self.assertEqual(events, ["lock", "write"])
        self.assertEqual(cache._l3_tombstone_pending, {11: 7})
        cache.tree_core.finish_mamba_host_eviction.assert_not_called()
        # The ACK stays in the FIFO for the rank-synced count.
        self.assertEqual(cache.cache_controller.ack_backup_queue.qsize(), 1)
        cache._drain_storage_control_queues_impl(0, 0, 1, 0, {}, False)
        self.assertEqual(events, ["lock", "write", "unlock", "free"])
        cache.tree_core.finish_mamba_host_eviction.assert_called_once_with(7)
        self.assertEqual((cache._l3_tombstone_pending, cache.ongoing_backup), ({}, {}))
        self.assertTrue(cache.cache_controller.ack_backup_queue.empty())

    def test_a_caller_short_of_slots_finishes_the_oldest_write_as_before(self):
        cache, _, events = _async_fixture(available=lambda events: events.count("free"))
        self.assertEqual(cache._drive_host_eviction(1, ComponentType.MAMBA), 1)
        self.assertEqual(events, ["lock", "write", "unlock", "free"])
        self.assertEqual(cache._l3_tombstone_pending, {})
        # Taken locally, as in synchronous mode: nothing left for the synced drain.
        self.assertTrue(cache.cache_controller.ack_backup_queue.empty())

    def test_a_failed_write_is_never_freed_by_the_drain(self):
        cache, operation, events = _async_fixture(state_ok=False)
        cache._drive_host_eviction(1, ComponentType.MAMBA)
        with self.assertRaisesRegex(RuntimeError, "backup failed"):
            cache._drain_storage_control_queues_impl(0, 0, 1, 0, {}, False)
        self.assertEqual(events, ["lock", "write"])
        self.assertTrue(operation.l3_tombstone_quarantined)
        self.assertIn(operation.id, cache.ongoing_backup)
        cache.tree_core.finish_mamba_host_eviction.assert_not_called()

    def test_a_victim_revived_before_its_drain_is_kept(self):
        cache, _, events = _async_fixture()
        cache._drive_host_eviction(1, ComponentType.MAMBA)
        # A prefix hit restored the state to the device: the walk no longer frees it.
        cache.tree_core.finish_mamba_host_eviction.return_value = _step()
        cache._drain_storage_control_queues_impl(0, 0, 1, 0, {}, False)
        self.assertEqual(events, ["lock", "write", "unlock"])
        self.assertEqual((cache._l3_tombstone_pending, cache.ongoing_backup), ({}, {}))

    def test_reserve_refill_writes_only_the_deficit(self):
        cache, _, events = _async_fixture(available=lambda events: 1)
        cache._l3_tombstone_reserve = 4
        cache._l3_tombstone_pending = {5: 3}
        ids = iter([11, 12, 13])
        cache.cache_controller.write_storage.side_effect = lambda *a, **k: (events.append("write"), next(ids))[1]
        cache.tree_core.begin_mamba_host_eviction = Mock(
            side_effect=[_step(pending=7), _step(pending=8), _step(pending=9)]
        )
        cache._refill_mamba_tombstone_reserve()
        self.assertEqual(cache._l3_tombstone_pending, {5: 3, 11: 7, 12: 8})
        self.assertEqual(
            [c.args[0] for c in cache.tree_core.begin_mamba_host_eviction.call_args_list], [2, 1]
        )
        cache.tree_core.finish_mamba_host_eviction.assert_not_called()


class TestSavedTokenThreshold(unittest.TestCase):
    def _cache(self, enabled=True):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(page_size=64)
        cache._l3_saved_token_threshold = enabled
        cache.prefetch_threshold = 8192
        return cache

    def test_positive_checkpoint_and_saved_token_boundaries(self):
        cache = self._cache()
        for tree, hit, accepted in [
            (0, 6336, False), (0, 8192, True),
            (4096, 4032, False), (4096, 4096, True),
            (65536, 64, True), (65536, 0, False),
        ]:
            with self.subTest(tree=tree, hit=hit):
                op = SimpleNamespace(stats_tree_kv_tokens=tree, storage_hit_count=hit)
                self.assertEqual(cache._storage_prefetch_meets_threshold(op), accepted)
        op = SimpleNamespace(stats_tree_kv_tokens=65536, storage_hit_count=4096)
        self.assertFalse(self._cache(enabled=False)._storage_prefetch_meets_threshold(op))

    def test_partial_allocation_minimum_keeps_saved_threshold(self):
        cache = self._cache()
        for tree, expected in [(0, 8192), (4096, 4096), (4097, 4096), (8192, 64)]:
            self.assertEqual(
                cache._storage_prefetch_min_hit_tokens(SimpleNamespace(stats_tree_kv_tokens=tree)),
                expected,
            )

    def _hit_drain(self):
        cache = self._cache()
        cache.host_memory_mode = "cache"
        cache._prefetch_outcome_stats = defaultdict(int)
        cache.enable_storage_metrics = False
        cache._invalidate_absent_from_hit_query = Mock()
        cache._record_storage_prefetch_hit = Mock()
        cache._account_prefetch_outcome = Mock()
        cache._resolve_storage_prefetch_tokens = Mock()
        cache._alloc_prefetch_aux_staging = Mock(return_value=True)
        cache._finish_storage_prefetch = Mock()
        cache.revoke_pending_prefetch = Mock()
        cache.evict_host = Mock()
        handle = CacheRequestHandle("saved", 0)
        operation = SimpleNamespace(
            handle=handle, storage_hit_count=4096, stats_tree_kv_tokens=4096,
            storage_start=4096, hash_value=["h"] * 64, is_terminated=lambda: False,
        )
        cache.ongoing_prefetch = {
            handle: _OngoingPrefetch(
                0, None, None, operation, None,
                {ComponentType.MAMBA: [PoolTransfer(name=PoolName.MAMBA, keys=["h"])]},
            )
        }
        cc = SimpleNamespace(
            prefetch_hit_queue=Queue(), ack_prefetch_queue=Queue(), ack_backup_queue=Queue(),
            host_mem_release_queue=Queue(), prefetch_buffer=Queue(),
            allocate_storage_hit=Mock(return_value=(torch.arange(4096), 4096)),
        )
        cache.cache_controller = cc
        cc.prefetch_hit_queue.put(operation)
        return cache, cc, operation

    def test_real_hit_drain_accepts_short_suffix_and_passes_adjusted_alloc_minimum(self):
        cache, cc, operation = self._hit_drain()
        cache._drain_storage_control_queues_impl(1, 0, 0, 0, {}, False)
        self.assertEqual(cc.allocate_storage_hit.call_args.kwargs["min_tokens"], 4096)
        self.assertFalse(cc.allocate_storage_hit.call_args.kwargs["allow_partial"])
        self.assertIs(cc.prefetch_buffer.get_nowait(), operation)
        cache.revoke_pending_prefetch.assert_not_called()
        cache._account_prefetch_outcome.assert_called_once_with(operation, revoked=False)

    def test_mamba_capacity_shortfall_does_not_shorten_confirmed_checkpoint(self):
        cache, cc, operation = self._hit_drain()
        cc.alloc_prefetch_host_buffers = Mock(return_value=None)
        cc.mem_pool_host = SimpleNamespace(
            anchor_entry=SimpleNamespace(host_pool=SimpleNamespace()),
            available_size=Mock(return_value=2048),
        )
        cc.allocate_storage_hit = HybridCacheController.allocate_storage_hit.__get__(cc)
        cache.storage_prefetch_retries = Mock()
        cache._log_storage_prefetch_deferred = Mock()
        cache._drain_storage_control_queues_impl(1, 0, 0, 0, {}, False)
        self.assertEqual([call.args[1] for call in cc.alloc_prefetch_host_buffers.call_args_list], [4096, 4096])
        cc.mem_pool_host.available_size.assert_not_called()
        self.assertEqual(operation.storage_hit_count, 4096)
        self.assertTrue(cc.prefetch_buffer.empty())
        cache.revoke_pending_prefetch.assert_called_once_with(operation.handle)
        cache._alloc_prefetch_aux_staging.assert_not_called()


class TestL3LifetimeAccounting(unittest.TestCase):
    def test_finish_is_once_and_counts_consumption_without_metrics(self):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache._l3_lifetime_ledger = Mock()
        cache.enable_storage_metrics = False
        cache._storage_prefetch_hit_remaining_by_reqid = {}
        handle = CacheRequestHandle("ledger", 0)
        cache.ongoing_prefetch = {
            handle: SimpleNamespace(operation=SimpleNamespace(id=19, storage_start=65536))
        }
        cache._record_storage_prefetch_hit(handle, 4096)
        cache._resolve_storage_prefetch_tokens(handle, 64)
        cache._finish_storage_prefetch(handle, 4032, None)
        cache._finish_storage_prefetch(handle, 4032, None)
        calls = cache._l3_lifetime_ledger.emit.call_args_list
        self.assertEqual([call.args[0] for call in calls], ["storage_prefetch_resolved", "storage_prefetch_finish"])
        self.assertEqual(calls[0].kwargs["resolved_tokens"], 64)
        self.assertEqual(calls[1].kwargs["fulfilled_tokens"], 4032)
        self.assertEqual(calls[1].kwargs["storage_start"], 65536)
        self.assertEqual(cache._l3_ledger_prefetch, {})

    def test_zero_query_has_no_checkpoint_endpoint(self):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(page_size=64)
        cache._l3_lifetime_ledger = Mock()
        cache._l3_trace = None
        cache._prefetch_outcome_stats = defaultdict(int)
        op = SimpleNamespace(
            stats_requested_tokens=4096, deadline_expired=False, storage_hit_count=0,
            all_hash_values=["end"], hash_value=[], request_id="miss", id=20,
            storage_start=65536, stats_tree_kv_tokens=32768, last_hash="anchor",
            query_pool_hit_pages={PoolName.KV: 64},
        )
        cache._account_prefetch_outcome(op, revoked=True)
        event = cache._l3_lifetime_ledger.emit.call_args
        self.assertEqual(event.args[0], "storage_lookup")
        self.assertEqual((event.kwargs["M"], event.kwargs["K"]), (32768, 65536))
        self.assertIsNone(event.kwargs["D"])
        self.assertIsNone(event.kwargs["checkpoint_hash"])

    def test_lookup_events_carry_the_root_to_anchor_path(self):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache._l3_lifetime_ledger = Mock()
        cache.tree_core = Mock()
        cache.tree_core.get_prefix_hash_values.return_value = ["a" * 64, "b" * 64]
        cache.tree_core.get_hash_values.return_value = ["c" * 64]
        cache._ledger_lookup_event("storage_lookup_issue", "r1", 7, 192, 16384, 64, op_id=3)
        event = cache._l3_lifetime_ledger.emit.call_args
        self.assertEqual(event.args[0], "storage_lookup_issue")
        self.assertEqual(event.kwargs["path"], ["a" * 16, "b" * 16, "c" * 16])
        self.assertEqual((event.kwargs["path_pages"], event.kwargs["storage_start"], event.kwargs["op_id"]), (3, 192, 3))
        cache.tree_core.get_prefix_hash_values.assert_called_once_with(7)
        cache.tree_core.get_hash_values.assert_called_once_with(7)

    def test_a_too_short_lookup_is_declined_with_a_ledger_event(self):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.tree_core = Mock(is_eagle=False)  # enable_storage / page_size forward to tree_core
        cache.enable_storage = True
        cache.cache_controller = Mock()
        cache.cache_controller.get_prefetch_submission.return_value = None
        cache.storage_prefetch_retries = Mock()
        cache._storage_prefetch_wait = {}
        cache.host_memory_mode = "cache"
        cache.ongoing_prefetch = {}
        cache.buffer_pipeline = None
        cache.tree_core.prefetch_anchor_info.return_value = (None, None)
        cache.tree_core.get_prefix_hash_values.return_value = ["a" * 64]
        cache.tree_core.get_hash_values.return_value = ["b" * 64]
        cache.page_size = 64
        cache.prefetch_threshold = 8192
        cache._prefetch_outcome_stats = defaultdict(int)
        cache._l3_lifetime_ledger = Mock()
        handle = CacheRequestHandle("short", 0)
        cache.prefetch_from_storage(handle, 7, list(range(1024)), matched_prefix_tokens=list(range(128)))
        self.assertEqual(cache._prefetch_outcome_stats["declined_too_short"], 1)
        event = cache._l3_lifetime_ledger.emit.call_args
        self.assertEqual(event.args[0], "storage_lookup_declined")
        self.assertEqual(
            (event.kwargs["reason"], event.kwargs["storage_start"], event.kwargs["prefetch_length"], event.kwargs["path_pages"]),
            ("too_short", 128, 1024, 2),
        )
        cache.cache_controller.submit_prefetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
