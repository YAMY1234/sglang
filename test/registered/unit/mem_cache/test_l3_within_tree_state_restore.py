"""State-only L3 recovery with real CPU radix nodes and mocked storage IO."""

import threading
import time
import unittest
from array import array
from hashlib import sha256
from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle, MatchPrefixParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    MambaStateRestoreOperation,
)
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.base import CacheTransferPhase
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture, match_prefix
from sglang.test.kits.unified_radix_cache.mixins import UnifiedRadixCacheFixtures

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestWithinTreeRestore(UnifiedRadixCacheFixtures, unittest.TestCase):
    cfg = CacheConfig(
        page_size=4,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        num_layers=2,
        full_attention_layer_ids=(0,),
        kv_size=64,
        max_context_len=64,
    )

    def setUp(self):
        # Hashing is a separate native-kernel test surface. Keep this ownership
        # and state-transfer test CPU-only without invoking the JIT builder.
        def cpu_hash(tokens, prior, page_size):
            hashes = []
            for start in range(0, len(tokens), page_size):
                prior = sha256(
                    (prior or b"")
                    + array("q", tokens[start : start + page_size]).tobytes()
                ).digest()
                hashes.append(prior.hex())
            return hashes

        patcher = mock.patch("sglang.srt.mem_cache.utils.get_native_hash", cpu_hash)
        patcher.start()
        self.addCleanup(patcher.stop)

    def fixture(self):
        cache, allocator, req_pool = build_fixture(self.cfg, tree_core_backend="python")
        tokens = list(range(16))
        self._insert(cache, allocator, req_pool, tokens[:4])
        self._insert(cache, allocator, req_pool, tokens[:12])
        tail = match_prefix(cache, tokens[:12]).last_device_node
        self._evict_device_component(cache, tail, ComponentType.MAMBA)
        cache.tree_core.enable_hicache = True
        cache.enable_storage = True
        cache.host_memory_mode = "cache"
        cache.prefetch_threshold = 4
        pool = LogicalHostPool(16, 1)
        cache.components[ComponentType.MAMBA]._mamba_pool_host = pool
        cache.host_pool_group = SimpleNamespace(
            free=lambda value, pool: state_pool.free(value)
        )
        state_pool = pool
        cc = HybridCacheController.__new__(HybridCacheController)
        cc.page_size = 4
        cc.prefetch_queue = Queue()
        cc.prefetch_hits_sync_groups = []
        cc._all_reduce = mock.Mock()
        cc.mem_pool_host = SimpleNamespace(layout_lease=nullcontext)
        hashes = get_storage_hash_str(RadixKey(array("q", tokens)), page_size=4)
        cc.storage_backend = SimpleNamespace(
            batch_exists_mamba_states=lambda keys: [key == hashes[1] for key in keys],
            batch_get_v2=mock.Mock(return_value={PoolName.MAMBA: [True]}),
        )
        cache.cache_controller = cc
        return cache, allocator, pool, cc, tokens

    def start(self, cache, tokens, callback=None):
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
        return cache.prefetch_mamba_state_within_tree(
            CacheRequestHandle("within-tree", 0),
            tokens,
            4,
            12,
            match.last_visited_node_id,
            match.last_host_node,
            callback or mock.Mock(),
        )

    def complete(self, cache, cc):
        op = cc.prefetch_queue.get_nowait()
        cc._storage_hit_query(op)
        self.assertEqual(cc._sync_prefetch_hit_query(op, 0), 0)
        cache._finish_mamba_state_restore(op)
        return op

    def test_disabled_control_cannot_restore_missing_state(self):
        cache, _, pool, cc, tokens = self.fixture()
        with envs.SGLANG_HICACHE_L3_READ_WITHIN_TREE_KV.override(False):
            self.assertFalse(self.start(cache, tokens))
        self.assertTrue(cc.prefetch_queue.empty())
        self.assertEqual(pool.available_size(), 16)
        self.assertEqual(match_prefix(cache, tokens).host_hit_length, 0)

    def test_restores_interior_checkpoint_without_loading_or_reallocating_kv(self):
        cache, allocator, pool, cc, tokens = self.fixture()
        available = allocator.available_size()
        callback = mock.Mock()
        with (
            envs.SGLANG_HICACHE_L3_READ_WITHIN_TREE_KV.override(True),
            envs.SGLANG_HICACHE_L3_SAVED_TOKEN_THRESHOLD.override(True),
        ):
            self.assertTrue(self.start(cache, tokens, callback))
            op = self.complete(cache, cc)
        self.assertEqual(op.state_end_tokens, 8)
        self.assertEqual(op.state_saved_tokens, 4)
        match = match_prefix(cache, tokens)
        self.assertEqual(cache.mamba_storage_match_length(tokens), 8)
        self.assertEqual(match.mamba_host_hit_length, 1)
        full = cache.tree_core.build_hicache_transfers(
            ComponentType.FULL, match.best_match_node, CacheTransferPhase.LOAD_BACK
        )[0]
        self.assertEqual(full.host_indices.numel(), 0)
        state = cache.tree_core.build_hicache_transfers(
            ComponentType.MAMBA, match.best_match_node, CacheTransferPhase.LOAD_BACK
        )[0]
        self.assertEqual(state.host_indices.numel(), 1)
        self.assertEqual(match.full_kv_hit_length, 12)
        self.assertEqual(allocator.available_size(), available)
        self.assertEqual(pool.available_size(), 15)
        transfers = cc.storage_backend.batch_get_v2.call_args.args[0]
        self.assertEqual([t.name for t in transfers], [PoolName.MAMBA])
        self.assertFalse(cache.prefetch_loaded_tokens_by_reqid)
        callback.assert_called_once_with()

    def test_missing_state_and_failed_get_release_only_new_slot(self):
        for missing in (True, False):
            with self.subTest(missing=missing):
                cache, allocator, pool, cc, tokens = self.fixture()
                available = allocator.available_size()
                if missing:
                    cc.storage_backend.batch_exists_mamba_states = lambda keys: (
                        [False] * len(keys)
                    )
                else:
                    cc.storage_backend.batch_get_v2.return_value = {
                        PoolName.MAMBA: [False]
                    }
                with envs.SGLANG_HICACHE_L3_READ_WITHIN_TREE_KV.override(True):
                    self.start(cache, tokens)
                    op = self.complete(cache, cc)
                self.assertEqual(op.state_end_tokens, 0)
                self.assertEqual(pool.available_size(), 16)
                self.assertEqual(allocator.available_size(), available)
                self.assertEqual(match_prefix(cache, tokens).host_hit_length, 0)

    def test_cancel_waits_for_worker_before_releasing_slot_and_pins(self):
        cache, _, pool, cc, tokens = self.fixture()
        callback = mock.Mock()
        with envs.SGLANG_HICACHE_L3_READ_WITHIN_TREE_KV.override(True):
            self.start(cache, tokens, callback)
            cache._cancel_mamba_state_restore(CacheRequestHandle("within-tree", 0))
            self.assertEqual(pool.available_size(), 15)
            self.complete(cache, cc)
        self.assertEqual(pool.available_size(), 16)
        callback.assert_not_called()

    def test_checkpoint_at_k_enters_host_lru_only_after_pin_release(self):
        cache, _, pool, cc, tokens = self.fixture()
        cc.storage_backend.batch_exists_mamba_states = lambda keys: [False, True]
        with envs.SGLANG_HICACHE_L3_READ_WITHIN_TREE_KV.override(True):
            self.start(cache, tokens)
            op = self.complete(cache, cc)
        self.assertEqual(op.state_end_tokens, 12)
        match = match_prefix(cache, tokens)
        node = cache.tree_core.node_by_id(match.best_match_node)
        self.assertEqual(node.component_data[ComponentType.MAMBA].host_lock_ref, 0)
        self.assertTrue(
            cache.tree_core.host_lru_lists[ComponentType.MAMBA].in_list(node)
        )
        self.assertEqual(cache.mamba_storage_match_length(tokens), 12)
        self.assertEqual(pool.available_size(), 15)

    def test_no_host_slot_still_enqueues_and_releases_pins_after_consensus(self):
        cache, _, pool, cc, tokens = self.fixture()
        with (
            envs.SGLANG_HICACHE_L3_READ_WITHIN_TREE_KV.override(True),
            mock.patch.object(
                cache.components[ComponentType.MAMBA],
                "alloc_prefetch_staging",
                return_value=None,
            ),
        ):
            self.assertTrue(self.start(cache, tokens))
            op = self.complete(cache, cc)
        self.assertEqual(op.state_end_tokens, 0)
        self.assertEqual(op.restore_locks, [])
        self.assertEqual(pool.available_size(), 16)
        self.assertEqual(cache._prefetch_outcome_stats["within_tree_alloc_failed"], 1)

    def test_delayed_peer_only_blocks_worker_not_intake(self):
        cache, _, _, cc, tokens = self.fixture()
        peer_arrived, worker_waiting = threading.Event(), threading.Event()

        def wait_for_peer(*args):
            worker_waiting.set()
            if not peer_arrived.wait(2):
                raise AssertionError("test peer never arrived")

        cc._all_reduce.side_effect = wait_for_peer
        with envs.SGLANG_HICACHE_L3_READ_WITHIN_TREE_KV.override(True):
            self.assertTrue(self.start(cache, tokens))
            op = cc.prefetch_queue.get_nowait()
            cc._storage_hit_query(op)
            worker = threading.Thread(target=lambda: cc._sync_prefetch_hit_query(op, 0))
            worker.start()
            self.assertTrue(worker_waiting.wait(1))
            # The scheduler is free to relay the request while its worker waits.
            self.assertTrue(worker.is_alive())
            peer_arrived.set()
            worker.join(2)
            self.assertFalse(worker.is_alive())
            cache._finish_mamba_state_restore(op)


class TestNonBlockingStateRestoreLane(UnifiedRadixCacheFixtures, unittest.TestCase):
    """SGLANG_HICACHE_L3_WITHIN_TREE_NONBLOCKING: own lane, gate, no chaining, deadline."""

    cfg = TestWithinTreeRestore.cfg
    setUp = TestWithinTreeRestore.setUp
    fixture = TestWithinTreeRestore.fixture
    start = TestWithinTreeRestore.start

    def lane_fixture(self, min_gap=4, deadline=2):
        cache, allocator, pool, cc, tokens = self.fixture()
        cc.state_restore_lane = True
        cc.state_restore_queue = Queue()
        cc.state_restore_done_queue = Queue()
        cc.state_sync_groups = ["state"]
        cc.prefetch_hits_sync_groups = ["bulk"]
        cc.pp_prefetch_decisions = {}
        for patcher in (
            envs.SGLANG_HICACHE_L3_READ_WITHIN_TREE_KV.override(True),
            envs.SGLANG_HICACHE_L3_WITHIN_TREE_NONBLOCKING.override(True),
            envs.SGLANG_HICACHE_L3_WITHIN_TREE_MIN_GAP.override(min_gap),
            envs.SGLANG_HICACHE_L3_WITHIN_TREE_DEADLINE_SKIPS.override(deadline),
        ):
            patcher.__enter__()
            self.addCleanup(patcher.__exit__, None, None, None)
        return cache, allocator, pool, cc, tokens

    def test_gate_skips_short_stateless_span(self):
        cache, _, pool, cc, tokens = self.lane_fixture(min_gap=12)
        self.assertFalse(self.start(cache, tokens))  # 12 - 4 = 8 < 12
        self.assertTrue(cc.state_restore_queue.empty())
        self.assertTrue(cc.prefetch_queue.empty())
        self.assertEqual(pool.available_size(), 16)

    def test_restore_runs_on_lane_groups_and_drains_from_done_queue(self):
        cache, _, pool, cc, tokens = self.lane_fixture()
        self.assertTrue(self.start(cache, tokens, None))
        self.assertTrue(cc.prefetch_queue.empty())
        op = cc.state_restore_queue.get_nowait()
        cc.run_state_restore(op)
        groups = {call.args[2][0] for call in cc._all_reduce.call_args_list}
        self.assertEqual(groups, {"state"})
        self.assertEqual(cc.state_sync_stats["state_bitmap_n"], 1)
        self.assertEqual(cc.state_sync_stats["state_done_n"], 1)
        cache._drain_state_restores(cc.state_restore_done_queue.qsize())
        self.assertEqual(op.state_end_tokens, 8)
        self.assertEqual(cache.mamba_storage_match_length(tokens), 8)
        self.assertEqual(pool.available_size(), 15)
        self.assertFalse(cache._mamba_state_restores)

    def test_deadline_admits_without_state_then_late_read_still_attaches(self):
        cache, _, pool, cc, tokens = self.lane_fixture(deadline=2)
        request = CacheRequestHandle("within-tree", 0)
        self.assertTrue(self.start(cache, tokens, None))
        self.assertFalse(cache.check_prefetch_progress(request))
        self.assertFalse(cache.check_prefetch_progress(request))
        self.assertTrue(cache.check_prefetch_progress(request))  # third pass: expired
        stats = cache._prefetch_outcome_stats
        self.assertEqual(stats["within_tree_admission_skips"], 2)
        self.assertEqual(stats["within_tree_expired"], 1)
        self.assertNotIn(request, cache._mamba_state_restores)
        self.assertEqual(pool.available_size(), 15)  # the worker still owns the slot
        op = cc.state_restore_queue.get_nowait()
        cc.run_state_restore(op)
        cache._drain_state_restores(1)
        self.assertEqual(cache.mamba_storage_match_length(tokens), 8)
        self.assertEqual(op.restore_locks, [])

    def test_late_restore_does_not_drop_a_newer_pending_entry(self):
        cache, _, _, cc, tokens = self.lane_fixture(deadline=0)
        request = CacheRequestHandle("within-tree", 0)
        self.assertTrue(self.start(cache, tokens, None))
        self.assertTrue(cache.check_prefetch_progress(request))  # expires at once
        old = cc.state_restore_queue.get_nowait()
        self.assertTrue(self.start(cache, tokens, None))  # e.g. a retracted re-queue
        newer = cache._mamba_state_restores[request]
        self.assertIsNot(old, newer)
        cc.run_state_restore(old)
        cache._drain_state_restores(1)
        self.assertIs(cache._mamba_state_restores[request], newer)


class TestStateLaneCollectiveOrder(unittest.TestCase):
    """Two fake ranks enqueue state and bulk ops in different interleavings. Each
    gloo group must see its ops in the same order on both ranks, as a real
    collective pairs the k-th call of one rank with the k-th call of the other."""

    class PairedGroups:
        def __init__(self):
            self.cond = threading.Condition()
            self.posted = {}  # (group, k) -> {rank: tensor}
            self.calls = {}  # (rank, group) -> count
            self.mismatch = []

        def all_reduce(self, rank, tensor, op, groups):
            group = groups[0]
            with self.cond:
                k = self.calls.get((rank, group), 0)
                self.calls[(rank, group)] = k + 1
                slot = self.posted.setdefault((group, k), {})
                slot[rank] = tensor.clone()
                self.cond.notify_all()
                if not self.cond.wait_for(lambda: len(slot) == 2, timeout=5):
                    raise AssertionError(f"rank {rank} waited alone on {group}[{k}]")
                a, b = slot[0], slot[1]
                if a.shape != b.shape or a.reshape(-1)[0] != b.reshape(-1)[0]:
                    self.mismatch.append((group, k))
                    return
                tensor.copy_(torch.minimum(a, b))

    def controller(self, rank, comm):
        cc = HybridCacheController.__new__(HybridCacheController)
        cc.page_size = 64
        cc.state_restore_lane = True
        cc.state_restore_queue = Queue()
        cc.state_restore_done_queue = Queue()
        cc.state_sync_groups = ["state"]
        cc.prefetch_hits_sync_groups = ["bulk"]
        cc.storage_stop_event = threading.Event()
        cc._all_reduce = lambda tensor, op, groups: comm.all_reduce(rank, tensor, op, groups)
        cc.query_mamba_states = lambda hashes: [False] * len(hashes)
        cc.load_mamba_state = mock.Mock(return_value=True)
        cc.bulk_queue = Queue()
        return cc

    def state_op(self, marker):
        # matched_tokens doubles as the op's identity in the packed bitmap's head.
        op = MambaStateRestoreOperation(
            CacheRequestHandle(f"s{marker}", 0),
            RadixKey(array("q", range(256))),
            64 * marker,
            256,
            torch.tensor([1]),
            1,
        )
        op.state_hashes = [str(i) for i in range(4)]
        op.state_exists = [False, False, False, True]
        return op

    def test_lane_order_is_independent_of_interleaving_with_bulk_ops(self):
        comm = self.PairedGroups()
        ranks = [self.controller(0, comm), self.controller(1, comm)]
        interleavings = (["s1", "b1", "s2", "b2", "s3"], ["b1", "b2", "s1", "s2", "s3"])
        for cc, order in zip(ranks, interleavings):
            for item in order:
                if item.startswith("s"):
                    cc.state_restore_queue.put(self.state_op(int(item[1:])))
                else:
                    cc.bulk_queue.put(int(item[1:]))

        def bulk_worker(cc):
            while not cc.bulk_queue.empty():
                marker = cc.bulk_queue.get_nowait()
                cc._all_reduce(torch.tensor([marker, 7]), None, cc.prefetch_hits_sync_groups)

        with mock.patch.object(
            HybridCacheController, "_storage_hit_query", lambda self, op: None
        ):
            threads = []
            for cc in ranks:
                threads.append(threading.Thread(target=cc.state_restore_thread_func))
                threads.append(threading.Thread(target=bulk_worker, args=(cc,)))
            for t in threads:
                t.start()
            for cc in ranks:
                deadline = time.monotonic() + 5
                while cc.state_restore_done_queue.qsize() < 3 and time.monotonic() < deadline:
                    time.sleep(0.01)
                cc.storage_stop_event.set()
            for t in threads:
                t.join(5)
                self.assertFalse(t.is_alive())
        self.assertEqual(comm.mismatch, [])
        for cc in ranks:
            done = [cc.state_restore_done_queue.get_nowait().matched_tokens for _ in range(3)]
            self.assertEqual(done, [64, 128, 192])
        self.assertEqual(comm.calls[(0, "state")], comm.calls[(1, "state")])
        self.assertEqual(comm.calls[(0, "bulk")], 2)

    def test_control_one_fifo_with_rank_local_interleaving_misorders(self):
        # The hazard the lane avoids: one queue/group, items in each rank's own order.
        comm = self.PairedGroups()
        interleavings = ([101, 1, 102, 2], [1, 2, 101, 102])

        def worker(rank, items):
            for marker in items:
                comm.all_reduce(rank, torch.tensor([marker, 7]), None, ["bulk"])

        threads = [threading.Thread(target=worker, args=(r, o)) for r, o in enumerate(interleavings)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        self.assertNotEqual(comm.mismatch, [])


class TestStateOnlyQueryAndConsensus(unittest.TestCase):
    def operation(self, *, endpoint=8192, minimum=8192):
        cc = HybridCacheController.__new__(HybridCacheController)
        cc.page_size = 64
        cc.prefetch_hits_sync_groups = []
        cc._all_reduce = mock.Mock()
        cc.load_mamba_state = mock.Mock(return_value=True)
        op = MambaStateRestoreOperation(
            CacheRequestHandle("r", 0),
            RadixKey(array("q", range(16384))),
            0,
            16384,
            torch.tensor([1]),
            minimum,
        )
        op.state_hashes = [str(i) for i in range(256)]
        op.state_exists = [False] * 256
        op.state_exists[endpoint // 64 - 1] = True
        return cc, op

    def test_saved_threshold_exact_boundary_and_b_without_c(self):
        for endpoint, minimum, accepted in (
            (8128, 8192, False),
            (8192, 8192, True),
            (64, 1, True),
        ):
            cc, op = self.operation(endpoint=endpoint, minimum=minimum)
            cc._sync_prefetch_hit_query(op, 0)
            self.assertEqual(bool(op.state_end_tokens), accepted)
            self.assertEqual(cc.load_mamba_state.called, accepted)

    def test_sparse_rank_disagreement_never_selects_a_missing_checkpoint(self):
        cc, op = self.operation()
        op.state_exists[255] = True

        def peer_bitmap(packed, *args):
            if packed.numel() > 1:
                packed[256] = 0

        cc._all_reduce.side_effect = peer_bitmap
        cc._sync_prefetch_hit_query(op, 0)
        self.assertEqual(op.state_end_tokens, 8192)
        cc.load_mamba_state.assert_called_once_with("127", op.state_host_indices)

    def test_peer_get_failure_prevents_attach(self):
        cc, op = self.operation()
        cc._all_reduce.side_effect = lambda packed, *args: (
            packed.zero_() if packed.numel() == 1 else None
        )
        cc._sync_prefetch_hit_query(op, 0)
        self.assertEqual(op.state_end_tokens, 0)
        self.assertEqual(op.state_outcome, "transfer_failed")

    def test_state_query_requires_all_components_but_no_kv_objects(self):
        store = MooncakeStore.__new__(MooncakeStore)
        store._get_hybrid_page_component_keys = mock.Mock(
            return_value=(["a_t", "a_c", "b_t", "b_c"], 2)
        )
        store._tag_keys = lambda keys: keys
        store._batch_exist = mock.Mock(return_value=[1, 1, 1, 0])
        store.batch_exists = mock.Mock(side_effect=AssertionError("must not probe KV"))
        self.assertEqual(store.batch_exists_mamba_states(["a", "b"]), [True, False])
        store.batch_exists.assert_not_called()


if __name__ == "__main__":
    unittest.main()
