"""Unit tests for HiCache PP synchronization."""

import unittest
from collections import deque
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache import pmicro_hoststats
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestUnifiedPPSyncBatching(unittest.TestCase):
    def _make_cache(self, pp_rank, write_ready, load_ready):
        cache = object.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(
            enable_storage=False,
            write_back_duplicate_reclaim_digest=0,
        )
        cache.pp_rank = pp_rank
        cache.pp_size = 2
        cache.host_memory_mode = "cache"
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.buffer_pipeline = None
        cache.linker = None
        cache._drain_async_work = MagicMock()
        cache._all_reduce = MagicMock()
        cache.writing_check = MagicMock()
        cache.loading_check = MagicMock()
        cache.cache_controller = SimpleNamespace(
            start_writing=MagicMock(),
            ack_write_queue=[
                SimpleNamespace(
                    finish_event=SimpleNamespace(query=MagicMock(return_value=ready))
                )
                for ready in write_ready
            ],
            ack_load_queue=[
                SimpleNamespace(
                    finish_event=SimpleNamespace(query=MagicMock(return_value=ready))
                )
                for ready in load_ready
            ],
        )
        return cache

    def test_pp_batches_write_and_load_counts_once(self):
        leader = self._make_cache(0, [True, False], [True, True])
        leader.check_hicache_events()

        leader._all_reduce.assert_called_once()
        self.assertEqual(leader._all_reduce.call_args.args[0].tolist(), [1, 2, 0, 0])
        leader.writing_check.assert_called_once_with(finish_count=1)
        leader.loading_check.assert_called_once_with(finish_count=2)

        follower = self._make_cache(1, [True], [True])

        def reduce_to_min(counts, _):
            counts.copy_(torch.tensor([1, 1, 0, 0], dtype=torch.int64))

        follower._all_reduce.side_effect = reduce_to_min
        follower.check_hicache_events()

        for queue in (
            follower.cache_controller.ack_write_queue,
            follower.cache_controller.ack_load_queue,
        ):
            queue[0].finish_event.query.assert_not_called()
        follower._all_reduce.assert_called_once()
        follower.writing_check.assert_called_once_with(finish_count=1)
        follower.loading_check.assert_called_once_with(finish_count=1)

    def test_buffer_mode_follower_drains_rank_local_completions(self):
        cache = self._make_cache(1, [True, False], [True, True])
        cache.host_memory_mode = "buffer_only"
        cache.tree_core.enable_storage = True
        cache._all_reduce_attn_groups = MagicMock()
        cache._drain_storage_control_queues_impl = MagicMock()
        cc = cache.cache_controller
        for size, name in enumerate(
            (
                "prefetch_hit_queue",
                "ack_prefetch_queue",
                "ack_backup_queue",
                "host_mem_release_queue",
            ),
            start=1,
        ):
            queue = Queue()
            for _ in range(size):
                queue.put(object())
            setattr(cc, name, queue)

        cache.check_hicache_events()

        cache._all_reduce.assert_not_called()
        cache._all_reduce_attn_groups.assert_called_once()
        cache.writing_check.assert_called_once_with(finish_count=1)
        cache.loading_check.assert_called_once_with(finish_count=2)
        cache._drain_storage_control_queues_impl.assert_called_once_with(
            n_storage_hit=1,
            n_ack_prefetch=2,
            n_backup=3,
            n_release=4,
            extra_release_counts={},
            log_metrics=True,
        )


class _NoWaitQueue(Queue):
    """An ack queue whose get() records, instead of performing, a blocking wait."""

    def __init__(self):
        super().__init__()
        self.would_block = 0

    def get(self, block=True, timeout=None):
        if self.empty():
            self.would_block += 1
            raise AssertionError("drain would block on a missing storage ack")
        return super().get(block=False)


class TestL3AckLag(unittest.TestCase):
    """SGLANG_HICACHE_L3_ACK_LAG_STEPS: PP0 relays a lagged storage write-ack count."""

    def _rank(self, pp_rank, lag=0):
        cache = object.__new__(UnifiedRadixCache)
        cache.pp_rank = pp_rank
        cache.pp_size = 2
        cache.host_memory_mode = "cache"
        cache.tree_core = SimpleNamespace(
            enable_storage=True, write_back_duplicate_reclaim_digest=0
        )
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.buffer_pipeline = None
        cache._shm_count_exchange = lambda: None
        cache.ongoing_backup = {}
        cache.ongoing_prefetch = {}
        cache._write_behind_inflight = {}
        cache._write_behind_mamba_inflight = {}
        cache._l3_wb_ack_credit = False
        cache.storage_existence_cache = MagicMock()
        cache.released = []
        cache.dec_host_lock_ref = lambda node_id, _: cache.released.append(node_id)
        cache.cache_controller = SimpleNamespace(
            ack_write_queue=[],
            ack_load_queue=[],
            extra_host_mem_release_queues={},
            prefetch_hit_queue=Queue(),
            ack_prefetch_queue=Queue(),
            ack_backup_queue=_NoWaitQueue(),
            host_mem_release_queue=Queue(),
        )
        with envs.SGLANG_HICACHE_L3_ACK_LAG_STEPS.override(lag):
            cache._init_ack_lag()
        return cache

    def _put_ack(self, cache, op_id):
        cache.ongoing_backup[op_id] = (op_id, None)
        cache.cache_controller.ack_backup_queue.put(
            SimpleNamespace(
                id=op_id, hash_value=None, pool_transfers=(), completed_tokens=0
            )
        )

    def _step(self, cache, relayed=None):
        """One scheduler step: sync counts (PP0 relays, PP1 receives), then drain."""
        sent = []

        def relay(counts, _):
            if relayed is not None:
                counts.copy_(relayed)
            sent.append(counts.clone())

        cache._all_reduce = relay
        _, _, sizes, _ = cache._sync_hicache_ready_counts()
        cache._drain_storage_control_queues_impl(
            n_storage_hit=sizes[0],
            n_ack_prefetch=sizes[1],
            n_backup=sizes[2],
            n_release=sizes[3],
            extra_release_counts={},
            log_metrics=False,
        )
        return sizes[2], sent[0]

    def _run(self, lag, delay, steps=12, tail=4):
        """One put per step on both ranks for ``steps`` steps; PP0's ack lands at
        once, PP1's ``delay`` steps later. Returns per-step published counts,
        PP0 ready counts and both ranks."""
        leader, follower = self._rank(0, lag), self._rank(1, lag)
        published, ready = [], []
        for t in range(steps + delay + tail):
            if t < steps:
                self._put_ack(leader, t)
            if 0 <= t - delay < steps:
                self._put_ack(follower, t - delay)
            ready.append(leader.cache_controller.ack_backup_queue.qsize())
            n, relayed = self._step(leader)
            published.append(n)
            self._step(follower, relayed)
        return published, ready, leader, follower

    def test_default_off_publishes_ready_count(self):
        leader = self._rank(0)
        self.assertEqual(leader._l3_ack_lag_steps, 0)
        self.assertIsNone(leader._l3_ack_lag_hist)
        for op_id in range(3):
            self._put_ack(leader, op_id)
        self.assertEqual(self._step(leader)[0], 3)
        self.assertEqual(leader.released, [0, 1, 2])
        self.assertEqual(leader._l3_ack_removed, 0)

    def test_default_off_follower_blocks_on_late_ack(self):
        with self.assertRaisesRegex(AssertionError, "would block"):
            self._run(lag=0, delay=1)

    def test_lag_only_on_pp0_of_a_cache_mode_pipeline(self):
        self.assertEqual(self._rank(0, lag=2)._l3_ack_lag_steps, 2)
        self.assertEqual(self._rank(1, lag=2)._l3_ack_lag_steps, 0)
        single = object.__new__(UnifiedRadixCache)
        single.pp_rank, single.pp_size, single.host_memory_mode = 0, 1, "cache"
        buffer = object.__new__(UnifiedRadixCache)
        buffer.pp_rank, buffer.pp_size, buffer.host_memory_mode = 0, 2, "buffer_only"
        for cache in (single, buffer):
            with envs.SGLANG_HICACHE_L3_ACK_LAG_STEPS.override(2):
                cache._init_ack_lag()
            self.assertEqual(cache._l3_ack_lag_steps, 0)

    def test_lag_publishes_acks_ready_d_steps_ago_and_never_more_than_ready(self):
        for lag in (1, 2, 3):
            with self.subTest(lag=lag):
                published, ready, leader, _ = self._run(lag, delay=0, steps=8)
                # One ack per step, ready at once: each is released lag steps later.
                self.assertEqual(published[:lag], [0] * lag)
                self.assertEqual(published[lag : lag + 8], [1] * 8)
                self.assertTrue(all(p <= r for p, r in zip(published, ready)))
                self.assertEqual(sum(published), 8)
                self.assertEqual(leader.released, list(range(8)))

    def test_follower_with_acks_within_lag_never_blocks_and_drains_same_order(self):
        for lag, delay in ((1, 1), (2, 1), (2, 2), (3, 2)):
            with self.subTest(lag=lag, delay=delay):
                published, _, leader, follower = self._run(lag, delay)
                queue = follower.cache_controller.ack_backup_queue
                self.assertEqual(queue.would_block, 0)
                self.assertEqual(leader.released, list(range(12)))
                self.assertEqual(follower.released, leader.released)
                self.assertEqual(sum(published), 12)

    def test_bursty_acks_and_out_of_band_removals_stay_bounded(self):
        leader = self._rank(0, lag=2)
        queue = leader.cache_controller.ack_backup_queue
        arrivals = [3, 0, 5, 1, 0, 0, 2, 0, 0, 0]
        published = []
        for t, k in enumerate(arrivals):
            for i in range(k):
                self._put_ack(leader, 10 * t + i)
            if t == 3:
                # A write-back wait takes one ack off the front outside the sync.
                leader._drain_storage_control_queues_impl(0, 0, 1, 0, {}, False)
            ready = queue.qsize()
            n = self._step(leader)[0]
            self.assertLessEqual(n, ready)
            published.append(n)
        self.assertEqual(published, [0, 0, 3, 0, 4, 1, 0, 0, 2, 0])
        self.assertTrue(queue.empty())
        self.assertEqual(leader._l3_ack_removed, 11)

    def test_local_flush_drains_lagged_acks_and_restarts_counts(self):
        leader = self._rank(0, lag=3)
        for op_id in range(4):
            self._put_ack(leader, op_id)
        self.assertEqual(self._step(leader)[0], 0)
        leader.drain_storage_control_queues_local()
        self.assertEqual(leader.released, [0, 1, 2, 3])
        self.assertTrue(leader.cache_controller.ack_backup_queue.empty())
        self.assertEqual((leader._l3_ack_removed, len(leader._l3_ack_lag_hist)), (0, 0))

    def test_drain_times_blocking_backup_gets_under_hoststats(self):
        pmicro_hoststats.take_ackdrain_field()
        follower = self._rank(1)
        follower.cache_controller.ack_backup_queue = Queue()
        for op_id in range(2):
            self._put_ack(follower, op_id)
        with patch.object(pmicro_hoststats, "ON", True):
            follower._drain_storage_control_queues_impl(0, 0, 2, 0, {}, False)
        self.assertEqual(pmicro_hoststats._ackdrain["n"], 2)
        self.assertEqual(pmicro_hoststats._ackdrain["blocked"], 0)
        self.assertRegex(
            pmicro_hoststats.take_ackdrain_field(),
            r"^ackdrain n=2 blocked=0 block_ms=[\d.]+ max_ms=[\d.]+$",
        )
        self.assertEqual(
            pmicro_hoststats.take_ackdrain_field(),
            "ackdrain n=0 blocked=0 block_ms=0.0 max_ms=0.000",
        )


class TestUnifiedPPSyncWindow(unittest.TestCase):
    def test_window_waits_only_rounds_older_than_the_window(self):
        for window in (1, 3):
            with self.subTest(window=window):
                cache = object.__new__(UnifiedRadixCache)
                cache.pp_sync_window = window
                cache.pp_sync_rounds = deque()
                cache.work_list = []
                rounds = []
                for _ in range(5):
                    cache._drain_async_work()
                    rounds.append([MagicMock(), MagicMock()])
                    cache.work_list.extend(rounds[-1])
                cache._drain_async_work()
                # Round k's sends are waited when round k + window opens.
                waited = [all(w.wait.called for w in r) for r in rounds]
                self.assertEqual(waited, [k + window <= 5 for k in range(5)])
                self.assertFalse(any(w.wait.called for w in cache.work_list))


if __name__ == "__main__":
    unittest.main()
