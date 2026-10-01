"""Unit tests for HiCache PP synchronization."""

import inspect
import pickle
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers import scheduler_pp_mixin
from sglang.srt.mem_cache.unified_radix_cache import (
    _HICACHE_PP_ENVELOPE_SIZE,
    _HICACHE_PP_IDENTITY,
    _HICACHE_PP_PREFETCH_START,
    _HICACHE_PP_QUEUE_SLOTS,
    _HICACHE_PP_STORAGE_START,
    _HICACHE_PP_TERMINATE,
    UnifiedRadixCache,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _FakeWork:
    def __init__(self):
        self.waited = False

    def wait(self, timeout=None):
        self.waited = True


class _FakeDirectWork:
    def __init__(self, on_wait=None):
        self.completed = False
        self.on_wait = on_wait
        self.waited = False

    def is_completed(self):
        return self.completed

    def wait(self):
        self.waited = True
        if self.on_wait is not None:
            self.on_wait()
        self.completed = True


class _FakeDirectTransport:
    """Message matching fake where send may precede the destination irecv."""

    def __init__(self):
        self.sends = {}
        self.recvs = {}

    def isend(self, tensor, src, dst, tag):
        work = _FakeDirectWork()
        self.sends[(src, dst, tag)] = (tensor.clone(), work)
        self._match(src, dst, tag)
        return work

    def irecv(self, tensor, src, dst, tag):
        work = _FakeDirectWork()
        self.recvs[(src, dst, tag)] = (tensor, work)
        self._match(src, dst, tag)
        return work

    def _match(self, src, dst, tag):
        key = (src, dst, tag)
        if key not in self.sends or key not in self.recvs:
            return
        sent, send_work = self.sends[key]
        recv, recv_work = self.recvs[key]
        recv.copy_(sent)
        send_work.completed = True
        recv_work.completed = True


class _Holder:
    """Minimal carrier exposing only what _drain_async_work touches."""


class TestPPSyncDrain(CustomTestCase):
    def test_drain_waits_all_and_clears(self):
        holder = _Holder()
        works = [_FakeWork(), _FakeWork(), _FakeWork()]
        holder.work_list = list(works)

        UnifiedRadixCache._drain_async_work(holder)

        self.assertTrue(all(w.waited for w in works))
        self.assertEqual(holder.work_list, [])

    def test_drain_empty_is_noop(self):
        holder = _Holder()
        holder.work_list = []

        UnifiedRadixCache._drain_async_work(holder)

        self.assertEqual(holder.work_list, [])


class TestUnifiedPPSyncBatching(CustomTestCase):
    def _make_cache(self, pp_rank, write_ready, load_ready):
        cache = object.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(
            enable_storage=False,
            write_back_duplicate_reclaim_digest=0,
        )
        cache.pp_rank = pp_rank
        cache.pp_size = 2
        cache.pp_group = object()
        cache.host_memory_mode = "cache"
        cache._hicache_storage_configured = False
        cache._hicache_pp_sync_round = 0
        cache._hicache_pp_prefetch_pending = {}
        cache._hicache_pp_prefetch_results = {}
        cache._hicache_pp_prefetch_keys = {}
        cache._hicache_pp_prefetch_inflight = set()
        cache._hicache_pp_write_acks_consumed = 0
        cache._hicache_pp_write_ack_snapshots = {}
        cache._hicache_pp_round_reservations = {}
        cache._hicache_pp_reserved_counts = [0] * _HICACHE_PP_QUEUE_SLOTS
        cache._hicache_pp_sync_state_logged = False
        cache.work_list = []
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.buffer_pipeline = None
        cache.linker = None
        cache._drain_async_work = MagicMock()
        cache._all_reduce_attn_groups = MagicMock()
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
        follower = self._make_cache(1, [True], [True])
        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)
        leader._apply_hicache_pp_ring_payload(final)
        follower._apply_hicache_pp_ring_payload(final)

        leader.writing_check.assert_called_once_with(finish_count=1)
        leader.loading_check.assert_called_once_with(finish_count=1)

        follower.writing_check.assert_called_once_with(finish_count=1)
        follower.loading_check.assert_called_once_with(finish_count=1)

    def test_ready_count_uses_slowest_pp_ack_prefix(self):
        """A PP0-ready ACK cannot be popped before every stage has it queued."""
        leader = self._make_cache(0, [], [True])
        follower = self._make_cache(1, [], [])
        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)
        leader._apply_hicache_pp_ring_payload(final)
        follower._apply_hicache_pp_ring_payload(final)

        leader.loading_check.assert_called_once_with(finish_count=0)
        follower.loading_check.assert_called_once_with(finish_count=0)

    def test_buffer_mode_follower_drains_rank_local_completions(self):
        cache = self._make_cache(1, [True, False], [True, True])
        cache.host_memory_mode = "buffer_only"
        cache._hicache_storage_configured = True
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

    def _make_storage_cache(self, pp_rank, backup_count):
        cache = object.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(
            enable_storage=True,
            write_back_duplicate_reclaim_digest=0,
        )
        cache.pp_rank = pp_rank
        cache.pp_size = 2
        cache.pp_group = object()
        cache.host_memory_mode = "cache"
        cache._hicache_storage_configured = True
        cache._hicache_pp_sync_round = 0
        cache._hicache_pp_prefetch_pending = {}
        cache._hicache_pp_prefetch_results = {}
        cache._hicache_pp_prefetch_keys = {}
        cache._hicache_pp_prefetch_inflight = set()
        cache._hicache_pp_write_acks_consumed = 0
        cache._hicache_pp_write_ack_snapshots = {}
        cache._hicache_pp_round_reservations = {}
        cache._hicache_pp_reserved_counts = [0] * _HICACHE_PP_QUEUE_SLOTS
        cache._hicache_pp_sync_state_logged = False
        cache.work_list = []
        cache.enable_storage = True
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.buffer_pipeline = None
        cache.linker = None
        cache._drain_async_work = MagicMock()
        cache._all_reduce_attn_groups = MagicMock()
        cache.flush_pending_backups = MagicMock()
        cache.writing_check = MagicMock()
        cache.loading_check = MagicMock()
        cache.dec_host_lock_ref = MagicMock()
        cache.ongoing_backup = {}

        backup_queue = Queue()
        for operation_id in range(backup_count):
            operation = SimpleNamespace(id=operation_id, completed_tokens=1)
            backup_queue.put(operation)
            cache.ongoing_backup[operation_id] = (object(), object())

        cache.cache_controller = SimpleNamespace(
            ack_write_queue=[],
            ack_load_queue=[],
            prefetch_hit_queue=Queue(),
            ack_prefetch_queue=Queue(),
            ack_backup_queue=backup_queue,
            host_mem_release_queue=Queue(),
            extra_host_mem_release_queues={},
            mem_pool_host=MagicMock(),
        )

        def broadcast_pp0_counts(counts, _):
            if counts.numel() == 8:
                counts.copy_(torch.tensor([0, 0, 0, 0, 2, 0, 0, 0]))
            else:
                counts.zero_()

        cache._all_reduce = MagicMock(side_effect=broadcast_pp0_counts)
        return cache

    def test_pp_drain_uses_common_prefix_when_backup_counts_diverge(self):
        """A lagging PP stage must not wait for a nonexistent backup ACK."""
        leader = self._make_storage_cache(pp_rank=0, backup_count=2)
        follower = self._make_storage_cache(pp_rank=1, backup_count=1)
        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)
        leader._apply_hicache_pp_ring_payload(final)
        follower._apply_hicache_pp_ring_payload(final)

        self.assertEqual(leader.cache_controller.ack_backup_queue.qsize(), 1)
        self.assertEqual(follower.cache_controller.ack_backup_queue.qsize(), 0)
        leader.dec_host_lock_ref.assert_called_once()
        follower.dec_host_lock_ref.assert_called_once()

    def test_single_stage_direct_storage_drain_uses_unified_consumer(self):
        cache = self._make_storage_cache(pp_rank=0, backup_count=1)
        cache.pp_size = 1

        self.assertTrue(cache.drain_storage_control_queues())

        self.assertEqual(cache.cache_controller.ack_backup_queue.qsize(), 0)
        cache.dec_host_lock_ref.assert_called_once()

    def test_pp_rejects_divergent_duplicate_reclaim_digest(self):
        leader = self._make_storage_cache(pp_rank=0, backup_count=0)
        follower = self._make_storage_cache(pp_rank=1, backup_count=0)
        leader.tree_core.write_back_duplicate_reclaim_digest = 11
        follower.tree_core.write_back_duplicate_reclaim_digest = 17
        errors = []

        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)
        for cache in (leader, follower):
            try:
                cache._apply_hicache_pp_ring_payload(final)
            except Exception as error:
                errors.append((cache.pp_rank, error))

        self.assertEqual({rank for rank, _ in errors}, {0, 1})
        self.assertTrue(all(isinstance(error, AssertionError) for _, error in errors))
        self.assertTrue(
            all(
                "duplicate-reclaim victims diverged" in str(error)
                for _, error in errors
            )
        )

    def test_storage_config_keeps_mixed_enablement_nonblocking(self):
        leader = self._make_storage_cache(pp_rank=0, backup_count=1)
        follower = self._make_storage_cache(pp_rank=1, backup_count=1)
        follower.enable_storage = False
        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)
        leader._apply_hicache_pp_ring_payload(final)
        follower._apply_hicache_pp_ring_payload(final)

        self.assertEqual(leader.cache_controller.ack_backup_queue.qsize(), 0)
        self.assertEqual(follower.cache_controller.ack_backup_queue.qsize(), 1)
        leader.dec_host_lock_ref.assert_called_once()
        follower.dec_host_lock_ref.assert_not_called()

    def test_different_prefetch_keys_share_fixed_envelope(self):
        leader = self._make_cache(0, [], [])
        follower = self._make_cache(1, [], [])
        for cache in (leader, follower):
            cache._pp_sync = MagicMock()
        leader._register_hicache_pp_prefetch_verdict("terminate", "req-a", True)
        follower._register_hicache_pp_prefetch_verdict("ready", "req-b", False)
        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)

        self.assertEqual(proposal.envelope.numel(), _HICACHE_PP_ENVELOPE_SIZE)
        self.assertEqual(final.envelope.numel(), _HICACHE_PP_ENVELOPE_SIZE)
        leader._pp_sync.assert_not_called()
        follower._pp_sync.assert_not_called()

    def test_ready_count_and_loading_check_do_not_consume_ack_twice(self):
        leader = self._make_cache(0, [], [True])
        follower = self._make_cache(1, [], [True])
        for cache in (leader, follower):
            del cache.writing_check
            del cache.loading_check
            cache.metrics_collector = None
            cache.ongoing_load_back = {cache.pp_rank: (object(), object(), object())}
            cache.dec_lock_ref = MagicMock()
            cache.dec_host_lock_ref = MagicMock()
            cache.tree_core.finish_load_back = MagicMock()
            ack = cache.cache_controller.ack_load_queue[0]
            ack.finish_event.synchronize = MagicMock()
            ack.node_ids = [cache.pp_rank]
            ack.num_tokens_by_pool = {}
            ack.num_bytes = 0
            ack.timing_enabled = False

        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)
        leader.loading_check()
        follower.loading_check()
        leader._apply_hicache_pp_ring_payload(final)
        follower._apply_hicache_pp_ring_payload(final)
        leader.loading_check()
        follower.loading_check()

        self.assertEqual(leader.cache_controller.ack_load_queue, [])
        self.assertEqual(follower.cache_controller.ack_load_queue, [])

    def test_blocking_write_during_pending_round_is_not_consumed_twice(self):
        leader = self._make_cache(0, [True], [])
        follower = self._make_cache(1, [True], [])
        for cache in (leader, follower):
            del cache.writing_check
            cache.metrics_collector = None
            cache.ongoing_write_through = {cache.pp_rank: object()}
            cache._finish_write_through_ack = MagicMock(
                side_effect=lambda ack_id, cache=cache: cache.ongoing_write_through.pop(
                    ack_id
                )
            )
            ack = cache.cache_controller.ack_write_queue[0]
            ack.finish_event.synchronize = MagicMock()
            ack.node_ids = [cache.pp_rank]

        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)
        leader.writing_check(write_back=True)
        follower.writing_check(write_back=True)
        leader._apply_hicache_pp_ring_payload(final)
        follower._apply_hicache_pp_ring_payload(final)

        self.assertEqual(leader.cache_controller.ack_write_queue, [])
        self.assertEqual(follower.cache_controller.ack_write_queue, [])


class TestHiCachePPConsensusRing(CustomTestCase):
    """Regression coverage for piggybacking on the scheduler consensus ring."""

    def setUp(self):
        self.helper = TestUnifiedPPSyncBatching()

    def test_ring_uses_common_ack_prefix(self):
        leader = self.helper._make_cache(0, [], [True])
        follower = self.helper._make_cache(1, [], [])

        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)
        leader._apply_hicache_pp_ring_payload(final)
        follower._apply_hicache_pp_ring_payload(final)

        leader.loading_check.assert_called_once_with(finish_count=0)
        follower.loading_check.assert_called_once_with(finish_count=0)

    def test_single_stage_ring_applies_prefetch_verdict_on_same_round(self):
        leader = self.helper._make_cache(0, [], [])
        leader.pp_size = 1
        self.assertIsNone(
            leader._register_hicache_pp_prefetch_verdict(
                "terminate", "req-shared", True
            )
        )

        proposal = leader._build_hicache_pp_ring_payload()
        self.assertNotIn(
            leader._hicache_pp_prefetch_tag("terminate", "req-shared"),
            leader._hicache_pp_prefetch_results,
        )

        leader._apply_hicache_pp_ring_payload(proposal)
        self.assertTrue(
            leader._register_hicache_pp_prefetch_verdict(
                "terminate", "req-shared", False
            )
        )

    def test_ring_keeps_fixed_shape_without_hicache_collective(self):
        leader = self.helper._make_cache(0, [], [])
        follower = self.helper._make_cache(1, [], [])
        leader._pp_sync = MagicMock()
        follower._pp_sync = MagicMock()
        leader._register_hicache_pp_prefetch_verdict("terminate", "req-a", True)
        follower._register_hicache_pp_prefetch_verdict("ready", "req-b", False)

        with patch.object(torch.distributed, "all_reduce") as all_reduce:
            proposal = leader._build_hicache_pp_ring_payload()
            final = follower._build_hicache_pp_ring_payload(proposal)

        self.assertEqual(proposal.envelope.numel(), _HICACHE_PP_ENVELOPE_SIZE)
        self.assertEqual(final.envelope.numel(), _HICACHE_PP_ENVELOPE_SIZE)
        all_reduce.assert_not_called()
        leader._pp_sync.assert_not_called()
        follower._pp_sync.assert_not_called()

    def test_ring_first_and_last_round_have_no_pending_work(self):
        leader = self.helper._make_cache(0, [], [])
        follower = self.helper._make_cache(1, [], [])

        self.assertFalse(leader._apply_hicache_pp_ring_payload(None))
        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)
        self.assertTrue(leader._apply_hicache_pp_ring_payload(final))
        self.assertTrue(follower._apply_hicache_pp_ring_payload(final))
        self.assertFalse(leader._apply_hicache_pp_ring_payload(final))
        self.assertFalse(follower._apply_hicache_pp_ring_payload(final))

    def test_ring_result_is_applied_at_existing_consensus_consumer(self):
        leader = self.helper._make_cache(0, [], [])
        follower = self.helper._make_cache(1, [], [])
        proposal = leader._build_hicache_pp_ring_payload()
        final = follower._build_hicache_pp_ring_payload(proposal)
        scheduler = scheduler_pp_mixin.SchedulerPPMixin()
        scheduler.tree_cache = SimpleNamespace(
            _apply_hicache_pp_ring_payload=MagicMock()
        )

        forwarded = scheduler.process_bootstrapped_queue(
            scheduler_pp_mixin._PPBootstrapPayload(None, final)
        )

        transport_copy = pickle.loads(pickle.dumps(forwarded))
        self.assertEqual(len(forwarded), 2)
        self.assertEqual(transport_copy.rids, forwarded.rids)
        scheduler.tree_cache._apply_hicache_pp_ring_payload.assert_called_once_with(
            final
        )
        self.assertIs(forwarded.hicache, final)

    def test_overlapping_rounds_offer_each_queue_item_once(self):
        leader = self.helper._make_cache(0, [], [True])
        follower = self.helper._make_cache(1, [], [True])
        first = follower._build_hicache_pp_ring_payload(
            leader._build_hicache_pp_ring_payload()
        )
        second = follower._build_hicache_pp_ring_payload(
            leader._build_hicache_pp_ring_payload()
        )

        self.assertEqual(int(second.envelope[1]), 0)
        for cache in (leader, follower):
            cache._apply_hicache_pp_ring_payload(second)
            cache._apply_hicache_pp_ring_payload(first)
            self.assertFalse(cache._apply_hicache_pp_ring_payload(first))
            self.assertEqual(
                [
                    item.kwargs["finish_count"]
                    for item in cache.loading_check.call_args_list
                ],
                [0, 1],
            )

    def test_unaccepted_local_claim_is_reoffered_after_result(self):
        leader = self.helper._make_storage_cache(0, backup_count=5)
        follower = self.helper._make_storage_cache(1, backup_count=2)
        first = follower._build_hicache_pp_ring_payload(
            leader._build_hicache_pp_ring_payload()
        )
        leader._apply_hicache_pp_ring_payload(first)
        follower._apply_hicache_pp_ring_payload(first)

        next_proposal = leader._build_hicache_pp_ring_payload()

        self.assertEqual(int(first.envelope[_HICACHE_PP_STORAGE_START + 2]), 2)
        self.assertEqual(int(next_proposal.envelope[_HICACHE_PP_STORAGE_START + 2]), 3)

    def test_follower_pp0_only_slots_do_not_reserve_counts(self):
        follower = self.helper._make_cache(1, [], [])
        follower._register_hicache_pp_prefetch_verdict(
            "terminate", "follower-only", True
        )
        before = list(follower._hicache_pp_reserved_counts)

        payload = follower._build_hicache_pp_ring_payload()

        self.assertEqual(
            int(payload.envelope[_HICACHE_PP_TERMINATE]), _HICACHE_PP_IDENTITY
        )
        self.assertTrue(
            torch.all(
                payload.envelope[_HICACHE_PP_PREFETCH_START:] == _HICACHE_PP_IDENTITY
            )
        )
        self.assertEqual(follower._hicache_pp_reserved_counts, before)


class TestHiCachePPPrefetchDirectFanout(CustomTestCase):
    """Regressions for the one-round PP0-only verdict fast path."""

    def setUp(self):
        self.helper = TestUnifiedPPSyncBatching()

    def _make_cache(self, pp_rank):
        cache = self.helper._make_cache(pp_rank, [], [])
        cache.pp_size = 4
        cache._hicache_pp_prefetch_fanout_last_applied_round = 0
        return cache

    @staticmethod
    def _parallel(pp_rank, pp_size=2):
        return SimpleNamespace(
            pp_rank=pp_rank,
            pp_size=pp_size,
            tp_size=1,
            attn_dp_rank=0,
            attn_cp_size=1,
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_cp_rank=0,
        )

    def _make_scheduler(self, cache):
        scheduler = scheduler_pp_mixin.SchedulerPPMixin()
        scheduler.enable_hierarchical_cache = True
        scheduler.tree_cache = cache
        scheduler.world_group = SimpleNamespace(cpu_group=object())
        scheduler._pp_init_hicache_prefetch_fanout()
        return scheduler

    def test_verdict_is_visible_exactly_one_round_later(self):
        """A registered verdict is hidden in k and committed on every stage for k+1."""
        caches = [self._make_cache(rank) for rank in range(4)]
        for cache in caches:
            cache._register_hicache_pp_prefetch_verdict(
                "terminate", "req-shared", cache.pp_rank == 0
            )

        payload = caches[0]._build_hicache_pp_prefetch_fanout(round_id=1)
        tag = caches[0]._hicache_pp_prefetch_tag("terminate", "req-shared")
        self.assertTrue(all(tag not in c._hicache_pp_prefetch_results for c in caches))

        for cache in caches:
            self.assertTrue(cache._apply_hicache_pp_prefetch_fanout(payload))
        self.assertTrue(
            all(
                cache._register_hicache_pp_prefetch_verdict(
                    "terminate", "req-shared", False
                )
                for cache in caches
            )
        )

    def test_all_stages_apply_identical_verdict_vector(self):
        """One PP0 payload, not rank-local completion timing, defines every result map."""
        caches = [self._make_cache(rank) for rank in range(4)]
        for rid, verdict in (("req-a", True), ("req-b", False)):
            caches[0]._register_hicache_pp_prefetch_verdict("terminate", rid, verdict)
        payload = caches[0]._build_hicache_pp_prefetch_fanout(round_id=1)

        for cache in caches:
            cache._apply_hicache_pp_prefetch_fanout(payload.clone())

        self.assertTrue(
            all(
                cache._hicache_pp_prefetch_results
                == caches[0]._hicache_pp_prefetch_results
                for cache in caches[1:]
            )
        )

    def test_replay_is_idempotent_and_out_of_order_is_rejected(self):
        """A stale or skipped direct round must not publish a second verdict."""
        cache = self._make_cache(0)
        first = cache._build_hicache_pp_prefetch_fanout(round_id=1)
        second = cache._build_hicache_pp_prefetch_fanout(round_id=2)

        with self.assertRaisesRegex(RuntimeError, "expected direct verdict round 1"):
            cache._apply_hicache_pp_prefetch_fanout(second)
        self.assertTrue(cache._apply_hicache_pp_prefetch_fanout(first))
        self.assertFalse(cache._apply_hicache_pp_prefetch_fanout(first))

    def test_direct_verdict_does_not_touch_ready_count_reservations(self):
        """The fast path must not regain ownership of R1 queue reservations."""
        cache = self.helper._make_cache(0, [], [True])
        cache._hicache_pp_prefetch_fanout_last_applied_round = 0
        cache._build_hicache_pp_ring_payload()
        reserved = list(cache._hicache_pp_reserved_counts)
        reservations = dict(cache._hicache_pp_round_reservations)

        direct = cache._build_hicache_pp_prefetch_fanout(round_id=1)
        cache._apply_hicache_pp_prefetch_fanout(direct)

        self.assertEqual(cache._hicache_pp_reserved_counts, reserved)
        self.assertEqual(cache._hicache_pp_round_reservations, reservations)

    def test_direct_post_and_commit_keep_ring_wait_out_of_the_cycle(self):
        """The fe21022a k+1 wait cycle returns if post depends on a ring receive."""
        source = inspect.getsource(
            scheduler_pp_mixin.SchedulerPPMixin.event_loop_pp_disagg_prefill
        )

        select = source.index("self.get_new_batch_prefill")
        post = source.index("self._pp_post_hicache_prefetch_fanout")
        ring_receive = source.index("self._pp_commit_comm_work(send_transfer_work)")
        commit = source.index("self._pp_commit_hicache_prefetch_fanout")
        process = source.index("self._process_hicache_events()")
        self.assertLess(select, post)
        self.assertLess(ring_receive, commit)
        self.assertLess(commit, process)

        leader = self._make_scheduler(self._make_cache(0))
        fake_ring_receive = _FakeDirectWork()

        def post_while_ring_is_pending(tensor, dst, group, tag):
            self.assertFalse(fake_ring_receive.is_completed())
            return _FakeDirectWork()

        with (
            patch.object(
                scheduler_pp_mixin, "get_parallel", return_value=self._parallel(0)
            ),
            patch.object(
                torch.distributed,
                "isend",
                side_effect=post_while_ring_is_pending,
            ) as isend,
        ):
            leader._pp_post_hicache_prefetch_fanout()
        isend.assert_called_once()
        self.assertFalse(fake_ring_receive.waited)

    def test_isend_before_follower_irecv_keeps_the_fixed_payload(self):
        """A fast PP0 must not lose V_k while a follower is still posting irecv."""
        leader_cache = self._make_cache(0)
        follower_cache = self._make_cache(1)
        leader_cache._register_hicache_pp_prefetch_verdict(
            "terminate", "req-shared", True
        )
        follower_cache._register_hicache_pp_prefetch_verdict(
            "terminate", "req-shared", False
        )
        leader = self._make_scheduler(leader_cache)
        follower = self._make_scheduler(follower_cache)
        transport = _FakeDirectTransport()

        with (
            patch.object(
                scheduler_pp_mixin, "get_parallel", return_value=self._parallel(0)
            ),
            patch.object(
                torch.distributed,
                "isend",
                side_effect=lambda tensor, dst, group, tag: transport.isend(
                    tensor, 0, dst, tag
                ),
            ),
        ):
            leader._pp_post_hicache_prefetch_fanout()
        self.assertFalse(leader._hicache_pp_prefetch_fanout.works[0].is_completed())

        with (
            patch.object(
                scheduler_pp_mixin, "get_parallel", return_value=self._parallel(1)
            ),
            patch.object(
                torch.distributed,
                "irecv",
                side_effect=lambda tensor, src, group, tag: transport.irecv(
                    tensor, src, 1, tag
                ),
            ),
        ):
            follower._pp_post_hicache_prefetch_fanout()
        self.assertTrue(leader._hicache_pp_prefetch_fanout.works[0].is_completed())

        with (
            patch.object(
                scheduler_pp_mixin, "get_parallel", return_value=self._parallel(1)
            ),
            patch.object(
                scheduler_pp_mixin,
                "attn_cp_tp_broadcast_pyobj",
                side_effect=lambda payload: payload,
            ),
        ):
            follower._pp_commit_hicache_prefetch_fanout()
        with (
            patch.object(
                scheduler_pp_mixin, "get_parallel", return_value=self._parallel(0)
            ),
            patch.object(
                scheduler_pp_mixin,
                "attn_cp_tp_broadcast_pyobj",
                side_effect=lambda payload: payload,
            ),
        ):
            leader._pp_commit_hicache_prefetch_fanout()

        tag = leader_cache._hicache_pp_prefetch_tag("terminate", "req-shared")
        self.assertTrue(leader_cache._hicache_pp_prefetch_results[tag])
        self.assertTrue(follower_cache._hicache_pp_prefetch_results[tag])

    def test_late_direct_work_warns_with_expected_and_actual_round(self):
        """A stuck direct receive stays blocking but emits a round-addressable warning."""
        payload = torch.full((3,), -1, dtype=torch.int64)
        work = _FakeDirectWork(on_wait=lambda: payload.__setitem__(0, 7))

        with (
            patch.object(
                scheduler_pp_mixin.time,
                "monotonic",
                side_effect=(0.0, 31.0),
            ),
            patch.object(scheduler_pp_mixin.time, "sleep"),
            patch.object(scheduler_pp_mixin.logger, "warning") as warning,
        ):
            scheduler_pp_mixin._pp_wait_hicache_prefetch_fanout_work(
                work,
                expected_round_id=7,
                payload=payload,
            )

        self.assertTrue(work.waited)
        warning.assert_called_once()
        self.assertIn("expected_round_id=%d", warning.call_args.args[0])
        self.assertIn("actual_round_id=%s", warning.call_args.args[0])
        self.assertEqual(warning.call_args.args[1:], (7, -1))


if __name__ == "__main__":
    unittest.main()
