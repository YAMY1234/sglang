import gc
import os
import tempfile
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from sglang.srt.disaggregation import l16_loopstat
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.decode_hicache_mixin import (
    HiCacheRestoreGatedKVReceiver,
)
from sglang.srt.disaggregation.fake.conn import FakeKVManager, FakeKVReceiver
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    poll_and_all_reduce_with_staging,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ReqKvInfo
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.separate_buffer_allocator_double import (
    bind_separate_buffer_capacity,
)
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class FakeReceiver:
    def __init__(self):
        self.clear_called = False
        self.conclude_state = None

    def clear(self):
        self.clear_called = True

    def failure_exception(self):
        return None


class TestDecodeQueueCleanup(CustomTestCase):
    def setUp(self):
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def test_paged_swa_retraction_resume_uses_physical_page_budget(self):
        # resume_retracted_reqs reads the retraction backend off the disagg
        # bag, so the case publishes a config instead of injecting one.
        override = get_context().override_server_args(
            disaggregation_decode_retraction_backup="cpu_tensor"
        )
        override.install()
        self.addCleanup(override.restore)

        page_size = 128
        fill_len = 574
        physical_tokens_per_req = 5 * page_size
        physical_available = 18 * page_size

        reqs = [
            SimpleNamespace(
                rid=f"req-{i}",
                origin_input_ids=[0] * fill_len,
                output_ids=[],
                is_retracted=True,
                retraction_backup=None,
                load_kv_cache=MagicMock(),
            )
            for i in range(4)
        ]

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.retracted_queue = reqs.copy()
        queue.num_reserved_decode_tokens = 0
        queue.req_to_token_pool = SimpleNamespace(available_size=lambda: len(reqs))
        queue.token_to_kv_pool_allocator = MagicMock(page_size=page_size)
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue.tree_cache = MagicMock()
        queue.scheduler = SimpleNamespace(
            sliding_window_size=2047,
            server_args=SimpleNamespace(disable_radix_cache=True),
            enable_overlap=False,
        )
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._swa_aware_allocatable_token_budgets = MagicMock(
            return_value=(physical_available, physical_available)
        )
        queue._allocatable_token_budgets = MagicMock(
            side_effect=lambda **_: physical_available
        )
        queue._swa_tail_allocatable_token_budget = MagicMock(
            side_effect=lambda **_: physical_available
        )

        def pre_alloc(_req):
            nonlocal physical_available
            self.assertGreaterEqual(physical_available, physical_tokens_per_req)
            physical_available -= physical_tokens_per_req

        queue._pre_alloc = MagicMock(side_effect=pre_alloc)

        resumed = queue.resume_retracted_reqs()

        self.assertEqual(resumed, reqs[:3])
        self.assertEqual(queue.retracted_queue, reqs[3:])
        self.assertEqual(physical_available, 3 * page_size)
        self.assertEqual(queue._pre_alloc.call_count, 3)

    def test_prealloc_abort_clears_receiver_before_removing_request(self):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="abort-prealloc",
            bootstrap_room=42,
            kv=ReqKvInfo(),
            finished_reason=None,
            return_logprob=False,
        )
        decode_req = SimpleNamespace(
            req=req, kv_receiver=receiver, waiting_for_input=True
        )

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.tp_rank = 0
        queue.gloo_group = object()
        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue._resolve_pending_reqs = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        # `_uses_swa_reservation` consults the allocator once tail prealloc is
        # off, so this abort path needs one even though it never allocates.
        queue.token_to_kv_pool_allocator = MagicMock()
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.enable_lora = False
        scheduler.metrics_reporter.enable_metrics = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        with patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            return_value=[KVPoll.Failed],
        ) as poll:
            preallocated, failed = queue.pop_preallocated()

        poll.assert_called_once_with([receiver], queue.gloo_group)
        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_prealloc_abort_also_drops_from_pending_reqs(self):
        # Same DecodeRequest lives in both queue and pending_reqs (add() slow
        # path). Aborting must drop it from both, and compare by identity since
        # DecodeRequest's dataclass __eq__ would compare the tensor receiver.
        class BadEqReceiver(FakeReceiver):
            def __eq__(self, other):
                raise TypeError("use identity comparison, not value equality")

            __hash__ = object.__hash__

        receiver = BadEqReceiver()
        req = SimpleNamespace(
            rid="abort-shared",
            finished_reason=FINISH_ABORT("aborted"),
            kv=ReqKvInfo(),
            return_logprob=False,
        )
        decode_req = SimpleNamespace(req=req, kv_receiver=receiver)

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = [decode_req]  # same object, dual ownership
        queue.retracted_queue = []
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        # `_uses_swa_reservation` consults the allocator once tail prealloc is
        # off, so this abort path needs one even though it never allocates.
        queue.token_to_kv_pool_allocator = MagicMock()
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.enable_lora = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        # Must not raise on the receiver __eq__ above.
        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(all(r is not decode_req for r in queue.pending_reqs))
        self.assertIsNone(decode_req.kv_receiver)

    def test_swa_reclaim_failure_rejects_only_request(self):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="swa-reclaim-failed",
            origin_input_ids=[1, 2, 3],
            kv=ReqKvInfo(),
            output_ids=[],
            finished_reason=None,
            return_logprob=False,
            sampling_params=SimpleNamespace(max_new_tokens=1),
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            waiting_for_input=True,
            is_rebootstrap=False,
        )

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = [decode_req]
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._swa_aware_allocatable_token_budgets = MagicMock(
            return_value=(1024, 1024)
        )
        queue._prealloc_required_tokens = MagicMock(return_value=(3, 3))
        queue._prealloc_kv_lens = MagicMock(return_value=(3, 3))
        queue._reclaim_swa_tail_capacity = MagicMock(
            return_value=(
                "SWA eviction insufficient: needed=64, available=0, "
                "req=swa-reclaim-failed"
            )
        )
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        queue._pre_alloc = MagicMock()
        # SWA tail pre-allocation runs on a paged pool.
        queue.token_to_kv_pool_allocator = MagicMock(page_size=64)
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue.tree_cache = MagicMock()
        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 1
        # Non-hybrid pools have no mamba allocator; MagicMock would otherwise
        # auto-create one and break the `available_size() <= 0` comparison in
        # pop_preallocated.
        queue.req_to_token_pool.mamba_allocator = None
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 1

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.enable_lora = False
        scheduler.server_args.disaggregation_decode_enable_radix_cache = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertEqual(queue.pending_reqs, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        queue._pre_alloc.assert_not_called()
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_ensure_prefill_info_tolerates_cleared_receiver(self):
        # A req whose kv_receiver was already cleared must not crash on .abort().
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._max_ensure_retries = 1
        queue._ensure_retry_interval = 0
        queue._ensure_retry_count = {"127.0.0.1:11500": 0}
        queue._ensure_last_attempt_time = {}
        queue.kv_manager = MagicMock()
        queue.kv_manager.try_ensure_parallel_info.return_value = False

        cleared_req = SimpleNamespace(
            req=SimpleNamespace(rid="cleared"), kv_receiver=None
        )
        addr_to_reqs = {"127.0.0.1:11500": [cleared_req]}

        ready, remaining = queue._ensure_prefill_info(addr_to_reqs)

        self.assertEqual(ready, {})
        self.assertEqual(remaining, [])

    def test_prefetches_prefill_dp_rank_query(self):
        addr = "127.0.0.1:11500"
        executor = MagicMock()
        future = Future()
        future.set_result({"7": 1})
        executor.submit.return_value = future

        def decode_req(room):
            return SimpleNamespace(
                req=SimpleNamespace(
                    bootstrap_host="127.0.0.1",
                    bootstrap_port=11500,
                    bootstrap_room=room,
                ),
                kv_receiver=MagicMock(),
            )

        first = decode_req(7)
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pending_reqs = [first]
        queue._prefill_dp_rank_queries = {}
        queue.kv_manager = SimpleNamespace(
            prefill_info_table={addr: object()},
            _ensure_prefill_recompute_executor=lambda: executor,
        )
        queue._resolve_prefill_dp_rank = MagicMock(return_value=None)
        queue._ensure_prefill_info = lambda groups: (groups, [])

        queue.prefetch_prefill_dp_rank_queries()
        tail = decode_req(8)
        queue.pending_reqs.append(tail)
        with patch(
            "sglang.srt.disaggregation.decode.CommonKVReceiver.query_prefill_dp_ranks",
            return_value={"8": 2},
        ) as query:
            queue._resolve_pending_reqs()

        _, called_addr, called_rooms = executor.submit.call_args.args
        self.assertEqual((called_addr, called_rooms), (addr, [7]))
        query.assert_called_once_with(addr, [8])
        first.kv_receiver.init.assert_called_once_with(1)
        tail.kv_receiver.init.assert_called_once_with(2)
        self.assertEqual(queue.pending_reqs, [])

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_transfer_failure_cleanup_respects_deferred_release_gates(
        self, mock_poll, mock_prepare_abort, mock_release_kv_cache
    ):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="failed-transfer",
            bootstrap_room=7,
            return_logprob=False,
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            metadata_buffer_index=3,
            hicache_restore_status=HiCacheRestoreResult.READY,
        )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.enable_staging = False
        queue.enable_deferred_kv_release = False
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tp_rank = 0
        queue.tree_cache = MagicMock()
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        queue.spec_algorithm = MagicMock()
        queue.spec_algorithm.is_none.return_value = True
        queue._clean_hicache_prefetch_resources = MagicMock()

        scheduler = MagicMock()
        scheduler.enable_decode_hicache = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        scheduler.metrics_reporter.enable_metrics = False
        queue.scheduler = scheduler

        mock_poll.return_value = [KVPoll.Failed]

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )
        mock_prepare_abort.assert_called_once()
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, checkpoint=False
        )

        receiver = FakeReceiver()
        receiver.kv_mgr = FakeKVManager.__new__(FakeKVManager)
        decode_req.kv_receiver = receiver
        queue.queue = [decode_req]
        queue.enable_deferred_kv_release = True
        queue.req_to_metadata_buffer_idx_allocator.reset_mock()
        mock_release_kv_cache.reset_mock()

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, checkpoint=False
        )

        receiver = MagicMock()
        decode_req.kv_receiver = receiver
        decode_req.host_staged = True
        queue.enable_host_receive = True
        queue.enable_deferred_kv_release = False
        queue._defer_release = MagicMock()
        queue.queue = [decode_req]
        queue.req_to_metadata_buffer_idx_allocator.reset_mock()
        mock_release_kv_cache.reset_mock()
        self.assertEqual(queue.pop_transferred(), [])
        receiver.abort.assert_called_once_with()
        queue._defer_release.assert_called_once_with(decode_req)
        receiver.clear.assert_not_called()
        queue.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
        mock_release_kv_cache.assert_not_called()

        queue.queue = [decode_req]
        decode_req.req.finished_reason = FINISH_ABORT("cancelled")
        with (
            patch.object(
                queue, "_poll_with_metadata_gate", return_value=[KVPoll.Success]
            ),
            patch(
                "sglang.srt.disaggregation.decode.discard_kv_cache_backup"
            ) as discard,
        ):
            self.assertEqual(queue.pop_transferred(), [])
            discard.assert_called_once_with(req, queue.tree_cache, "host_pool")
        receiver.clear.assert_called_once_with()
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        mock_release_kv_cache.assert_not_called()

    def test_fake_receiver_initializes_deferred_release_state(self):
        manager = MagicMock()
        receiver = FakeKVReceiver(manager, "")

        self.assertIs(receiver.kv_mgr, manager)
        self.assertFalse(receiver.abort_notified)

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_failed_fake_transfer_releases_at_once_under_deferred_release(
        self, mock_poll, mock_prepare_abort, mock_release_kv_cache
    ):
        # A health-check request gets a FakeKVReceiver over the real transfer
        # manager, so a failed one reaches the deferred-release path.
        receiver = FakeKVReceiver(MagicMock(enable_deferred_decode_kv_release=True), "")
        req = SimpleNamespace(
            rid="health-check", bootstrap_room=0, return_logprob=False
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            metadata_buffer_index=3,
            hicache_restore_status=HiCacheRestoreResult.READY,
        )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.enable_staging = False
        queue.enable_host_receive = False
        queue.enable_deferred_kv_release = True
        queue._defer_release = MagicMock()
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tp_rank = 0
        queue.tree_cache = MagicMock()
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        queue.spec_algorithm = MagicMock()
        queue.spec_algorithm.is_none.return_value = True
        queue._clean_hicache_prefetch_resources = MagicMock()
        queue.scheduler = MagicMock(enable_decode_hicache=False, enable_hisparse=False)
        queue.scheduler.metrics_reporter.enable_metrics = False
        mock_poll.return_value = [KVPoll.Failed]

        self.assertEqual(queue.pop_transferred(), [])
        self.assertFalse(receiver.abort_notified)
        queue._defer_release.assert_not_called()
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, checkpoint=False
        )

    def test_retracted_decode_requests_keep_scheduler_non_idle(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.cur_batch_for_debug = None
        scheduler.enable_overlap = False
        scheduler.running_mbs = []
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[object()]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(
            queue=[], has_pending_deferred_releases=lambda: False
        )
        scheduler.decode_offload_manager = None
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_lmcache = False

        self.assertFalse(scheduler.is_fully_idle())


class _GateReceiver:
    def __init__(self, poll):
        self._poll = poll
        self.require_staging = False

    def poll(self):
        return self._poll

    def clear(self):
        pass

    def failure_exception(self):
        raise KVTransferError(
            7, "Failed due to an unknown reason from another rank", True
        )


def _gated_decode_req(poll, restore_status):
    return SimpleNamespace(
        req=SimpleNamespace(rid="r", bootstrap_room=7, return_logprob=False),
        kv_receiver=_GateReceiver(poll),
        metadata_buffer_index=3,
        hicache_restore_status=restore_status,
    )


class TestDecodeHiCacheRestoreGate(CustomTestCase):
    def test_gate_folds_restore_outcome_into_terminal_poll_only(self):
        cases = [
            (KVPoll.Success, HiCacheRestoreResult.READY, KVPoll.Success),
            (KVPoll.Success, HiCacheRestoreResult.PENDING, KVPoll.Transferring),
            (KVPoll.Success, HiCacheRestoreResult.FAILED, KVPoll.Failed),
            (KVPoll.Transferring, HiCacheRestoreResult.FAILED, KVPoll.Transferring),
            (
                KVPoll.WaitingForInput,
                HiCacheRestoreResult.FAILED,
                KVPoll.WaitingForInput,
            ),
            (KVPoll.Failed, HiCacheRestoreResult.PENDING, KVPoll.Failed),
        ]
        for poll, status, expected in cases:
            with self.subTest(poll=poll, status=status):
                gated = HiCacheRestoreGatedKVReceiver(_gated_decode_req(poll, status))
                self.assertEqual(gated.poll(), expected)

    def test_failed_restore_keeps_reserving_admission_budget(self):
        def dr(status, lock):
            return SimpleNamespace(
                prefix_match=SimpleNamespace(restore_token_count=3000),
                hicache_restore_status=status,
                hicache_restore_lock=lock,
            )

        prealloc = DecodePreallocQueue.__new__(DecodePreallocQueue)
        prealloc.scheduler = SimpleNamespace(enable_decode_hicache=True)
        prealloc.transfer_queue = SimpleNamespace(
            queue=[
                dr(HiCacheRestoreResult.PENDING, None),
                dr(HiCacheRestoreResult.FAILED, None),
                dr(HiCacheRestoreResult.PENDING, object()),
                dr(HiCacheRestoreResult.READY, object()),
            ]
        )
        self.assertEqual(prealloc._hicache_pending_restore_tokens(), 6000)

    def test_staging_poll_uses_gated_pollers(self):
        decode_req = _gated_decode_req(KVPoll.Success, HiCacheRestoreResult.PENDING)
        with patch(
            "sglang.srt.disaggregation.utils._all_reduce_polls",
            side_effect=lambda polls, group: polls,
        ):
            polls = poll_and_all_reduce_with_staging(
                [decode_req],
                staging_handler=MagicMock(),
                gloo_group=MagicMock(),
                pollers=[HiCacheRestoreGatedKVReceiver(decode_req)],
            )
        self.assertEqual(polls, [KVPoll.Transferring])

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    def test_pop_transferred_branches_on_reduced_poll_only(
        self, mock_prepare_abort, mock_release_kv_cache
    ):
        def make_queue(decode_req, peer_poll):
            queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
            queue.queue = [decode_req]
            queue.enable_staging = False
            queue.enable_host_receive = False
            queue.enable_deferred_kv_release = False
            queue.gloo_group = MagicMock()
            queue.req_to_metadata_buffer_idx_allocator = MagicMock()
            queue.tp_rank = 0
            queue.tree_cache = MagicMock()
            queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
            queue._clean_hicache_prefetch_resources = MagicMock()
            queue._commit_transfer_to_req = MagicMock()
            queue._record_transfer_failure = MagicMock()
            scheduler = MagicMock()
            scheduler.enable_decode_hicache = True
            scheduler.enable_hisparse = False
            scheduler.metrics_reporter.enable_metrics = False
            queue.scheduler = scheduler

            def reduce_with_peer(
                pollers, group, decode_reqs=None, metadata_buffers=None
            ):
                return [min(int(p.poll()), int(peer_poll)) for p in pollers]

            patcher = patch(
                "sglang.srt.disaggregation.decode.poll_and_all_reduce",
                side_effect=reduce_with_peer,
            )
            patcher.start()
            self.addCleanup(patcher.stop)
            return queue

        decode_req = _gated_decode_req(KVPoll.Success, HiCacheRestoreResult.FAILED)
        queue = make_queue(decode_req, peer_poll=KVPoll.Failed)
        self.assertEqual(queue.pop_transferred(), [])
        self.assertEqual(queue.queue, [])
        queue._commit_transfer_to_req.assert_not_called()
        queue._record_transfer_failure.assert_called_once_with("HiCache restore failed")
        mock_prepare_abort.assert_called_once()

        mock_prepare_abort.reset_mock()
        decode_req = _gated_decode_req(KVPoll.Success, HiCacheRestoreResult.READY)
        queue = make_queue(decode_req, peer_poll=KVPoll.Failed)
        self.assertEqual(queue.pop_transferred(), [])
        queue._commit_transfer_to_req.assert_not_called()
        queue._record_transfer_failure.assert_called_once_with(
            "Failed due to an unknown reason from another rank"
        )
        mock_prepare_abort.assert_called_once()

        mock_prepare_abort.reset_mock()
        mock_release_kv_cache.reset_mock()
        decode_req = _gated_decode_req(KVPoll.Transferring, HiCacheRestoreResult.FAILED)
        queue = make_queue(decode_req, peer_poll=KVPoll.Success)
        self.assertEqual(queue.pop_transferred(), [])
        self.assertEqual(queue.queue, [decode_req])
        mock_prepare_abort.assert_not_called()
        mock_release_kv_cache.assert_not_called()


class _VirtualClock:
    """perf_counter_ns for the loopstat tests: the mocked stages advance it; a read moves it by tick_us."""

    def __init__(self):
        self.ns = 10**9
        self.tick_us = 0

    def __call__(self):
        self.ns += self.tick_us * 1000
        return self.ns

    def step(self, us):
        self.ns += us * 1000


class _SplitReq(SimpleNamespace):
    def finished(self):
        return self.done


class _SplitBatch:
    def __init__(self, clock, reqs):
        self.clock, self.reqs, self.batch_is_full = clock, list(reqs), False

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return not self.reqs

    def copy(self):
        return self

    def process_prebuilt(self, future_map):
        self.clock.step(200)

    def filter_batch(self, reqs_to_exclude=None):
        # update_running_batch's filter costs 30 us, the prebuilt batch's own filter nothing
        if reqs_to_exclude is not None:
            self.clock.step(30)
        self.reqs = [r for r in self.reqs if not r.finished()]

    def merge_batch(self, other):
        self.clock.step(40)
        self.reqs = self.reqs + other.reqs

    def check_decode_mem(self):
        self.clock.step(20)
        return True

    def prepare_for_decode(self):
        self.clock.step(100)


class _NoCudaEvent:
    def record(self, stream=None):
        pass

    def query(self):
        return True

    def elapsed_time(self, other):
        return 1.0


class TestPlbLoopstatSplit(CustomTestCase):
    """SGLANG_PLB_LOOPSTAT_SPLIT: the loopstat CSV keeps its fixed columns (offline tools read them by name); on,
    each row appends the nb / rb sub-stages, whose remainders close each stage, and an idle iteration drops them."""

    FIXED = (
        "it,t_ms,fct,B,gbs,nnew,npa,npre,nxfer,nwait,idle,ar_n,knob,mode,pf,in,pq,pq_rel,pq_res,pp,pp_rs,pp_hs,"
        "pp_alloc,pp_send,px,px_poll,ar,nb,rb,prw,pr,smp"
    )
    SPLIT = (
        "nadm,nflt,pvm,nb_new,nb_pb,nb_res,nb_mrg,nb_flt,nb_mem,nb_pfd,nb_fl,nb_oth,rb_sl,rb_fi,rb_dp,rb_dg,rb_vp,"
        "rb_vf,rb_pv,rb_ex,rb_out,rb_oth,nb_cpu,rb_cpu,gc"
    )

    def _open_stat(self, split, clock):
        # a split LoopStat registers a GC callback for the process lifetime
        self.addCleanup(gc.callbacks.__delitem__, slice(len(gc.callbacks), None))
        out_dir = enter_scope(self, tempfile.TemporaryDirectory())
        for scope in (
            published_topology(role="scheduler"),
            patch.dict(os.environ, {"SGLANG_TRTL16_LOOPSTAT": out_dir}),
            envs.SGLANG_PLB_LOOPSTAT_SPLIT.override(split),
            patch.object(l16_loopstat, "_now", clock),
            patch.object(l16_loopstat, "STAT", None),
            patch.object(l16_loopstat, "SUB", None),
            patch.object(l16_loopstat.LoopStat, "_ev", lambda self: _NoCudaEvent()),
        ):
            enter_scope(self, scope)

    def _rows(self, stat, clock):
        stat._flush(clock.ns)
        for f in (stat.f, stat.fg, stat.ft):
            f.close()
        with open(stat.f.name) as f:
            return f.read().splitlines()

    def run_loop(self, split):
        """The real overlap loop with nb, update_running_batch and process_decode_queue: iteration 1 is idle, 2 admits
        a and b, 3 drops the finished a, 4 admits c. Every mocked step advances the virtual clock by a fixed cost."""
        clock = _VirtualClock()
        self._open_stat(split, clock)
        state = SimpleNamespace(it=0)
        reqs = {rid: _SplitReq(rid=rid, done=False) for rid in "abc"}
        arrivals = {2: "ab", 4: "c"}

        def ingest_requests():
            if state.it == 4:
                raise StopIteration
            state.it += 1
            reqs["a"].done = state.it >= 3

        def new_prebuilt_batch(running_batch):
            if not sched.waiting_queue:
                clock.step(10)
                return None
            clock.step(300)
            batch, sched.waiting_queue = _SplitBatch(clock, sched.waiting_queue), []
            return batch

        def run_batch(batch):
            sched.forward_ct += 1
            clock.step(5000)
            return GenerationBatchResult()

        sched = Scheduler.__new__(Scheduler)
        sched.gracefully_exit = False
        sched._engine_paused = False
        sched.enable_decode_hicache = False
        sched.enable_hisparse = False
        sched.enable_skip_finishing_decode = False
        sched.scheduler_stage_metrics = None
        sched.chunked_req = None
        sched.kv_checksum_computer = None
        sched.future_map = None
        sched.forward_stream = None
        sched.forward_ct = 0
        sched.server_args = SimpleNamespace(cuda_graph_bs_decode=[1, 2, 4, 8])
        sched.waiting_queue = []
        sched.running_batch = _SplitBatch(clock, [])
        sched.disagg_decode_prealloc_queue = SimpleNamespace(
            prefetch_prefill_dp_rank_queries=Mock(),
            flush_deferred_sends=Mock(),
            resume_retracted_reqs=Mock(return_value=[]),
            retracted_queue=[],
            pop_preallocated=Mock(return_value=([], [])),
            queue=[],
        )
        sched.disagg_decode_transfer_queue = SimpleNamespace(
            resolve_deferred_releases=Mock(),
            extend=Mock(),
            pop_transferred=lambda: [reqs[r] for r in arrivals.get(state.it, "")],
            staging_handler=SimpleNamespace(
                plb_defer=True, plb_flush=lambda stream: clock.step(70)
            ),
            queue=[],
        )
        sched.ingest_requests = ingest_requests
        sched._record_scheduler_state_for_paused_engine = Mock()
        sched._get_new_prebuilt_batch = new_prebuilt_batch
        sched.batch_result_processor = SimpleNamespace(
            process_batch_result_prebuilt=lambda batch: clock.step(50)
        )
        sched.new_token_ratio_tracker = SimpleNamespace(decay_step=Mock())
        sched.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch: batch
        )
        sched.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=lambda batch, chunked_req: batch
        )
        sched.is_disable_overlap_for_batch = lambda batch, last_batch: bool(
            clock.step(5)
        )
        sched._apply_war_barrier = lambda: clock.step(15)
        sched.run_batch = run_batch
        sched.process_batch_result = lambda batch, result: clock.step(400)
        sched.launch_batch_sample_if_needed = Mock()
        sched.on_idle = Mock()
        with self.assertRaises(StopIteration):
            Scheduler.event_loop_overlap_disagg_decode(sched)
        return self._rows(l16_loopstat.STAT, clock)

    def test_split_columns_follow_the_fixed_ones(self):
        off, on = self.run_loop(False), self.run_loop(True)
        n = len(self.FIXED.split(","))
        self.assertEqual(off[0], self.FIXED)
        self.assertEqual(on[0], self.FIXED + "," + self.SPLIT)
        # the switch leaves every fixed field of every row as it was
        self.assertEqual(
            [r.split(",") for r in off[1:]], [r.split(",")[:n] for r in on[1:]]
        )
        fixed = [dict(zip(self.FIXED.split(","), r.split(","))) for r in off[1:]]
        self.assertEqual(
            [
                tuple(int(d[k]) for k in ("B", "nnew", "idle", "nb", "rb", "pr"))
                for d in fixed
            ],
            [
                (2, 2, 1, 775, 5015, 0),
                (1, 0, 0, 235, 5015, 400),
                (2, 1, 0, 815, 5015, 400),
            ],
        )
        split = [
            dict(zip(self.SPLIT.split(","), map(int, r.split(",")[n:]))) for r in on[1:]
        ]
        for row in split:
            self.assertGreaterEqual(row.pop("nb_cpu"), 0)
            self.assertGreaterEqual(row.pop("rb_cpu"), 0)
        same = dict.fromkeys(
            "rb_sl rb_fi rb_dp rb_dg rb_vp rb_vf rb_pv rb_ex".split(), 0
        )
        same.update(nb_flt=30, nb_mem=20, nb_pfd=100, nb_fl=70, nb_oth=5)
        same.update(rb_out=15, rb_oth=5000, gc=0, pvm=0)
        # the idle iteration's nb_new (10) and nb_fl (70) must not reach the next row
        self.assertEqual(
            split,
            [
                dict(same, nadm=2, nflt=0, nb_new=300, nb_pb=200, nb_res=50, nb_mrg=0),
                dict(same, nadm=0, nflt=1, nb_new=10, nb_pb=0, nb_res=0, nb_mrg=0),
                dict(same, nadm=1, nflt=0, nb_new=300, nb_pb=200, nb_res=50, nb_mrg=40),
            ],
        )

    def test_gc_pause_lands_in_its_row(self):
        """A collection during a row counts in that row's gc column; an idle iteration drops its own."""
        clock = _VirtualClock()
        self._open_stat(True, clock)
        sched = SimpleNamespace(
            server_args=SimpleNamespace(cuda_graph_bs_decode=[1, 2]), forward_ct=0
        )
        stat = l16_loopstat.init(sched)
        batch = _SplitBatch(clock, [_SplitReq(rid="a", done=False)])
        for collect, idle in ((True, False), (True, True), (False, False)):
            stat.begin()
            if collect:
                # each clock read inside the collection (its start and stop) moves 3 us
                clock.tick_us = 3
                gc.collect()
                clock.tick_us = 0
            stat.mark("pf")
            stat.end(None if idle else batch)
        rows = self._rows(stat, clock)
        self.assertEqual([int(r.split(",")[-1]) for r in rows[1:]], [3, 0])


if __name__ == "__main__":
    unittest.main()
