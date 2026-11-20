"""HiCache progress must not depend on whether a prefill batch is admitted."""

import unittest
import unittest.mock
from types import SimpleNamespace
from unittest.mock import Mock, call

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import NextBatchPlan
from sglang.srt.managers.schedule_policy import PrefillStopStats
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestSchedulerHiCacheEvents(unittest.TestCase):
    def setUp(self):
        self.calls = Mock()
        self.scheduler = s = Scheduler.__new__(Scheduler)
        s.scheduler_stage_metrics = None
        s.prefill_stop_stats = PrefillStopStats()
        s.enable_hierarchical_cache = True
        s.enable_unified_cache_external_linker = False
        s.enable_lmcache = False
        s.enable_hicache_storage = True
        s.tree_cache = SimpleNamespace(check_hicache_events=self.calls.drain)
        s._process_storage_prefetch_retries = self.calls.retry
        s.process_pending_chunked_abort = Mock()
        s.process_prefill_chunk = Mock()
        s.dp_attn_adapter = Mock()
        s.dp_attn_adapter.maybe_prepare_mlp_sync_batch.return_value = None
        self.running_batch = SimpleNamespace(
            batch_is_full=False,
            is_prefill_only=False,
            is_empty=lambda: True,
            batch_size=lambda: 0,
            filter_batch=lambda: None,
        )

    def test_pd_prefill_drains_before_admission_even_with_empty_queue(self):
        s = self.scheduler
        s.resolve_waiting_queue_bootstrap = Mock()
        s.get_new_batch_prefill = self.calls.admit
        s.get_new_batch_prefill.return_value = NextBatchPlan(
            batch_to_run=None, running_batch=self.running_batch
        )
        for waiting_queue in ([], [SimpleNamespace(rid="pending_l3")]):
            with self.subTest(waiting=bool(waiting_queue)):
                s.waiting_queue = waiting_queue
                self.calls.reset_mock()
                for _ in range(2):
                    plan = s.get_next_disagg_prefill_batch_to_run(
                        running_batch=self.running_batch, last_batch=None
                    )
                    self.assertIsNone(plan.batch_to_run)
                self.assertEqual(
                    self.calls.mock_calls,
                    [call.drain(), call.retry(), call.admit(self.running_batch)] * 2,
                )

    def test_unified_drains_when_prefill_is_deferred(self):
        s = self.scheduler
        s.enable_fpm = False
        s._abort_on_waiting_timeout = Mock()
        s._abort_on_running_timeout = Mock()
        s.dllm_config = None
        s.chunked_req = None
        s.enable_hisparse = False
        s.require_mlp_sync = False
        s._should_defer_prefill = self.calls.defer
        s._should_defer_prefill.return_value = True
        s.get_new_batch_prefill = Mock()
        s.dp_attn_adapter.maybe_convert_decode_to_extend.return_value = None
        s._arm_prefill_decode_interval = Mock()
        s.ngram_embedding_manager = Mock()
        s.ngram_embedding_manager.prepare_for_forward.return_value = None

        plan = s.get_next_batch_to_run(self.running_batch, None)

        self.assertIsNone(plan.batch_to_run)
        s.get_new_batch_prefill.assert_not_called()
        self.assertEqual(
            self.calls.mock_calls,
            [
                call.drain(),
                call.retry(),
                call.defer(self.running_batch),
            ],
        )

    def test_pp_prefill_drains_before_admission(self):
        s = self.scheduler
        s.init_pp_loop_state = Mock()
        s.pp_loop_size = 1
        enter_scope(self, published_topology(pp_size=2))
        s.pp_group = SimpleNamespace(is_last_rank=True)
        s.running_mbs = [self.running_batch]
        s.last_mbs = [None]
        s.ingest_requests = Mock(return_value=[])
        s.waiting_queue = []
        s.disagg_prefill_bootstrap_queue = SimpleNamespace(pop_bootstrapped=lambda: [])
        s._pp_pd_get_prefill_transferred_ids = Mock(return_value=[])
        s._pp_commit_comm_work = Mock()
        s.get_new_batch_prefill = self.calls.admit
        # Stop the infinite loop at admission; no PP transport or GPU is needed.
        s.get_new_batch_prefill.side_effect = StopIteration

        with self.assertRaises(StopIteration):
            s.event_loop_pp_disagg_prefill()

        self.assertEqual(
            self.calls.mock_calls,
            [call.drain(), call.retry(), call.admit(self.running_batch)],
        )


class TestLinkerVerdictsRideTheRequestRelay(unittest.TestCase):
    """SGLANG_LINKER_ASYNC_BATCHED_RELAY: PP0's admission verdicts ride the
    request list it relays each round; every stage applies them before its
    pass, never processes them as a request, and forwards them on."""

    def _scheduler(self, received):
        s = Scheduler.__new__(Scheduler)
        s._deferred_input_requests = []
        s._poll_timeout_aborts = lambda: []
        s.enable_unified_cache_external_linker = True
        s.tree_cache = Mock()
        s.request_receiver = Mock()
        s.request_receiver.recv_requests.side_effect = lambda local_reqs: [*received, *local_reqs]
        s.metrics_reporter = Mock()
        s.process_input_requests = Mock()
        return s

    def _ingest(self, s, pp_rank):
        from sglang.srt.managers import scheduler as scheduler_module

        parallel = SimpleNamespace(pp_rank=pp_rank, attn_tp_rank=0, attn_cp_rank=0)
        with unittest.mock.patch.object(scheduler_module, "get_parallel", return_value=parallel):
            return s.ingest_requests()

    def test_pp0_sends_applies_and_forwards_its_verdicts(self):
        from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
            LinkerAdmissionVerdicts,
        )

        verdicts = LinkerAdmissionVerdicts(ready={"r": (3, 3)})
        s = self._scheduler(received=["req"])
        s.tree_cache.linker_take_verdicts.return_value = verdicts
        forwarded = self._ingest(s, pp_rank=0)
        s.tree_cache.linker_apply_verdicts.assert_called_once_with(verdicts)
        s.process_input_requests.assert_called_once_with(["req"])
        self.assertEqual(forwarded, ["req", verdicts])

    def _actions_scheduler(self, work):
        from sglang.srt.mem_cache.unified_cache.pp_linker_relay import PPLinkerActions

        actions = PPLinkerActions(actions=[("e", 1, "h", 1, "load", 2)])
        s = self._scheduler(received=["req", actions])
        s.tree_cache.ring_wait_work = work
        return s, actions

    def test_actions_apply_at_ingest_by_default(self):
        s, actions = self._actions_scheduler(work=None)
        forwarded = self._ingest(s, pp_rank=3)
        s.tree_cache.linker_apply_actions.assert_called_once_with([actions])
        self.assertEqual(forwarded, ["req", actions])

    def test_deferred_actions_wait_for_the_ring_wait_and_still_ride_on(self):
        # SGLANG_LINKER_PP_DEFER_ACTIONS: the stage forwards them as before, applies
        # the loads (private slots, ahead of admission) after this round's requests,
        # and the rest from the ring-wait queue.
        from collections import deque

        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.unified_cache.pp_linker_relay import PPLinkerActions

        work = deque()
        s, load = self._actions_scheduler(work)
        rest = PPLinkerActions(actions=[("e", 2, "g", 1, "finish", True), ("e", 3, "f", 2, "stop", None)])
        s.request_receiver.recv_requests.side_effect = lambda local_reqs: ["req", load, rest]
        with envs.SGLANG_LINKER_PP_DEFER_ACTIONS.override(True):
            forwarded = self._ingest(s, pp_rank=3)
        s.process_input_requests.assert_called_once_with(["req"])
        s.tree_cache.linker_apply_actions.assert_called_once_with([load])
        self.assertEqual(forwarded, ["req", load, rest])
        self.assertEqual(len(work), 1)
        work.popleft()()
        s.tree_cache.linker_apply_actions.assert_called_with([rest])
        self.assertEqual(s.tree_cache.linker_apply_actions.call_count, 2)

    def test_deferred_actions_without_loads_all_wait(self):
        from collections import deque

        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.unified_cache.pp_linker_relay import PPLinkerActions

        work = deque()
        s, _ = self._actions_scheduler(work)
        rest = PPLinkerActions(actions=[("e", 1, "g", 1, "finish", False)])
        s.request_receiver.recv_requests.side_effect = lambda local_reqs: ["req", rest]
        with envs.SGLANG_LINKER_PP_DEFER_ACTIONS.override(True):
            self._ingest(s, pp_rank=3)
        s.tree_cache.linker_apply_actions.assert_not_called()
        work.popleft()()
        s.tree_cache.linker_apply_actions.assert_called_once_with([rest])

    def test_a_later_stage_applies_and_forwards_without_processing_them(self):
        from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
            LinkerAdmissionVerdicts,
        )

        verdicts = LinkerAdmissionVerdicts(ready={})
        s = self._scheduler(received=["req", verdicts])
        forwarded = self._ingest(s, pp_rank=3)
        s.tree_cache.linker_take_verdicts.assert_not_called()
        s.tree_cache.linker_apply_verdicts.assert_called_once_with(verdicts)
        s.process_input_requests.assert_called_once_with(["req"])
        self.assertEqual(forwarded, ["req", verdicts])


if __name__ == "__main__":
    unittest.main()
