"""CPU coverage for continuous input polling and coordinated prefill completion."""

import multiprocessing
import tempfile
import threading
import time
import unittest
from array import array
from collections import Counter, defaultdict, deque
from contextlib import contextmanager, nullcontext
from functools import partial
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch
import torch.distributed as dist

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager, CommonKVSender
from sglang.srt.disaggregation.pp_consensus_store import PPConsensusStore
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import DisaggregationMode, pp_sync_polls
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
    ShutdownReq,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    NextBatchPlan,
    Req,
    ScheduleBatch,
)
from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.managers.scheduler import Scheduler, _dispatch_event_loop_once
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


class _StopLoop(Exception):
    pass


class _Event:
    def __init__(self, ready=False, recorded=True):
        self.ready = ready
        self.recorded = recorded
        self.queries = 0
        self.synchronizations = 0
        self.blocking_synchronizations = 0
        self.records = 0

    def record(self):
        self.recorded = True
        self.records += 1

    def query(self):
        assert self.recorded, "Polled a copy before submission"
        self.queries += 1
        return self.ready

    def synchronize(self):
        assert self.recorded, "Waited for a copy before submission"
        self.synchronizations += 1
        if not self.ready:
            self.blocking_synchronizations += 1
        self.ready = True


def _batch(forward_iter, reqs=None):
    if reqs is None:
        reqs = [Req(str(forward_iter), "", [1, 2, 3, 4], SamplingParams())]
    return ScheduleBatch(
        reqs=reqs,
        forward_iter=forward_iter,
        forward_mode=ForwardMode.EXTEND,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )


def _result(event=None, delayed=False):
    result = GenerationBatchResult(
        next_token_ids=torch.tensor([11]),
        copy_done=event if event is not None else _Event(),
    )
    if delayed:
        result.delay_sample_func = lambda: result
    return result


def _input(batch):
    return TokenizedGenerateReqInput(
        rid=str(batch.forward_iter),
        input_text="",
        input_ids=array("q", [1, 2, 3, 4]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )


class _Scheduler(SchedulerDisaggregationPrefillMixin):
    """Run real scheduling and result handling with controlled intake and GPU events."""

    launch_batch_sample_if_needed = Scheduler.launch_batch_sample_if_needed
    _copy_result_to_cpu = Scheduler._copy_result_to_cpu
    pause_generation = Scheduler.pause_generation
    continue_generation = Scheduler.continue_generation

    def __init__(
        self,
        arrivals,
        results,
        ready_at,
        pause_at=None,
        resume_at=None,
        pause_mode="in_place",
    ):
        self.arrivals = arrivals
        self.batches = {b.forward_iter: b for b in arrivals if b is not None}
        self.results = results
        self.copy_events = {name: result.copy_done for name, result in results.items()}
        self.ready_at = ready_at
        self.ready_batches = deque()
        self.request_receiver = SimpleNamespace(recv_requests=self.recv_requests)
        self._deferred_input_requests = []
        self._incoming_requests = []
        self.pause_at = pause_at
        self.resume_at = resume_at
        self.pause_mode = pause_mode
        self.paused_states = []
        self.iteration = -1
        self.actions = []
        self.admission_attempts = []
        self.chunk_processes = Counter()
        self.result_processes = Counter()
        self.depths = []
        self.scheduler_stage_metrics = None
        self.disaggregation_mode = DisaggregationMode.PREFILL
        self.spec_algorithm = SpeculativeAlgorithm.NONE
        self.is_generation = True
        self.dllm_config = None
        self.tp_cpu_group = None
        self.tp_size = 1
        self.attn_cp_cpu_group = None
        self.attn_tp_cpu_group = None
        self.last_batch = None
        self.result_queue = deque()
        self._deferred_responses = None
        self.chunked_req = None
        self.decode_offload_manager = None
        self.running_batch = ScheduleBatch(reqs=[])
        self.waiting_queue = []
        self.cur_batch_for_debug = None
        self._engine_paused = False
        self._pause_started_at = None
        self._pause_intervals = deque()
        self.gracefully_exit = False
        self.enable_staging = False
        self.enable_overlap = True
        self.enable_async_d2h_copy = False
        self.enable_overlap_mlx = False
        self.disagg_prefill_bootstrap_queue = SimpleNamespace(
            pop_bootstrapped=lambda: []
        )
        self.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=lambda batch, chunked_req: batch
        )
        self.forward_stream_ctx = nullcontext()
        self.copy_stream_ctx = nullcontext()
        self.forward_stream = Mock(spec=["wait_stream"])
        self.copy_stream = Mock(spec=["wait_stream"])
        self.schedule_stream = object()
        self._relay_forward_payload = Mock()
        self.tree_cache = Mock(spec=["flush_pending_backups", "finish"])
        self.batch_result_processor = Mock(
            spec=["snapshot_auxiliary_output_starts", "move_logprobs_to_cpu"]
        )
        self.metrics_reporter = Mock(
            spec=[
                "report_prefill_stats",
                "log_batch_result_stats",
                "update_device_timer",
                "record_scheduler_active",
            ]
        )
        self.metrics_reporter.current_scheduler_metrics_enabled = False
        self.kv_events_publisher = Mock(spec=["publish_kv_events"])
        self.publish_load_snapshot = Mock()
        self.load_publisher = Mock(spec=["publish_load_stat"])
        self.load_inquirer = SimpleNamespace(get_loads=Mock())
        self.enable_fpm = False
        self._record_step_counters = Mock()
        self._maybe_clear_mm_inputs = Mock()
        self.maybe_send_health_check_signal = Mock()
        self.disagg_prefill_inflight_queue = []
        self.send_kv_chunk = Mock()
        self.process_pending_chunked_abort = Mock()
        self.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch: batch
        )

    def _process_hicache_events(self, should_retry_storage_prefetch=True):
        pass

    def _poll_timeout_aborts(self):
        return []

    def ingest_requests(self, stop_at_pause=False):
        # GPU completion and incoming traffic advance even when intake is held.
        self.iteration += 1
        if self.iteration == len(self.arrivals):
            raise _StopLoop
        self.depths.append(len(self.result_queue))
        for name, iteration in self.ready_at.items():
            event = self.copy_events[name]
            event.ready = event.ready or self.iteration >= iteration
        self._incoming_requests.extend(self._inputs_for_step())
        return Scheduler.ingest_requests(self, stop_at_pause=stop_at_pause)

    def recv_requests(self, local_reqs=None):
        self.actions.append((self.iteration, "receive", None))
        if self.tp_cpu_group is not None:
            # A completion decision must never reorder the next intake collective.
            tick = torch.tensor([self.iteration])
            dist.broadcast(tick, src=0, group=self.tp_cpu_group)
            assert tick.item() == self.iteration
        inputs = self._incoming_requests
        self._incoming_requests = []
        return inputs + (local_reqs or [])

    def _inputs_for_step(self):
        inputs = []
        if self.iteration == self.pause_at:
            inputs.append(PauseGenerationReqInput(mode=self.pause_mode))
        if self.iteration == self.resume_at:
            inputs.append(ContinueGenerationReqInput(torch_empty_cache=False))
        if (batch := self.arrivals[self.iteration]) is not None:
            inputs.append(_input(batch))
        return inputs

    def process_input_requests(self, recv_reqs):
        for req in recv_reqs:
            name = (
                req.rid
                if isinstance(req, TokenizedGenerateReqInput)
                else type(req).__name__
            )
            self.actions.append((self.iteration, "dispatch", name))
            if isinstance(req, TokenizedGenerateReqInput):
                self.ready_batches.append(self.batches[int(req.rid)])
            elif isinstance(req, PauseGenerationReqInput):
                assert len(self.result_queue) <= 1, "Pause bypassed a result wait"
                self.pause_generation(req)
            elif isinstance(req, ContinueGenerationReqInput):
                self.continue_generation(req)
            elif isinstance(req, ShutdownReq):
                Scheduler.handle_shutdown(self, req)
            else:
                raise AssertionError(f"Unexpected input: {req}")

    def _record_scheduler_state_for_paused_engine(self):
        self.actions.append((self.iteration, "pause", None))
        self.paused_states.append((self.last_batch, tuple(self.result_queue)))

    def process_prefill_chunk(self, last_batch, running_batch):
        if last_batch is not None:
            self.chunk_processes[last_batch.forward_iter] += 1
        SchedulerDisaggregationPrefillMixin.process_prefill_chunk(
            self, last_batch=last_batch, running_batch=running_batch
        )

    def get_new_batch_prefill(self, running_batch):
        self.admission_attempts.append(self.iteration)
        batch = self.ready_batches.popleft() if self.ready_batches else None
        if batch is not None:
            self.actions.append((self.iteration, "prepare_batch", batch.forward_iter))
            self.waiting_queue = [
                req for req in self.waiting_queue if req not in batch.reqs
            ]
        return NextBatchPlan(batch_to_run=batch, running_batch=running_batch)

    def run_batch(self, batch):
        assert len(self.result_queue) < 2, "Overwrote a live forward's keep-alive slot"
        self.actions.append((self.iteration, "launch", batch.forward_iter))
        result = self.results[batch.forward_iter]
        if result.delay_sample_func is not None:

            def sample():
                self.actions.append((self.iteration, "sample", batch.forward_iter))
                return result

            result.delay_sample_func = sample
        return result

    war_barrier_is_coarse = True

    def _war_barrier_is_coarse(self):
        return self.war_barrier_is_coarse

    def _apply_war_barrier(self):
        self.actions.append((self.iteration, "barrier", None))

    def process_batch_result(self, batch, result):
        assert all(pending is not result for _, pending in self.result_queue)
        self.result_processes[batch.forward_iter] += 1
        self.actions.append((self.iteration, "process_result", batch.forward_iter))
        Scheduler.process_batch_result(self, batch, result)
        event = self.copy_events[batch.forward_iter]
        assert event.ready and event.recorded
        assert result.delay_sample_func is None
        self.actions.append((self.iteration, "result", batch.forward_iter))

    def process_disagg_prefill_inflight_queue(self):
        self.actions.append((self.iteration, "inflight", None))

    def on_idle(self):
        self.actions.append((self.iteration, "idle", None))


def _run_loop(scheduler, enabled=True, **topology):
    with (
        published_topology(**topology),
        envs.SGLANG_ENABLE_DISAGG_PREFILL_CONTINUOUS_INPUT_POLLING.override(enabled),
        patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache"),
    ):
        scheduler.tp_size = get_parallel().tp_size
        try:
            _dispatch_event_loop_once(scheduler)
        except _StopLoop:
            return
        if scheduler.gracefully_exit:
            return
    raise AssertionError("Scheduler exited before all arrivals were polled")


@contextmanager
def _single_rank_gloo_group():
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _rank_consensus_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        scheduler = _Scheduler(
            [_batch(1), None, _batch(2)] + [None] * 6,
            {name: _result() for name in (1, 2)},
            {1: (2, 5)[rank], 2: (7, 6)[rank]},
        )
        scheduler.tp_cpu_group = dist.group.WORLD
        _run_loop(scheduler, tp_size=2, ranks={"world_rank": rank})
        finished = [a for a in scheduler.actions if a[1] == "result"]
        assert finished == [(5, "result", 1), (7, "result", 2)], finished
        assert max(scheduler.depths) == 2
        for event in scheduler.copy_events.values():
            assert event.synchronizations == 1
            assert event.blocking_synchronizations == 0

        for depth in (1, 2):
            for mode in ("in_place", "retract"):
                arrivals = [
                    _batch(1),
                    _batch(2) if depth == 2 else None,
                    None,
                    _batch(3),
                ]
                names = (1, 2, 3) if depth == 2 else (1, 3)
                scheduler = _Scheduler(
                    arrivals + [None] * 8,
                    {
                        name: _result(_Event(recorded=False), delayed=True)
                        for name in names
                    },
                    {
                        1: (3, 5)[rank],
                        3: (10, 9)[rank],
                        **({2: (8, 7)[rank]} if depth == 2 else {}),
                    },
                    pause_at=2,
                    resume_at=7,
                    pause_mode=mode,
                )
                scheduler.tp_cpu_group = dist.group.WORLD
                _run_loop(scheduler, tp_size=2, ranks={"world_rank": rank})
                assert (5, "result", 1) in scheduler.actions
                assert (6, "dispatch", "PauseGenerationReqInput") in scheduler.actions
                assert (6, "dispatch", "3") in scheduler.actions
                assert (7, "launch", 3) in scheduler.actions
                assert not set(range(2, 7)).intersection(scheduler.admission_attempts)
                assert all(
                    (step, "receive", None) in scheduler.actions for step in range(3, 7)
                )
                assert scheduler.result_processes == {name: 1 for name in names}
                assert [a[2] for a in scheduler.actions if a[1] == "result"] == list(
                    names
                )
                assert not scheduler._deferred_input_requests
                for name, event in scheduler.copy_events.items():
                    assert event.synchronizations == 1
                    assert event.blocking_synchronizations == int(
                        name == 2 and mode == "retract"
                    )
                    assert event.records == 1

        # An unsubmitted copy on either rank is a lifecycle error on every rank.
        with published_topology(tp_size=2, ranks={"world_rank": rank}):
            batch_result_completion_status = torch.empty(1, dtype=torch.int32)
            for missing_event in (True, False):
                result = _result()
                if rank == 0:
                    if missing_event:
                        result.copy_done = None
                    else:
                        result.delay_sample_func = lambda: None
                        result.copy_done.recorded = False
                scheduler.result_queue = deque([(_batch(99), result)])
                try:
                    scheduler.is_disagg_prefill_batch_result_ready(
                        result, batch_result_completion_status
                    )
                except RuntimeError as error:
                    assert "must be submitted" in str(error)
                else:
                    raise AssertionError("An unsubmitted copy was polled")
    finally:
        dist.destroy_process_group()


class TestPrefillResultPolling(CustomTestCase):
    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_empty_polls_and_arrival_bursts_preserve_storage_retry_pacing(self):
        class CapacityBlockedScheduler(_Scheduler):
            _process_hicache_events = Scheduler._process_hicache_events
            _process_storage_prefetch_retries = (
                Scheduler._process_storage_prefetch_retries
            )
            _retry_storage_prefetch = Scheduler._retry_storage_prefetch

            def _prefetch_kvcache(self, req, storage_hit_end):
                self.reissues.append(self.iteration)
                self.tree_cache.storage_prefetch_retries.poll_miss(req.rid)

            def get_new_batch_prefill(self, running_batch):
                if self.iteration > 0 and not self.result_processes[1]:
                    self.admission_attempts.append(self.iteration)
                    self.policy.calc_priority(self.waiting_queue, running_batch)
                    return NextBatchPlan(batch_to_run=None, running_batch=running_batch)
                return super().get_new_batch_prefill(running_batch)

        for arrivals_during_wait in (False, True):
            with self.subTest(arrivals_during_wait=arrivals_during_wait):
                arrivals = [_batch(1)] + [None] * 13
                if arrivals_during_wait:
                    arrivals[2:12] = [_batch(i) for i in range(2, 12)]
                scheduler = CapacityBlockedScheduler(arrivals, {1: _result()}, {1: 13})
                scheduler.enable_hierarchical_cache = True
                scheduler.enable_hicache_storage = True
                scheduler.reissues = []
                retries = StoragePrefetchRetries()
                radix_cache = RadixCache.create_simulated()
                matched_at = []

                def match_prefix(params):
                    matched_at.append(scheduler.iteration)
                    return radix_cache.match_prefix(params)

                scheduler.tree_cache = SimpleNamespace(
                    check_hicache_events=Mock(),
                    flush_pending_backups=Mock(),
                    storage_prefetch_retries=retries,
                    supports_fast_match_prefix=lambda: True,
                    swa_reprefill_tail_tokens=lambda req=None: 0,
                    match_prefix=match_prefix,
                )
                scheduler.policy = SchedulePolicy(
                    "fcfs", scheduler.tree_cache, False, False, False
                )
                scheduler.ngram_embedding_manager.prepare_for_forward = Mock(
                    side_effect=lambda batch, chunked_req: batch
                )
                head, paced = (_batch(i).reqs[0] for i in (90, 91))
                for req in (head, paced):
                    req.origin_input_ids = array("q", req.origin_input_ids)
                    req.pending_bootstrap = False
                    req.disagg_kv_sender = SimpleNamespace(
                        poll=lambda: KVPoll.WaitingForInput
                    )
                scheduler.waiting_queue = [head, paced]
                retries.poll_miss(paced.rid)

                with _single_rank_gloo_group():
                    _run_loop(scheduler, hicache_storage_prefetch_retry_poll_interval=2)

                attempts = list(range(12)) if arrivals_during_wait else [0, 1]
                self.assertEqual(scheduler.admission_attempts, attempts)
                self.assertEqual(
                    matched_at, [step for step in attempts[1:] for _ in range(2)]
                )
                self.assertEqual(
                    scheduler.ngram_embedding_manager.prepare_for_forward.call_count,
                    len(attempts),
                )
                # Empty polls still receive inputs and query completion after a miss.
                self.assertEqual(scheduler.copy_events[1].queries, 13)
                self.assertEqual(
                    [
                        step
                        for step, action, _ in scheduler.actions
                        if action == "receive"
                    ],
                    list(range(14)),
                )
                self.assertEqual(scheduler.reissues, [])
                self.assertEqual(paced.storage_prefetch_retry_attempts, 0)
                self.assertIn((13, "result", 1), scheduler.actions)
                # The next outer admission still gets the retry due at step three.
                self.assertEqual(
                    retries.pop_ready([head, paced], 2, 8), [(paced, None)]
                )

    def test_shutdown_during_intake_stops_new_batches_in_both_modes(self):
        class StoppingScheduler(_Scheduler):
            def _inputs_for_step(self):
                inputs = super()._inputs_for_step()
                if self.iteration == shutdown_at:
                    inputs.insert(0, ShutdownReq())
                return inputs

        for enabled in (False, True):
            for shutdown_at in (0, 1, 2):
                with self.subTest(enabled=enabled, shutdown_at=shutdown_at):
                    scheduler = StoppingScheduler(
                        [_batch(1), _batch(2), _batch(3), None],
                        {
                            name: _result(
                                _Event(ready=True, recorded=False), delayed=True
                            )
                            for name in (1, 2, 3)
                        },
                        {},
                    )
                    _run_loop(scheduler, enabled=enabled)

                    self.assertTrue(scheduler.gracefully_exit)
                    self.assertEqual(
                        scheduler.admission_attempts, list(range(shutdown_at))
                    )
                    self.assertEqual(
                        [a[2] for a in scheduler.actions if a[1] == "launch"],
                        list(range(1, shutdown_at + 1)),
                    )
                    self.assertEqual(
                        scheduler.result_processes,
                        {name: 1 for name in range(1, shutdown_at + 1)},
                    )
                    self.assertFalse(scheduler.result_queue)

    def test_only_disabled_and_unsupported_modes_keep_blocking_results(self):
        for mode in (
            "disabled",
            "independent_dp",
            "attention_dp",
            "cp",
            "speculation",
            "embedding",
            "dllm",
        ):
            with self.subTest(mode=mode):
                scheduler = _Scheduler([_batch(1), None, None], {1: _result()}, {1: 2})
                topology = {}
                if mode == "independent_dp":
                    topology = {"dp_size": 2}
                elif mode == "attention_dp":
                    topology = {"dp_size": 2, "tp_size": 2, "enable_dp_attention": True}
                elif mode == "cp":
                    topology = {"attn_cp_size": 2, "tp_size": 2}
                elif mode == "speculation":
                    scheduler.spec_algorithm = SpeculativeAlgorithm.EAGLE
                elif mode == "embedding":
                    scheduler.is_generation = False
                elif mode == "dllm":
                    scheduler.dllm_config = object()
                _run_loop(scheduler, enabled=mode != "disabled", **topology)
                blocking = mode != "independent_dp"
                self.assertIn((1 if blocking else 2, "result", 1), scheduler.actions)
                self.assertEqual(scheduler.copy_events[1].queries == 0, blocking)
                self.assertEqual(
                    scheduler.copy_events[1].blocking_synchronizations, int(blocking)
                )

    def test_dispatch_clears_polling_state_outside_prefill_overlap(self):
        scheduler = _Scheduler([], {}, {})
        for mode, pp_size, overlap, loop in (
            (DisaggregationMode.PREFILL, 2, True, "event_loop_pp_disagg_prefill"),
            (DisaggregationMode.PREFILL, 1, False, "event_loop_normal_disagg_prefill"),
            (DisaggregationMode.DECODE, 1, True, "event_loop_overlap_disagg_decode"),
        ):
            with self.subTest(mode=mode, pp_size=pp_size, overlap=overlap):
                scheduler.disaggregation_mode = mode
                scheduler.enable_overlap = overlap
                scheduler.enable_continuous_input_polling = True
                with (
                    published_topology(pp_size=pp_size),
                    patch.object(scheduler, loop, create=True) as run_loop,
                    patch.object(
                        scheduler, "_is_continuous_input_polling_enabled"
                    ) as check_support,
                ):
                    _dispatch_event_loop_once(scheduler)
                run_loop.assert_called_once_with()
                check_support.assert_not_called()
                self.assertFalse(scheduler.enable_continuous_input_polling)

    def test_coarse_war_barrier_follows_previous_result_processing(self):
        steps = ("launch", "process_result", "inflight", "barrier")
        for coarse, polling, ready_at, expected in (
            # A whole-forward barrier waits until the previous result is processed.
            (True, False, 1, ["launch", "process_result", "inflight", "barrier"]),
            (True, True, 1, ["launch", "process_result", "inflight", "barrier"]),
            # Polling for an unfinished result still applies it before intake.
            (True, True, 2, ["launch", "barrier"]),
            # A read-done barrier keeps its place right after the launch.
            (False, False, 1, ["launch", "barrier", "process_result", "inflight"]),
            (False, True, 2, ["launch", "barrier"]),
        ):
            with self.subTest(coarse=coarse, polling=polling, ready_at=ready_at):
                scheduler = _Scheduler(
                    [_batch(1), _batch(2)] + [None] * 4,
                    {name: _result() for name in (1, 2)},
                    {1: ready_at, 2: ready_at + 1},
                )
                scheduler.war_barrier_is_coarse = coarse
                _run_loop(scheduler, enabled=polling)
                # Step 1 launches batch 2 while batch 1's result is pending.
                self.assertEqual(
                    [a[1] for a in scheduler.actions if a[0] == 1 and a[1] in steps],
                    expected,
                )

    def test_chunk_handoffs_across_polling_and_retract_pause(self):
        for enabled, pause_at, blocking_ends in (
            (True, None, ()),
            (False, 1, (4,)),
            (True, 1, (4,)),
            (True, 2, (8,)),
            (True, 4, (12,)),
        ):
            with self.subTest(enabled=enabled, pause_at=pause_at):
                req = Req("chunked", "", list(range(12)), SamplingParams())
                req.metadata_buffer_index = 1
                req.pending_bootstrap = False
                batches = [_batch(end, [req]) for end in (4, 8, 12)]
                results = {
                    b.forward_iter: GenerationBatchResult(
                        next_token_ids=torch.tensor([11]), copy_done=_Event()
                    )
                    for b in batches
                }
                checkpoint_ends = []

                class ChunkScheduler(_Scheduler):
                    def run_batch(self, batch):
                        end = int(batch.forward_iter)
                        req.extend_end = end
                        self.chunked_req = req if end < 12 else None
                        batch.chunked_req = self.chunked_req
                        if self.chunked_req is not None:
                            req.inflight_middle_chunks += 1
                        return super().run_batch(batch)

                    def checkpoint_disagg_prefill(self, req):
                        checkpoint_ends.append(req.extend_end)
                        super().checkpoint_disagg_prefill(req)

                if pause_at == 1:
                    arrivals = [batches[0], None, None, batches[1], batches[2]]
                    ready_at = {4: 20, 8: 7, 12: 8}
                elif pause_at == 2:
                    arrivals = [batches[0], batches[1], None, None, batches[2]]
                    ready_at = {4: 3, 8: 20, 12: 8}
                elif pause_at == 4:
                    # The second pending result is the request's final chunk.
                    arrivals = [batches[0], batches[1], None, batches[2], None]
                    ready_at = {4: 1, 8: 5, 12: 20}
                else:
                    arrivals = [batches[0], None, batches[1], None, batches[2]]
                    ready_at = {4: 4, 8: 6, 12: 8}
                scheduler = ChunkScheduler(
                    arrivals + [None] * 5,
                    results,
                    ready_at,
                    pause_at=pause_at,
                    resume_at=pause_at + 2 if pause_at is not None else None,
                    pause_mode="retract",
                )
                _run_loop(scheduler, enabled=enabled)

                self.assertEqual(
                    scheduler.send_kv_chunk.call_args_list,
                    [
                        call(req, last_chunk=False, end_idx=4),
                        call(req, last_chunk=False, end_idx=8),
                        call(req, last_chunk=True),
                    ],
                )
                # Pause must not take a cache snapshot that resume repeats.
                self.assertEqual(checkpoint_ends, [4, 8, 12])
                self.assertEqual(req.inflight_middle_chunks, 0)
                self.assertEqual(list(req.output_ids), [11])
                self.assertEqual(scheduler.disagg_prefill_inflight_queue, [req])
                self.assertEqual(scheduler.result_processes, {4: 1, 8: 1, 12: 1})
                for end, event in scheduler.copy_events.items():
                    self.assertEqual(event.synchronizations, 1)
                    self.assertEqual(
                        event.blocking_synchronizations,
                        int(not enabled or end in blocking_ends),
                    )

    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_retract_resume_requeues_optimistic_chunk_after_result_finishes(self):
        for depth, yield_before_pause in ((1, False), (2, False), (2, True)):
            with self.subTest(depth=depth, yield_before_pause=yield_before_pause):
                req = Req("optimistic", "", list(range(12)), SamplingParams())
                req.metadata_buffer_index = 1
                req.pending_bootstrap = True
                req.kv.req_pool_idx = 1
                req.kv.kv_allocated_len = 8
                req.disagg_kv_sender = SimpleNamespace(
                    poll=lambda: KVPoll.Bootstrapping
                )
                chunk_count = 1 if yield_before_pause else depth
                chunks = [_batch(end, [req]) for end in (4, 8)[:chunk_count]]
                other = _batch(100)
                other.reqs[0].disagg_kv_sender = SimpleNamespace(
                    poll=lambda: KVPoll.WaitingForInput
                )

                class OptimisticScheduler(_Scheduler):
                    _release_aborted_request = Scheduler._release_aborted_request

                    def run_batch(self, batch):
                        if req in batch.reqs:
                            req.extend_end = batch.forward_iter
                            self.chunked_req = batch.chunked_req = req
                            req.inflight_middle_chunks += 1
                        return super().run_batch(batch)

                    def process_input_requests(self, inputs):
                        super().process_input_requests(inputs)
                        if any(
                            isinstance(item, TokenizedGenerateReqInput)
                            and item.rid == "100"
                            for item in inputs
                        ):
                            self.waiting_queue.extend(other.reqs)

                pause_at = 2 if yield_before_pause else (1 if depth == 1 else 3)
                ready_at = {**{batch.forward_iter: 20 for batch in chunks}, 100: 7}
                if depth == 2:
                    ready_at[4] = pause_at + 1
                scheduler = OptimisticScheduler(
                    chunks + [other] + [None] * 8,
                    {batch.forward_iter: _result() for batch in [*chunks, other]},
                    ready_at,
                    pause_at=pause_at,
                    resume_at=pause_at + 2,
                    pause_mode="retract",
                )
                scheduler.disagg_prefill_pending_chunk_rids = set()
                scheduler.processed_tokens_counter = 0
                scheduler.metrics_reporter.enable_metrics = False
                release_steps = []

                def release(req, tree_cache, checkpoint):
                    self.assertEqual(req.inflight_middle_chunks, 0)
                    release_steps.append(scheduler.iteration)
                    req.kv.req_pool_idx = None
                    req.kv.mark_kv_released()

                with (
                    _single_rank_gloo_group(),
                    patch(
                        "sglang.srt.disaggregation.prefill.release_kv_cache",
                        side_effect=release,
                    ) as release_kv,
                ):
                    _run_loop(scheduler, optimistic_prefill_attempts=2)

                release_kv.assert_called_once_with(
                    req, scheduler.tree_cache, checkpoint=False
                )
                self.assertEqual(
                    release_steps, [3 if depth == 1 or yield_before_pause else 5]
                )
                self.assertEqual(scheduler.waiting_queue, [req])
                self.assertEqual(req.prefill_attempt_count, 1)
                self.assertEqual(req.inflight_middle_chunks, 0)
                self.assertEqual(list(req.output_ids), [])
                self.assertIsNone(scheduler.chunked_req)
                self.assertEqual(
                    scheduler.result_processes,
                    {b.forward_iter: 1 for b in [*chunks, other]},
                )
                scheduler.send_kv_chunk.assert_called_once_with(
                    other.reqs[0], last_chunk=True
                )

    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_ranks_agree_on_fifo_completion_and_copy_submission(self):
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = str(Path(directory) / "rendezvous")
            processes = [
                context.Process(target=_rank_consensus_worker, args=(rank, rendezvous))
                for rank in range(2)
            ]
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(timeout=60)
                self.assertEqual([process.exitcode for process in processes], [0, 0])
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)


class _ForwardLaunched(Exception):
    pass


class _EarlySendScheduler(SchedulerDisaggregationPrefillMixin):
    """Just enough Scheduler state for run_batch to reach the forward launch."""

    def __init__(self):
        self.scheduler_stage_metrics = None
        self.metrics_reporter = SimpleNamespace(record_scheduler_active=Mock())
        self.forward_ct = 0
        self._sched_idled = False
        self.processed_tokens_counter = 0
        self.scripted_scheduler_hook = None
        self.profiler_manager = SimpleNamespace(_profile_batch_predicate=Mock())
        self.forward_sleep_time = None
        self.disaggregation_mode = DisaggregationMode.PREFILL
        self.enable_staging = False
        self.enable_overlap = False
        self.token_to_kv_pool_allocator = SimpleNamespace(page_size=64)
        self.send_kv_chunk = Mock()

    @property
    def is_generation(self):
        raise _ForwardLaunched


class TestCachedPrefixEarlySend(CustomTestCase):
    def test_batch_that_starts_a_hicache_load_back_does_not_early_send(self):
        """A request can match pages whose HiCache load-back was submitted in the same
        build; early-sending them would read KV the H2D has not written yet (#44070)."""
        for consumer_index, expect_send in ((-1, True), (0, False)):
            scheduler = _EarlySendScheduler()
            req = SimpleNamespace(
                prefix_len=128,
                host_hit_length=0,
                start_send_idx=0,
                pending_bootstrap=False,
                early_send_prefix_end=None,
            )
            batch = SimpleNamespace(
                reqs=[req],
                hicache_consumer_index=consumer_index,
                extend_num_tokens=32,
                forward_mode=SimpleNamespace(
                    is_split_prefill=lambda: False, is_prebuilt=lambda: False
                ),
            )
            with self.subTest(consumer_index=consumer_index):
                with envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.override(True):
                    with self.assertRaises(_ForwardLaunched):
                        Scheduler.run_batch(scheduler, batch)
                if expect_send:
                    scheduler.send_kv_chunk.assert_called_once_with(
                        req, last_chunk=False, end_idx=128
                    )
                else:
                    scheduler.send_kv_chunk.assert_not_called()


class _RelayPPGroup:
    """In-process PP group: a stage's send_object lands in the next stage's inbox."""

    def __init__(self, rank, size, inboxes):
        self.rank_in_group = rank
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == size - 1
        self._inboxes = inboxes

    def broadcast_object(self, obj, src=0):
        # PPConsensusStore shares PP0's address; every stage holds one inbox map.
        if obj is not None:
            self._inboxes["store"] = obj
        return self._inboxes["store"]

    def send_object(self, obj, dst, async_send=False, tag=0):
        self._inboxes[(dst, tag)].append(obj)
        return []

    def recv_object(self, src, tag=0):
        return self._inboxes[(self.rank_in_group, tag)].popleft()


class TestPPPrefillReleaseRelay(CustomTestCase):
    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_every_stage_releases_what_pp0_decides_from_all_stages(self):
        ok, sending, failed = KVPoll.Success, KVPoll.Transferring, KVPoll.Failed
        # Room 1 is done on every stage, room 2 still sends on stage 2, and room 3
        # failed on stage 1 only.
        statuses = {1: (ok, ok, ok), 2: (ok, ok, sending), 3: (ok, failed, ok)}
        pp_size = 3
        inboxes = defaultdict(deque)
        stores = [
            PPConsensusStore(pp_size, k, _RelayPPGroup(k, pp_size, inboxes))
            for k in range(pp_size)
        ]
        try:
            for room, per_stage in statuses.items():
                for stage, status in enumerate(per_stage):
                    stores[stage][room] = status
            deadline = time.monotonic() + 5
            while any(
                stores[0].collect(room) != list(per_stage)
                for room, per_stage in statuses.items()
            ):
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)

            schedulers = []
            for stage in range(pp_size):
                kv_mgr = SimpleNamespace(
                    request_status=stores[stage],
                    _staging_outstanding={},
                    defer_decode_allocation=False,
                    reports_transfer_latency=False,
                )
                for check in ("check_status", "check_status_pp_consensus"):
                    setattr(
                        kv_mgr,
                        check,
                        partial(getattr(CommonKVManager, check), kv_mgr),
                    )
                queue = []
                for room in statuses:
                    sender = CommonKVSender.__new__(CommonKVSender)
                    sender.conclude_state = None
                    sender.bootstrap_room = room
                    sender.kv_mgr = kv_mgr
                    sender.trace_ctx = Mock()
                    sender.clear = Mock()
                    sender.get_transfer_metric = Mock()
                    time_stats = Mock()
                    time_stats.compute_and_observe_kv_transfer_metrics.return_value = {}
                    queue.append(
                        SimpleNamespace(
                            rid=f"r{room}",
                            disagg_kv_sender=sender,
                            pending_bootstrap=False,
                            return_routed_experts=False,
                            return_logprob=False,
                            finished_reason=None,
                            cache_request_handle=None,
                            bootstrap_host="10.0.0.1",
                            time_stats=time_stats,
                        )
                    )
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.scheduler_stage_metrics = None
                scheduler.pp_group = _RelayPPGroup(stage, pp_size, inboxes)
                scheduler.pp_release_sync_work_list = []
                scheduler.disagg_prefill_inflight_queue = queue
                scheduler.tree_cache = Mock()
                scheduler.output_streamer = Mock()
                scheduler.handle_inflight_transfer_failure = Mock()
                scheduler.req_to_metadata_buffer_idx_allocator = Mock()
                schedulers.append(scheduler)

            with (
                _single_rank_gloo_group(),
                published_topology(),
                patch("sglang.srt.disaggregation.prefill.release_kv_cache") as release,
                patch(
                    "sglang.srt.disaggregation.prefill.maybe_release_metadata_buffer"
                ),
            ):
                for scheduler in schedulers:
                    scheduler.attn_cp_cpu_group = dist.group.WORLD
                    scheduler.attn_tp_cpu_group = dist.group.WORLD
                # Stages run in pipeline order; each receives PP0's decision.
                for stage, scheduler in enumerate(schedulers):
                    with get_parallel().override(pp_size=pp_size, pp_rank=stage):
                        polls = scheduler._pp_pd_relay_prefill_release_polls()
                        self.assertEqual(polls, [ok, sending, failed])
                        done = scheduler.process_disagg_prefill_inflight_queue(
                            pp_consensus_polls=polls
                        )
                    self.assertEqual([req.rid for req in done], ["r1", "r3"])
                    self.assertEqual(
                        [req.rid for req in scheduler.disagg_prefill_inflight_queue],
                        ["r2"],
                    )
                    scheduler.handle_inflight_transfer_failure.assert_called_once()
                    self.assertEqual(
                        scheduler.handle_inflight_transfer_failure.call_args.args[
                            0
                        ].rid,
                        "r3",
                    )
                self.assertEqual(
                    [c.args[0].rid for c in release.call_args_list], ["r1"] * pp_size
                )

                # A stage whose inflight queue differs from PP0's fails loudly.
                extra = Mock(poll_pp_consensus=Mock(return_value=sending))
                schedulers[0].disagg_prefill_inflight_queue.append(
                    SimpleNamespace(rid="r4", disagg_kv_sender=extra)
                )
                with get_parallel().override(pp_size=pp_size, pp_rank=0):
                    schedulers[0]._pp_pd_relay_prefill_release_polls()
                with get_parallel().override(pp_size=pp_size, pp_rank=1):
                    with self.assertRaises(AssertionError):
                        schedulers[1]._pp_pd_relay_prefill_release_polls()
        finally:
            for store in stores:
                store.close()

    def test_pp0_fails_a_room_stuck_past_the_transfer_timeout_on_every_stage(self):
        # SGLANG_DISAGGREGATION_TRANSFER_TIMEOUT: room 2 still sends on stage 2 two
        # minutes after its last chunk; PP0's consensus poll fails it once, the relay
        # carries that, and every stage releases it through the Failed path once.
        # Room 3 also still sends but its last chunk is recent: it waits.
        ok, sending, failed = KVPoll.Success, KVPoll.Transferring, KVPoll.Failed
        statuses = {1: (ok, ok, ok), 2: (ok, ok, sending), 3: (ok, ok, sending)}
        pp_size = 3
        inboxes = defaultdict(deque)
        stores = [
            PPConsensusStore(pp_size, k, _RelayPPGroup(k, pp_size, inboxes))
            for k in range(pp_size)
        ]
        try:
            for room, per_stage in statuses.items():
                for stage, status in enumerate(per_stage):
                    stores[stage][room] = status
            deadline = time.monotonic() + 5
            while any(
                stores[0].collect(room) != list(per_stage)
                for room, per_stage in statuses.items()
            ):
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)

            schedulers, managers = [], []
            for stage in range(pp_size):
                kv_mgr = SimpleNamespace(
                    request_status=stores[stage],
                    _staging_outstanding={},
                    defer_decode_allocation=False,
                    reports_transfer_latency=False,
                    transfer_timeout=60.0,
                    failure_lock=threading.Lock(),
                    _timed_out_rooms=set(),
                    room_rids={room: f"r{room}" for room in statuses},
                    pp_rank=stage,
                    enable_deferred_decode_kv_release=False,
                    conclude_failure=Mock(),
                )
                for method in (
                    "check_status",
                    "check_status_pp_consensus",
                    "fail_room_on_timeout",
                ):
                    setattr(
                        kv_mgr,
                        method,
                        partial(getattr(CommonKVManager, method), kv_mgr),
                    )
                managers.append(kv_mgr)
                queue = []
                for room in statuses:
                    sender = CommonKVSender.__new__(CommonKVSender)
                    sender.conclude_state = None
                    sender.bootstrap_room = room
                    sender.kv_mgr = kv_mgr
                    sender._last_chunk_time = time.perf_counter() - (
                        120 if room == 2 else 1
                    )
                    sender.trace_ctx = Mock()
                    sender.clear = Mock()
                    sender.get_transfer_metric = Mock()
                    time_stats = Mock()
                    time_stats.compute_and_observe_kv_transfer_metrics.return_value = {}
                    queue.append(
                        SimpleNamespace(
                            rid=f"r{room}",
                            disagg_kv_sender=sender,
                            pending_bootstrap=False,
                            return_routed_experts=False,
                            return_logprob=False,
                            finished_reason=None,
                            cache_request_handle=None,
                            bootstrap_host="10.0.0.1",
                            time_stats=time_stats,
                        )
                    )
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.scheduler_stage_metrics = None
                scheduler.pp_group = _RelayPPGroup(stage, pp_size, inboxes)
                scheduler.pp_release_sync_work_list = []
                scheduler.disagg_prefill_inflight_queue = queue
                scheduler.tree_cache = Mock()
                scheduler.output_streamer = Mock()
                scheduler.handle_inflight_transfer_failure = Mock()
                scheduler.req_to_metadata_buffer_idx_allocator = Mock()
                schedulers.append(scheduler)

            with (
                _single_rank_gloo_group(),
                published_topology(),
                patch("sglang.srt.disaggregation.prefill.release_kv_cache") as release,
                patch(
                    "sglang.srt.disaggregation.prefill.maybe_release_metadata_buffer"
                ),
                self.assertLogs(
                    "sglang.srt.disaggregation.common.conn", level="ERROR"
                ) as logs,
            ):
                for scheduler in schedulers:
                    scheduler.attn_cp_cpu_group = dist.group.WORLD
                    scheduler.attn_tp_cpu_group = dist.group.WORLD
                for round_ in range(2):
                    for stage, scheduler in enumerate(schedulers):
                        with get_parallel().override(pp_size=pp_size, pp_rank=stage):
                            polls = scheduler._pp_pd_relay_prefill_release_polls()
                            done = scheduler.process_disagg_prefill_inflight_queue(
                                pp_consensus_polls=polls
                            )
                        if round_ == 0:
                            self.assertEqual(polls, [ok, failed, sending])
                            self.assertEqual([req.rid for req in done], ["r1", "r2"])
                        else:  # released once: room 2 is not polled again
                            self.assertEqual(polls, [sending])
                            self.assertEqual(done, [])
                        self.assertEqual(
                            [
                                req.rid
                                for req in scheduler.disagg_prefill_inflight_queue
                            ],
                            ["r3"],
                        )
                        scheduler.handle_inflight_transfer_failure.assert_called_once()
                        self.assertEqual(
                            scheduler.handle_inflight_transfer_failure.call_args.args[
                                0
                            ].rid,
                            "r2",
                        )
                self.assertEqual(
                    [c.args[0].rid for c in release.call_args_list], ["r1"] * pp_size
                )
            # One line for the room, from PP0, naming what it was stuck in.
            timeouts = [r for r in logs.output if "PD transfer timeout" in r]
            self.assertEqual(len(timeouts), 1)
            self.assertIn("room 2 rid r2 stage 0 stuck in Transferring", timeouts[0])
            self.assertIn("stage statuses Success,Success,Transferring", timeouts[0])
            managers[0].conclude_failure.assert_called_once()
            self.assertEqual(
                managers[0].conclude_failure.call_args.kwargs["bootstrap_room"], 2
            )
            for kv_mgr in managers[1:]:
                kv_mgr.conclude_failure.assert_not_called()
        finally:
            for store in stores:
                store.close()

    def _relay_once(
        self, statuses, last_ages, *, any_state, pp0_staging=None, rounds=1, inflight_age=None
    ):
        """One PP0-decided release pass over 3 stages. A None status: that stage never
        wrote the room. Returns PP0's polls per round, the rooms each stage failed, and
        the ERROR log lines."""
        pp_size = len(next(iter(statuses.values())))
        inboxes = defaultdict(deque)
        stores = [
            PPConsensusStore(pp_size, k, _RelayPPGroup(k, pp_size, inboxes))
            for k in range(pp_size)
        ]
        try:
            for room, per_stage in statuses.items():
                for stage, status in enumerate(per_stage):
                    if status is not None:
                        stores[stage][room] = status
            deadline = time.monotonic() + 5
            while any(
                stores[0].collect(room) != list(per_stage)
                for room, per_stage in statuses.items()
            ):
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            schedulers = []
            for stage in range(pp_size):
                kv_mgr = SimpleNamespace(
                    request_status=stores[stage],
                    _staging_outstanding=dict(pp0_staging or {}) if stage == 0 else {},
                    defer_decode_allocation=False,
                    reports_transfer_latency=False,
                    transfer_timeout=60.0,
                    transfer_timeout_any_state=any_state,
                    bootstrap_timeout=3600,
                    failure_lock=threading.Lock(),
                    _timed_out_rooms=set(),
                    room_rids={room: f"r{room}" for room in statuses},
                    pp_rank=stage,
                    enable_deferred_decode_kv_release=False,
                    conclude_failure=Mock(),
                )
                for method in ("check_status", "check_status_pp_consensus", "fail_room_on_timeout"):
                    setattr(kv_mgr, method, partial(getattr(CommonKVManager, method), kv_mgr))
                queue = []
                for room in statuses:
                    sender = CommonKVSender.__new__(CommonKVSender)
                    sender.conclude_state = None
                    sender.bootstrap_room = room
                    sender.kv_mgr = kv_mgr
                    sender.init_time = time.time()
                    age = last_ages.get(room)
                    sender._last_chunk_time = None if age is None else time.perf_counter() - age
                    since = (inflight_age or {}).get(room)
                    if since is not None and stage == 0:
                        sender._inflight_since = time.perf_counter() - since
                    sender.trace_ctx = Mock()
                    sender.clear = Mock()
                    sender.get_transfer_metric = Mock()
                    time_stats = Mock()
                    time_stats.compute_and_observe_kv_transfer_metrics.return_value = {}
                    queue.append(
                        SimpleNamespace(
                            rid=f"r{room}", disagg_kv_sender=sender, pending_bootstrap=False,
                            return_routed_experts=False, return_logprob=False,
                            finished_reason=None, cache_request_handle=None,
                            bootstrap_host="10.0.0.1", time_stats=time_stats,
                        )
                    )
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.scheduler_stage_metrics = None
                scheduler.pp_group = _RelayPPGroup(stage, pp_size, inboxes)
                scheduler.pp_release_sync_work_list = []
                scheduler.disagg_prefill_inflight_queue = queue
                scheduler.tree_cache = Mock()
                scheduler.output_streamer = Mock()
                scheduler.handle_inflight_transfer_failure = Mock()
                scheduler.req_to_metadata_buffer_idx_allocator = Mock()
                schedulers.append(scheduler)
            pp0_polls = []
            with (
                _single_rank_gloo_group(),
                published_topology(),
                patch("sglang.srt.disaggregation.prefill.release_kv_cache"),
                patch("sglang.srt.disaggregation.prefill.maybe_release_metadata_buffer"),
                self.assertLogs("sglang.srt.disaggregation", level="WARNING") as logs,
            ):
                import logging
                logging.getLogger("sglang.srt.disaggregation").warning("relay start")
                for scheduler in schedulers:
                    scheduler.attn_cp_cpu_group = dist.group.WORLD
                    scheduler.attn_tp_cpu_group = dist.group.WORLD
                for _ in range(rounds):
                    for stage, scheduler in enumerate(schedulers):
                        with get_parallel().override(pp_size=pp_size, pp_rank=stage):
                            polls = scheduler._pp_pd_relay_prefill_release_polls()
                            scheduler.process_disagg_prefill_inflight_queue(
                                pp_consensus_polls=polls
                            )
                        if stage == 0:
                            pp0_polls.append(polls)
            failed = [
                [c.args[0].rid for c in s.handle_inflight_transfer_failure.call_args_list]
                for s in schedulers
            ]
            return pp0_polls, failed, [r for r in logs.output if "PD transfer timeout" in r]
        finally:
            for store in stores:
                store.close()

    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_inflight_bootstrapping_consensus_is_a_crash_not_a_hang(self):
        # A stage that never wrote an inflight room makes PP0's consensus Bootstrapping,
        # which the inflight queue rejects outright; so the H134 hang, with no crash,
        # was not this state.
        ok = KVPoll.Success
        with self.assertRaisesRegex(RuntimeError, "Unexpected poll state 1 for req r2"):
            self._relay_once({1: (ok, ok, ok), 2: (ok, ok, None)}, {1: 1, 2: 120}, any_state=True)

    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_any_state_deadline_fails_a_room_whose_last_chunk_never_left(self):
        # Room 2 waits for input on stage 2 and PP0 never enqueued its last chunk, so the
        # plain deadline (from the last chunk) never starts. ANY_STATE counts from the
        # room's first inflight poll on PP0 instead and fails it there, on every stage.
        ok, waiting = KVPoll.Success, KVPoll.WaitingForInput
        statuses = {1: (ok, ok, ok), 2: (ok, ok, waiting)}
        polls, failed, lines = self._relay_once(statuses, {1: 1}, any_state=False, rounds=2)
        self.assertEqual(polls, [[ok, waiting], [waiting]])
        self.assertEqual(failed, [[], [], []])
        self.assertEqual(lines, [])
        polls, failed, lines = self._relay_once(
            statuses, {1: 1}, any_state=True, inflight_age={2: 120}
        )
        self.assertEqual(polls, [[ok, KVPoll.Failed]])
        self.assertEqual(failed, [["r2"], ["r2"], ["r2"]])
        (line,) = lines
        self.assertIn(
            "room 2 rid r2 stage 0 stuck in WaitingForInput with no last chunk enqueued here",
            line,
        )
        self.assertIn("stage statuses Success,Success,WaitingForInput", line)

    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_any_state_deadline_fails_a_success_held_by_staging(self):
        # Every stage says Success but PP0 still counts a staging chunk: the poll holds it
        # as Transferring and returns before any deadline. ANY_STATE applies the deadline.
        ok = KVPoll.Success
        statuses = {1: (ok, ok, ok), 2: (ok, ok, ok)}
        polls, failed, lines = self._relay_once(
            statuses, {1: 1, 2: 120}, any_state=False, pp0_staging={2: 1}, rounds=2
        )
        self.assertEqual(polls, [[ok, KVPoll.Transferring], [KVPoll.Transferring]])
        self.assertEqual(failed, [[], [], []])
        polls, failed, lines = self._relay_once(
            statuses, {1: 1, 2: 120}, any_state=True, pp0_staging={2: 1}
        )
        self.assertEqual(polls, [[ok, KVPoll.Failed]])
        self.assertEqual(failed, [["r2"], ["r2"], ["r2"]])
        (line,) = lines
        self.assertIn("room 2 rid r2 stage 0 stuck in Transferring", line)

    def test_inflight_diag_logs_every_room_once_per_request(self):
        # SGLANG_PD_INFLIGHT_DIAG: SIGUSR2 sets a flag; the next inflight pass logs each
        # room's state on this stage (PP0 also each stage's), then clears the flag.
        from sglang.srt.disaggregation.prefill import request_inflight_diag

        kv_mgr = SimpleNamespace(
            request_status={7: KVPoll.Transferring, 8: KVPoll.WaitingForInput},
            transfer_infos={7: {"a": SimpleNamespace(required_dst_info_num=4)}},
            _staging_outstanding={7: 1},
            pp_rank=0,
        )
        kv_mgr.describe_room = partial(CommonKVManager.describe_room, kv_mgr)
        queue = []
        for room in (7, 8):
            sender = CommonKVSender.__new__(CommonKVSender)
            sender.conclude_state = None
            sender.kv_mgr = kv_mgr
            sender._last_chunk_time = time.perf_counter() - 30 if room == 7 else None
            queue.append(
                SimpleNamespace(
                    rid=f"r{room}", bootstrap_room=room, disagg_kv_sender=sender,
                    pending_bootstrap=False,
                )
            )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.scheduler_stage_metrics = None
        scheduler.output_streamer = Mock()
        scheduler.disagg_prefill_inflight_queue = queue
        polls = [KVPoll.Transferring, KVPoll.WaitingForInput]
        with published_topology(), self.assertLogs(
            "sglang.srt.disaggregation.prefill", level="WARNING"
        ) as logs:
            request_inflight_diag()
            scheduler.process_disagg_prefill_inflight_queue(pp_consensus_polls=polls)
            scheduler.process_disagg_prefill_inflight_queue(pp_consensus_polls=polls)
        lines = [r for r in logs.output if "PD inflight diag" in r]
        self.assertEqual(len(lines), 3)  # header + two rooms, and nothing on the second pass
        self.assertIn("PP0, 2 inflight rooms", lines[0])
        self.assertRegex(
            lines[1],
            r"room 7 rid r7 poll 3 conclude None pending_bootstrap False last_chunk_age 3\d\.\ds "
            r"local Transferring staging_outstanding 1 dst_infos 1/4",
        )
        self.assertIn("room 8 rid r8 poll 2", lines[2])
        self.assertIn("last_chunk_age - local WaitingForInput staging_outstanding 0 dst_infos 0/None", lines[2])

    def test_store_and_status_rules_reach_success_at_twice_the_rooms(self):
        # Stress repro of the PP consensus store plus update_status at 2 x 48 rooms per
        # micro-batch: per stage, three threads write each room through the real rules
        # (scheduler: sender init; bootstrap thread: dst info; worker: Transferring then
        # Success after the dst info), init and dst info in random order. PP0 must see
        # every room's consensus reach Success with no stage entry left None. Room 0
        # never gets stage 2's dst info: its consensus stays Bootstrapping, the state the
        # bootstrap queue holds such a room in (it never reaches the inflight queue).
        import random

        pp_size, rooms = 3, list(range(2 * 48 * 4))
        inboxes = defaultdict(deque)
        stores = [
            PPConsensusStore(pp_size, k, _RelayPPGroup(k, pp_size, inboxes))
            for k in range(pp_size)
        ]
        try:
            managers = []
            for stage in range(pp_size):
                mgr = SimpleNamespace(
                    request_status=stores[stage], _deferred_ack_targets={},
                    _deferred_ack_poisoned_rooms=set(), _staging_outstanding={},
                    deferred_bootstrap=None, _timed_out_rooms=set(),
                )
                mgr.update_status = partial(CommonKVManager.update_status, mgr)
                managers.append(mgr)

            def stage_threads(stage, mgr, seed):
                rng = random.Random(seed)
                info = {room: threading.Event() for room in rooms}

                def scheduler():
                    for room in rooms:
                        mgr.update_status(room, KVPoll.Bootstrapping)
                        if rng.random() < 0.01:
                            time.sleep(0.0005)

                def bootstrap():
                    for room in rng.sample(rooms, len(rooms)):
                        if room == 0 and stage == 2:
                            continue
                        mgr.update_status(room, KVPoll.WaitingForInput)
                        info[room].set()

                def worker():
                    for room in rooms:
                        if room == 0 and stage == 2:
                            continue
                        info[room].wait(5)
                        mgr.update_status(room, KVPoll.Transferring)
                        mgr.update_status(room, KVPoll.Success)

                return [threading.Thread(target=f) for f in (scheduler, bootstrap, worker)]

            threads = [
                t for stage, mgr in enumerate(managers) for t in stage_threads(stage, mgr, stage)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(10)
            consensus = partial(CommonKVManager.check_status_pp_consensus, managers[0])
            deadline = time.monotonic() + 10
            while any(consensus(room) != KVPoll.Success for room in rooms[1:]):
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            self.assertEqual(consensus(0), KVPoll.Bootstrapping)
            self.assertEqual(stores[0].collect(0), [KVPoll.Success, KVPoll.Success, KVPoll.Bootstrapping])
        finally:
            for store in stores:
                store.close()

    def test_relay_window_waits_only_rounds_older_than_the_window(self):
        for window in (1, 3):
            with self.subTest(window=window):
                sends = []

                def send_object(obj, dst, async_send=False, tag=0):
                    sends.append(Mock())
                    return [SimpleNamespace(work=sends[-1])]

                group = SimpleNamespace(send_object=send_object)
                work_list = []
                for k in range(6):
                    self.assertEqual(
                        pp_sync_polls([k], group, 0, 3, work_list, window=window), [k]
                    )
                    # Round k's send is waited when round k + window opens.
                    self.assertEqual(
                        [send.wait.called for send in sends],
                        [j + window <= k for j in range(k + 1)],
                    )


class _ForwardEvent:
    """A launched forward's event: ready once the test says the forward ended."""

    def __init__(self):
        self.ready = False
        self.synchronizations = 0

    def record(self, stream=None):
        pass

    def query(self):
        return self.ready

    def synchronize(self):
        self.synchronizations += 1
        self.ready = True


class _PPEarlySendScheduler(SchedulerDisaggregationPrefillMixin):
    def __init__(self, is_last_rank):
        self.pp_group = SimpleNamespace(is_last_rank=is_last_rank)
        self.chunked_req = None
        self.forward_stream_ctx = nullcontext()
        self.device_module = SimpleNamespace(
            Event=_ForwardEvent, current_stream=lambda: None
        )
        self.disagg_prefill_early_sent_rids = set()
        self.disagg_prefill_early_send_pending = deque()
        self.disagg_prefill_early_send_stats = {"early": 0, "waited": 0}
        self.sent = []

        def send(req, last_chunk=False, end_idx=None, **kwargs):
            # The forward's event rides on the sender to the transfer worker,
            # which consumes it with the first chunk.
            sender = req.disagg_kv_sender
            wait_event = sender._early_send_wait_event
            sender._early_send_wait_event = None
            self.sent.append((req.rid, last_chunk, kwargs, wait_event))

        self._send_kv_chunk = send


def _early_req(rid, room, status=KVPoll.WaitingForInput, **overrides):
    kv_mgr = SimpleNamespace(request_status={room: status}, _staging_outstanding={})
    fields = dict(
        rid=rid,
        bootstrap_room=room,
        origin_input_ids=[1] * 8,
        extend_end=8,
        pending_bootstrap=False,
        metadata_buffer_index=0,
        discard_output_reason=None,
        to_finish=None,
        finished_reason=None,
        return_logprob=False,
        return_sampling_mask=False,
        grammar=None,
        bootstrap_host="10.0.0.1",
        disagg_kv_sender=SimpleNamespace(
            kv_mgr=kv_mgr,
            supports_host_tensor_indices=True,
            _early_send_wait_event=None,
        ),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestPPPrefillEarlySend(CustomTestCase):
    def test_stage_sends_last_chunk_at_its_forward_and_leaves_aux_to_last_stage(self):
        done = _early_req("done", 1)
        chunked = _early_req("chunked", 2, extend_end=4)
        logprob = _early_req("logprob", 3, return_logprob=True)
        aborted = _early_req("aborted", 4, to_finish=FINISH_ABORT("abort"))
        batch = SimpleNamespace(reqs=[done, chunked, logprob, aborted], forward_iter=5)
        launch_event = _ForwardEvent()

        # A stage before the last sends now, fenced on its forward, without aux.
        stage = _PPEarlySendScheduler(is_last_rank=False)
        stage.chunked_req = chunked
        stage._pp_pd_early_send_last_chunks(batch, None, launch_event)
        no_aux = dict(first_output_id=None, send_aux=False)
        self.assertEqual(stage.sent, [("done", True, no_aux, launch_event)])
        self.assertEqual(stage.disagg_prefill_early_sent_rids, {"done"})

        # The last stage waits for its forward's first token, then sends it as aux.
        last = _PPEarlySendScheduler(is_last_rank=True)
        last.chunked_req = chunked
        result = SimpleNamespace(next_token_ids=torch.tensor([7, 8, 9, 10]))
        last._pp_pd_early_send_last_chunks(batch, result, launch_event)
        last._pp_pd_flush_early_sends()
        self.assertEqual(last.sent, [])
        (_, ready, _, _), = last.disagg_prefill_early_send_pending
        ready.ready = True
        last._pp_pd_flush_early_sends()
        self.assertEqual(
            last.sent, [("done", True, dict(first_output_id=7, send_aux=True), None)]
        )
        self.assertFalse(last.disagg_prefill_early_send_pending)

        # Before a micro-batch's result is processed its sends are forced out; an
        # abort that arrived since launch leaves the request to the result path.
        last = _PPEarlySendScheduler(is_last_rank=True)
        late = _early_req("late", 5)
        last._pp_pd_early_send_last_chunks(
            SimpleNamespace(reqs=[done, late], forward_iter=6),
            SimpleNamespace(next_token_ids=torch.tensor([3, 4])),
            launch_event,
        )
        late.to_finish = FINISH_ABORT("abort")
        last._pp_pd_flush_early_sends(block_upto_forward_iter=5)
        self.assertEqual(last.sent, [])
        last._pp_pd_flush_early_sends(block_upto_forward_iter=6)
        self.assertEqual([s[0] for s in last.sent], ["done"])
        self.assertEqual(last.sent[0][2], dict(first_output_id=3, send_aux=True))

    def test_result_skips_the_early_sent_chunk_after_its_transfer_ends(self):
        stage = _PPEarlySendScheduler(is_last_rank=False)
        req = _early_req("r", 1, status=KVPoll.Transferring)
        self.assertFalse(stage._settle_early_sent_last_chunk(req, forget=True))
        stage.disagg_prefill_early_sent_rids.add("r")

        # The checkpoint after it may dedupe (free) pages the transfer still reads.
        def finish():
            time.sleep(0.05)
            req.disagg_kv_sender.kv_mgr.request_status[1] = KVPoll.Success

        threading.Thread(target=finish).start()
        start = time.monotonic()
        self.assertTrue(stage._settle_early_sent_last_chunk(req, forget=True))
        self.assertGreaterEqual(time.monotonic() - start, 0.04)
        self.assertEqual(stage.disagg_prefill_early_send_stats["waited"], 1)
        self.assertFalse(stage._settle_early_sent_last_chunk(req, forget=True))

    def test_gate_is_off_by_default_and_needs_pp_mooncake_and_aux_free_results(self):
        stage = _PPEarlySendScheduler(is_last_rank=False)
        stage.kv_checksum_computer = None
        stage.enable_staging = False
        stage.enable_unified_cache_external_linker = False
        stage.spec_algorithm = SpeculativeAlgorithm.DFLASH
        with published_topology(pp_size=2):
            self.assertFalse(stage._pp_pd_early_send_enabled())
            with envs.SGLANG_PP_PD_PREFILL_EARLY_SEND.override(True):
                self.assertTrue(stage._pp_pd_early_send_enabled())
                stage.spec_algorithm = SpeculativeAlgorithm.EAGLE
                self.assertFalse(stage._pp_pd_early_send_enabled())
        with published_topology(pp_size=1):
            with envs.SGLANG_PP_PD_PREFILL_EARLY_SEND.override(True):
                stage.spec_algorithm = SpeculativeAlgorithm.DFLASH
                self.assertFalse(stage._pp_pd_early_send_enabled())

    def test_prefill_complete_sends_early_only_what_decode_reserved_first(self):
        stage = _PPEarlySendScheduler(is_last_rank=False)
        stage.kv_checksum_computer = None
        stage.enable_staging = False
        stage.enable_unified_cache_external_linker = False
        stage.spec_algorithm = SpeculativeAlgorithm.DFLASH
        with published_topology(
            pp_size=2, disaggregation_decode_allocation_policy="prefill_complete"
        ):
            with envs.SGLANG_PP_PD_PREFILL_EARLY_SEND.override(True):
                self.assertTrue(stage._pp_pd_early_send_enabled())
        # Early readiness (below SGLANG_PREFILL_COMPLETE_MIN_WORK_TOKENS): decode
        # reserved during prefill. Late: readiness waits for the result.
        early = _early_req("early", 1)
        late = _early_req(
            "late", 2, status=KVPoll.Bootstrapping, pending_bootstrap=True
        )
        batch = SimpleNamespace(reqs=[early, late], forward_iter=3)
        launch_event = _ForwardEvent()
        stage._pp_pd_early_send_last_chunks(batch, None, launch_event)
        no_aux = dict(first_output_id=None, send_aux=False)
        self.assertEqual(stage.sent, [("early", True, no_aux, launch_event)])
        self.assertEqual(stage.disagg_prefill_early_sent_rids, {"early"})


class TestDeferredEarlySettle(CustomTestCase):
    """SGLANG_PP_PD_DEFER_EARLY_SETTLE: a result does not wait for its early-sent last
    chunk; the checkpoint's device frees wait in the inflight queue instead."""

    @staticmethod
    def _req(status):
        kv_mgr = SimpleNamespace(request_status={7: status}, _staging_outstanding={})
        return SimpleNamespace(
            rid="r", bootstrap_room=7, disagg_kv_sender=SimpleNamespace(kv_mgr=kv_mgr)
        )

    def _scheduler(self, cache=None):
        mixin = SchedulerDisaggregationPrefillMixin
        sched = SimpleNamespace(
            disagg_prefill_early_sent_rids={"r"},
            disagg_prefill_early_send_stats={"early": 1, "waited": 0},
            _settle_early_sent_last_chunk=Mock(return_value=True),
            _send_kv_chunk=Mock(),
            kv_checksum_computer=None,
            tree_cache=cache,
        )
        sched._apply_deferred_frees = lambda req: mixin._apply_deferred_frees(sched, req)
        return sched

    def test_result_does_not_wait_and_last_chunk_is_not_resent(self):
        mixin = SchedulerDisaggregationPrefillMixin
        for enabled, status, deferred in (
            (True, KVPoll.Transferring, True),
            (True, KVPoll.Success, True),
            (False, KVPoll.Transferring, False),
        ):
            with self.subTest(enabled=enabled, status=status):
                sched, req = self._scheduler(), self._req(status)
                with envs.SGLANG_PP_PD_DEFER_EARLY_SETTLE.override(enabled):
                    mixin._pp_pd_settle_early_sends(sched, SimpleNamespace(reqs=[req]))
                self.assertEqual(getattr(req, "disagg_settle_deferred", False), deferred)
                self.assertEqual(sched._settle_early_sent_last_chunk.called, not deferred)
                if deferred:
                    mixin.send_kv_chunk(sched, req, last_chunk=True)
                    sched._send_kv_chunk.assert_not_called()
                    sched._settle_early_sent_last_chunk.assert_not_called()
                    self.assertNotIn("r", sched.disagg_prefill_early_sent_rids)

    def test_ranks_with_different_local_status_defer_alike(self):
        # PP stages send their own layers: one may have finished while another still sends.
        mixin = SchedulerDisaggregationPrefillMixin
        decisions = []
        for status in (KVPoll.Success, KVPoll.Transferring, KVPoll.WaitingForInput, None):
            sched, req = self._scheduler(), self._req(status)
            req.bootstrap_room = 7
            with envs.SGLANG_PP_PD_DEFER_EARLY_SETTLE.override(True):
                mixin._pp_pd_settle_early_sends(sched, SimpleNamespace(reqs=[req]))
            decisions.append(
                (req.disagg_settle_deferred, sched._settle_early_sent_last_chunk.called)
            )
        self.assertEqual(decisions, [(True, False)] * 4)

    def test_checkpoint_frees_are_held_until_the_transfer_ends(self):
        from sglang.srt.mem_cache.unified_cache.cache_action import FreeDeviceKV
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as URC

        cache = SimpleNamespace(token_to_kv_pool_allocator=Mock())
        cache._apply_cache_action = lambda a: URC._apply_cache_action(cache, a)
        cache._apply_cache_actions = lambda a: URC._apply_cache_actions(cache, a)
        cache.defer_device_frees = lambda sink: URC.defer_device_frees(cache, sink)
        cache.apply_deferred_frees = lambda sink: URC.apply_deferred_frees(cache, sink)
        dup = torch.arange(64)

        def checkpoint(req, c):  # the insert finds the prefix in the tree: free the copy
            c._apply_cache_actions([FreeDeviceKV(indices=[dup])])

        sched = self._scheduler(cache)
        req = SimpleNamespace(pending_bootstrap=False, disagg_settle_deferred=True)
        with patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache", checkpoint):
            SchedulerDisaggregationPrefillMixin.checkpoint_disagg_prefill(sched, req)
        free = cache.token_to_kv_pool_allocator.free_segment
        free.assert_not_called()
        self.assertEqual(len(req.disagg_deferred_frees), 1)
        sched._apply_deferred_frees(req)  # the inflight queue saw Success or Failed
        free.assert_called_once_with(dup, start_pos=0)
        self.assertIsNone(req.disagg_deferred_frees)
        # outside the deferral, frees go straight to the allocator
        cache._apply_cache_actions([FreeDeviceKV(indices=[dup])])
        self.assertEqual(free.call_count, 2)

    def test_idle_pool_log_reports_held_frees(self):
        sched = self._scheduler()
        sched.pool_stats_observer = SimpleNamespace(
            get_pool_stats=lambda: SimpleNamespace(
                full_num_used=0, full_available_size=900, full_evictable_size=100,
                full_capacity=1000, mamba_num_used=0, mamba_available_size=8,
            )
        )
        sched.disagg_prefill_inflight_queue = [SimpleNamespace(disagg_deferred_frees=[1])]
        with self.assertLogs("sglang.srt.disaggregation.prefill", level="INFO") as logs:
            SchedulerDisaggregationPrefillMixin._log_pool_at_idle(sched)
        self.assertIn("full used=0 available=900", logs.output[0])
        self.assertIn("holding deferred frees=1", logs.output[0])

    def test_idle_pool_log_without_early_send(self):
        # without early send its stats are never set
        sched = self._scheduler()
        del sched.disagg_prefill_early_send_stats
        sched.pool_stats_observer = SimpleNamespace(
            get_pool_stats=lambda: SimpleNamespace(
                full_num_used=0, full_available_size=900, full_evictable_size=100,
                full_capacity=1000, mamba_num_used=0, mamba_available_size=8,
            )
        )
        sched.disagg_prefill_inflight_queue = []
        with self.assertLogs("sglang.srt.disaggregation.prefill", level="INFO") as logs:
            SchedulerDisaggregationPrefillMixin._log_pool_at_idle(sched)
        self.assertIn("deferred settles=0", logs.output[0])


if __name__ == "__main__":
    unittest.main()
