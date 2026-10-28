import ast
import contextlib
import os
import unittest
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    HiCacheRestoreResult,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.decode_hicache_mixin import (
    HiCacheRestoreGatedKVReceiver,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=13, suite="base-a-test-cpu")


def _run_as_single_rank(test: unittest.TestCase) -> None:
    # The transfer queue sizes its release collectives from the gloo group;
    # these tests have no process group, so answer as one rank.
    world_size = patch("torch.distributed.get_world_size", return_value=1)
    world_size.start()
    test.addCleanup(world_size.stop)


class TestDecodeQueueCleanup(CustomTestCase):
    def setUp(self):
        _run_as_single_rank(self)
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

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


class TestDeferredPreallocMetadata(CustomTestCase):
    def test_metadata_send_waits_for_nonblocking_index_copies(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.alloc.return_value = 7
        queue.req_to_token_pool = SimpleNamespace(
            req_index_to_mamba_index_mapping=torch.tensor(
                [[9], [11], [13]], dtype=torch.int32
            ),
            translate_mamba_indices=lambda indices: indices,
        )
        queue.transfer_queue = SimpleNamespace(enable_staging=False)
        queue._deferred_sends = deque()
        queue._num_published_destinations = 0

        receiver = SimpleNamespace(require_staging=False, send_metadata=MagicMock())
        decode_req = SimpleNamespace(
            req=SimpleNamespace(
                bootstrap_room=3,
                finished_reason=None,
                kv=SimpleNamespace(req_pool_idx=1),
                time_stats=MagicMock(),
            ),
            kv_receiver=receiver,
            metadata_buffer_index=-1,
            is_rebootstrap=False,
        )
        kv_indices = torch.arange(16, 24, dtype=torch.int64)
        event = MagicMock()
        event.query.return_value = False
        torch_empty = torch.empty

        def unpinned_empty(shape, *, dtype, pin_memory):
            self.assertTrue(pin_memory)
            return torch_empty(shape, dtype=dtype)

        with (
            patch(
                "sglang.srt.disaggregation.decode.torch.empty",
                side_effect=unpinned_empty,
            ),
            patch(
                "sglang.srt.disaggregation.decode.torch.cuda.Event",
                return_value=event,
            ),
        ):
            queue._defer_kv_metadata(
                decode_req,
                kv_indices,
                4,
                has_state=True,
                metadata_kwargs={"decode_prefix_len": 5},
            )

        self.assertEqual(decode_req.metadata_buffer_index, 7)
        self.assertEqual(queue._num_published_destinations, 1)
        decode_req.req.time_stats.set_decode_transfer_queue_entry_time.assert_called_once()
        event.record.assert_called_once()

        queue.flush_deferred_sends()
        receiver.send_metadata.assert_not_called()
        self.assertEqual(len(queue._deferred_sends), 1)

        event.query.return_value = True
        queue.flush_deferred_sends()
        self.assertEqual(len(queue._deferred_sends), 0)
        args, kwargs = receiver.send_metadata.call_args
        np.testing.assert_array_equal(args[0], np.array([4, 5], dtype=np.int32))
        self.assertEqual(args[1], 7)
        np.testing.assert_array_equal(args[2][0][0], np.array([11], dtype=np.int32))
        self.assertEqual(kwargs, {"decode_prefix_len": 5})


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
    def setUp(self):
        super().setUp()
        _run_as_single_rank(self)

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
            queue.metadata_buffers.clear_slot = lambda idx: (
                queue.metadata_buffers.bootstrap_room.__setitem__(idx, 0)
            )
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


def _queue_scheduler():
    scheduler = SimpleNamespace(
        scheduler_stage_metrics=None,
        enable_decode_hicache=False,
        enable_hisparse=False,
        waiting_queue=[],
        polling_count=0,
        polling_interval=1,
        disagg_decode_transfer_queue=SimpleNamespace(
            resolve_deferred_releases=MagicMock(),
            pop_transferred=MagicMock(return_value=[]),
            extend=MagicMock(),
        ),
        disagg_decode_prealloc_queue=SimpleNamespace(
            flush_deferred_sends=MagicMock(),
            resume_retracted_reqs=MagicMock(return_value=[]),
            retracted_queue=[],
            pop_preallocated=MagicMock(return_value=([], [])),
        ),
    )
    scheduler._preallocate_new_requests = lambda: (
        SchedulerDisaggregationDecodeMixin._preallocate_new_requests(scheduler)
    )
    return scheduler


def _admit_req(**overrides):
    fields = dict(
        grammar=None,
        output_dsa_topk_indices=None,
        kv=SimpleNamespace(retraction_backup=None),
        hidden_states_tensor=object(),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestPDDecRuntimeSwitch(CustomTestCase):
    def test_all_five_readers_follow_shared_and_component_switches(self):
        # Evaluate the production predicates without constructing CUDA backends
        # or entering the scheduler's event loop.
        runtime = Path(__file__).resolve().parents[4] / "python/sglang/srt"
        shared = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"
        component = "SGLANG_ENABLE_NV_MEGA_PDDEC_RUNTIME_OPTIMIZATION"
        predicates = []
        for relative in (
            "disaggregation/decode.py",
            "layers/attention/trtllm_mha_backend.py",
            "speculative/eagle_draft_extend_cuda_graph_runner.py",
        ):
            tree = ast.parse((runtime / relative).read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.BoolOp):
                    continue
                reads = {
                    value.func.value.attr
                    for value in node.values
                    if isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "get"
                    and isinstance(value.func.value, ast.Attribute)
                    and isinstance(value.func.value.value, ast.Name)
                    and value.func.value.value.id == "envs"
                }
                # Other runtime sections share these files and the master switch.
                # Count this component's five readers without gating ARRIVE.
                if component in reads:
                    self.assertIn(shared, reads, (relative, node.lineno))
                    self.assertIsInstance(node.op, ast.And)
                    predicates.append(
                        (relative, compile(ast.Expression(node), relative, "eval"))
                    )
        self.assertEqual(len(predicates), 5)
        scheduler = SimpleNamespace(
            enable_overlap=True,
            spec_algorithm=SimpleNamespace(is_eagle=lambda: True),
            enable_hisparse=False,
            decode_seq_len_splits=4,
        )
        context = dict(
            envs=envs,
            self=scheduler,
            fm=SimpleNamespace(is_draft_extend_v2=lambda: True),
            batch=SimpleNamespace(reqs=[_admit_req()]),
            get_disagg=lambda: SimpleNamespace(
                disaggregation_decode_enable_radix_cache=False,
                disaggregation_decode_host_receive_threshold=1,
            ),
        )
        for master in (False, True):
            for override in (None, "0", "1"):
                with self.subTest(master=master, component=override):
                    with patch.dict(os.environ), getattr(envs, shared).override(master):
                        os.environ.pop(component, None)
                        if override is not None:
                            os.environ[component] = override
                        expected = master and override != "0"
                        for relative, predicate in predicates:
                            self.assertEqual(
                                eval(predicate, context), expected, relative
                            )
                        # Arrive and other components still read the shared switch.
                        self.assertEqual(getattr(envs, shared).get(), master)


class TestAdmissionOverlap(CustomTestCase):
    def test_scope_uses_admitted_requests_and_keeps_host_read_paths_stock(self):
        scheduler = SimpleNamespace(
            enable_overlap=True,
            spec_algorithm=SimpleNamespace(is_eagle=lambda: True),
            enable_hisparse=False,
        )
        disagg = SimpleNamespace(
            disaggregation_decode_host_receive_threshold=1,
            disaggregation_decode_enable_radix_cache=False,
        )

        def scope(reqs):
            return SchedulerDisaggregationDecodeMixin._nonblocking_admit_in_scope(
                scheduler, SimpleNamespace(reqs=reqs)
            )

        with patch("sglang.srt.disaggregation.decode.get_disagg", return_value=disagg):
            with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(False):
                self.assertFalse(scope([_admit_req()]))
            with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True):
                self.assertTrue(scope([_admit_req()]))
                disagg.disaggregation_decode_enable_radix_cache = True
                self.assertFalse(scope([_admit_req()]))
                disagg.disaggregation_decode_enable_radix_cache = False
                for req in (
                    _admit_req(grammar=object()),
                    _admit_req(output_dsa_topk_indices=object()),
                    _admit_req(kv=SimpleNamespace(retraction_backup=object())),
                    _admit_req(hidden_states_tensor=None),
                ):
                    self.assertFalse(scope([req]))
                scheduler.enable_hisparse = True
                self.assertFalse(scope([_admit_req()]))
                scheduler.enable_hisparse = False
                scheduler.enable_overlap = False
                self.assertFalse(scope([_admit_req()]))

    def test_seed_is_ordered_on_forward_stream_without_schedule_wait(self):
        for nonblocking in (False, True):
            batch = MagicMock()
            batch.is_empty.return_value = False
            calls = []

            @contextlib.contextmanager
            def forward_context():
                calls.append("enter_forward")
                yield
                calls.append("leave_forward")

            scheduler = SimpleNamespace(
                _get_new_prebuilt_batch=lambda _: batch,
                _nonblocking_admit_in_scope=lambda _: nonblocking,
                _verify_kv_checksums=lambda _: calls.append("verify"),
                ngram_embedding_manager=MagicMock(),
                chunked_req=None,
                future_map=object(),
                enable_overlap=True,
                schedule_stream=MagicMock(),
                forward_stream=MagicMock(),
                forward_stream_ctx=forward_context(),
            )
            batch.process_prebuilt.side_effect = lambda _: calls.append("seed")
            result = SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch(
                scheduler, object()
            )
            self.assertIs(result, batch)
            if nonblocking:
                self.assertEqual(
                    calls, ["verify", "enter_forward", "seed", "leave_forward"]
                )
                scheduler.schedule_stream.wait_stream.assert_not_called()
                scheduler.forward_stream.wait_stream.assert_called_once_with(
                    scheduler.schedule_stream
                )
                batch.req_pool_indices.record_stream.assert_called_once_with(
                    scheduler.forward_stream
                )
                batch.seq_lens.record_stream.assert_called_once_with(
                    scheduler.forward_stream
                )
            else:
                self.assertEqual(calls, ["verify", "seed"])
                scheduler.schedule_stream.wait_stream.assert_called_once_with(
                    scheduler.forward_stream
                )
                scheduler.forward_stream.wait_stream.assert_not_called()

    def test_deferred_prealloc_polls_and_flushes_once_and_preserves_retraction(self):
        disagg = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False,
            disaggregation_decode_host_receive_threshold=1,
            disaggregation_decode_polling_interval=1,
        )
        with patch("sglang.srt.disaggregation.decode.get_disagg", return_value=disagg):
            for defer in (False, True):
                scheduler = _queue_scheduler()
                deferred = SchedulerDisaggregationDecodeMixin.process_decode_queue(
                    scheduler, defer_prealloc=defer
                )
                self.assertEqual(deferred, defer)
                queue = scheduler.disagg_decode_prealloc_queue
                queue.flush_deferred_sends.assert_called_once()
                scheduler.disagg_decode_transfer_queue.pop_transferred.assert_called_once()
                self.assertEqual(queue.pop_preallocated.call_count, int(not defer))
                if deferred:
                    scheduler._preallocate_new_requests()
                queue.pop_preallocated.assert_called_once()
            scheduler = _queue_scheduler()
            scheduler.disagg_decode_prealloc_queue.retracted_queue.append(object())
            self.assertFalse(
                SchedulerDisaggregationDecodeMixin.process_decode_queue(
                    scheduler, defer_prealloc=True
                )
            )
            scheduler.disagg_decode_prealloc_queue.pop_preallocated.assert_not_called()


if __name__ == "__main__":
    unittest.main()
