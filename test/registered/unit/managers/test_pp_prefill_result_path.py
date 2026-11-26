"""The PP PD prefill result path: the local result that skips the output ring,
the relay of a launch before an older result's wait, the host copy of a received
result, and the PP communication stream's choice among the held streams.
"""

import itertools
import unittest
from collections import deque
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.disaggregation.utils import TransferBackend
from sglang.srt.environ import envs
from sglang.srt.managers import scheduler_pp_mixin as pp_mod
from sglang.srt.managers import utils as managers_utils
from sglang.srt.managers.scheduler_pp_mixin import PPBatchMetadata, SchedulerPPMixin
from sglang.srt.managers.utils import GenerationBatchResult, allocate_distinct_stream
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Stream:
    def __init__(self, name):
        self.name = name
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event)

    def wait_stream(self, stream):
        self.waited.append(stream)


class _Event:
    def __init__(self):
        self.stream = None
        self.after = None

    def record(self, stream=None):
        self.stream = stream
        # What the stream had waited when the event was recorded.
        self.after = list(stream.waited)


class _StreamContext:
    def __init__(self, device, stream):
        self.device = device
        self.stream = stream

    def __enter__(self):
        self.outer, self.device.current = self.device.current, self.stream

    def __exit__(self, *exc):
        self.device.current = self.outer


class _Device:
    def __init__(self):
        self.current = _Stream("current")

    def Event(self):
        return _Event()

    def Stream(self):
        return _Stream("side")

    def current_stream(self):
        return self.current

    def stream(self, stream):
        return _StreamContext(self, stream)


class _Stage(SchedulerPPMixin):
    def __init__(self, is_last_rank):
        self.device = "cpu"
        self.device_module = _Device()
        self.pp_group = SimpleNamespace(is_last_rank=is_last_rank)
        self._pp_pd_local_result_enabled = True
        self._pp_local_outputs = {}
        self.forward_stream = _Stream("forward")
        self.forward_stream_ctx = self.device_module.stream(self.forward_stream)
        self.copy_stream = _Stream("copy")
        self.copy_stream_ctx = self.device_module.stream(self.copy_stream)
        self.schedule_stream = _Stream("schedule")
        self.pp_comm_stream = None


def _req(**flags):
    return SimpleNamespace(
        grammar=flags.get("grammar"),
        return_sampling_mask=flags.get("return_sampling_mask", False),
        return_routed_experts=flags.get("return_routed_experts", False),
    )


def _batch(*reqs, forward_mode=ForwardMode.EXTEND, return_logprob=False):
    return SimpleNamespace(
        reqs=list(reqs) or [_req(), _req()],
        forward_mode=forward_mode,
        return_logprob=return_logprob,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        spec_info=None,
    )


def _forward_result():
    return SimpleNamespace(
        can_run_cuda_graph=False,
        spec_verify_out_cache_loc=None,
        fpm_timing=None,
        forward_launch=None,
        unified_forward_done=None,
        routed_experts_output=None,
    )


def _exec(capture_routed_experts=False):
    features = SimpleNamespace(enable_return_routed_experts=capture_routed_experts)
    return mock.patch.object(
        pp_mod, "get_exec", return_value=SimpleNamespace(features=features)
    )


def _topology(**fields):
    return mock.patch.object(
        pp_mod, "get_parallel", return_value=SimpleNamespace(**fields)
    )


class TestLocalResultDecision(CustomTestCase):
    def test_only_plain_extend_batches_skip_the_ring(self):
        stage = _Stage(is_last_rank=False)
        with _exec():
            self.assertTrue(stage._pp_pd_local_result(_batch()))
            for batch in (
                _batch(return_logprob=True),
                _batch(forward_mode=ForwardMode.DECODE),
                _batch(_req(), _req(grammar=object())),
                _batch(_req(return_sampling_mask=True)),
                _batch(_req(return_routed_experts=True)),
            ):
                self.assertFalse(stage._pp_pd_local_result(batch))
        # The output message carries every batch's routed experts while captured.
        with _exec(capture_routed_experts=True):
            self.assertFalse(stage._pp_pd_local_result(_batch()))
        stage._pp_pd_local_result_enabled = False
        with _exec():
            self.assertFalse(stage._pp_pd_local_result(_batch()))

    def test_engages_only_on_the_mooncake_backend(self):
        stage = _Stage(is_last_rank=False)
        stage._pp_spec_relay = False
        stage.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(sampling_observer=None)
        )
        stage.transfer_backend = TransferBackend.MOONCAKE
        self.assertFalse(stage._pp_pd_local_result_engages())
        with (
            envs.SGLANG_PP_PD_LOCAL_RESULT.override(True),
            mock.patch.object(pp_mod, "_is_npu", False),
        ):
            self.assertTrue(stage._pp_pd_local_result_engages())
            # Another sender would still write the earlier stages' placeholder aux.
            stage.transfer_backend = TransferBackend.NIXL
            with self.assertRaises(ValueError):
                stage._pp_pd_local_result_engages()


class TestLastStageLocalResult(CustomTestCase):
    def test_processes_its_own_held_output_without_the_ring(self):
        stage = _Stage(is_last_rank=True)
        stage._pp_wait_forward_dependencies = lambda: None
        stage.run_batch = lambda batch, pp_proxy_tensors: _forward_result()
        token_ids = torch.tensor([5, 6])
        stage._pp_prepare_tensor_dict = lambda result, batch: {
            "next_token_ids": token_ids
        }
        stage._pp_send_dict_to_next_stage = mock.Mock()
        stage._pp_recv_dict_from_prev_stage = mock.Mock()
        processed = []
        stage._pp_prep_batch_result = lambda batch, meta, outputs: (
            processed.append(outputs) or SimpleNamespace(logits_output=None)
        )
        batch = _batch()
        mbs, metadata, queue = [batch, None], [None, None], deque()
        stage.mb_metadata = metadata

        with _exec():
            stage._pp_launch_batch(0, batch, None, metadata, queue)
        self.assertTrue(metadata[0].pd_local_result)
        ((ready, outputs),) = queue
        # Held for later iterations: copied out of buffers a later forward reuses.
        held = outputs.tensors["next_token_ids"]
        self.assertIsNot(held, token_ids)
        self.assertTrue(torch.equal(held, token_ids))
        self.assertIsNot(ready, metadata[0].forward_done)
        self.assertIs(ready.stream, stage.forward_stream)

        with (
            _topology(pp_rank=1, pp_size=2),
            mock.patch.object(pp_mod, "_is_npu", False),
        ):
            # The iteration that would return micro-batch 0 to the first stage.
            _, result, _, work = stage._pp_send_recv_and_preprocess_output_tensors(
                0, 1, mbs, metadata, queue, None
            )
            self.assertIsNone(result)
            self.assertEqual(work, [])
            self.assertEqual(list(stage._pp_local_outputs), [0])
            # Without a PP comm stream the skipped send fenced the scheduler
            # stream on this forward; a chunked request's next KV send needs it.
            self.assertEqual(stage.device_module.current.waited, [ready])
            # The iteration that processes micro-batch 0.
            pp_outputs, result, completion, _ = (
                stage._pp_send_recv_and_preprocess_output_tensors(
                    1, 0, mbs, metadata, queue, None
                )
            )
        self.assertIsNone(pp_outputs)
        self.assertEqual(processed, [outputs])
        self.assertEqual(stage._pp_local_outputs, {})
        stage._pp_send_dict_to_next_stage.assert_not_called()
        stage._pp_recv_dict_from_prev_stage.assert_not_called()
        self.assertIs(completion.stream, stage.copy_stream)
        self.assertEqual(
            completion.after,
            [stage.schedule_stream, ready, metadata[0].forward_done],
        )

    def test_comm_stream_takes_no_fence_for_a_held_output(self):
        stage = _Stage(is_last_rank=True)
        stage.pp_comm_stream = _Stream("comm")
        stage.mb_metadata = [PPBatchMetadata(False, pd_local_result=True)]
        ready, outputs = _Event(), object()
        work = stage._pp_send_output_to_next_stage(
            0, [_batch()], deque([(ready, outputs)]), None
        )
        self.assertEqual(work, [])
        self.assertEqual(stage._pp_local_outputs, {0: (ready, outputs)})
        self.assertEqual(stage.device_module.current.waited, [])


class TestEarlierStageLocalResult(CustomTestCase):
    def test_processes_a_host_placeholder_after_its_own_forward(self):
        stage = _Stage(is_last_rank=False)
        forward_done = object()
        metadata = PPBatchMetadata(
            can_run_cuda_graph=False, forward_done=forward_done, pd_local_result=True
        )
        batch = _batch()
        batch.spec_info = object()
        _, result, completion = stage._pp_make_local_result(batch, metadata, 0)
        # The last stage alone sends the draft state to decode.
        self.assertIsNone(batch.spec_info)
        self.assertIsNone(result.next_draft_input)
        self.assertEqual(result.next_token_ids.device.type, "cpu")
        self.assertEqual(result.next_token_ids.dtype, torch.int64)
        self.assertEqual(result.next_token_ids.tolist(), [0, 0])
        # Its KV is written before it is checkpointed or sent; the scheduler's
        # own stream takes no dependency on the forward.
        self.assertIsNot(completion.stream, stage.device_module.current)
        self.assertEqual(completion.after, [forward_done])
        self.assertEqual(stage.device_module.current.waited, [])


class _Stop(Exception):
    pass


class TestRelayBeforeD2H(CustomTestCase):
    def _one_iteration(self, relay_before_d2h, is_last_rank=False):
        """Run one PP PD prefill loop iteration that launches a batch and
        processes an older one; return the order of its steps."""
        order = []
        scheduler = mock.Mock()
        scheduler.pp_loop_size = 2
        scheduler.pp_group = SimpleNamespace(is_last_rank=is_last_rank)
        scheduler._pp_pd_local_result_engages.return_value = False
        scheduler._pp_prefill_host_output_copy_engages.return_value = False
        scheduler._pp_prefill_relay_before_d2h_engages.return_value = relay_before_d2h
        scheduler.running_mbs = [mock.Mock(), mock.Mock()]
        scheduler.last_mbs = [None, None]
        scheduler.mbs = [None, object()]
        scheduler.enable_staging = False
        scheduler.get_new_batch_prefill.return_value = SimpleNamespace(
            batch_to_run=object(), running_batch=mock.Mock()
        )
        scheduler.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = (
            lambda batch: batch
        )
        result = object()
        scheduler._pp_launch_batch.side_effect = lambda *args: (
            order.append("launch") or (result, object())
        )
        d2h_event = SimpleNamespace(synchronize=lambda: order.append("d2h"))
        scheduler._pp_commit_send_output_work_and_preprocess_output_tensors.side_effect = (
            lambda *args: order.append("exchange") or (None, object(), d2h_event)
        )
        scheduler._pp_process_batch_result.side_effect = lambda *args: order.append(
            "process"
        )
        scheduler.process_disagg_prefill_transfer_queue.side_effect = lambda: (
            order.append("release")
        )
        scheduler._pp_relay_to_next_stage.side_effect = lambda reqs, launched: (
            order.append("relay") if launched is result else order.append("bad")
        )
        scheduler.invariant_checker.run.side_effect = _Stop
        with (
            _topology(pp_size=2, pp_async_batch_depth=0),
            self.assertRaises(_Stop),
        ):
            SchedulerPPMixin.event_loop_pp_disagg_prefill(scheduler)
        return order

    def test_relays_the_launch_before_the_older_result(self):
        self.assertEqual(
            self._one_iteration(relay_before_d2h=True),
            ["launch", "exchange", "relay", "d2h", "process", "release"],
        )
        self.assertEqual(
            self._one_iteration(relay_before_d2h=False),
            ["launch", "exchange", "d2h", "process", "release", "relay"],
        )
        self.assertEqual(
            self._one_iteration(relay_before_d2h=False, is_last_rank=True),
            ["launch", "exchange", "d2h", "process", "release"],
        )

    def test_engages_on_the_earlier_stages_of_a_supported_layout(self):
        stage = _Stage(is_last_rank=False)
        stage._pp_spec_relay = False
        supported = dict(tp_size=1, attn_dp_enabled=False, pp_async_batch_depth=0)
        self.assertFalse(stage._pp_prefill_relay_before_d2h_engages())
        with (
            envs.SGLANG_PP_PREFILL_RELAY_BEFORE_D2H.override(True),
            mock.patch.object(pp_mod, "_is_npu", False),
        ):
            with _topology(**supported):
                self.assertTrue(stage._pp_prefill_relay_before_d2h_engages())
                stage.pp_group = SimpleNamespace(is_last_rank=True)
                self.assertFalse(stage._pp_prefill_relay_before_d2h_engages())
            for unsupported in (
                {"tp_size": 2},
                {"attn_dp_enabled": True},
                {"pp_async_batch_depth": 1},
            ):
                with (
                    self.subTest(**unsupported),
                    _topology(**{**supported, **unsupported}),
                ):
                    with self.assertRaises(ValueError):
                        stage._pp_prefill_relay_before_d2h_engages()


class TestHostOutputCopy(CustomTestCase):
    def _receive(self, host_output_copy):
        stage = _Stage(is_last_rank=False)
        stage._pp_prefill_host_output_copy = host_output_copy
        stage.spec_algorithm = SpeculativeAlgorithm.EAGLE
        stage._pp_send_output_to_next_stage = lambda *args: []
        stage._pp_recv_dict_from_prev_stage = lambda: ({}, None)
        result = SimpleNamespace(logits_output=None, copy_to_cpu=mock.Mock())
        stage._pp_prep_batch_result = lambda batch, meta, outputs: result
        metadata = PPBatchMetadata(can_run_cuda_graph=False, forward_done=object())
        with (
            _topology(pp_rank=0, pp_size=2),
            mock.patch.object(pp_mod, "_is_npu", False),
        ):
            _, received, completion, _ = (
                stage._pp_send_recv_and_preprocess_output_tensors(
                    0, 0, [_batch()], [metadata], deque(), None
                )
            )
        self.assertIs(received, result)
        return result, completion

    def test_copies_the_received_result_and_draft_on_arrival(self):
        result, completion = self._receive(host_output_copy=True)
        result.copy_to_cpu.assert_called_once_with(
            return_logprob=False,
            return_hidden_states=False,
            return_pd_prefill_draft=True,
            fence_logprob_copy_devices=True,
        )
        self.assertIs(result.copy_done, completion)
        result, _ = self._receive(host_output_copy=False)
        result.copy_to_cpu.assert_not_called()

    def test_draft_snapshot_leaves_the_device_draft_in_place(self):
        draft = SimpleNamespace(
            topk_p=torch.ones(2, 1),
            topk_index=torch.zeros(2, 1, dtype=torch.long),
            hidden_states=torch.ones(2, 4),
            dsa_topk_indices=None,
        )
        copy_done = mock.Mock()
        result = GenerationBatchResult(
            next_token_ids=torch.tensor([3, 4]),
            next_draft_input=draft,
            copy_done=copy_done,
        )
        result.copy_to_cpu(
            return_logprob=False,
            return_hidden_states=False,
            return_pd_prefill_draft=True,
        )
        self.assertIs(result.next_draft_input, draft)
        topk_p, topk_index, hidden_states, dsa_topk_indices = result.prefill_host_draft
        self.assertTrue(torch.equal(topk_p, draft.topk_p))
        self.assertTrue(torch.equal(topk_index, draft.topk_index))
        self.assertTrue(torch.equal(hidden_states, draft.hidden_states))
        self.assertIsNone(dsa_topk_indices)
        copy_done.record.assert_called_once_with()

    def test_copy_from_another_device_joins_the_copy_stream(self):
        source_streams = {}

        def current_stream(device=None):
            return source_streams.setdefault(device, _Stream(str(device)))

        copy_stream = mock.Mock(device=torch.device("cuda", 0))
        with (
            mock.patch.object(managers_utils, "_async_d2h", return_value="host"),
            mock.patch.object(torch.cuda, "Event", _Event),
            mock.patch.object(torch.cuda, "current_stream", current_stream),
        ):
            local = SimpleNamespace(device=torch.device("cuda", 0))
            self.assertEqual(
                managers_utils._async_logprob_d2h(local, copy_stream), "host"
            )
            copy_stream.wait_event.assert_not_called()
            other = SimpleNamespace(device=torch.device("cuda", 1))
            managers_utils._async_logprob_d2h(other, copy_stream)
        ((joined,), _) = copy_stream.wait_event.call_args
        self.assertIs(joined.stream, source_streams[torch.device("cuda", 1)])


class TestCommStreamAvoidsHeldStreams(CustomTestCase):
    def test_skips_a_side_stream_a_model_holds(self):
        class FakeStream:
            def __init__(self, handle):
                self.cuda_stream = handle

        pool = itertools.cycle(range(4))
        device_module = SimpleNamespace(Stream=lambda priority: FakeStream(next(pool)))
        with mock.patch.object(torch.cuda, "Stream", FakeStream):
            model = SimpleNamespace(alt_stream=FakeStream(0))
            held = pp_mod._held_cuda_streams()
            comm_stream = allocate_distinct_stream(device_module, held)
        self.assertIn(model.alt_stream, held)
        self.assertNotEqual(comm_stream.cuda_stream, 0)


if __name__ == "__main__":
    unittest.main()
