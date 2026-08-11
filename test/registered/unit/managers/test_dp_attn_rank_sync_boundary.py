import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers.communicator import LayerCommunicator  # noqa: E402
from sglang.srt.managers.scheduler_components import dp_attn  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.model_executor.forward_context import (  # noqa: E402
    ForwardContext,
    forward_context,
)
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _TboPreparer:
    def prepare_all_gather(self, _local_batch):
        return False, ForwardMode.IDLE.value

    def compute_output(self, _tp0_info):
        return None, ForwardMode.IDLE.value


class TestDPAttnRankSyncBoundary(CustomTestCase):
    def _prepare(
        self,
        *,
        use_nccl: bool,
        disable_overlap_schedule: bool = False,
        rank_sync_event=None,
        rank_sync_ordering_required: bool = True,
        rank_sync_boundary_enabled: bool = False,
        rank_sync_requires_stream_fallback: bool = False,
    ):
        schedule_stream = MagicMock()
        forward_stream = object()
        device_module = SimpleNamespace(current_stream=lambda: schedule_stream)
        tp_group = SimpleNamespace(
            device="cuda",
            device_group=object(),
            cpu_group=object(),
        )
        model_runner = SimpleNamespace(forward_stream=forward_stream)
        rank_sync_runner = SimpleNamespace(
            rank_sync_done_event=rank_sync_event,
            rank_sync_ordering_required=rank_sync_ordering_required,
            rank_sync_boundary_enabled=rank_sync_boundary_enabled,
            rank_sync_requires_stream_fallback=rank_sync_requires_stream_fallback,
        )

        def fake_all_gather(sync_info, **_kwargs):
            sync_info.global_num_tokens = [0]
            sync_info.tp0_info_cpu = MagicMock()

        with (
            envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False),
            envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.override(
                use_nccl
            ),
            patch.object(dp_attn, "world_dp_gather_enabled", return_value=False),
            patch.object(dp_attn, "check_cuda_graph_backend", return_value=False),
            patch.object(
                dp_attn, "TboDPAttentionPreparer", return_value=_TboPreparer()
            ),
            patch.object(
                dp_attn.torch, "get_device_module", return_value=device_module
            ),
            patch.object(
                dp_attn.MLPSyncBatchInfo,
                "all_gather",
                autospec=True,
                side_effect=fake_all_gather,
            ),
        ):
            dp_attn.prepare_mlp_sync_batch_raw(
                local_batch=None,
                model_runner=model_runner,
                dp_size=1,
                attn_tp_size=1,
                attn_cp_size=1,
                tp_group=tp_group,
                get_idle_batch=MagicMock(),
                disable_cuda_graph=False,
                require_mlp_tp_gather=False,
                disable_overlap_schedule=disable_overlap_schedule,
                offload_tags=set(),
                rank_sync_runner=rank_sync_runner,
            )

        return schedule_stream, forward_stream, rank_sync_runner

    def test_gpu_metadata_sync_waits_only_for_rank_sync_event(self):
        rank_sync_event = object()
        schedule_stream, _, runner = self._prepare(
            use_nccl=True, rank_sync_event=rank_sync_event
        )

        schedule_stream.wait_event.assert_called_once_with(rank_sync_event)
        schedule_stream.wait_stream.assert_not_called()
        self.assertIsNone(runner.rank_sync_done_event)

    def test_required_boundary_falls_back_when_unpublished(self):
        schedule_stream, forward_stream, _ = self._prepare(use_nccl=True)

        schedule_stream.wait_stream.assert_called_once_with(forward_stream)
        schedule_stream.wait_event.assert_not_called()

    def test_unrelated_model_does_not_wait(self):
        schedule_stream, _, runner = self._prepare(
            use_nccl=True,
            rank_sync_ordering_required=False,
            rank_sync_event=object(),
            rank_sync_requires_stream_fallback=True,
        )

        schedule_stream.wait_event.assert_not_called()
        schedule_stream.wait_stream.assert_not_called()
        self.assertIsNotNone(runner.rank_sync_done_event)
        self.assertTrue(runner.rank_sync_requires_stream_fallback)

    def test_first_sync_skips_wait_when_narrow_boundary_is_enabled(self):
        schedule_stream, _, _ = self._prepare(
            use_nccl=True,
            rank_sync_boundary_enabled=True,
        )

        schedule_stream.wait_event.assert_not_called()
        schedule_stream.wait_stream.assert_not_called()

    def test_required_predecessor_explicitly_falls_back(self):
        schedule_stream, forward_stream, runner = self._prepare(
            use_nccl=True,
            rank_sync_boundary_enabled=True,
            rank_sync_requires_stream_fallback=True,
        )

        schedule_stream.wait_event.assert_not_called()
        schedule_stream.wait_stream.assert_called_once_with(forward_stream)
        self.assertFalse(runner.rank_sync_requires_stream_fallback)

    def test_gloo_does_not_wait_or_consume_gpu_boundary(self):
        rank_sync_event = object()
        schedule_stream, _, runner = self._prepare(
            use_nccl=False, rank_sync_event=rank_sync_event
        )

        schedule_stream.wait_event.assert_not_called()
        schedule_stream.wait_stream.assert_not_called()
        self.assertIs(runner.rank_sync_done_event, rank_sync_event)

    def test_non_overlap_does_not_add_redundant_wait(self):
        schedule_stream, _, _ = self._prepare(
            use_nccl=True,
            disable_overlap_schedule=True,
            rank_sync_event=object(),
        )

        schedule_stream.wait_event.assert_not_called()
        schedule_stream.wait_stream.assert_not_called()


class TestLayerRankSyncPublish(CustomTestCase):
    def test_last_layer_records_after_postprocess(self):
        event = MagicMock()
        communicator = object.__new__(LayerCommunicator)
        communicator.is_last_layer = True
        communicator.allow_reduce_scatter = True
        communicator._context = object()
        communicator._communicate_summable_tensor_pair_fn = MagicMock(
            return_value=("hidden-out", "residual-out")
        )

        with (
            forward_context(
                ForwardContext(attn_backend=object(), rank_sync_done_event=event)
            ),
        ):
            output = communicator.postprocess_layer(
                hidden_states="hidden-in",
                residual="residual-in",
                forward_batch=object(),
            )

        self.assertEqual(output, ("hidden-out", "residual-out"))
        communicator._communicate_summable_tensor_pair_fn.assert_called_once()
        event.record.assert_called_once_with()


class TestQwen35MTPRankSyncCapability(CustomTestCase):
    def test_boundary_requires_collective_free_logits_tail(self):
        model = object.__new__(Qwen3_5ForCausalLMMTP)

        self.assertTrue(model.requires_overlap_scheduler_rank_sync_ordering)

        model.logits_processor = SimpleNamespace(do_tensor_parallel_all_gather=False)
        self.assertTrue(model.rank_sync_boundary_after_last_layer_communication)

        model.logits_processor.do_tensor_parallel_all_gather = True
        self.assertFalse(model.rank_sync_boundary_after_last_layer_communication)


if __name__ == "__main__":
    unittest.main()
