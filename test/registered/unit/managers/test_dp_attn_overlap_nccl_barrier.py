import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler_components import dp_attn  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _TboPreparer:
    def prepare_all_gather(self, _local_batch):
        return False, ForwardMode.IDLE.value

    def compute_output(self, _tp0_info):
        return None, ForwardMode.IDLE.value


class TestDPAttnOverlapNCCLBarrier(CustomTestCase):
    def _prepare(
        self,
        *,
        use_nccl: bool,
        disable_overlap_schedule: bool,
        skip_all_gather: bool = False,
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

        def fake_all_gather(sync_info, **_kwargs):
            sync_info.global_num_tokens = [0]
            sync_info.tp0_info = MagicMock()

        with (
            envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(skip_all_gather),
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
            )

        return schedule_stream, forward_stream

    def test_overlap_nccl_waits_for_previous_forward(self):
        schedule_stream, forward_stream = self._prepare(
            use_nccl=True, disable_overlap_schedule=False
        )
        schedule_stream.wait_stream.assert_called_once_with(forward_stream)

    def test_gloo_overlap_does_not_add_gpu_barrier(self):
        schedule_stream, _ = self._prepare(
            use_nccl=False, disable_overlap_schedule=False
        )
        schedule_stream.wait_stream.assert_not_called()

    def test_non_overlap_does_not_add_redundant_barrier(self):
        schedule_stream, _ = self._prepare(use_nccl=True, disable_overlap_schedule=True)
        schedule_stream.wait_stream.assert_not_called()

    def test_skipped_all_gather_does_not_add_redundant_barrier(self):
        schedule_stream, _ = self._prepare(
            use_nccl=True,
            disable_overlap_schedule=False,
            skip_all_gather=True,
        )
        schedule_stream.wait_stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
