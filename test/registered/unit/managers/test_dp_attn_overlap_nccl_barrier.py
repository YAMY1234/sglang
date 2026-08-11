import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers.attention.base_attn_backend import (  # noqa: E402
    SharedReadBoundary,
)
from sglang.srt.layers.attention.trtllm_mha_backend import (  # noqa: E402
    TRTLLMHAAttnBackend,
)
from sglang.srt.managers.scheduler_components import dp_attn  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (  # noqa: E402
    EAGLEDraftExtendCudaGraphRunner,
)

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

    def test_overlap_nccl_does_not_wait_for_entire_forward(self):
        schedule_stream, _ = self._prepare(
            use_nccl=True, disable_overlap_schedule=False
        )
        schedule_stream.wait_stream.assert_not_called()

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


class TestDraftExtendWARBoundary(CustomTestCase):
    def test_trtllm_mha_nccl_gather_publishes_after_draft_replay(self):
        backend = object.__new__(TRTLLMHAAttnBackend)
        with (
            envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.override(True),
            envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False),
        ):
            self.assertIs(
                backend.shared_read_boundary(ForwardMode.DRAFT_EXTEND_V2),
                SharedReadBoundary.POST_REPLAY,
            )

    def test_trtllm_mha_gloo_gather_keeps_in_replay_boundary(self):
        backend = object.__new__(TRTLLMHAAttnBackend)
        with envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.override(
            False
        ):
            self.assertIs(
                backend.shared_read_boundary(ForwardMode.DRAFT_EXTEND_V2),
                SharedReadBoundary.IN_REPLAY,
            )

    def test_in_replay_boundary_requires_captured_event_node(self):
        runner = object.__new__(EAGLEDraftExtendCudaGraphRunner)
        runner.draft_extend_attn_backend = MagicMock(
            shared_read_boundary=MagicMock(return_value=SharedReadBoundary.IN_REPLAY)
        )
        runner.forward_mode = ForwardMode.DRAFT_EXTEND_V2

        runner._war_read_done_node_planted = False
        self.assertIs(runner._war_read_done_boundary(), SharedReadBoundary.UNKNOWN)

        runner._war_read_done_node_planted = True
        self.assertIs(runner._war_read_done_boundary(), SharedReadBoundary.IN_REPLAY)


if __name__ == "__main__":
    unittest.main()
