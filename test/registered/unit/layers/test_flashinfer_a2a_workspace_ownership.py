import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
    FlashinferCombineInput,
    FlashinferDispatcher,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class _FakeMoeAlltoAll:
    def __init__(self, workspace_payload):
        self.workspace_payload = workspace_payload
        self.payload_in_workspace = None

    def dispatch(
        self,
        _token_selected_experts,
        input_payloads,
        runtime_max_tokens_per_rank,
        **_kwargs,
    ):
        hidden_states, topk_ids, topk_weights = input_payloads
        num_recv_tokens = 2 * runtime_max_tokens_per_rank
        return (
            torch.empty(
                num_recv_tokens,
                hidden_states.shape[-1],
                dtype=hidden_states.dtype,
            ),
            torch.empty(
                num_recv_tokens,
                topk_ids.shape[-1],
                dtype=topk_ids.dtype,
            ),
            torch.empty(
                num_recv_tokens,
                topk_weights.shape[-1],
                dtype=topk_weights.dtype,
            ),
        )

    def get_combine_payload_tensor_in_workspace(
        self, runtime_max_tokens_per_rank, hidden_size, dtype
    ):
        expected_shape = (2, runtime_max_tokens_per_rank, hidden_size)
        assert self.workspace_payload.shape == expected_shape
        assert self.workspace_payload.dtype == dtype
        return self.workspace_payload

    def combine(
        self,
        hidden_states,
        runtime_max_tokens_per_rank,
        *,
        payload_in_workspace,
    ):
        self.payload_in_workspace = payload_in_workspace
        return hidden_states[0]


class TestFlashinferA2AWorkspaceOwnership(unittest.TestCase):
    @staticmethod
    def _make_dispatcher(workspace_payload):
        dispatcher = FlashinferDispatcher.__new__(FlashinferDispatcher)
        dispatcher.ep_size = 2
        dispatcher.router_topk = 1
        dispatcher.hidden_size = 4
        dispatcher.quant_config = {}
        dispatcher.allocate_combine_payload_in_workspace = True
        dispatcher._workspace_combine_payload = None
        dispatcher.moe_a2a = _FakeMoeAlltoAll(workspace_payload)
        return dispatcher

    @staticmethod
    def _dispatch(dispatcher):
        hidden_states = torch.empty(3, 4)
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones(3, 1),
            topk_ids=torch.zeros(3, 1, dtype=torch.int32),
            router_logits=None,
        )
        with patch(
            "sglang.srt.layers.moe.token_dispatcher.flashinfer.get_dp_global_num_tokens",
            return_value=None,
        ):
            return dispatcher.dispatch(hidden_states, topk_output)

    def test_exact_workspace_payload_uses_in_place_combine(self):
        workspace_payload = torch.empty(2, 3, 4)
        dispatcher = self._make_dispatcher(workspace_payload)
        dispatch_output = self._dispatch(dispatcher)

        self.assertEqual(
            dispatch_output.moe_output.data_ptr(), workspace_payload.data_ptr()
        )
        dispatcher.combine(FlashinferCombineInput(dispatch_output.moe_output))

        self.assertTrue(dispatcher.moe_a2a.payload_in_workspace)
        self.assertFalse(hasattr(dispatcher, "_workspace_combine_payload"))

    def test_unquantized_mtp_runner_writes_to_workspace_payload(self):
        workspace_payload = torch.empty(2, 3, 4)
        dispatcher = self._make_dispatcher(workspace_payload)
        dispatch_output = self._dispatch(dispatcher)

        method = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)
        method.use_flashinfer_cutlass = True
        method.moe_runner_config = SimpleNamespace(activation="silu")
        method.runner = SimpleNamespace(runner_backend=MoeRunnerBackend.TRITON)
        layer = SimpleNamespace(
            w13_weight=torch.empty(1, 8, 4),
            w2_weight=torch.empty(1, 4, 4),
            moe_ep_size=2,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
        )

        def fake_cutlass_fused_moe(**kwargs):
            self.assertIs(kwargs["output"], dispatch_output.moe_output)
            self.assertTrue(kwargs["enable_alltoall"])
            return [kwargs["output"]]

        flashinfer_a2a_backend = SimpleNamespace(is_flashinfer=lambda: True)
        with (
            patch(
                "sglang.srt.layers.quantization.unquant.flashinfer_cutlass_fused_moe",
                side_effect=fake_cutlass_fused_moe,
            ),
            patch(
                "sglang.srt.layers.quantization.unquant.get_moe_a2a_backend",
                return_value=flashinfer_a2a_backend,
            ),
        ):
            combine_input = method.forward_cuda(layer, dispatch_output)

        self.assertEqual(
            combine_input.hidden_states.data_ptr(), workspace_payload.data_ptr()
        )
        dispatcher.combine(combine_input)

        self.assertTrue(dispatcher.moe_a2a.payload_in_workspace)
        self.assertFalse(hasattr(dispatcher, "_workspace_combine_payload"))

    def test_external_payload_uses_copy_combine(self):
        workspace_payload = torch.empty(2, 3, 4)
        external_payload = torch.empty(6, 4)
        dispatcher = self._make_dispatcher(workspace_payload)
        self._dispatch(dispatcher)

        dispatcher.combine(FlashinferCombineInput(external_payload))

        self.assertFalse(dispatcher.moe_a2a.payload_in_workspace)
        self.assertFalse(hasattr(dispatcher, "_workspace_combine_payload"))

    def test_same_storage_with_different_offset_uses_copy_combine(self):
        backing = torch.empty(7, 4)
        workspace_payload = backing[:6].view(2, 3, 4)
        offset_alias = backing[1:]
        self.assertEqual(
            workspace_payload.untyped_storage().data_ptr(),
            offset_alias.untyped_storage().data_ptr(),
        )
        self.assertNotEqual(workspace_payload.data_ptr(), offset_alias.data_ptr())
        dispatcher = self._make_dispatcher(workspace_payload)
        self._dispatch(dispatcher)

        dispatcher.combine(FlashinferCombineInput(offset_alias))

        self.assertFalse(dispatcher.moe_a2a.payload_in_workspace)
        self.assertFalse(hasattr(dispatcher, "_workspace_combine_payload"))


if __name__ == "__main__":
    unittest.main()
