"""Regression coverage for padding-aware Qwen MoE routing."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import torch

import sglang.srt.models.qwen2_moe as qwen2_moe
from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _NoA2ABackend:
    def is_deepep(self):
        return False

    def is_deepep_v2(self):
        return False

    def is_mori(self):
        return False


class TestQwen2MoePaddingBoundary(CustomTestCase):
    def _block(self):
        return SimpleNamespace(
            shared_expert_gate=None,
            alt_stream=None,
            tp_size=1,
            _forward_shared_experts=Mock(return_value=None),
            _forward_router_experts=Mock(
                side_effect=lambda hidden_states, **_kwargs: hidden_states
            ),
        )

    def test_normal_path_forwards_real_token_boundary_to_router(self):
        block = self._block()
        hidden_states = torch.randn(8, 16)
        boundary = torch.tensor([5], dtype=torch.int32)
        forward_batch = SimpleNamespace(num_token_non_padded=boundary)

        with (
            patch.object(
                qwen2_moe,
                "get_moe_a2a_backend",
                return_value=_NoA2ABackend(),
            ),
            patch.object(qwen2_moe, "is_npu", return_value=False),
        ):
            output = Qwen2MoeSparseMoeBlock.forward(
                block, hidden_states, forward_batch
            )

        self.assertTrue(torch.equal(output, hidden_states))
        block._forward_router_experts.assert_called_once()
        routed_args, routed_kwargs = block._forward_router_experts.call_args
        self.assertEqual(routed_args[0].data_ptr(), hidden_states.data_ptr())
        self.assertIs(routed_kwargs["num_token_non_padded"], boundary)
        self.assertFalse(routed_kwargs["defer_finalize"])

    def test_router_forwards_real_token_boundary_to_topk(self):
        hidden_states = torch.randn(8, 16)
        router_logits = torch.randn(8, 4)
        boundary = torch.tensor([5], dtype=torch.int32)
        block = SimpleNamespace(
            gate=Mock(return_value=(router_logits, None)),
            topk=Mock(return_value=sentinel.topk_output),
            enable_shared_expert_fusion=False,
            experts=Mock(return_value=hidden_states),
        )

        output = Qwen2MoeSparseMoeBlock._forward_router_experts(
            block,
            hidden_states,
            num_token_non_padded=boundary,
        )

        self.assertIs(output, hidden_states)
        block.topk.assert_called_once_with(
            hidden_states,
            router_logits,
            num_token_non_padded=boundary,
        )
        block.experts.assert_called_once_with(hidden_states, sentinel.topk_output)

    def test_capture_dual_stream_receives_real_token_boundary(self):
        block = self._block()
        block.alt_stream = object()
        block.forward_normal_dual_stream = Mock(
            return_value=(torch.randn(8, 16), None)
        )
        hidden_states = torch.randn(8, 16)
        boundary = torch.tensor([5], dtype=torch.int32)
        forward_batch = SimpleNamespace(num_token_non_padded=boundary)

        with (
            patch.object(
                qwen2_moe,
                "get_moe_a2a_backend",
                return_value=_NoA2ABackend(),
            ),
            patch.object(qwen2_moe, "is_npu", return_value=False),
            patch.object(qwen2_moe, "get_is_capture_mode", return_value=True),
            patch.object(torch.compiler, "is_compiling", return_value=False),
        ):
            output = Qwen2MoeSparseMoeBlock.forward(
                block, hidden_states, forward_batch
            )

        self.assertTrue(
            torch.equal(output, block.forward_normal_dual_stream.return_value[0])
        )
        block.forward_normal_dual_stream.assert_called_once()
        dual_args, dual_kwargs = block.forward_normal_dual_stream.call_args
        self.assertEqual(dual_args[0].data_ptr(), hidden_states.data_ptr())
        self.assertIs(dual_kwargs["num_token_non_padded"], boundary)
        self.assertFalse(dual_kwargs["use_fused_gate"])
        self.assertFalse(dual_kwargs["defer_finalize"])


if __name__ == "__main__":
    unittest.main()
