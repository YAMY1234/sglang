from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.models import qwen2_moe


def test_idle_rank_skips_dense_router_and_enters_expert_dispatcher():
    block = qwen2_moe.Qwen2MoeSparseMoeBlock.__new__(qwen2_moe.Qwen2MoeSparseMoeBlock)
    torch.nn.Module.__init__(block)
    block.tp_size = 1
    block.alt_stream = None
    block.topk = Mock()
    empty_topk_output = object()
    block.topk.empty_topk_output.return_value = empty_topk_output
    expected_output = torch.empty((0, 16))
    block.experts = Mock(return_value=expected_output)
    block._forward_shared_experts = Mock(
        side_effect=AssertionError("idle rank entered shared experts")
    )
    block._forward_router_experts = Mock(
        side_effect=AssertionError("idle rank entered router")
    )
    flashinfer_backend = SimpleNamespace(is_deepep=lambda: False)
    hidden_states = torch.empty((0, 16))

    with patch.object(
        qwen2_moe, "get_moe_a2a_backend", return_value=flashinfer_backend
    ):
        output = block.forward(hidden_states)

    block.topk.empty_topk_output.assert_called_once_with(hidden_states.device)
    block.experts.assert_called_once()
    dispatched_hidden_states, dispatched_topk = block.experts.call_args.args
    assert dispatched_hidden_states.shape == hidden_states.shape
    assert dispatched_topk is empty_topk_output
    assert output.shape == expected_output.shape
    assert output.dtype == expected_output.dtype
