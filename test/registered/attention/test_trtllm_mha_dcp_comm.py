from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend


def _backend() -> TRTLLMHAAttnBackend:
    backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
    backend.dcp_group = object()
    backend.q_data_type = torch.bfloat16
    return backend


def test_merge_dcp_partial_output_uses_a2a_backend():
    backend = _backend()
    partial_o = torch.randn(2, 8, 16)
    partial_lse = torch.randn(2, 8)
    merged = torch.randn(2, 2, 16)

    with (
        patch(
            "sglang.srt.layers.attention.trtllm_mha_backend.get_parallel",
            return_value=SimpleNamespace(dcp_comm_backend="a2a"),
        ),
        patch(
            "sglang.srt.layers.attention.trtllm_mha_backend.dcp_a2a_lse_reduce",
            return_value=merged,
        ) as a2a_merge,
        patch(
            "sglang.srt.layers.attention.trtllm_mha_backend.cp_lse_ag_out_rs_mha"
        ) as ag_rs_merge,
    ):
        result = backend._merge_dcp_partial_output(partial_o, partial_lse)

    assert result.dtype == torch.bfloat16
    a2a_merge.assert_called_once()
    args, kwargs = a2a_merge.call_args
    assert args[0] is partial_o
    assert args[1] is partial_lse
    assert args[2] is backend.dcp_group
    assert kwargs == {"is_lse_base_on_e": False, "comm_backend": "a2a"}
    ag_rs_merge.assert_not_called()


def test_merge_dcp_partial_output_keeps_ag_rs_default():
    backend = _backend()
    partial_o = torch.randn(2, 8, 16)
    partial_lse = torch.randn(2, 8)
    merged = torch.randn(2, 2, 16)

    with (
        patch(
            "sglang.srt.layers.attention.trtllm_mha_backend.get_parallel",
            return_value=SimpleNamespace(dcp_comm_backend="ag_rs"),
        ),
        patch(
            "sglang.srt.layers.attention.trtllm_mha_backend.dcp_a2a_lse_reduce"
        ) as a2a_merge,
        patch(
            "sglang.srt.layers.attention.trtllm_mha_backend.cp_lse_ag_out_rs_mha",
            return_value=merged,
        ) as ag_rs_merge,
    ):
        result = backend._merge_dcp_partial_output(partial_o, partial_lse)

    assert result.dtype == torch.bfloat16
    ag_rs_merge.assert_called_once()
    args, kwargs = ag_rs_merge.call_args
    assert args[0] is partial_o
    assert args[1] is partial_lse
    assert args[2] is backend.dcp_group
    assert kwargs == {"is_lse_base_on_e": False}
    a2a_merge.assert_not_called()
