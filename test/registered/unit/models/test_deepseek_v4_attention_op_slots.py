import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.models import deepseek_v4
from sglang.srt.models.deepseek_v4 import MQALayer, MQAOpSlots
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_ENV_GATE = "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate"
_ROWS, _LOCAL_HEADS, _PADDED_HEADS, _DIM, _GROUPS, _RANK = 3, 16, 64, 8, 2, 4
_DECODE = SimpleNamespace(
    forward_mode=SimpleNamespace(
        is_decode_or_idle=lambda: True,
        is_decode=lambda: True,
        is_target_verify=lambda: False,
        is_extend=lambda: False,
        is_extend_without_speculative=lambda: False,
    )
)


def _layer():
    layer = MQALayer.__new__(MQALayer)
    torch.nn.Module.__init__(layer)
    layer.__dict__.update(
        alt_streams=None,
        attn_mqa=SimpleNamespace(v_head_dim=_DIM),
        compress_ratio=4,
        n_local_heads=_LOCAL_HEADS,
        n_local_groups=_GROUPS,
        o_lora_rank=_RANK,
        head_dim=_DIM,
        qk_rope_head_dim=4,
        freqs_cis=None,
        wo_a_fp8=False,
        use_fused_wo_a=False,
        use_flashinfer_mxfp8_wo_b=False,
        use_npu_arch35_mxfp8_wo_a=False,
        _kernel_num_heads=lambda rows: _PADDED_HEADS,
        _local_attn_sink=lambda heads: torch.zeros(heads),
        _forward_prepare=mock.Mock(
            return_value=(torch.randn(_ROWS, _LOCAL_HEADS, _DIM), None)
        ),
        wo_a=SimpleNamespace(
            weight=torch.zeros(_GROUPS * _RANK, _LOCAL_HEADS * _DIM // _GROUPS)
        ),
        wq_b=mock.Mock(return_value=(torch.randn(_ROWS, _LOCAL_HEADS * _DIM), None)),
        wqkv_a=mock.Mock(return_value=("stock qkv_a", None)),
        _normalize_q_lora=mock.Mock(return_value=("stock normed", "stock quant")),
    )
    return layer


@contextlib.contextmanager
def _cuda_decode_step():
    """Yields (backend, stock wo_a GEMM) for a CUDA-like decode step on CPU."""
    backend = mock.Mock()
    backend.forward.side_effect = lambda q, **_: torch.randn(_ROWS, q.shape[1], _DIM)
    with contextlib.ExitStack() as stack:
        for name, value in {
            "_is_hip": False,
            "_is_npu": False,
            "_is_cuda": False,
            "_is_gfx95_supported": False,
            "_is_gfx942_supported": False,
            "_NPU_BF16_WO_A_GEMM": False,
        }.items():
            stack.enter_context(mock.patch.object(deepseek_v4, name, value))
        stack.enter_context(
            mock.patch.object(deepseek_v4, "get_attn_backend", return_value=backend)
        )
        stack.enter_context(mock.patch.object(deepseek_v4, "fused_rope_inplace"))
        for gate in ("is_unified_kv_triton", "is_unified_kv_fp8"):
            stack.enter_context(mock.patch(f"{_ENV_GATE}.{gate}", return_value=False))
        yield (
            backend,
            stack.enter_context(
                mock.patch.object(
                    deepseek_v4, "_apply_wo_a_bf16_matmul", return_value="stock wo_a"
                )
            ),
        )


class TestAttentionOpSlots(unittest.TestCase):
    def test_set_slots_replace_attention_and_wo_a_with_local_heads(self):
        """A caller's attention gets the unpadded query, and its wo_a replaces SGLang's."""
        layer = _layer()
        slots = MQAOpSlots(
            sparse_decode=mock.Mock(),
            unpadded_q=True,
            wo_a=mock.Mock(return_value="caller wo_a"),
        )
        positions = torch.arange(_ROWS)
        with _cuda_decode_step() as (backend, stock_wo_a):
            out = layer.forward_before_wo_b(
                torch.randn(_ROWS, 32), positions, _DECODE, op_slots=slots
            )
        self.assertEqual(out, "caller wo_a")
        stock_wo_a.assert_not_called()
        self.assertIs(layer._forward_prepare.call_args.kwargs["op_slots"], slots)
        kwargs = backend.forward.call_args.kwargs
        self.assertIs(kwargs["sparse_decode"], slots.sparse_decode)
        self.assertEqual(kwargs["q"].shape[1], _LOCAL_HEADS)
        self.assertEqual(kwargs["attn_sink"].shape[0], _LOCAL_HEADS)
        o, wo_a_positions = slots.wo_a.call_args.args
        self.assertEqual(tuple(o.shape), (_ROWS, _LOCAL_HEADS, _DIM))
        self.assertIs(wo_a_positions, positions)

    def test_unset_slots_pad_the_query_and_run_the_stock_wo_a(self):
        """Without slots, attention takes the padded query and SGLang's wo_a runs."""
        layer = _layer()
        with _cuda_decode_step() as (backend, stock_wo_a):
            out = layer.forward_before_wo_b(
                torch.randn(_ROWS, 32), torch.arange(_ROWS), _DECODE
            )
        self.assertEqual(out, "stock wo_a")
        self.assertIs(
            layer._forward_prepare.call_args.kwargs["op_slots"],
            deepseek_v4._NO_OP_SLOTS,
        )
        kwargs = backend.forward.call_args.kwargs
        self.assertNotIn("sparse_decode", kwargs)
        self.assertEqual(kwargs["q"].shape[1], _PADDED_HEADS)
        self.assertEqual(
            tuple(stock_wo_a.call_args.args[0].shape), (_ROWS, _GROUPS, 64)
        )

    def test_prepare_op_slots_replace_only_their_op(self):
        """Each prepare slot replaces one op; unset, SGLang's op gets the same input."""
        layer = _layer()
        x, q_lora, positions = torch.randn(_ROWS, 32), torch.randn(_ROWS, 6), None
        slots = MQAOpSlots(
            wqkv_a=mock.Mock(return_value="caller qkv_a"),
            q_norm_quant=mock.Mock(return_value=("caller normed", "caller quant")),
            q_b=mock.Mock(return_value="caller q"),
        )
        self.assertEqual(layer._compute_qkv_a(x, slots), "caller qkv_a")
        self.assertEqual(
            layer._normalize_q_lora_or_slot(q_lora, slots),
            ("caller normed", "caller quant"),
        )
        self.assertEqual(
            layer._compute_q_b(q_lora, positions, "q_out", slots), "caller q"
        )
        slots.q_b.assert_called_once_with(q_lora, positions, "q_out")
        layer.wqkv_a.assert_not_called()
        layer._normalize_q_lora.assert_not_called()
        layer.wq_b.assert_not_called()

        self.assertEqual(layer._compute_qkv_a(x), "stock qkv_a")
        layer.wqkv_a.assert_called_once_with(x)
        self.assertEqual(
            layer._normalize_q_lora_or_slot(q_lora, deepseek_v4._NO_OP_SLOTS),
            ("stock normed", "stock quant"),
        )
        layer._normalize_q_lora.assert_called_once_with(q_lora)
        layer.use_mega_attention = True  # the stock arm that returns wq_b's view
        q = layer._compute_q_b(q_lora, positions)
        layer.wq_b.assert_called_once_with(q_lora)
        self.assertEqual(tuple(q.shape), (_ROWS, _LOCAL_HEADS, _DIM))


if __name__ == "__main__":
    unittest.main()
