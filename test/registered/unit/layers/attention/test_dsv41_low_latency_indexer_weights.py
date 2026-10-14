"""DSV4.1 indexer weights_proj on ll_bf16 (SGLANG_OPT_DSV41_LOW_LATENCY_INDEXER_WEIGHTS).

CPU: the 1-8-row routing and the flag-off path. GPU (SM90+ with CuTe DSL): the kernel
matches the FP32 reference at the served shape for 1-8 rows, as in muLLM b5a4d16f
tests/models/test_deepseek_v41_low_latency_aux_projections.py::test_kernel_parity.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.gemm import ll_bf16
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4 import dsv41_sparse
from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Indexer
from sglang.test.test_utils import CustomTestCase

HIDDEN = 5120
INDEX_HEADS = 32


def _indexer(hidden: int = 64) -> DeepseekV41Indexer:
    config = SimpleNamespace(
        index_n_heads=INDEX_HEADS,
        index_head_dim=128,
        qk_rope_head_dim=64,
        index_topk=512,
        kv_source_layer_ids=[],
        candidate_source_layer_id=-1,
        candidate_topk_blocks=0,
        candidate_block_size=0,
        q_lora_rank=32,
        hidden_size=hidden,
        rms_norm_eps=1e-6,
    )
    return DeepseekV41Indexer(
        config, layer_id=2, head_dim=512, quant_config=None, prefix="indexer"
    )


class TestIndexerWeightsRouting(CustomTestCase):
    def test_flag_off_keeps_the_stock_gemm(self):
        with envs.SGLANG_OPT_DSV41_LOW_LATENCY_INDEXER_WEIGHTS.override(False):
            indexer = _indexer()
        self.assertEqual(indexer.weights_proj_low_latency_max_m, 0)
        x = torch.randn(6, 64, dtype=torch.bfloat16)
        with patch.object(ll_bf16, "ll_bf16_gemm") as ll:
            w = indexer.head_weights_raw(x)
        ll.assert_not_called()
        self.assertEqual((w.dtype, w.shape), (torch.bfloat16, (6, INDEX_HEADS)))

    def test_flag_on_needs_the_kernel(self):
        with (
            envs.SGLANG_OPT_DSV41_LOW_LATENCY_INDEXER_WEIGHTS.override(True),
            patch.object(ll_bf16, "is_available", return_value=False),
        ):
            self.assertEqual(_indexer().weights_proj_low_latency_max_m, 0)
        with (
            envs.SGLANG_OPT_DSV41_LOW_LATENCY_INDEXER_WEIGHTS.override(True),
            patch.object(ll_bf16, "is_available", return_value=True),
        ):
            self.assertEqual(
                _indexer().weights_proj_low_latency_max_m,
                dsv41_sparse._LOW_LATENCY_INDEXER_MAX_ROWS,
            )

    def test_switches_at_1_to_8_contiguous_bf16_rows(self):
        indexer = _indexer()
        indexer.weights_proj_low_latency_max_m = 8

        def fake(x, w):
            return torch.full((x.shape[0], w.shape[0]), 2.5, dtype=torch.float32)

        with patch.object(ll_bf16, "ll_bf16_gemm", side_effect=fake) as ll:
            w = indexer.head_weights_raw(torch.zeros(8, 64, dtype=torch.bfloat16))
            # fp32 accumulation comes back rounded to the layer's bf16 output
            self.assertEqual(w.dtype, torch.bfloat16)
            self.assertTrue(torch.all(w == 2.5))
            indexer.head_weights_raw(torch.zeros(9, 64, dtype=torch.bfloat16))
            indexer.head_weights_raw(torch.zeros(64, 6, dtype=torch.bfloat16).T)
            indexer.head_weights_raw(torch.zeros(0, 64, dtype=torch.bfloat16))
        self.assertEqual(ll.call_count, 1)

    def test_narrow_output_dispatches_to_the_dot_product(self):
        for m in range(1, 9):
            self.assertEqual(
                ll_bf16.dispatch(M=m, K=HIDDEN, N=INDEX_HEADS).backend, "dotprod"
            )


def _gpu_ok() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (9, 0)
        and ll_bf16.is_available()
    )


@unittest.skipUnless(_gpu_ok(), "needs an SM90+ GPU with CuTe DSL")
class TestIndexerWeightsGpuParity(CustomTestCase):
    def test_kernel_parity(self):
        for m in range(1, 9):
            torch.manual_seed(INDEX_HEADS + m)
            x = torch.randn(m, HIDDEN, device="cuda").bfloat16()
            w = (torch.randn(INDEX_HEADS, HIDDEN, device="cuda") / HIDDEN**0.5).bfloat16()
            out = ll_bf16.ll_bf16_gemm(x, w)
            self.assertEqual((out.dtype, out.shape), (torch.float32, (m, INDEX_HEADS)))
            ref = x.float() @ w.float().T
            torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
            cublas = torch.mm(x, w.T, out_dtype=torch.float32)
            torch.testing.assert_close(out, cublas, atol=1e-4, rtol=1e-4)

    def test_head_weights_match_the_stock_gemm(self):
        with envs.SGLANG_OPT_DSV41_LOW_LATENCY_INDEXER_WEIGHTS.override(True):
            indexer = _indexer(HIDDEN).cuda()
        self.assertEqual(indexer.weights_proj_low_latency_max_m, 8)
        torch.nn.init.normal_(indexer.weights_proj.weight, std=HIDDEN**-0.5)
        for m in (1, 6, 8):
            x = torch.randn(m, HIDDEN, device="cuda").bfloat16()
            got = indexer.head_weights_raw(x)
            stock = (x.float() @ indexer.weights_proj.weight.float().T).bfloat16()
            # Only the K summation order differs: one bf16 ulp at most.
            torch.testing.assert_close(got, stock, atol=1e-2, rtol=8e-3)


if __name__ == "__main__":
    unittest.main()
