"""DSV4.1 low-latency compressor projections: ll_bf16 dispatch and the 1-8-row routing."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.gemm import ll_bf16
from sglang.kernels.ops.gemm.ll_bf16 import CompileKey, dispatch
from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Compressor
from sglang.test.test_utils import CustomTestCase


class TestLlBf16Dispatch(CustomTestCase):
    def test_tuned_compressor_shapes(self):
        # GB300 configs measured by the source branch at the DSpark verify size.
        self.assertEqual(
            dispatch(M=6, K=5120, N=1024),
            CompileKey(backend="splitk", split_k=4, num_stages=4),
        )
        self.assertEqual(
            dispatch(M=6, K=5120, N=512), CompileKey(backend="dotprod", M=6, K=5120, bs=64)
        )
        self.assertEqual(
            dispatch(M=8, K=5120, N=512),
            CompileKey(backend="splitk", split_k=4, num_stages=6),
        )

    def test_untuned_shapes_use_the_default_dispatch(self):
        dotprod = CompileKey(backend="dotprod", M=3, K=5120, bs=128)
        self.assertEqual(dispatch(M=3, K=5120, N=1024), dotprod)
        # Narrow N would leave split-K with too few CTAs.
        self.assertEqual(dispatch(M=7, K=5120, N=32).backend, "dotprod")
        self.assertEqual(
            dispatch(M=7, K=5120, N=512),
            CompileKey(backend="splitk", split_k=6, num_stages=4),
        )


class TestLlBf16Operands(CustomTestCase):
    def test_trainable_weight_reaches_the_kernel_detached(self):
        """An nn.Linear weight requires grad; DLPack export of it fails at graph capture."""
        seen = []
        w = torch.nn.Parameter(torch.zeros(1024, 5120, dtype=torch.bfloat16))
        x = torch.zeros(6, 5120, dtype=torch.bfloat16)
        with (
            patch.object(ll_bf16, "compile_kernel", return_value=lambda *a: seen.extend(a)),
            patch.object(ll_bf16, "_stream", return_value=None),
        ):
            ll_bf16.ll_bf16_gemm(x, w)
        tensors = [a for a in seen if isinstance(a, torch.Tensor)]
        self.assertEqual(len(tensors), 3)
        self.assertFalse(any(t.requires_grad for t in tensors))


class TestCompressorRouting(CustomTestCase):
    def _compressor(self, ratio, max_rows):
        c = DeepseekV41Compressor(64, 32, ratio, 1e-6, fused_compress=True)
        c.low_latency_max_rows = max_rows
        return c

    def test_ratio2_fp32_projection_switches_at_1_to_8_rows(self):
        c = self._compressor(2, 8)
        with (
            patch.object(ll_bf16, "ll_bf16_gemm", return_value="ll") as ll,
            patch(
                "sglang.srt.layers.attention.dsv4.dsv41_sparse.linear_bf16_fp32",
                return_value="cublas",
            ) as cublas,
        ):
            self.assertEqual(c.project_fused(torch.zeros(8, 64, dtype=torch.bfloat16)), "ll")
            self.assertEqual(
                c.project_fused(torch.zeros(9, 64, dtype=torch.bfloat16)), "cublas"
            )
            self.assertEqual(
                c.project_fused(torch.zeros(64, 6, dtype=torch.bfloat16).T), "cublas"
            )
        self.assertEqual(ll.call_count, 1)
        self.assertEqual(cublas.call_count, 2)

    def test_ratio1_keeps_bf16_and_flag_off_keeps_cublas(self):
        x = torch.zeros(6, 64, dtype=torch.bfloat16)
        with patch(
            "sglang.srt.layers.low_latency_lm_head.cutedsl_low_m_mm", return_value="ll"
        ):
            self.assertEqual(self._compressor(1, 8).project(x), ("ll", None))
            kv, _ = self._compressor(1, 0).project(x)
        self.assertEqual((kv.dtype, kv.shape), (torch.bfloat16, (6, 32)))


if __name__ == "__main__":
    unittest.main()
