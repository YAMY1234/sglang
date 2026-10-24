import unittest

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    silu_and_mul_static_quant_fp8,
    static_quant_fp8,
)
from sglang.srt.layers.activation import silu_and_mul
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.fp8 import TestFP8Base

register_cuda_ci(est_time=4, stage="base-b", runner_config="1-gpu-large")


_is_cuda = is_cuda()


class TestPerTokenGroupQuantFP8(TestFP8Base):
    def test_per_token_group_quant_fp8(self):
        if _is_cuda and torch.cuda.get_device_capability()[0] < 9:
            return

        A, A_quant_gt, scale_gt = self._make_A(
            M=self.M, K=self.K, group_size=self.group_size, out_dtype=self.quant_type
        )
        A_quant, scale = per_token_group_quant_fp8(
            x=A.to(torch.bfloat16), group_size=self.group_size
        )
        torch.testing.assert_close(scale, scale_gt)
        diff = (A_quant.to(torch.float16) - A_quant_gt.to(torch.float16)).abs()
        diff_count = (diff > 1e-5).count_nonzero()
        assert diff_count / diff.numel() < 1e-4


class TestSiluAndMulStaticQuantFP8(unittest.TestCase):
    """The one-launch SiLU-and-mul + static FP8 quant must equal silu_and_mul then
    static_quant_fp8 bit for bit, eager and replayed in a CUDA graph."""

    def test_matches_the_two_launches(self):
        if not _is_cuda or torch.cuda.get_device_capability()[0] < 9:
            return
        g = torch.Generator(device="cuda").manual_seed(0)
        for rows, d2, std, scale in (
            (1, 256, 1.0, 0.01),
            (28, 256, 3.0, 0.02),
            (896, 256, 8.0, 0.05),
            (83, 1024, 0.3, 0.001),
        ):
            x = (torch.randn(rows, d2, device="cuda", generator=g) * std).to(
                torch.bfloat16
            )
            s = torch.tensor([scale], device="cuda")
            want, _ = static_quant_fp8(silu_and_mul(x), s)
            got = silu_and_mul_static_quant_fp8(x, s)
            self.assertTrue(torch.equal(got.view(torch.uint8), want.view(torch.uint8)))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = silu_and_mul_static_quant_fp8(x, s)
            x.copy_((torch.randn(rows, d2, device="cuda", generator=g) * std))
            graph.replay()
            want, _ = static_quant_fp8(silu_and_mul(x), s)
            self.assertTrue(torch.equal(out.view(torch.uint8), want.view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()
