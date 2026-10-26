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
    def test_matches_two_launches_and_cuda_graph_replay(self):
        if not _is_cuda or torch.cuda.get_device_capability()[0] < 9:
            return

        generator = torch.Generator(device="cuda").manual_seed(0)
        for rows, width, std, scale in (
            (1, 256, 1.0, 0.01),
            (28, 256, 3.0, 0.02),
            (896, 256, 8.0, 0.05),
            (83, 1024, 0.3, 0.001),
        ):
            with self.subTest(rows=rows, width=width):
                x = (
                    torch.randn(rows, width, device="cuda", generator=generator) * std
                ).to(torch.bfloat16)
                input_scale = torch.tensor([scale], device="cuda")
                expected, _ = static_quant_fp8(silu_and_mul(x), input_scale)
                actual = silu_and_mul_static_quant_fp8(x, input_scale)
                self.assertTrue(
                    torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
                )

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replayed = silu_and_mul_static_quant_fp8(x, input_scale)
                x.copy_(
                    torch.randn(rows, width, device="cuda", generator=generator) * std
                )
                graph.replay()
                expected, _ = static_quant_fp8(silu_and_mul(x), input_scale)
                self.assertTrue(
                    torch.equal(replayed.view(torch.uint8), expected.view(torch.uint8))
                )

    def test_flashinfer_bmm_takes_prequantized_rows(self):
        if not _is_cuda or torch.cuda.get_device_capability()[0] < 9:
            return

        from sglang.srt.layers.quantization.fp8_utils import (
            apply_fp8_linear_bmm_flashinfer,
        )

        generator = torch.Generator(device="cuda").manual_seed(1)
        rows, intermediate_size, hidden_size = 224, 128, 4096
        x = (
            torch.randn(rows, 2 * intermediate_size, device="cuda", generator=generator)
            * 2
        ).to(torch.bfloat16)
        weight = (
            torch.randn(
                hidden_size, intermediate_size, device="cuda", generator=generator
            )
            * 0.05
        ).to(torch.float8_e4m3fn)
        weight_scale = torch.tensor([0.01], device="cuda")
        input_scale = torch.tensor([0.02], device="cuda")
        expected = apply_fp8_linear_bmm_flashinfer(
            silu_and_mul(x), weight.t(), weight_scale, input_scale
        )
        actual = apply_fp8_linear_bmm_flashinfer(
            silu_and_mul_static_quant_fp8(x, input_scale),
            weight.t(),
            weight_scale,
            input_scale,
        )
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(
            torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        )


if __name__ == "__main__":
    unittest.main()
