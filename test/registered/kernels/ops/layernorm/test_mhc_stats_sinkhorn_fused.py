"""The one-launch prefill mHC triplet matches the two-launch bf16x3 path bitwise, including
partial row blocks and back-to-back launches that reuse the row-block semaphores."""

import unittest

import torch

from sglang.kernels.ops.layernorm.mhc import (
    hc_mix_stats_sinkhorn_bf16x3,
    split_bf16_hc_weight,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HC = 4
MIX = (2 + HC) * HC
ITERS, RMS_EPS, HC_EPS = 20, 1e-6, 1e-6


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "Blackwell only",
)
class TestFusedStatsSinkhorn(CustomTestCase):
    def test_matches_two_launch_path_bitwise(self):
        from sglang.kernels.ops.layernorm.mhc_stats_sinkhorn_fused import (
            hc_mix_stats_sinkhorn_bf16x3_fused,
        )

        g = torch.Generator(device="cpu").manual_seed(0)
        # 5000 leaves a partial last row block; repeats reuse the semaphores.
        for H, m in ((4096, 4096), (4096, 5000), (4096, 8192), (4096, 8192), (5120, 16384)):
            fn = (torch.randn(MIX, HC * H, generator=g) * 0.02).cuda()
            parts = split_bf16_hc_weight(fn)
            scale = torch.tensor([0.7, 1.3, 0.9]).cuda()
            base = (torch.randn(MIX, generator=g) * 0.3).cuda()
            x = (torch.randn(m, HC * H, generator=g) * 2).to(torch.bfloat16).cuda()
            args = (x, parts, scale, base, ITERS, RMS_EPS, HC_EPS)
            ref = hc_mix_stats_sinkhorn_bf16x3(*args)
            got = hc_mix_stats_sinkhorn_bf16x3_fused(*args)
            for name, a, b in zip(("pre", "post", "comb"), ref, got):
                diff = (a - b).abs().max().item()
                self.assertTrue(torch.equal(a, b), f"H={H} m={m} {name} max diff {diff}")


if __name__ == "__main__":
    unittest.main()
