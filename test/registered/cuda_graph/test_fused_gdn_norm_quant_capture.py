"""CUDA-graph capture/replay test for fused_rmsnorm_gated_static_fp8_quant.

Captures the fused GDN gated-norm+quant kernel in a raw ``torch.cuda.CUDAGraph``,
then rewrites the static x/z buffers AND the device scale value, replays, and
compares against an eager run on the same data — proving the scale is read
through its device pointer each replay, never baked in at capture. Replays x100
to check the stability of the graph-pool output allocation.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

# M = 9 covers the ROWS_PER_BLOCK=4 remainder tile; N = qwen3.5 head_v_dim.
M, N = 9, 128
EPS = 1e-6


def _make_inputs(seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda", generator=gen)
    z = torch.randn(M, N, dtype=torch.bfloat16, device="cuda", generator=gen)
    scale = (
        torch.rand(1, dtype=torch.float32, device="cuda", generator=gen) * 0.05 + 1e-3
    )
    return x, z, scale


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFusedGdnNormQuantCapture(CustomTestCase):

    def _run_case(self, activation):
        from sglang.srt.layers.quantization.fp8_kernel import (
            fused_rmsnorm_gated_static_fp8_quant,
        )

        torch.manual_seed(0)
        weight = torch.rand(N, dtype=torch.bfloat16, device="cuda") * 2 - 1

        # Static IO buffers, rewritten in place between replays.
        static_x, static_z, static_scale = _make_inputs(seed=0)

        def _launch():
            return fused_rmsnorm_gated_static_fp8_quant(
                static_x, static_z, weight, EPS, static_scale, activation=activation
            )

        def _eager(x, z, scale):
            return fused_rmsnorm_gated_static_fp8_quant(
                x.clone(), z.clone(), weight, EPS, scale, activation=activation
            )

        # Eager warmup on a side stream (JIT-compiles the kernel; capture must
        # not be the first launch).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _launch()
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_q = _launch()

        def _check_replay(x, z, scale):
            ref_q = _eager(x, z, scale)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(out_q, ref_q), activation)

        # Fresh data + a NEW device scale value must flow through the replay:
        # the kernel reads the scale by pointer, never a baked-in constant.
        for seed in (1, 2):
            x, z, scale = _make_inputs(seed)
            static_x.copy_(x)
            static_z.copy_(z)
            static_scale.copy_(scale)
            g.replay()
            _check_replay(x, z, scale)

        # Replay x100 on fixed inputs: the graph-pool output allocation must
        # be stable.
        x, z, scale = _make_inputs(seed=3)
        static_x.copy_(x)
        static_z.copy_(z)
        static_scale.copy_(scale)
        for _ in range(100):
            g.replay()
        _check_replay(x, z, scale)

    def test_swish(self):
        self._run_case("swish")

    def test_sigmoid(self):
        self._run_case("sigmoid")


if __name__ == "__main__":
    unittest.main()
