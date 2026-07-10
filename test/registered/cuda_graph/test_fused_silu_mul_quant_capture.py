"""CUDA-graph capture/replay test for fused_silu_mul_static_fp8_quant.

Captures the fused kernel in a raw ``torch.cuda.CUDAGraph``, then rewrites the
static input buffer AND the device scale value, replays, and compares against
an eager run on the same data — proving no capture-time constant snuck in
(precedent hazard: ``torch.empty`` outputs captured with stale data, cf. the
fused-QK IMA bug class). Replays x100 to check the stability of the graph-pool
output allocation.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

M = 8


def _make_inputs(intermediate, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        M, 2 * intermediate, dtype=torch.bfloat16, device="cuda", generator=gen
    )
    scale = (
        torch.rand(1, dtype=torch.float32, device="cuda", generator=gen) * 0.05 + 1e-3
    )
    return x, scale


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFusedSiluMulQuantCapture(CustomTestCase):

    def _run_case(self, intermediate):
        from sglang.srt.layers.quantization.fp8_kernel import (
            fused_silu_mul_static_fp8_quant,
        )

        msg = f"intermediate={intermediate}"

        # Static IO buffers, rewritten in place between replays.
        static_x, static_scale = _make_inputs(intermediate, seed=0)

        def _launch():
            return fused_silu_mul_static_fp8_quant(static_x, static_scale)

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

        def _check_replay(x, scale):
            ref_q = fused_silu_mul_static_fp8_quant(x, scale)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(out_q, ref_q), msg)

        # Fresh data + a NEW device scale value must flow through the replay:
        # the kernel reads the scale by pointer, never a baked-in constant.
        for seed in (1, 2):
            x, scale = _make_inputs(intermediate, seed)
            static_x.copy_(x)
            static_scale.copy_(scale)
            g.replay()
            _check_replay(x, scale)

        # Replay x100 on fixed inputs: the graph-pool output allocation must
        # be stable.
        x, scale = _make_inputs(intermediate, seed=3)
        static_x.copy_(x)
        static_scale.copy_(scale)
        for _ in range(100):
            g.replay()
        _check_replay(x, scale)

    def test_single_tile(self):
        self._run_case(intermediate=1408)

    def test_multi_tile(self):
        # > _FUSED_SILU_QUANT_MAX_SINGLE_TILE_N: exercises the loop path.
        self._run_case(intermediate=16384)


if __name__ == "__main__":
    unittest.main()
