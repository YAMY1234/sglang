"""CUDA-graph capture/replay test for fused_gemma_rmsnorm_static_fp8_quant.

Captures the fused kernel in a raw ``torch.cuda.CUDAGraph``, then rewrites the
static input/residual buffers AND the device scale value, replays, and compares
against an eager run on the same data — proving no capture-time constant snuck
in (precedent hazard: ``torch.empty`` outputs captured with stale data, cf. the
fused-QK IMA bug class). Replays x100 to check the stability of the graph-pool
output allocations.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

M, N = 8, 2048
EPS = 1e-6


def _make_inputs(seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda", generator=gen)
    res = torch.randn(M, N, dtype=torch.bfloat16, device="cuda", generator=gen)
    scale = (
        torch.rand(1, dtype=torch.float32, device="cuda", generator=gen) * 0.05 + 1e-3
    )
    return x, res, scale


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFusedRmsNormQuantCapture(CustomTestCase):

    def _run_case(self, has_res, bf16_out):
        from sglang.srt.layers.quantization.fp8_kernel import (
            fused_gemma_rmsnorm_static_fp8_quant,
        )

        msg = f"has_res={has_res} bf16_out={bf16_out}"
        torch.manual_seed(0)
        weight = torch.rand(N, dtype=torch.bfloat16, device="cuda") * 2 - 1

        # Static IO buffers, rewritten in place between replays.
        static_x, static_res, static_scale = _make_inputs(seed=0)

        def _launch():
            return fused_gemma_rmsnorm_static_fp8_quant(
                static_x,
                weight,
                EPS,
                static_scale,
                residual=static_res if has_res else None,
                bf16_out=bf16_out,
            )

        def _eager(x, res, scale):
            res_c = res.clone() if has_res else None
            q, y = fused_gemma_rmsnorm_static_fp8_quant(
                x.clone(), weight, EPS, scale, residual=res_c, bf16_out=bf16_out
            )
            return q, y, res_c

        # Eager warmup on a side stream (JIT-compiles the kernel; capture must
        # not be the first launch).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _launch()
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_q, out_bf16 = _launch()

        def _check_replay(x, res, scale):
            from sglang.srt.layers.quantization.fp8_kernel import static_quant_fp8

            ref_q, ref_y, ref_res = _eager(x, res, scale)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(out_q, ref_q), msg)
            if bf16_out:
                self.assertTrue(torch.equal(out_bf16, ref_y), msg)
                # Dual-output invariant (Phase 2): the fp8 co-output is the
                # static quant of the bf16 co-output, bitwise, on every replay.
                inv_q, _ = static_quant_fp8(out_bf16, static_scale, repeat_scale=False)
                self.assertTrue(
                    torch.equal(out_q.view(torch.uint8), inv_q.view(torch.uint8)), msg
                )
            if has_res:
                self.assertTrue(torch.equal(static_res, ref_res), msg)

        # Fresh data + a NEW device scale value must flow through the replay:
        # the kernel reads the scale by pointer, never a baked-in constant.
        for seed in (1, 2):
            x, res, scale = _make_inputs(seed)
            static_x.copy_(x)
            static_res.copy_(res)
            static_scale.copy_(scale)
            g.replay()
            _check_replay(x, res, scale)

        # Replay x100 on fixed inputs: graph-pool output allocations must be
        # stable (the residual is refilled each time, as replay updates it in
        # place).
        x, res, scale = _make_inputs(seed=3)
        static_scale.copy_(scale)
        for _ in range(100):
            static_x.copy_(x)
            static_res.copy_(res)
            g.replay()
        _check_replay(x, res, scale)

    def test_norm_only(self):
        self._run_case(has_res=False, bf16_out=False)

    def test_fused_add(self):
        self._run_case(has_res=True, bf16_out=False)

    def test_fused_add_bf16_out(self):
        self._run_case(has_res=True, bf16_out=True)

    def test_norm_only_bf16_out(self):
        self._run_case(has_res=False, bf16_out=True)


if __name__ == "__main__":
    unittest.main()
