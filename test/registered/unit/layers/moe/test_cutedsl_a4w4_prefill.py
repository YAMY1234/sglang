"""Repack of MXFP4 experts to the CuteDSL NVFP4 layout (SGLANG_K31_PREFILL_MOE_A4W4).

The repack must not change a value, and it must not leave memory behind: whatever it frees has to be releasable
before the KV pool is sized, so only the persistent NVFP4 tensors stay allocated afterwards.
"""

import gc
import unittest

import torch

from sglang.srt.layers.moe.moe_runner import cutedsl_a4w4_prefill as a4w4
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")


def _mxfp4(num_experts, rows, k, span, seed=0, device="cpu"):
    gen = torch.Generator(device=device).manual_seed(seed)
    w = torch.randint(
        0,
        256,
        (num_experts, rows, k // 2),
        dtype=torch.uint8,
        generator=gen,
        device=device,
    )
    base = torch.randint(110, 125, (num_experts, 1, 1), generator=gen, device=device)
    off = torch.randint(
        0, span + 1, (num_experts, rows, k // 32), generator=gen, device=device
    )
    return w, (base + off).to(torch.uint8)


class TestCuteDslA4W4Repack(CustomTestCase):
    def test_scales_keep_every_block_value(self):
        # Spans up to 17 binades fit E4M3's exact powers of two (2^-9 .. 2^8).
        w, s = _mxfp4(4, 128, 256, span=17)
        sf, g = a4w4._nvfp4_scales(w, s)
        nv = sf.view(torch.float8_e4m3fn).float() * torch.exp2(g.float())[:, None, None]
        mx = torch.exp2(s.float() - 127).repeat_interleave(2, dim=-1)
        self.assertTrue(torch.equal(nv, mx))

    def test_span_beyond_e4m3_raises_only_for_nonzero_blocks(self):
        w, s = _mxfp4(1, 128, 256, span=0)
        s[0, 0, 0] = s[0, 0, 0] - 30  # 30 binades below the expert's max
        w[0, 0, :16] = 0x80  # that block holds only +-0: still exact
        a4w4._nvfp4_scales(w, s)
        w[0, 0, 0] = 0x11
        with self.assertRaises(ValueError):
            a4w4._nvfp4_scales(w, s)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_repack_moves_rows_and_frees_its_temporaries(self):
        e, hidden, inter = 192, 1024, 512
        dev = torch.device("cuda")
        w13, s13 = _mxfp4(e, 2 * inter, hidden, span=11, seed=1, device=dev)
        w2, s2 = _mxfp4(e, hidden, inter, span=3, seed=2, device=dev)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()

        out = a4w4.prepare_weights(w13, s13, w2, s2, beta=4.0, lbeta=25.0)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base

        rows = a4w4._up_gate_interleave_rows(inter, dev)
        self.assertTrue(torch.equal(out.w13, w13.index_select(1, rows)))
        self.assertEqual(out.w2.data_ptr(), w2.data_ptr())
        del rows

        # As after the layer's params are rebound: only the new persistent tensors remain.
        input_bytes = sum(t.numel() for t in (w13, s13, s2))
        del w13, s13, s2
        gc.collect()
        torch.cuda.empty_cache()
        new_bytes = sum(
            t.numel() * t.element_size()
            for t in (
                out.w13,
                out.w13_sf,
                out.w2_sf,
                out.w1_alpha,
                out.w2_alpha,
                out.unit_scale,
            )
        )
        slack = 64 * 512  # the caching allocator rounds each block up to 512 bytes
        self.assertLessEqual(
            abs(torch.cuda.memory_allocated() - (base - input_bytes + new_bytes)), slack
        )
        self.assertLessEqual(
            torch.cuda.memory_reserved() - torch.cuda.memory_allocated(), 2 << 20
        )
        # Chunked: the transient footprint is a few chunks of scales, not several full scale-sized copies.
        chunk_scale = a4w4._REPACK_EXPERT_CHUNK * 2 * inter * (hidden // 16) * 2
        self.assertLessEqual(peak, new_bytes + 8 * chunk_scale)


if __name__ == "__main__":
    unittest.main()
