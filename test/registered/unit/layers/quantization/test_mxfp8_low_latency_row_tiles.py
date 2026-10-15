"""Low-latency MXFP8 projections past 8 rows (SGLANG_OPT_DSV41_LOW_LATENCY_MXFP8_MAX_ROWS).

FlashInfer's cutedsl_low_latency GEMM runs one CTA per 8-row tile and reads each tile's
activation scales from a 128-row atom of its own; with mxfp8_quantize's plain swizzled
scales its output past 8 rows is garbage. CPU: the flag's range and which inputs take the
row-tiled path. GPU (SM100 family): the row-tile quantizer is mxfp8_quantize re-laid out,
and the row-tiled GEMM matches cute-dsl mm_mxfp8 at 9-48 rows (eager, graph, autotuned)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import fp8_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _max_rows(value):
    fp8_utils.mxfp8_low_latency_max_rows.cache_clear()
    with envs.SGLANG_OPT_DSV41_LOW_LATENCY_MXFP8_MAX_ROWS.override(value):
        return fp8_utils.mxfp8_low_latency_max_rows()


class TestRowBound(CustomTestCase):
    def tearDown(self):
        fp8_utils.mxfp8_low_latency_max_rows.cache_clear()

    def test_range(self):
        self.assertEqual(_max_rows(8), 8)
        self.assertEqual(_max_rows(48), 48)
        for bad in (7, 65):
            with self.assertRaises(ValueError):
                _max_rows(bad)

    def test_only_bf16_rows_past_eight_take_the_row_tiled_path(self):
        layer = SimpleNamespace(weight=torch.zeros(1792, 5120), _mxfp8_low_latency_logged=True)
        scale = torch.zeros(10)
        bf16 = lambda rows: torch.zeros(rows, 5120, dtype=torch.bfloat16)  # noqa: E731
        # Pre-quantized rows carry plain swizzled scales: the kernel reads them only to 8 rows.
        quantized = lambda rows: (torch.zeros(rows, 5120, dtype=torch.float8_e4m3fn), scale)  # noqa: E731
        cases = [
            (8, bf16(8), "ll"),
            (8, bf16(9), None),
            (48, bf16(8), "ll"),
            (48, bf16(12), "wide"),
            (48, bf16(49), None),
            (48, quantized(8), "ll"),
            (48, quantized(12), None),
        ]
        with (
            patch.object(fp8_utils, "mxfp8_low_latency_supported", return_value=True),
            patch.object(fp8_utils, "_mxfp8_low_latency_wide_linear", return_value="wide"),
            patch.object(fp8_utils, "flashinfer_mxfp8_blockscaled_linear", return_value="ll", create=True),
        ):
            for bound, x, want in cases:
                fp8_utils.mxfp8_low_latency_max_rows.cache_clear()
                with envs.SGLANG_OPT_DSV41_LOW_LATENCY_MXFP8_MAX_ROWS.override(bound):
                    got = fp8_utils.maybe_mxfp8_low_latency_linear(layer, x, scale, None)
                self.assertEqual(got, want, (bound, x))


def _gpu_ready() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
        and fp8_utils.mxfp8_low_latency_supported()
    )


@unittest.skipUnless(_gpu_ready(), "needs an SM100-family GPU with FlashInfer cute-dsl")
class TestRowTiles(CustomTestCase):
    SHAPES = ((1792, 5120), (8192, 1280), (5120, 2048))  # TP4 wqkv_a, wq_b, wo_b

    def test_quantizer_is_mxfp8_quantize_in_per_tile_atoms(self):
        from flashinfer import mxfp8_quantize

        from sglang.kernels.ops.layernorm.mxfp8_epilogue import mxfp8_quantize_row_tiles

        g = torch.Generator(device="cuda").manual_seed(0)
        for k in (5120, 1280, 2048):
            for rows in (9, 12, 16, 33, 48):
                x = (torch.randn(rows, k, device="cuda", generator=g) * 3).to(torch.bfloat16)
                x[rows // 2, :64] = 0  # an all-zero group takes scale byte 0
                q, s = mxfp8_quantize_row_tiles(x)
                want_q, want_s = mxfp8_quantize(x, is_sf_swizzled_layout=False)
                want_s = want_s.view(torch.uint8).view(rows, k // 32)
                self.assertTrue(torch.equal(q.view(torch.uint8), want_q.view(torch.uint8)))
                r = torch.arange(rows, device="cuda")[:, None]
                grp = torch.arange(k // 32, device="cuda")[None, :]
                off = ((r // 8) * (k // 128) + grp // 4) * 512 + (r % 8) * 16 + grp % 4
                self.assertTrue(torch.equal(s[off], want_s))

    def test_matches_cute_dsl_9_to_48_rows(self):
        from flashinfer import mm_mxfp8, mxfp8_quantize
        from flashinfer.autotuner import autotune

        g = torch.Generator(device="cuda").manual_seed(1)
        for n, k in self.SHAPES:
            w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
            w_q, w_s = mxfp8_quantize(w, is_sf_swizzled_layout=True)
            for rows in (9, 12, 16, 24, 40, 48):
                x = torch.randn(rows, k, device="cuda", generator=g).to(torch.bfloat16)
                x_q, x_s = mxfp8_quantize(x, is_sf_swizzled_layout=True)
                want = mm_mxfp8(x_q, w_q.t(), x_s, w_s, out_dtype=torch.bfloat16, backend="cute-dsl").float()
                tol = dict(rtol=2**-7, atol=2**-7 * want.abs().max().item())
                torch.testing.assert_close(
                    fp8_utils._mxfp8_low_latency_wide_linear(x, w_q, w_s, None).float(), want, **tol
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = fp8_utils._mxfp8_low_latency_wide_linear(x, w_q, w_s, None)
                out.zero_()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(out.float(), want, **tol)
            # One tuning pass over the row buckets: its scale buffers must fit the per-tile layout.
            with autotune(True):
                tuned = fp8_utils._mxfp8_low_latency_wide_linear(x, w_q, w_s, None)
            torch.testing.assert_close(tuned.float(), want, **tol)


if __name__ == "__main__":
    unittest.main()
