"""The NVFP4 HiCache host codec probes its Triton kernels once per row width and
falls back, with a warning, when they fail or differ from the torch twin.

CPU-only: the Triton kernels are patched, and on CPU the fallback is the twin.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.pool_host import nvfp4_host
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.quant_ref_utils import break_fp4_bytes, nvfp4_kv_quantize
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_WARNING = "HiCache NVFP4 host codec fallback: torch, triton fused kernels failed: "


class TestNVFP4CodecFallback(CustomTestCase):
    def setUp(self):
        nvfp4_host._codec_fallback.cache_clear()
        self.addCleanup(nvfp4_host._codec_fallback.cache_clear)

    def test_failing_triton_kernels_select_the_torch_twin_once_per_width(self):
        calls = []

        def broken(*args, **kwargs):
            calls.append(kwargs["dim"])
            raise RuntimeError("no triton")

        with (
            patch.object(nvfp4_host, "encode_fp8_rows_to_nvfp4", broken),
            self.assertLogs(nvfp4_host.logger, level="WARNING") as logs,
        ):
            codecs = [nvfp4_host.nvfp4_row_codec(dim, "cpu") for dim in (32, 32, 48)]
        self.assertEqual(calls, [32, 48])
        self.assertEqual([codec.backend for codec in codecs], ["torch"] * 3)
        self.assertEqual(
            [record.getMessage() for record in logs.records],
            [_WARNING + "RuntimeError('no triton')"] * 2,
        )

        # The twin still encodes FlashInfer's layout and round-trips within NVFP4 error.
        codec, dim = codecs[0], 32
        x = (
            torch.randn(8, dim)
            * torch.tensor([0.01, 1.0, 30.0, 400.0]).repeat(2)[:, None]
        )
        rows = x.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        idx = torch.arange(rows.shape[0])
        enc = torch.zeros((1, rows.shape[0], codec.row_bytes), dtype=torch.uint8)
        codec.encode_rows(rows, idx, enc, page_size=rows.shape[0])
        dec = torch.zeros_like(rows)
        codec.decode_rows(enc[0], idx, dec)

        values = rows.view(torch.float8_e4m3fn).float()
        ref_fp4, ref_sf = nvfp4_kv_quantize(values, torch.tensor([1.0 / 6.0]))
        self.assertTrue(torch.equal(enc[0, :, dim // 2 :], ref_sf))
        self.assertTrue(
            torch.equal(
                break_fp4_bytes(enc[0, :, : dim // 2]), break_fp4_bytes(ref_fp4)
            )
        )
        blocks = values.view(-1, dim // 16, 16)
        amax = blocks.abs().amax(-1, keepdim=True)
        err = (dec.view(torch.float8_e4m3fn).float().view_as(blocks) - blocks).abs()
        self.assertTrue((err <= amax * 0.25 + 2**-9).all())

    def test_kernels_that_differ_from_the_twin_are_not_used(self):
        def write_nothing(*args, **kwargs):
            pass

        with (
            patch.object(nvfp4_host, "encode_fp8_rows_to_nvfp4", write_nothing),
            patch.object(nvfp4_host, "decode_nvfp4_rows_to_fp8", write_nothing),
            self.assertLogs(nvfp4_host.logger, level="WARNING") as logs,
        ):
            codec = nvfp4_host.nvfp4_row_codec(32, "cpu")
        self.assertEqual(codec.backend, "torch")
        (message,) = [record.getMessage() for record in logs.records]
        self.assertTrue(message.startswith(_WARNING + "differ from the torch twin"))


if __name__ == "__main__":
    unittest.main()
