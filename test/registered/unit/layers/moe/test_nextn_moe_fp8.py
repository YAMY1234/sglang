"""SGLANG_NEXTN_MOE_FP8: the draft MoE's FP8 copy (quantizers, switch, routing hand-off); CPU only."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import unittest
from unittest import mock

import torch

from sglang.srt.layers.moe import nextn_moe_fp8 as m
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import BypassedTopKOutput
from sglang.test.test_utils import CustomTestCase


def _weights(e=2, n=256, k=384, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(e, n, k, generator=g) * 0.02
    w[0, :128, :128] *= 50  # one block with a much larger range
    w[1, 128:, 256:] = 0  # an all-zero block
    return w.to(torch.bfloat16)


class TestMode(CustomTestCase):
    def test_values(self):
        for v, want in (
            ("", None),
            ("0", None),
            ("off", None),
            ("1", "mxfp8"),
            ("mxfp8", "mxfp8"),
            ("block", "block"),
            ("FP8", "block"),
        ):
            with mock.patch.dict(os.environ, {m.ENV: v}):
                self.assertEqual(m.nextn_moe_fp8_mode(), want, v)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(m.nextn_moe_fp8_mode())
        with mock.patch.dict(os.environ, {m.ENV: "nvfp4"}):
            with self.assertRaises(ValueError):
                m.nextn_moe_fp8_mode()

    def test_defer_finalize_default_off(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(m.defer_finalize_requested())
        for v, want in (("0", False), ("", False), ("1", True), ("on", True)):
            with mock.patch.dict(os.environ, {m.DEFER_ENV: v}):
                self.assertEqual(m.defer_finalize_requested(), want, v)


class TestBlockFp8(CustomTestCase):
    def test_scales_and_error(self):
        w = _weights()
        q, s = m.quantize_block_fp8(w)
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(s.shape), (2, 2, 3))
        x = w.float().view(2, 2, 128, 3, 128)
        amax = x.abs().amax(dim=(2, 4))
        # amax / 448 per 128 x 128 block (an all-zero block keeps scale 1 and codes 0)
        want = torch.where(amax > 0, amax / 448, torch.ones_like(amax))
        torch.testing.assert_close(s, want, rtol=0, atol=0)
        qf = q.float().view(2, 2, 128, 3, 128)
        nz = amax > 0
        self.assertTrue(torch.all(qf.abs().amax(dim=(2, 4))[nz] == 448))
        self.assertTrue(torch.all(qf[~nz[:, :, None, :, None].expand_as(qf)] == 0))
        deq = (qf * s[:, :, None, :, None]).view_as(w)
        # E4M3: 3 mantissa bits, so at most 1/16 of the block's amax per element
        err = (deq - w.float()).abs().view(2, 2, 128, 3, 128).amax(dim=(2, 4))
        self.assertTrue(torch.all(err <= amax / 16 + 1e-12))


class TestMxfp8(CustomTestCase):
    def test_scales_and_error(self):
        w = _weights()
        q, s = m.quantize_mxfp8(w)
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(s.dtype, torch.uint8)
        self.assertEqual(tuple(s.shape), (2, 256, 384 // 32))
        x = w.float().view(2, 256, 12, 32)
        amax = x.abs().amax(-1)
        scale = torch.exp2(s.float() - 127)
        nz = amax > 0
        # the smallest power of two that keeps amax / scale <= 448
        self.assertTrue(torch.all(amax[nz] / scale[nz] <= 448))
        self.assertTrue(torch.all(amax[nz] / scale[nz] > 224))
        self.assertTrue(torch.all(s[~nz] == 0))
        deq = q.float().view(2, 256, 12, 32) * scale[..., None]
        err = (deq - x).abs().amax(-1)
        self.assertTrue(torch.all(err <= amax / 16 + 1e-12))


class _Runner:
    def __init__(self):
        self.calls = []

    def run(self, dispatch_output, quant_info):
        self.calls.append((dispatch_output, quant_info))
        return "out"


class TestRun(CustomTestCase):
    def _dispatch(self, non_padded):
        topk = BypassedTopKOutput(
            hidden_states=torch.zeros(4, 8),
            raw_router_logits=torch.zeros(4, 16),
            topk_config=None,
            num_token_non_padded=non_padded,
        )
        return StandardDispatchOutput(
            hidden_states=torch.zeros(4, 8), hidden_states_scale=None, topk_output=topk
        )

    def test_no_copy_runs_bf16(self):
        runner = _Runner()
        self.assertIsNone(m.run(object(), runner, self._dispatch(None)))
        self.assertEqual(runner.calls, [])

    def test_copy_keeps_logit_routing_of_every_row(self):
        runner = _Runner()
        info = object()
        with mock.patch.object(m, "quant_info", return_value=info):
            out = m.run(object(), runner, self._dispatch(torch.tensor(3)))
        self.assertEqual(out, "out")
        (dispatch, got), = runner.calls
        self.assertIs(got, info)
        # the BF16 MoE ignores num_token_non_padded; the FP8 call must not materialize top-k
        self.assertIsNone(dispatch.topk_output.num_token_non_padded)
        self.assertIsInstance(dispatch.topk_output, BypassedTopKOutput)


if __name__ == "__main__":
    unittest.main()
