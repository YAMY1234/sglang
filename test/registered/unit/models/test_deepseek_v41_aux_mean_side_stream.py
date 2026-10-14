"""DSpark aux means on the mHC stats stream: dispatch and model-loop wiring."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.test.test_utils import CustomTestCase

ROWS, MULT, HIDDEN = 6, 4, 64


def _residual():
    return torch.randn(ROWS, MULT, HIDDEN, dtype=torch.bfloat16)


class TestAuxMeanDispatch(CustomTestCase):
    def test_inline_without_stream_matches_mean(self):
        residual = _residual()
        out = mhc.mean_on_side_stream(residual, None)
        self.assertTrue(torch.equal(out, residual.mean(dim=1)))

    def test_triplet_appends_mean_after_coefficients(self):
        residual = _residual()
        order = []
        triplet = (torch.zeros(1), torch.zeros(1), torch.zeros(1))

        def fake_mix(hc, x, stream):
            order.append("mix")
            return triplet

        def fake_mean(x, stream):
            order.append("mean")
            return x.mean(dim=1)

        sink = []
        with (
            patch.object(mhc, "mix_stats", side_effect=fake_mix),
            patch.object(mhc, "mean_on_side_stream", side_effect=fake_mean),
        ):
            out = mhc._compute_triplet(None, residual, None, None, sink)
        self.assertIs(out, triplet)
        self.assertEqual(order, ["mix", "mean"])
        self.assertEqual(len(sink), 1)
        self.assertTrue(torch.equal(sink[0], residual.mean(dim=1)))

    def test_precomputed_triplet_still_fills_sink(self):
        residual = _residual()
        triplet = (torch.zeros(1), torch.zeros(1), torch.zeros(1))
        sink = []
        with patch.object(mhc, "mix_stats") as mix:
            out = mhc._compute_triplet(None, residual, None, triplet, sink)
        mix.assert_not_called()
        self.assertIs(out, triplet)
        self.assertTrue(torch.equal(sink[0], residual.mean(dim=1)))

    def test_no_sink_is_unchanged(self):
        residual = _residual()
        triplet = (torch.zeros(1), torch.zeros(1), torch.zeros(1))
        with (
            patch.object(mhc, "mix_stats", return_value=triplet),
            patch.object(mhc, "mean_on_side_stream") as mean,
        ):
            out = mhc._compute_triplet(None, residual, None)
        mean.assert_not_called()
        self.assertIs(out, triplet)


if __name__ == "__main__":
    unittest.main()
