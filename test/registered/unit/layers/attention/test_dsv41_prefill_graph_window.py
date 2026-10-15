"""DSV4.1 prefill CUDA graph: the captured ratio-1/2 indexer takes the full top-k, so it
may only be captured while every compressed position of the graph's longest context
fits in the candidate window (2048 blocks x 8 on DeepSeek-V4.1-Flash); otherwise the
sources stay in their eager break instead of failing the capture."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest

from sglang.srt.layers.attention.deepseek_v4_backend import (
    _low_ratio_graph_fits_candidates,
)

WINDOW = 2048 * 8


class TestLowRatioGraphWindow(unittest.TestCase):
    def test_boundary_is_per_smallest_ratio(self):
        fits = lambda ratios, n: _low_ratio_graph_fits_candidates(
            low_ratios=ratios, candidate_window=WINDOW, max_seq_len=n
        )
        self.assertTrue(fits((1, 2), 16384))
        self.assertFalse(fits((1, 2), 16384 + 256))
        self.assertTrue(fits((2,), 32768))
        self.assertFalse(fits((1, 2), None))

    def test_without_candidates_the_graph_is_unbounded(self):
        self.assertTrue(
            _low_ratio_graph_fits_candidates(
                low_ratios=(1, 2), candidate_window=0, max_seq_len=262144
            )
        )


if __name__ == "__main__":
    unittest.main()
