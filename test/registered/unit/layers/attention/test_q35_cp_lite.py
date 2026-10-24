"""CP-lite refuses LL exchange polls narrower than their CTA: such a poll holds a word in two
warps, and its block-wide count could let one of them return the previous call's copy."""

import unittest

from sglang.srt.layers.attention import q35_cp_lite as cl
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLLPollWidths(CustomTestCase):
    def test_shipped_shapes_are_accepted(self):
        # TP8 / TEP8: cp 4 x 4 heads (256 words, or 128 in two gather chunks); TEP16: cp 8 x 2 heads
        for q_heads, chunks in ((4, 1), (4, 2), (2, 1)):
            cl._check_ll_polls(q_heads, chunks)

    def test_polls_narrower_than_the_cta_are_refused(self):
        # TP8 with SGLANG_PSM_CP_GATHER_CHUNKS=4, TEP16 in two chunks, one head a rank: 64 words each
        for q_heads, chunks in ((4, 4), (2, 2), (1, 1)):
            with self.subTest(q_heads=q_heads, chunks=chunks):
                with self.assertRaisesRegex(ValueError, "polling race"):
                    cl._check_ll_polls(q_heads, chunks)


if __name__ == "__main__":
    unittest.main()
