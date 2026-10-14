"""Pinned low-latency MXFP8 tactics: which problems pin, and the tuner hook's lifetime."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys
import types
import unittest
from unittest.mock import patch

from sglang.srt.layers.quantization.fp8_utils import (
    _MXFP8_LOW_LATENCY_GEMM_MODULE,
    MXFP8_LOW_LATENCY_PINNED_TACTICS,
    mxfp8_low_latency_pinned_tactic,
    mxfp8_low_latency_pinned_tactics,
)
from sglang.test.test_utils import CustomTestCase

WQKV_A = (1792, 5120)
OFFERED = [(256, 8, 1, 1), (128, 10, 1, 4), (256, 6, 1, 1), (256, 4, 1, 2)]


def _fake_ll_gemm_module():
    module = types.ModuleType(_MXFP8_LOW_LATENCY_GEMM_MODULE)
    module.autotune_tactics = lambda problem_mnkl, *args, **kwargs: list(OFFERED)
    return module


class TestPinnedTacticLookup(CustomTestCase):
    def test_bucket_eight_pins_on_gb300(self):
        n, k = WQKV_A
        for rows in (5, 6, 8):
            self.assertEqual(
                mxfp8_low_latency_pinned_tactic(sm_version=103, n=n, k=k, rows=rows),
                (128, 10, 1, 4),
            )

    def test_other_rows_shapes_and_gpus_do_not_pin(self):
        n, k = WQKV_A
        for rows in (1, 2, 4, 9):
            self.assertIsNone(
                mxfp8_low_latency_pinned_tactic(sm_version=103, n=n, k=k, rows=rows)
            )
        self.assertIsNone(
            mxfp8_low_latency_pinned_tactic(sm_version=100, n=n, k=k, rows=6)
        )
        self.assertIsNone(
            mxfp8_low_latency_pinned_tactic(sm_version=103, n=n, k=4096, rows=6)
        )

    def test_table_holds_the_three_tp4_projections(self):
        self.assertEqual(
            sorted(MXFP8_LOW_LATENCY_PINNED_TACTICS),
            [(103, 1792, 5120), (103, 5120, 2048), (103, 8192, 1280)],
        )


class TestPinnedTacticsHook(CustomTestCase):
    def setUp(self):
        self.module = _fake_ll_gemm_module()
        self.original = self.module.autotune_tactics
        patcher = patch.dict(
            sys.modules, {_MXFP8_LOW_LATENCY_GEMM_MODULE: self.module}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_pinned_problem_offers_only_the_pin(self):
        n, k = WQKV_A
        with mxfp8_low_latency_pinned_tactics(sm_version=103):
            # The runner passes (N, M, K, L).
            self.assertEqual(self.module.autotune_tactics((n, 6, k, 1)), [(128, 10, 1, 4)])
            self.assertEqual(self.module.autotune_tactics((n, 4, k, 1)), OFFERED)
        self.assertIs(self.module.autotune_tactics, self.original)

    def test_pin_missing_from_offer_falls_back_to_tuning(self):
        n, k = 5120, 2048  # pinned (256, 4, 1, 2) is in OFFERED; drop it
        self.module.autotune_tactics = lambda *a, **kw: [(256, 8, 1, 1)]
        with mxfp8_low_latency_pinned_tactics(sm_version=103):
            self.assertEqual(self.module.autotune_tactics((n, 8, k, 1)), [(256, 8, 1, 1)])

    def test_restored_after_exception(self):
        with self.assertRaises(RuntimeError):
            with mxfp8_low_latency_pinned_tactics(sm_version=103):
                raise RuntimeError
        self.assertIs(self.module.autotune_tactics, self.original)


if __name__ == "__main__":
    unittest.main()
