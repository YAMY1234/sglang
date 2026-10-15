"""Pinned low-latency MXFP8 tactics: which problems pin, and the tuner hook's lifetime."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import sys
import types
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from sglang.srt.layers.quantization import fp8_utils
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


def _tune_like_flashinfer(module, cache, *, n, k, rows):
    """FlashInfer 0.7.0.post1 for <= 8 rows: a call tunes profiles at every power of two
    below `rows` and at `rows` (get_hybrid_num_tokens_buckets), each filed under the next
    power of two (map_to_hybrid_bucket_uncapped); a filed bucket is not re-tuned."""
    profiles = sorted({1 << i for i in range(rows.bit_length()) if 1 << i <= rows} | {rows})
    for m in profiles:
        bucket = 1 << (m - 1).bit_length()
        # The first offered tactic stands in for the fastest one.
        cache.setdefault((n, k, bucket), module.autotune_tactics((n, m, k, 1))[0])
    tactic = cache[(n, k, 1 << (rows - 1).bit_length())]
    if tactic not in module.autotune_tactics((n, rows, k, 1)):
        raise ValueError(f"Invalid low-latency GEMM tactic: {tactic}")
    return tactic


class TestPinnedTacticsAcrossAutotunePasses(CustomTestCase):
    def setUp(self):
        self.module = _fake_ll_gemm_module()
        patcher = patch.dict(
            sys.modules, {_MXFP8_LOW_LATENCY_GEMM_MODULE: self.module}
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        gb300 = patch.object(
            fp8_utils.torch.cuda, "get_device_capability", return_value=(10, 3)
        )
        gb300.start()
        self.addCleanup(gb300.stop)

    def _startup(self, *, decode_rows, pin_every_pass):
        cache = {}
        n, k = WQKV_A
        decode_pass = (
            fp8_utils.maybe_mxfp8_low_latency_pinned_tactics()
            if pin_every_pass
            else nullcontext()
        )
        with decode_pass:
            _tune_like_flashinfer(self.module, cache, n=n, k=k, rows=decode_rows)
        with fp8_utils.mxfp8_low_latency_pinned_tactics(sm_version=103):
            return _tune_like_flashinfer(self.module, cache, n=n, k=k, rows=8)

    def test_six_row_decode_pass_files_bucket_eight_unpinned(self):
        # max_running_requests=1: the decode pass is 1 request x 6 verify rows.
        with self.assertRaisesRegex(ValueError, "Invalid low-latency GEMM tactic"):
            self._startup(decode_rows=6, pin_every_pass=False)

    def test_pinned_decode_pass_keeps_the_pin_for_every_row_count(self):
        with patch.dict(
            os.environ, {"SGLANG_OPT_DSV41_LOW_LATENCY_MXFP8_PINNED_TACTICS": "1"}
        ):
            for rows in range(1, 9):
                with self.subTest(decode_rows=rows):
                    self.assertEqual(
                        self._startup(decode_rows=rows, pin_every_pass=True),
                        (128, 10, 1, 4),
                    )

    def test_flag_off_adds_no_hook(self):
        with patch.dict(
            os.environ, {"SGLANG_OPT_DSV41_LOW_LATENCY_MXFP8_PINNED_TACTICS": "0"}
        ):
            original = self.module.autotune_tactics
            with fp8_utils.maybe_mxfp8_low_latency_pinned_tactics():
                self.assertIs(self.module.autotune_tactics, original)


if __name__ == "__main__":
    unittest.main()
