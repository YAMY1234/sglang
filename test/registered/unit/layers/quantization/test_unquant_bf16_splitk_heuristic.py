"""
Unit tests for the opt-in BF16 split-K heuristic route in UnquantizedLinearMethod
(SGLANG_ENABLE_BF16_SPLITK_GEMM_HEURISTIC).

Listed (N, K) shapes take warp split-K up to M = 32 and TGV with K split over its
batch axis above; keys of the tuned split-K tables keep their route. The switch is
read only on SM107 with the cutedsl backend; off, the dispatch is unchanged.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

import sglang.srt.layers.quantization.unquant as unquant
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.unquant import (
    Bf16SplitKRoute,
    bf16_splitk_heuristic_route,
)
from sglang.test.test_utils import CustomTestCase

_FUSED_QKV_A = (2624, 6144)
_O_PROJ = (6144, 16384)
_DENSE_DOWN = (6144, 12288)
_INDEXER_WQ_B = (4096, 2048)
_DENSE_GATE_UP = (24576, 6144)
_WARP = Bf16SplitKRoute(kernel="warp", tactic=None, split_k=1)

# A small listed shape for dispatch tests: N % 16 == 0 and K % 256 == 0.
_N, _K = 64, 256
_BF16_RTOL, _BF16_ATOL = 1.6e-2, 1e-2


def _tgv(tactic: int, split_k: int) -> Bf16SplitKRoute:
    return Bf16SplitKRoute(kernel="tgv", tactic=tactic, split_k=split_k)


class TestBf16SplitKHeuristicRoute(CustomTestCase):
    def _route(self, m, shape, num_sms=148):
        return bf16_splitk_heuristic_route(m, *shape, num_sms=num_sms)

    def test_warp_splitk_serves_m_up_to_32(self):
        for shape in (_FUSED_QKV_A, _O_PROJ, _DENSE_DOWN, _INDEXER_WQ_B):
            for m in (1, 16, 32):
                with self.subTest(shape=shape, m=m):
                    self.assertEqual(self._route(m, shape), _WARP)

    def test_unlisted_shapes_and_m_keep_their_route(self):
        cases = (
            (8, (6144, 4096)),  # o_proj at attention TP 4
            (0, _O_PROJ),
            (129, _FUSED_QKV_A),
            (33, _INDEXER_WQ_B),
            (32, _DENSE_GATE_UP),
            (65, _DENSE_GATE_UP),
            (127, _DENSE_GATE_UP),
        )
        for m, shape in cases:
            with self.subTest(shape=shape, m=m):
                self.assertIsNone(self._route(m, shape, num_sms=1024))

    def test_tgv_token_tile_covers_m(self):
        # 2 * ceil(6144 / 128) = 96 CTAs; no K split fits in 148 SMs.
        self.assertEqual(self._route(64, _O_PROJ), _tgv(23, 1))
        self.assertEqual(self._route(100, _O_PROJ), _tgv(24, 1))

    def test_tgv_splits_k_within_one_wave(self):
        # 42 CTAs: x8 leaves 768-wide K slices, x6 and x4 exceed 148 SMs.
        self.assertEqual(self._route(96, _FUSED_QKV_A), _tgv(24, 3))
        # 96 CTAs x2 fill 192 SMs exactly.
        self.assertEqual(self._route(64, _O_PROJ, num_sms=192), _tgv(23, 2))

    def test_tgv_takes_wider_pairs_when_narrow_ones_exceed_a_wave(self):
        # 128-row pairs need 384 CTAs, 256-row pairs 192.
        self.assertEqual(self._route(64, _DENSE_GATE_UP, num_sms=200), _tgv(27, 1))
        self.assertEqual(self._route(128, _DENSE_GATE_UP, num_sms=200), _tgv(28, 1))
        self.assertIsNone(self._route(64, _DENSE_GATE_UP))


class TestBf16SplitKHeuristicDispatch(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.weight = torch.randn(_N, _K, dtype=torch.bfloat16) / 16.0  # 1 / sqrt(K)
        self.calls = []
        unquant._resolve_bf16_splitk_heuristic_route.cache_clear()
        self.addCleanup(unquant._resolve_bf16_splitk_heuristic_route.cache_clear)

    def _heuristic(self, *, enabled=True, default_tactic=None):
        def run_warp(a, b, out, pdl, tactic):
            self.calls.append(("warp", tactic))
            torch.matmul(a, b, out=out)

        def run_tgv(x, weight, out, *, tactic, split_k):
            self.calls.append(("tgv", tactic, split_k))
            torch.matmul(x, weight.t(), out=out)

        return patch.multiple(
            unquant,
            _BF16_SPLITK_HEURISTIC_M_RANGES={(_N, _K): ((1, 128),)},
            _enable_bf16_splitk_heuristic=enabled,
            _bf16_splitk_heuristic_num_sms=8,
            _heuristic_warp_splitk_default_tactic=default_tactic
            or (lambda m, n, k: f"default-{m}"),
            _run_heuristic_warp_splitk=run_warp,
            _run_tgv_splitk=run_tgv,
        )

    def _dispatch(self, x, bias=None):
        output = unquant._bf16_gemm_dispatch_impl(x, self.weight, bias)
        torch.testing.assert_close(
            output,
            F.linear(x, self.weight, bias),
            rtol=_BF16_RTOL,
            atol=_BF16_ATOL,
        )
        return output

    def test_switch_off_keeps_the_default_dispatch(self):
        x = torch.randn(8, _K, dtype=torch.bfloat16)
        with self._heuristic(enabled=False):
            self._dispatch(x)
            self.assertFalse(unquant.prefer_bf16_splitk_over_fused_a(x, self.weight))
        self.assertEqual(self.calls, [])

    def test_listed_shape_takes_warp_splitk_up_to_m32(self):
        with self._heuristic():
            for m in (1, 32):
                self._dispatch(torch.randn(m, _K, dtype=torch.bfloat16))
            # Leading dims fold into M.
            output = self._dispatch(torch.randn(2, 4, _K, dtype=torch.bfloat16))
        self.assertEqual(output.shape, (2, 4, _N))
        self.assertEqual(
            self.calls,
            [("warp", "default-1"), ("warp", "default-32"), ("warp", "default-8")],
        )

    def test_listed_shape_takes_tgv_above_m32(self):
        # 2 CTAs fit 8 SMs; K = 256 is too short to split.
        with self._heuristic():
            for m in (64, 128):
                self._dispatch(torch.randn(m, _K, dtype=torch.bfloat16))
        self.assertEqual(self.calls, [("tgv", 23, 1), ("tgv", 24, 1)])

    def test_other_m_and_inputs_keep_the_default_dispatch(self):
        with self._heuristic():
            self._dispatch(torch.randn(129, _K, dtype=torch.bfloat16))
            self._dispatch(
                torch.randn(8, _K, dtype=torch.bfloat16),
                bias=torch.randn(_N, dtype=torch.bfloat16),
            )
            # Not contiguous.
            self._dispatch(torch.randn(_K, 8, dtype=torch.bfloat16).t())
            # Not 32-byte aligned.
            self._dispatch(
                torch.randn(8 * _K + 1, dtype=torch.bfloat16)[1:].view(8, _K)
            )
        self.assertEqual(self.calls, [])

    def test_shape_without_a_warp_tactic_keeps_the_default_dispatch(self):
        def no_tactic(m, n, k):
            raise ValueError(f"no legal tactic for shape {(m, n, k)}")

        with self._heuristic(default_tactic=no_tactic):
            self._dispatch(torch.randn(8, _K, dtype=torch.bfloat16))
        self.assertEqual(self.calls, [])

    def test_tuned_splitk_tables_keep_precedence(self):
        def tuned(x, weight, bias):
            self.calls.append("tuned")
            return F.linear(x, weight, bias)

        with (
            self._heuristic(),
            patch.multiple(
                unquant,
                _enable_bf16_splitk_gemm=True,
                use_bf16_splitk_gemm=lambda m, n, k: m == 4,
                _bf16_splitk_gemm=tuned,
            ),
        ):
            for m in (4, 8):
                self._dispatch(torch.randn(m, _K, dtype=torch.bfloat16))
        self.assertEqual(self.calls, ["tuned", ("warp", "default-8")])

    def test_fused_a_yields_to_the_splitk_route(self):
        x = torch.randn(16, _K, dtype=torch.bfloat16)
        unlisted = torch.randn(_N, 2 * _K, dtype=torch.bfloat16)
        with self._heuristic():
            self.assertTrue(unquant.prefer_bf16_splitk_over_fused_a(x, self.weight))
            self.assertFalse(
                unquant.prefer_bf16_splitk_over_fused_a(
                    torch.randn(16, 2 * _K, dtype=torch.bfloat16), unlisted
                )
            )


class TestBf16SplitKHeuristicInit(CustomTestCase):
    def setUp(self):
        # initialize_bf16_gemm_config rebinds module globals; restore them all.
        saved = dict(vars(unquant))
        self.addCleanup(vars(unquant).update, saved)

    def _initialize(self, *, enabled, backend="auto", device_sm=107):
        platform = SimpleNamespace(
            is_sm90=False, is_sm100=True, is_sm120=False, device_sm=device_sm
        )
        exec_ = SimpleNamespace(
            kernel=SimpleNamespace(
                bf16_gemm_backend=backend, disable_flashinfer_autotune=True
            ),
            deterministic=SimpleNamespace(enable_deterministic_inference=False),
        )
        flashinfer_kernels = SimpleNamespace(
            default_tactic=None,
            prefer_direct_bf16_gemm_sm100=None,
            run_direct_dense=None,
            SplitKTactic=None,
            run_splitk_dense=None,
            WarpSplitKTactic=None,
            run_warp_splitk_dense=None,
        )
        modules = {
            "flashinfer.gemm.kernels.dense_bf16_gemm_direct": flashinfer_kernels,
            "flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk": flashinfer_kernels,
            "flashinfer.gemm.kernels.dense_bf16_gemm_warp_splitk": flashinfer_kernels,
            "sglang.kernels.ops.gemm.cutedsl_bf16_gemm": SimpleNamespace(
                cutedsl_bf16_gemm=None,
                use_cutedsl_bf16_gemm=None,
                tgv_splitk_bf16_gemm_out="tgv split-K",
            ),
            "sglang.kernels.ops.gemm.dense_bf16_gemm_warp_splitk": SimpleNamespace(
                default_tactic="warp default tactic",
                run_warp_splitk_dense="warp split-K",
            ),
        }
        with (
            envs.SGLANG_ENABLE_BF16_SPLITK_GEMM_HEURISTIC.override(enabled),
            patch.dict(sys.modules, modules),
            patch.object(unquant, "get_platform", return_value=platform),
            patch.object(unquant, "get_exec", return_value=exec_),
            patch.object(unquant, "get_device_core_count", return_value=148),
            patch.object(torch.cuda, "current_device", return_value=0),
        ):
            unquant.initialize_bf16_gemm_config()

    def test_switch_on_sm107_cutedsl_enables_the_route(self):
        self._initialize(enabled=True)
        self.assertTrue(unquant._enable_bf16_splitk_heuristic)
        self.assertEqual(unquant._bf16_splitk_heuristic_num_sms, 148)
        self.assertEqual(
            unquant._heuristic_warp_splitk_default_tactic, "warp default tactic"
        )
        self.assertEqual(unquant._run_heuristic_warp_splitk, "warp split-K")
        self.assertEqual(unquant._run_tgv_splitk, "tgv split-K")

    def test_route_is_off_without_the_switch_sm107_or_cutedsl(self):
        for kwargs in (
            dict(enabled=False),
            dict(enabled=True, device_sm=100),
            dict(enabled=True, backend="torch"),
        ):
            with self.subTest(**kwargs):
                self._initialize(enabled=True)
                self._initialize(**kwargs)
                self.assertFalse(unquant._enable_bf16_splitk_heuristic)


if __name__ == "__main__":
    unittest.main()
