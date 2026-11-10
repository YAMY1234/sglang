"""SM107 bitwise small-M BF16 GEMM routes: tables, gating, fallback (CPU)."""

import hashlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import sm107_bf16_smallm as routes
from sglang.srt.layers.quantization import unquant
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# FlashInfer 0.7.2 dense_bf16_gemm_sm100_splitk._SMEM_CAPACITY_BYTES on SM10x.
_SMEM_CAPACITY = 227 * 1024
_SPLITK = "sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk"


def _fake_splitk(source: bytes):
    """sys.modules entries for the two imports install() makes, the split-K module backed by ``source``."""
    path = tempfile.NamedTemporaryFile(suffix=".py", delete=False)
    path.write(source)
    path.close()
    splitk = types.ModuleType(_SPLITK)
    splitk.__file__ = path.name
    splitk._SMEM_CAPACITY_BYTES = _SMEM_CAPACITY
    driver = types.ModuleType("cuda.bindings.driver")
    driver.CUstream = object
    bindings = types.ModuleType("cuda.bindings")
    bindings.driver = driver
    return {
        _SPLITK: splitk,
        "cuda": types.ModuleType("cuda"),
        "cuda.bindings": bindings,
        "cuda.bindings.driver": driver,
    }


class TestRouteTable(CustomTestCase):
    def test_stock_tuned_keys_keep_the_stock_route(self):
        """The routes are consulted before the split-K allowlists, so a shared key would change a tuned shape."""
        stock = (
            set(unquant._BF16_SPLITK_TUNED_TACTICS)
            | set(unquant._BF16_WARP_SPLITK_TUNED_TACTICS)
            | set(unquant._BF16_SM107_WARP_SPLITK_TUNED_TACTICS)
        )
        self.assertEqual(set(routes.ROUTES) & stock, set())

    def test_cluster_routes_follow_cublas_slicing(self):
        """Bitwise only at cuBLAS 13.8's slicing: 4 K slices at M <= 16, 8 at M 17-32, each tiling K evenly."""
        cluster = {key: t for key, (kind, t) in routes.ROUTES.items() if kind == "cl"}
        self.assertEqual(len(cluster), 70)
        for (m, n, k), tactic in cluster.items():
            mma_m, mma_n, split_k, stages, cta_k = tactic
            self.assertEqual(split_k, 4 if m <= 16 else 8, (m, n, k))
            self.assertEqual((k // split_k) % cta_k, 0, (m, n, k))
            self.assertLessEqual(-(-m // mma_n), 4, (m, n, k))
            self.assertTrue(routes._cluster_ok(k, tactic, _SMEM_CAPACITY), (m, n, k))

    def test_tgv_routes_are_valid_tiles(self):
        kinds = [kind for kind, _ in routes.ROUTES.values()]
        self.assertEqual(
            (len(kinds), kinds.count("tgv"), kinds.count("tgvc")), (87, 12, 5)
        )
        for key, (kind, tactic) in routes.ROUTES.items():
            if kind == "tgvc":
                self.assertTrue(routes._tgvc_ok(*tactic), key)
        self.assertFalse(
            routes._tgvc_ok(64, 72, 8, True)
        )  # 2-CTA tiles step by 16 columns
        self.assertFalse(
            routes._tgvc_ok(64, 256, 20, True)
        )  # over the shared-memory budget


class TestInstall(CustomTestCase):
    def tearDown(self):
        routes._routes = None
        routes._fi = None

    def test_refused_off_sm107_or_without_split_k(self):
        for kwargs in (
            dict(device_sm=100, splitk_enabled=True, cutedsl_available=True),
            dict(device_sm=103, splitk_enabled=True, cutedsl_available=True),
            dict(device_sm=107, splitk_enabled=False, cutedsl_available=True),
            dict(device_sm=107, splitk_enabled=True, cutedsl_available=False),
        ):
            with self.subTest(**kwargs):
                self.assertFalse(routes.install(**kwargs))
                self.assertIsNone(routes._routes)

    def test_installed_only_against_the_checked_split_k_source(self):
        source = b"# split-K kernel source\n"
        digest = hashlib.sha256(source).hexdigest()
        with mock.patch.dict(sys.modules, _fake_splitk(source)):
            self.assertFalse(
                routes.install(
                    device_sm=107, splitk_enabled=True, cutedsl_available=True
                )
            )
            self.assertIsNone(routes._routes)
            with mock.patch.object(routes, "_FI_SPLITK_SHA256", digest):
                self.assertTrue(
                    routes.install(
                        device_sm=107, splitk_enabled=True, cutedsl_available=True
                    )
                )
        self.assertIs(routes._routes, routes.ROUTES)

    def test_the_vendored_split_k_source_is_the_pinned_one(self):
        import sglang.kernels.ops.gemm as gemm_ops

        path = gemm_ops.__path__[0] + "/dense_bf16_gemm_sm100_splitk.py"
        with open(path, "rb") as f:
            self.assertEqual(
                hashlib.sha256(f.read()).hexdigest(), routes._FI_SPLITK_SHA256
            )


class TestNonbitwise(CustomTestCase):
    def tearDown(self):
        routes._routes = None
        routes._fi = None
        routes._checked = False
        routes._nonbitwise = frozenset()

    def test_opt_in_adds_only_valid_disjoint_keys(self):
        self.assertEqual(set(routes.NONBITWISE_ROUTES) & set(routes.ROUTES), set())
        for (m, n, k), (kind, tactic) in routes.NONBITWISE_ROUTES.items():
            self.assertEqual(kind, "cl")
            self.assertTrue(routes._cluster_ok(k, tactic, _SMEM_CAPACITY), (m, n, k))
        source = b"# split-K kernel source\n"
        digest = hashlib.sha256(source).hexdigest()
        for on in (False, True):
            with (
                mock.patch.dict(sys.modules, _fake_splitk(source)),
                mock.patch.object(routes, "_FI_SPLITK_SHA256", digest),
                envs.SGLANG_ENABLE_SM107_BF16_SMALLM_NONBITWISE_ROUTES.override(on),
            ):
                self.assertTrue(
                    routes.install(
                        device_sm=107, splitk_enabled=True, cutedsl_available=True
                    )
                )
            extra = set(routes.NONBITWISE_ROUTES) if on else set()
            self.assertEqual(set(routes._routes), set(routes.ROUTES) | extra)
            self.assertEqual(routes._nonbitwise, frozenset(extra))

    def test_load_check_drops_only_the_keys_past_the_bound(self):
        """Bitwise keys are compared bit for bit; each non-bitwise key by relative L2."""

        def stock(x, w):
            return torch.ones(x.shape[0], w.shape[0], dtype=torch.bfloat16)

        def routed(x, w):
            out = stock(x, w)
            if (x.shape[0], *w.shape) == (24, 2560, 6144):
                out *= 1.01  # relative L2 1e-2, past the 1e-3 bound
            return out

        cheap = mock.patch.object(
            torch,
            "randn",
            lambda *shape, **kw: torch.zeros(*shape, device=kw.get("device")),
        )
        installed = {**routes.ROUTES, **routes.NONBITWISE_ROUTES}
        with (
            cheap,
            mock.patch.object(routes, "_routes", installed),
            mock.patch.object(
                routes, "_nonbitwise", frozenset(routes.NONBITWISE_ROUTES)
            ),
            mock.patch.object(routes, "gemm", routed),
        ):
            self.assertTrue(routes.precompile(stock_gemm=stock, device="cpu"))
            self.assertNotIn((24, 2560, 6144), routes._routes)
            self.assertNotIn((24, 2560, 6144), routes._nonbitwise)
            self.assertEqual(len(routes._nonbitwise), len(routes.NONBITWISE_ROUTES) - 1)
            self.assertTrue(set(routes.ROUTES) <= set(routes._routes))


class TestFailClosed(CustomTestCase):
    def tearDown(self):
        routes._routes = None
        routes._fi = None
        routes._checked = False

    def test_missing_split_k_module_keeps_the_stock_dispatch(self):
        with mock.patch.dict(sys.modules, {_SPLITK: None}):
            self.assertFalse(
                routes.install(
                    device_sm=107, splitk_enabled=True, cutedsl_available=True
                )
            )
        self.assertIsNone(routes._routes)

    def test_load_check_turns_the_routes_off_on_a_mismatch(self):
        """The source pin sees only the route side; the stock side (cuBLAS's slicing) is checked at load."""

        def stock(x, w):
            return torch.zeros(x.shape[0], w.shape[0], dtype=torch.bfloat16)

        def off_by_one_bit(x, w):
            out = stock(x, w)
            if (x.shape[0], *w.shape) == (6, 512, 2560):
                out.view(torch.int16)[0, 0] = 1
            return out

        cheap = mock.patch.object(
            torch,
            "randn",
            lambda *shape, **kw: torch.zeros(*shape, device=kw.get("device")),
        )
        with cheap, mock.patch.object(routes, "_routes", routes.ROUTES):
            with mock.patch.object(routes, "gemm", stock):
                self.assertTrue(routes.precompile(stock_gemm=stock, device="cpu"))
                self.assertIs(routes._routes, routes.ROUTES)
            with mock.patch.object(routes, "gemm", off_by_one_bit):
                self.assertFalse(routes.precompile(stock_gemm=stock, device="cpu"))
                self.assertIsNone(routes._routes)

    def test_only_the_full_precompile_checks_the_routes(self):
        """A model precompiling its own split-K keys (GLM-5 on SM107) leaves the routes unchecked, so unserved."""
        with (
            mock.patch.object(unquant, "_enable_bf16_splitk_gemm", True),
            mock.patch.object(unquant, "_sm107_smallm_on", True),
            mock.patch.object(routes, "precompile", return_value=True) as check,
            mock.patch.object(unquant, "_bf16_splitk_gemm_out"),
            mock.patch.object(torch.cuda, "current_device", return_value="cpu"),
            mock.patch.object(torch.cuda, "synchronize"),
        ):
            self.assertTrue(unquant.precompile_splitk_tactics(keys=[(1, 8, 8)]))
            check.assert_not_called()
            self.assertTrue(unquant.precompile_splitk_tactics())
            check.assert_called_once()
            self.assertTrue(unquant._sm107_smallm_on)


class TestDispatch(CustomTestCase):
    def setUp(self):
        self.calls = []

        def runner(kind):
            def run(x_2d, weight, out, tactic):
                self.calls.append((kind, tuple(x_2d.shape), tactic))
                out.fill_(7.0)

            return run

        patches = [
            mock.patch.object(routes, "_routes", routes.ROUTES),
            mock.patch.object(routes, "_checked", True),
            mock.patch.object(
                routes,
                "_RUNNERS",
                {kind: runner(kind) for kind in ("cl", "tgv", "tgvc")},
            ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_listed_key_runs_its_route_unlisted_and_strided_fall_back(self):
        weight = torch.zeros(512, 2560, dtype=torch.bfloat16)
        x = torch.ones(2, 3, 2560, dtype=torch.bfloat16)  # M = 6, a router cluster key
        out = routes.gemm(x, weight)
        self.assertEqual(self.calls, [("cl", (6, 2560), (64, 8, 4, 5, 128))])
        self.assertEqual(tuple(out.shape), (2, 3, 512))
        self.assertTrue(bool((out == 7.0).all()))

        self.assertIsNone(
            routes.gemm(torch.ones(8, 2560, dtype=torch.bfloat16), weight)
        )
        wide = torch.ones(6, 2 * 2560, dtype=torch.bfloat16)[:, :2560]
        self.assertIsNone(routes.gemm(wide, weight))
        base = torch.ones(6 * 2560 + 1, dtype=torch.bfloat16)
        self.assertIsNone(routes.gemm(base[1:].view(6, 2560), weight))  # 2-byte offset
        self.assertIsNone(
            routes.gemm(
                torch.ones(6, 2560, dtype=torch.bfloat16), weight.t().contiguous().t()
            )
        )
        self.assertEqual(len(self.calls), 1)

    def test_routes_serve_only_after_the_load_check(self):
        x = torch.ones(6, 2560, dtype=torch.bfloat16)
        weight = torch.zeros(512, 2560, dtype=torch.bfloat16)
        with mock.patch.object(routes, "_checked", False):
            self.assertIsNone(routes.gemm(x, weight))
        self.assertEqual(self.calls, [])

    def test_bias_and_addend_keep_the_stock_dispatch(self):
        """The routes have no bias or addend epilogue: those calls must reach the stock dispatch."""
        x = torch.randn(6, 2560).to(torch.bfloat16)
        weight = torch.randn(512, 2560).to(torch.bfloat16)
        bias = torch.randn(512).to(torch.bfloat16)
        with mock.patch.object(unquant, "_sm107_smallm_on", True):
            routed = unquant._bf16_gemm_dispatch_impl(x, weight, None)
            self.assertTrue(bool((routed == 7.0).all()))
            torch.testing.assert_close(
                unquant._bf16_gemm_dispatch_impl(x, weight, bias),
                torch.nn.functional.linear(x, weight, bias),
            )
            addend = torch.zeros(6, 512, dtype=torch.bfloat16)
            torch.testing.assert_close(
                unquant._bf16_gemm_dispatch_impl(x, weight, None, addend=addend),
                torch.nn.functional.linear(x, weight),
            )
        self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
