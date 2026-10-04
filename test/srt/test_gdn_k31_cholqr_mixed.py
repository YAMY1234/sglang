"""Mixed-precision k31 CholeskyQR2 (SGLANG_GDN_K31_CHOLQR_MIXED): accuracy bounds and switch-off identity.

CPU only. The mixed path changes rounding, not the column space; it is not byte-exact
against the fp64 path, so (a) and (b) assert measured accuracy bounds instead.
"""

import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

PYTHON = Path(__file__).resolve().parents[2] / "python"


def _load():
    # Namespace stubs so only the two leaf modules execute (no sglang package import).
    for name in ("sglang", "sglang.srt", "sglang.srt.duet", "sglang.srt.layers",
                 "sglang.srt.layers.attention", "sglang.srt.layers.attention.linear",
                 "sglang.srt.layers.attention.linear.kernels"):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(PYTHON / name.replace(".", "/"))]
            sys.modules[name] = package
    return importlib.import_module("sglang.srt.layers.attention.linear.kernels.gdn_prefill_reference")


def _envs(enabled):
    env = types.ModuleType("sglang.srt.environ")
    env.envs = types.SimpleNamespace(
        SGLANG_GDN_K31_CHOLQR_MIXED=types.SimpleNamespace(get=lambda: enabled))
    return mock.patch.dict(sys.modules, {"sglang.srt.environ": env})


def _with_condition(kappa, m=128, n=24, batch=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    u, _ = torch.linalg.qr(torch.randn(batch, m, n, dtype=torch.float64, generator=g))
    v, _ = torch.linalg.qr(torch.randn(batch, n, n, dtype=torch.float64, generator=g))
    sigma = torch.logspace(0, -torch.log10(torch.tensor(kappa)).item(), n, dtype=torch.float64)
    return (u * sigma @ v.transpose(-1, -2)).float()


def _orth(q):
    q = q.double()
    eye = torch.eye(q.shape[-1], dtype=torch.float64)
    return (q.transpose(-1, -2) @ q - eye).norm(dim=(-2, -1)).max().item()


def _recon(q, y):
    q, y = q.double(), y.double()
    return ((y - q @ (q.transpose(-1, -2) @ y)).norm(dim=(-2, -1)) / y.norm(dim=(-2, -1))).max().item()


class MixedCholQRTest(unittest.TestCase):
    def setUp(self):
        self.ref = _load()

    def test_orthogonality_matches_fp64_cholqr2_across_condition_numbers(self):
        # Bound: within 4x the fp64 two-pass result plus 1e-5. Measured: equal up to kappa 1e4,
        # about 2x at 1e5-1e6 where both paths are jitter-limited (the real-state gate is the residual).
        for kappa in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6):
            y = _with_condition(kappa)
            old = self.ref._orth_cholqr2(y)
            new = self.ref._orth_cholqr2(y, mixed=True)
            self.assertTrue(torch.isfinite(new).all(), kappa)
            self.assertLessEqual(_orth(new), 4 * _orth(old) + 1e-5, kappa)
            self.assertLessEqual(_recon(new, y), 2 * _recon(old, y) + 1e-7, kappa)

    def test_rank_deficient_and_zero_inputs_stay_finite(self):
        g = torch.Generator().manual_seed(1)
        cases = {
            "zero": torch.zeros(4, 128, 24),
            "rank1": torch.randn(4, 128, 1, generator=g).expand(4, 128, 24).contiguous(),
            "rank8": torch.randn(4, 128, 8, generator=g) @ torch.randn(4, 8, 24, generator=g),
        }
        for name, y in cases.items():
            new = self.ref._orth_cholqr2(y, mixed=True)
            self.assertTrue(torch.isfinite(new).all(), name)
            if name == "zero":
                self.assertEqual(new.abs().max().item(), 0.0)
            else:
                self.assertLessEqual(_recon(new, y), 1e-6, name)

    def test_k31_factors_match_fp64_path(self):
        g = torch.Generator().manual_seed(2)
        b, hv, v, k, r, rmax = 1, 8, 128, 128, 16, 32
        # Decaying spectrum, as in prompt-end GDN states.
        left = torch.randn(b, hv, v, 48, generator=g)
        right = torch.randn(b, hv, 48, k, generator=g)
        s = (left * torch.logspace(0, -4, 48)) @ right
        vbar = torch.randn(hv, v, generator=g)
        omega = torch.randn(b, hv, v, r + 8, generator=g)
        out = {}
        for mixed in (False, True):
            a, u, w = self.ref.factorize_prefill_k31(s, vbar, r, rmax, torch.float32, omega,
                                                      mixed_cholqr=mixed)
            stored = vbar[None, :, :, None] * a[:, :, None, :] + w.transpose(-1, -2) @ u
            residual = ((s - stored).norm(dim=(-2, -1)) / s.norm(dim=(-2, -1)))
            projector = u[:, :, :r].transpose(-1, -2) @ u[:, :, :r]
            out[mixed] = (residual, projector, a)
        # Same accounting as the R3-a gate: residual_max within 1e-3 relative of fp64.
        self.assertLessEqual(out[True][0].max().item(), out[False][0].max().item() * (1 + 1e-3) + 1e-6)
        self.assertLessEqual((out[True][1] - out[False][1]).norm(dim=(-2, -1)).max().item() / 4, 1e-4)
        torch.testing.assert_close(out[True][2], out[False][2], rtol=0, atol=0)

    def test_switch_off_is_the_old_fp64_path_bitwise(self):
        def old_cholqr2(y):
            yd = y.double()
            for _ in range(2):
                g = yd.transpose(-1, -2) @ yd
                g = g + (1e-7 * g.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-30) * torch.eye(
                    g.shape[-1], device=g.device, dtype=g.dtype)
                chol = torch.linalg.cholesky_ex(g)[0]
                yd = torch.linalg.solve_triangular(chol, yd.transpose(-1, -2), upper=False).transpose(-1, -2)
            return yd.to(y.dtype)

        y = _with_condition(1e4, batch=8)
        self.assertTrue(torch.equal(self.ref._orth_cholqr2(y), old_cholqr2(y)))
        g = torch.Generator().manual_seed(3)
        s = torch.randn(1, 4, 128, 128, generator=g)
        vbar, omega = torch.randn(4, 128, generator=g), torch.randn(1, 4, 128, 24, generator=g)
        with _envs(False):
            default = self.ref.factorize_prefill_k31(s, vbar, 16, 32, torch.float16, omega)
        with mock.patch.object(self.ref, "_orth_cholqr2", side_effect=lambda y, mixed=False: old_cholqr2(y)):
            legacy = self.ref.factorize_prefill_k31(s, vbar, 16, 32, torch.float16, omega, mixed_cholqr=False)
        for x, y in zip(default, legacy):
            self.assertTrue(torch.equal(x, y))
        with _envs(True), mock.patch.object(self.ref, "_orth_cholqr2", wraps=self.ref._orth_cholqr2) as spy:
            self.ref.factorize_prefill_k31(s, vbar, 16, 32, torch.float16, omega)
        self.assertEqual([c.kwargs["mixed"] for c in spy.call_args_list], [True, True, True])


REAL_STATES = Path(__file__).resolve().parent / "fixtures" / "k31_real_states_fixture.pt"


class RealPromptStatesTest(unittest.TestCase):
    """Prompt-end GDN states from lowc-ab task10diag (j989452 F0), kappa(Y) ~ 1e8-1e11.

    These heads made the plain fp32 Cholesky fail under the 1e-6 relative jitter and
    turned whole layers of stored factors non-finite in production.
    """

    def setUp(self):
        self.ref = _load()
        self.fix = torch.load(REAL_STATES, weights_only=False)

    def _factor(self, mixed):
        dense, vbar, omega = self.fix["dense"], self.fix["vbar"], self.fix["omega"]
        return self.ref.factorize_prefill_k31(dense[None], vbar, 16, 32, torch.float16,
                                              omega[None], mixed_cholqr=mixed)

    def test_old_plain_jitter_fails_on_these_states(self):
        m, n = 128, 24
        dense, vbar, omega = self.fix["dense"], self.fix["vbar"], self.fix["omega"]
        a = torch.einsum("hvk,hv->hk", dense, vbar) / vbar.square().sum(-1)[:, None]
        y = (dense - vbar[:, :, None] * a[:, None, :]).transpose(-1, -2) @ omega
        first, _ = self.ref._cholqr_fp32(y, shifted=True)
        g = first.transpose(-1, -2) @ first
        g = g + (1e-6 * g.diagonal(dim1=-2, dim2=-1).mean(-1) + 1e-30)[:, None, None] * torch.eye(n)
        # Most of these heads fail outright (8/12 here; per-head rounding differs from the batched capture).
        self.assertGreaterEqual(int((torch.linalg.cholesky_ex(g)[1] != 0).sum()), 6)

    def test_real_states_stay_finite_and_match_fp64_residual(self):
        before = self.ref.mixed_cholqr_fallbacks("cpu")
        out = {mixed: self._factor(mixed) for mixed in (False, True)}
        dense, vbar = self.fix["dense"][None], self.fix["vbar"]
        residual = {}
        for mixed, (a, u, w) in out.items():
            self.assertTrue(all(torch.isfinite(t.float()).all() for t in (a, u, w)), mixed)
            stored = vbar[None, :, :, None] * a[:, :, None, :] + w.float().transpose(-1, -2) @ u.float()
            residual[mixed] = (dense - stored).norm(dim=(-2, -1)) / dense.norm(dim=(-2, -1))
        # R3-a gate accounting: per-head residual within 1% of the fp64 path.
        self.assertLessEqual((residual[True] / residual[False]).max().item(), 1.01)
        self.assertEqual(self.ref.mixed_cholqr_fallbacks("cpu"), before)

    def test_fp32_stage_failure_falls_back_to_fp64_input_and_counts(self):
        y = _with_condition(1e3, batch=4)
        before = self.ref.mixed_cholqr_fallbacks("cpu")
        real = self.ref._cholqr_fp32

        def broken(x, shifted):
            q, info = real(x, shifted)
            if shifted:
                return q, info
            q = q.clone()
            q[1] = float("nan")
            info = info.clone()
            info[1] = 3
            return q, info

        with mock.patch.object(self.ref, "_cholqr_fp32", side_effect=broken):
            staged = self.ref._mixed_fp32_stage(y)
        self.assertTrue(torch.isfinite(staged).all())
        self.assertTrue(torch.equal(staged[1], y[1].double()))
        self.assertEqual(self.ref.mixed_cholqr_fallbacks("cpu") - before, 1)

    def test_real_states_r8_are_finite_with_default_off_identity(self):
        dense, vbar = self.fix["dense"][None], self.fix["vbar"]
        omega = self.fix["omega"][None, ..., :16]
        outputs = {}
        before = self.ref.mixed_cholqr_fallbacks("cpu")
        for enabled in (False, True):
            with _envs(enabled):
                outputs[enabled] = self.ref.factorize_prefill_k31(
                    dense, vbar, 8, 16, torch.float16, omega)
            self.assertTrue(all(torch.isfinite(t).all() for t in outputs[enabled]))
        explicit = self.ref.factorize_prefill_k31(
            dense, vbar, 8, 16, torch.float16, omega, mixed_cholqr=False)
        for left, right in zip(outputs[False], explicit):
            self.assertTrue(torch.equal(left, right))
        residual = {}
        for enabled, (a, u, w) in outputs.items():
            stored = vbar[None, :, :, None] * a[:, :, None, :] + w.float().transpose(-1, -2) @ u.float()
            residual[enabled] = (dense - stored).norm(dim=(-2, -1)) / dense.norm(dim=(-2, -1))
        self.assertLessEqual(residual[True].max(), residual[False].max() * 1.01)
        self.assertEqual(self.ref.mixed_cholqr_fallbacks("cpu"), before)


if __name__ == "__main__":
    os.environ.setdefault("SGLANG_GDN_K31_EIGH", "torch")
    unittest.main()
