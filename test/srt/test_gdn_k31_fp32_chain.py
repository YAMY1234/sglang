"""K4.1 fp32 k31 prompt-end chain (SGLANG_GDN_K31_FP32_CHAIN): switch-off identity and CPU numerics.

CPU only. Without CUDA the Triton Jacobi kernels run only under TRITON_INTERPRET=1 (set it in the environment
before starting the process); otherwise the kernel tests skip. On CPU the k31 eigh dispatcher resolves to
torch.linalg.eigh unless a test forces "jacobi", so tests that name the served eigh force it and need the kernel.

Admissible evidence is the real-state fixture (12 prompt-end heads, kappa(Y) ~ 1e8-1e11) and the served dumps checked by
test/srt/k31_fp32_check.py on GPU. The graded synthetic sets below are coverage only. Numbers are printed (-s).

Run: TRITON_INTERPRET=1 python -m pytest -s test/srt/test_gdn_k31_fp32_chain.py
"""

import importlib
import itertools
import logging
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
KERNELS = "sglang.srt.layers.attention.linear.kernels"
REF_NAME = KERNELS + ".gdn_prefill_reference"
REF_PATH = "python/sglang/srt/layers/attention/linear/kernels/gdn_prefill_reference.py"
POOL_PATH = "python/sglang/srt/mem_cache/gdn_factored_pool.py"
# The fork commit this line branched from: the admitted served path the switch-off must reproduce byte for byte.
BASE_REV = os.environ.get("K41_BASE_REV", "c44d5446d63")
REAL_STATES = Path(__file__).resolve().parent / "fixtures" / "k31_real_states_fixture.pt"
KERNEL_OK = torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"  # the fallback counters key on tensor.device
CHAIN_ENV = "SGLANG_GDN_K31_FP32_CHAIN"


def _load(name):
    # Namespace stubs so only the leaf modules execute (no sglang package import).
    for pkg in ("sglang", "sglang.srt", "sglang.srt.duet", "sglang.srt.layers", "sglang.srt.layers.attention",
                "sglang.srt.layers.attention.linear", KERNELS, "sglang.srt.mem_cache"):
        if pkg not in sys.modules:
            package = types.ModuleType(pkg)
            package.__path__ = [str(PYTHON / pkg.replace(".", "/"))]
            sys.modules[pkg] = package
    return importlib.import_module(name)


def _frozen(relpath, name, package):
    """The BASE_REV source of a file as a module (git show); None when git or the revision is unavailable."""
    if name in sys.modules:
        return sys.modules[name]
    try:
        src = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE_REV}:{relpath}"], check=True,
                             capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    module = types.ModuleType(name)
    module.__file__ = f"{BASE_REV}:{relpath}"
    module.__package__ = package
    sys.modules[name] = module
    exec(compile(src, module.__file__, "exec"), module.__dict__)
    return module


def _fixture(device="cpu"):
    fix = torch.load(REAL_STATES, weights_only=False)
    return fix["dense"][None].to(device), fix["vbar"].to(device), fix["omega"][None].to(device)


def _graded(decades, heads=16, seed=0, device="cpu"):
    """Synthetic coverage only: content singular values logspace(0, -decades, 128) on random bases, plus a sink."""
    g = torch.Generator().manual_seed(seed)
    left, _ = torch.linalg.qr(torch.randn(heads, 128, 128, dtype=torch.float64, generator=g))
    right, _ = torch.linalg.qr(torch.randn(heads, 128, 128, dtype=torch.float64, generator=g))
    sigma = torch.logspace(0, -decades, 128, dtype=torch.float64)
    vbar = torch.randn(heads, 128, generator=g)
    sink = vbar.double()[:, :, None] * torch.randn(heads, 1, 128, dtype=torch.float64, generator=g) * 0.3
    dense = ((left * sigma) @ right.transpose(-1, -2) + sink).float()
    omega = torch.randn(1, heads, 128, 24, generator=g)
    return dense[None].to(device), vbar.to(device), omega.to(device)


def _cliff(strong, heads=16, tail=1e-6, seed=0, device="cpu"):
    """Synthetic coverage only: `strong` content directions (1 .. 0.1) then a flat tail `tail`, plus a sink."""
    g = torch.Generator().manual_seed(seed)
    left, _ = torch.linalg.qr(torch.randn(heads, 128, 128, dtype=torch.float64, generator=g))
    right, _ = torch.linalg.qr(torch.randn(heads, 128, 128, dtype=torch.float64, generator=g))
    sigma = torch.full((128,), tail, dtype=torch.float64)
    sigma[:strong] = torch.logspace(0, -1, strong, dtype=torch.float64)
    vbar = torch.randn(heads, 128, generator=g)
    sink = vbar.double()[:, :, None] * torch.randn(heads, 1, 128, dtype=torch.float64, generator=g) * 0.3
    dense = ((left * sigma) @ right.transpose(-1, -2) + sink).float()
    omega = torch.randn(1, heads, 128, 24, generator=g)
    return dense[None].to(device), vbar.to(device), omega.to(device)


def _stored(vbar, a, u, w):
    return vbar[None, :, :, None] * a[:, :, None, :] + w.float().transpose(-1, -2) @ u.float()


def _residual(dense, vbar, out):
    return (dense - _stored(vbar, *out)).norm(dim=(-2, -1)).flatten().double()


def _nonfinite(out):
    return sum(int((~torch.isfinite(t.float())).sum()) for t in out)


def _summary(label, ratio):
    ratio = ratio.cpu()
    q = torch.quantile(ratio, torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64)).tolist()
    line = (f"[K41] {label}: heads={ratio.numel()} ratio min={ratio.min().item():.6f} median={q[0]:.6f} "
            f"p90={q[1]:.6f} p99={q[2]:.6f} max={ratio.max().item():.6f} "
            f"frac<=1.01={(ratio <= 1.01).double().mean().item():.4f}")
    print(line, flush=True)
    return line


def _factor(ref, dense, vbar, omega, **kw):
    return ref.factorize_prefill_k31(dense, vbar, 16, 32, torch.float16, omega, **kw)


class _Base(unittest.TestCase):
    def setUp(self):
        self.ref = _load(REF_NAME)

    def reference(self, dense, vbar, omega):
        """fp64 reference path: fp64 CholeskyQR2 and torch fp64 eigh."""
        with mock.patch.object(self.ref, "K31_EIGH", "torch"):
            return _factor(self.ref, dense, vbar, omega, mixed_cholqr=False, mixed_eigh=False)

    def gate(self, label, dense, vbar, omega, out, ref_out, *, admissible):
        ratio = _residual(dense, vbar, out) / _residual(dense, vbar, ref_out)
        _summary(label, ratio)
        self.assertEqual(_nonfinite(out), 0, label)
        if admissible:
            # Pre-registered (docs/220 K4.1): <= 1.01 on >= 99.9 % of heads, max <= 1.02.
            self.assertGreaterEqual((ratio <= 1.01).double().mean().item(), 0.999, label)
            self.assertLessEqual(ratio.max().item(), 1.02, label)
        return ratio


class TournamentScheduleTest(unittest.TestCase):
    def test_24_round_robin_covers_every_pair_once_and_self_pairs_the_padding(self):
        n, m = 32, 24
        seen = set()
        for turn in range(m - 1):
            partner = []
            for x in range(n):
                p = turn if x == m - 1 else (m - 1 if x == turn else (2 * turn - x + 2 * (m - 1)) % (m - 1))
                partner.append(p if x < m else x)
            for x in range(n):
                self.assertEqual(partner[partner[x]], x)
                if x >= m:
                    self.assertEqual(partner[x], x)
                else:
                    self.assertNotEqual(partner[x], x)
                    self.assertLess(partner[x], m)
                    seen.add((min(x, partner[x]), max(x, partner[x])))
        self.assertEqual(len(seen), m * (m - 1) // 2)


class SwitchOffIdentityTest(_Base):
    """Switch off: the BASE_REV functions, byte for byte."""

    def setUp(self):
        super().setUp()
        self.old_ref = _frozen(REF_PATH, KERNELS + "._k41_frozen_reference", KERNELS)
        if self.old_ref is None:
            self.skipTest(f"git show {BASE_REV} unavailable")

    def _inputs(self):
        g = torch.Generator().manual_seed(3)
        yield "synthetic", (torch.randn(1, 4, 128, 128, generator=g), torch.randn(4, 128, generator=g),
                            torch.randn(1, 4, 128, 24, generator=g))
        yield "fixture", _fixture()

    def test_reference_function_bitwise(self):
        for (name, (dense, vbar, omega)), mixed_cholqr, mixed_eigh in itertools.product(
                self._inputs(), (False, True), (False, True)):
            with self.subTest(name=name, mixed_cholqr=mixed_cholqr, mixed_eigh=mixed_eigh):
                old = _factor(self.old_ref, dense, vbar, omega, mixed_cholqr=mixed_cholqr, mixed_eigh=mixed_eigh)
                for kw in ({}, {"fp32_chain": 0}):
                    new = _factor(self.ref, dense, vbar, omega, mixed_cholqr=mixed_cholqr, mixed_eigh=mixed_eigh,
                                  **kw)
                    for x, y in zip(old, new):
                        self.assertTrue(torch.equal(x, y))

    def test_env_resolved_cholqr_bitwise(self):
        dense, vbar, omega = _fixture()
        for flag in ("0", "1"):
            with mock.patch.dict(os.environ, {"SGLANG_GDN_K31_CHOLQR_MIXED": flag, CHAIN_ENV: "0"}):
                old = _factor(self.old_ref, dense, vbar, omega, mixed_eigh=True)
                new = _factor(self.ref, dense, vbar, omega, mixed_eigh=True)
            for x, y in zip(old, new):
                self.assertTrue(torch.equal(x, y))

    @unittest.skipUnless(KERNEL_OK, "Triton Jacobi needs CUDA or TRITON_INTERPRET=1")
    def test_served_jacobi_path_bitwise(self):
        # The admitted served eigh (fp32 _fp32_jacobi + fp64 Newton) executes, not the CPU torch fallback.
        dense, vbar, omega = _fixture(DEV)
        with mock.patch.object(self.old_ref, "K31_EIGH", "jacobi"):
            old = _factor(self.old_ref, dense, vbar, omega, mixed_cholqr=True, mixed_eigh=True)
        with mock.patch.object(self.ref, "K31_EIGH", "jacobi"):
            new = _factor(self.ref, dense, vbar, omega, mixed_cholqr=True, mixed_eigh=True)
        for x, y in zip(old, new):
            self.assertTrue(torch.equal(x, y))

    def test_admitted_kernel_source_unchanged(self):
        # Text of the admitted kernel and its wrappers (decorators included) equals BASE_REV's.
        import ast
        rel = "python/sglang/srt/layers/attention/linear/kernels/gdn_k31_eigh_mixed.py"
        old_src = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE_REV}:{rel}"], check=True,
                                 capture_output=True, text=True).stdout
        new_src = (ROOT / rel).read_text()

        def segments(src):
            tree = ast.parse(src)
            lines = src.splitlines()
            return {node.name: "\n".join(lines[(node.decorator_list[0].lineno if node.decorator_list else node.lineno)
                                                 - 1:node.end_lineno])
                    for node in tree.body if isinstance(node, ast.FunctionDef)}

        old, new = segments(old_src), segments(new_src)
        for name in ("_fp32_jacobi", "fp32_vectors", "correct_once", "eigh"):
            self.assertEqual(new[name], old[name], name)
        self.assertTrue(new_src.startswith(old_src.rstrip("\n")))  # the change is append-only


class FactorizeLayersGateTest(unittest.TestCase):
    """factorize_layers: switch resolution, the shared geometry gate, the log line, and switch-off bytes."""

    def setUp(self):
        self.ref = _load(REF_NAME)
        self.pool = _load("sglang.srt.mem_cache.gdn_factored_pool")
        self.old_ref = _frozen(REF_PATH, KERNELS + "._k41_frozen_reference", KERNELS)

    def _inputs(self, h=24, b=1, layers=36):
        dense, vbar, omega = _fixture()
        idx = torch.arange(layers * h) % dense.shape[1]
        states = dense[0, idx].reshape(layers, 1, h, 128, 128).expand(layers, b, h, 128, 128)
        return list(states.unbind(0)), vbar[idx].reshape(layers, h, 128), omega[:, idx[:h]].expand(b, h, 128, 24)

    def _cfg(self, module, **changes):
        cfg = module.FactoredGDNConfig(r=16, m=16, dtype=torch.float16, init_method="k31", init_oversample=8,
                                       decode_method="iter")
        for name, value in changes.items():
            setattr(cfg, name, value)
        return cfg

    def _selected(self, env, inputs, **changes):
        states, vbar, omega = inputs
        with mock.patch.dict(os.environ, env), mock.patch.object(self.pool, "factorize_dense",
                                                                 return_value=(None, None, None)) as call:
            try:
                self.pool.factorize_layers(states, vbar, self._cfg(self.pool, **changes), omega=omega)
            except TypeError:
                pass  # the stubbed factors are not sliceable; only the call arguments matter
        return call.call_args.kwargs["fp32_chain"], call.call_args.kwargs["mixed_eigh"]

    def test_switch_resolution_and_gate(self):
        h24 = self._inputs(24)
        for value, expected in (("", 0), ("0", 0), ("1", 1), ("2", 2)):
            with self.subTest(value=value):
                self.assertEqual(self._selected({CHAIN_ENV: value}, h24), (expected, True))
        env = {CHAIN_ENV: "2"}
        self.assertEqual(self._selected(env, self._inputs(48)), (2, True))
        self.assertEqual(self._selected(dict(env, SGLANG_GDN_K31_MIXED_EIGH="0"), h24), (0, False))
        self.assertEqual(self._selected(dict(env, SGLANG_GDN_K31_MIXED_EIGH_ANY_BATCH="0"), self._inputs(24, b=2)),
                         (0, False))
        self.assertEqual(self._selected(dict(env, SGLANG_GDN_K31_MIXED_EIGH_ANY_BATCH="1"), self._inputs(24, b=2)),
                         (2, True))
        self.assertEqual(self._selected(env, self._inputs(12)), (0, False))
        self.assertEqual(self._selected(env, self._inputs(24, layers=35)), (0, False))
        self.assertEqual(self._selected(env, h24, dtype=torch.bfloat16), (0, False))
        self.assertEqual(self._selected(env, h24, decode_method="warm"), (0, False))
        with self.assertRaises(ValueError):
            self._selected({CHAIN_ENV: "3"}, h24)

    def test_activation_logged_once(self):
        h24 = self._inputs(24)
        self.pool.factorize_layers.__dict__.pop("_fp32_chain_logged", None)
        with self.assertLogs(self.pool.logger, logging.INFO) as logs:
            self._selected({CHAIN_ENV: "1"}, h24)
            self._selected({CHAIN_ENV: "1"}, h24)
        self.assertEqual(sum("K31_FP32_CHAIN_ACTIVE stage=1 36 1 24" in line for line in logs.output), 1)

    def test_switch_off_layers_bitwise_against_base_pool(self):
        if self.old_ref is None:
            self.skipTest(f"git show {BASE_REV} unavailable")
        old_pool = _frozen(POOL_PATH, "sglang.srt.mem_cache._k41_frozen_pool", "sglang.srt.mem_cache")
        for h, env in ((24, {}), (24, {CHAIN_ENV: "0"}), (48, {CHAIN_ENV: "0", "SGLANG_GDN_K31_CHOLQR_MIXED": "1"})):
            with self.subTest(h=h, env=env):
                states, vbar, omega = self._inputs(h)
                with mock.patch.dict(os.environ, env):
                    with mock.patch.dict(sys.modules, {REF_NAME: self.old_ref}):
                        old = old_pool.factorize_layers(states, vbar, self._cfg(old_pool), omega=omega)
                    new = self.pool.factorize_layers(states, vbar, self._cfg(self.pool), omega=omega)
                for row_old, row_new in zip(old, new):
                    for x, y in zip(row_old, row_new):
                        self.assertTrue(torch.equal(x, y))


class Stage1NumericsTest(_Base):
    """fp32 shifted CholeskyQR3 + today's eigh, against the fp64 reference path."""

    def test_fixture_gate_cpu_eigh(self):
        dense, vbar, omega = _fixture()
        before = self.ref.fp32_chain_cholqr_fallbacks("cpu")
        today = _factor(self.ref, dense, vbar, omega, mixed_cholqr=True, mixed_eigh=True)
        out = _factor(self.ref, dense, vbar, omega, mixed_eigh=True, fp32_chain=1)
        self.gate("stage1 fixture (CPU torch eigh)", dense, vbar, omega, out, self.reference(dense, vbar, omega),
                  admissible=True)
        after = self.ref.fp32_chain_cholqr_fallbacks("cpu")
        print(f"[K41] stage1 fixture fallbacks [any, pass1, pass2, pass3] = "
              f"{[x - y for x, y in zip(after, before)]}", flush=True)
        self.assertEqual(after, before)
        self.assertTrue(torch.equal(out[0], today[0]))

    @unittest.skipUnless(KERNEL_OK, "Triton Jacobi needs CUDA or TRITON_INTERPRET=1")
    def test_fixture_gate_served_eigh(self):
        dense, vbar, omega = _fixture(DEV)
        before = self.ref.fp32_chain_cholqr_fallbacks(DEV)
        with mock.patch.object(self.ref, "K31_EIGH", "jacobi"):
            today = _factor(self.ref, dense, vbar, omega, mixed_cholqr=True, mixed_eigh=True)
            out = _factor(self.ref, dense, vbar, omega, mixed_eigh=True, fp32_chain=1)
        ref_out = self.reference(dense, vbar, omega)
        self.gate("today served (mixed CholQR + mixed Jacobi eigh) fixture", dense, vbar, omega, today, ref_out,
                  admissible=False)
        self.gate("stage1 fixture (mixed Jacobi eigh)", dense, vbar, omega, out, ref_out, admissible=True)
        self.assertEqual(self.ref.fp32_chain_cholqr_fallbacks(DEV), before)
        self.assertTrue(torch.equal(out[0], today[0]))

    def test_graded_synthetic_coverage(self):
        for decades in (3, 6, 9):
            dense, vbar, omega = _graded(decades, seed=decades)
            before = self.ref.fp32_chain_cholqr_fallbacks("cpu")
            out = _factor(self.ref, dense, vbar, omega, mixed_eigh=True, fp32_chain=1)
            self.gate(f"stage1 graded 1e-{decades} (synthetic, CPU torch eigh)", dense, vbar, omega, out,
                      self.reference(dense, vbar, omega), admissible=False)
            delta = [x - y for x, y in zip(self.ref.fp32_chain_cholqr_fallbacks("cpu"), before)]
            print(f"[K41] stage1 graded 1e-{decades} fallbacks = {delta}", flush=True)

    def test_cliff_synthetic_coverage(self):
        # Content rank below / near r = 16 with a flat tail: the weak-direction regime the fp32 jitter floor hits.
        for strong, tail in ((8, 1e-6), (16, 1e-6), (20, 1e-6), (12, 1e-9)):
            dense, vbar, omega = _cliff(strong, tail=tail, seed=strong)
            before = self.ref.fp32_chain_cholqr_fallbacks("cpu")
            out = _factor(self.ref, dense, vbar, omega, mixed_eigh=True, fp32_chain=1)
            self.gate(f"stage1 cliff {strong} strong + tail {tail:.0e} (synthetic, CPU torch eigh)", dense, vbar,
                      omega, out, self.reference(dense, vbar, omega), admissible=False)
            delta = [x - y for x, y in zip(self.ref.fp32_chain_cholqr_fallbacks("cpu"), before)]
            print(f"[K41] stage1 cliff {strong}/{tail:.0e} fallbacks = {delta}", flush=True)

    def test_kappa_sweep_cholqr3_fp32(self):
        # Orthogonality ||Q^T Q - I||_F and reconstruction ||Y - Q Q^T Y|| / ||Y|| (max over 64 matrices).
        from test_gdn_k31_cholqr_mixed import _orth, _recon, _with_condition
        before = self.ref.fp32_chain_cholqr_fallbacks("cpu")
        for kappa in (1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11):
            y = _with_condition(kappa)
            fp64 = self.ref._orth_cholqr2(y)
            mixed = self.ref._orth_cholqr2(y, mixed=True)
            new = self.ref._orth_cholqr3_fp32(y)
            print(f"[K41] kappa={kappa:.0e} orth fp64={_orth(fp64):.2e} mixed={_orth(mixed):.2e} "
                  f"fp32x3={_orth(new):.2e} | recon fp64={_recon(fp64, y):.2e} mixed={_recon(mixed, y):.2e} "
                  f"fp32x3={_recon(new, y):.2e}", flush=True)
            self.assertTrue(torch.isfinite(new).all(), kappa)
            self.assertEqual(new.dtype, torch.float32)
            # Measured (CPU): orthogonality floor ~8e-5 up to kappa 1e4 (the plain-pass jitter sqrt(m) u trace(G);
            # fp64 CholQR2 ~5e-7). From kappa 1e5 the columns with sigma_i(Y) below ~2e-4 ||Y||_F are damped by
            # that jitter instead of normalised (orth 2e-3 at 1e5, 0.12 at 1e6, ~1.2 from 1e7: O(1) on those
            # columns), while the reconstruction stays ~1e-5. Bounds record that, they are not a design claim.
            if kappa <= 1e4:
                self.assertLessEqual(_orth(new), 2e-4, kappa)
            self.assertLessEqual(_recon(new, y), 5e-5, kappa)
        print(f"[K41] kappa sweep fallbacks = "
              f"{[x - y for x, y in zip(self.ref.fp32_chain_cholqr_fallbacks('cpu'), before)]}", flush=True)

    def test_failed_pass_keeps_previous_output_and_counts(self):
        from test_gdn_k31_cholqr_mixed import _with_condition
        y = _with_condition(1e3, batch=4)
        real = self.ref._cholqr_fp32
        calls = []

        def broken(x, shifted):
            q, info = real(x, shifted)
            calls.append(q)
            if len(calls) == 2:  # the first plain pass fails on matrix 1
                q, info = q.clone(), info.clone()
                q[1] = float("nan")
                info[1] = 3
            return q, info

        before = self.ref.fp32_chain_cholqr_fallbacks("cpu")
        with mock.patch.object(self.ref, "_cholqr_fp32", side_effect=broken):
            out = self.ref._orth_cholqr3_fp32(y)
        self.assertTrue(torch.isfinite(out).all())
        delta = [x - y for x, y in zip(self.ref.fp32_chain_cholqr_fallbacks("cpu"), before)]
        self.assertEqual(delta, [1, 0, 1, 0])
        # matrix 1's third pass ran on the first pass output; the others on the second pass output
        third_alone, _ = real(calls[0][1:2], False)
        torch.testing.assert_close(out[1:2], third_alone, rtol=0, atol=0)
        mixed_before = self.ref.mixed_cholqr_fallbacks("cpu")
        self.assertGreaterEqual(mixed_before, 1)  # the pool's K31_CHOLQR_MIXED_FALLBACK reader includes the chain

    def test_zero_and_rank_deficient_stay_finite(self):
        g = torch.Generator().manual_seed(1)
        cases = {
            "zero": torch.zeros(4, 128, 24),
            "rank8": torch.randn(4, 128, 8, generator=g) @ torch.randn(4, 8, 24, generator=g),
        }
        for name, y in cases.items():
            out = self.ref._orth_cholqr3_fp32(y)
            self.assertTrue(torch.isfinite(out).all(), name)


@unittest.skipUnless(KERNEL_OK, "Triton Jacobi needs CUDA or TRITON_INTERPRET=1")
class Stage2NumericsTest(_Base):
    """Stage 1 plus the fp32 24-tournament Jacobi eigh (no fp64 Newton)."""

    def setUp(self):
        super().setUp()
        self.mixed = _load(KERNELS + ".gdn_k31_eigh_mixed")

    def test_tournament_jacobi_matches_fp64_eigh(self):
        g = torch.Generator().manual_seed(4)
        q, _ = torch.linalg.qr(torch.randn(6, 24, 24, dtype=torch.float64, generator=g))
        lam = torch.logspace(3, -1, 24, dtype=torch.float64)
        gram = (q * lam) @ q.transpose(-1, -2)
        d, z, count = self.mixed.eigh_fp32_active(gram.float().to(DEV))
        d_ref, z_ref = torch.linalg.eigh(gram)
        d, z = d.cpu().double(), z.cpu().double()
        self.assertEqual(z.shape, (6, 24, 24))
        self.assertLessEqual(((d - d_ref).abs() / d_ref.abs().amax(-1, keepdim=True)).max().item(), 1e-5)
        eye = torch.eye(24, dtype=torch.float64)
        self.assertLessEqual((z.transpose(-1, -2) @ z - eye).abs().max().item(), 1e-5)
        proj = z[..., -16:] @ z[..., -16:].transpose(-1, -2) - z_ref[..., -16:] @ z_ref[..., -16:].transpose(-1, -2)
        self.assertLessEqual(proj.norm(dim=(-2, -1)).max().item(), 1e-4)
        self.assertTrue(((count >= 1) & (count <= 12)).all())
        print(f"[K41] tournament Jacobi synthetic sweeps = {count.tolist()}", flush=True)

    def test_fixture_gate(self):
        dense, vbar, omega = _fixture(DEV)
        fb_before = self.ref.fp32_chain_cholqr_fallbacks(DEV)
        hist_before = self.ref.fp32_chain_sweep_histogram(DEV)
        with mock.patch.object(self.ref, "K31_EIGH", "jacobi"):
            today = _factor(self.ref, dense, vbar, omega, mixed_cholqr=True, mixed_eigh=True)
        out = _factor(self.ref, dense, vbar, omega, mixed_eigh=True, fp32_chain=2)
        self.gate("stage2 fixture", dense, vbar, omega, out, self.reference(dense, vbar, omega), admissible=True)
        hist = self.ref.fp32_chain_sweep_histogram(DEV)
        hist = [x - (hist_before[i] if i < len(hist_before) else 0) for i, x in enumerate(hist)]
        sweeps = torch.repeat_interleave(torch.arange(len(hist)), torch.tensor(hist))
        print(f"[K41] stage2 fixture sweep histogram (index = sweeps) = {hist}; median = "
              f"{sweeps.median().item()}", flush=True)
        self.assertEqual(self.ref.fp32_chain_cholqr_fallbacks(DEV), fb_before)
        self.assertTrue(torch.equal(out[0], today[0]))
        self.assertEqual(sum(hist), dense.shape[1])
        self.assertLessEqual(sweeps.median().item(), 8)  # pre-registered

    def test_graded_synthetic_coverage(self):
        for decades in (3, 6):
            dense, vbar, omega = _graded(decades, heads=8, seed=decades, device=DEV)
            hist_before = self.ref.fp32_chain_sweep_histogram(DEV)
            out = _factor(self.ref, dense, vbar, omega, mixed_eigh=True, fp32_chain=2)
            self.gate(f"stage2 graded 1e-{decades} (synthetic)", dense, vbar, omega, out,
                      self.reference(dense, vbar, omega), admissible=False)
            hist = self.ref.fp32_chain_sweep_histogram(DEV)
            print(f"[K41] stage2 graded 1e-{decades} sweep histogram = "
                  f"{[x - (hist_before[i] if i < len(hist_before) else 0) for i, x in enumerate(hist)]}", flush=True)


if __name__ == "__main__":
    unittest.main()
