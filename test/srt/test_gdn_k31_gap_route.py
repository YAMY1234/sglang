"""B' (docs/223 §4.3): gap-routed k31 eigh. CPU only; solvers injected (the Triton kernels need CUDA).

Checks the cut criterion on the padded ascending spectrum, the fixed-capacity flagged-first gather, the exact
replacement by the fp64 solver's result for routed matrices only, the untouched mixed result elsewhere, and the
routed / overflow counters.
"""
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def _load():
    name = "sglang.srt.layers.attention.linear.kernels.gdn_k31_eigh_mixed"
    for pkg in ("sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.attention",
                "sglang.srt.layers.attention.linear", "sglang.srt.layers.attention.linear.kernels"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [str(ROOT / "python" / pkg.replace(".", "/"))]
            sys.modules[pkg] = m
    spec = importlib.util.spec_from_file_location(name, ROOT / "python" / (name.replace(".", "/") + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mixed = _load()


def spectrum_matrix(values, seed):
    """Symmetric fp64 32x32 with the given 24 eigenvalues and 8 padded (smallest) ones."""
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(32, 32, dtype=torch.float64, generator=g))
    vals = torch.cat([torch.full((8,), -10.0, dtype=torch.float64), torch.tensor(sorted(values), dtype=torch.float64)])
    return q @ torch.diag(vals) @ q.T


def mixed_fake(m):
    d, z = torch.linalg.eigh(m)
    return d, z * 1.0  # stands in for the fp32-basis solver


def fp64_fake(m):
    d, z = torch.linalg.eigh(m)
    return d + 7.0, -z  # distinguishable marker for routed matrices


class GapRouteTest(unittest.TestCase):
    def batch(self):
        base = [1.0 / (i + 1) for i in range(24)]          # healthy gaps
        tie = list(base); tie[16] = tie[15] * (1 - 1e-5)    # l16 ~ l17 (descending index 15 / 16) with energy
        dead = [1.0] + [1e-12] * 23                          # cut below the floor -> not routed
        mats = [spectrum_matrix(base, 0), spectrum_matrix(tie, 1), spectrum_matrix(dead, 2),
                spectrum_matrix(tie, 3), spectrum_matrix(base, 4)]
        return torch.stack(mats)

    def test_flags(self):
        d, _ = torch.linalg.eigh(self.batch())
        self.assertEqual(mixed.gap_route_flags(d, 16).tolist(), [False, True, False, True, False])

    def test_routed_rows_take_fp64_result_others_untouched(self):
        g = self.batch()
        mixed._GAP_ROUTE_STATS.clear()
        d, z = mixed.eigh_gap_routed(g, capacity=0.4, fp64_solver=fp64_fake, mixed_solver=mixed_fake)
        dm, zm = mixed_fake(g)
        df, zf = fp64_fake(g)
        for i in range(5):
            routed = i in (1, 3)
            self.assertTrue(torch.equal(d[i], (df if routed else dm)[i]), i)
            self.assertTrue(torch.equal(z[i], (zf if routed else zm)[i]), i)
        self.assertEqual(mixed.gap_route_stats("cpu"), (2, 0))

    def test_overflow_keeps_mixed_and_counts(self):
        g = self.batch()
        mixed._GAP_ROUTE_STATS.clear()
        d, _ = mixed.eigh_gap_routed(g, capacity=0.2, fp64_solver=fp64_fake, mixed_solver=mixed_fake)  # cap 1
        dm, _ = mixed_fake(g)
        df, _ = fp64_fake(g)
        self.assertTrue(torch.equal(d[1], df[1]))   # first flagged (stable order) is routed
        self.assertTrue(torch.equal(d[3], dm[3]))   # second flagged overflows -> mixed
        self.assertEqual(mixed.gap_route_stats("cpu"), (2, 1))

    def test_no_flags_is_pure_mixed(self):
        base = [1.0 / (i + 1) for i in range(24)]
        g = torch.stack([spectrum_matrix(base, s) for s in range(3)])
        d, z = mixed.eigh_gap_routed(g, capacity=0.4, fp64_solver=fp64_fake, mixed_solver=mixed_fake)
        dm, zm = mixed_fake(g)
        self.assertTrue(torch.equal(d, dm) and torch.equal(z, zm))


if __name__ == "__main__":
    unittest.main()
