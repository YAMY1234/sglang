"""CPU checks of sglang.srt.duet.state_factor.small_eigh (lead #1557): backend selection by dimension / device /
override, the exactness of padding below the Gershgorin bound (nvfp4-perf line's method, run here with
torch.linalg.eigh standing in for the power-of-two kernel), and the torch fallback.

Run from this directory:  python -m unittest test_duet_small_eigh
"""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/sglang/srt/models"))
from lightning_duet._common import load as _load  # noqa: E402

sf = _load("state_factor")


def gram(batch, n, rank=None):
    torch.manual_seed(1234 + n)
    a = torch.randn(batch, n, rank or n, dtype=torch.float64)
    return a @ a.transpose(-1, -2)


def projector(z, k):
    v = z[..., -k:]
    return v @ v.transpose(-1, -2)


class Backend(unittest.TestCase):
    def test_selection_table(self):
        self.assertEqual(sf.small_eigh_backend(16, "cpu", "auto"), "torch")
        self.assertEqual(sf.small_eigh_backend(16, "cuda", "torch"), "torch")
        self.assertEqual(sf.small_eigh_backend(16, "cuda", "auto"), "jacobi")
        self.assertEqual(sf.small_eigh_backend(24, "cuda", "auto"), "jacobi-padded")   # r16 + oversample 8
        self.assertEqual(sf.small_eigh_backend(24, "cuda", "jacobi"), "jacobi-padded")
        self.assertEqual(sf.small_eigh_backend(32, "cuda", "auto"), "jacobi")
        self.assertEqual(sf.small_eigh_backend(100, "cuda", "auto"), "torch")           # beyond the kernel's size
        with self.assertRaises(ValueError):
            sf.small_eigh_backend(100, "cuda", "jacobi")
        with self.assertRaises(ValueError):
            sf.small_eigh_backend(16, "cuda", "fast")

    def test_env_override_is_read(self):
        import os
        prev = os.environ.get(sf.SMALL_EIGH_ENV)
        try:
            os.environ[sf.SMALL_EIGH_ENV] = "torch"
            self.assertEqual(sf.small_eigh_backend(16, "cuda"), "torch")
        finally:
            if prev is None:
                os.environ.pop(sf.SMALL_EIGH_ENV, None)
            else:
                os.environ[sf.SMALL_EIGH_ENV] = prev


class Padding(unittest.TestCase):
    def test_padded_eigenpairs_lie_below_the_spectrum_and_the_rest_is_exact(self):
        for shift in (0.0, -3.0, 50.0):   # psd, indefinite, strongly shifted spectra
            g = gram(2, 24) + shift * torch.eye(24, dtype=torch.float64)
            padded = sf.pad_below_spectrum(g)
            self.assertEqual(tuple(padded.shape[-2:]), (32, 32))
            self.assertTrue(torch.equal(padded[..., :24, :24], g))
            self.assertTrue(torch.equal(padded[..., :24, 24:], torch.zeros(2, 24, 8, dtype=torch.float64)))
            all_values = torch.linalg.eigvalsh(padded)
            discarded = torch.linalg.eigvalsh(padded[..., 24:, 24:])
            torch.testing.assert_close(all_values[..., :8], discarded, rtol=0, atol=1e-13)   # padding is the bottom 8
            self.assertTrue(bool((discarded.amax(-1) < torch.linalg.eigvalsh(g).amin(-1)).all()))
            # the dispatcher with the kernel stubbed by torch.linalg.eigh reproduces the direct decomposition
            d, z = sf.small_eigh(g, override="jacobi", _solver=torch.linalg.eigh)
            dr, zr = torch.linalg.eigh(g)
            torch.testing.assert_close(d, dr, rtol=1e-10, atol=1e-13)
            torch.testing.assert_close(z.transpose(-1, -2) @ z, torch.eye(24, dtype=torch.float64).expand(2, -1, -1), rtol=0, atol=1e-13)
            torch.testing.assert_close(projector(z, 16), projector(zr, 16), rtol=0, atol=1e-10)
            torch.testing.assert_close(z @ torch.diag_embed(d) @ z.transpose(-1, -2), g, rtol=0, atol=1e-12)

    def test_power_of_two_goes_straight_to_the_solver_and_torch_fallback_is_exact(self):
        g = gram(3, 16)
        seen = []
        def solver(x):
            seen.append(tuple(x.shape[-2:])); return torch.linalg.eigh(x)
        d, z = sf.small_eigh(g, override="jacobi", _solver=solver)
        self.assertEqual(seen, [(16, 16)])
        dr, zr = torch.linalg.eigh(g)
        self.assertTrue(torch.equal(d, dr) and torch.equal(z, zr))
        d2, z2 = sf.small_eigh(g, override="torch")
        self.assertTrue(torch.equal(d2, dr) and torch.equal(z2, zr))
        with self.assertRaises(ValueError):
            sf.small_eigh(g.float())


if __name__ == "__main__":
    unittest.main()
