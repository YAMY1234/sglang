"""#1019: the capturable fp64 Jacobi eigh of the k31 prompt-end truncation against torch.linalg.eigh (the reference's
call), under the Triton interpreter on CPU.  Eigenvectors are compared through projectors (they are unique up to sign)."""
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import unittest

import torch

from sglang.srt.layers.attention.linear.kernels import gdn_k31_eigh, gdn_prefill_reference as ref
from sglang.srt.layers.attention.linear.kernels.gdn_prefill_reference import K31_SEED, factorize_prefill_k31


def gram(batch, rank, n=16, width=128, seed=0):
    """Gram matrices bm bm^T of the truncation (bm = q^T x, m = 16 rows) with a decaying spectrum, plus the jitter."""
    g = torch.Generator().manual_seed(seed)
    bm = torch.randn(batch, n, width, generator=g, dtype=torch.float64)
    sv = torch.logspace(0, -3, n, dtype=torch.float64)
    sv[rank:] = 0
    u, _ = torch.linalg.qr(torch.randn(batch, n, n, generator=g, dtype=torch.float64))
    bm = u @ (sv[:, None] * torch.linalg.qr(bm.transpose(-1, -2))[0].transpose(-1, -2))
    G = bm @ bm.transpose(-1, -2)
    return G + (1e-7 * G.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-30) * torch.eye(n, dtype=G.dtype)


def projector(z, cols):
    v = z[..., cols]
    return v @ v.transpose(-1, -2)


class K31EighTest(unittest.TestCase):
    def test_r16_non_power_of_two_and_indefinite_spectra(self):
        for shift in (0., -0.25):
            G = gram(2, 24, n=24) + shift * torch.eye(24, dtype=torch.float64)
            d, z = gdn_k31_eigh.eigh(G)
            dr, zr = torch.linalg.eigh(G)
            torch.testing.assert_close(d, dr, rtol=1e-10, atol=1e-13)
            torch.testing.assert_close(z.transpose(-1, -2) @ z,
                                       torch.eye(24, dtype=torch.float64).expand(2, -1, -1),
                                       rtol=0, atol=1e-13)
            torch.testing.assert_close(projector(z, slice(-16, None)),
                                       projector(zr, slice(-16, None)), rtol=0, atol=1e-10)
            torch.testing.assert_close(z @ torch.diag_embed(d) @ z.transpose(-1, -2),
                                       G, rtol=0, atol=1e-13)

    def test_matches_torch_eigh(self):
        G = gram(24, 16)
        d, z = gdn_k31_eigh.eigh(G)
        dr, zr = torch.linalg.eigh(G)
        self.assertTrue(torch.all(d[:, 1:] >= d[:, :-1]))                       # ascending, as eigh
        self.assertLess(float(((d - dr).abs() / dr.abs().amax(-1, keepdim=True)).max()), 1e-13)
        eye = torch.eye(16, dtype=torch.float64)
        self.assertLess(float((z.transpose(-1, -2) @ z - eye).abs().max()), 1e-13)  # orthonormal
        # the r = 8 split the truncation uses: 1e-12 (#1019)
        self.assertLess(float((projector(z, slice(-8, None)) - projector(zr, slice(-8, None))).abs().max()), 1e-12)
        # every split r: within the Davis-Kahan bound of two backward-stable solvers, ~eps |G| / gap_r per matrix
        for r in range(1, 16):
            diff = (projector(z, slice(-r, None)) - projector(zr, slice(-r, None))).abs().amax((-1, -2))
            gap = dr[:, 16 - r] - dr[:, 15 - r]
            bound = 64 * torch.finfo(torch.float64).eps * dr[:, -1] / gap
            self.assertTrue(bool((diff <= bound).all()), (r, float((diff / bound).max())))
        self.assertLess(float((z @ torch.diag_embed(d) @ z.transpose(-1, -2) - G).abs().max()), 1e-14)

    def test_rank_deficient_content(self):
        # a short prompt: content rank 5 < r, the rest is the jitter plateau (any basis of it is valid in both)
        G = gram(8, 5, seed=1)
        d, z = gdn_k31_eigh.eigh(G)
        dr, zr = torch.linalg.eigh(G)
        self.assertLess(float((projector(z, slice(-5, None)) - projector(zr, slice(-5, None))).abs().max()), 1e-12)
        self.assertLess(float(((d - dr).abs() / dr.abs().amax(-1, keepdim=True)).max()), 1e-13)

    def test_truncation_jacobi_vs_torch(self):
        torch.manual_seed(3)
        B, HV, V, K, r = 1, 6, 128, 128, 8
        s = torch.randn(B, HV, V, 12) @ torch.randn(B, HV, 12, K) + 0.01 * torch.randn(B, HV, V, K)
        vbar = torch.randn(HV, V)
        omega = torch.randn(1, HV, V, r + 8, generator=torch.Generator().manual_seed(K31_SEED))
        old = ref.K31_EIGH
        try:
            ref.K31_EIGH = "torch"
            a0, u0, w0 = factorize_prefill_k31(s, vbar, r, 16, torch.float32, omega)
            ref.K31_EIGH = "jacobi"
            a1, u1, w1 = factorize_prefill_k31(s, vbar, r, 16, torch.float32, omega)
        finally:
            ref.K31_EIGH = old
        state = lambda a, u, w: vbar[None, :, :, None] * a[:, :, None, :] + torch.einsum("bhrv,bhrk->bhvk", w, u)
        s0, s1 = state(a0, u0, w0), state(a1, u1, w1)
        self.assertTrue(torch.equal(a0, a1))                                     # the sink does not involve eigh
        self.assertLess(float((s0 - s1).abs().max() / s0.abs().max()), 1e-6)     # fp32 factors: ulp-level only
        # factor rows agree up to sign
        sign = torch.sign((u0 * u1).sum(-1, keepdim=True))
        self.assertLess(float((u0 - sign * u1).abs().max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
