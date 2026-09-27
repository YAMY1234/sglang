"""#873: the k31 prompt-final truncation equals Mingyuan's unified StateFactor on the same state and directions.

The reference below is copied verbatim from origin/minma/0913 twinstar/duet/state.py (sha256 45436a7c9c30cb6f...):
_orthonormalize, _small_eigh, truncate_rank and StateFactor.sink/forward(warm=False) for side="right" (gated delta,
S (B, H, Dk, Dv), sink direction on Dv). Our serving layout is S^T per head (sglang (V, K)).
"""
import unittest

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_prefill_reference import K31_SEED, factorize_prefill_k31

OVERSAMPLE, POWER = 8, 1


def _orthonormalize(Y):
    Yd = Y.double()
    for _ in range(2):
        G = Yd.transpose(-1, -2) @ Yd
        G = G + (1e-7 * G.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-30) * torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
        L = torch.linalg.cholesky(G)
        Yd = torch.linalg.solve_triangular(L, Yd.transpose(-1, -2), upper=False).transpose(-1, -2)
    return Yd.to(Y.dtype)


def _small_eigh(G):
    G = G.double()
    G = G + (1e-7 * G.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-30) * torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
    return torch.linalg.eigh(G)[1].to(torch.float32)


def truncate_rank(S, r, prev=None):
    Sf = S.float()
    B, H, P, N = Sf.shape
    m = min(r + OVERSAMPLE, P, N)
    g = torch.Generator(device=Sf.device)
    g.manual_seed(0x5EED)
    use_prev = prev is not None and tuple(prev.shape) == (B, H, N, r) and prev.device == Sf.device
    omega = torch.randn(B, H, N, m - (r if use_prev else 0), generator=g, device=Sf.device, dtype=torch.float32)
    if use_prev:
        omega = torch.cat([prev.float(), omega], -1)
    with torch.no_grad():
        Y = Sf @ omega
        for _ in range(POWER):
            Y = Sf @ _orthonormalize(Sf.transpose(-1, -2) @ _orthonormalize(Y))
        Q = _orthonormalize(Y)
        Bm = Q.transpose(-1, -2) @ Sf
        W = _small_eigh(Bm @ Bm.transpose(-1, -2))
        U = Q @ W[..., -r:]
    UtS = U.transpose(-1, -2) @ Sf
    return U @ UtS


def reference_stored_form(S, d, r):
    """StateFactor.forward(l, S, warm=False), explicit sink, side="right"."""
    Sf = S.float()
    n2 = (d * d).sum(-1).clamp_min(1e-12)
    a = torch.einsum("bhkv,hv->bhk", Sf, d) / n2[None, :, None]
    sink = a[:, :, :, None] * d[None, :, None, :]
    return sink + truncate_rank(Sf - sink, r)


class K31PrefillTruncationTest(unittest.TestCase):
    def _state(self, heads=4, dk=32, dv=32):
        torch.manual_seed(0)
        low = torch.randn(1, heads, dk, 6) @ torch.randn(1, heads, 6, dv)          # strong low-rank content
        d = torch.randn(heads, dv)
        sink = torch.randn(1, heads, dk, 1) * d[None, :, None, :] * 3.0
        return low + sink + 0.05 * torch.randn(1, heads, dk, dv), d

    def test_matches_reference_batch1(self):
        S, d = self._state()
        r = 8
        ref = reference_stored_form(S, d, r)
        g = torch.Generator().manual_seed(K31_SEED)
        omega = torch.randn(1, S.shape[1], S.shape[-1], r + OVERSAMPLE, generator=g)      # the reference's batch-1 draw
        s_sgl = S.transpose(-1, -2).contiguous()                                            # (B, H, V, K)
        a, U, W = factorize_prefill_k31(s_sgl, d, r, 16, torch.float32, omega)
        ours = d[None, :, :, None] * a[:, :, None, :] + W.transpose(-1, -2) @ U             # (B, H, V, K)
        self.assertTrue(torch.allclose(ours.transpose(-1, -2), ref, rtol=1e-5, atol=1e-5),
                        float((ours.transpose(-1, -2) - ref).abs().max()))
        self.assertEqual(int((U[:, :, r:] != 0).sum()), 0)

    def test_zero_sink_direction_is_pure_low_rank(self):
        S, _ = self._state()
        d = torch.zeros(S.shape[1], S.shape[-1])
        g = torch.Generator().manual_seed(K31_SEED)
        omega = torch.randn(1, S.shape[1], S.shape[-1], 16, generator=g)
        a, U, W = factorize_prefill_k31(S.transpose(-1, -2).contiguous(), d, 8, 16, torch.float32, omega)
        self.assertEqual(float(a.abs().sum()), 0.0)

    def test_requires_directions(self):
        S, d = self._state()
        with self.assertRaises(ValueError):
            factorize_prefill_k31(S.transpose(-1, -2), d, 8, 16, torch.float32, None)


if __name__ == "__main__":
    unittest.main()
