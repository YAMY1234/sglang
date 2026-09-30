"""Explicit sink + warm-started rank-r truncation of a recurrent state (origin/minma/0913 twinstar/duet/state.py).

Moved verbatim from the Kimi line (twinstar_sgl/kimi_duet_math.py: `_orthonormalize`, `truncate_rank`, `project_state`;
right-side sink, S (B, H, Dk, Dv), d on Dv -- gated delta / KDA) and the Lightning line (models/lightning_duet/state.py:
`orthonormalize`, `factorize`; left-side sink, S (B, H, P, N), d on P -- Mamba-2, returns the factors and the warm basis).
Both are bitwise against the pinned reference on CPU.  `truncate_rank_exact` is the reference's ground truth
(state.py L22-36) for tests and probes.  Flash-Next's factored-pool version (`factorize_prefill_k31`,
`FactoredGDNPool.truncate_warm`) keeps its graph-safe fp64 Jacobi variant in the GDN kernels package.

Constants of the reference: OVERSAMPLE = 8, POWER = 1, Omega generator seed 0x5EED (drawn per call, batch-1 per request
under lead ruling #990), CholeskyQR2 in fp64 with jitter 1e-7 * mean diag + 1e-30, small Gram eigh in fp64 with the same jitter.
"""
from __future__ import annotations

import torch

OVERSAMPLE = 8
POWER = 1
OMEGA_SEED = 0x5EED
EXACT_EIGH_DTYPE = torch.float64


def orthonormalize(y):
    """CholeskyQR2 in fp64 (state.py L39-50); returns y's dtype."""
    yd = y.double()
    for _ in range(2):
        gram = yd.transpose(-1, -2) @ yd
        jitter = 1e-7 * gram.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-30
        gram = gram + jitter * torch.eye(gram.shape[-1], device=y.device, dtype=gram.dtype)
        lower = torch.linalg.cholesky(gram)
        yd = torch.linalg.solve_triangular(lower, yd.transpose(-1, -2), upper=False).transpose(-1, -2)
    return yd.to(y.dtype)


_orthonormalize = orthonormalize   # Kimi-line name


def truncate_rank(state, rank, prev=None):
    """Warm-started randomized subspace truncation (state.py L61-98) on the last two dims of `state` (B, H, P, N).
    Returns (rank-r state in fp32, right factor V (B, H, N, r) for the next warm start); (state, None) when nothing
    is truncated."""
    if rank <= 0 or rank >= min(state.shape[-2:]):
        return state, None
    sf = state.float()
    batch, heads, p, n = sf.shape
    m = min(rank + OVERSAMPLE, p, n)
    generator = torch.Generator(device=sf.device).manual_seed(OMEGA_SEED)
    use_prev = prev is not None and prev.shape == (batch, heads, n, rank) and prev.device == sf.device
    omega = torch.randn(batch, heads, n, m - (rank if use_prev else 0), generator=generator, device=sf.device, dtype=torch.float32)
    if use_prev:
        omega = torch.cat([prev.float(), omega], -1)
    y = sf @ omega
    y = sf @ _orthonormalize(sf.transpose(-1, -2) @ _orthonormalize(y))
    q = _orthonormalize(y)
    bm = q.transpose(-1, -2) @ sf
    gram = (bm @ bm.transpose(-1, -2)).double()
    jitter = 1e-7 * gram.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-30
    gram = gram + jitter * torch.eye(gram.shape[-1], device=sf.device, dtype=gram.dtype)
    vectors = torch.linalg.eigh(gram)[1].float()
    u = q @ vectors[..., -rank:]
    uts = u.transpose(-1, -2) @ sf
    return u @ uts, _orthonormalize(uts.transpose(-1, -2))


def truncate_rank_exact(state, rank):
    """Reference ground truth (state.py L22-36): best rank-r approximation via the fp64 Gram eigendecomposition."""
    if rank <= 0 or rank >= min(state.shape[-2], state.shape[-1]):
        return state
    sf = state.float()
    gram = sf @ sf.transpose(-1, -2)
    gram = gram + 1e-6 * gram.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * torch.eye(gram.shape[-1], device=gram.device)
    _, u = torch.linalg.eigh(gram.to(EXACT_EIGH_DTYPE))
    u = u[..., -rank:].to(sf.dtype)
    return u @ (u.transpose(-1, -2) @ sf)


def project_state(state, direction, rank, prev=None, *, explicit=True):
    """Right-side sink (gated delta / KDA): state (B, H, Dk, Dv), direction (H, Dv).  a = S d / |d|^2, sink = a d^T,
    stored = sink + truncate(S - sink); returns (stored form in state's dtype, warm basis)."""
    if rank <= 0:
        return state, None
    sf = state.float()
    if not explicit:
        content, warm = truncate_rank(sf, rank, prev)
        return content.to(state.dtype), warm
    direction = direction.to(sf.device, sf.dtype)
    n2 = (direction * direction).sum(-1).clamp_min(1e-12)
    coeff = torch.einsum("bhkv,hv->bhk", sf, direction) / n2[None, :, None]
    sink = coeff[..., :, None] * direction[None, :, None, :]
    content, warm = truncate_rank(sf - sink, rank, prev)
    return (sink + content).to(state.dtype), warm


def factorize_left(state, direction, rank, previous=None):
    """Left-side sink (Mamba-2), Lightning line's `factorize`: state (B, H, P, N), direction (H, P).
    Returns (sink coefficient a (B, H, N), left U (B, H, P, r), right U^T C (B, H, r, N), next warm basis (B, H, N, r));
    stored form = d a^T + left @ right."""
    state = state.float()
    d = direction.float()
    norm = d.square().sum(-1).clamp_min(1e-12)
    coeff = torch.einsum("bhpn,hp->bhn", state, d) / norm[None, :, None]
    content = state - d[None, :, :, None] * coeff[:, :, None, :]
    batch, heads, p, n = content.shape
    if not 0 < rank < min(p, n):
        raise ValueError("content rank must be in (0, min(P,N))")
    width = min(rank + OVERSAMPLE, p, n)
    generator = torch.Generator(device=state.device).manual_seed(OMEGA_SEED)
    warm = previous is not None
    if warm and previous.shape != (batch, heads, n, rank):
        raise ValueError("warm basis shape differs from slot geometry")
    omega = torch.randn(
        batch, heads, n, width - (rank if warm else 0),
        generator=generator, device=state.device, dtype=torch.float32,
    )
    if warm:
        omega = torch.cat([previous.float(), omega], -1)
    y = content @ omega
    y = content @ orthonormalize(content.transpose(-1, -2) @ orthonormalize(y))
    q = orthonormalize(y)
    reduced = q.transpose(-1, -2) @ content
    gram = (reduced @ reduced.transpose(-1, -2)).double()
    eps = 1e-7 * gram.diagonal(dim1=-2, dim2=-1).mean(-1) + 1e-30
    gram = gram + eps[..., None, None] * torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    rotation = torch.linalg.eigh(gram)[1].float()
    left = q @ rotation[..., -rank:]
    right = left.transpose(-1, -2) @ content
    next_warm = orthonormalize(right.transpose(-1, -2))
    return coeff, left, right, next_warm
