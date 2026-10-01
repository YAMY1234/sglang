"""Explicit sink + warm-started rank-r truncation of a recurrent state (origin/minma/0913 twinstar/duet/state.py).

Shared by the Kimi line (twinstar_sgl/kimi_duet_math.py: `_orthonormalize`, `truncate_rank`, `project_state`;
right-side sink, S (B, H, Dk, Dv), d on Dv -- gated delta / KDA) and the Lightning line (models/lightning_duet/state.py:
`orthonormalize`, `factorize`; left-side sink, S (B, H, P, N), d on P -- Mamba-2, returns the factors and the warm basis).
`project_state(side="left")` also exposes Mamba's dense stored form without transposing the content.
Both sides are byte-equal to the pinned reference on CPU (test_duet_reference). `truncate_rank_exact` is the reference's ground truth
(state.py L22-36) for tests and probes.  Flash-Next's factored-pool version (`factorize_prefill_k31`,
`FactoredGDNPool.truncate_warm`) keeps its graph-safe fp64 Jacobi variant in the GDN kernels package.

Constants of the reference: OVERSAMPLE = 8, POWER = 1, Omega generator seed 0x5EED (drawn per call, batch-1 per request
under lead ruling #990), CholeskyQR2 in fp64 with jitter 1e-7 * mean diag + 1e-30, small Gram eigh in fp64 with the same jitter.
"""

from __future__ import annotations

import os

import torch

OVERSAMPLE = 8
POWER = 1
OMEGA_SEED = 0x5EED
EXACT_EIGH_DTYPE = torch.float64

# ----------------------------------------------------------------------------- small symmetric eigh (lead #1557)
# One dispatcher for every "small Gram" eigendecomposition on the serving path.  The graph-capturable fp64 Jacobi
# kernel (layers/attention/linear/kernels/gdn_k31_eigh.py) only takes power-of-two sizes; r16 + OVERSAMPLE = 24 is
# not one.  The nvfp4-perf line's answer -- pad to the next power of two with a diagonal block strictly below the
# matrix's Gershgorin lower bound and drop those eigenpairs -- is exact (block diagonal, padded eigenvalues below
# the spectrum) and passed its CPU numerical gate (eigenvalues 1e-13, top-16 projector 1e-10); it is the common
# implementation.  `SGLANG_GDN_K31_EIGH` keeps its meaning: auto (CUDA -> Jacobi, else torch) | jacobi | torch.
SMALL_EIGH_ENV = "SGLANG_GDN_K31_EIGH"
JACOBI_MAX_DIM = 64  # the kernel is one program per matrix with N**2 registers; above this use torch


def _is_power_of_two(n):
    return n >= 1 and (n & (n - 1)) == 0


def small_eigh_backend(dim, device=None, override=None):
    """Which solver `small_eigh` uses for a dim x dim fp64 symmetric matrix: "torch" | "jacobi" | "jacobi-padded".

    override: None reads SMALL_EIGH_ENV (auto|jacobi|torch).  CPU devices always resolve to torch.
    """
    mode = (override if override is not None else os.environ.get(SMALL_EIGH_ENV, "auto")) or "auto"
    if mode not in ("auto", "jacobi", "torch"):
        raise ValueError(f"{SMALL_EIGH_ENV} must be auto | jacobi | torch, got {mode!r}")
    on_cuda = device is not None and torch.device(device).type == "cuda"
    if mode == "torch" or (mode == "auto" and not on_cuda):
        return "torch"
    padded = dim if _is_power_of_two(dim) else 1 << (dim - 1).bit_length()
    if padded > JACOBI_MAX_DIM:
        if mode == "jacobi":
            raise ValueError(f"Jacobi small eigh supports dims up to {JACOBI_MAX_DIM}, got {dim}")
        return "torch"
    return "jacobi" if padded == dim else "jacobi-padded"


def _jacobi_eigh(gram):
    """The graph-safe fp64 Jacobi kernel (power-of-two sizes).  Imported lazily: it needs triton."""
    from sglang.srt.layers.attention.linear.kernels.gdn_k31_eigh import eigh
    return eigh(gram)


def pad_below_spectrum(gram):
    """Embed an n x n symmetric matrix into the next power-of-two size with a diagonal block strictly below its
    Gershgorin lower bound (nvfp4-perf line, #1019 follow-up).  The padded eigenpairs are then exactly the
    smallest ones and carry no weight on the original coordinates."""
    n = gram.shape[-1]
    padded_n = 1 << (n - 1).bit_length()
    padded = torch.zeros(*gram.shape[:-2], padded_n, padded_n, dtype=gram.dtype, device=gram.device)
    padded[..., :n, :n] = gram
    bound = gram.abs().sum(-1).amax(-1)
    diagonal = -(2 * bound + 1)
    padded[..., n:, n:] = diagonal[..., None, None] * torch.eye(padded_n - n, dtype=gram.dtype, device=gram.device)
    return padded


def small_eigh(gram, *, override=None, _solver=None):
    """Batched symmetric fp64 eigendecomposition -> (eigenvalues ascending, eigenvector columns), as
    torch.linalg.eigh.  Adapters call this instead of choosing a solver themselves (lead #1557).

    `_solver` injects the power-of-two kernel for CPU tests of the padding path."""
    if gram.dtype != torch.float64 or gram.shape[-1] != gram.shape[-2] or gram.shape[-1] < 1:
        raise ValueError("small_eigh takes square nonempty fp64 matrices")
    n = gram.shape[-1]
    backend = small_eigh_backend(n, gram.device, override)
    if backend == "torch":
        return torch.linalg.eigh(gram)
    solver = _solver or _jacobi_eigh
    if backend == "jacobi":
        return solver(gram.contiguous())
    d, z = solver(pad_below_spectrum(gram))
    return d[..., -n:].contiguous(), z[..., :n, -n:].contiguous()



def orthonormalize(y):
    """CholeskyQR2 in fp64 (state.py L39-50); returns y's dtype."""
    yd = y.double()
    for _ in range(2):
        gram = yd.transpose(-1, -2) @ yd
        jitter = (
            1e-7 * gram.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-30
        )
        gram = gram + jitter * torch.eye(
            gram.shape[-1], device=y.device, dtype=gram.dtype
        )
        lower = torch.linalg.cholesky(gram)
        yd = torch.linalg.solve_triangular(
            lower, yd.transpose(-1, -2), upper=False
        ).transpose(-1, -2)
    return yd.to(y.dtype)


_orthonormalize = orthonormalize  # Kimi-line name


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
    use_prev = (
        prev is not None
        and prev.shape == (batch, heads, n, rank)
        and prev.device == sf.device
    )
    omega = torch.randn(
        batch,
        heads,
        n,
        m - (rank if use_prev else 0),
        generator=generator,
        device=sf.device,
        dtype=torch.float32,
    )
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
    gram = gram + 1e-6 * gram.diagonal(dim1=-2, dim2=-1).mean(-1)[
        ..., None, None
    ] * torch.eye(gram.shape[-1], device=gram.device)
    _, u = torch.linalg.eigh(gram.to(EXACT_EIGH_DTYPE))
    u = u[..., -rank:].to(sf.dtype)
    return u @ (u.transpose(-1, -2) @ sf)


def project_state(state, direction, rank, prev=None, *, explicit=True, side="right"):
    """Project a dense state with the reference's explicit or implicit sink.

    For state (B, H, P, N), side="right" (KDA/GDN) takes direction (H, N)
    and sink = (S d / |d|^2) d^T. side="left" (Mamba) takes direction (H, P)
    and sink = d (S^T d / |d|^2)^T. Both truncate the original P x N content;
    transposing for a left sink would change Omega and the warm-start basis.
    Returns (stored form in state's dtype, right warm basis (B, H, N, r)).
    """
    if side not in ("right", "left"):
        raise ValueError(f"unknown DUET sink side: {side!r}")
    if rank <= 0:
        return state, None
    sf = state.float()
    if not explicit:
        content, warm = truncate_rank(sf, rank, prev)
        return content.to(state.dtype), warm
    direction = direction.to(sf.device, sf.dtype)
    n2 = (direction * direction).sum(-1).clamp_min(1e-12)
    if side == "left":
        coeff = torch.einsum("bhpn,hp->bhn", sf, direction) / n2[None, :, None]
        sink = direction[None, :, :, None] * coeff[:, :, None, :]
    else:
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
        batch,
        heads,
        n,
        width - (rank if warm else 0),
        generator=generator,
        device=state.device,
        dtype=torch.float32,
    )
    if warm:
        omega = torch.cat([previous.float(), omega], -1)
    y = content @ omega
    y = content @ orthonormalize(content.transpose(-1, -2) @ orthonormalize(y))
    q = orthonormalize(y)
    reduced = q.transpose(-1, -2) @ content
    gram = (reduced @ reduced.transpose(-1, -2)).double()
    eps = 1e-7 * gram.diagonal(dim1=-2, dim2=-1).mean(-1) + 1e-30
    gram = gram + eps[..., None, None] * torch.eye(
        gram.shape[-1], dtype=gram.dtype, device=gram.device
    )
    rotation = torch.linalg.eigh(gram)[1].float()
    left = q @ rotation[..., -rank:]
    right = left.transpose(-1, -2) @ content
    next_warm = orthonormalize(right.transpose(-1, -2))
    return coeff, left, right, next_warm
