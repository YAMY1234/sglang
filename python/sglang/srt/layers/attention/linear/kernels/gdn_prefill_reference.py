"""Frozen 21a0ea28 P-end truncation algebra for v3 serving.

The service uses a deterministic per-layer probe for batching/radix repeatability;
this is not the reference process's global RNG sequence. The complete K1 gate
validates that difference. Decode's r8/W8 kernels remain separately controlled.
"""
import os

import torch


def _orth_ns(y):
    # Frozen 21a0ea28 twinstar/probes/gdnstep.py: eight Newton-Schulz rounds.
    g = y.transpose(-1, -2) @ y
    eye = torch.eye(g.shape[-1], device=g.device, dtype=g.dtype)
    trace = g.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
    gn = g / trace.clamp_min(1e-30) + 1e-6 * eye
    a, z = gn, eye.expand_as(gn)
    for _ in range(8):
        t = .5 * (3 * eye - z @ a)
        a, z = a @ t, t @ z
    return y @ (z / trace.clamp_min(1e-30).sqrt())


def factorize_prefill_reference(s, vbar, r, rmax, dtype, iters=2, oversample=8,
                    method="iter", omega=None):
    """Paper's P-end algebra, deterministic existing service probes.

    This isolates the truncation algorithm, not the HF RNG sequence or decode
    truncation. Diagnostic only until complete fixed-threshold K1 validation.
    """
    if method != "iter":
        raise ValueError("paper diagnostic only supports prefill iteration")
    s = s.float()
    vb = vbar.float()
    a = torch.einsum("bhvk,hv->bhk", s, vb) / vb.square().sum(-1).clamp_min(1e-30)[None, :, None]
    c = s - vb[None, :, :, None] * a[:, :, None, :]
    x = c.transpose(-1, -2)
    if omega is None:
        gen = torch.Generator(device=s.device).manual_seed(0)
        omega = torch.randn(*s.shape[:2], s.shape[-2], r+oversample,
                            device=s.device, dtype=torch.float32, generator=gen)
    y = x @ omega
    for _ in range(iters):
        y = x @ _orth_ns(x.transpose(-1, -2) @ _orth_ns(y))
    q = _orth_ns(y)
    b = q.transpose(-1, -2) @ x
    if oversample:
        g = (b @ b.transpose(-1, -2)).double()
        jitter = 1e-7*(g.diagonal(dim1=-2, dim2=-1).sum(-1)/g.shape[-1])[..., None, None] + 1e-30
        g = g + jitter*torch.eye(g.shape[-1], device=g.device, dtype=g.dtype)
        z = torch.linalg.eigh(g)[1][..., -r:].flip(-1).float()
        b, q = z.transpose(-1, -2) @ b, q @ z
    u = torch.zeros(*s.shape[:2], rmax, s.shape[-1], device=s.device, dtype=dtype)
    w = torch.zeros(*s.shape[:2], rmax, s.shape[-2], device=s.device, dtype=dtype)
    u[:, :, :r], w[:, :, :r] = q.transpose(-1, -2), b
    return a, u, w


# ---------------------------------------------------------------------------- k31-r4096-u (#873)
# Mingyuan's unified prompt-final truncation (origin/minma/0913 twinstar/duet/state.py: StateFactor.forward with
# warm=False -> truncate_rank), reproduced operation for operation on the sglang (V, K) layout: explicit sink
# a = S v_bar / |v_bar|^2 (clamp 1e-12), content S - sink truncated by a cold randomized subspace iteration with
# m = r + 8 fixed directions, POWER = 1 re-orthonormalised power step, CholeskyQR2 in fp64 (jitter 1e-7 of the mean
# diagonal + 1e-30) and a jittered fp64 eigendecomposition of the small (m x m) Gram; U = Q W[..., -r:].
# The fixed directions come from the caller (the pool's batch-1 draw with seed 0x5EED, see FactoredGDNPool).
K31_SEED = 0x5EED
K31_OVERSAMPLE = 8
K31_POWER = 1


# CUDA uses Jacobi to avoid the host synchronization in torch.linalg.eigh.
# The torch override retains the eager reference for numerical comparisons.
K31_EIGH = os.environ.get("SGLANG_GDN_K31_EIGH", "auto")
_K31_FUSED_ORTH_SETTING = os.environ.get("SGLANG_PFACTOR4_FUSED_ORTH", "0")
if _K31_FUSED_ORTH_SETTING not in ("0", "1"):
    raise ValueError("SGLANG_PFACTOR4_FUSED_ORTH must be 0 or 1")
K31_FUSED_ORTH = _K31_FUSED_ORTH_SETTING == "1"
K31_JACOBI_LAUNCH = os.environ.get("SGLANG_PFACTOR4_JACOBI_LAUNCH", "1")
if K31_JACOBI_LAUNCH not in ("1", "shape"):
    raise ValueError("SGLANG_PFACTOR4_JACOBI_LAUNCH must be 1 or shape")


def k31_graph_safe(device=None) -> bool:
    return K31_EIGH == "jacobi" or (K31_EIGH == "auto" and (device is None or torch.device(device).type == "cuda"))


def _orth_cholqr2(y):
    if K31_FUSED_ORTH and y.is_cuda and y.dtype == torch.float32 and y.shape[-2:] == (128, 16):
        from .gdn_k31_orth import orth_cholqr2

        return orth_cholqr2(y)
    return _orth_cholqr2_reference(y)


def _orth_cholqr2_reference(y):
    yd = y.double()
    for _ in range(2):
        g = yd.transpose(-1, -2) @ yd
        g = g + (1e-7 * g.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-30) * torch.eye(
            g.shape[-1], device=g.device, dtype=g.dtype)
        # Jitter keeps the Gram positive definite; skip the host-side info check.
        chol = torch.linalg.cholesky_ex(g)[0]
        yd = torch.linalg.solve_triangular(chol, yd.transpose(-1, -2), upper=False).transpose(-1, -2)
    return yd.to(y.dtype)


def _small_eigh_fp64(g):
    g = g.double()
    g = g + (1e-7 * g.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-30) * torch.eye(
        g.shape[-1], device=g.device, dtype=g.dtype)
    if k31_graph_safe(g.device):
        from .gdn_k31_eigh import eigh, shape_warps
        width = shape_warps(g) if K31_JACOBI_LAUNCH == "shape" else 1
        return eigh(g, num_warps=width)[1].to(torch.float32)
    return torch.linalg.eigh(g)[1].to(torch.float32)


def factorize_prefill_k31(s, vbar, r, rmax, dtype, omega):
    """s (B, HV, V, K) sglang layout, vbar (HV, V), omega (B, HV, V, r + 8) -> a (B, HV, K) fp32, U (B, HV, RMAX, K),
    W (B, HV, RMAX, V) in `dtype`, rows >= r zero; stored form = vbar a^T + W^T U (= sink + U_ref (U_ref^T C))."""
    if omega is None:
        raise ValueError("k31 prompt-final truncation needs the pool's fixed directions")
    s = s.float()
    vb = vbar.float()
    a = torch.einsum("bhvk,hv->bhk", s, vb) / vb.square().sum(-1).clamp_min(1e-12)[None, :, None]
    x = (s - vb[None, :, :, None] * a[:, :, None, :]).transpose(-1, -2)   # (B, HV, K, V) = reference S - sink (Dk x Dv)
    y = x @ omega.float()                                                  # (B, HV, K, m)
    for _ in range(K31_POWER):
        y = x @ _orth_cholqr2(x.transpose(-1, -2) @ _orth_cholqr2(y))
    q = _orth_cholqr2(y)
    bm = q.transpose(-1, -2) @ x                                           # (B, HV, m, V)
    wr = _small_eigh_fp64(bm @ bm.transpose(-1, -2))                       # ascending energy
    u_ref = q @ wr[..., -r:]                                               # (B, HV, K, r)
    uts = u_ref.transpose(-1, -2) @ x                                      # (B, HV, r, V)
    u = torch.zeros(*s.shape[:2], rmax, s.shape[-1], device=s.device, dtype=dtype)
    w = torch.zeros(*s.shape[:2], rmax, s.shape[-2], device=s.device, dtype=dtype)
    u[:, :, :r], w[:, :, :r] = u_ref.transpose(-1, -2), uts
    return a, u, w
