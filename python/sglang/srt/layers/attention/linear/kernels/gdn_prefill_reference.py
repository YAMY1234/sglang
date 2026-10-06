"""Frozen 21a0ea28 P-end truncation algebra for v3 serving.

The service uses a deterministic per-layer probe for batching/radix repeatability;
this is not the reference process's global RNG sequence. The complete K1 gate
validates that difference. Decode's r8/W8 kernels remain separately controlled.
"""
import contextlib
import math
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


def k31_graph_safe(device=None) -> bool:
    return K31_EIGH == "jacobi" or (K31_EIGH == "auto" and (device is None or torch.device(device).type == "cuda"))


# Shifted CholeskyQR (Fukaya et al. 2020): s = 11 (m n + n (n + 1)) u ||Y||_F^2 with fp32 u = 2^-24.
CHOLQR_SHIFT_CONST = 11.0
FP32_UNIT_ROUNDOFF = 2.0 ** -24
# Prompt-end states reach kappa(Y) ~ 1e8-1e11 (docs/170 s18.7), beyond fp32. The plain
# pass jitter sqrt(m) u trace(G) covers the fp32 Gram rounding error; 1e-6 mean(diag) did not.
_MIXED_FALLBACKS = {}


@contextlib.contextmanager
def _ieee_fp32_matmul():
    """The jitter bounds assume IEEE fp32 Gram products, under either precision API."""
    matmul = torch.backends.cuda.matmul
    if hasattr(matmul, "fp32_precision"):
        old = matmul.fp32_precision
        matmul.fp32_precision = "ieee"
        try:
            yield
        finally:
            matmul.fp32_precision = old
    else:
        old = matmul.allow_tf32
        matmul.allow_tf32 = False
        try:
            yield
        finally:
            matmul.allow_tf32 = old


def _cholqr_fp32(y, shifted):
    """One fp32 CholeskyQR pass on (..., m, n) -> (q, cholesky info); same column space as y."""
    m, n = y.shape[-2], y.shape[-1]
    with _ieee_fp32_matmul():
        g = y.transpose(-1, -2) @ y
    trace = g.diagonal(dim1=-2, dim2=-1).sum(-1)
    if shifted:
        jitter = CHOLQR_SHIFT_CONST * (m * n + n * (n + 1)) * FP32_UNIT_ROUNDOFF * trace
    else:
        jitter = math.sqrt(m) * FP32_UNIT_ROUNDOFF * trace
    g = g + (jitter + 1e-30)[..., None, None] * torch.eye(n, device=g.device, dtype=g.dtype)
    chol, info = torch.linalg.cholesky_ex(g)
    q = torch.linalg.solve_triangular(chol, y.transpose(-1, -2), upper=False).transpose(-1, -2)
    return q, info


def mixed_cholqr_fallbacks(device) -> int:
    """Matrices whose fp32 stage failed and took the fp64 input instead (host sync; call rarely)."""
    count = _MIXED_FALLBACKS.get(torch.device(device))
    return 0 if count is None else int(count.item())


def _mixed_fp32_stage(y):
    """Shifted + plain fp32 passes; any matrix that fails falls back to y for the fp64 pass."""
    first, info_first = _cholqr_fp32(y.float(), shifted=True)
    second, info_second = _cholqr_fp32(first, shifted=False)
    ok = (info_first == 0) & (info_second == 0) & torch.isfinite(second).all(-1).all(-1)
    count = _MIXED_FALLBACKS.get(y.device)
    if count is None:
        count = _MIXED_FALLBACKS[y.device] = torch.zeros((), dtype=torch.int64, device=y.device)
    # Device-side guard: no host sync, valid inside the prefill commit graph.
    count += (~ok).sum()
    return torch.where(ok[..., None, None], second.double(), y.double())



def _householder_fp32(y, *, repeats=1):
    """Reduced Householder Q; research only, model/numerical gates separate.

    Unlike the shifted Gram method, this does not damp small singular
    directions. Rank-deficient inputs may complete a different basis.
    """
    with _ieee_fp32_matmul():
        q = y.float()
        for _ in range(repeats):
            q = torch.linalg.qr(q, mode="reduced")[0]
    return q.to(y.dtype)

def _orth_cholqr2(y, *, mixed=False):
    """CholeskyQR2 in fp64. mixed: shifted fp32 pass, plain fp32 pass, then the same final fp64 pass."""
    if mixed:
        hh_mode = os.environ.get("SGLANG_GDN_K31_HOUSEHOLDER_FP32", "0")
        if hh_mode not in ("0", "1", "2", "3"):
            raise ValueError("K31_HOUSEHOLDER_FP32 expects 0/1/2/3")
        if hh_mode in ("2", "3"):
            return _householder_fp32(y, repeats=3 if hh_mode == "3" else 1)
        yd = _mixed_fp32_stage(y)
        if hh_mode == "1":
            return _householder_fp32(yd).to(y.dtype)
        passes = 1
    else:
        yd = y.double()
        passes = 2
    for _ in range(passes):
        g = yd.transpose(-1, -2) @ yd
        g = g + (1e-7 * g.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-30) * torch.eye(
            g.shape[-1], device=g.device, dtype=g.dtype)
        # Jitter keeps the Gram positive definite; skip the host-side info check.
        chol = torch.linalg.cholesky_ex(g)[0]
        yd = torch.linalg.solve_triangular(chol, yd.transpose(-1, -2), upper=False).transpose(-1, -2)
    return yd.to(y.dtype)


def _small_eigh_fp64(g, *, mixed_eigh=False):
    g = g.double()
    g = g + (1e-7 * g.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-30) * torch.eye(
        g.shape[-1], device=g.device, dtype=g.dtype)
    # lead #1557: one dispatcher for the small Gram eigh (power-of-two -> Jacobi kernel, other sizes -> padded
    # Jacobi, CPU / SGLANG_GDN_K31_EIGH=torch -> torch.linalg.eigh).
    from sglang.srt.duet.state_factor import small_eigh
    # k31 production admission #1588: same-threshold early exit. Keep the
    # generic small_eigh policy unchanged for other adapters; explicit 0 is
    # the fixed-twelve-sweep control for paired diagnostics.
    def production_jacobi(matrix):
        from .gdn_k31_eigh import eigh
        return eigh(matrix, early_exit=os.environ.get(
            "SGLANG_GDN_K31_EIGH_EARLY_EXIT", "1") == "1")
    solver = production_jacobi
    if mixed_eigh:
        from .gdn_k31_eigh_mixed import eigh as solver
    return small_eigh(g, override=K31_EIGH, _solver=solver)[1].to(torch.float32)


def factorize_prefill_k31(s, vbar, r, rmax, dtype, omega, *, mixed_eigh=False, mixed_cholqr=None):
    """s (B, HV, V, K) sglang layout, vbar (HV, V), omega (B, HV, V, r + 8) -> a (B, HV, K) fp32, U (B, HV, RMAX, K),
    W (B, HV, RMAX, V) in `dtype`, rows >= r zero; stored form = vbar a^T + W^T U (= sink + U_ref (U_ref^T C))."""
    if omega is None:
        raise ValueError("k31 prompt-final truncation needs the pool's fixed directions")
    if mixed_cholqr is None:
        from sglang.srt.environ import envs

        mixed_cholqr = envs.SGLANG_GDN_K31_CHOLQR_MIXED.get()
    s = s.float()
    vb = vbar.float()
    a = torch.einsum("bhvk,hv->bhk", s, vb) / vb.square().sum(-1).clamp_min(1e-12)[None, :, None]
    x = (s - vb[None, :, :, None] * a[:, :, None, :]).transpose(-1, -2)   # (B, HV, K, V) = reference S - sink (Dk x Dv)
    y = x @ omega.float()                                                  # (B, HV, K, m)
    for _ in range(K31_POWER):
        y = x @ _orth_cholqr2(x.transpose(-1, -2) @ _orth_cholqr2(y, mixed=mixed_cholqr), mixed=mixed_cholqr)
    q = _orth_cholqr2(y, mixed=mixed_cholqr)
    bm = q.transpose(-1, -2) @ x                                           # (B, HV, m, V)
    wr = _small_eigh_fp64(bm @ bm.transpose(-1, -2), mixed_eigh=mixed_eigh)                       # ascending energy
    u_ref = q @ wr[..., -r:]                                               # (B, HV, K, r)
    uts = u_ref.transpose(-1, -2) @ x                                      # (B, HV, r, V)
    u = torch.zeros(*s.shape[:2], rmax, s.shape[-1], device=s.device, dtype=dtype)
    w = torch.zeros(*s.shape[:2], rmax, s.shape[-2], device=s.device, dtype=dtype)
    u[:, :, :r], w[:, :, :r] = u_ref.transpose(-1, -2), uts
    return a, u, w
