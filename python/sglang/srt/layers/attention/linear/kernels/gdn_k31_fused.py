"""Fused prompt-end k31 (docs/223 §9, round-2 R1; Triton first per lead 22:0x Q3).

Same algorithm as gdn_prefill_reference.factorize_prefill_k31 on the served recipe
(SGLANG_GDN_K31_CHOLQR_MIXED=1 + mixed eigh), one program per (row, layer, head) matrix:

  phase 1 (_k31_phase1): C = S - vbar a^T kept in registers; y = C^T omega; three mixed
    CholeskyQR orthonormalizations (shifted fp32 pass, plain fp32 pass, per-matrix fallback
    to the fp32 input with the shared fallback counter, then one fp64 pass with 1e-7 jitter);
    one power step; small Gram (bm bm^T) in fp64 with jitter, pad_below_spectrum and the amax
    normalization of fp32_vectors.
  jacobi  (gdn_k31_eigh_mixed._fp32_jacobi): the admitted kernel, launched exactly as fp32_vectors.
  phase 2 (_k31_finalize): correct_once (fp64 Newton, `safe` mask on the padded spectrum),
    stable ascending order, U = (q wr)^T and W = (C q wr)^T stored in `dtype`, rows >= r zero.

`a` is computed with the reference's torch ops (bitwise with the service path).
Matmuls on fp32 data use IEEE fp32 products (the service's precision); fp64 products are
explicit FMA loops. Opt-in: SGLANG_GDN_K31_FUSED=1 (gdn_factored_pool.factorize_layers).
"""
import math
import os

import torch
import triton
import triton.language as tl

from .gdn_k31_eigh_mixed import _fp32_jacobi

M_DIRECTIONS = 24          # r + oversample on the served recipe
M_PAD = 32                 # padded small dimension (= eigh N)
DIM = 128
CHOLQR_SHIFT = 11.0 * (DIM * M_DIRECTIONS + M_DIRECTIONS * (M_DIRECTIONS + 1)) * 2.0 ** -24
CHOLQR_PLAIN = math.sqrt(DIM) * 2.0 ** -24
SQRT_EPS32 = torch.finfo(torch.float32).eps ** .5


@triton.jit
def _col(A, j, DIMS: tl.constexpr):
    c = tl.arange(0, DIMS)
    return tl.sum(tl.where(c[None, :] == j, A, 0.), 1)


@triton.jit
def _row(A, i, DIMS: tl.constexpr):
    c = tl.arange(0, DIMS)
    return tl.sum(tl.where(c[:, None] == i, A, 0.), 0)


@triton.jit
def _cholesky(G, N: tl.constexpr, DIMS: tl.constexpr):
    """Lower Cholesky of the leading N x N block (lower triangle read); identity on the pad.
    Returns (L, failed pivots) -- a pivot that is not finite and > 0 marks a failed factorization (an inf
    pivot is where LAPACK / cuSOLVER turn NaN and report info > 0)."""
    c = tl.arange(0, DIMS)
    L = tl.zeros((DIMS, DIMS), dtype=G.dtype)
    A = G
    bad = 0
    for j in tl.static_range(N):
        column = _col(A, j, DIMS)
        d = tl.sum(tl.where(c == j, column, 0.), 0)
        bad += tl.where((d > 0.) & (d < float("inf")), 0, 1)
        pivot = tl.sqrt(d)
        lower = tl.where(c > j, column / pivot, 0.)
        L = tl.where(c[None, :] == j, tl.where(c == j, pivot, lower)[:, None], L)
        A = A - lower[:, None] * lower[None, :]
    L = tl.where((c[:, None] == c[None, :]) & (c[:, None] >= N), 1., L)
    return L, bad


@triton.jit
def _solve_lt(Y, L, N: tl.constexpr, DIMS: tl.constexpr):
    """Q with Q L^T = Y (column forward substitution, as solve_triangular(L, Y^T)); pad columns stay zero."""
    c = tl.arange(0, DIMS)
    Q = tl.zeros(Y.shape, dtype=Y.dtype)
    for i in tl.static_range(N):
        row = _row(L, i, DIMS)
        pivot = tl.sum(tl.where(c == i, row, 0.), 0)
        row = tl.where(c < i, row, 0.)
        q = (_col(Y, i, DIMS) - tl.sum(Q * row[None, :], 1)) / pivot
        Q = tl.where(c[None, :] == i, q[:, None], Q)
    return Q


@triton.jit
def _gram64(Y, N: tl.constexpr, DIMS: tl.constexpr):
    """Y^T Y for fp64 Y (rows, DIMS); exactly symmetric."""
    c = tl.arange(0, DIMS)
    G = tl.zeros((DIMS, DIMS), dtype=tl.float64)
    for j in tl.static_range(N):
        g = tl.sum(Y * _col(Y, j, DIMS)[:, None], 0)
        G = tl.where(c[:, None] == j, g[None, :], G)
    return G


@triton.jit
def _cholqr_fp32(Y, JITTER: tl.constexpr, N: tl.constexpr, DIMS: tl.constexpr):
    """gdn_prefill_reference._cholqr_fp32: IEEE fp32 Gram, jitter * trace on the diagonal."""
    c = tl.arange(0, DIMS)
    eye = c[:, None] == c[None, :]
    g = tl.dot(tl.trans(Y), Y, input_precision="ieee")
    trace = tl.sum(tl.sum(tl.where(eye & (c[:, None] < N), g, 0.), 1), 0)
    jitter = JITTER * trace + 1e-30
    g = tl.where(eye & (c[:, None] < N), g + jitter, g)
    L, bad = _cholesky(g, N, DIMS)
    return _solve_lt(Y, L, N, DIMS), bad


@triton.jit
def _orth_mixed(Y, CNT_ptr, SHIFT: tl.constexpr, PLAIN: tl.constexpr, N: tl.constexpr, DIMS: tl.constexpr):
    """gdn_prefill_reference._orth_cholqr2(y, mixed=True) for one fp32 (rows, DIMS) matrix -> (fp32 Q, ok)."""
    c = tl.arange(0, DIMS)
    eye = c[:, None] == c[None, :]
    first, bad_first = _cholqr_fp32(Y, SHIFT, N, DIMS)
    second, bad_second = _cholqr_fp32(first, PLAIN, N, DIMS)
    finite = (second == second) & (tl.abs(second) < float("inf")) & (first == first) & (tl.abs(first) < float("inf"))
    nonfinite = tl.sum(tl.sum(tl.where(finite, 0, 1), 1), 0)
    ok = (bad_first == 0) & (bad_second == 0) & (nonfinite == 0)
    tl.atomic_add(CNT_ptr, tl.where(ok, 0, 1).to(tl.int64), mask=~ok)
    yd = tl.where(ok, second, Y).to(tl.float64)
    g = _gram64(yd, N, DIMS)
    mean = tl.sum(tl.sum(tl.where(eye & (c[:, None] < N), g, 0.), 1), 0) / N
    g = tl.where(eye & (c[:, None] < N), g + (1e-7 * mean + 1e-30), g)
    g = tl.where(eye & (c[:, None] >= N), 1., g)
    L, _ = _cholesky(g, N, DIMS)
    return _solve_lt(yd, L, N, DIMS).to(tl.float32), ok


@triton.jit
def _load_content(S_ptr, VB_ptr, A_ptr, pid, lh, D: tl.constexpr):
    v = tl.arange(0, D)
    k = tl.arange(0, D)
    s = tl.load(S_ptr + pid * D * D + v[:, None] * D + k[None, :])
    vb = tl.load(VB_ptr + lh * D + v)
    a = tl.load(A_ptr + pid * D + k)
    return s - vb[:, None] * a[None, :]                        # C (V, K) = (S - sink); x = C^T


@triton.jit
def _k31_phase1(S_ptr, VB_ptr, A_ptr, OM_ptr, Q_ptr, G64_ptr, GN_ptr, CNT_ptr, DBG_ptr,
                LH, H, om_sb, om_sh,
                D: tl.constexpr, M: tl.constexpr, MP: tl.constexpr,
                SHIFT: tl.constexpr, PLAIN: tl.constexpr, DEBUG: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    row = pid // LH
    lh = pid % LH
    head = lh % H
    v = tl.arange(0, D)
    k = tl.arange(0, D)
    c = tl.arange(0, MP)
    eye = c[:, None] == c[None, :]
    C = _load_content(S_ptr, VB_ptr, A_ptr, pid, lh, D)
    om = tl.load(OM_ptr + row * om_sb + head * om_sh + v[:, None] * M + c[None, :], mask=c[None, :] < M, other=0.)
    y = tl.dot(tl.trans(C), om, input_precision="ieee")      # x @ omega       (K, MP)
    o1, ok1 = _orth_mixed(y, CNT_ptr, SHIFT, PLAIN, M, MP)
    z = tl.dot(C, o1, input_precision="ieee")                # x^T @ orth(y)   (V, MP)
    o2, ok2 = _orth_mixed(z, CNT_ptr, SHIFT, PLAIN, M, MP)
    y2 = tl.dot(tl.trans(C), o2, input_precision="ieee")     # power step      (K, MP)
    q, ok3 = _orth_mixed(y2, CNT_ptr, SHIFT, PLAIN, M, MP)
    tl.store(Q_ptr + pid * D * MP + k[:, None] * MP + c[None, :], q)
    t = tl.dot(C, q, input_precision="ieee")                 # bm^T            (V, MP)
    g = tl.dot(tl.trans(t), t, input_precision="ieee").to(tl.float64)
    real = c < M
    mean = tl.sum(tl.sum(tl.where(eye & real[:, None], g, 0.), 1), 0) / M
    g = tl.where(eye & real[:, None], g + (1e-7 * mean + 1e-30), g)
    # pad_below_spectrum: pad diagonal strictly below the Gershgorin lower bound
    bound = tl.max(tl.sum(tl.where(real[:, None] & real[None, :], tl.abs(g), 0.), 1), 0)
    g = tl.where(eye & ~real[:, None], -(2 * bound + 1), g)
    tl.store(G64_ptr + pid * MP * MP + c[:, None] * MP + c[None, :], g)
    # fp32_vectors: normalize by amax before the fp32 cast
    scale = tl.maximum(tl.max(tl.max(tl.abs(g), 1), 0), 1e-300)
    tl.store(GN_ptr + pid * MP * MP + c[:, None] * MP + c[None, :], (g / scale).to(tl.float32))
    if DEBUG:
        base = DBG_ptr + pid * 5 * D * MP + k[:, None] * MP + c[None, :]
        tl.store(base, y)
        tl.store(base + D * MP, o1)
        tl.store(base + 2 * D * MP, z)
        tl.store(base + 3 * D * MP, o2)
        tl.store(base + 4 * D * MP, y2)


@triton.jit
def _mm64(A, B, DIMS: tl.constexpr):
    acc = tl.zeros((DIMS, DIMS), dtype=tl.float64)
    for k in tl.static_range(DIMS):
        acc += _col(A, k, DIMS)[:, None] * _row(B, k, DIMS)[None, :]
    return acc


@triton.jit
def _k31_finalize(S_ptr, VB_ptr, A_ptr, Q_ptr, G64_ptr, Z_ptr, U_ptr, W_ptr, DBG_ptr,
                  LH,
                  D: tl.constexpr, M: tl.constexpr, MP: tl.constexpr, R: tl.constexpr, RMAX: tl.constexpr,
                  SQRT_EPS: tl.constexpr, DEBUG: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    lh = pid % LH
    c = tl.arange(0, MP)
    eye = c[:, None] == c[None, :]
    g = tl.load(G64_ptr + pid * MP * MP + c[:, None] * MP + c[None, :])
    q = tl.load(Z_ptr + pid * MP * MP + c[:, None] * MP + c[None, :]).to(tl.float64)
    # gdn_k31_eigh_mixed.correct_once
    gq = _mm64(g, q, MP)
    metric = _mm64(tl.trans(q), q, MP)
    projected = _mm64(tl.trans(q), gq, MP)
    values = tl.sum(tl.where(eye, projected, 0.), 1) / tl.sum(tl.where(eye, metric, 0.), 1)
    error = metric - tl.where(eye, 1., 0.)
    gaps = values[None, :] - values[:, None]
    scale = tl.maximum(tl.max(tl.abs(values), 0), 1e-300)
    safe = tl.abs(gaps) > SQRT_EPS * scale
    residual = projected - error * values[None, :]
    update = tl.where(safe, residual / tl.where(safe, gaps, 1.), -.5 * error)
    corrected = q + _mm64(q, update, MP)
    corrected = _mm64(corrected, .5 * (3 * tl.where(eye, 1., 0.) - _mm64(tl.trans(corrected), corrected, MP)), MP)
    corrected = corrected / tl.maximum(tl.sqrt(tl.sum(corrected * corrected, 0)), 1e-300)[None, :]
    values = tl.sum(corrected * _mm64(g, corrected, MP), 0)
    # stable ascending argsort (ties by index)
    rank = tl.sum(((values[None, :] < values[:, None])
                   | ((values[None, :] == values[:, None]) & (c[None, :] < c[:, None]))).to(tl.int32), 1)
    order = tl.sum(tl.where(rank[:, None] == c[None, :], c[:, None], 0), 0)
    vectors = tl.gather(corrected, tl.broadcast_to(order[None, :], (MP, MP)), 1)
    if DEBUG:
        tl.store(DBG_ptr + pid * (MP + 1) * MP + c[:, None] * MP + c[None, :], vectors)
        tl.store(DBG_ptr + pid * (MP + 1) * MP + MP * MP + c, tl.gather(values, order, 0))
    # small_eigh unpad (rows < M) and the top-R columns; fp32 as the reference's .to(float32)
    wr = tl.where((c[:, None] < M) & (c[None, :] >= MP - R), vectors, 0.).to(tl.float32)
    k = tl.arange(0, D)
    qf = tl.load(Q_ptr + pid * D * MP + k[:, None] * MP + c[None, :])
    u = tl.dot(qf, wr, input_precision="ieee")                # u_ref in columns MP-R.. (K, MP)
    C = _load_content(S_ptr, VB_ptr, A_ptr, pid, lh, D)
    w = tl.dot(C, u, input_precision="ieee")                  # (u_ref^T x)^T   (V, MP)
    keep = c[None, :] >= MP - R
    rows = c[None, :] - (MP - R)
    tl.store(U_ptr + pid * RMAX * D + rows * D + k[:, None], u.to(U_ptr.dtype.element_ty), mask=keep)
    tl.store(W_ptr + pid * RMAX * D + rows * D + k[:, None], w.to(W_ptr.dtype.element_ty), mask=keep)
    if RMAX > R:
        zr = R + tl.arange(0, RMAX - R)
        zero = tl.zeros((RMAX - R, D), dtype=U_ptr.dtype.element_ty)
        tl.store(U_ptr + pid * RMAX * D + zr[:, None] * D + k[None, :], zero)
        tl.store(W_ptr + pid * RMAX * D + zr[:, None] * D + k[None, :], zero)


def fused_supported(s, vbar, omega, r, rmax, dtype):
    """Geometry the kernels are written for (the R3-a served shapes)."""
    return (s.dim() == 4 and s.shape[-1] == DIM and s.shape[-2] == DIM and vbar.shape[-1] == DIM
            and omega is not None and omega.dim() == 4 and omega.shape[-1] == M_DIRECTIONS
            and omega.shape[-2] == DIM and r == 16 and rmax >= r and (rmax - r) in (0, 16, 32)
            and dtype in (torch.float16, torch.float32))


def _fallback_counter(device):
    # gdn_prefill_reference._mixed_fp32_stage keys the shared counter by the tensor's device
    from . import gdn_prefill_reference as ref
    count = ref._MIXED_FALLBACKS.get(device)
    if count is None:
        count = ref._MIXED_FALLBACKS[device] = torch.zeros((), dtype=torch.int64, device=device)
    return count


def factorize_k31_fused(s, vbar, omega, heads, r, rmax, dtype, *, warps=None, debug=None):
    """s (B, LH, V, K) with LH = layers * heads, vbar (LH, V), omega (B or 1, heads, V, 24) -> a, U, W as
    factorize_prefill_k31(mixed_eigh=True, mixed_cholqr=True) on the same inputs (omega shared per layer).
    debug: dict that receives the per-matrix stage tensors (tests)."""
    if not fused_supported(s, vbar, omega, r, rmax, dtype):
        raise ValueError("k31 fused: unsupported geometry")
    if warps is None:  # layout round of the micro (docs/223 §9.2); phase 1, finalize
        warps = tuple(int(x) for x in os.environ.get("SGLANG_GDN_K31_FUSED_WARPS", "4,4").split(","))
    s = s.float().contiguous()
    vb = vbar.float()
    # reference ops, bitwise with the service path
    a = torch.einsum("bhvk,hv->bhk", s, vb) / vb.square().sum(-1).clamp_min(1e-12)[None, :, None]
    b, lh = s.shape[:2]
    n = b * lh
    vb = vb.contiguous()
    a = a.contiguous()
    om = omega.float()
    if om.stride(-1) != 1 or om.stride(-2) != M_DIRECTIONS:
        om = om.contiguous()
    om_sb = 0 if om.shape[0] == 1 else om.stride(0)
    dev = s.device
    q = torch.empty(n, DIM, M_PAD, device=dev, dtype=torch.float32)
    g64 = torch.empty(n, M_PAD, M_PAD, device=dev, dtype=torch.float64)
    gn = torch.empty(n, M_PAD, M_PAD, device=dev, dtype=torch.float32)
    d = torch.empty(n, M_PAD, device=dev, dtype=torch.float32)
    z = torch.empty(n, M_PAD, M_PAD, device=dev, dtype=torch.float32)
    sweeps = torch.empty(n, device=dev, dtype=torch.int32)
    u = torch.empty(b, lh, rmax, DIM, device=dev, dtype=dtype)
    w = torch.empty(b, lh, rmax, DIM, device=dev, dtype=dtype)
    dbg1 = torch.empty(n, 5, DIM, M_PAD, device=dev, dtype=torch.float32) if debug is not None else q
    dbg2 = torch.empty(n, M_PAD + 1, M_PAD, device=dev, dtype=torch.float64) if debug is not None else g64
    _k31_phase1[(n,)](s, vb, a, om, q, g64, gn, _fallback_counter(dev), dbg1, lh, heads, om_sb, om.stride(1),
                      D=DIM, M=M_DIRECTIONS, MP=M_PAD, SHIFT=CHOLQR_SHIFT, PLAIN=CHOLQR_PLAIN,
                      DEBUG=debug is not None, num_warps=warps[0])
    # the admitted fp32 Jacobi, launched as gdn_k31_eigh_mixed.fp32_vectors does
    _fp32_jacobi[(n,)](gn, d, z, sweeps, N=32, SWEEPS=12, EARLY_EXIT=True, num_warps=1)
    _k31_finalize[(n,)](s, vb, a, q, g64, z, u, w, dbg2, lh,
                        D=DIM, M=M_DIRECTIONS, MP=M_PAD, R=r, RMAX=rmax, SQRT_EPS=SQRT_EPS32,
                        DEBUG=debug is not None, num_warps=warps[1])
    if debug is not None:
        debug.update(q=q, g64=g64, gn=gn, z=z, stages=dbg1, eig=dbg2)
    return a, u, w
