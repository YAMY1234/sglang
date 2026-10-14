"""Mamba-2 direct factor recurrence, with a left sink and device W-step expiry.

S[P,N] = direction[P] * a[N] + U[RMAX,P].T @ V[RMAX,N].
One CTA owns each (slot, head); callers must supply unique nonnegative slots.
Negative slots are padding. Output observes the updated, *untruncated* state;
expiry is fused after readout, before the next state reader.
The append and FP32 MGS2 subspace iteration use the same GDN K1/K2 helpers.
There is no GDN delta correction. FP32 factors; no dense state in the step.
"""
import torch
import triton
import triton.language as tl

from .gdn_factored import (
    _gram_wwt, _small_matmul_cols, _mgs, _project_rows,
    TRUNC_ITERS, MGS_REL_TOL,
)


@triton.jit
def _compress(u, v, r, RM: tl.constexpr, R: tl.constexpr,
              FULL: tl.constexpr, ITERS: tl.constexpr, REL_TOL: tl.constexpr):
    # Exactly GDN K2's FP32 diagonal warm start and MGS2 subspace iteration.
    gram = _gram_wwt(v, r, RM, FULL)
    rows, keep = r < FULL, r < R
    d = tl.where(rows, tl.sum(tl.where(r[:, None] == r[None, :], gram, 0.), 1), -1.)
    better = (d[None, :] > d[:, None]) | ((d[None, :] == d[:, None]) & (r[None, :] < r[:, None]))
    order = tl.sum(better.to(tl.int32), 1)
    z = tl.where((order[:, None] == r[None, :]) & keep[None, :] & rows[:, None], 1., 0.)
    for _ in tl.static_range(ITERS):
        z = _small_matmul_cols(gram, z, r, RM, R)
        z = _mgs(z, r, R, 2, REL_TOL)
    return _project_rows(z, u, r, RM, R), _project_rows(z, v, r, RM, R)


@triton.jit
def _step(X, B, C, DT, DECAY, D, BIAS, DIR, FA, U, V, COUNT, SLOTS, Y,
          H: tl.constexpr, G: tl.constexpr, P: tl.constexpr, N: tl.constexpr,
          RM: tl.constexpr, XS: tl.constexpr, BS: tl.constexpr, CS: tl.constexpr,
          DTS: tl.constexpr, DS: tl.constexpr, IS: tl.constexpr,
          NATIVE_DT: tl.constexpr, FUSED_TRUNC: tl.constexpr,
          R: tl.constexpr, FULL: tl.constexpr, ITERS: tl.constexpr, REL_TOL: tl.constexpr):
    row, h = tl.program_id(0), tl.program_id(1)
    slot = tl.load(SLOTS + row * IS).to(tl.int64)
    p, n, r = tl.arange(0, P), tl.arange(0, N), tl.arange(0, RM)
    yp = Y + (row * H + h) * P + p
    if slot < 0:
        tl.store(yp, 0.)
        return
    sh = slot * H + h
    cnt = tl.load(COUNT + sh)
    tl.device_assert((cnt >= 0) & (cnt < RM), "Mamba factor count out of bounds")
    group = h // (H // G)
    x = tl.load(X + row * XS + h * P + p).to(tl.float32)
    b = tl.load(B + row * BS + group * N + n).to(tl.float32)
    c = tl.load(C + row * CS + group * N + n).to(tl.float32)
    dt = tl.load(DT + row * DTS + h).to(tl.float32)
    if NATIVE_DT:
        dt += tl.load(BIAS + h).to(tl.float32)
        dt = tl.where(dt <= 20., tl.log(1. + tl.exp(dt)), dt)
        decay = tl.exp(dt * tl.load(DECAY + h).to(tl.float32))
    else:
        decay = tl.load(DECAY + row * DS + h).to(tl.float32)
    direction = tl.load(DIR + h * P + p)
    sx = dt * x
    coeff = tl.sum(sx * direction, 0) / tl.maximum(tl.sum(direction * direction, 0), 1.e-12)
    residual = sx - coeff * direction
    a = decay * tl.load(FA + sh * N + n) + coeff * b
    tl.store(FA + sh * N + n, a)
    up = U + sh * RM * P + r[:, None] * P + p[None, :]
    vp = V + sh * RM * N + r[:, None] * N + n[None, :]
    u = tl.load(up, r[:, None] < cnt, 0.)
    v = tl.load(vp, r[:, None] < cnt, 0.)
    proj = tl.sum(u * residual[None, :], 1)
    tail = residual - tl.sum(u * proj[:, None], 0)
    # Always reorthogonalize: non-unit x and rank-deficient inputs are valid.
    correction = tl.sum(u * tail[None, :], 1)
    tail -= tl.sum(u * correction[:, None], 0)
    proj += correction
    norm = tl.sqrt(tl.sum(tail * tail, 0))
    base_norm = tl.sqrt(tl.sum(residual * residual, 0))
    keep = norm > 1.e-6 * base_norm
    new_u = tl.where(keep, tail / tl.maximum(norm, 1.e-30), 0.)
    proj = tl.where(r == cnt, tl.where(keep, norm, 0.), proj)
    new_v = decay * v + proj[:, None] * b[None, :]
    read = tl.sum(new_v * c[None, :], 1)
    y = direction * tl.sum(a * c, 0) + tl.sum(u * read[:, None], 0)
    y += new_u * tl.sum(tl.where(r == cnt, read, 0.), 0)
    y += tl.load(D + h).to(tl.float32) * x
    tl.store(yp, y.to(Y.dtype.element_ty))
    if FUSED_TRUNC and cnt + 1 >= FULL:
        full_u = tl.where((r == cnt)[:, None], new_u[None, :], u)
        un, vn = _compress(full_u, new_v, r, RM, R, FULL, ITERS, REL_TOL)
        tl.store(up, un, r[:, None] < R)
        tl.store(vp, vn, r[:, None] < R)
        tl.store(COUNT + sh, R)
    else:
        tl.store(vp, new_v, r[:, None] <= cnt)
        tl.store(U + sh * RM * P + cnt * P + p, new_u)
        tl.store(COUNT + sh, cnt + 1)


@triton.jit
def _truncate(U, V, COUNT, SLOTS, H: tl.constexpr, P: tl.constexpr,
              N: tl.constexpr, RM: tl.constexpr, R: tl.constexpr,
              FULL: tl.constexpr, IS: tl.constexpr, ITERS: tl.constexpr, REL_TOL: tl.constexpr,
              LUS: tl.constexpr = 0, LVS: tl.constexpr = 0, LCS: tl.constexpr = 0):
    row, h = tl.program_id(0), tl.program_id(1)
    layer = tl.program_id(2).to(tl.int64)
    U += layer * LUS
    V += layer * LVS
    COUNT += layer * LCS
    slot = tl.load(SLOTS + row * IS).to(tl.int64)
    if slot < 0:
        return
    sh = slot * H + h
    if tl.load(COUNT + sh) < FULL:
        return
    r, p, n = tl.arange(0, RM), tl.arange(0, P), tl.arange(0, N)
    up = U + sh * RM * P + r[:, None] * P + p[None, :]
    vp = V + sh * RM * N + r[:, None] * N + n[None, :]
    v = tl.load(vp, r[:, None] < FULL, 0.)
    u = tl.load(up, r[:, None] < FULL, 0.)
    un, vn = _compress(u, v, r, RM, R, FULL, ITERS, REL_TOL)
    tl.store(up, un, r[:, None] < R)
    tl.store(vp, vn, r[:, None] < R)
    tl.store(COUNT + sh, R)


def truncate(u, v, count, slots, rank=16, window=16):
    """Device-side expiry; safe for changing phases/slots during graph replay."""
    _, heads, capacity, p = u.shape
    _truncate[(slots.numel(), heads)](
        u, v, count, slots, heads, p, v.shape[-1], capacity,
        rank, rank + window, slots.stride(0), TRUNC_ITERS, MGS_REL_TOL, num_warps=4,
        enable_fp_fusion=False,
    )


def truncate_layers(u, v, count, slots, rank=16, window=16):
    """One post-order launch for every layer, using GDN's FP32 MGS helpers.

    Caller orders after all producers, and joins before the next state reader.
    Device counts choose expiry; no host phase check is captured.
    """
    layers, size, heads, capacity, p = u.shape
    assert v.shape[:4] == u.shape[:4] and count.shape == (layers, size, heads)
    assert all(t.is_contiguous() for t in (u, v, count))
    _truncate[(slots.numel(), heads, layers)](
        u, v, count, slots, heads, p, v.shape[-1], capacity,
        rank, rank + window, slots.stride(0), TRUNC_ITERS, MGS_REL_TOL,
        u.stride(0), v.stride(0), count.stride(0), num_warps=4,
        enable_fp_fusion=False,
    )


def step(x, b, c, dt, decay, d, direction, a, u, v, count, slots,
         *, rank=16, window=16, out=None, do_truncate=True, dt_bias=None):
    """Inputs x[B,H,P], B/C[B,G,N], dt/decay[B,H], D[H].

    dt=softplus(raw_dt+bias), decay=exp(dt*A) are supplied by the caller to
    preserve the bridge's activation rounding in tests. With dt_bias, dt is raw
    and decay is A[H]: the native SSU softplus/exp is fused in this kernel.
    State buffers are contiguous;
    x/B/C may be views into a packed convolution output (row stride supported).
    """
    batch, heads, p = x.shape
    n, groups = b.shape[-1], b.shape[1]
    capacity = u.shape[-2]
    assert p in (32, 64, 128) and n in (64, 128, 256)
    assert heads % groups == 0 and 0 < rank < min(p, n)
    assert rank + window <= capacity and capacity & (capacity - 1) == 0
    assert u.shape == (a.shape[0], heads, capacity, p)
    assert v.shape == (a.shape[0], heads, capacity, n)
    assert count.shape == (a.shape[0], heads) and slots.shape == (batch,)
    assert all(t.is_contiguous() for t in (a, u, v, count, direction, d))
    assert all(t.stride(-1) == 1 for t in (x, b, c, dt, decay))
    assert x.stride(1) == p and b.stride(1) == n and c.stride(1) == n
    if out is None:
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
    assert out.is_contiguous() and out.shape == x.shape
    _step[(batch, heads)](
        x, b, c, dt, decay, d, dt_bias if dt_bias is not None else d, direction, a, u, v, count, slots, out,
        heads, groups, p, n, capacity, x.stride(0), b.stride(0), c.stride(0),
        dt.stride(0), decay.stride(0), slots.stride(0),
        dt_bias is not None, do_truncate, rank, rank + window, TRUNC_ITERS, MGS_REL_TOL,
        num_warps=4, enable_fp_fusion=dt_bias is not None,
    )
    return out
