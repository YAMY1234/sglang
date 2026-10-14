"""Mamba-2 direct factor recurrence, with a left sink and device W-step expiry.

S[P,N] = direction[P] * a[N] + U[RMAX,P].T @ V[RMAX,N].
One CTA owns each (slot, head); callers must supply unique nonnegative slots.
Negative slots are padding. Output observes the updated, *untruncated* state;
expiry is a subsequent launch on the same stream, before the next state reader.
The append/MGS and small-Gram Jacobi algorithms are adapted from GDN K1/K2.
There is no GDN delta correction. FP32 factors; no dense state in the step.
"""
import torch
import triton
import triton.language as tl

from .gdn_truncate import _jacobi_vectors


@triton.jit
def _step(X, B, C, DT, DECAY, D, DIR, FA, U, V, COUNT, SLOTS, Y,
          H: tl.constexpr, G: tl.constexpr, P: tl.constexpr, N: tl.constexpr,
          RM: tl.constexpr, XS: tl.constexpr, BS: tl.constexpr, CS: tl.constexpr,
          DTS: tl.constexpr, DS: tl.constexpr, IS: tl.constexpr):
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
    tl.store(yp, y)
    tl.store(vp, new_v, r[:, None] <= cnt)
    tl.store(U + sh * RM * P + cnt * P + p, new_u)
    tl.store(COUNT + sh, cnt + 1)


@triton.jit
def _truncate(U, V, COUNT, SLOTS, H: tl.constexpr, P: tl.constexpr,
              N: tl.constexpr, RM: tl.constexpr, R: tl.constexpr,
              FULL: tl.constexpr, IS: tl.constexpr, SWEEPS: tl.constexpr):
    row, h = tl.program_id(0), tl.program_id(1)
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
    gram = tl.dot(v, tl.trans(v), input_precision="ieee")
    z = _jacobi_vectors(gram, RM, R, SWEEPS)
    u = tl.load(up, r[:, None] < FULL, 0.)
    un = tl.dot(tl.trans(z), u, input_precision="ieee")
    vn = tl.dot(tl.trans(z), v, input_precision="ieee")
    tl.store(up, un, r[:, None] < R)
    tl.store(vp, vn, r[:, None] < R)
    tl.store(COUNT + sh, R)


def truncate(u, v, count, slots, rank=16, window=16):
    """Device-side expiry; safe for changing phases/slots during graph replay."""
    _, heads, capacity, p = u.shape
    _truncate[(slots.numel(), heads)](
        u, v, count, slots, heads, p, v.shape[-1], capacity,
        rank, rank + window, slots.stride(0), 8, num_warps=4,
        enable_fp_fusion=False,
    )


def step(x, b, c, dt, decay, d, direction, a, u, v, count, slots,
         *, rank=16, window=16, out=None, do_truncate=True):
    """Inputs x[B,H,P], B/C[B,G,N], dt/decay[B,H], D[H].

    dt=softplus(raw_dt+bias), decay=exp(dt*A) are supplied by the caller to
    preserve the bridge's activation rounding. State buffers are contiguous;
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
        x, b, c, dt, decay, d, direction, a, u, v, count, slots, out,
        heads, groups, p, n, capacity, x.stride(0), b.stride(0), c.stride(0),
        dt.stride(0), decay.stride(0), slots.stride(0),
        num_warps=4, enable_fp_fusion=False,
    )
    if do_truncate:
        truncate(u, v, count, slots, rank, window)
    return out
