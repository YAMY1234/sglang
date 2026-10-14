"""KDA factors, with an orthonormal *value* basis.

S[K,V] = a[K] vbar[V]^T + C[R,K]^T Q[R,V].  Unlike GDN's
scalar decay, KDA's diagonal key decay does not preserve a key basis.
Keeping Q on the value side makes decay an exact coefficient update and
lets expiry use GDN's unchanged FP32 MGS subspace iteration (Q, C).

One CTA owns one live (slot, head). Live indices must be unique; negative
indices are graph padding. Step, expiry and the next reader share a stream.
No dense state, host counter, synchronization or cross-CTA publication.
"""

import torch
import triton
import triton.language as tl

from .gdn_factored import (
    _gram_wwt,
    _mgs,
    _mgs_rolled,
    _project_rows,
    _small_matmul_cols,
)


@triton.jit
def _step(
    mixed,
    gate,
    beta_raw,
    alog,
    bias,
    vb_ptr,
    a_ptr,
    q_ptr,
    c_ptr,
    count_ptr,
    indices,
    output,
    scale,
    SM: tl.constexpr,
    SG: tl.constexpr,
    SB: tl.constexpr,
    SI: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    CAP: tl.constexpr,
    RMAX: tl.constexpr,
    R: tl.constexpr,
    LOWER: tl.constexpr,
):
    pid = tl.program_id(0)
    row, head = pid // HV, pid % HV
    kh = head // (HV // H)
    slot = tl.load(indices + row * SI).to(tl.int64)
    kk, vv, rr = tl.arange(0, K), tl.arange(0, V), tl.arange(0, RMAX)
    op = output + (row * HV + head) * V + vv
    if slot < 0:
        tl.store(op, tl.full((V,), 0, tl.float32))
        return
    x = mixed + row * SM
    query = tl.load(x + kh * K + kk).to(tl.float32)
    key = tl.load(x + H * K + kh * K + kk).to(tl.float32)
    value = tl.load(x + 2 * H * K + head * V + vv).to(tl.float32)
    query = query * tl.rsqrt(tl.sum(query * query) + 1.0e-6) * scale
    key = key * tl.rsqrt(tl.sum(key * key) + 1.0e-6)
    z = tl.load(gate + row * SG + head * K + kk).to(tl.float32)
    z += tl.load(bias + head * K + kk).to(tl.float32)
    A = tl.exp(tl.load(alog + head).to(tl.float32))
    if LOWER is None:
        log_decay = -A * tl.where(z <= 20.0, tl.log(1.0 + tl.exp(z)), z)
    else:
        log_decay = LOWER * tl.sigmoid(A * z)
    decay = tl.exp(log_decay)
    beta = tl.sigmoid(tl.load(beta_raw + row * SB + head).to(tl.float32))
    vb = tl.load(vb_ptr + head * V + vv).to(tl.float32)
    base = slot * HV + head
    ap = a_ptr + base * K + kk
    a = tl.load(ap).to(tl.float32) * decay
    a = a + beta * key * (1.0 - tl.sum(a * key))
    tl.store(ap, a)
    cp = count_ptr + base
    count = tl.load(cp)
    qp = q_ptr + base * CAP * V + rr[:, None] * V + vv[None, :]
    wp = c_ptr + base * CAP * K + rr[:, None] * K + kk[None, :]
    Q = tl.load(qp, (rr < count)[:, None], 0).to(tl.float32)
    C = tl.load(wp, (rr < count)[:, None], 0).to(tl.float32) * decay[None, :]
    ck = tl.sum(C * key[None, :], 1)
    residual = value - vb - tl.sum(Q * ck[:, None], 0)
    # Two-pass GS. Q is unchanged by key decay and stays orthonormal.
    proj = tl.sum(Q * residual[None, :], 1)
    orth = residual - tl.sum(Q * proj[:, None], 0)
    corr = tl.sum(Q * orth[None, :], 1)
    orth -= tl.sum(Q * corr[:, None], 0)
    proj += corr
    norm = tl.sqrt(tl.sum(orth * orth))
    keep = norm > 1.0e-7 * tl.sqrt(tl.sum(residual * residual))
    newq = tl.where(keep, orth / tl.maximum(norm, 1.0e-30), 0.0)
    proj = tl.where(rr == count, tl.where(keep, norm, 0.0), proj)
    Q = tl.where((rr == count)[:, None], newq[None, :], Q)
    C += proj[:, None] * (beta * key)[None, :]
    out = vb * tl.sum(a * query) + tl.sum(Q * tl.sum(C * query[None, :], 1)[:, None], 0)
    # Read the Wth output before truncation. Each slot owns its device count.
    # Exactly GDN K2's FP32 elementwise/reduction MGS path (no tl.dot/eigh).
    if count + 1 >= CAP:
        G = _gram_wwt(C, rr, RMAX, CAP)
        rows, kept = rr < CAP, rr < R
        d = tl.where(
            rows, tl.sum(tl.where(rr[:, None] == rr[None, :], G, 0.0), 1), -1.0
        )
        better = (d[None, :] > d[:, None]) | (
            (d[None, :] == d[:, None]) & (rr[None, :] < rr[:, None])
        )
        order = tl.sum(better.to(tl.int32), 1)
        Z = tl.where(
            (order[:, None] == rr[None, :]) & kept[None, :] & rows[:, None], 1.0, 0.0
        )
        for _ in tl.static_range(3):
            Z = _small_matmul_cols(G, Z, rr, RMAX, R)
            Z = _mgs(Z, rr, R, 2, 1.0e-4)
        Q = _project_rows(Z, Q, rr, RMAX, R)
        C = _project_rows(Z, C, rr, RMAX, R)
        tl.store(wp, C, kept[:, None])
        tl.store(qp, Q, kept[:, None])
        tl.store(cp, R)
    else:
        tl.store(wp, C, (rr <= count)[:, None])
        tl.store(qp, Q, (rr == count)[:, None])
        tl.store(cp, count + 1)
    tl.store(op, out)


@triton.jit
def _apply_content_gram(U, V, Z, rr, cc, R: tl.constexpr):
    # Z is (Dv, padded rank); apply S_c^T S_c without a dense S_c.
    result = tl.zeros_like(Z)
    for j in range(R):
        z = tl.sum(tl.where((cc == j)[None, :], Z, 0.0), 1)
        vz = tl.sum(V * z[None, :], 1)
        uz = tl.sum(U * vz[:, None], 0)
        uu = tl.sum(U * uz[None, :], 1)
        y = tl.sum(V * uu[:, None], 0)
        result = tl.where((cc == j)[None, :], y[:, None], result)
    return result


@triton.jit
def _append_step(
    mixed,
    gate,
    beta_raw,
    alog,
    bias,
    vb_ptr,
    a_ptr,
    v_ptr,
    u_ptr,
    log_decay_ptr,
    exponent_ptr,
    count_ptr,
    indices,
    output,
    SM: tl.constexpr,
    SG: tl.constexpr,
    SB: tl.constexpr,
    SI: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    VDIM: tl.constexpr,
    CAP: tl.constexpr,
    RP: tl.constexpr,
    R: tl.constexpr,
    LOWER: tl.constexpr,
):
    # Ownership: one CTA per live slot/head; same-stream read, append and cut.
    pid = tl.program_id(0)
    row, head = pid // HV, pid % HV
    kh = head // (HV // H)
    slot = tl.load(indices + row * SI).to(tl.int64)
    kk, vv, rr = tl.arange(0, K), tl.arange(0, VDIM), tl.arange(0, RP)
    op = output + (row * HV + head) * VDIM + vv
    if slot < 0:
        tl.store(op, tl.full((VDIM,), 0, tl.float32))
        return
    x = mixed + row * SM
    q = tl.load(x + kh * K + kk).to(tl.float32)
    k = tl.load(x + H * K + kh * K + kk).to(tl.float32)
    v = tl.load(x + 2 * H * K + head * VDIM + vv).to(tl.float32)
    q *= tl.rsqrt(tl.sum(q * q) + 1.0e-6) * (K**-0.5)
    k *= tl.rsqrt(tl.sum(k * k) + 1.0e-6)
    z = tl.load(gate + row * SG + head * K + kk).to(tl.float32)
    z += tl.load(bias + head * K + kk).to(tl.float32)
    A = tl.exp(tl.load(alog + head).to(tl.float32))
    if LOWER is None:
        lg = -A * tl.where(z <= 20.0, tl.log(1.0 + tl.exp(z)), z)
    else:
        lg = LOWER * tl.sigmoid(A * z)
    beta = tl.sigmoid(tl.load(beta_raw + row * SB + head).to(tl.float32))
    base = slot * HV + head
    ap = a_ptr + base * K + kk
    dp = log_decay_ptr + base * K + kk
    logd = tl.load(dp) + lg
    a = tl.load(ap) * tl.exp(lg)
    a += beta * k * (1.0 - tl.sum(a * k))
    tl.store(ap, a)
    vb = tl.load(vb_ptr + head * VDIM + vv)
    cp = count_ptr + base
    count = tl.load(cp)
    up = u_ptr + base * CAP * K + rr[:, None] * K + kk[None, :]
    vp = v_ptr + base * CAP * VDIM + rr[:, None] * VDIM + vv[None, :]
    ep = exponent_ptr + base * CAP + rr
    raw_u = tl.load(up, (rr < count)[:, None], 0).to(tl.float32)
    V = tl.load(vp, (rr < count)[:, None], 0).to(tl.float32)
    exponent = tl.load(ep, rr < count, 0)
    # Compute products in log space to avoid exp(logd) underflow or 1/d overflow.
    U = tl.where(
        raw_u == 0,
        0.0,
        tl.exp(tl.log(tl.abs(raw_u)) + logd[None, :] + exponent[:, None])
        * tl.where(raw_u < 0, -1.0, 1.0),
    )
    residual = v - vb - tl.sum(V * tl.sum(U * k[None, :], 1)[:, None], 0)
    # A per-column scalar keeps new U/d representable. Old columns are immutable.
    new_exponent = tl.max(-logd, 0)
    new_u = (beta * k * tl.exp(-logd - new_exponent)).to(u_ptr.dtype.element_ty)
    new_v = residual.to(v_ptr.dtype.element_ty)
    physical_u = tl.where(
        new_u == 0,
        0.0,
        tl.exp(tl.log(tl.abs(new_u.to(tl.float32))) + logd + new_exponent)
        * tl.where(new_u < 0, -1.0, 1.0),
    )
    U = tl.where((rr == count)[:, None], physical_u[None, :], U)
    V = tl.where((rr == count)[:, None], new_v[None, :].to(tl.float32), V)
    out = vb * tl.sum(a * q) + tl.sum(V * tl.sum(U * q[None, :], 1)[:, None], 0)
    if count + 1 >= CAP:
        # Non-orthogonal immutable columns require S^T S, not just U U^T.
        energy = tl.sum(U * U, 1) * tl.sum(V * V, 1)
        energy = tl.where(rr < CAP, energy, -1.0)
        order = tl.sum(
            (energy[None, :] > energy[:, None])
            | ((energy[None, :] == energy[:, None]) & (rr[None, :] < rr[:, None])),
            1,
        )
        Z = tl.zeros((VDIM, RP), tl.float32)
        for j in range(R):
            seed = tl.sum(tl.where((order == j)[:, None], V, 0.0), 0)
            Z = tl.where((rr == j)[None, :], seed[:, None], Z)
        Z = _mgs_rolled(Z, rr, R, 2, 1.0e-4)
        for _ in tl.static_range(3):
            Z = _apply_content_gram(U, V, Z, rr, rr, R)
            Z = _mgs_rolled(Z, rr, R, 2, 1.0e-4)
        unew = tl.zeros_like(U)
        for j in range(R):
            zj = tl.sum(tl.where((rr == j)[None, :], Z, 0.0), 1)
            weights = tl.sum(V * zj[None, :], 1)
            uj = tl.sum(U * weights[:, None], 0)
            unew = tl.where((rr == j)[:, None], uj[None, :], unew)
        tl.store(up, unew, (rr < R)[:, None])
        tl.store(vp, tl.trans(Z), (rr < R)[:, None])
        tl.store(dp, 0.0)
        tl.store(ep, 0.0, rr < R)
        tl.store(cp, R)
    else:
        tl.store(u_ptr + base * CAP * K + count * K + kk, new_u)
        tl.store(v_ptr + base * CAP * VDIM + count * VDIM + vv, new_v)
        tl.store(exponent_ptr + base * CAP + count, new_exponent)
        tl.store(dp, logd)
        tl.store(cp, count + 1)
    tl.store(op, out)


def packed_decode(
    mixed,
    gate,
    beta,
    alog,
    bias,
    vbar,
    a,
    basis,
    coeff,
    count,
    indices,
    rank,
    window,
    *,
    lower_bound=None,
    log_decay=None,
    column_exponent=None,
):
    """Post-step expiry: every Wth output is read before its state is cut.

    Sink is FP32. BF16 columns are immutable between append/expiry; their
    key decay is delayed in FP32 log_decay plus one exponent per column.
    FP32 storage retains the original orthogonal-basis implementation. Contiguous (slots, heads, capacity, V/K).
    Count starts at R, so the first cut is exactly after W steps.
    """
    batch, heads = mixed.shape[0], basis.shape[1]
    capacity, values, keys = basis.shape[2], basis.shape[3], coeff.shape[3]
    assert basis.dtype == coeff.dtype
    assert a.dtype == torch.float32
    assert basis.dtype in (torch.float32, torch.bfloat16)
    assert capacity == rank + window
    assert all(x.is_contiguous() for x in (a, basis, coeff, count, vbar))
    assert triton.next_power_of_2(keys) == keys
    assert triton.next_power_of_2(values) == values
    h = (mixed.shape[-1] - heads * values) // (2 * keys)
    gate, beta = gate.reshape(batch, -1), beta.reshape(batch, -1)
    out = mixed.new_empty(1, batch, heads, values)
    if basis.dtype == torch.bfloat16 or log_decay is not None:
        assert log_decay is not None and column_exponent is not None
        assert log_decay.shape == a.shape and log_decay.dtype == torch.float32
        assert column_exponent.shape == basis.shape[:-1]
        assert column_exponent.dtype == torch.float32
        assert log_decay.is_contiguous() and column_exponent.is_contiguous()
        _append_step[(batch * heads,)](
            mixed,
            gate,
            beta,
            alog,
            bias,
            vbar,
            a,
            basis,
            coeff,
            log_decay,
            column_exponent,
            count,
            indices,
            out,
            mixed.stride(0),
            gate.stride(0),
            beta.stride(0),
            indices.stride(0),
            h,
            heads,
            keys,
            values,
            capacity,
            triton.next_power_of_2(capacity),
            rank,
            lower_bound,
            num_warps=4,
        )
        return out
    _step[(batch * heads,)](
        mixed,
        gate,
        beta,
        alog,
        bias,
        vbar,
        a,
        basis,
        coeff,
        count,
        indices,
        out,
        keys**-0.5,
        mixed.stride(0),
        gate.stride(0),
        beta.stride(0),
        indices.stride(0),
        h,
        heads,
        keys,
        values,
        capacity,
        triton.next_power_of_2(capacity),
        rank,
        lower_bound,
        num_warps=4,
    )
    return out
