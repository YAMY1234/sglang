"""Block (chunk) form of the factored GDN verify window + one-launch commit (#ssmon-opus, docs/120).

The frozen verify path runs the r8/W8 step primitive four times per layer (append, W8 cut inside the window),
then the commit restores the entry state and replays the accepted inputs through the same primitive.  Here:

  * verify (one launch per layer, replaces the stock verify kernel): the persistent pool is READ-ONLY.  For the four
    inputs t = 0..3 of a (request, value head) the only serial work is the Gram-Schmidt of k_t against
    [U0; khat_0..khat_{t-1}] (K-vector reductions).  Everything that touches the (RMAX, V) coefficient matrix W is
    rewritten in block form: W_{t-1} = G_{t-1} W0 + sum_{j<t} h_{j,t-1} cfull_j delta_j^T, so W_{t-1}^T x needs only
    W0^T x (one small block product for all t) plus 4x4 scalar coefficients; no per-input W update, no FP16 round trip.
    No truncation inside the window (the up-to-19 rows live in registers), i.e. the B-arm cut rule (#746/#753 gate).
  * per input the kernel records a_t, khat_t (FP16-rounded, as the pool stores U), cfull_t, delta_t and g_t.
  * commit (ONE launch for all layers, all rows): rebuild the state after the accepted input s from the untouched pool
    entry and the records, cut RFULL.. -> R when count >= RFULL (count <= 19, 32-row tile), publish to the request slot
    and, from the same program (no read-after-write race on the entry), to the radix track slot.
State per (slot, head) is unchanged: a (K) fp32, U (RMAX=16, K) fp16, W (RMAX, V) fp16, count in [r, r+m).
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .gdn_factored import GS_EPS, MGS_REL_TOL, TRUNC_ITERS, _mgs

CHUNK_WARPS = int(os.environ.get("SGLANG_GDN_CHUNK_WARPS", "4"))
COMMIT_WARPS = int(os.environ.get("SGLANG_GDN_CHUNK_COMMIT_WARPS", "4"))
RC = 32  # record width of cfull: [0, 16) entry rows, [16, 20) appended rows j = 0..3


@triton.jit
def _row(X, offs, i):
    """Row i of a small (N, D) tile as a (D,) vector (select + reduce over N)."""
    return tl.sum(tl.where((offs == i)[:, None], X, 0.0), axis=0)


@triton.jit
def _col(X, offs, j):
    """Column j of a small (N, M) tile as a (N,) vector."""
    return tl.sum(tl.where((offs == j)[None, :], X, 0.0), axis=1)


@triton.jit
def _factored_chunk_verify_kernel(
    mixed, gate_a, gate_b, A_log, dt_bias, vbar,
    pa, pu, pw, pcount, indices,
    output, rec_a, rec_k, rec_c, rec_d, rec_g,
    scale, gs_eps,
    MIXED_ROW: tl.constexpr, MIXED_STEP: tl.constexpr,
    A_ROW: tl.constexpr, A_STEP: tl.constexpr, B_ROW: tl.constexpr, B_STEP: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    RMAX: tl.constexpr, T: tl.constexpr, RCW: tl.constexpr,
):
    pid = tl.program_id(0)
    i_n = pid // HV
    i_hv = pid % HV
    i_h = i_hv // (HV // H)
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V)
    offs_r = tl.arange(0, RMAX)
    offs_t = tl.arange(0, T)
    offs_c = tl.arange(0, RCW)
    # record row (n, t, hv) = (n*T + t)*HV + hv
    rec_row = (i_n * T + offs_t) * HV + i_hv  # (T,)
    slot = tl.load(indices + i_n).to(tl.int64)
    if slot < 0:
        tl.store(output + rec_row[:, None] * V + offs_v[None, :], tl.zeros([T, V], dtype=tl.float32).to(output.dtype.element_ty))
        return
    base = slot * HV + i_hv
    c0 = tl.load(pcount + base)
    rmask = offs_r < c0
    U0 = tl.load(pu + base * RMAX * K + offs_r[:, None] * K + offs_k[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    W0 = tl.load(pw + base * RMAX * V + offs_r[:, None] * V + offs_v[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    a = tl.load(pa + base * K + offs_k)
    vb = tl.load(vbar + i_hv * V + offs_v).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)

    # ---- the four inputs as (T, .) tiles: stock packed layout and stock gate formula
    p_mixed = mixed + i_n * MIXED_ROW + offs_t[:, None] * MIXED_STEP
    Q = tl.load(p_mixed + i_h * K + offs_k[None, :]).to(tl.float32)
    Kt = tl.load(p_mixed + H * K + i_h * K + offs_k[None, :]).to(tl.float32)
    Vt = tl.load(p_mixed + 2 * H * K + i_hv * V + offs_v[None, :]).to(tl.float32)
    ga = tl.load(gate_a + i_n * A_ROW + offs_t * A_STEP + i_hv).to(tl.float32)
    gb = tl.load(gate_b + i_n * B_ROW + offs_t * B_STEP + i_hv).to(tl.float32)
    x = ga + dt_bias_val
    softplus_x = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    gt = tl.exp(-tl.exp(A_log_val) * softplus_x)  # (T,)
    beta = tl.sigmoid(gb).to(gate_b.dtype.element_ty).to(tl.float32)  # (T,)
    QN = Q / tl.sqrt(tl.sum(Q * Q, axis=1) + 1e-6)[:, None] * scale
    KN = Kt / tl.sqrt(tl.sum(Kt * Kt, axis=1) + 1e-6)[:, None]

    # ---- block products against the entry basis (independent of the serial chain)
    CB = tl.sum(U0[:, None, :] * KN[None, :, :], axis=2)  # (RMAX, T): U0 k_t
    QB = tl.sum(U0[:, None, :] * QN[None, :, :], axis=2)  # (RMAX, T): U0 q_t
    PK = tl.sum(U0[:, None, :] * CB[:, :, None], axis=0)  # (T, K): U0^T (U0 k_t)

    # ---- serial part: Gram-Schmidt of k_t against [U0; khat_<t], sink recurrence
    KH = tl.zeros([T, K], dtype=tl.float32)  # appended basis rows (FP16-rounded, as stored)
    CA = tl.zeros([T, T], dtype=tl.float32)  # [i, t]: khat_i . k_t (i < t), second pass included
    CF = tl.zeros([T, T], dtype=tl.float32)  # [i, t]: cfull_t on appended row i (i <= t)
    QA = tl.zeros([T, T], dtype=tl.float32)  # [i, t]: khat_i . q_t (i <= t)
    SQ = tl.zeros([T], dtype=tl.float32)     # a_t . q_t
    for t in tl.static_range(T):
        kn = _row(KN, offs_t, t)
        qn = _row(QN, offs_t, t)
        g_t = tl.sum(tl.where(offs_t == t, gt, 0.0), axis=0)
        b_t = tl.sum(tl.where(offs_t == t, beta, 0.0), axis=0)
        # sink: exact key-side vector recurrence
        a = g_t * (a - b_t * kn * tl.sum(kn * a, axis=0)) + b_t * kn
        tl.store(rec_a + ((i_n * T + t) * HV + i_hv) * K + offs_k, a)
        SQ = tl.where(offs_t == t, tl.sum(a * qn, axis=0), SQ)
        cb = _col(CB, offs_t, t)
        ca = tl.sum(KH * kn[None, :], axis=1)  # (T,), rows >= t are zero rows of KH
        kp = kn - _row(PK, offs_t, t) - tl.sum(KH * ca[:, None], axis=0)
        nrm2 = tl.sum(kp * kp, axis=0)
        if nrm2 < 0.25:  # k nearly in the span: one more pass ("twice is enough"), program-uniform
            cb2 = tl.sum(U0 * kp[None, :], axis=1)
            ca2 = tl.sum(KH * kp[None, :], axis=1)
            kp = kp - tl.sum(U0 * cb2[:, None], axis=0) - tl.sum(KH * ca2[:, None], axis=0)
            cb = cb + cb2
            ca = ca + ca2
            nrm2 = tl.sum(kp * kp, axis=0)
            CB = tl.where((offs_t == t)[None, :], cb[:, None], CB)  # c_t entry part incl. the second pass
        nrm = tl.sqrt(nrm2)
        keep = nrm > gs_eps
        khat = tl.where(keep, kp / tl.maximum(nrm, gs_eps), 0.0)
        clast = tl.where(keep, nrm, 0.0)
        khat16 = khat.to(tl.float16).to(tl.float32)
        # query coefficients on appended rows i < t (stored rows) and on the new row (unrounded, as the primitive)
        qa = tl.sum(KH * qn[None, :], axis=1) + tl.where(offs_t == t, tl.sum(khat * qn, axis=0), 0.0)
        KH = tl.where((offs_t == t)[:, None], khat16[None, :], KH)
        CA = tl.where((offs_t == t)[None, :], ca[:, None], CA)
        CF = tl.where((offs_t == t)[None, :], (ca + tl.where(offs_t == t, clast, 0.0))[:, None], CF)
        QA = tl.where((offs_t == t)[None, :], qa[:, None], QA)
        tl.store(rec_k + ((i_n * T + t) * HV + i_hv) * K + offs_k, khat16.to(rec_k.dtype.element_ty))
        # cfull_t record: entry rows in [0, 16), appended rows in [16, 16 + T)
        cfe = tl.sum(tl.where(offs_c[:, None] == offs_r[None, :], cb[None, :], 0.0), axis=1)
        cfa = tl.sum(tl.where(offs_c[:, None] == (RMAX + offs_t)[None, :],
                              (ca + tl.where(offs_t == t, clast, 0.0))[None, :], 0.0), axis=1)
        tl.store(rec_c + ((i_n * T + t) * HV + i_hv) * RCW + offs_c, cfe + cfa)

    # ---- V side in block form: M = W0^T [c_t | cq_t], 4x4 coefficient matrices, then 4 V-vector steps
    MC = tl.sum(CB[:, :, None] * W0[:, None, :], axis=0)  # (T, V): W0^T c_t (entry rows)
    MQ = tl.sum(QB[:, :, None] * W0[:, None, :], axis=0)  # (T, V): W0^T cq_t
    # SS[j, t] = cfull_j . c_t ; RR[j, t] = cfull_j . cq_t  (entry rows + appended rows)
    SS = tl.sum(CB[:, :, None] * CB[:, None, :], axis=0) + tl.sum(CF[:, :, None] * CA[:, None, :], axis=0)
    RR = tl.sum(CB[:, :, None] * QB[:, None, :], axis=0) + tl.sum(CF[:, :, None] * QA[:, None, :], axis=0)
    D = tl.zeros([T, V], dtype=tl.float32)
    hv_ = tl.zeros([T], dtype=tl.float32)  # h_{j,t-1} = prod_{j<i<=t-1} g_i for j < t
    G = 1.0
    for t in tl.static_range(T):
        g_t = tl.sum(tl.where(offs_t == t, gt, 0.0), axis=0)
        b_t = tl.sum(tl.where(offs_t == t, beta, 0.0), axis=0)
        ss = _col(SS, offs_t, t)
        rr = _col(RR, offs_t, t)
        rtt = tl.sum(tl.where(offs_t == t, rr, 0.0), axis=0)
        mvec = G * _row(MC, offs_t, t) + tl.sum(D * (hv_ * ss)[:, None], axis=0)
        wq = G * _row(MQ, offs_t, t) + tl.sum(D * (hv_ * rr)[:, None], axis=0)
        v = _row(Vt, offs_t, t)
        delta = b_t * ((v - vb) - g_t * mvec)
        out = vb * tl.sum(tl.where(offs_t == t, SQ, 0.0), axis=0) + g_t * wq + delta * rtt
        tl.store(output + ((i_n * T + t) * HV + i_hv) * V + offs_v, out.to(output.dtype.element_ty))
        tl.store(rec_d + ((i_n * T + t) * HV + i_hv) * V + offs_v, delta)
        D = tl.where((offs_t == t)[:, None], delta[None, :], D)
        hv_ = tl.where(offs_t == t, 1.0, hv_ * g_t)
        G = G * g_t
    tl.store(rec_g + rec_row, gt)


@triton.jit
def _factored_commit_select_kernel(
    pa, pu, pw, pcount, stale, dense_of, dense_required, prefix_valid,
    rec_a, rec_k, rec_c, rec_d, rec_g,
    src_slots, steps, track_slots, track_steps,
    LAYER_A: tl.constexpr, LAYER_U: tl.constexpr, LAYER_W: tl.constexpr, LAYER_COUNT: tl.constexpr,
    LAYER_RA: tl.constexpr, LAYER_RK: tl.constexpr, LAYER_RC: tl.constexpr, LAYER_RD: tl.constexpr,
    LAYER_RG: tl.constexpr,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, RMAX: tl.constexpr, RP: tl.constexpr,
    R: tl.constexpr, RFULL: tl.constexpr, T: tl.constexpr, RCW: tl.constexpr,
    ITERS: tl.constexpr, REL_TOL: tl.constexpr,
    HAS_TRACK: tl.constexpr, HAS_DENSE_OF: tl.constexpr, HAS_DENSE_REQUIRED: tl.constexpr,
    HAS_PREFIX_VALID: tl.constexpr,
):
    pid = tl.program_id(0)
    layer = tl.program_id(1).to(tl.int64)
    i_n = pid // HV
    i_hv = pid % HV
    slot = tl.load(src_slots + i_n).to(tl.int64)
    step = tl.load(steps + i_n)
    if HAS_TRACK:
        tslot = tl.load(track_slots + i_n).to(tl.int64)
        tstep = tl.load(track_steps + i_n)
    else:
        tslot = slot * 0 - 1
        tstep = step * 0 - 1
    if (slot < 0) | (step < 0):
        return
    pa += layer * LAYER_A
    pu += layer * LAYER_U
    pw += layer * LAYER_W
    pcount += layer * LAYER_COUNT
    rec_a += layer * LAYER_RA
    rec_k += layer * LAYER_RK
    rec_c += layer * LAYER_RC
    rec_d += layer * LAYER_RD
    rec_g += layer * LAYER_RG
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V)
    offs_p = tl.arange(0, RP)  # 32-row tile: entry rows + up to T appended rows
    offs_c = tl.arange(0, RCW)
    offs_t = tl.arange(0, T)
    base = slot * HV + i_hv
    c0 = tl.load(pcount + base)
    emask = offs_p < c0
    U0 = tl.load(pu + base * RMAX * K + offs_p[:, None] * K + offs_k[None, :], mask=emask[:, None], other=0.0).to(tl.float32)
    W0 = tl.load(pw + base * RMAX * V + offs_p[:, None] * V + offs_v[None, :], mask=emask[:, None], other=0.0).to(tl.float32)
    rrow = (i_n * T + offs_t) * HV + i_hv  # (T,)
    gts = tl.load(rec_g + rrow)  # (T,)
    KH = tl.load(rec_k + rrow[:, None] * K + offs_k[None, :]).to(tl.float32)  # (T, K)
    Dt = tl.load(rec_d + rrow[:, None] * V + offs_v[None, :])  # (T, V)
    # cfull_j mapped to tile rows: entry row r < c0 -> record r; appended row c0 + i -> record RMAX + i
    ridx = tl.where(offs_p < c0, offs_p, RMAX + offs_p - c0)
    rvalid = offs_p < c0 + T
    for which in tl.static_range(2):
        if which == 0:
            dst, s = tslot, tstep  # track copy first (it never exceeds the accepted prefix)
        else:
            dst, s = slot, step
        if (dst >= 0) & (s >= 0):
            n = c0 + s + 1
            # W_s = G_s W0 + sum_{j<=s} h_{j,s} cfull_j delta_j^T ; U_s = [U0; khat_0..khat_s]
            Gs = tl.reduce(tl.where(offs_t <= s, gts, 1.0), 0, _mul)
            W = Gs * W0
            U = U0
            for j in tl.static_range(T):
                if j <= s:
                    h = tl.reduce(tl.where((offs_t > j) & (offs_t <= s), gts, 1.0), 0, _mul)
                    cj = tl.load(rec_c + ((i_n * T + j) * HV + i_hv) * RCW + ridx, mask=rvalid, other=0.0)
                    cj = tl.where(offs_p < c0 + j + 1, cj, 0.0)
                    dj = tl.sum(tl.where((offs_t == j)[:, None], Dt, 0.0), axis=0)
                    W = W + h * cj[:, None] * dj[None, :]
                    kj = tl.sum(tl.where((offs_t == j)[:, None], KH, 0.0), axis=0)
                    U = tl.where((offs_p == c0 + j)[:, None], kj[None, :], U)
            if n >= RFULL:  # cut n (<= RFULL + T - 1) rows -> R: K0 iter (G = W W^T, subspace iteration + MGS2)
                rows = offs_p < n
                keep = offs_p < R
                Gm = tl.dot(W, tl.trans(W), input_precision="ieee")
                d = tl.where(rows, tl.sum(tl.where(offs_p[:, None] == offs_p[None, :], Gm, 0.0), axis=1), -1.0)
                better = (d[None, :] > d[:, None]) | ((d[None, :] == d[:, None]) & (offs_p[None, :] < offs_p[:, None]))
                rank = tl.sum(better.to(tl.int32), axis=1)
                Z = tl.where((rank[:, None] == offs_p[None, :]) & keep[None, :] & rows[:, None], 1.0, 0.0)
                for _ in tl.static_range(ITERS):
                    Z = tl.dot(Gm, Z, input_precision="ieee")
                    Z = _mgs(Z, offs_p, R, 2, REL_TOL)
                Zt = tl.trans(Z)
                U = tl.where(keep[:, None], tl.dot(Zt, U, input_precision="ieee"), 0.0)
                W = tl.where(keep[:, None], tl.dot(Zt, W, input_precision="ieee"), 0.0)
                n = n * 0 + R
            else:
                U = tl.where((offs_p < n)[:, None], U, 0.0)
                W = tl.where((offs_p < n)[:, None], W, 0.0)
            a_s = tl.load(rec_a + ((i_n * T + s) * HV + i_hv) * K + offs_k)
            out = dst * HV + i_hv
            smask = (offs_p < RMAX)[:, None]
            tl.store(pa + out * K + offs_k, a_s)
            tl.store(pu + out * RMAX * K + offs_p[:, None] * K + offs_k[None, :], U.to(pu.dtype.element_ty), mask=smask)
            tl.store(pw + out * RMAX * V + offs_p[:, None] * V + offs_v[None, :], W.to(pw.dtype.element_ty), mask=smask)
            tl.store(pcount + out, n)
            if (layer == 0) & (i_hv == 0):
                tl.store(stale + dst, 1)
                if HAS_DENSE_OF:
                    tl.store(dense_of + dst, -1)
                if HAS_DENSE_REQUIRED:
                    tl.store(dense_required + dst, 0)
                if HAS_PREFIX_VALID:
                    tl.store(prefix_valid + dst, 0)


@triton.jit
def _mul(a, b):
    return a * b


def chunk_verify(mixed, gate_a, gate_b, *, A_log, dt_bias, vbar, pa, pu, pw, pcount, indices,
                 records, layer, scale, num_q_heads, output=None):
    """mixed (B, T, width) bf16; gate_a/gate_b (B, T, HV); pool layer tensors; records dict of per-layer buffers."""
    batch, tokens, _ = mixed.shape
    hv, rmax, k = pu.shape[1], pu.shape[2], pu.shape[3]
    v = pw.shape[-1]
    if output is None:
        output = mixed.new_empty(batch, tokens, hv, v)
    _factored_chunk_verify_kernel[(batch * hv,)](
        mixed, gate_a, gate_b, A_log, dt_bias, vbar, pa, pu, pw, pcount, indices,
        output, records['a'][layer], records['k'][layer], records['c'][layer], records['d'][layer],
        records['g'][layer], scale, GS_EPS,
        mixed.stride(0), mixed.stride(1), gate_a.stride(0), gate_a.stride(1),
        gate_b.stride(0), gate_b.stride(1),
        num_q_heads, hv, k, v, rmax, tokens, RC, num_warps=CHUNK_WARPS)
    return output


def commit_select(pool, records, src_slots, steps, track_slots=None, track_steps=None, *, r, rfull,
                  trunc_iters=None):
    """One launch: rebuild accepted states from the untouched entry + records, cut when due, publish."""
    n = src_slots.numel()
    if n == 0:
        return
    layers, _, hv, rmax, k = pool.U.shape
    v = pool.W.shape[-1]
    tokens = records['a'].shape[2]
    has_track = track_slots is not None
    _factored_commit_select_kernel[(n * hv, layers)](
        pool.a, pool.U, pool.W, pool.count, pool.stale,
        pool.dense_of if pool.dense_of is not None else pool.stale,
        pool.dense_required if pool.dense_required is not None else pool.stale,
        pool.prefix_valid if pool.prefix_valid is not None else pool.stale,
        records['a'], records['k'], records['c'], records['d'], records['g'],
        src_slots, steps, track_slots if has_track else src_slots, track_steps if has_track else steps,
        pool.a.stride(0), pool.U.stride(0), pool.W.stride(0), pool.count.stride(0),
        records['a'].stride(0), records['k'].stride(0), records['c'].stride(0), records['d'].stride(0),
        records['g'].stride(0),
        hv, k, v, rmax, 32, r, rfull, tokens, RC, trunc_iters or TRUNC_ITERS, MGS_REL_TOL,
        has_track, pool.dense_of is not None, pool.dense_required is not None, pool.prefix_valid is not None,
        num_warps=COMMIT_WARPS)


def allocate_records(layers, max_batch, tokens, hv, k, v, device):
    """Per-layer verify records; layout (L, B, T, HV, .) so a layer's rows are contiguous for the verify launch."""
    return dict(
        a=torch.zeros(layers, max_batch, tokens, hv, k, dtype=torch.float32, device=device),
        k=torch.zeros(layers, max_batch, tokens, hv, k, dtype=torch.float16, device=device),
        c=torch.zeros(layers, max_batch, tokens, hv, RC, dtype=torch.float32, device=device),
        d=torch.zeros(layers, max_batch, tokens, hv, v, dtype=torch.float32, device=device),
        g=torch.zeros(layers, max_batch, tokens, hv, dtype=torch.float32, device=device),
    )
