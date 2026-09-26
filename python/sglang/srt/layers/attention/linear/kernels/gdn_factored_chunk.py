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

CHUNK_WARPS = int(os.environ.get("SGLANG_GDN_CHUNK_WARPS", "1"))  # j883641 sweep: 1 warp best (B1-B32)
COMMIT_WARPS = int(os.environ.get("SGLANG_GDN_CHUNK_COMMIT_WARPS", "2"))
CHUNK_BV = int(os.environ.get("SGLANG_GDN_CHUNK_BV", "0"))  # 0 = whole V per program
COMMIT_IMPL = os.environ.get("SGLANG_GDN_CHUNK_COMMIT", "block")  # block | tile (32-row reference)
MODE = os.environ.get("SGLANG_GDN_CHUNK_MODE", "dense")  # dense: dense verify from factors + factor chain at commit
DENSE_IMPL = os.environ.get("SGLANG_GDN_DENSE_IMPL", "tile")  # tile (default) | wy: S0 x via factors (j885132 bench: 4x slower at BV 32)
DENSE_BV = int(os.environ.get("SGLANG_GDN_DENSE_BV", "16"))  # j885310 sweep (tile impl): 16 x 1 warp best B16/B32
DENSE_WARPS = int(os.environ.get("SGLANG_GDN_DENSE_WARPS", "1"))
DENSE_DOT = os.environ.get("SGLANG_GDN_DENSE_DOT", "tf32x3")  # WY impl: tensor-core fp32 emulation (ieee = FMA loops)
COMMIT_SPLIT = os.environ.get("SGLANG_GDN_CHUNK_COMMIT_SPLIT", "1") == "1"  # chain / cut-solve / publish kernels
PUBLISH_WARPS = int(os.environ.get("SGLANG_GDN_CHUNK_PUBLISH_WARPS", "2"))
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
    RMAX: tl.constexpr, T: tl.constexpr, RCW: tl.constexpr, BV: tl.constexpr,
):
    # One program per (request, value head, V block of BV columns).  The K side (Gram-Schmidt chain, sink) is
    # recomputed by every V block (it is the short part); K-side records are written by block 0 only.
    NVB: tl.constexpr = V // BV
    pid = tl.program_id(0)
    i_vb = pid % NVB
    i_n = pid // (HV * NVB)
    i_hv = (pid // NVB) % HV
    i_h = i_hv // (HV // H)
    offs_k = tl.arange(0, K)
    offs_v = i_vb * BV + tl.arange(0, BV)
    offs_r = tl.arange(0, RMAX)
    offs_t = tl.arange(0, T)
    offs_c = tl.arange(0, RCW)
    # record row (n, t, hv) = (n*T + t)*HV + hv
    rec_row = (i_n * T + offs_t) * HV + i_hv  # (T,)
    slot = tl.load(indices + i_n).to(tl.int64)
    if slot < 0:
        tl.store(output + rec_row[:, None] * V + offs_v[None, :], tl.zeros([T, BV], dtype=tl.float32).to(output.dtype.element_ty))
        return
    base = slot * HV + i_hv
    c0 = tl.load(pcount + base)
    rmask = offs_r < c0
    U0 = tl.load(pu + base * RMAX * K + offs_r[:, None] * K + offs_k[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
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

    # ---- products against the entry basis (independent of the serial chain); 2-D per input, no 3-D temporaries
    CB = tl.zeros([RMAX, T], dtype=tl.float32)  # U0 k_t
    QB = tl.zeros([RMAX, T], dtype=tl.float32)  # U0 q_t
    PK = tl.zeros([T, K], dtype=tl.float32)     # U0^T (U0 k_t)
    for t in tl.static_range(T):
        cb = tl.sum(U0 * _row(KN, offs_t, t)[None, :], axis=1)
        qb = tl.sum(U0 * _row(QN, offs_t, t)[None, :], axis=1)
        CB = tl.where((offs_t == t)[None, :], cb[:, None], CB)
        QB = tl.where((offs_t == t)[None, :], qb[:, None], QB)
        PK = tl.where((offs_t == t)[:, None], tl.sum(U0 * cb[:, None], axis=0)[None, :], PK)

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
        if i_vb == 0:
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
        if i_vb == 0:
            tl.store(rec_k + ((i_n * T + t) * HV + i_hv) * K + offs_k, khat16.to(rec_k.dtype.element_ty))
        # cfull_t record: entry rows in [0, 16), appended rows in [16, 16 + T)
        cfe = tl.sum(tl.where(offs_c[:, None] == offs_r[None, :], cb[None, :], 0.0), axis=1)
        cfa = tl.sum(tl.where(offs_c[:, None] == (RMAX + offs_t)[None, :],
                              (ca + tl.where(offs_t == t, clast, 0.0))[None, :], 0.0), axis=1)
        if i_vb == 0:
            tl.store(rec_c + ((i_n * T + t) * HV + i_hv) * RCW + offs_c, cfe + cfa)

    # ---- V side in block form: M = W0^T [c_t | cq_t], 4x4 coefficient matrices, then 4 V-vector steps
    W0 = tl.load(pw + base * RMAX * V + offs_r[:, None] * V + offs_v[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    # SS[j, t] = cfull_j . c_t ; RR[j, t] = cfull_j . cq_t  (entry rows + appended rows)
    SS = tl.sum(CB[:, :, None] * CB[:, None, :], axis=0) + tl.sum(CF[:, :, None] * CA[:, None, :], axis=0)
    RR = tl.sum(CB[:, :, None] * QB[:, None, :], axis=0) + tl.sum(CF[:, :, None] * QA[:, None, :], axis=0)
    D = tl.zeros([T, BV], dtype=tl.float32)
    hv_ = tl.zeros([T], dtype=tl.float32)  # h_{j,t-1} = prod_{j<i<=t-1} g_i for j < t
    G = 1.0
    for t in tl.static_range(T):
        g_t = tl.sum(tl.where(offs_t == t, gt, 0.0), axis=0)
        b_t = tl.sum(tl.where(offs_t == t, beta, 0.0), axis=0)
        ss = _col(SS, offs_t, t)
        rr = _col(RR, offs_t, t)
        rtt = tl.sum(tl.where(offs_t == t, rr, 0.0), axis=0)
        mvec = G * tl.sum(W0 * _col(CB, offs_t, t)[:, None], axis=0) + tl.sum(D * (hv_ * ss)[:, None], axis=0)
        wq = G * tl.sum(W0 * _col(QB, offs_t, t)[:, None], axis=0) + tl.sum(D * (hv_ * rr)[:, None], axis=0)
        v = _row(Vt, offs_t, t)
        delta = b_t * ((v - vb) - g_t * mvec)
        out = vb * tl.sum(tl.where(offs_t == t, SQ, 0.0), axis=0) + g_t * wq + delta * rtt
        tl.store(output + ((i_n * T + t) * HV + i_hv) * V + offs_v, out.to(output.dtype.element_ty))
        tl.store(rec_d + ((i_n * T + t) * HV + i_hv) * V + offs_v, delta)
        D = tl.where((offs_t == t)[:, None], delta[None, :], D)
        hv_ = tl.where(offs_t == t, 1.0, hv_ * g_t)
        G = G * g_t
    if i_vb == 0:
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
    for which in range(2):  # runtime loop: one code copy for the track slot (first) and the request slot
        dst = tl.where(which == 0, tslot, slot)
        s = tl.where(which == 0, tstep, step)
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
                # W8 cadence (B-arm protocol): keep R factors, rows R..n-(RFULL-R)-1 stay zero, count - R carries the
                # accepted inputs past the cut point, so the next cut lands after RFULL - R accepted inputs in total
                n = n - (RFULL - R)
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
def _mgs_blocks(QE, QA, offs_c, RKEEP: tl.constexpr, PASSES: tl.constexpr, REL_TOL: tl.constexpr):
    """The frozen `_mgs` (MGS2 over the first RKEEP columns, rank tolerance on the first pass) applied to the row-stacked
    matrix [QE; QA] without materialising it: every column inner product sums the entry block and the appended block."""
    n0 = tl.sqrt(tl.sum(QE * QE, axis=0) + tl.sum(QA * QA, axis=0))
    for p in tl.static_range(PASSES):
        for j in range(RKEEP):  # runtime loop: the unrolled 2 x 8 column steps made one cut program ~300 us (code size)
            colj = offs_c == j
            ye = tl.sum(tl.where(colj[None, :], QE, 0.0), axis=1)
            ya = tl.sum(tl.where(colj[None, :], QA, 0.0), axis=1)
            proj = tl.where(offs_c < j, tl.sum(QE * ye[:, None], axis=0) + tl.sum(QA * ya[:, None], axis=0), 0.0)
            ye = ye - tl.sum(QE * proj[None, :], axis=1)
            ya = ya - tl.sum(QA * proj[None, :], axis=1)
            n = tl.sqrt(tl.sum(ye * ye, axis=0) + tl.sum(ya * ya, axis=0))
            n0j = tl.sum(tl.where(colj, n0, 0.0), axis=0)
            ok = n > 1e-12
            if p == 0:
                ok = ok & (n > REL_TOL * n0j)
            ye = tl.where(ok, ye / tl.maximum(n, 1e-30), 0.0)
            ya = tl.where(ok, ya / tl.maximum(n, 1e-30), 0.0)
            QE = tl.where(colj[None, :], ye[:, None], QE)
            QA = tl.where(colj[None, :], ya[:, None], QA)
    return QE, QA


@triton.jit
def _factored_commit_block_kernel(
    pa, pu, pw, pcount, stale, dense_of, dense_required, prefix_valid,
    rec_a, rec_k, rec_c, rec_d, rec_g,
    src_slots, steps, track_slots, track_steps,
    LAYER_A: tl.constexpr, LAYER_U: tl.constexpr, LAYER_W: tl.constexpr, LAYER_COUNT: tl.constexpr,
    LAYER_RA: tl.constexpr, LAYER_RK: tl.constexpr, LAYER_RC: tl.constexpr, LAYER_RD: tl.constexpr,
    LAYER_RG: tl.constexpr,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, RMAX: tl.constexpr,
    R: tl.constexpr, RFULL: tl.constexpr, T: tl.constexpr, RCW: tl.constexpr,
    ITERS: tl.constexpr, REL_TOL: tl.constexpr,
    HAS_TRACK: tl.constexpr, HAS_DENSE_OF: tl.constexpr, HAS_DENSE_REQUIRED: tl.constexpr,
    HAS_PREFIX_VALID: tl.constexpr,
):
    """Same result as `_factored_commit_select_kernel` (entry + records -> accepted state, frozen K0 cut when count >=
    RFULL, publish), but the up-to-(RMAX + T) rows are kept as an entry block (RMAX rows) and an appended block (T rows):
    no (2 RMAX, K) tiles, the cut's Gram matrix and subspace iteration run on (RMAX|T, RMAX) blocks, and the projected
    rows are one (RMAX, RMAX) x (RMAX, K) product plus T outer products."""
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
    offs_r = tl.arange(0, RMAX)
    offs_t = tl.arange(0, T)
    base = slot * HV + i_hv
    c0 = tl.load(pcount + base)
    emask = offs_r < c0
    W0 = tl.load(pw + base * RMAX * V + offs_r[:, None] * V + offs_v[None, :], mask=emask[:, None], other=0.0).to(tl.float32)
    rrow = (i_n * T + offs_t) * HV + i_hv  # (T,)
    gts = tl.load(rec_g + rrow)
    Dt = tl.load(rec_d + rrow[:, None] * V + offs_v[None, :])  # (T, V) delta_j
    CE = tl.load(rec_c + rrow[None, :] * RCW + offs_r[:, None])  # (RMAX, T): cfull_j on entry rows
    CA = tl.load(rec_c + rrow[None, :] * RCW + RMAX + offs_t[:, None])  # (T, T): [i, j] cfull_j on appended row i
    for which in range(2):  # track slot first (never beyond the accepted prefix), then the request slot
        dst = tl.where(which == 0, tslot, slot)
        s = tl.where(which == 0, tstep, step)
        if (dst >= 0) & (s >= 0):
            n = c0 + s + 1
            Gs = tl.reduce(tl.where(offs_t <= s, gts, 1.0), 0, _mul)
            # h_j = prod_{j < i <= s} g_i for j <= s, 0 for j > s
            hj = tl.zeros([T], dtype=tl.float32)
            for j in tl.static_range(T):
                hval = tl.reduce(tl.where((offs_t > j) & (offs_t <= s), gts, 1.0), 0, _mul)
                hj = tl.where((offs_t == j) & (j <= s), hval, hj)
            WE = Gs * W0  # (RMAX, V)
            WA = tl.zeros([T, V], dtype=tl.float32)  # (T, V); rows i > s stay zero
            for j in tl.static_range(T):
                h = tl.sum(tl.where(offs_t == j, hj, 0.0), axis=0)
                dj = tl.sum(tl.where((offs_t == j)[:, None], Dt, 0.0), axis=0)
                WE = WE + (h * tl.sum(tl.where((offs_t == j)[None, :], CE, 0.0), axis=1))[:, None] * dj[None, :]
                WA = WA + (h * tl.sum(tl.where((offs_t == j)[None, :], CA, 0.0), axis=1))[:, None] * dj[None, :]
            amask = offs_t <= s
            WA = tl.where(amask[:, None], WA, 0.0)
            KH = tl.load(rec_k + rrow[:, None] * K + offs_k[None, :], mask=amask[:, None], other=0.0).to(tl.float32)
            U0 = tl.load(pu + base * RMAX * K + offs_r[:, None] * K + offs_k[None, :], mask=emask[:, None], other=0.0).to(tl.float32)
            if n >= RFULL:
                # ---- frozen K0 cut on the stacked rows [entry (c0); appended (s + 1)] -> R
                GEE = tl.dot(WE, tl.trans(WE), input_precision="ieee")  # (RMAX, RMAX)
                GEA = tl.zeros([RMAX, T], dtype=tl.float32)
                GAA = tl.zeros([T, T], dtype=tl.float32)
                for i in tl.static_range(T):
                    wa = tl.sum(tl.where((offs_t == i)[:, None], WA, 0.0), axis=0)
                    GEA = tl.where((offs_t == i)[None, :], tl.sum(WE * wa[None, :], axis=1)[:, None], GEA)
                    GAA = tl.where((offs_t == i)[None, :], tl.sum(WA * wa[None, :], axis=1)[:, None], GAA)
                dE = tl.where(emask, tl.sum(tl.where(offs_r[:, None] == offs_r[None, :], GEE, 0.0), axis=1), -1.0)
                dA = tl.where(amask, tl.sum(tl.where(offs_t[:, None] == offs_t[None, :], GAA, 0.0), axis=1), -1.0)
                # rank = number of rows with a larger diagonal (ties: smaller stacked index first; entry before appended)
                rankE = (tl.sum(((dE[None, :] > dE[:, None]) | ((dE[None, :] == dE[:, None]) & (offs_r[None, :] < offs_r[:, None]))).to(tl.int32), axis=1)
                         + tl.sum((dA[None, :] > dE[:, None]).to(tl.int32), axis=1))
                rankA = (tl.sum(((dA[None, :] > dA[:, None]) | ((dA[None, :] == dA[:, None]) & (offs_t[None, :] < offs_t[:, None]))).to(tl.int32), axis=1)
                         + tl.sum((dE[None, :] >= dA[:, None]).to(tl.int32), axis=1))
                ZE = tl.where((rankE[:, None] == offs_r[None, :]) & (offs_r < R)[None, :] & emask[:, None], 1.0, 0.0)  # (RMAX, RMAX)
                ZA = tl.where((rankA[:, None] == offs_r[None, :]) & (offs_r < R)[None, :] & amask[:, None], 1.0, 0.0)  # (T, RMAX)
                for _ in range(ITERS):
                    # explicit per-appended-row products (the 3-D broadcast-reduce forms were rewritten by the
                    # compiler into K=4 transposed dots; GPU cut quality disagreed with the interpreter, j884332)
                    YE = tl.dot(GEE, ZE, input_precision="ieee")
                    YA = tl.zeros([T, RMAX], dtype=tl.float32)
                    for i in tl.static_range(T):
                        sel = (offs_t == i)
                        ga_i = tl.sum(tl.where(sel[None, :], GEA, 0.0), axis=1)  # (RMAX,) column i of GEA
                        gaa_i = tl.sum(tl.where(sel[None, :], GAA, 0.0), axis=1)  # (T,) column i of GAA (symmetric)
                        za_i = tl.sum(tl.where(sel[:, None], ZA, 0.0), axis=0)  # (RMAX,) row i of ZA
                        YE = YE + ga_i[:, None] * za_i[None, :]
                        ya_i = tl.sum(ZE * ga_i[:, None], axis=0) + tl.sum(ZA * gaa_i[:, None], axis=0)
                        YA = tl.where(sel[:, None], ya_i[None, :], YA)
                    ZE, ZA = _mgs_blocks(YE, YA, offs_r, R, 2, REL_TOL)
                keep = (offs_r < R)[:, None]
                Un = tl.dot(tl.trans(ZE), U0, input_precision="ieee")  # (RMAX, K): row k = sum_r Z[r, k] U[r]
                Wn = tl.dot(tl.trans(ZE), WE, input_precision="ieee")
                for i in tl.static_range(T):
                    za = tl.sum(tl.where((offs_t == i)[:, None], ZA, 0.0), axis=0)  # (RMAX,) row i of ZA
                    Un = Un + za[:, None] * tl.sum(tl.where((offs_t == i)[:, None], KH, 0.0), axis=0)[None, :]
                    Wn = Wn + za[:, None] * tl.sum(tl.where((offs_t == i)[:, None], WA, 0.0), axis=0)[None, :]
                Uout = tl.where(keep, Un, 0.0)
                Wout = tl.where(keep, Wn, 0.0)
                # W8 cadence (B-arm protocol): keep R factors, rows R..n-(RFULL-R)-1 stay zero, count - R carries the
                # accepted inputs past the cut point, so the next cut lands after RFULL - R accepted inputs in total
                n = n - (RFULL - R)
            else:
                # ---- no cut: appended row i goes to pool row c0 + i (< RMAX since n < RFULL <= RMAX)
                Uout = tl.where(emask[:, None], U0, 0.0)
                Wout = tl.where(emask[:, None], WE, 0.0)
                for i in tl.static_range(T):
                    at = (offs_r == c0 + i) & (i <= s)
                    Uout = tl.where(at[:, None], tl.sum(tl.where((offs_t == i)[:, None], KH, 0.0), axis=0)[None, :], Uout)
                    Wout = tl.where(at[:, None], tl.sum(tl.where((offs_t == i)[:, None], WA, 0.0), axis=0)[None, :], Wout)
            a_s = tl.load(rec_a + ((i_n * T + s) * HV + i_hv) * K + offs_k)
            out = dst * HV + i_hv
            tl.store(pa + out * K + offs_k, a_s)
            tl.store(pu + out * RMAX * K + offs_r[:, None] * K + offs_k[None, :], Uout.to(pu.dtype.element_ty))
            tl.store(pw + out * RMAX * V + offs_r[:, None] * V + offs_v[None, :], Wout.to(pw.dtype.element_ty))
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
def _factored_dense_verify_kernel(
    mixed, gate_a, gate_b, A_log, dt_bias, vbar,
    pa, pu, pw, pcount, indices,
    output, rec_k, rec_d, rec_g, rec_b,
    scale,
    MIXED_ROW: tl.constexpr, MIXED_STEP: tl.constexpr,
    A_ROW: tl.constexpr, A_STEP: tl.constexpr, B_ROW: tl.constexpr, B_STEP: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    RMAX: tl.constexpr, T: tl.constexpr, BV: tl.constexpr,
):
    """Verify from compact factors with the DENSE recurrence (stock verify structure, no Gram-Schmidt on the critical
    path).  The factored state is an exact representation of the dense GDN state S = vbar a^T + W^T U, so the entry
    block S[v-block, :] is rebuilt in registers with one fp16 x fp16 -> fp32 product (contraction depth RMAX), then the
    four inputs run S <- g S + d k^T, d = beta (v - g S k), o = S q.  Records k_t, d_t, g_t, beta_t for the commit, which
    converts them to factor updates (sink recurrence, Gram-Schmidt append, due cut).  Pool is read-only."""
    NVB: tl.constexpr = V // BV
    pid = tl.program_id(0)
    i_vb = pid % NVB
    i_n = pid // (HV * NVB)
    i_hv = (pid // NVB) % HV
    i_h = i_hv // (HV // H)
    offs_k = tl.arange(0, K)
    offs_v = i_vb * BV + tl.arange(0, BV)
    offs_r = tl.arange(0, RMAX)
    offs_t = tl.arange(0, T)
    slot = tl.load(indices + i_n).to(tl.int64)
    if slot < 0:
        for t in tl.static_range(T):
            tl.store(output + ((i_n * T + t) * HV + i_hv) * V + offs_v, tl.zeros([BV], dtype=tl.float32).to(output.dtype.element_ty))
        return
    base = slot * HV + i_hv
    # count and factor tiles in one round trip; rows >= count are masked in registers (the pool may hold stale rows)
    c0 = tl.load(pcount + base)
    U0 = tl.load(pu + base * RMAX * K + offs_r[:, None] * K + offs_k[None, :])
    W0 = tl.load(pw + base * RMAX * V + offs_r[:, None] * V + offs_v[None, :])
    a = tl.load(pa + base * K + offs_k)
    vb = tl.load(vbar + i_hv * V + offs_v).to(tl.float32)
    rmask = offs_r < c0
    U0 = tl.where(rmask[:, None], U0, 0.0).to(U0.dtype)
    W0 = tl.where(rmask[:, None], W0, 0.0).to(W0.dtype)
    S = vb[:, None] * a[None, :] + tl.dot(tl.trans(W0), U0)  # (BV, K) fp32
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    for t in tl.static_range(T):
        p = mixed + i_n * MIXED_ROW + t * MIXED_STEP
        q = tl.load(p + i_h * K + offs_k).to(tl.float32)
        k = tl.load(p + H * K + i_h * K + offs_k).to(tl.float32)
        v = tl.load(p + 2 * H * K + i_hv * V + offs_v).to(tl.float32)
        ga = tl.load(gate_a + i_n * A_ROW + t * A_STEP + i_hv).to(tl.float32)
        gb = tl.load(gate_b + i_n * B_ROW + t * B_STEP + i_hv).to(tl.float32)
        x = ga + dt_bias_val
        softplus_x = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
        g = tl.exp(-tl.exp(A_log_val) * softplus_x)
        beta = tl.sigmoid(gb).to(gate_b.dtype.element_ty).to(tl.float32)
        qn = q / tl.sqrt(tl.sum(q * q) + 1e-6) * scale
        kn = k / tl.sqrt(tl.sum(k * k) + 1e-6)
        d = beta * (v - g * tl.sum(S * kn[None, :], axis=1))
        S = g * S + d[:, None] * kn[None, :]
        o = tl.sum(S * qn[None, :], axis=1)
        row = (i_n * T + t) * HV + i_hv
        tl.store(output + row * V + offs_v, o.to(output.dtype.element_ty))
        tl.store(rec_d + row * V + offs_v, d)
        if i_vb == 0:
            tl.store(rec_k + row * K + offs_k, kn)
            tl.store(rec_g + row, g)
            tl.store(rec_b + row, beta)


@triton.jit
def _factored_dense_verify_wy_kernel(
    mixed, gate_a, gate_b, A_log, dt_bias, vbar,
    pa, pu, pw, pcount, indices,
    output, rec_k, rec_d, rec_g, rec_b,
    scale,
    MIXED_ROW: tl.constexpr, MIXED_STEP: tl.constexpr,
    A_ROW: tl.constexpr, A_STEP: tl.constexpr, B_ROW: tl.constexpr, B_STEP: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    RMAX: tl.constexpr, T: tl.constexpr, BV: tl.constexpr, DOT_PREC: tl.constexpr = "tf32x3",
):
    """Same maths as `_factored_dense_verify_kernel` without materialising the (BV, K) state: for the 2T vectors
    x in [k_0..k_{T-1}, q_0..q_{T-1}], S0 x = vbar (a.x) + W0^T (U0 x) through the rank-RMAX factors (two small
    products), and the T-step recurrence S_t = g_t S_{t-1} + d_t k_t^T is unrolled in WY form:
      S_{t-1} k_t = G_{t-1} S0 k_t + sum_{j<t} h_{j,t-1} (k_j.k_t) d_j,   d_t = beta_t (v_t - g_t S_{t-1} k_t),
      o_t = S_t q_t = G_t S0 q_t + sum_{j<=t} h_{j,t} (k_j.q_t) d_j,
    with G_t = prod_{i<=t} g_i and h_{j,t} = prod_{j<i<=t} g_i.  No reduction over K per input."""
    NVB: tl.constexpr = V // BV
    X2: tl.constexpr = 16  # 2T = 8 vectors padded to the smallest dot tile
    pid = tl.program_id(0)
    i_vb = pid % NVB
    i_n = pid // (HV * NVB)
    i_hv = (pid // NVB) % HV
    i_h = i_hv // (HV // H)
    offs_k = tl.arange(0, K)
    offs_v = i_vb * BV + tl.arange(0, BV)
    offs_r = tl.arange(0, RMAX)
    offs_t = tl.arange(0, T)
    offs_x = tl.arange(0, X2)
    slot = tl.load(indices + i_n).to(tl.int64)
    if slot < 0:
        for t in tl.static_range(T):
            tl.store(output + ((i_n * T + t) * HV + i_hv) * V + offs_v, tl.zeros([BV], dtype=tl.float32).to(output.dtype.element_ty))
        return
    base = slot * HV + i_hv
    c0 = tl.load(pcount + base)
    rmask = offs_r < c0
    # ---- the 2T input vectors as rows of one (X2, K) tile: rows 0..T-1 = k_t, rows T..2T-1 = q_t (unit-normalised)
    trow = offs_x % T
    is_q = (offs_x >= T) & (offs_x < 2 * T)
    live = offs_x < 2 * T
    p = mixed + i_n * MIXED_ROW + trow[:, None] * MIXED_STEP
    col = tl.where(is_q, i_h * K, H * K + i_h * K)
    X = tl.load(p + col[:, None] + offs_k[None, :], mask=live[:, None], other=0.0).to(tl.float32)
    X = X / tl.sqrt(tl.sum(X * X, axis=1) + 1e-6)[:, None]
    X = tl.where(is_q[:, None], X * scale, X)
    U0 = tl.load(pu + base * RMAX * K + offs_r[:, None] * K + offs_k[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    W0 = tl.load(pw + base * RMAX * V + offs_r[:, None] * V + offs_v[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    a = tl.load(pa + base * K + offs_k)
    vb = tl.load(vbar + i_hv * V + offs_v).to(tl.float32)
    UX = tl.dot(U0, tl.trans(X), input_precision=DOT_PREC)  # (RMAX, X2)
    AX = tl.sum(X * a[None, :], axis=1)  # (X2,)
    S0X = vb[:, None] * AX[None, :] + tl.dot(tl.trans(W0), UX, input_precision=DOT_PREC)  # (BV, X2)
    GX = tl.dot(X, tl.trans(X), input_precision=DOT_PREC)  # (X2, X2): k_i.k_j, k_i.q_j
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    ga = tl.load(gate_a + i_n * A_ROW + offs_t * A_STEP + i_hv).to(tl.float32)
    gb = tl.load(gate_b + i_n * B_ROW + offs_t * B_STEP + i_hv).to(tl.float32)
    xg = ga + dt_bias_val
    softplus_x = tl.where(xg <= 20.0, tl.log(1.0 + tl.exp(xg)), xg)
    gts = tl.exp(-tl.exp(A_log_val) * softplus_x)  # (T,)
    bts = tl.sigmoid(gb).to(gate_b.dtype.element_ty).to(tl.float32)  # (T,)
    D = tl.zeros([T, BV], dtype=tl.float32)
    hv_ = tl.zeros([T], dtype=tl.float32)  # h_{j,t-1} for j < t
    G = 1.0
    for t in tl.static_range(T):
        g = tl.sum(tl.where(offs_t == t, gts, 0.0), axis=0)
        b = tl.sum(tl.where(offs_t == t, bts, 0.0), axis=0)
        kk = tl.sum(tl.where((offs_x == t)[None, :] & (offs_x < T)[:, None], GX, 0.0), axis=1)  # (X2,) k_j.k_t over rows j
        kq = tl.sum(tl.where((offs_x == T + t)[None, :] & (offs_x < T)[:, None], GX, 0.0), axis=1)  # (X2,) k_j.q_t
        kkT = tl.sum(tl.where(offs_x[:, None] == offs_t[None, :], kk[:, None], 0.0), axis=0)  # (T,)
        kqT = tl.sum(tl.where(offs_x[:, None] == offs_t[None, :], kq[:, None], 0.0), axis=0)  # (T,)
        s0k = tl.sum(tl.where((offs_x == t)[None, :], S0X, 0.0), axis=1)  # (BV,)
        s0q = tl.sum(tl.where((offs_x == T + t)[None, :], S0X, 0.0), axis=1)  # (BV,)
        sk = G * s0k + tl.sum(D * (hv_ * kkT)[:, None], axis=0)
        v = tl.load(mixed + i_n * MIXED_ROW + t * MIXED_STEP + 2 * H * K + i_hv * V + offs_v).to(tl.float32)
        d = b * (v - g * sk)
        D = tl.where((offs_t == t)[:, None], d[None, :], D)
        hv_ = tl.where(offs_t == t, 1.0, hv_ * g)  # now h_{j,t} for j <= t
        G = G * g
        o = G * s0q + tl.sum(D * (hv_ * kqT)[:, None], axis=0)
        row = (i_n * T + t) * HV + i_hv
        tl.store(output + row * V + offs_v, o.to(output.dtype.element_ty))
        tl.store(rec_d + row * V + offs_v, d)
        if i_vb == 0:
            kn = tl.sum(tl.where((offs_x == t)[:, None], X, 0.0), axis=0)
            tl.store(rec_k + row * K + offs_k, kn)
            tl.store(rec_g + row, g)
            tl.store(rec_b + row, b)


@triton.jit
def _factored_commit_dense_kernel(
    pa, pu, pw, pcount, stale, dense_of, dense_required, prefix_valid, vbar,
    rec_k, rec_d, rec_g, rec_b,
    src_slots, steps, track_slots, track_steps, gs_eps,
    LAYER_A: tl.constexpr, LAYER_U: tl.constexpr, LAYER_W: tl.constexpr, LAYER_COUNT: tl.constexpr,
    LAYER_VBAR: tl.constexpr, LAYER_RK: tl.constexpr, LAYER_RD: tl.constexpr, LAYER_RG: tl.constexpr,
    LAYER_RB: tl.constexpr,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, RMAX: tl.constexpr,
    R: tl.constexpr, RFULL: tl.constexpr, T: tl.constexpr,
    ITERS: tl.constexpr, REL_TOL: tl.constexpr,
    HAS_TRACK: tl.constexpr, HAS_DENSE_OF: tl.constexpr, HAS_DENSE_REQUIRED: tl.constexpr,
    HAS_PREFIX_VALID: tl.constexpr,
):
    """Commit for the dense verify: turn the recorded (k_t, d_t, g_t, beta_t) into the factored update of the frozen
    step primitive -- sink a_t = g (a - beta k (k.a)) + beta k, content delta_t = d_t - beta (1 - g k.a) vbar,
    Gram-Schmidt append of k_t (twice-is-enough, FP16-rounded rows) -- then rebuild W for the accepted input in block
    form, cut r+m.. -> r when due and publish (same second phase as `_factored_commit_block_kernel`)."""
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
    vbar += layer * LAYER_VBAR
    rec_k += layer * LAYER_RK
    rec_d += layer * LAYER_RD
    rec_g += layer * LAYER_RG
    rec_b += layer * LAYER_RB
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V)
    offs_r = tl.arange(0, RMAX)
    offs_t = tl.arange(0, T)
    base = slot * HV + i_hv
    c0 = tl.load(pcount + base)
    emask = offs_r < c0
    rrow = (i_n * T + offs_t) * HV + i_hv  # (T,)
    gts = tl.load(rec_g + rrow)
    bts = tl.load(rec_b + rrow)
    vb = tl.load(vbar + i_hv * V + offs_v).to(tl.float32)
    # ---- phase 1: factor chain over the four recorded inputs (inputs after the accepted one are never published)
    U0 = tl.load(pu + base * RMAX * K + offs_r[:, None] * K + offs_k[None, :], mask=emask[:, None], other=0.0).to(tl.float32)
    a = tl.load(pa + base * K + offs_k)
    KHall = tl.zeros([T, K], dtype=tl.float32)
    AR = tl.zeros([T, K], dtype=tl.float32)
    Dt = tl.zeros([T, V], dtype=tl.float32)
    CE = tl.zeros([RMAX, T], dtype=tl.float32)
    CA = tl.zeros([T, T], dtype=tl.float32)
    for j in tl.static_range(T):
        row = (i_n * T + j) * HV + i_hv
        kn = tl.load(rec_k + row * K + offs_k)
        dd = tl.load(rec_d + row * V + offs_v)
        g = tl.sum(tl.where(offs_t == j, gts, 0.0), axis=0)
        b = tl.sum(tl.where(offs_t == j, bts, 0.0), axis=0)
        ka = tl.sum(kn * a, axis=0)
        a = g * (a - b * kn * ka) + b * kn
        AR = tl.where((offs_t == j)[:, None], a[None, :], AR)
        Dt = tl.where((offs_t == j)[:, None], (dd - b * (1.0 - g * ka) * vb)[None, :], Dt)
        cb = tl.sum(U0 * kn[None, :], axis=1)
        ca = tl.sum(KHall * kn[None, :], axis=1)
        kp = kn - tl.sum(U0 * cb[:, None], axis=0) - tl.sum(KHall * ca[:, None], axis=0)
        nrm2 = tl.sum(kp * kp, axis=0)
        if nrm2 < 0.25:
            cb2 = tl.sum(U0 * kp[None, :], axis=1)
            ca2 = tl.sum(KHall * kp[None, :], axis=1)
            kp = kp - tl.sum(U0 * cb2[:, None], axis=0) - tl.sum(KHall * ca2[:, None], axis=0)
            cb = cb + cb2
            ca = ca + ca2
            nrm2 = tl.sum(kp * kp, axis=0)
        nrm = tl.sqrt(nrm2)
        keep_k = nrm > gs_eps
        khat = tl.where(keep_k, kp / tl.maximum(nrm, gs_eps), 0.0)
        clast = tl.where(keep_k, nrm, 0.0)
        KHall = tl.where((offs_t == j)[:, None], khat.to(tl.float16).to(tl.float32)[None, :], KHall)
        CE = tl.where((offs_t == j)[None, :], cb[:, None], CE)
        CA = tl.where((offs_t == j)[None, :], (ca + tl.where(offs_t == j, clast, 0.0))[:, None], CA)
    W0 = tl.load(pw + base * RMAX * V + offs_r[:, None] * V + offs_v[None, :], mask=emask[:, None], other=0.0).to(tl.float32)
    # ---- phase 2: rebuild, cut when due, publish (verbatim from the block commit)
    for which in range(2):  # track slot first (never beyond the accepted prefix), then the request slot
        dst = tl.where(which == 0, tslot, slot)
        s = tl.where(which == 0, tstep, step)
        if (dst >= 0) & (s >= 0):
            n = c0 + s + 1
            Gs = tl.reduce(tl.where(offs_t <= s, gts, 1.0), 0, _mul)
            # h_j = prod_{j < i <= s} g_i for j <= s, 0 for j > s
            hj = tl.zeros([T], dtype=tl.float32)
            for j in tl.static_range(T):
                hval = tl.reduce(tl.where((offs_t > j) & (offs_t <= s), gts, 1.0), 0, _mul)
                hj = tl.where((offs_t == j) & (j <= s), hval, hj)
            WE = Gs * W0  # (RMAX, V)
            WA = tl.zeros([T, V], dtype=tl.float32)  # (T, V); rows i > s stay zero
            for j in tl.static_range(T):
                h = tl.sum(tl.where(offs_t == j, hj, 0.0), axis=0)
                dj = tl.sum(tl.where((offs_t == j)[:, None], Dt, 0.0), axis=0)
                WE = WE + (h * tl.sum(tl.where((offs_t == j)[None, :], CE, 0.0), axis=1))[:, None] * dj[None, :]
                WA = WA + (h * tl.sum(tl.where((offs_t == j)[None, :], CA, 0.0), axis=1))[:, None] * dj[None, :]
            amask = offs_t <= s
            WA = tl.where(amask[:, None], WA, 0.0)
            KH = tl.where(amask[:, None], KHall, 0.0)
            U0 = tl.load(pu + base * RMAX * K + offs_r[:, None] * K + offs_k[None, :], mask=emask[:, None], other=0.0).to(tl.float32)
            if n >= RFULL:
                # ---- frozen K0 cut on the stacked rows [entry (c0); appended (s + 1)] -> R
                GEE = tl.dot(WE, tl.trans(WE), input_precision="ieee")  # (RMAX, RMAX)
                GEA = tl.zeros([RMAX, T], dtype=tl.float32)
                GAA = tl.zeros([T, T], dtype=tl.float32)
                for i in tl.static_range(T):
                    wa = tl.sum(tl.where((offs_t == i)[:, None], WA, 0.0), axis=0)
                    GEA = tl.where((offs_t == i)[None, :], tl.sum(WE * wa[None, :], axis=1)[:, None], GEA)
                    GAA = tl.where((offs_t == i)[None, :], tl.sum(WA * wa[None, :], axis=1)[:, None], GAA)
                dE = tl.where(emask, tl.sum(tl.where(offs_r[:, None] == offs_r[None, :], GEE, 0.0), axis=1), -1.0)
                dA = tl.where(amask, tl.sum(tl.where(offs_t[:, None] == offs_t[None, :], GAA, 0.0), axis=1), -1.0)
                # rank = number of rows with a larger diagonal (ties: smaller stacked index first; entry before appended)
                rankE = (tl.sum(((dE[None, :] > dE[:, None]) | ((dE[None, :] == dE[:, None]) & (offs_r[None, :] < offs_r[:, None]))).to(tl.int32), axis=1)
                         + tl.sum((dA[None, :] > dE[:, None]).to(tl.int32), axis=1))
                rankA = (tl.sum(((dA[None, :] > dA[:, None]) | ((dA[None, :] == dA[:, None]) & (offs_t[None, :] < offs_t[:, None]))).to(tl.int32), axis=1)
                         + tl.sum((dE[None, :] >= dA[:, None]).to(tl.int32), axis=1))
                ZE = tl.where((rankE[:, None] == offs_r[None, :]) & (offs_r < R)[None, :] & emask[:, None], 1.0, 0.0)  # (RMAX, RMAX)
                ZA = tl.where((rankA[:, None] == offs_r[None, :]) & (offs_r < R)[None, :] & amask[:, None], 1.0, 0.0)  # (T, RMAX)
                for _ in range(ITERS):
                    # explicit per-appended-row products (the 3-D broadcast-reduce forms were rewritten by the
                    # compiler into K=4 transposed dots; GPU cut quality disagreed with the interpreter, j884332)
                    YE = tl.dot(GEE, ZE, input_precision="ieee")
                    YA = tl.zeros([T, RMAX], dtype=tl.float32)
                    for i in tl.static_range(T):
                        sel = (offs_t == i)
                        ga_i = tl.sum(tl.where(sel[None, :], GEA, 0.0), axis=1)  # (RMAX,) column i of GEA
                        gaa_i = tl.sum(tl.where(sel[None, :], GAA, 0.0), axis=1)  # (T,) column i of GAA (symmetric)
                        za_i = tl.sum(tl.where(sel[:, None], ZA, 0.0), axis=0)  # (RMAX,) row i of ZA
                        YE = YE + ga_i[:, None] * za_i[None, :]
                        ya_i = tl.sum(ZE * ga_i[:, None], axis=0) + tl.sum(ZA * gaa_i[:, None], axis=0)
                        YA = tl.where(sel[:, None], ya_i[None, :], YA)
                    ZE, ZA = _mgs_blocks(YE, YA, offs_r, R, 2, REL_TOL)
                keep = (offs_r < R)[:, None]
                Un = tl.dot(tl.trans(ZE), U0, input_precision="ieee")  # (RMAX, K): row k = sum_r Z[r, k] U[r]
                Wn = tl.dot(tl.trans(ZE), WE, input_precision="ieee")
                for i in tl.static_range(T):
                    za = tl.sum(tl.where((offs_t == i)[:, None], ZA, 0.0), axis=0)  # (RMAX,) row i of ZA
                    Un = Un + za[:, None] * tl.sum(tl.where((offs_t == i)[:, None], KH, 0.0), axis=0)[None, :]
                    Wn = Wn + za[:, None] * tl.sum(tl.where((offs_t == i)[:, None], WA, 0.0), axis=0)[None, :]
                Uout = tl.where(keep, Un, 0.0)
                Wout = tl.where(keep, Wn, 0.0)
                # W8 cadence (B-arm protocol): keep R factors, rows R..n-(RFULL-R)-1 stay zero, count - R carries the
                # accepted inputs past the cut point, so the next cut lands after RFULL - R accepted inputs in total
                n = n - (RFULL - R)
            else:
                # ---- no cut: appended row i goes to pool row c0 + i (< RMAX since n < RFULL <= RMAX)
                Uout = tl.where(emask[:, None], U0, 0.0)
                Wout = tl.where(emask[:, None], WE, 0.0)
                for i in tl.static_range(T):
                    at = (offs_r == c0 + i) & (i <= s)
                    Uout = tl.where(at[:, None], tl.sum(tl.where((offs_t == i)[:, None], KH, 0.0), axis=0)[None, :], Uout)
                    Wout = tl.where(at[:, None], tl.sum(tl.where((offs_t == i)[:, None], WA, 0.0), axis=0)[None, :], Wout)
            a_s = tl.sum(tl.where((offs_t == s)[:, None], AR, 0.0), axis=0)
            out = dst * HV + i_hv
            tl.store(pa + out * K + offs_k, a_s)
            tl.store(pu + out * RMAX * K + offs_r[:, None] * K + offs_k[None, :], Uout.to(pu.dtype.element_ty))
            tl.store(pw + out * RMAX * V + offs_r[:, None] * V + offs_v[None, :], Wout.to(pw.dtype.element_ty))
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
def _commit_chain_kernel(
    pa, pu, pw, pcount, stale, dense_of, dense_required, prefix_valid, vbar,
    rec_k, rec_d, rec_g, rec_b, rec_c, rec_z,
    src_slots, steps, track_slots, track_steps, gs_eps,
    LAYER_A: tl.constexpr, LAYER_U: tl.constexpr, LAYER_W: tl.constexpr, LAYER_COUNT: tl.constexpr,
    LAYER_VBAR: tl.constexpr, LAYER_RK: tl.constexpr, LAYER_RD: tl.constexpr, LAYER_RG: tl.constexpr,
    LAYER_RB: tl.constexpr, LAYER_RC: tl.constexpr, LAYER_RZ: tl.constexpr,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, RMAX: tl.constexpr,
    R: tl.constexpr, RFULL: tl.constexpr, T: tl.constexpr,
    ITERS: tl.constexpr, REL_TOL: tl.constexpr,
    HAS_TRACK: tl.constexpr, HAS_DENSE_OF: tl.constexpr, HAS_DENSE_REQUIRED: tl.constexpr,
    HAS_PREFIX_VALID: tl.constexpr,
):
    """Split commit 1/3 (1 warp): the factor chain of `_factored_commit_dense_kernel` phase 1.  Overwrites the verify
    records in place (k_t -> FP16-rounded khat_t, d_t -> content delta_t), writes the cfull coefficients (CE, CA) to
    scratch and publishes the sink vector a for both destinations (the cut never changes a)."""
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
    vbar += layer * LAYER_VBAR
    rec_k += layer * LAYER_RK
    rec_d += layer * LAYER_RD
    rec_g += layer * LAYER_RG
    rec_b += layer * LAYER_RB
    rec_c += layer * LAYER_RC
    rec_z += layer * LAYER_RZ
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V)
    offs_r = tl.arange(0, RMAX)
    offs_t = tl.arange(0, T)
    base = slot * HV + i_hv
    c0 = tl.load(pcount + base)
    emask = offs_r < c0
    rrow = (i_n * T + offs_t) * HV + i_hv  # (T,)
    gts = tl.load(rec_g + rrow)
    bts = tl.load(rec_b + rrow)
    # coefficient scratch per (row, head): CE (RMAX, T) then CA (T, T)
    cbase = (i_n * HV + i_hv) * (RMAX + T) * T
    vb = tl.load(vbar + i_hv * V + offs_v).to(tl.float32)
    # ---- phase 1: factor chain over the four recorded inputs (inputs after the accepted one are never published)
    U0 = tl.load(pu + base * RMAX * K + offs_r[:, None] * K + offs_k[None, :], mask=emask[:, None], other=0.0).to(tl.float32)
    a = tl.load(pa + base * K + offs_k)
    KHall = tl.zeros([T, K], dtype=tl.float32)
    AR = tl.zeros([T, K], dtype=tl.float32)
    Dt = tl.zeros([T, V], dtype=tl.float32)
    CE = tl.zeros([RMAX, T], dtype=tl.float32)
    CA = tl.zeros([T, T], dtype=tl.float32)
    for j in tl.static_range(T):
        row = (i_n * T + j) * HV + i_hv
        kn = tl.load(rec_k + row * K + offs_k)
        dd = tl.load(rec_d + row * V + offs_v)
        g = tl.sum(tl.where(offs_t == j, gts, 0.0), axis=0)
        b = tl.sum(tl.where(offs_t == j, bts, 0.0), axis=0)
        ka = tl.sum(kn * a, axis=0)
        a = g * (a - b * kn * ka) + b * kn
        AR = tl.where((offs_t == j)[:, None], a[None, :], AR)
        Dt = tl.where((offs_t == j)[:, None], (dd - b * (1.0 - g * ka) * vb)[None, :], Dt)
        cb = tl.sum(U0 * kn[None, :], axis=1)
        ca = tl.sum(KHall * kn[None, :], axis=1)
        kp = kn - tl.sum(U0 * cb[:, None], axis=0) - tl.sum(KHall * ca[:, None], axis=0)
        nrm2 = tl.sum(kp * kp, axis=0)
        if nrm2 < 0.25:
            cb2 = tl.sum(U0 * kp[None, :], axis=1)
            ca2 = tl.sum(KHall * kp[None, :], axis=1)
            kp = kp - tl.sum(U0 * cb2[:, None], axis=0) - tl.sum(KHall * ca2[:, None], axis=0)
            cb = cb + cb2
            ca = ca + ca2
            nrm2 = tl.sum(kp * kp, axis=0)
        nrm = tl.sqrt(nrm2)
        keep_k = nrm > gs_eps
        khat = tl.where(keep_k, kp / tl.maximum(nrm, gs_eps), 0.0)
        clast = tl.where(keep_k, nrm, 0.0)
        KHall = tl.where((offs_t == j)[:, None], khat.to(tl.float16).to(tl.float32)[None, :], KHall)
        CE = tl.where((offs_t == j)[None, :], cb[:, None], CE)
        CA = tl.where((offs_t == j)[None, :], (ca + tl.where(offs_t == j, clast, 0.0))[:, None], CA)
    tl.store(rec_c + cbase + offs_r[:, None] * T + offs_t[None, :], CE)
    tl.store(rec_c + cbase + RMAX * T + offs_t[:, None] * T + offs_t[None, :], CA)
    tl.store(rec_k + rrow[:, None] * K + offs_k[None, :], KHall)
    tl.store(rec_d + rrow[:, None] * V + offs_v[None, :], Dt)
    for which in range(2):
        dst = tl.where(which == 0, tslot, slot)
        s = tl.where(which == 0, tstep, step)
        if (dst >= 0) & (s >= 0):
            a_s = tl.sum(tl.where((offs_t == s)[:, None], AR, 0.0), axis=0)
            tl.store(pa + (dst * HV + i_hv) * K + offs_k, a_s)


@triton.jit
def _commit_cutsolve_kernel(
    pa, pu, pw, pcount, stale, dense_of, dense_required, prefix_valid, vbar,
    rec_k, rec_d, rec_g, rec_b, rec_c, rec_z,
    src_slots, steps, track_slots, track_steps, gs_eps,
    LAYER_A: tl.constexpr, LAYER_U: tl.constexpr, LAYER_W: tl.constexpr, LAYER_COUNT: tl.constexpr,
    LAYER_VBAR: tl.constexpr, LAYER_RK: tl.constexpr, LAYER_RD: tl.constexpr, LAYER_RG: tl.constexpr,
    LAYER_RB: tl.constexpr, LAYER_RC: tl.constexpr, LAYER_RZ: tl.constexpr,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, RMAX: tl.constexpr,
    R: tl.constexpr, RFULL: tl.constexpr, T: tl.constexpr,
    ITERS: tl.constexpr, REL_TOL: tl.constexpr,
    HAS_TRACK: tl.constexpr, HAS_DENSE_OF: tl.constexpr, HAS_DENSE_REQUIRED: tl.constexpr,
    HAS_PREFIX_VALID: tl.constexpr,
):
    """Split commit 2/3 (1 warp, only programs with a due cut do work): rebuild W for the accepted input in block form,
    form the Gram blocks and run the frozen subspace iteration + MGS2 on (RMAX|T, RMAX) tiles -- every reduction stays
    inside one warp; write the kept directions Z (entry rows, appended rows) x R to scratch."""
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
    vbar += layer * LAYER_VBAR
    rec_k += layer * LAYER_RK
    rec_d += layer * LAYER_RD
    rec_g += layer * LAYER_RG
    rec_b += layer * LAYER_RB
    rec_c += layer * LAYER_RC
    rec_z += layer * LAYER_RZ
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V)
    offs_r = tl.arange(0, RMAX)
    offs_t = tl.arange(0, T)
    base = slot * HV + i_hv
    c0 = tl.load(pcount + base)
    emask = offs_r < c0
    rrow = (i_n * T + offs_t) * HV + i_hv  # (T,)
    gts = tl.load(rec_g + rrow)
    bts = tl.load(rec_b + rrow)
    # coefficient scratch per (row, head): CE (RMAX, T) then CA (T, T)
    cbase = (i_n * HV + i_hv) * (RMAX + T) * T
    W0 = tl.load(pw + base * RMAX * V + offs_r[:, None] * V + offs_v[None, :], mask=emask[:, None], other=0.0).to(tl.float32)
    CE = tl.load(rec_c + cbase + offs_r[:, None] * T + offs_t[None, :])  # (RMAX, T)
    CA = tl.load(rec_c + cbase + RMAX * T + offs_t[:, None] * T + offs_t[None, :])  # (T, T)
    Dt = tl.load(rec_d + rrow[:, None] * V + offs_v[None, :])  # (T, V) delta_f (written by the chain kernel)
    for which in range(2):
        dst = tl.where(which == 0, tslot, slot)
        s = tl.where(which == 0, tstep, step)
        if (dst >= 0) & (s >= 0) & (c0 + s + 1 >= RFULL):
            Gs = tl.reduce(tl.where(offs_t <= s, gts, 1.0), 0, _mul)
            # h_j = prod_{j < i <= s} g_i for j <= s, 0 for j > s
            hj = tl.zeros([T], dtype=tl.float32)
            for j in tl.static_range(T):
                hval = tl.reduce(tl.where((offs_t > j) & (offs_t <= s), gts, 1.0), 0, _mul)
                hj = tl.where((offs_t == j) & (j <= s), hval, hj)
            WE = Gs * W0  # (RMAX, V)
            WA = tl.zeros([T, V], dtype=tl.float32)  # (T, V); rows i > s stay zero
            for j in tl.static_range(T):
                h = tl.sum(tl.where(offs_t == j, hj, 0.0), axis=0)
                dj = tl.sum(tl.where((offs_t == j)[:, None], Dt, 0.0), axis=0)
                WE = WE + (h * tl.sum(tl.where((offs_t == j)[None, :], CE, 0.0), axis=1))[:, None] * dj[None, :]
                WA = WA + (h * tl.sum(tl.where((offs_t == j)[None, :], CA, 0.0), axis=1))[:, None] * dj[None, :]
            amask = offs_t <= s
            WA = tl.where(amask[:, None], WA, 0.0)
                # ---- frozen K0 cut on the stacked rows [entry (c0); appended (s + 1)] -> R
            GEE = tl.dot(WE, tl.trans(WE), input_precision="ieee")  # (RMAX, RMAX)
            GEA = tl.zeros([RMAX, T], dtype=tl.float32)
            GAA = tl.zeros([T, T], dtype=tl.float32)
            for i in tl.static_range(T):
                wa = tl.sum(tl.where((offs_t == i)[:, None], WA, 0.0), axis=0)
                GEA = tl.where((offs_t == i)[None, :], tl.sum(WE * wa[None, :], axis=1)[:, None], GEA)
                GAA = tl.where((offs_t == i)[None, :], tl.sum(WA * wa[None, :], axis=1)[:, None], GAA)
            dE = tl.where(emask, tl.sum(tl.where(offs_r[:, None] == offs_r[None, :], GEE, 0.0), axis=1), -1.0)
            dA = tl.where(amask, tl.sum(tl.where(offs_t[:, None] == offs_t[None, :], GAA, 0.0), axis=1), -1.0)
            # rank = number of rows with a larger diagonal (ties: smaller stacked index first; entry before appended)
            rankE = (tl.sum(((dE[None, :] > dE[:, None]) | ((dE[None, :] == dE[:, None]) & (offs_r[None, :] < offs_r[:, None]))).to(tl.int32), axis=1)
                     + tl.sum((dA[None, :] > dE[:, None]).to(tl.int32), axis=1))
            rankA = (tl.sum(((dA[None, :] > dA[:, None]) | ((dA[None, :] == dA[:, None]) & (offs_t[None, :] < offs_t[:, None]))).to(tl.int32), axis=1)
                     + tl.sum((dE[None, :] >= dA[:, None]).to(tl.int32), axis=1))
            ZE = tl.where((rankE[:, None] == offs_r[None, :]) & (offs_r < R)[None, :] & emask[:, None], 1.0, 0.0)  # (RMAX, RMAX)
            ZA = tl.where((rankA[:, None] == offs_r[None, :]) & (offs_r < R)[None, :] & amask[:, None], 1.0, 0.0)  # (T, RMAX)
            for _ in range(ITERS):
                # explicit per-appended-row products (the 3-D broadcast-reduce forms were rewritten by the
                # compiler into K=4 transposed dots; GPU cut quality disagreed with the interpreter, j884332)
                YE = tl.dot(GEE, ZE, input_precision="ieee")
                YA = tl.zeros([T, RMAX], dtype=tl.float32)
                for i in tl.static_range(T):
                    sel = (offs_t == i)
                    ga_i = tl.sum(tl.where(sel[None, :], GEA, 0.0), axis=1)  # (RMAX,) column i of GEA
                    gaa_i = tl.sum(tl.where(sel[None, :], GAA, 0.0), axis=1)  # (T,) column i of GAA (symmetric)
                    za_i = tl.sum(tl.where(sel[:, None], ZA, 0.0), axis=0)  # (RMAX,) row i of ZA
                    YE = YE + ga_i[:, None] * za_i[None, :]
                    ya_i = tl.sum(ZE * ga_i[:, None], axis=0) + tl.sum(ZA * gaa_i[:, None], axis=0)
                    YA = tl.where(sel[:, None], ya_i[None, :], YA)
                ZE, ZA = _mgs_blocks(YE, YA, offs_r, R, 2, REL_TOL)
            zb = ((i_n * 2 + which) * HV + i_hv) * (RMAX + T) * R
            tl.store(rec_z + zb + offs_r[:, None] * R + offs_r[None, :], ZE, mask=(offs_r < R)[None, :])
            tl.store(rec_z + zb + RMAX * R + offs_t[:, None] * R + offs_r[None, :], ZA, mask=(offs_r < R)[None, :])


@triton.jit
def _commit_publish_kernel(
    pa, pu, pw, pcount, stale, dense_of, dense_required, prefix_valid, vbar,
    rec_k, rec_d, rec_g, rec_b, rec_c, rec_z,
    src_slots, steps, track_slots, track_steps, gs_eps,
    LAYER_A: tl.constexpr, LAYER_U: tl.constexpr, LAYER_W: tl.constexpr, LAYER_COUNT: tl.constexpr,
    LAYER_VBAR: tl.constexpr, LAYER_RK: tl.constexpr, LAYER_RD: tl.constexpr, LAYER_RG: tl.constexpr,
    LAYER_RB: tl.constexpr, LAYER_RC: tl.constexpr, LAYER_RZ: tl.constexpr,
    HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, RMAX: tl.constexpr,
    R: tl.constexpr, RFULL: tl.constexpr, T: tl.constexpr,
    ITERS: tl.constexpr, REL_TOL: tl.constexpr,
    HAS_TRACK: tl.constexpr, HAS_DENSE_OF: tl.constexpr, HAS_DENSE_REQUIRED: tl.constexpr,
    HAS_PREFIX_VALID: tl.constexpr,
):
    """Split commit 3/3: rebuild U/W for the accepted input, apply the solved Z when the cut is due, publish."""
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
    vbar += layer * LAYER_VBAR
    rec_k += layer * LAYER_RK
    rec_d += layer * LAYER_RD
    rec_g += layer * LAYER_RG
    rec_b += layer * LAYER_RB
    rec_c += layer * LAYER_RC
    rec_z += layer * LAYER_RZ
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V)
    offs_r = tl.arange(0, RMAX)
    offs_t = tl.arange(0, T)
    base = slot * HV + i_hv
    c0 = tl.load(pcount + base)
    emask = offs_r < c0
    rrow = (i_n * T + offs_t) * HV + i_hv  # (T,)
    gts = tl.load(rec_g + rrow)
    bts = tl.load(rec_b + rrow)
    # coefficient scratch per (row, head): CE (RMAX, T) then CA (T, T)
    cbase = (i_n * HV + i_hv) * (RMAX + T) * T
    W0 = tl.load(pw + base * RMAX * V + offs_r[:, None] * V + offs_v[None, :], mask=emask[:, None], other=0.0).to(tl.float32)
    U0 = tl.load(pu + base * RMAX * K + offs_r[:, None] * K + offs_k[None, :], mask=emask[:, None], other=0.0).to(tl.float32)
    CE = tl.load(rec_c + cbase + offs_r[:, None] * T + offs_t[None, :])  # (RMAX, T)
    CA = tl.load(rec_c + cbase + RMAX * T + offs_t[:, None] * T + offs_t[None, :])  # (T, T)
    Dt = tl.load(rec_d + rrow[:, None] * V + offs_v[None, :])  # (T, V) delta_f (written by the chain kernel)
    KHall = tl.load(rec_k + rrow[:, None] * K + offs_k[None, :])
    for which in range(2):
        dst = tl.where(which == 0, tslot, slot)
        s = tl.where(which == 0, tstep, step)
        if (dst >= 0) & (s >= 0):
            n = c0 + s + 1
            Gs = tl.reduce(tl.where(offs_t <= s, gts, 1.0), 0, _mul)
            # h_j = prod_{j < i <= s} g_i for j <= s, 0 for j > s
            hj = tl.zeros([T], dtype=tl.float32)
            for j in tl.static_range(T):
                hval = tl.reduce(tl.where((offs_t > j) & (offs_t <= s), gts, 1.0), 0, _mul)
                hj = tl.where((offs_t == j) & (j <= s), hval, hj)
            WE = Gs * W0  # (RMAX, V)
            WA = tl.zeros([T, V], dtype=tl.float32)  # (T, V); rows i > s stay zero
            for j in tl.static_range(T):
                h = tl.sum(tl.where(offs_t == j, hj, 0.0), axis=0)
                dj = tl.sum(tl.where((offs_t == j)[:, None], Dt, 0.0), axis=0)
                WE = WE + (h * tl.sum(tl.where((offs_t == j)[None, :], CE, 0.0), axis=1))[:, None] * dj[None, :]
                WA = WA + (h * tl.sum(tl.where((offs_t == j)[None, :], CA, 0.0), axis=1))[:, None] * dj[None, :]
            amask = offs_t <= s
            WA = tl.where(amask[:, None], WA, 0.0)
            KH = tl.where(amask[:, None], KHall, 0.0)
            if n >= RFULL:
                zb = ((i_n * 2 + which) * HV + i_hv) * (RMAX + T) * R
                ZE = tl.load(rec_z + zb + offs_r[:, None] * R + offs_r[None, :], mask=(offs_r < R)[None, :], other=0.0)
                ZA = tl.load(rec_z + zb + RMAX * R + offs_t[:, None] * R + offs_r[None, :], mask=(offs_r < R)[None, :], other=0.0)
                keep = (offs_r < R)[:, None]
                Un = tl.dot(tl.trans(ZE), U0, input_precision="ieee")  # (RMAX, K): row k = sum_r Z[r, k] U[r]
                Wn = tl.dot(tl.trans(ZE), WE, input_precision="ieee")
                for i in tl.static_range(T):
                    za = tl.sum(tl.where((offs_t == i)[:, None], ZA, 0.0), axis=0)  # (RMAX,) row i of ZA
                    Un = Un + za[:, None] * tl.sum(tl.where((offs_t == i)[:, None], KH, 0.0), axis=0)[None, :]
                    Wn = Wn + za[:, None] * tl.sum(tl.where((offs_t == i)[:, None], WA, 0.0), axis=0)[None, :]
                Uout = tl.where(keep, Un, 0.0)
                Wout = tl.where(keep, Wn, 0.0)
                # W8 cadence (B-arm protocol): keep R factors, rows R..n-(RFULL-R)-1 stay zero, count - R carries the
                # accepted inputs past the cut point, so the next cut lands after RFULL - R accepted inputs in total
                n = n - (RFULL - R)
            else:
                # ---- no cut: appended row i goes to pool row c0 + i (< RMAX since n < RFULL <= RMAX)
                Uout = tl.where(emask[:, None], U0, 0.0)
                Wout = tl.where(emask[:, None], WE, 0.0)
                for i in tl.static_range(T):
                    at = (offs_r == c0 + i) & (i <= s)
                    Uout = tl.where(at[:, None], tl.sum(tl.where((offs_t == i)[:, None], KH, 0.0), axis=0)[None, :], Uout)
                    Wout = tl.where(at[:, None], tl.sum(tl.where((offs_t == i)[:, None], WA, 0.0), axis=0)[None, :], Wout)
            out = dst * HV + i_hv
            tl.store(pu + out * RMAX * K + offs_r[:, None] * K + offs_k[None, :], Uout.to(pu.dtype.element_ty))
            tl.store(pw + out * RMAX * V + offs_r[:, None] * V + offs_v[None, :], Wout.to(pw.dtype.element_ty))
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
    bv = CHUNK_BV if CHUNK_BV else v
    _factored_chunk_verify_kernel[(batch * hv * (v // bv),)](
        mixed, gate_a, gate_b, A_log, dt_bias, vbar, pa, pu, pw, pcount, indices,
        output, records['a'][layer], records['k'][layer], records['c'][layer], records['d'][layer],
        records['g'][layer], scale, GS_EPS,
        mixed.stride(0), mixed.stride(1), gate_a.stride(0), gate_a.stride(1),
        gate_b.stride(0), gate_b.stride(1),
        num_q_heads, hv, k, v, rmax, tokens, RC, bv, num_warps=CHUNK_WARPS)
    return output


def dense_verify(mixed, gate_a, gate_b, *, A_log, dt_bias, vbar, pa, pu, pw, pcount, indices,
                 records, layer, scale, num_q_heads, output=None):
    batch, tokens, _ = mixed.shape
    hv, rmax, k = pu.shape[1], pu.shape[2], pu.shape[3]
    v = pw.shape[-1]
    if output is None:
        output = mixed.new_empty(batch, tokens, hv, v)
    bv = min(DENSE_BV, v)
    kernel = _factored_dense_verify_wy_kernel if DENSE_IMPL == 'wy' else _factored_dense_verify_kernel
    kernel[(batch * hv * (v // bv),)](
        mixed, gate_a, gate_b, A_log, dt_bias, vbar, pa, pu, pw, pcount, indices,
        output, records['k'][layer], records['d'][layer], records['g'][layer], records['b'][layer], scale,
        mixed.stride(0), mixed.stride(1), gate_a.stride(0), gate_a.stride(1),
        gate_b.stride(0), gate_b.stride(1),
        num_q_heads, hv, k, v, rmax, tokens, bv, num_warps=DENSE_WARPS,
        **(dict(DOT_PREC=DENSE_DOT) if DENSE_IMPL == 'wy' else {}))
    return output


def verify(*args, **kwargs):
    return (dense_verify if MODE == 'dense' else chunk_verify)(*args, **kwargs)


def commit_select(pool, records, src_slots, steps, track_slots=None, track_steps=None, *, r, rfull,
                  trunc_iters=None):
    """One launch: rebuild accepted states from the untouched entry + records, cut when due, publish."""
    n = src_slots.numel()
    if n == 0:
        return
    layers, _, hv, rmax, k = pool.U.shape
    v = pool.W.shape[-1]
    tokens = records['g'].shape[2]
    has_track = track_slots is not None
    if MODE == 'dense' and COMMIT_SPLIT:
        args = (pool.a, pool.U, pool.W, pool.count, pool.stale,
                pool.dense_of if pool.dense_of is not None else pool.stale,
                pool.dense_required if pool.dense_required is not None else pool.stale,
                pool.prefix_valid if pool.prefix_valid is not None else pool.stale, pool.vbar,
                records['k'], records['d'], records['g'], records['b'], records['c'], records['z'],
                src_slots, steps, track_slots if has_track else src_slots, track_steps if has_track else steps, GS_EPS,
                pool.a.stride(0), pool.U.stride(0), pool.W.stride(0), pool.count.stride(0), pool.vbar.stride(0),
                records['k'].stride(0), records['d'].stride(0), records['g'].stride(0), records['b'].stride(0),
                records['c'].stride(0), records['z'].stride(0))
        meta = dict(HV=hv, K=k, V=v, RMAX=rmax, R=r, RFULL=rfull, T=tokens,
                    ITERS=trunc_iters or TRUNC_ITERS, REL_TOL=MGS_REL_TOL,
                    HAS_TRACK=has_track, HAS_DENSE_OF=pool.dense_of is not None,
                    HAS_DENSE_REQUIRED=pool.dense_required is not None, HAS_PREFIX_VALID=pool.prefix_valid is not None)
        _commit_chain_kernel[(n * hv, layers)](*args, **meta, num_warps=1)
        _commit_cutsolve_kernel[(n * hv, layers)](*args, **meta, num_warps=1)
        _commit_publish_kernel[(n * hv, layers)](*args, **meta, num_warps=PUBLISH_WARPS)
        return
    if MODE == 'dense':
        _factored_commit_dense_kernel[(n * hv, layers)](
            pool.a, pool.U, pool.W, pool.count, pool.stale,
            pool.dense_of if pool.dense_of is not None else pool.stale,
            pool.dense_required if pool.dense_required is not None else pool.stale,
            pool.prefix_valid if pool.prefix_valid is not None else pool.stale, pool.vbar,
            records['k'], records['d'], records['g'], records['b'],
            src_slots, steps, track_slots if has_track else src_slots, track_steps if has_track else steps, GS_EPS,
            pool.a.stride(0), pool.U.stride(0), pool.W.stride(0), pool.count.stride(0), pool.vbar.stride(0),
            records['k'].stride(0), records['d'].stride(0), records['g'].stride(0), records['b'].stride(0),
            HV=hv, K=k, V=v, RMAX=rmax, R=r, RFULL=rfull, T=tokens,
            ITERS=trunc_iters or TRUNC_ITERS, REL_TOL=MGS_REL_TOL,
            HAS_TRACK=has_track, HAS_DENSE_OF=pool.dense_of is not None,
            HAS_DENSE_REQUIRED=pool.dense_required is not None, HAS_PREFIX_VALID=pool.prefix_valid is not None,
            num_warps=COMMIT_WARPS)
        return
    kernel = _factored_commit_block_kernel if COMMIT_IMPL == 'block' else _factored_commit_select_kernel
    extra = {} if COMMIT_IMPL == 'block' else dict(RP=32)
    kernel[(n * hv, layers)](
        pool.a, pool.U, pool.W, pool.count, pool.stale,
        pool.dense_of if pool.dense_of is not None else pool.stale,
        pool.dense_required if pool.dense_required is not None else pool.stale,
        pool.prefix_valid if pool.prefix_valid is not None else pool.stale,
        records['a'], records['k'], records['c'], records['d'], records['g'],
        src_slots, steps, track_slots if has_track else src_slots, track_steps if has_track else steps,
        pool.a.stride(0), pool.U.stride(0), pool.W.stride(0), pool.count.stride(0),
        records['a'].stride(0), records['k'].stride(0), records['c'].stride(0), records['d'].stride(0),
        records['g'].stride(0),
        HV=hv, K=k, V=v, RMAX=rmax, R=r, RFULL=rfull, T=tokens, RCW=RC,
        ITERS=trunc_iters or TRUNC_ITERS, REL_TOL=MGS_REL_TOL,
        HAS_TRACK=has_track, HAS_DENSE_OF=pool.dense_of is not None,
        HAS_DENSE_REQUIRED=pool.dense_required is not None, HAS_PREFIX_VALID=pool.prefix_valid is not None,
        num_warps=COMMIT_WARPS, **extra)


def allocate_records(layers, max_batch, tokens, hv, k, v, device, mode=None):
    """Per-layer verify records; layout (L, B, T, HV, .) so a layer's rows are contiguous for the verify launch."""
    if (mode or MODE) == 'dense':
        return dict(
            k=torch.zeros(layers, max_batch, tokens, hv, k, dtype=torch.float32, device=device),
            d=torch.zeros(layers, max_batch, tokens, hv, v, dtype=torch.float32, device=device),
            g=torch.zeros(layers, max_batch, tokens, hv, dtype=torch.float32, device=device),
            b=torch.zeros(layers, max_batch, tokens, hv, dtype=torch.float32, device=device),
            # split-commit scratch: cfull coefficients (RMAX + T, T) and solved directions (2 destinations)
            c=torch.zeros(layers, max_batch, hv, 16 + tokens, tokens, dtype=torch.float32, device=device),
            z=torch.zeros(layers, max_batch, 2, hv, 16 + tokens, 8, dtype=torch.float32, device=device),
        )
    return dict(
        a=torch.zeros(layers, max_batch, tokens, hv, k, dtype=torch.float32, device=device),
        k=torch.zeros(layers, max_batch, tokens, hv, k, dtype=torch.float16, device=device),
        c=torch.zeros(layers, max_batch, tokens, hv, RC, dtype=torch.float32, device=device),
        d=torch.zeros(layers, max_batch, tokens, hv, v, dtype=torch.float32, device=device),
        g=torch.zeros(layers, max_batch, tokens, hv, dtype=torch.float32, device=device),
    )
