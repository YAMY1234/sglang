"""Factored GDN decode state: step kernel + slot-expiry truncation kernel (TwinStar K1, docs/62).

Source: twinstar/kernels/gdn_factored.py (K0, docs/60; twinstar-pd-models PR #124) -- the `_factored_step_kernel`,
`_mgs` and `_truncate_iter_kernel` algorithms are copied verbatim in their maths.  Differences of this copy:
  * packed decode addressing: q, k, v are read from the stock `mixed_qkv` row ([2*H*K + HV*V], q/k head = hv // (HV // H)),
    exactly like `fused_recurrent_gated_delta_rule_packed_decode_kernel`;
  * the GDN gate is computed in-kernel with the stock formula (g = -exp(A_log) * softplus(a + dt_bias, threshold 20),
    beta = bf16(sigmoid(b)) -- the same expressions, hence the same values, as the stock packed kernel / fused_gdn_gating);
  * state slot indirection through `ssm_state_indices` (a padded row, slot < 0, writes a zero output and returns);
  * the truncation is a SEPARATE launch per layer per decode step whose programs exit immediately unless the slot's
    `count == RFULL` (= r + m): no host-side schedule, slots with unsynchronised counts, CUDA-graph safe.  (A first
    version fused the truncation into the step program behind a scalar branch; the branch's tl.dot tiles made the
    plain step 6x slower and mis-computed the factors, docs/62 §3.2.)
  * the per-slot `stale` flag is set to 1 (the factored form is authoritative, the dense state of this slot is stale).

State per (slot, head): a (K) fp32 sink vector; U (RMAX, K) orthonormal key-side basis rows; W (RMAX, V) coefficient rows;
count int32 in [r, r + m].  S (sglang layout, (V, K)) = vbar a^T + W^T U  (K0 layout S^T = a vbar^T + U^T W).
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from .gdn_truncate import _jacobi_vectors, truncate as jacobi_truncate, truncate_tensor

TRUNC_METHOD = os.environ.get("SGLANG_GDN_FACTORED_TRUNC_METHOD", "mgs")
TENSOR_ITERS = int(os.environ.get("SGLANG_GDN_FACTORED_TENSOR_ITERS", "3"))
TENSOR_PASSES = int(os.environ.get("SGLANG_GDN_FACTORED_TENSOR_PASSES", "-1"))
TENSOR_EXTENSION = None
if TRUNC_METHOD == "tensor":
    from .gdn_jacobi_cuda import load_extension
    TENSOR_EXTENSION = load_extension()

JACOBI_SWEEPS = int(os.environ.get("SGLANG_GDN_FACTORED_JACOBI_SWEEPS", "5"))

GS_EPS = 1e-4  # k within EPS of span(U) appends a zero column (docs/60 §1)
MGS_REL_TOL = 1e-4  # rank tolerance of the truncation's Gram-Schmidt (docs/60 §3.1: 1e-4 .. 1e-2 stable; 0 blows up)
TRUNC_ITERS = int(os.environ.get("SGLANG_GDN_FACTORED_TRUNC_ITERS", "3"))  # subspace-iteration rounds (docs/60 §3.1: 3 rounds <= 1.09x the exact cut)
STEP_WARPS = 1  # K0 GB300 sweep for RMAX = 16 (docs/60 §3.2)


def _step_warps(rmax):
    """Keep verify and accepted replay on the same deferred-policy layout."""
    value = (int(os.environ.get('SGLANG_GDN_VERIFY_APPEND_WARPS', '1'))
             if rmax == 32 and os.environ.get('SGLANG_GDN_VERIFY_DEFER_CUT', '0') == '1'
             else STEP_WARPS)
    if value not in (1, 2, 4, 8):
        raise ValueError('append warp count must be one of 1, 2, 4, 8')
    return value

TRUNC_WARPS = int(os.environ.get("SGLANG_GDN_FACTORED_TRUNC_WARPS", "4"))  # K1 split expiry launch (fallback)
# K2 (docs/63 §4, AGA 784052 sweep): the expiry truncation is latency-bound (one program = a serial chain of ~200 small
# reductions); at RMAX 16 one warp keeps every 16x16 reduction inside a warp (1/8 of the slots expiring: 22-43 us vs
# 33-63 us at 4 warps), at RMAX 32 one warp spills (110-255 us) and 4 warps is best (60-190 us).
TRUNC_WARPS_BY_RMAX = {16: int(os.environ.get("SGLANG_GDN_FACTORED_TRUNC_WARPS16", "1")), 32: int(os.environ.get("SGLANG_GDN_FACTORED_TRUNC_WARPS32", "4"))}
# kernel = "split" (expiry-truncation launch + step launch per layer per step; with async_stream the truncation runs off
# the critical path) | "fused" (one launch: the expiring program truncates in registers with reductions only -- no tl.dot
# tiles in the branch, docs/62 §3.2 -- then appends).  All overridable by the env / the flag string.
DEFAULT_KERNEL = os.environ.get("SGLANG_GDN_FACTORED_KERNEL", "split")
FUSED_WARPS = {16: int(os.environ.get("SGLANG_GDN_FACTORED_FUSED_WARPS16", "1")), 32: int(os.environ.get("SGLANG_GDN_FACTORED_FUSED_WARPS32", "4"))}


@triton.jit
def _mgs(Y, offs_c, RKEEP: tl.constexpr, PASSES: tl.constexpr, REL_TOL: tl.constexpr):
    """Modified Gram-Schmidt over the first RKEEP columns of Y (RP, RK), PASSES times ("twice is enough").
    A column whose first-pass residual is below REL_TOL x its original norm is numerically dependent and is dropped
    (zero column) -- normalising its rounding noise gives a non-orthogonal basis whose row norms grow at every truncation
    (docs/60 §3.1, the m = 1 blow-up).  Elementwise ops + reductions only."""
    Q = Y
    n0 = tl.sqrt(tl.sum(Y * Y, axis=0))  # (RK,) original column norms
    for p in tl.static_range(PASSES):
        for j in tl.static_range(RKEEP):
            colj = offs_c == j
            y = tl.sum(tl.where(colj[None, :], Q, 0.0), axis=1)  # (RP,)
            proj = tl.where(offs_c < j, tl.sum(Q * y[:, None], axis=0), 0.0)  # (RK,) Q^T y on previous columns
            y = y - tl.sum(Q * proj[None, :], axis=1)
            n = tl.sqrt(tl.sum(y * y, axis=0))
            n0j = tl.sum(tl.where(colj, n0, 0.0), axis=0)
            ok = n > 1e-12
            if p == 0:
                ok = ok & (n > REL_TOL * n0j)
            y = tl.where(ok, y / tl.maximum(n, 1e-30), 0.0)
            Q = tl.where(colj[None, :], y[:, None], Q)
    return Q


@triton.jit
def _factored_packed_step_kernel(
    mixed_qkv,
    a_gate,
    b_gate,
    A_log,
    dt_bias,
    vbar,
    a_ptr,
    u_ptr,
    w_ptr,
    cnt_ptr,
    stale_ptr,
    ssm_state_indices,
    o,
    scale,
    gs_eps,
    stride_mixed_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_idx: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    RMAX: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    dst_a, dst_u, dst_w, dst_count,
    OUT_OF_PLACE: tl.constexpr,
    OUT_ROW_STRIDE: tl.constexpr,
    WRITE_OUTPUT: tl.constexpr = True,
    LAYER_MIXED: tl.constexpr = 0, LAYER_GATE_A: tl.constexpr = 0,
    LAYER_GATE_B: tl.constexpr = 0, LAYER_LOG: tl.constexpr = 0,
    LAYER_BIAS: tl.constexpr = 0, LAYER_VBAR: tl.constexpr = 0,
    LAYER_A: tl.constexpr = 0, LAYER_U: tl.constexpr = 0,
    LAYER_W: tl.constexpr = 0, LAYER_COUNT: tl.constexpr = 0,
    RECORD_INPUTS: tl.constexpr = False,
    record_mixed=None, record_a=None, record_b=None, record_written=None,
    RECORD_MIXED_ROW: tl.constexpr = 0, RECORD_GATE_ROW: tl.constexpr = 0,
    RECORD_WRITTEN_ROW: tl.constexpr = 0,
    CONDITIONAL_STEP: tl.constexpr = False, accepted_steps=None,
    INPUT_STEP: tl.constexpr = 0, RAW_APPEND: tl.constexpr = False,
):
    layer = tl.program_id(1).to(tl.int64)
    mixed_qkv += layer * LAYER_MIXED
    a_gate += layer * LAYER_GATE_A
    b_gate += layer * LAYER_GATE_B
    A_log += layer * LAYER_LOG
    dt_bias += layer * LAYER_BIAS
    vbar += layer * LAYER_VBAR
    a_ptr += layer * LAYER_A
    u_ptr += layer * LAYER_U
    w_ptr += layer * LAYER_W
    cnt_ptr += layer * LAYER_COUNT
    pid = tl.program_id(0)  # b * HV + hv
    i_n = pid // HV
    i_hv = pid % HV
    i_h = i_hv // (HV // H)
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V)
    offs_r = tl.arange(0, RMAX)

    state_idx = tl.load(ssm_state_indices + i_n * stride_idx).to(tl.int64)
    if CONDITIONAL_STEP:
        if tl.load(accepted_steps + i_n) < INPUT_STEP:
            return
    p_o = o + i_n * OUT_ROW_STRIDE + i_hv * V + offs_v
    if state_idx < 0:
        if WRITE_OUTPUT:
            tl.store(p_o, tl.zeros([V], dtype=tl.float32).to(p_o.dtype.element_ty))
        return

    # ---- inputs (stock packed layout) and gate (stock formula)
    p_mixed = mixed_qkv + i_n * stride_mixed_tok
    if WRITE_OUTPUT:
        q_raw = tl.load(p_mixed + i_h * K + offs_k)
        q = q_raw.to(tl.float32)
    k_raw = tl.load(p_mixed + (H * K) + i_h * K + offs_k)
    v_raw = tl.load(p_mixed + (2 * H * K) + i_hv * V + offs_v)
    a_raw = tl.load(a_gate + i_n * stride_a_tok + i_hv)
    b_raw = tl.load(b_gate + i_n * stride_b_tok + i_hv)
    k, v = k_raw.to(tl.float32), v_raw.to(tl.float32)
    a_val, b_val = a_raw.to(tl.float32), b_raw.to(tl.float32)
    if RECORD_INPUTS:
        # Each value head owns its v/gates; only one head in a q/k group
        # records the shared q/k, avoiding concurrent stores to one address.
        dest = record_mixed + i_n * RECORD_MIXED_ROW
        if i_hv % (HV // H) == 0:
            tl.store(dest + i_h * K + offs_k, q_raw)
            tl.store(dest + H * K + i_h * K + offs_k, k_raw)
        tl.store(dest + 2 * H * K + i_hv * V + offs_v, v_raw)
        tl.store(record_a + i_n * RECORD_GATE_ROW + i_hv, a_raw)
        tl.store(record_b + i_n * RECORD_GATE_ROW + i_hv, b_raw)
        if i_hv == 0:
            tl.store(record_written + i_n * RECORD_WRITTEN_ROW, True)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta = tl.sigmoid(b_val).to(b_gate.dtype.element_ty).to(tl.float32)
    gt = tl.exp(g_val)
    if WRITE_OUTPUT:
        qn = q / tl.sqrt(tl.sum(q * q) + 1e-6) * scale
    kn = k / tl.sqrt(tl.sum(k * k) + 1e-6)
    vb = tl.load(vbar + i_hv * V + offs_v).to(tl.float32)

    # ---- sink: exact key-side vector recurrence
    p_a = a_ptr + (state_idx * HV + i_hv) * K + offs_k
    a = tl.load(p_a)
    a_new = gt * (a - beta * kn * tl.sum(kn * a, axis=0)) + beta * kn
    if OUT_OF_PLACE:
        tl.store(dst_a + (state_idx * HV + i_hv) * K + offs_k, a_new)
    else:
        tl.store(p_a, a_new)
    if WRITE_OUTPUT:
        out = vb * tl.sum(a_new * qn, axis=0)

    # ---- content: Gram-Schmidt of k against the orthonormal basis, rank-1 update of the coefficients (K0 step)
    p_cnt = cnt_ptr + state_idx * HV + i_hv
    cnt = tl.load(p_cnt)
    rmask = offs_r < cnt
    u_tile = u_ptr + (state_idx * HV + i_hv) * RMAX * K + offs_r[:, None] * K + offs_k[None, :]
    w_tile = w_ptr + (state_idx * HV + i_hv) * RMAX * V + offs_r[:, None] * V + offs_v[None, :]
    U = tl.load(u_tile, mask=rmask[:, None], other=0.0).to(tl.float32)  # (RMAX, K)
    W = tl.load(w_tile, mask=rmask[:, None], other=0.0).to(tl.float32)  # (RMAX, V)
    if RAW_APPEND:
        # Temporary verify factors need not be orthogonal: for S = U^T W,
        # S' = gt*S + kn*delta^T is an exact rank-one append. The accepted
        # prefix is still replayed with Gram-Schmidt into the persistent pool.
        tl.static_assert(not OUT_OF_PLACE)
        c_raw = tl.sum(U * kn[None, :], axis=1)
        m_raw = tl.sum(W * c_raw[:, None], axis=0)
        delta_raw = beta * ((v - vb) - gt * m_raw)
        if WRITE_OUTPUT:
            cq_raw = tl.sum(U * qn[None, :], axis=1)
            out = out + gt * tl.sum(W * cq_raw[:, None], axis=0) + delta_raw * tl.sum(kn * qn, axis=0)
            tl.store(p_o, out.to(p_o.dtype.element_ty))
        is_new_raw = offs_r == cnt
        next_w = tl.where(is_new_raw[:, None], delta_raw[None, :], gt * W)
        tl.store(w_tile, next_w.to(w_ptr.dtype.element_ty), mask=(offs_r <= cnt)[:, None])
        tl.store(u_ptr + (state_idx * HV + i_hv) * RMAX * K + cnt * K + offs_k,
                 kn.to(u_ptr.dtype.element_ty), mask=offs_k < K * (cnt < RMAX))
        tl.store(p_cnt, cnt + 1)
        tl.store(stale_ptr + state_idx, 1)
        return
    c = tl.sum(U * kn[None, :], axis=1)  # (RMAX,) rows >= cnt are 0
    kp = kn - tl.sum(U * c[:, None], axis=0)
    nrm2 = tl.sum(kp * kp, axis=0)
    if nrm2 < 0.25:  # k nearly in span(U): one more pass ("twice is enough"); program-uniform branch
        c2 = tl.sum(U * kp[None, :], axis=1)
        kp = kp - tl.sum(U * c2[:, None], axis=0)
        c = c + c2
        nrm2 = tl.sum(kp * kp, axis=0)
    nrm = tl.sqrt(nrm2)
    keep = nrm > gs_eps
    khat = tl.where(keep, kp / tl.maximum(nrm, gs_eps), 0.0)
    clast = tl.where(keep, nrm, 0.0)
    mvec = tl.sum(W * c[:, None], axis=0)  # (V,)  S_c^T k
    delta = beta * ((v - vb) - gt * mvec)
    is_new = offs_r == cnt
    cfull = tl.where(is_new, clast, c)
    if WRITE_OUTPUT:
        cq = tl.sum(U * qn[None, :], axis=1) + tl.where(is_new, tl.sum(khat * qn, axis=0), 0.0)
        out = out + gt * tl.sum(W * cq[:, None], axis=0) + delta * tl.sum(cfull * cq, axis=0)
    if OUT_OF_PLACE:
        # Preserve inactive rows bit-for-bit as the old whole-state copy did.
        # The math above, dtype rounding and post-step W8 cut are unchanged.
        old_u = tl.load(u_tile)
        old_w = tl.load(w_tile)
        new_u = tl.where(is_new[:, None], khat[None, :].to(u_ptr.dtype.element_ty), old_u)
        new_w = tl.where((offs_r <= cnt)[:, None],
                         (gt * W + cfull[:, None] * delta[None, :]).to(w_ptr.dtype.element_ty), old_w)
        tl.store(dst_u + (state_idx * HV + i_hv) * RMAX * K + offs_r[:, None] * K + offs_k[None, :], new_u)
        tl.store(dst_w + (state_idx * HV + i_hv) * RMAX * V + offs_r[:, None] * V + offs_v[None, :], new_w)
        tl.store(dst_count + state_idx * HV + i_hv, cnt + 1)
    else:
        tl.store(w_tile, (gt * W + cfull[:, None] * delta[None, :]).to(w_ptr.dtype.element_ty), mask=(offs_r <= cnt)[:, None])
        tl.store(u_ptr + (state_idx * HV + i_hv) * RMAX * K + cnt * K + offs_k, khat.to(u_ptr.dtype.element_ty),
                 mask=offs_k < K * (cnt < RMAX))
        tl.store(p_cnt, cnt + 1)
    tl.store(stale_ptr + state_idx, 1)
    if WRITE_OUTPUT:
        tl.store(p_o, out.to(p_o.dtype.element_ty))


@triton.jit
def _factored_expiry_truncate_kernel(
    u_ptr,
    w_ptr,
    cnt_ptr,
    ssm_state_indices,
    stride_idx: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    RMAX: tl.constexpr,
    R: tl.constexpr,
    RFULL: tl.constexpr,
    ITERS: tl.constexpr,
    REL_TOL: tl.constexpr,
    STRIDE_LAYER_U: tl.constexpr = 0,
    STRIDE_LAYER_W: tl.constexpr = 0,
    STRIDE_LAYER_COUNT: tl.constexpr = 0,
    VERIFY_GATHER: tl.constexpr = False,
    DEFERRED_CUT: tl.constexpr = False,
):
    """Slot-expiry truncation (K0 `_truncate_iter_kernel`, RP = RK = RMAX): one program per (b, hv); returns at once
    unless the slot's count == RFULL.  G = W W^T; Z0 = the R coordinate directions with the largest |W_j|^2; ITERS rounds
    of Z <- MGS2(G Z) with the rank tolerance; U[:R] = Z^T U, W[:R] = Z^T W, count = R."""
    pid = tl.program_id(0)
    layer = tl.program_id(1).to(tl.int64)
    u_ptr += layer * STRIDE_LAYER_U
    w_ptr += layer * STRIDE_LAYER_W
    cnt_ptr += layer * STRIDE_LAYER_COUNT
    i_n = pid // HV
    i_hv = pid % HV
    state_idx = tl.load(ssm_state_indices + i_n * stride_idx).to(tl.int64)
    if state_idx < 0:
        return
    p_cnt = cnt_ptr + state_idx * HV + i_hv
    cnt = tl.load(p_cnt)
    if cnt < RFULL:  # (>= rather than ==: a slot that somehow overshot -- e.g. stepped once at RFULL -- is still cut, K2)
        return
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V)
    offs_r = tl.arange(0, RMAX)
    rows = offs_r < (cnt if DEFERRED_CUT else RFULL)
    keep = offs_r < R
    u_tile = u_ptr + (state_idx * HV + i_hv) * RMAX * K + offs_r[:, None] * K + offs_k[None, :]
    w_tile = w_ptr + (state_idx * HV + i_hv) * RMAX * V + offs_r[:, None] * V + offs_v[None, :]
    W = tl.load(w_tile, mask=rows[:, None], other=0.0).to(tl.float32)  # (RMAX, V)
    G = tl.dot(W, tl.trans(W), input_precision="ieee")  # (RMAX, RMAX); rows/cols >= RFULL are 0
    # warm start: rank the diagonal (ties broken by index), Z0[i, rank_i] = 1 for rank_i < R
    d = tl.where(rows, tl.sum(tl.where(offs_r[:, None] == offs_r[None, :], G, 0.0), axis=1), -1.0)
    better = (d[None, :] > d[:, None]) | ((d[None, :] == d[:, None]) & (offs_r[None, :] < offs_r[:, None]))
    rank = tl.sum(better.to(tl.int32), axis=1)  # (RMAX,)
    Z = tl.where((rank[:, None] == offs_r[None, :]) & keep[None, :] & rows[:, None], 1.0, 0.0)  # (RMAX, RMAX)
    for _ in range(ITERS):
        Z = tl.dot(G, Z, input_precision="ieee")
        if VERIFY_GATHER:
            Z = _mgs_verify_gather(Z, offs_r, R, 2, REL_TOL)
        else:
            Z = _mgs(Z, offs_r, R, 2, REL_TOL)
    Zt = tl.trans(Z)  # (RMAX, RMAX): row j (< R) = kept direction j
    U = tl.load(u_tile, mask=rows[:, None], other=0.0).to(tl.float32)
    Un = tl.dot(Zt, U, input_precision="ieee")  # (RMAX, K)
    Wn = tl.dot(Zt, W, input_precision="ieee")  # (RMAX, V)
    if DEFERRED_CUT:
        # Carry the accepted-token phase across a late commit. The surplus
        # rows are zero placeholders, not additional retained singular modes.
        # Thus count-r == total_accepted % 8 after every committed crossing.
        tl.store(u_tile, tl.where(keep[:, None], Un, 0.0).to(u_ptr.dtype.element_ty))
        tl.store(w_tile, tl.where(keep[:, None], Wn, 0.0).to(w_ptr.dtype.element_ty))
        tl.store(p_cnt, cnt - (RFULL - R))
    else:
        tl.store(u_tile, Un.to(u_ptr.dtype.element_ty), mask=keep[:, None])
        tl.store(w_tile, Wn.to(w_ptr.dtype.element_ty), mask=keep[:, None])
        tl.store(p_cnt, cnt * 0 + R)


@triton.jit
def _factored_verify_window_kernel(
    mixed, gate_a, gate_b, A_log, dt_bias, vbar,
    fa, fu, fw, count, stale, indices, output, scale, gs_eps,
    MIXED_ROW: tl.constexpr, MIXED_STEP: tl.constexpr,
    A_ROW: tl.constexpr, A_STEP: tl.constexpr,
    B_ROW: tl.constexpr, B_STEP: tl.constexpr, INDEX_STRIDE: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    RMAX: tl.constexpr, R: tl.constexpr, RFULL: tl.constexpr,
    ITERS: tl.constexpr, REL_TOL: tl.constexpr, TOKENS: tl.constexpr,
    GATHER: tl.constexpr = False,
    DEFERRED_CUT: tl.constexpr = False,
):
    # Deliberately call the ORIGINAL post-order primitives, including typed
    # stores/reloads. No pre-order K2 kernel or alternate reduction algorithm.
    for step in range(TOKENS):
        _factored_packed_step_kernel(
            mixed + step*MIXED_STEP, gate_a + step*A_STEP, gate_b + step*B_STEP,
            A_log, dt_bias, vbar, fa, fu, fw, count, stale, indices,
            output + step*HV*V, scale, gs_eps,
            MIXED_ROW, A_ROW, B_ROW, INDEX_STRIDE, H, HV, K, V, RMAX, 20.0,
            fa, fu, fw, count, False, TOKENS*HV*V)
        tl.debug_barrier()
        if not DEFERRED_CUT:
            _factored_expiry_truncate_kernel(
                fu, fw, count, indices, INDEX_STRIDE, HV, K, V, RMAX, R, RFULL,
                ITERS, REL_TOL, VERIFY_GATHER=GATHER)
        tl.debug_barrier()


@triton.jit
def _factored_verify_append_window_kernel(
    mixed, gate_a, gate_b, A_log, dt_bias, vbar,
    fa, fu, fw, count, stale, indices, output, scale, gs_eps,
    MIXED_ROW: tl.constexpr, MIXED_STEP: tl.constexpr,
    A_ROW: tl.constexpr, A_STEP: tl.constexpr,
    B_ROW: tl.constexpr, B_STEP: tl.constexpr, INDEX_STRIDE: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    RMAX: tl.constexpr, R: tl.constexpr, RFULL: tl.constexpr,
    ITERS: tl.constexpr, REL_TOL: tl.constexpr, TOKENS: tl.constexpr,
    GATHER: tl.constexpr = False, DEFERRED_CUT: tl.constexpr = True,
    RAW_APPEND: tl.constexpr = False, RECORD_INPUTS: tl.constexpr = False,
    record_mixed=None, record_a=None, record_b=None, record_written=None,
    RECORD_MIXED_ROW: tl.constexpr = 0, RECORD_MIXED_STEP: tl.constexpr = 0,
    RECORD_GATE_ROW: tl.constexpr = 0, RECORD_GATE_STEP: tl.constexpr = 0,
    RECORD_WRITTEN_ROW: tl.constexpr = 0, RECORD_WRITTEN_STEP: tl.constexpr = 0,
):
    # No truncation primitive exists in this candidate's verify kernel.
    for step in range(TOKENS):
        rm, ra, rb, rw = record_mixed, record_a, record_b, record_written
        if RECORD_INPUTS:
            rm += step * RECORD_MIXED_STEP
            ra += step * RECORD_GATE_STEP
            rb += step * RECORD_GATE_STEP
            rw += step * RECORD_WRITTEN_STEP
        _factored_packed_step_kernel(
            mixed + step*MIXED_STEP, gate_a + step*A_STEP, gate_b + step*B_STEP,
            A_log, dt_bias, vbar, fa, fu, fw, count, stale, indices,
            output + step*HV*V, scale, gs_eps,
            MIXED_ROW, A_ROW, B_ROW, INDEX_STRIDE, H, HV, K, V, RMAX, 20.0,
            fa, fu, fw, count, False, TOKENS*HV*V,
            RAW_APPEND=RAW_APPEND, RECORD_INPUTS=RECORD_INPUTS, record_mixed=rm, record_a=ra, record_b=rb,
            record_written=rw, RECORD_MIXED_ROW=RECORD_MIXED_ROW,
            RECORD_GATE_ROW=RECORD_GATE_ROW, RECORD_WRITTEN_ROW=RECORD_WRITTEN_ROW)
        tl.debug_barrier()


@triton.jit
def _mgs_verify_gather(Y, offs_c, RKEEP: tl.constexpr, PASSES: tl.constexpr, REL_TOL: tl.constexpr):
    """Modified Gram-Schmidt over the first RKEEP columns of Y (RP, RK), PASSES times ("twice is enough").
    A column whose first-pass residual is below REL_TOL x its original norm is numerically dependent and is dropped
    (zero column) -- normalising its rounding noise gives a non-orthogonal basis whose row norms grow at every truncation
    (docs/60 §3.1, the m = 1 blow-up).  Elementwise ops + reductions only."""
    Q = Y
    n0 = tl.sqrt(tl.sum(Y * Y, axis=0))  # (RK,) original column norms
    for p in tl.static_range(PASSES):
        for j in tl.static_range(RKEEP):
            colj = offs_c == j
            y = tl.gather(Q, tl.full((Q.shape[0], 1), j, tl.int32), axis=1).reshape((Q.shape[0],))  # (RP,)
            proj = tl.where(offs_c < j, tl.sum(Q * y[:, None], axis=0), 0.0)  # (RK,) Q^T y on previous columns
            y = y - tl.sum(Q * proj[None, :], axis=1)
            n = tl.sqrt(tl.sum(y * y, axis=0))
            n0j = tl.sum(tl.where(colj, n0, 0.0), axis=0)
            ok = n > 1e-12
            if p == 0:
                ok = ok & (n > REL_TOL * n0j)
            y = tl.where(ok, y / tl.maximum(n, 1e-30), 0.0)
            Q = tl.where(colj[None, :], y[:, None], Q)
    return Q


@triton.jit
def _truncate_verify_resident(U_all, W_all, K: tl.constexpr, V: tl.constexpr,
                              RMAX: tl.constexpr, R: tl.constexpr,
                              RFULL: tl.constexpr, ITERS: tl.constexpr,
                              REL_TOL: tl.constexpr, GATHER: tl.constexpr):
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V)
    offs_r = tl.arange(0, RMAX)
    rows = offs_r < RFULL
    keep = offs_r < R
    W = tl.where(rows[:, None], W_all, 0.0).to(tl.float32)  # (RMAX, V)
    G = tl.dot(W, tl.trans(W), input_precision="ieee")  # (RMAX, RMAX); rows/cols >= RFULL are 0
    # warm start: rank the diagonal (ties broken by index), Z0[i, rank_i] = 1 for rank_i < R
    d = tl.where(rows, tl.sum(tl.where(offs_r[:, None] == offs_r[None, :], G, 0.0), axis=1), -1.0)
    better = (d[None, :] > d[:, None]) | ((d[None, :] == d[:, None]) & (offs_r[None, :] < offs_r[:, None]))
    rank = tl.sum(better.to(tl.int32), axis=1)  # (RMAX,)
    Z = tl.where((rank[:, None] == offs_r[None, :]) & keep[None, :] & rows[:, None], 1.0, 0.0)  # (RMAX, RMAX)
    for _ in range(ITERS):
        Z = tl.dot(G, Z, input_precision="ieee")
        if GATHER:
            Z = _mgs_verify_gather(Z, offs_r, R, 2, REL_TOL)
        else:
            Z = _mgs(Z, offs_r, R, 2, REL_TOL)
    Zt = tl.trans(Z)  # (RMAX, RMAX): row j (< R) = kept direction j
    U = tl.where(rows[:, None], U_all, 0.0).to(tl.float32)
    Un = tl.dot(Zt, U, input_precision="ieee")  # (RMAX, K)
    Wn = tl.dot(Zt, W, input_precision="ieee")  # (RMAX, V)
    U_all = tl.where(keep[:, None], Un.to(U_all.dtype), U_all)
    W_all = tl.where(keep[:, None], Wn.to(W_all.dtype), W_all)
    return U_all, W_all


@triton.jit
def _factored_verify_resident_kernel(
    mixed, gate_a, gate_b, A_log, dt_bias, vbar,
    fa, fu, fw, count, stale, indices, output, scale, gs_eps,
    MIXED_ROW: tl.constexpr, MIXED_STEP: tl.constexpr,
    A_ROW: tl.constexpr, A_STEP: tl.constexpr,
    B_ROW: tl.constexpr, B_STEP: tl.constexpr, INDEX_STRIDE: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    RMAX: tl.constexpr, R: tl.constexpr, RFULL: tl.constexpr,
    ITERS: tl.constexpr, REL_TOL: tl.constexpr, TOKENS: tl.constexpr,
    BATCH: tl.constexpr, GATHER: tl.constexpr, HEAD_MAJOR: tl.constexpr,
    DEFERRED_CUT: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if HEAD_MAJOR:
        i_n, i_hv = pid % BATCH, pid // BATCH
    else:
        i_n, i_hv = pid // HV, pid % HV
    i_h = i_hv // (HV // H)
    offs_k, offs_v = tl.arange(0, K), tl.arange(0, V)
    offs_r = tl.arange(0, RMAX)
    state_idx = tl.load(indices + i_n * INDEX_STRIDE).to(tl.int64)
    if state_idx < 0:
        for step in range(TOKENS):
            tl.store(output + (i_n * TOKENS + step) * HV * V + i_hv * V + offs_v, 0)
        return
    base = state_idx * HV + i_hv
    p_a = fa + base*K + offs_k
    u_tile = fu + base*RMAX*K + offs_r[:, None]*K + offs_k[None, :]
    w_tile = fw + base*RMAX*V + offs_r[:, None]*V + offs_v[None, :]
    a, U_all, W_all = tl.load(p_a), tl.load(u_tile), tl.load(w_tile)
    cnt = tl.load(count + base)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    vb = tl.load(vbar + i_hv * V + offs_v).to(tl.float32)
    WRITE_OUTPUT: tl.constexpr = True
    SOFTPLUS_THRESHOLD: tl.constexpr = 20.0
    for step in range(TOKENS):
        # ---- inputs (stock packed layout) and gate (stock formula)
        p_mixed = mixed + i_n * MIXED_ROW + step * MIXED_STEP
        if WRITE_OUTPUT:
            q = tl.load(p_mixed + i_h * K + offs_k).to(tl.float32)
        k = tl.load(p_mixed + (H * K) + i_h * K + offs_k).to(tl.float32)
        v = tl.load(p_mixed + (2 * H * K) + i_hv * V + offs_v).to(tl.float32)
        a_val = tl.load(gate_a + i_n * A_ROW + step * A_STEP + i_hv).to(tl.float32)
        b_val = tl.load(gate_b + i_n * B_ROW + step * B_STEP + i_hv).to(tl.float32)
        x = a_val + dt_bias_val
        softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
        g_val = -tl.exp(A_log_val) * softplus_x
        beta = tl.sigmoid(b_val).to(gate_b.dtype.element_ty).to(tl.float32)
        gt = tl.exp(g_val)
        if WRITE_OUTPUT:
            qn = q / tl.sqrt(tl.sum(q * q) + 1e-6) * scale
        kn = k / tl.sqrt(tl.sum(k * k) + 1e-6)

        # ---- sink: exact key-side vector recurrence
        a_new = gt * (a - beta * kn * tl.sum(kn * a, axis=0)) + beta * kn
        if WRITE_OUTPUT:
            out = vb * tl.sum(a_new * qn, axis=0)

        # ---- content: Gram-Schmidt of k against the orthonormal basis, rank-1 update of the coefficients (K0 step)
        rmask = offs_r < cnt
        U = tl.where(rmask[:, None], U_all, 0.0).to(tl.float32)  # (RMAX, K)
        W = tl.where(rmask[:, None], W_all, 0.0).to(tl.float32)  # (RMAX, V)
        c = tl.sum(U * kn[None, :], axis=1)  # (RMAX,) rows >= cnt are 0
        kp = kn - tl.sum(U * c[:, None], axis=0)
        nrm2 = tl.sum(kp * kp, axis=0)
        if nrm2 < 0.25:  # k nearly in span(U): one more pass ("twice is enough"); program-uniform branch
            c2 = tl.sum(U * kp[None, :], axis=1)
            kp = kp - tl.sum(U * c2[:, None], axis=0)
            c = c + c2
            nrm2 = tl.sum(kp * kp, axis=0)
        nrm = tl.sqrt(nrm2)
        keep = nrm > gs_eps
        khat = tl.where(keep, kp / tl.maximum(nrm, gs_eps), 0.0)
        clast = tl.where(keep, nrm, 0.0)
        mvec = tl.sum(W * c[:, None], axis=0)  # (V,)  S_c^T k
        delta = beta * ((v - vb) - gt * mvec)
        is_new = offs_r == cnt
        cfull = tl.where(is_new, clast, c)
        if WRITE_OUTPUT:
            cq = tl.sum(U * qn[None, :], axis=1) + tl.where(is_new, tl.sum(khat * qn, axis=0), 0.0)
            out = out + gt * tl.sum(W * cq[:, None], axis=0) + delta * tl.sum(cfull * cq, axis=0)
        # The original stores round U/W to FP16 after EVERY input, including
        # immediately before a due cut. Inactive rows retain their original bits.
        U_all = tl.where(is_new[:, None], khat[None, :].to(fu.dtype.element_ty), U_all)
        W_all = tl.where((offs_r <= cnt)[:, None],
                         (gt * W + cfull[:, None] * delta[None, :]).to(fw.dtype.element_ty), W_all)
        a = a_new
        cnt += 1
        tl.store(output + (i_n * TOKENS + step) * HV * V + i_hv * V + offs_v,
                 out.to(output.dtype.element_ty))
        if not DEFERRED_CUT:
            if cnt >= RFULL:
                U_all, W_all = _truncate_verify_resident(U_all, W_all, K, V, RMAX,
                    R, RFULL, ITERS, REL_TOL, GATHER)
                cnt = cnt * 0 + R
    tl.store(p_a, a)
    tl.store(u_tile, U_all)
    tl.store(w_tile, W_all)
    tl.store(count + base, cnt)
    tl.store(stale + state_idx, 1)


@triton.jit
def _factored_verify_append_resident_kernel(
    mixed, gate_a, gate_b, A_log, dt_bias, vbar,
    fa, fu, fw, count, stale, indices, output, scale, gs_eps,
    MIXED_ROW: tl.constexpr, MIXED_STEP: tl.constexpr,
    A_ROW: tl.constexpr, A_STEP: tl.constexpr,
    B_ROW: tl.constexpr, B_STEP: tl.constexpr, INDEX_STRIDE: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    RMAX: tl.constexpr, R: tl.constexpr, RFULL: tl.constexpr,
    ITERS: tl.constexpr, REL_TOL: tl.constexpr, TOKENS: tl.constexpr,
    BATCH: tl.constexpr, GATHER: tl.constexpr, HEAD_MAJOR: tl.constexpr,
):
    # Unroll only this append-only body. The dynamic loop triggers a Triton
    # dominance failure for multi-warp U/W loop-carried tiles (j882858).
    pid = tl.program_id(0)
    if HEAD_MAJOR:
        i_n, i_hv = pid % BATCH, pid // BATCH
    else:
        i_n, i_hv = pid // HV, pid % HV
    i_h = i_hv // (HV // H)
    offs_k, offs_v = tl.arange(0, K), tl.arange(0, V)
    offs_r = tl.arange(0, RMAX)
    state_idx = tl.load(indices + i_n * INDEX_STRIDE).to(tl.int64)
    if state_idx < 0:
        for step in tl.static_range(TOKENS):
            tl.store(output + (i_n * TOKENS + step) * HV * V + i_hv * V + offs_v, 0)
        return
    base = state_idx * HV + i_hv
    p_a = fa + base*K + offs_k
    u_tile = fu + base*RMAX*K + offs_r[:, None]*K + offs_k[None, :]
    w_tile = fw + base*RMAX*V + offs_r[:, None]*V + offs_v[None, :]
    a, U_all, W_all = tl.load(p_a), tl.load(u_tile), tl.load(w_tile)
    cnt = tl.load(count + base)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    vb = tl.load(vbar + i_hv * V + offs_v).to(tl.float32)
    WRITE_OUTPUT: tl.constexpr = True
    SOFTPLUS_THRESHOLD: tl.constexpr = 20.0
    for step in tl.static_range(TOKENS):
        # ---- inputs (stock packed layout) and gate (stock formula)
        p_mixed = mixed + i_n * MIXED_ROW + step * MIXED_STEP
        if WRITE_OUTPUT:
            q = tl.load(p_mixed + i_h * K + offs_k).to(tl.float32)
        k = tl.load(p_mixed + (H * K) + i_h * K + offs_k).to(tl.float32)
        v = tl.load(p_mixed + (2 * H * K) + i_hv * V + offs_v).to(tl.float32)
        a_val = tl.load(gate_a + i_n * A_ROW + step * A_STEP + i_hv).to(tl.float32)
        b_val = tl.load(gate_b + i_n * B_ROW + step * B_STEP + i_hv).to(tl.float32)
        x = a_val + dt_bias_val
        softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
        g_val = -tl.exp(A_log_val) * softplus_x
        beta = tl.sigmoid(b_val).to(gate_b.dtype.element_ty).to(tl.float32)
        gt = tl.exp(g_val)
        if WRITE_OUTPUT:
            qn = q / tl.sqrt(tl.sum(q * q) + 1e-6) * scale
        kn = k / tl.sqrt(tl.sum(k * k) + 1e-6)

        # ---- sink: exact key-side vector recurrence
        a_new = gt * (a - beta * kn * tl.sum(kn * a, axis=0)) + beta * kn
        if WRITE_OUTPUT:
            out = vb * tl.sum(a_new * qn, axis=0)

        # ---- content: Gram-Schmidt of k against the orthonormal basis, rank-1 update of the coefficients (K0 step)
        rmask = offs_r < cnt
        U = tl.where(rmask[:, None], U_all, 0.0).to(tl.float32)  # (RMAX, K)
        W = tl.where(rmask[:, None], W_all, 0.0).to(tl.float32)  # (RMAX, V)
        c = tl.sum(U * kn[None, :], axis=1)  # (RMAX,) rows >= cnt are 0
        kp = kn - tl.sum(U * c[:, None], axis=0)
        nrm2 = tl.sum(kp * kp, axis=0)
        if nrm2 < 0.25:  # k nearly in span(U): one more pass ("twice is enough"); program-uniform branch
            c2 = tl.sum(U * kp[None, :], axis=1)
            kp = kp - tl.sum(U * c2[:, None], axis=0)
            c = c + c2
            nrm2 = tl.sum(kp * kp, axis=0)
        nrm = tl.sqrt(nrm2)
        keep = nrm > gs_eps
        khat = tl.where(keep, kp / tl.maximum(nrm, gs_eps), 0.0)
        clast = tl.where(keep, nrm, 0.0)
        mvec = tl.sum(W * c[:, None], axis=0)  # (V,)  S_c^T k
        delta = beta * ((v - vb) - gt * mvec)
        is_new = offs_r == cnt
        cfull = tl.where(is_new, clast, c)
        if WRITE_OUTPUT:
            cq = tl.sum(U * qn[None, :], axis=1) + tl.where(is_new, tl.sum(khat * qn, axis=0), 0.0)
            out = out + gt * tl.sum(W * cq[:, None], axis=0) + delta * tl.sum(cfull * cq, axis=0)
        # The original stores round U/W to FP16 after EVERY input, including
        # immediately before a due cut. Inactive rows retain their original bits.
        U_all = tl.where(is_new[:, None], khat[None, :].to(fu.dtype.element_ty), U_all)
        W_all = tl.where((offs_r <= cnt)[:, None],
                         (gt * W + cfull[:, None] * delta[None, :]).to(fw.dtype.element_ty), W_all)
        a = a_new
        cnt += 1
        tl.store(output + (i_n * TOKENS + step) * HV * V + i_hv * V + offs_v,
                 out.to(output.dtype.element_ty))
    tl.store(p_a, a)
    tl.store(u_tile, U_all)
    tl.store(w_tile, W_all)
    tl.store(count + base, cnt)
    tl.store(stale + state_idx, 1)


def factored_verify_window(mixed, gate_a, gate_b, *, fa, fu, fw, fcount,
                           stale, indices, arguments, recording=None):
    """Experimental exact-post-order four-input fusion; production opt-in only."""
    deferred = os.environ.get('SGLANG_GDN_VERIFY_DEFER_CUT', '0') == '1'
    append_warps = _step_warps(fu.shape[-2])
    if (TRUNC_METHOD != 'mgs' or not arguments.get('post_order') or
            arguments.get('async_stream') is not None or
            (arguments.get('kernel') or DEFAULT_KERNEL) != 'split' or
            (arguments['r'], arguments['rfull'], fu.shape[-2]) != (8, 16, 32 if deferred else 16)):
        raise ValueError('verify window fusion requires original r8/W8 post-order MGS')
    if (arguments.get('trunc_warps') or TRUNC_WARPS_BY_RMAX[16]) != STEP_WARPS:
        raise ValueError('verify fusion cannot change original warp counts')
    batch, tokens, _ = mixed.shape
    if tokens != 4:
        raise ValueError('verify window must contain four candidate inputs')
    hv, k, v = arguments['num_v_heads'], arguments['head_k_dim'], arguments['head_v_dim']
    output = mixed.new_empty(batch, tokens, hv, v)
    gluon = os.environ.get('SGLANG_GDN_VERIFY_WINDOW_GLUON', '0') == '1'
    old_resident = gluon or os.environ.get('SGLANG_GDN_VERIFY_WINDOW_REGISTER', '0') == '1'
    append_resident = deferred and os.environ.get('SGLANG_GDN_VERIFY_APPEND_RESIDENT', '0') == '1'
    resident = old_resident or append_resident
    raw_append = os.environ.get('SGLANG_GDN_VERIFY_APPEND_RAW', '0') == '1'
    if raw_append and (not deferred or resident):
        raise ValueError('raw verify append requires deferred non-resident verification')
    if deferred and old_resident:
        raise ValueError('deferred cut requires its explicitly named append-only resident body')
    selected = _factored_verify_resident_kernel if resident else _factored_verify_window_kernel
    if deferred:
        selected = _factored_verify_append_resident_kernel if append_resident else _factored_verify_append_window_kernel
    if gluon and os.environ.get('TRITON_INTERPRET', '0') != '1':
        from .gdn_verify_gluon import _factored_verify_gluon_kernel
        selected = _factored_verify_gluon_kernel
    tuning = dict(BATCH=batch,
                  GATHER=os.environ.get('SGLANG_GDN_VERIFY_MGS_GATHER', '0') == '1',
                  HEAD_MAJOR=os.environ.get('SGLANG_GDN_VERIFY_HEAD_MAJOR', '0') == '1') if resident else dict(
                      GATHER=os.environ.get('SGLANG_GDN_VERIFY_MGS_GATHER', '0') == '1',
                      DEFERRED_CUT=deferred)
    if deferred and not resident:
        tuning['RAW_APPEND'] = raw_append
    if recording is not None:
        if not deferred or resident:
            raise ValueError('fused input recording requires the append window primitive')
        rm, ra, rb, rw = (recording[name] for name in ('mixed','a','b','written'))
        if ra.stride()!=rb.stride():
            raise ValueError('recorded gate layouts must match')
        tuning.update(RECORD_INPUTS=True,record_mixed=rm,record_a=ra,record_b=rb,record_written=rw,
            RECORD_MIXED_ROW=rm.stride(0),RECORD_MIXED_STEP=rm.stride(1),
            RECORD_GATE_ROW=ra.stride(0),RECORD_GATE_STEP=ra.stride(1),
            RECORD_WRITTEN_ROW=rw.stride(0),RECORD_WRITTEN_STEP=rw.stride(1))
    compiled = selected[(batch*hv, 1)](
        mixed, gate_a, gate_b, arguments['A_log'], arguments['dt_bias'], arguments['vbar'],
        fa, fu, fw, fcount, stale, indices, output, arguments['scale'], GS_EPS,
        mixed.stride(0), mixed.stride(1), gate_a.stride(0), gate_a.stride(1),
        gate_b.stride(0), gate_b.stride(1), indices.stride(0),
        arguments['num_q_heads'], hv, k, v, fu.shape[-2], arguments['r'], arguments['rfull'],
        arguments.get('trunc_iters') or TRUNC_ITERS, MGS_REL_TOL, tokens,
        num_warps=append_warps, **tuning)
    if os.environ.get('SGLANG_GDN_VERIFY_DIAGNOSTICS', '0') == '1' and compiled is not None:
        global VERIFY_LAST_RESOURCES
        VERIFY_LAST_RESOURCES = dict(batch=batch, registers=getattr(compiled, 'n_regs', None),
            spills=getattr(compiled, 'n_spills', None), shared=getattr(compiled.metadata, 'shared', None),
            gluon=gluon, resident=resident, append_warps=append_warps, raw_append=raw_append)
    return output


# ============================================================================ K2: fused step + in-register expiry truncation
@triton.jit
def _gram_wwt(W, offs_r, RMAX: tl.constexpr, RFULL: tl.constexpr):
    """G = W W^T (RMAX x RMAX) of W (RMAX, V) fp32 with elementwise ops + reductions only (no tl.dot): column j of G is
    W w_j for j < RFULL; rows / columns >= RFULL are 0 (those W rows were loaded as 0)."""
    G = tl.zeros([RMAX, RMAX], dtype=tl.float32)
    for j in tl.static_range(RFULL):
        wj = tl.sum(tl.where((offs_r == j)[:, None], W, 0.0), axis=0)  # (V,) row j of W
        gj = tl.sum(W * wj[None, :], axis=1)  # (RMAX,) W w_j
        G = tl.where((offs_r == j)[None, :], gj[:, None], G)
    return G


@triton.jit
def _small_matmul_cols(G, Z, offs_r, RMAX: tl.constexpr, R: tl.constexpr):
    """Y = G Z on the first R columns of Z ((RMAX, RMAX) tiles), reductions only; columns >= R are 0."""
    Y = tl.zeros([RMAX, RMAX], dtype=tl.float32)
    for j in tl.static_range(R):
        zj = tl.sum(tl.where((offs_r == j)[None, :], Z, 0.0), axis=1)  # (RMAX,) column j of Z
        yj = tl.sum(G * zj[None, :], axis=1)  # (RMAX,) G z_j
        Y = tl.where((offs_r == j)[None, :], yj[:, None], Y)
    return Y


@triton.jit
def _project_rows(Z, X, offs_r, RMAX: tl.constexpr, R: tl.constexpr):
    """(Z^T X) on the first R kept directions: row i < R of the result = sum_j Z[j, i] X[j] (X (RMAX, D) fp32); rows >= R
    are 0.  Reductions only (the K1 split kernel used tl.dot here)."""
    Xn = tl.zeros_like(X)
    for i in tl.static_range(R):
        zi = tl.sum(tl.where((offs_r == i)[None, :], Z, 0.0), axis=1)  # (RMAX,) kept direction i (column i of Z)
        xi = tl.sum(X * zi[:, None], axis=0)  # (D,)
        Xn = tl.where((offs_r == i)[:, None], xi[None, :], Xn)
    return Xn


@triton.jit
def _factored_fused_step_kernel(
    mixed_qkv,
    a_gate,
    b_gate,
    A_log,
    dt_bias,
    vbar,
    a_ptr,
    u_ptr,
    w_ptr,
    cnt_ptr,
    stale_ptr,
    ssm_state_indices,
    o,
    scale,
    gs_eps,
    stride_mixed_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_idx: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    RMAX: tl.constexpr,
    R: tl.constexpr,
    RFULL: tl.constexpr,
    ITERS: tl.constexpr,
    REL_TOL: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    JACOBI: tl.constexpr = False,
    SWEEPS: tl.constexpr = 5,
):
    """K2 fused decode step (docs/63 §4): the maths of `_factored_packed_step_kernel`, plus -- for a program whose slot has
    count == RFULL -- the K0 `iter` truncation done in registers BEFORE the append, with elementwise ops and reductions
    only (G = W W^T column by column, Z <- MGS2(G Z) for ITERS rounds from the diagonal warm start, U[:R] = Z^T U,
    W[:R] = Z^T W).  Program-uniform branch; no tl.dot anywhere (docs/62 §3.2: dot tiles behind the branch made the plain
    step 6x slower).  One launch per layer per step replaces the K1 predicate launch + step launch."""
    pid = tl.program_id(0)  # b * HV + hv
    i_n = pid // HV
    i_hv = pid % HV
    i_h = i_hv // (HV // H)
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V)
    offs_r = tl.arange(0, RMAX)

    state_idx = tl.load(ssm_state_indices + i_n * stride_idx).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + offs_v
    if state_idx < 0:
        tl.store(p_o, tl.zeros([V], dtype=tl.float32).to(p_o.dtype.element_ty))
        return

    # ---- inputs (stock packed layout) and gate (stock formula)
    p_mixed = mixed_qkv + i_n * stride_mixed_tok
    q = tl.load(p_mixed + i_h * K + offs_k).to(tl.float32)
    k = tl.load(p_mixed + (H * K) + i_h * K + offs_k).to(tl.float32)
    v = tl.load(p_mixed + (2 * H * K) + i_hv * V + offs_v).to(tl.float32)
    a_val = tl.load(a_gate + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b_gate + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta = tl.sigmoid(b_val).to(b_gate.dtype.element_ty).to(tl.float32)
    gt = tl.exp(g_val)
    qn = q / tl.sqrt(tl.sum(q * q) + 1e-6) * scale
    kn = k / tl.sqrt(tl.sum(k * k) + 1e-6)
    vb = tl.load(vbar + i_hv * V + offs_v).to(tl.float32)

    # ---- sink: exact key-side vector recurrence
    p_a = a_ptr + (state_idx * HV + i_hv) * K + offs_k
    a = tl.load(p_a)
    a_new = gt * (a - beta * kn * tl.sum(kn * a, axis=0)) + beta * kn
    tl.store(p_a, a_new)
    out = vb * tl.sum(a_new * qn, axis=0)

    # ---- content factors
    p_cnt = cnt_ptr + state_idx * HV + i_hv
    cnt = tl.load(p_cnt)
    rmask = offs_r < cnt
    u_tile = u_ptr + (state_idx * HV + i_hv) * RMAX * K + offs_r[:, None] * K + offs_k[None, :]
    w_tile = w_ptr + (state_idx * HV + i_hv) * RMAX * V + offs_r[:, None] * V + offs_v[None, :]
    U = tl.load(u_tile, mask=rmask[:, None], other=0.0).to(tl.float32)  # (RMAX, K)
    W = tl.load(w_tile, mask=rmask[:, None], other=0.0).to(tl.float32)  # (RMAX, V)
    expired = cnt >= RFULL
    if expired:  # ---- slot expiry: truncate RFULL -> R in registers (K0 iter: G = W W^T, subspace iteration + MGS2)
        if JACOBI:
            G = tl.dot(W, tl.trans(W), input_precision="ieee")
            Z = _jacobi_vectors(G, RMAX, R, SWEEPS)
            U = tl.dot(tl.trans(Z), U, input_precision="ieee")
            W = tl.dot(tl.trans(Z), W, input_precision="ieee")
        else:
            G = _gram_wwt(W, offs_r, RMAX, RFULL)
            rows = offs_r < RFULL
            keep = offs_r < R
            d = tl.where(rows, tl.sum(tl.where(offs_r[:, None] == offs_r[None, :], G, 0.0), axis=1), -1.0)
            better = (d[None, :] > d[:, None]) | ((d[None, :] == d[:, None]) & (offs_r[None, :] < offs_r[:, None]))
            rank = tl.sum(better.to(tl.int32), axis=1)  # (RMAX,)
            Z = tl.where((rank[:, None] == offs_r[None, :]) & keep[None, :] & rows[:, None], 1.0, 0.0)  # (RMAX, RMAX)
            for _ in tl.static_range(ITERS):
                Z = _small_matmul_cols(G, Z, offs_r, RMAX, R)
                Z = _mgs(Z, offs_r, R, 2, REL_TOL)
            U = _project_rows(Z, U, offs_r, RMAX, R)
            W = _project_rows(Z, W, offs_r, RMAX, R)
        cnt = cnt * 0 + R
        rmask = offs_r < cnt
    # ---- append (K0 step): Gram-Schmidt of k against the orthonormal basis, rank-1 update of the coefficients
    c = tl.sum(U * kn[None, :], axis=1)  # (RMAX,) rows >= cnt are 0
    kp = kn - tl.sum(U * c[:, None], axis=0)
    nrm2 = tl.sum(kp * kp, axis=0)
    if nrm2 < 0.25:  # k nearly in span(U): one more pass ("twice is enough"); program-uniform branch
        c2 = tl.sum(U * kp[None, :], axis=1)
        kp = kp - tl.sum(U * c2[:, None], axis=0)
        c = c + c2
        nrm2 = tl.sum(kp * kp, axis=0)
    nrm = tl.sqrt(nrm2)
    keep_k = nrm > gs_eps
    khat = tl.where(keep_k, kp / tl.maximum(nrm, gs_eps), 0.0)
    clast = tl.where(keep_k, nrm, 0.0)
    mvec = tl.sum(W * c[:, None], axis=0)  # (V,)  S_c^T k
    delta = beta * ((v - vb) - gt * mvec)
    is_new = offs_r == cnt
    cfull = tl.where(is_new, clast, c)
    cq = tl.sum(U * qn[None, :], axis=1) + tl.where(is_new, tl.sum(khat * qn, axis=0), 0.0)
    out = out + gt * tl.sum(W * cq[:, None], axis=0) + delta * tl.sum(cfull * cq, axis=0)
    tl.store(w_tile, (gt * W + cfull[:, None] * delta[None, :]).to(w_ptr.dtype.element_ty), mask=(offs_r <= cnt)[:, None])
    if expired:  # the kept rows of U changed: store rows <= cnt (row cnt = khat)
        Ufull = tl.where(is_new[:, None], khat[None, :], U)
        tl.store(u_tile, Ufull.to(u_ptr.dtype.element_ty), mask=(offs_r <= cnt)[:, None])
    else:  # plain step: only the appended row (K1 step kernel)
        tl.store(u_ptr + (state_idx * HV + i_hv) * RMAX * K + cnt * K + offs_k, khat.to(u_ptr.dtype.element_ty),
                 mask=offs_k < K * (cnt < RMAX))
    tl.store(p_cnt, cnt + 1)
    tl.store(stale_ptr + state_idx, 1)
    tl.store(p_o, out.to(p_o.dtype.element_ty))


def factored_expiry_truncate(fu, fw, fcount, indices, r, rfull, *, trunc_warps=None, trunc_iters=None, method=None):
    """Dispatch the same expiry operation for decode and schedule-parity tests."""
    B = indices.numel()
    if B == 0:
        return
    _, HV, RMAX, K = fu.shape
    V = fw.shape[-1]
    tw = trunc_warps or TRUNC_WARPS_BY_RMAX.get(RMAX, TRUNC_WARPS)
    iters = trunc_iters or TRUNC_ITERS
    method = TRUNC_METHOD if method is None else method
    if method == "tensor" and RMAX == 32:
        truncate_tensor(fu, fw, fcount, indices, r, rfull,
                        iters=TENSOR_ITERS, passes=TENSOR_PASSES, extension=TENSOR_EXTENSION,
                        split=os.environ.get("SGLANG_GDN_FACTORED_TENSOR_SPLIT", "0") == "1",
                        rows=os.environ.get("SGLANG_GDN_FACTORED_TENSOR_ROWS", "0") == "1",
                        lanes=int(os.environ.get("SGLANG_GDN_FACTORED_TENSOR_LANES", "0")),
                        panel=os.environ.get("SGLANG_GDN_FACTORED_PANEL", "0") == "1",
                        lu_chol=int(os.environ.get("SGLANG_GDN_FACTORED_LU_CHOL", "0")),
                        lu=int(os.environ.get("SGLANG_GDN_FACTORED_LU", "0")),
                        chol=int(os.environ.get("SGLANG_GDN_FACTORED_CHOL", "0")),
                        whole=os.environ.get("SGLANG_GDN_FACTORED_TENSOR_WHOLE", "0") == "1",
                        parallel=os.environ.get("SGLANG_GDN_FACTORED_TENSOR_PARALLEL", "0") == "1",
                        parallel_lanes=int(os.environ.get("SGLANG_GDN_FACTORED_PARALLEL_LANES", "32")),
                        precision=os.environ.get("SGLANG_GDN_FACTORED_TENSOR_PRECISION", "ieee"))
    elif method in ("jacobi", "jacobi_split"):
        jacobi_truncate(fu, fw, fcount, indices, r, rfull,
                        sweeps=JACOBI_SWEEPS, split=method == "jacobi_split", warps=tw)
    else:
        _factored_expiry_truncate_kernel[(B * HV,)](
            fu, fw, fcount, indices, stride_idx=indices.stride(0),
            HV=HV, K=K, V=V, RMAX=RMAX, R=r, RFULL=rfull, ITERS=iters, REL_TOL=MGS_REL_TOL, num_warps=tw)


def factored_expiry_truncate_layers(fu, fw, fcount, indices, r, rfull, *, trunc_warps=None, trunc_iters=None,
                                    deferred_cut=False):
    """Flush all local layers after their steps, before radix tracking/next token.

    r8 groups the existing MGS programs; r16 uses the three-round LU tensor
    path. Counts and truncation mathematics are unchanged by launch grouping.
    """
    if r == 8 and fu.shape[-2] in (16, 32):
        assert TRUNC_METHOD in ("mgs", "tensor"), "r8 layer batching preserves the MGS path"
        if indices.numel() == 0:
            return
        layers, _, hv, rmax, k = fu.shape
        _factored_expiry_truncate_kernel[(indices.numel()*hv, layers)](
            fu, fw, fcount, indices, stride_idx=indices.stride(0), HV=hv, K=k, V=fw.shape[-1],
            RMAX=rmax, R=r, RFULL=rfull, ITERS=trunc_iters or TRUNC_ITERS, REL_TOL=MGS_REL_TOL,
            STRIDE_LAYER_U=fu.stride(0), STRIDE_LAYER_W=fw.stride(0),
            STRIDE_LAYER_COUNT=fcount.stride(0), DEFERRED_CUT=deferred_cut,
            num_warps=trunc_warps or TRUNC_WARPS_BY_RMAX[rmax])
        return
    assert TRUNC_METHOD == "tensor" and TENSOR_EXTENSION is not None
    assert os.environ.get("SGLANG_GDN_FACTORED_TENSOR_WHOLE", "0") == "1"
    assert os.environ.get("SGLANG_GDN_FACTORED_LU", "0") == "1"
    assert os.environ.get("SGLANG_GDN_FACTORED_PANEL", "0") == "0"
    assert os.environ.get("SGLANG_GDN_FACTORED_TENSOR_PARALLEL", "0") == "0"
    assert os.environ.get("SGLANG_GDN_FACTORED_CHOL", "0") == "0"
    assert os.environ.get("SGLANG_GDN_FACTORED_LU_CHOL", "0") == "0"
    assert TENSOR_ITERS == 3 and TENSOR_PASSES == -1
    assert r == 16 and fu.shape[-2] == 32
    TENSOR_EXTENSION.layers(fu, fw, fcount, indices, r, rfull, TENSOR_ITERS, TENSOR_PASSES)


def factored_packed_replay_layers(mixed, gate_a, gate_b, *, A_log, dt_bias,
                                  vbar, working, stale, indices, arguments, deferred_cut=False):
    """One accepted-input position across all layers; no discarded output.

    Each layer retains exactly the original append -> W8 cut dependency.
    Calls for successive positions remain ordered on the same CUDA stream.
    """
    fa, fu, fw, count = (working[n] for n in ('a', 'U', 'W', 'count'))
    layers, _, hv, rmax, k = fu.shape
    v = fw.shape[-1]
    assert mixed.shape[:2] == (layers, indices.numel())
    _factored_packed_step_kernel[(indices.numel() * hv, layers)](
        mixed, gate_a, gate_b, A_log, dt_bias, vbar, fa, fu, fw, count,
        stale, indices, mixed, arguments['scale'], GS_EPS,
        stride_mixed_tok=mixed.stride(1), stride_a_tok=gate_a.stride(1),
        stride_b_tok=gate_b.stride(1), stride_idx=indices.stride(0),
        H=arguments['num_q_heads'], HV=hv, K=k, V=v, RMAX=rmax,
        SOFTPLUS_THRESHOLD=20.0, dst_a=fa, dst_u=fu, dst_w=fw, dst_count=count,
        OUT_OF_PLACE=False, OUT_ROW_STRIDE=0, WRITE_OUTPUT=False,
        LAYER_MIXED=mixed.stride(0), LAYER_GATE_A=gate_a.stride(0),
        LAYER_GATE_B=gate_b.stride(0), LAYER_LOG=A_log.stride(0),
        LAYER_BIAS=dt_bias.stride(0), LAYER_VBAR=vbar.stride(0),
        LAYER_A=fa.stride(0), LAYER_U=fu.stride(0),
        LAYER_W=fw.stride(0), LAYER_COUNT=count.stride(0), num_warps=_step_warps(rmax))
    if not deferred_cut:
        factored_expiry_truncate_layers(fu, fw, count, indices,
            arguments['r'], arguments['rfull'], trunc_warps=arguments.get('trunc_warps'),
            trunc_iters=arguments.get('trunc_iters'))


def factored_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    vbar: torch.Tensor,
    fa: torch.Tensor,
    fu: torch.Tensor,
    fw: torch.Tensor,
    fcount: torch.Tensor,
    stale: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_q_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    r: int,
    rfull: int,
    out: Optional[torch.Tensor] = None,
    truncate: bool = True,
    kernel: Optional[str] = None,
    trunc_warps: Optional[int] = None,
    trunc_iters: Optional[int] = None,
    fused_warps: Optional[int] = None,
    async_stream: Optional[torch.cuda.Stream] = None,
    post_order: bool = False,
    state_dest: Optional[tuple] = None,
    raw_append: bool = False,
) -> torch.Tensor:
    """One factored decode step for a batch of rows.  kernel = "split" (expiry truncation launch for the slots with
    count >= rfull + step launch) | "fused" (K2: one launch, the expiring programs truncate in registers first, K1 order).
    Order of the split kernels: K1 = truncate (slots at rfull) THEN append; post_order (K2, docs/63 §4) = append THEN
    truncate the slots that have just reached rfull -- the same cut of the same 16 / 24-column state, the same sequence of
    (cut, append) operations, hence bitwise the same trajectory, executed one step earlier.  With async_stream (post
    order implied) the truncation launches on that stream after the step (fork by event) and runs off the critical path;
    the caller joins (`current.wait_stream(async_stream)`) before anything else may touch the pool (the last GDN layer of
    the forward; CUDA-graph capture needs the join inside the capture).  The truncation kernel is latency-bound (one
    program = a ~200-reduction serial chain, 20-250 us regardless of how few slots expire), so on the critical path every
    served decode step paid it once per layer.  Under the post order no slot is ever at count == rfull when a step starts
    (the truncation of the step that reached rfull precedes it); every producer of slot states must use one order.
    mixed_qkv [B, 2*H*K + HV*V] (after the causal conv), a, b [B, HV]; fa [S, HV, K] fp32, fu [S, HV, RMAX, K],
    fw [S, HV, RMAX, V], fcount [S, HV] int32, stale [S] int32 = this layer's factored pool; vbar [HV, V] fp32.
    Returns out [B, 1, HV, V] (stock packed-decode layout before the transpose)."""
    B = mixed_qkv.shape[0]
    S, HV, RMAX, K = fu.shape
    V = fw.shape[-1]
    assert HV == num_v_heads and K == head_k_dim and V == head_v_dim, (fu.shape, fw.shape, num_v_heads, head_k_dim, head_v_dim)
    assert RMAX & (RMAX - 1) == 0 and RMAX >= 16 and r < rfull <= RMAX, (RMAX, r, rfull)
    assert mixed_qkv.stride(-1) == 1 and a.stride(-1) == 1 and b.stride(-1) == 1
    assert fa.is_contiguous() and fu.is_contiguous() and fw.is_contiguous() and fcount.is_contiguous()
    assert ssm_state_indices.ndim == 1 and ssm_state_indices.shape[0] == B
    if out is None:
        out = mixed_qkv.new_empty(B, 1, HV, V)
    if state_dest is None:
        assert out.is_contiguous()
    else:
        if not (truncate and post_order and async_stream is None and
                (kernel or DEFAULT_KERNEL) == "split" and TRUNC_METHOD == "mgs"):
            raise ValueError("direct verify checkpoints require split, post-order MGS W8")
        if out.shape != (B, 1, HV, V) or out.stride(-1) != 1 or out.stride(-2) != V:
            raise ValueError("invalid direct verify output layout")
        for src, dst in zip((fa, fu, fw, fcount), state_dest):
            if src.shape != dst.shape or src.dtype != dst.dtype or not dst.is_contiguous():
                raise ValueError("direct verify state layout differs from source")
    if raw_append and (truncate or state_dest is not None or RMAX != 32):
        raise ValueError('raw append is only a no-cut temporary RMAX32 reference step')
    kernel = kernel or DEFAULT_KERNEL
    iters = trunc_iters or TRUNC_ITERS
    if kernel in ("fused", "jacobi_fused") and truncate:
        _factored_fused_step_kernel[(B * HV,)](
            mixed_qkv, a, b, A_log, dt_bias, vbar, fa, fu, fw, fcount, stale, ssm_state_indices, out,
            scale, GS_EPS,
            stride_mixed_tok=mixed_qkv.stride(0), stride_a_tok=a.stride(0), stride_b_tok=b.stride(0),
            stride_idx=ssm_state_indices.stride(0),
            H=num_q_heads, HV=HV, K=K, V=V, RMAX=RMAX, R=r, RFULL=rfull, ITERS=iters, REL_TOL=MGS_REL_TOL,
            SOFTPLUS_THRESHOLD=20.0, JACOBI=kernel == "jacobi_fused", SWEEPS=JACOBI_SWEEPS,
            num_warps=fused_warps or FUSED_WARPS.get(RMAX, 2),
        )
        return out
    assert kernel in ("split", "fused", "jacobi_fused"), kernel
    tw = trunc_warps or TRUNC_WARPS_BY_RMAX.get(RMAX, TRUNC_WARPS)

    def _truncate():
        cut_u, cut_w, cut_count = (fu, fw, fcount) if state_dest is None else state_dest[1:]
        factored_expiry_truncate(cut_u, cut_w, cut_count, ssm_state_indices, r, rfull,
                                 trunc_warps=tw, trunc_iters=iters)

    post = post_order or async_stream is not None
    if truncate and not post:
        _truncate()
    _factored_packed_step_kernel[(B * HV,)](
        mixed_qkv, a, b, A_log, dt_bias, vbar, fa, fu, fw, fcount, stale, ssm_state_indices, out,
        scale, GS_EPS,
        stride_mixed_tok=mixed_qkv.stride(0), stride_a_tok=a.stride(0), stride_b_tok=b.stride(0),
        stride_idx=ssm_state_indices.stride(0),
        H=num_q_heads, HV=HV, K=K, V=V, RMAX=RMAX, SOFTPLUS_THRESHOLD=20.0, num_warps=_step_warps(RMAX),
        dst_a=fa if state_dest is None else state_dest[0], dst_u=fu if state_dest is None else state_dest[1],
        dst_w=fw if state_dest is None else state_dest[2], dst_count=fcount if state_dest is None else state_dest[3],
        OUT_OF_PLACE=state_dest is not None, OUT_ROW_STRIDE=out.stride(0), RAW_APPEND=raw_append,
    )
    if truncate and post:
        if async_stream is None:
            _truncate()
        else:
            # fork: the side stream waits for the step (event), truncates the slots that just reached count == rfull; the
            # caller joins before the pool is touched again (graph capture: inside the capture)
            async_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(async_stream):
                _truncate()
    return out


# ============================================================================ batched orthonormalisation (prefill-end factorisation)
@triton.jit
def _orthonormalize_kernel(y_ptr, N, KC, NP: tl.constexpr, KP: tl.constexpr, PASSES: tl.constexpr, REL_TOL: tl.constexpr):
    """One program per matrix Y (N, KC) fp32 (batch-contiguous): Y <- two-pass MGS basis of col(Y) with K0's rank
    tolerance (numerically dependent columns -> zero).  Padded rows / columns (NP >= N, KP >= KC) are zero and stay zero."""
    pid = tl.program_id(0)
    offs_n = tl.arange(0, NP)
    offs_k = tl.arange(0, KP)
    mask = (offs_n[:, None] < N) & (offs_k[None, :] < KC)
    ptr = y_ptr + pid.to(tl.int64) * N * KC + offs_n[:, None] * KC + offs_k[None, :]
    Y = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
    Q = _mgs(Y, offs_k, KP, PASSES, REL_TOL)
    tl.store(ptr, Q.to(y_ptr.dtype.element_ty), mask=mask)


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


ORTH_WARPS = int(os.environ.get("SGLANG_GDN_FACTORED_ORTH_WARPS", "8"))  # K2 docs/63 §4.5: 384 matrices of 128 x 24: 1 / 2 / 4 / 8 warps = 2177 / 917 / 479 / 386 us (128-row tiles want more warps, unlike the truncation)


def orthonormalize_columns(Y: torch.Tensor, passes: int = 2, rel_tol: float = MGS_REL_TOL, num_warps: Optional[int] = None) -> torch.Tensor:
    """Batched two-pass modified Gram-Schmidt of Y (..., n, k) -> orthonormal columns (dependent columns zeroed), one
    Triton launch for the whole batch (the column-loop torch version costs ~200 launches per call and torch.linalg.qr
    loops over the batch in cusolver: 11.6 s vs 3.7 s vs stock 1.1 s for one 32-request run, AGA 783378 / 783233).
    Same maths as twinstar.kernels.gdn_factored.gram_schmidt (K0 reference).  Latency-bound like the truncation (a serial
    chain of 2 x k column steps): docs/63 §4.5 -- 9 launches x ~190 us per layer per extend at 4 warps was the largest
    K2 serving overhead (AGA 784747 profile: `_orthonormalize_kernel` the top GPU kernel of a served r16 engine)."""
    n, k = Y.shape[-2], Y.shape[-1]
    Yc = Y.float().contiguous().view(-1, n, k)
    if Yc.shape[0] == 0:
        return Yc.view_as(Y)
    _orthonormalize_kernel[(Yc.shape[0],)](Yc, n, k, NP=_pow2(n), KP=max(2, _pow2(k)), PASSES=passes, REL_TOL=rel_tol,
                                          num_warps=num_warps or ORTH_WARPS)
    return Yc.view(Y.shape)


# ============================================================================ masked slot copy (radix tracking)
@triton.jit
def _factored_track_copy_kernel(
    a_ptr, u_ptr, w_ptr, cnt_ptr, stale_ptr, src_idx, mask_ptr, dst_idx,
    stride_a_layer, stride_u_layer, stride_w_layer, stride_c_layer,
    A_ROW: tl.constexpr, U_ROW: tl.constexpr, W_ROW: tl.constexpr, C_ROW: tl.constexpr, BLOCK: tl.constexpr,
):
    """grid (B, L): copy (a, U, W, count) of slot src[i] -> dst[i] for layer l when mask[i]; dst becomes stale (factored-only).
    All offsets in int64: layer stride x layer id overflows int32 for a served-size pool (36 layers x 2932 slots x 24 heads x
    32 x 128 = 1e10 elements; K2 AGA 784499-784503 / 784662: illegal memory access as soon as the radix cache tracked a
    decode state)."""
    i = tl.program_id(0)
    l = tl.program_id(1).to(tl.int64)
    if tl.load(mask_ptr + i) == 0:
        return
    src = tl.load(src_idx + i).to(tl.int64)
    dst = tl.load(dst_idx + i).to(tl.int64)
    if src < 0 or dst < 0 or src == dst:
        return
    stride_a_layer = stride_a_layer.to(tl.int64)
    stride_u_layer = stride_u_layer.to(tl.int64)
    stride_w_layer = stride_w_layer.to(tl.int64)
    stride_c_layer = stride_c_layer.to(tl.int64)
    for s in range(0, A_ROW, BLOCK):
        offs = s + tl.arange(0, BLOCK)
        m = offs < A_ROW
        x = tl.load(a_ptr + l * stride_a_layer + src * A_ROW + offs, mask=m)
        tl.store(a_ptr + l * stride_a_layer + dst * A_ROW + offs, x, mask=m)
    for s in range(0, U_ROW, BLOCK):
        offs = s + tl.arange(0, BLOCK)
        m = offs < U_ROW
        x = tl.load(u_ptr + l * stride_u_layer + src * U_ROW + offs, mask=m)
        tl.store(u_ptr + l * stride_u_layer + dst * U_ROW + offs, x, mask=m)
    for s in range(0, W_ROW, BLOCK):
        offs = s + tl.arange(0, BLOCK)
        m = offs < W_ROW
        x = tl.load(w_ptr + l * stride_w_layer + src * W_ROW + offs, mask=m)
        tl.store(w_ptr + l * stride_w_layer + dst * W_ROW + offs, x, mask=m)
    offs = tl.arange(0, BLOCK)
    m = offs < C_ROW
    x = tl.load(cnt_ptr + l * stride_c_layer + src * C_ROW + offs, mask=m)
    tl.store(cnt_ptr + l * stride_c_layer + dst * C_ROW + offs, x, mask=m)
    if l == 0:
        tl.store(stale_ptr + dst, 1)


def factored_track_copy(
    fa: torch.Tensor, fu: torch.Tensor, fw: torch.Tensor, fcount: torch.Tensor, stale: torch.Tensor,
    src_idx: torch.Tensor, mask: torch.Tensor, dst_idx: torch.Tensor,
) -> None:
    """All-layers masked slot copy of the factored state (fa [L, S, HV, K], fu [L, S, HV, RMAX, K], fw [L, S, HV, RMAX, V],
    fcount [L, S, HV]); src_idx / dst_idx [B] (int32 or int64), mask [B] bool/int.  CUDA-graph safe (no host sync)."""
    B = src_idx.shape[0]
    L = fa.shape[0]
    if B == 0 or L == 0:
        return
    assert fa.is_contiguous() and fu.is_contiguous() and fw.is_contiguous() and fcount.is_contiguous()
    A_ROW = fa[0, 0].numel()
    U_ROW = fu[0, 0].numel()
    W_ROW = fw[0, 0].numel()
    C_ROW = fcount[0, 0].numel()
    BLOCK = 1024
    assert C_ROW <= BLOCK
    mask_i = mask if mask.dtype in (torch.int32, torch.int64, torch.uint8, torch.int8) else mask.to(torch.int32)
    _factored_track_copy_kernel[(B, L)](
        fa, fu, fw, fcount, stale, src_idx, mask_i, dst_idx,
        fa.stride(0), fu.stride(0), fw.stride(0), fcount.stride(0),
        A_ROW=A_ROW, U_ROW=U_ROW, W_ROW=W_ROW, C_ROW=C_ROW, BLOCK=BLOCK,
    )
