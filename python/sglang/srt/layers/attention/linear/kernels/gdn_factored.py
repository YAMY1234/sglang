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

from typing import Optional

import torch
import triton
import triton.language as tl

GS_EPS = 1e-4  # k within EPS of span(U) appends a zero column (docs/60 §1)
MGS_REL_TOL = 1e-4  # rank tolerance of the truncation's Gram-Schmidt (docs/60 §3.1: 1e-4 .. 1e-2 stable; 0 blows up)
TRUNC_ITERS = 3  # subspace-iteration rounds (docs/60 §3.1: 3 rounds <= 1.09x the exact cut)
STEP_WARPS = 1  # K0 GB300 sweep for RMAX = 16 (docs/60 §3.2)
TRUNC_WARPS = 4


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
):
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

    # ---- content: Gram-Schmidt of k against the orthonormal basis, rank-1 update of the coefficients (K0 step)
    p_cnt = cnt_ptr + state_idx * HV + i_hv
    cnt = tl.load(p_cnt)
    rmask = offs_r < cnt
    u_tile = u_ptr + (state_idx * HV + i_hv) * RMAX * K + offs_r[:, None] * K + offs_k[None, :]
    w_tile = w_ptr + (state_idx * HV + i_hv) * RMAX * V + offs_r[:, None] * V + offs_v[None, :]
    U = tl.load(u_tile, mask=rmask[:, None], other=0.0).to(tl.float32)  # (RMAX, K)
    W = tl.load(w_tile, mask=rmask[:, None], other=0.0).to(tl.float32)  # (RMAX, V)
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
    cq = tl.sum(U * qn[None, :], axis=1) + tl.where(is_new, tl.sum(khat * qn, axis=0), 0.0)
    out = out + gt * tl.sum(W * cq[:, None], axis=0) + delta * tl.sum(cfull * cq, axis=0)
    tl.store(w_tile, (gt * W + cfull[:, None] * delta[None, :]).to(w_ptr.dtype.element_ty), mask=(offs_r <= cnt)[:, None])
    tl.store(u_ptr + (state_idx * HV + i_hv) * RMAX * K + cnt * K + offs_k, khat.to(u_ptr.dtype.element_ty),
             mask=offs_k < K * (cnt < RMAX))
    tl.store(p_cnt, cnt + 1)
    tl.store(stale_ptr + state_idx, 1)
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
):
    """Slot-expiry truncation (K0 `_truncate_iter_kernel`, RP = RK = RMAX): one program per (b, hv); returns at once
    unless the slot's count == RFULL.  G = W W^T; Z0 = the R coordinate directions with the largest |W_j|^2; ITERS rounds
    of Z <- MGS2(G Z) with the rank tolerance; U[:R] = Z^T U, W[:R] = Z^T W, count = R."""
    pid = tl.program_id(0)
    i_n = pid // HV
    i_hv = pid % HV
    state_idx = tl.load(ssm_state_indices + i_n * stride_idx).to(tl.int64)
    if state_idx < 0:
        return
    p_cnt = cnt_ptr + state_idx * HV + i_hv
    cnt = tl.load(p_cnt)
    if cnt != RFULL:
        return
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V)
    offs_r = tl.arange(0, RMAX)
    rows = offs_r < RFULL
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
        Z = _mgs(Z, offs_r, R, 2, REL_TOL)
    Zt = tl.trans(Z)  # (RMAX, RMAX): row j (< R) = kept direction j
    U = tl.load(u_tile, mask=rows[:, None], other=0.0).to(tl.float32)
    Un = tl.dot(Zt, U, input_precision="ieee")  # (RMAX, K)
    Wn = tl.dot(Zt, W, input_precision="ieee")  # (RMAX, V)
    tl.store(u_tile, Un.to(u_ptr.dtype.element_ty), mask=keep[:, None])
    tl.store(w_tile, Wn.to(w_ptr.dtype.element_ty), mask=keep[:, None])
    tl.store(p_cnt, cnt * 0 + R)


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
) -> torch.Tensor:
    """One factored decode step for a batch of rows: expiry truncation launch (programs of slots with count == rfull) then
    the step launch.  mixed_qkv [B, 2*H*K + HV*V] (after the causal conv), a, b [B, HV]; fa [S, HV, K] fp32,
    fu [S, HV, RMAX, K], fw [S, HV, RMAX, V], fcount [S, HV] int32, stale [S] int32 = this layer's factored pool;
    vbar [HV, V] fp32.  Returns out [B, 1, HV, V] (stock packed-decode layout before the transpose)."""
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
    assert out.is_contiguous()
    if truncate:
        _factored_expiry_truncate_kernel[(B * HV,)](
            fu, fw, fcount, ssm_state_indices, stride_idx=ssm_state_indices.stride(0),
            HV=HV, K=K, V=V, RMAX=RMAX, R=r, RFULL=rfull, ITERS=TRUNC_ITERS, REL_TOL=MGS_REL_TOL, num_warps=TRUNC_WARPS,
        )
    _factored_packed_step_kernel[(B * HV,)](
        mixed_qkv, a, b, A_log, dt_bias, vbar, fa, fu, fw, fcount, stale, ssm_state_indices, out,
        scale, GS_EPS,
        stride_mixed_tok=mixed_qkv.stride(0), stride_a_tok=a.stride(0), stride_b_tok=b.stride(0),
        stride_idx=ssm_state_indices.stride(0),
        H=num_q_heads, HV=HV, K=K, V=V, RMAX=RMAX, SOFTPLUS_THRESHOLD=20.0, num_warps=STEP_WARPS,
    )
    return out


# ============================================================================ masked slot copy (radix tracking)
@triton.jit
def _factored_track_copy_kernel(
    a_ptr, u_ptr, w_ptr, cnt_ptr, stale_ptr, src_idx, mask_ptr, dst_idx,
    stride_a_layer, stride_u_layer, stride_w_layer, stride_c_layer,
    A_ROW: tl.constexpr, U_ROW: tl.constexpr, W_ROW: tl.constexpr, C_ROW: tl.constexpr, BLOCK: tl.constexpr,
):
    """grid (B, L): copy (a, U, W, count) of slot src[i] -> dst[i] for layer l when mask[i]; dst becomes stale (factored-only)."""
    i = tl.program_id(0)
    l = tl.program_id(1)
    if tl.load(mask_ptr + i) == 0:
        return
    src = tl.load(src_idx + i).to(tl.int64)
    dst = tl.load(dst_idx + i).to(tl.int64)
    if src < 0 or dst < 0 or src == dst:
        return
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
