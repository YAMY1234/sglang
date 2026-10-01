"""Unconnected candidate: split each head along V, retain one-warp K reductions.

Each CTA uses immutable pre-step a/count snapshots. New U rows are excluded
by the old count; W/output partitions are disjoint. Production has no caller.
The next numerical/performance gate must charge the snapshot preparation too.
"""
import triton
import triton.language as tl


@triton.jit
def _factored_tiled_v_step_kernel(
    mixed_qkv,
    a_gate,
    b_gate,
    A_log,
    dt_bias,
    vbar,
    a_ptr,
    a_snapshot_ptr,
    u_ptr,
    w_ptr,
    cnt_ptr,
    count_snapshot_ptr,
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
    V_TILE: tl.constexpr,
    LATE_W_LOAD: tl.constexpr = False,
):
    program = tl.program_id(0)
    partitions: tl.constexpr = V // V_TILE
    pid = program // partitions  # b * HV + hv
    v_part = program % partitions
    i_n = pid // HV
    i_hv = pid % HV
    i_h = i_hv // (HV // H)
    offs_k = tl.arange(0, K)
    offs_v = v_part * V_TILE + tl.arange(0, V_TILE)
    offs_r = tl.arange(0, RMAX)

    state_idx = tl.load(ssm_state_indices + i_n * stride_idx).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + offs_v
    if state_idx < 0:
        tl.store(p_o, tl.zeros([V_TILE], dtype=tl.float32).to(p_o.dtype.element_ty))
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
    # Snapshot source is immutable during this launch: the four CTAs must
    # not race with the one CTA publishing the updated sink or count.
    a = tl.load(a_snapshot_ptr + (state_idx * HV + i_hv) * K + offs_k)
    a_new = gt * (a - beta * kn * tl.sum(kn * a, axis=0)) + beta * kn
    if v_part == 0:
        tl.store(p_a, a_new)
    out = vb * tl.sum(a_new * qn, axis=0)

    # ---- content: Gram-Schmidt of k against the orthonormal basis, rank-1 update of the coefficients (K0 step)
    p_cnt = cnt_ptr + state_idx * HV + i_hv
    cnt = tl.load(count_snapshot_ptr + state_idx * HV + i_hv)
    rmask = offs_r < cnt
    u_tile = u_ptr + (state_idx * HV + i_hv) * RMAX * K + offs_r[:, None] * K + offs_k[None, :]
    w_tile = w_ptr + (state_idx * HV + i_hv) * RMAX * V + offs_r[:, None] * V + offs_v[None, :]
    U = tl.load(u_tile, mask=rmask[:, None], other=0.0).to(tl.float32)  # (RMAX, K)
    if not LATE_W_LOAD:
        W = tl.load(w_tile, mask=rmask[:, None], other=0.0).to(tl.float32)
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
    is_new = offs_r == cnt
    cfull = tl.where(is_new, clast, c)
    cq = tl.sum(U * qn[None, :], axis=1) + tl.where(is_new, tl.sum(khat * qn, axis=0), 0.0)
    # All U-dependent reductions finish before loading W. Keeping both full
    # tiles live across reorthogonalisation consumed 249 registers/thread at
    # B1/RMAX16. Expressions and publication order remain unchanged.
    if LATE_W_LOAD:
        W = tl.load(w_tile, mask=rmask[:, None], other=0.0).to(tl.float32)
    mvec = tl.sum(W * c[:, None], axis=0)  # (V,)  S_c^T k
    delta = beta * ((v - vb) - gt * mvec)
    out = out + gt * tl.sum(W * cq[:, None], axis=0) + delta * tl.sum(cfull * cq, axis=0)
    tl.store(w_tile, (gt * W + cfull[:, None] * delta[None, :]).to(w_ptr.dtype.element_ty), mask=(offs_r <= cnt)[:, None])
    if v_part == 0:
        tl.store(u_ptr + (state_idx * HV + i_hv) * RMAX * K + cnt * K + offs_k, khat.to(u_ptr.dtype.element_ty),
                 mask=offs_k < K * (cnt < RMAX))
        tl.store(p_cnt, cnt + 1)
        tl.store(stale_ptr + state_idx, 1)
    tl.store(p_o, out.to(p_o.dtype.element_ty))

