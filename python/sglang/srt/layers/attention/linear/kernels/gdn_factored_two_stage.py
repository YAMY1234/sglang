"""Unconnected experiment: one K-side producer, V-partitioned consumers.

No production caller. Scratch is 64 FP32 words per active batch/head; it
contains coefficients, four scalars and the pre-step count, not pool copies.
Both launches and scratch traffic must be charged to the candidate step.
"""
import triton
import triton.language as tl


@triton.jit
def _factored_k_producer(
    mixed_qkv, a_gate, b_gate, A_log, dt_bias, a_ptr, u_ptr,
    cnt_ptr, stale_ptr, ssm_state_indices, scratch, scale, gs_eps,
    stride_mixed_tok: tl.constexpr, stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr, stride_idx: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
    RMAX: tl.constexpr, SOFTPLUS_THRESHOLD: tl.constexpr,
    SCRATCH_WORDS: tl.constexpr,
):
    pid = tl.program_id(0)
    i_n, i_hv = pid // HV, pid % HV
    i_h = i_hv // (HV // H)
    offs_k, offs_r = tl.arange(0, K), tl.arange(0, RMAX)
    state_idx = tl.load(ssm_state_indices + i_n * stride_idx).to(tl.int64)
    if state_idx < 0:
        return
    p_mixed = mixed_qkv + i_n * stride_mixed_tok
    q = tl.load(p_mixed + i_h * K + offs_k).to(tl.float32)
    k = tl.load(p_mixed + H * K + i_h * K + offs_k).to(tl.float32)
    a_val = tl.load(a_gate + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b_gate + i_n * stride_b_tok + i_hv).to(tl.float32)
    x = a_val + tl.load(dt_bias + i_hv).to(tl.float32)
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(tl.load(A_log + i_hv).to(tl.float32)) * softplus_x
    beta = tl.sigmoid(b_val).to(b_gate.dtype.element_ty).to(tl.float32)
    gt = tl.exp(g_val)
    qn = q / tl.sqrt(tl.sum(q * q) + 1e-6) * scale
    kn = k / tl.sqrt(tl.sum(k * k) + 1e-6)
    p_a = a_ptr + (state_idx * HV + i_hv) * K + offs_k
    a = tl.load(p_a)
    a_new = gt * (a - beta * kn * tl.sum(kn * a, axis=0)) + beta * kn
    tl.store(p_a, a_new)
    sink = tl.sum(a_new * qn, axis=0)
    p_cnt = cnt_ptr + state_idx * HV + i_hv
    cnt = tl.load(p_cnt)
    u_tile = u_ptr + (state_idx * HV + i_hv) * RMAX * K + offs_r[:, None] * K + offs_k[None, :]
    U = tl.load(u_tile, mask=(offs_r < cnt)[:, None], other=0.0).to(tl.float32)
    c = tl.sum(U * kn[None, :], axis=1)
    kp = kn - tl.sum(U * c[:, None], axis=0)
    nrm2 = tl.sum(kp * kp, axis=0)
    if nrm2 < 0.25:
        c2 = tl.sum(U * kp[None, :], axis=1)
        kp = kp - tl.sum(U * c2[:, None], axis=0)
        c = c + c2
        nrm2 = tl.sum(kp * kp, axis=0)
    nrm = tl.sqrt(nrm2)
    keep = nrm > gs_eps
    khat = tl.where(keep, kp / tl.maximum(nrm, gs_eps), 0.0)
    cfull = tl.where(offs_r == cnt, tl.where(keep, nrm, 0.0), c)
    cq = tl.sum(U * qn[None, :], axis=1) + tl.where(offs_r == cnt, tl.sum(khat * qn, axis=0), 0.0)
    dot = tl.sum(cfull * cq, axis=0)
    base = scratch + pid * SCRATCH_WORDS
    tl.store(base + offs_r, c)
    tl.store(base + RMAX + offs_r, cfull)
    tl.store(base + 2 * RMAX + offs_r, cq)
    tl.store(base + 3 * RMAX, gt)
    tl.store(base + 3 * RMAX + 1, beta)
    tl.store(base + 3 * RMAX + 2, sink)
    tl.store(base + 3 * RMAX + 3, dot)
    tl.store(base + 3 * RMAX + 4, cnt.to(tl.float32))
    tl.store(u_ptr + (state_idx * HV + i_hv) * RMAX * K + cnt * K + offs_k,
             khat.to(u_ptr.dtype.element_ty), mask=offs_k < K * (cnt < RMAX))
    tl.store(p_cnt, cnt + 1)
    tl.store(stale_ptr + state_idx, 1)


@triton.jit
def _factored_v_consumer(
    mixed_qkv, vbar, w_ptr, ssm_state_indices, o, scratch,
    stride_mixed_tok: tl.constexpr, stride_idx: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    RMAX: tl.constexpr, SCRATCH_WORDS: tl.constexpr, V_TILE: tl.constexpr,
):
    program = tl.program_id(0)
    partitions: tl.constexpr = V // V_TILE
    pid, part = program // partitions, program % partitions
    i_n, i_hv = pid // HV, pid % HV
    # Preserve the original logical RMAX x V reduction layout. Mask memory
    # operations to the owned partition; no consumer reads U or updated a.
    offs_v, offs_r = tl.arange(0, V), tl.arange(0, RMAX)
    owned = (offs_v >= part * V_TILE) & (offs_v < (part + 1) * V_TILE)
    state_idx = tl.load(ssm_state_indices + i_n * stride_idx).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + offs_v
    if state_idx < 0:
        tl.store(p_o, tl.full((V,), 0., tl.float32).to(p_o.dtype.element_ty), owned)
        return
    base = scratch + pid * SCRATCH_WORDS
    c = tl.load(base + offs_r)
    cfull = tl.load(base + RMAX + offs_r)
    cq = tl.load(base + 2 * RMAX + offs_r)
    gt = tl.load(base + 3 * RMAX)
    beta = tl.load(base + 3 * RMAX + 1)
    sink = tl.load(base + 3 * RMAX + 2)
    dot = tl.load(base + 3 * RMAX + 3)
    cnt = tl.load(base + 3 * RMAX + 4).to(tl.int32)
    v = tl.load(mixed_qkv + i_n * stride_mixed_tok + 2 * H * K + i_hv * V + offs_v,
                owned, other=0.).to(tl.float32)
    vb = tl.load(vbar + i_hv * V + offs_v, owned, other=0.).to(tl.float32)
    w_tile = w_ptr + (state_idx * HV + i_hv) * RMAX * V + offs_r[:, None] * V + offs_v[None, :]
    W = tl.load(w_tile, mask=(offs_r < cnt)[:, None] & owned[None, :], other=0.).to(tl.float32)
    mvec = tl.sum(W * c[:, None], axis=0)
    delta = beta * ((v - vb) - gt * mvec)
    out = vb * sink
    out = out + gt * tl.sum(W * cq[:, None], axis=0) + delta * dot
    tl.store(w_tile, (gt * W + cfull[:, None] * delta[None, :]).to(w_ptr.dtype.element_ty),
             mask=(offs_r <= cnt)[:, None] & owned[None, :])
    tl.store(p_o, out.to(p_o.dtype.element_ty), owned)
