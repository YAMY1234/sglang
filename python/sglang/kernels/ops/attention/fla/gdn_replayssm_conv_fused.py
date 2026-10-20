# SPDX-License-Identifier: Apache-2.0
"""ReplaySSM GDN verify with the causal conv1d update (width 4, SiLU) in its prologue.

One launch replaces ``causal_conv1d_update`` followed by the circular verify kernel
(``gdn_replayssm_spec_circular_kernel``, verify launch) for the BF16-checkpoint verify of
7 linear draft tokens at head dim 128. Every byte it writes (verify output, rings, conv
state, conv window) matches the two stock kernels.

Each program convolves the q and k channels of its key head and its own v channels per
channel, in the stock conv kernel's operation order, and stages the result in a private
scratch that the verify body reads with the stock load pattern (so its reduction trees
match). Every program of a key-head group reads the old q/k conv state, so the last one
to take a ticket writes the new state.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.utils import is_tf32_supported


@triton.jit
def _prefetch_l2(ptrs):
    tl.inline_asm_elementwise(
        "prefetch.L2 [$1];\n\tmov.u32 $0, 0;",
        "=r,l",
        [ptrs.to(tl.int64)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _load_taps(ptrs, stride_w_width: tl.constexpr):
    return (
        tl.load(ptrs),
        tl.load(ptrs + stride_w_width),
        tl.load(ptrs + 2 * stride_w_width),
        tl.load(ptrs + 3 * stride_w_width),
    )


@triton.jit
def _load_state(ptrs, stride_cs_tok: tl.constexpr):
    return (
        tl.load(ptrs),
        tl.load(ptrs + stride_cs_tok),
        tl.load(ptrs + 2 * stride_cs_tok),
    )


@triton.jit
def _load_tokens(ptrs, stride_x_t: tl.constexpr):
    return (
        tl.load(ptrs),
        tl.load(ptrs + stride_x_t),
        tl.load(ptrs + 2 * stride_x_t),
        tl.load(ptrs + 3 * stride_x_t),
        tl.load(ptrs + 4 * stride_x_t),
        tl.load(ptrs + 5 * stride_x_t),
        tl.load(ptrs + 6 * stride_x_t),
    )


@triton.jit
def _conv_silu(c0, c1, c2, xt, w0, w1, w2, w3):
    # Operation order of _causal_conv1d_update_kernel: activation-dtype products summed in fp32.
    acc = tl.zeros(c0.shape, dtype=tl.float32)
    acc += c0 * w0
    acc += c1 * w1
    acc += c2 * w2
    acc += xt * w3
    return acc / (1 + tl.exp(-acc))


@triton.jit
def _conv_rows(dst, stride_dst: tl.constexpr, s0, s1, s2, w0, w1, w2, w3, x0, x1, x2, x3, x4, x5, x6):
    tl.store(dst, _conv_silu(s0, s1, s2, x0, w0, w1, w2, w3))
    tl.store(dst + stride_dst, _conv_silu(s1, s2, x0, x1, w0, w1, w2, w3))
    tl.store(dst + 2 * stride_dst, _conv_silu(s2, x0, x1, x2, w0, w1, w2, w3))
    tl.store(dst + 3 * stride_dst, _conv_silu(x0, x1, x2, x3, w0, w1, w2, w3))
    tl.store(dst + 4 * stride_dst, _conv_silu(x1, x2, x3, x4, w0, w1, w2, w3))
    tl.store(dst + 5 * stride_dst, _conv_silu(x2, x3, x4, x5, w0, w1, w2, w3))
    tl.store(dst + 6 * stride_dst, _conv_silu(x3, x4, x5, x6, w0, w1, w2, w3))


@triton.jit
def _window_rows(dst, s1, s2, x0, x1, x2, x3, x4, x5, x6):
    # Position p of a channel's window: old state column p + 1 for p < 2, else token p - 2.
    tl.store(dst, s1)
    tl.store(dst + 1, s2)
    tl.store(dst + 2, x0)
    tl.store(dst + 3, x1)
    tl.store(dst + 4, x2)
    tl.store(dst + 5, x3)
    tl.store(dst + 6, x4)
    tl.store(dst + 7, x5)
    tl.store(dst + 8, x6)


@triton.jit
def _state_rows(dst, stride_cs_tok: tl.constexpr, x4, x5, x6):
    tl.store(dst, x4)
    tl.store(dst + stride_cs_tok, x5)
    tl.store(dst + 2 * stride_cs_tok, x6)


@triton.jit
def gdn_replayssm_conv_fused_kernel(
    x,  # [total_tokens, >= conv_dim] pre-conv rows, channels q | k | v
    conv_w,  # [conv_dim, 4]
    conv_state,  # [slots, conv_dim, 3]
    conv_window,  # [slots, conv_dim, 9] physical window
    window_indices,  # [B]
    tickets,  # [>= B, H] int32, zero between launches
    scratch,  # [programs, 7, 2 * K + BV] activation dtype
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    d_cache,
    k_cache,
    g_cache,
    rawv_cache,
    rawk_cache,
    beta_cache,
    query_start_loc,
    ssm_state_indices,
    replay_indices,
    write_pos,
    cache_base,
    is_flush_flags,
    scale,
    stride_x_t: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_cs_slot: tl.constexpr,
    stride_cs_dim: tl.constexpr,
    stride_cs_tok: tl.constexpr,
    stride_win_slot: tl.constexpr,
    stride_win_dim: tl.constexpr,
    stride_window_indices: tl.constexpr,
    stride_a_t: tl.constexpr,
    stride_b_t: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_state_slot: tl.constexpr,
    stride_d_slot: tl.constexpr,
    stride_k_slot: tl.constexpr,
    stride_g_slot: tl.constexpr,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    stride_qsl: tl.constexpr,
    stride_indices: tl.constexpr,
    stride_replay_indices: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BS: tl.constexpr,
    BC: tl.constexpr,
    NK: tl.constexpr,
    BKT: tl.constexpr,
    NT: tl.constexpr,
    SCR_W: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    STORE_EXACT_RAW: tl.constexpr,
    STORE_D_RESIDUAL: tl.constexpr,
    STORE_K_RESIDUAL: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    USE_GDC: tl.constexpr = False,
    PREFETCH: tl.constexpr = True,
    REDUNDANT_QK: tl.constexpr = True,
):
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    i_hv = tl.program_id(2)
    i_h = i_hv // (HV // H)
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()

    o_v = i_v * BV + tl.arange(0, BV)
    o_s = tl.arange(0, BS)
    o_c = tl.arange(0, BC)
    mask_v = o_v < V

    bos = tl.load(query_start_loc + i_n * stride_qsl).to(tl.int64)
    eos = tl.load(query_start_loc + (i_n + 1) * stride_qsl).to(tl.int64)
    spec_len = eos - bos
    mask_s = o_s < spec_len
    o_s_safe = tl.where(mask_s, o_s, 0)

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices).to(tl.int64)
    replay_idx = tl.load(replay_indices + i_n * stride_replay_indices).to(tl.int64)

    p_o = o + (bos + o_s_safe[:, None]) * stride_o_t + i_hv * V + o_v[None, :]

    # Rows the stock conv skips (pad slot -1 == NULL_BLOCK_ID) get the stock zero output only.
    if state_idx <= NULL_BLOCK_ID:
        if USE_GDC:
            tl.extra.cuda.gdc_wait()
        full_mask = (o_s < spec_len)[:, None] & mask_v[None, :]
        tl.store(
            p_o,
            tl.zeros([BS, BV], dtype=tl.float32).to(p_o.dtype.element_ty),
            mask=full_mask,
        )
        return

    # Conv channels: q and k of key head i_h (shared by the group) and this program's v.
    ch_q = i_h * K + tl.arange(0, K)
    ch_k = H * K + i_h * K + tl.arange(0, K)
    ch_v = 2 * H * K + i_hv * V + o_v
    x_row = x + bos * stride_x_t
    cs_row = conv_state + state_idx * stride_cs_slot
    win_idx = tl.load(window_indices + i_n * stride_window_indices).to(tl.int64)
    win_row = conv_window + win_idx * stride_win_slot
    write_k = (i_v == 0) and (i_hv == i_h * (HV // H))
    scr = scratch + ((i_n * HV + i_hv) * (V // BV) + i_v) * (7 * SCR_W)

    # Static weights and the old conv state are not written by the prerequisite grid.
    wq0, wq1, wq2, wq3 = _load_taps(conv_w + ch_q * stride_w_dim, stride_w_width)
    wk0, wk1, wk2, wk3 = _load_taps(conv_w + ch_k * stride_w_dim, stride_w_width)
    wv0, wv1, wv2, wv3 = _load_taps(conv_w + ch_v * stride_w_dim, stride_w_width)
    sq0, sq1, sq2 = _load_state(cs_row + ch_q * stride_cs_dim, stride_cs_tok)
    sk0, sk1, sk2 = _load_state(cs_row + ch_k * stride_cs_dim, stride_cs_tok)
    sv0, sv1, sv2 = _load_state(cs_row + ch_v * stride_cs_dim, stride_cs_tok)
    if PREFETCH:
        _prefetch_l2(
            h0
            + state_idx * stride_state_slot
            + i_hv * V * K
            + i_v * BV * K
            + tl.arange(0, BV * K // 64) * 64
        )
        _prefetch_l2(
            k_cache
            + replay_idx * stride_k_slot
            + i_h * MAX_CACHE_LEN * K
            + tl.arange(0, MAX_CACHE_LEN * K // 64) * 64
        )

    o_t = tl.arange(0, NT)
    p_ticket = tickets + i_n * H + i_h + o_t * 0
    last_ticket = (HV // H) * (V // BV) - 1

    b_is_flush = tl.load(is_flush_flags + replay_idx) != 0
    if b_is_flush:
        # The stock verify launch skips flush rows; only the conv writes remain.
        if USE_GDC:
            tl.extra.cuda.gdc_wait()
        xq0, xq1, xq2, xq3, xq4, xq5, xq6 = _load_tokens(x_row + ch_q, stride_x_t)
        xk0, xk1, xk2, xk3, xk4, xk5, xk6 = _load_tokens(x_row + ch_k, stride_x_t)
        xv0, xv1, xv2, xv3, xv4, xv5, xv6 = _load_tokens(x_row + ch_v, stride_x_t)
        # Every program read the old q/k state above; all old-state reads precede the writes.
        tl.debug_barrier()
        ticket = tl.atomic_add(
            p_ticket, tl.where(o_t == 0, 1, 0), mask=o_t == 0, sem="release", scope="gpu"
        )
        _window_rows(win_row + ch_v * stride_win_dim, sv1, sv2, xv0, xv1, xv2, xv3, xv4, xv5, xv6)
        _state_rows(cs_row + ch_v * stride_cs_dim, stride_cs_tok, xv4, xv5, xv6)
        if write_k:
            _window_rows(win_row + ch_q * stride_win_dim, sq1, sq2, xq0, xq1, xq2, xq3, xq4, xq5, xq6)
            _window_rows(win_row + ch_k * stride_win_dim, sk1, sk2, xk0, xk1, xk2, xk3, xk4, xk5, xk6)
        if tl.sum(tl.where(o_t == 0, ticket, 0), axis=0) == last_ticket:
            _state_rows(cs_row + ch_q * stride_cs_dim, stride_cs_tok, xq4, xq5, xq6)
            _state_rows(cs_row + ch_k * stride_cs_dim, stride_cs_tok, xk4, xk5, xk6)
            tl.store(tickets + i_n * H + i_h, 0)
        return

    b_write_pos = tl.load(write_pos + replay_idx).to(tl.int64)
    b_cache_base = tl.load(cache_base + replay_idx).to(tl.int32)

    out_mask = mask_s[:, None] & mask_v[None, :]

    b_wp_i = b_write_pos.to(tl.int32)
    cache_valid = o_c < b_write_pos

    phys_c = (b_cache_base + o_c) & (MAX_CACHE_LEN - 1)
    phys_spec = (b_cache_base + b_wp_i + o_s) & (MAX_CACHE_LEN - 1)

    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)

    p_g_main = g_cache + replay_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + phys_c
    b_g_all = tl.load(p_g_main, mask=cache_valid, other=0.0).to(tl.float32)
    b_g_prefix = tl.cumsum(b_g_all, axis=0)
    b_g_total = tl.sum(b_g_all, axis=0)
    b_replay_decay = tl.where(cache_valid, tl.exp(b_g_total - b_g_prefix), 0.0)
    b_total_decay = tl.exp(b_g_total)

    p_d_main = d_cache + (
        replay_idx * stride_d_slot
        + (i_hv * MAX_CACHE_LEN + phys_c[None, :]) * V
        + o_v[:, None]
    )
    b_d_all = tl.load(
        p_d_main, mask=mask_v[:, None] & cache_valid[None, :], other=0.0
    ).to(tl.float32)
    b_d_scaled_fp32 = b_d_all * b_replay_decay[None, :]
    b_d_scaled = b_d_scaled_fp32.to(d_cache.dtype.element_ty)

    # The prerequisite grid writes the pre-conv rows and the gates.
    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    a_s = tl.load(a + (bos + o_s_safe) * stride_a_t + i_hv, mask=mask_s, other=0.0).to(
        tl.float32
    )
    b_s = tl.load(b + (bos + o_s_safe) * stride_b_t + i_hv, mask=mask_s, other=0.0).to(
        tl.float32
    )
    xq0, xq1, xq2, xq3, xq4, xq5, xq6 = _load_tokens(x_row + ch_q, stride_x_t)
    xk0, xk1, xk2, xk3, xk4, xk5, xk6 = _load_tokens(x_row + ch_k, stride_x_t)
    xv0, xv1, xv2, xv3, xv4, xv5, xv6 = _load_tokens(x_row + ch_v, stride_x_t)
    tl.debug_barrier()
    ticket = tl.atomic_add(
        p_ticket, tl.where(o_t == 0, 1, 0), mask=o_t == 0, sem="release", scope="gpu"
    )

    x_g = a_s + dt_bias_val
    softplus_x = tl.where(x_g <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x_g)), x_g)
    g_s = tl.where(mask_s, -tl.exp(A_log_val) * softplus_x, 0.0)
    beta_s = tl.where(mask_s, 1.0 / (1.0 + tl.exp(-b_s)), 0.0)
    G_s = tl.cumsum(g_s, axis=0)
    expG_s = tl.exp(G_s)

    if REDUNDANT_QK:
        _conv_rows(scr + tl.arange(0, K), SCR_W, sq0, sq1, sq2, wq0, wq1, wq2, wq3,
                   xq0, xq1, xq2, xq3, xq4, xq5, xq6)
        _conv_rows(scr + K + tl.arange(0, K), SCR_W, sk0, sk1, sk2, wk0, wk1, wk2, wk3,
                   xk0, xk1, xk2, xk3, xk4, xk5, xk6)
    elif write_k:
        # Timing-only diagnostic: one program per key head convolves q / k (other programs read garbage).
        _conv_rows(scr + tl.arange(0, K), SCR_W, sq0, sq1, sq2, wq0, wq1, wq2, wq3,
                   xq0, xq1, xq2, xq3, xq4, xq5, xq6)
        _conv_rows(scr + K + tl.arange(0, K), SCR_W, sk0, sk1, sk2, wk0, wk1, wk2, wk3,
                   xk0, xk1, xk2, xk3, xk4, xk5, xk6)
    _conv_rows(scr + 2 * K + tl.arange(0, BV), SCR_W, sv0, sv1, sv2, wv0, wv1, wv2, wv3,
               xv0, xv1, xv2, xv3, xv4, xv5, xv6)
    _window_rows(win_row + ch_v * stride_win_dim, sv1, sv2, xv0, xv1, xv2, xv3, xv4, xv5, xv6)
    _state_rows(cs_row + ch_v * stride_cs_dim, stride_cs_tok, xv4, xv5, xv6)
    if write_k:
        _window_rows(win_row + ch_q * stride_win_dim, sq1, sq2, xq0, xq1, xq2, xq3, xq4, xq5, xq6)
        _window_rows(win_row + ch_k * stride_win_dim, sk1, sk2, xk0, xk1, xk2, xk3, xk4, xk5, xk6)
    # The staged conv rows must be visible to every thread of the program.
    tl.debug_barrier()

    # From here on the stock verify body, reading q / k / v from the staged rows.
    if USE_QK_L2NORM_IN_KERNEL:
        qnorm_acc = tl.zeros([BS], dtype=tl.float32)
        knorm_acc = tl.zeros([BS], dtype=tl.float32)
        for kk in range(NK):
            o_kt = kk * BKT + tl.arange(0, BKT)
            mask_kt = o_kt < K
            ld = mask_s[:, None] & mask_kt[None, :]
            qn = tl.load(
                scr + o_s_safe[:, None] * SCR_W + o_kt[None, :],
                mask=ld,
                other=0.0,
            ).to(tl.float32)
            knn = tl.load(
                scr + o_s_safe[:, None] * SCR_W + K + o_kt[None, :],
                mask=ld,
                other=0.0,
            ).to(tl.float32)
            qnorm_acc += tl.sum(qn * qn, axis=1)
            knorm_acc += tl.sum(knn * knn, axis=1)
        q_rnorm = tl.where(mask_s, 1.0 / tl.sqrt(qnorm_acc + 1e-6), 0.0)
        k_rnorm = tl.where(mask_s, 1.0 / tl.sqrt(knorm_acc + 1e-6), 0.0)
    else:
        q_rnorm = tl.where(mask_s, 1.0, 0.0)
        k_rnorm = tl.where(mask_s, 1.0, 0.0)

    hw_q = tl.zeros([BV, BS], dtype=tl.float32)
    hw_k = tl.zeros([BV, BS], dtype=tl.float32)
    scores_q = tl.zeros([BC, BS], dtype=tl.float32)
    scores_k = tl.zeros([BC, BS], dtype=tl.float32)
    kk_mat = tl.zeros([BS, BS], dtype=tl.float32)
    kq_mat = tl.zeros([BS, BS], dtype=tl.float32)

    for kk in range(NK):
        o_kt = kk * BKT + tl.arange(0, BKT)
        mask_kt = o_kt < K
        q_tile = tl.load(
            scr + o_s_safe[:, None] * SCR_W + o_kt[None, :],
            mask=mask_kt[None, :],
            other=0.0,
        ).to(tl.float32)
        k_tile = tl.load(
            scr + o_s_safe[:, None] * SCR_W + K + o_kt[None, :],
            mask=mask_kt[None, :],
            other=0.0,
        ).to(tl.float32)
        k_raw_tile = k_tile
        q_precise = q_tile * (q_rnorm * scale)[:, None]
        k_precise = k_tile * k_rnorm[:, None]
        k_store_tile = k_precise.to(k_cache.dtype.element_ty)
        k_residual_tile = (k_precise - k_store_tile.to(tl.float32)).to(
            x.dtype.element_ty
        )
        q_tile = q_precise.to(x.dtype.element_ty)
        k_tile = k_store_tile

        p_h0 = (
            h0
            + state_idx * stride_state_slot
            + i_hv * V * K
            + o_v[:, None] * K
            + o_kt[None, :]
        )
        sc_tile = tl.load(p_h0, mask=mask_v[:, None] & mask_kt[None, :], other=0.0)

        qT = tl.trans(q_tile)
        kT = tl.trans(k_tile)
        kk_mat += tl.dot(k_tile, kT, input_precision=DOT_PRECISION)
        kq_mat += tl.dot(k_tile, qT, input_precision=DOT_PRECISION)
        if STORE_K_RESIDUAL:
            k_residual_T = tl.trans(k_residual_tile)
            kk_mat += tl.dot(k_tile, k_residual_T, input_precision=DOT_PRECISION)
            kk_mat += tl.dot(k_residual_tile, kT, input_precision=DOT_PRECISION)
            kk_mat += tl.dot(
                k_residual_tile, k_residual_T, input_precision=DOT_PRECISION
            )
            kq_mat += tl.dot(k_residual_tile, qT, input_precision=DOT_PRECISION)

        sc_tile = sc_tile.to(k_tile.dtype)
        hw_q += tl.dot(sc_tile, qT)
        hw_k += tl.dot(sc_tile, kT)
        if STORE_K_RESIDUAL:
            hw_k += tl.dot(sc_tile, k_residual_T)

        p_k = k_cache + (
            replay_idx * stride_k_slot
            + (i_h * MAX_CACHE_LEN + phys_c[:, None]) * K
            + o_kt[None, :]
        )
        khist_tile = tl.load(
            p_k, mask=cache_valid[:, None] & mask_kt[None, :], other=0.0
        )
        khist_tile = khist_tile.to(k_cache.dtype.element_ty)
        scores_q += tl.dot(khist_tile, qT, input_precision=DOT_PRECISION)
        scores_k += tl.dot(khist_tile, kT, input_precision=DOT_PRECISION)

        if write_k:
            spec_kt_mask = (
                mask_s[:, None]
                & mask_kt[None, :]
                & ((b_write_pos + o_s[:, None]) < MAX_CACHE_LEN)
            )
            if STORE_EXACT_RAW or STORE_K_RESIDUAL:
                p_cur_rawk = rawk_cache + (
                    replay_idx * stride_rawk_slot
                    + (i_h * MAX_CACHE_LEN + phys_spec[:, None]) * K
                    + o_kt[None, :]
                )
                if STORE_EXACT_RAW:
                    tl.store(
                        p_cur_rawk,
                        k_raw_tile.to(p_cur_rawk.dtype.element_ty),
                        mask=spec_kt_mask,
                    )
                else:
                    tl.store(p_cur_rawk, k_residual_tile, mask=spec_kt_mask)
            p_cur_k = k_cache + (
                replay_idx * stride_k_slot
                + (i_h * MAX_CACHE_LEN + phys_spec[:, None]) * K
                + o_kt[None, :]
            )
            tl.store(p_cur_k, k_store_tile, mask=spec_kt_mask)

    hw_q = b_total_decay * hw_q + tl.dot(
        b_d_scaled, scores_q.to(b_d_scaled.dtype), input_precision=DOT_PRECISION
    )
    hw_k = b_total_decay * hw_k + tl.dot(
        b_d_scaled, scores_k.to(b_d_scaled.dtype), input_precision=DOT_PRECISION
    )

    lower = (o_s[:, None] > o_s[None, :]) & mask_s[:, None] & mask_s[None, :]
    diff_ij = G_s[:, None] - G_s[None, :]
    A_mat = tl.where(lower, beta_s[:, None] * tl.exp(diff_ij) * kk_mat, 0.0)

    b_Ai = -A_mat
    for ii in range(2, BS):
        row = tl.sum(tl.where((o_s == ii)[:, None], -A_mat, 0.0), axis=0)
        row = tl.where(o_s < ii, row, 0.0)
        row = row + tl.sum(row[:, None] * b_Ai, axis=0)
        b_Ai = tl.where((o_s == ii)[:, None], row, b_Ai)
    T_mat = b_Ai + (o_s[:, None] == o_s[None, :]).to(tl.float32)

    p_v = scr + o_s_safe[None, :] * SCR_W + 2 * K + tl.arange(0, BV)[:, None]
    v_tile = tl.load(p_v, mask=mask_v[:, None] & mask_s[None, :], other=0.0).to(
        tl.float32
    )
    R_mat = beta_s[None, :] * (v_tile - expG_s[None, :] * hw_k)
    D_spec = tl.zeros([BV, BS], dtype=tl.float32)
    for j in tl.static_range(BS):
        Rj = tl.sum(tl.where((o_s == j)[None, :], R_mat, 0.0), axis=1)
        Tj = tl.sum(tl.where((o_s == j)[None, :], T_mat, 0.0), axis=1)
        D_spec += Rj[:, None] * Tj[None, :]

    causalF = (o_s[:, None] <= o_s[None, :]) & mask_s[:, None] & mask_s[None, :]
    diff_ji = G_s[None, :] - G_s[:, None]
    F_mat = tl.where(causalF, tl.exp(diff_ji) * kq_mat, 0.0)
    DF = tl.zeros([BV, BS], dtype=tl.float32)
    for j in tl.static_range(BS):
        Dj = tl.sum(tl.where((o_s == j)[None, :], D_spec, 0.0), axis=1)
        Fj = tl.sum(tl.where((o_s == j)[:, None], F_mat, 0.0), axis=0)
        DF += Dj[:, None] * Fj[None, :]
    O_tile = expG_s[None, :] * hw_q + DF

    tl.store(p_o, tl.trans(O_tile).to(p_o.dtype.element_ty), mask=out_mask)

    spec_pos = b_write_pos + o_s
    spec_store_mask = mask_s & (spec_pos < MAX_CACHE_LEN)
    p_cur_d = d_cache + (
        replay_idx * stride_d_slot
        + (i_hv * MAX_CACHE_LEN + phys_spec[None, :]) * V
        + o_v[:, None]
    )
    d_hi = D_spec.to(p_cur_d.dtype.element_ty)
    tl.store(
        p_cur_d,
        d_hi,
        mask=mask_v[:, None] & spec_store_mask[None, :],
    )
    if STORE_EXACT_RAW or STORE_D_RESIDUAL:
        p_cur_v = rawv_cache + (
            replay_idx * stride_rawv_slot
            + (i_hv * MAX_CACHE_LEN + phys_spec[None, :]) * V
            + o_v[:, None]
        )
        if STORE_EXACT_RAW:
            tl.store(
                p_cur_v,
                v_tile.to(p_cur_v.dtype.element_ty),
                mask=mask_v[:, None] & spec_store_mask[None, :],
            )
        else:
            tl.store(
                p_cur_v,
                (D_spec - d_hi.to(tl.float32)).to(p_cur_v.dtype.element_ty),
                mask=mask_v[:, None] & spec_store_mask[None, :],
            )
    if i_v == 0:
        p_cur_g = (
            g_cache + replay_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + phys_spec
        )
        tl.store(p_cur_g, g_s, mask=spec_store_mask)
        if STORE_EXACT_RAW:
            p_cur_beta = (
                beta_cache
                + replay_idx * stride_beta_slot
                + i_hv * MAX_CACHE_LEN
                + phys_spec
            )
            tl.store(p_cur_beta, beta_s, mask=spec_store_mask)

    # The store follows the ticket's data dependency; every reader released before it.
    if tl.sum(tl.where(o_t == 0, ticket, 0), axis=0) == last_ticket:
        _state_rows(cs_row + ch_q * stride_cs_dim, stride_cs_tok, xq4, xq5, xq6)
        _state_rows(cs_row + ch_k * stride_cs_dim, stride_cs_tok, xk4, xk5, xk6)
        tl.store(tickets + i_n * H + i_h, 0)


def gdn_replayssm_conv_fused_verify(
    x: torch.Tensor,  # [total_tokens, >= conv_dim] pre-conv rows, channels q | k | v
    conv_weight: torch.Tensor,  # [conv_dim, 4]
    conv_state: torch.Tensor,  # [slots, conv_dim, 3]
    conv_window: torch.Tensor,  # [slots, conv_dim, 9] or its [slots, 7, conv_dim, 3] view
    window_indices: torch.Tensor,  # [B]
    tickets: torch.Tensor,  # [>= B, H] int32 zeros, restored to zero by the kernel
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    checkpoint_state: torch.Tensor,  # [slots, HV, V, K] BF16
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    rawv_cache: torch.Tensor | None,
    rawk_cache: torch.Tensor | None,
    beta_cache: torch.Tensor | None,
    out: torch.Tensor,
    query_start_loc: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    replay_indices: torch.Tensor,
    write_pos: torch.Tensor,
    cache_base: torch.Tensor,
    is_flush: torch.Tensor,
    max_cache_len: int,
    max_spec_len: int,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    null_block_id: int = -1,
    block_v: int = 64,
    num_warps: int = 2,
    num_stages: int = 2,
    nk: int = 2,
    bs_min: int = 4,
    dot_precision: str = "tf32" if is_tf32_supported else "ieee",
    pdl: bool = False,
    maxnreg: int | None = None,
    prefetch: bool = True,
    scratch: torch.Tensor | None = None,
    redundant_qk: bool = True,
) -> torch.Tensor:
    num_slots, HV, V, K = checkpoint_state.shape
    H = k_cache.shape[1]
    B = query_start_loc.shape[0] - 1
    conv_dim, width = conv_weight.shape
    if conv_window.dim() == 4:
        stride_win_slot, stride_win_dim = conv_window.stride(0), conv_window.stride(2)
        assert conv_window.stride(1) == 1 and conv_window.stride(3) == 1
    else:
        stride_win_slot, stride_win_dim = conv_window.stride(0), conv_window.stride(1)
        assert conv_window.stride(2) == 1
    assert K == V == 128 and width == 4 and max_spec_len == 7 and null_block_id == -1
    assert conv_dim == 2 * H * K + HV * V and stride_win_dim == max_spec_len + 2
    assert checkpoint_state.dtype == torch.bfloat16 and x.dtype == torch.bfloat16
    assert max_cache_len & (max_cache_len - 1) == 0 and d_cache.shape[2] == max_cache_len
    if scale is None:
        scale = K**-0.5
    if is_flush.dtype != torch.int8:
        is_flush = is_flush.to(torch.int8)

    BK = triton.next_power_of_2(K)
    BV = block_v
    BS = max(bs_min, triton.next_power_of_2(max_spec_len))
    BC = max(16, triton.next_power_of_2(max_cache_len))
    grid = (triton.cdiv(V, BV), B, HV)
    if scratch is None:
        scratch = torch.empty(
            grid[0] * grid[1] * grid[2] * max_spec_len * (2 * K + BV),
            dtype=x.dtype,
            device=x.device,
        )
    store_exact_raw = (
        rawv_cache is not None and rawk_cache is not None and beta_cache is not None
    )
    store_residual = (
        rawv_cache is not None and rawk_cache is not None and beta_cache is None
    )
    rawv_buf = rawv_cache if rawv_cache is not None else d_cache
    rawk_buf = rawk_cache if rawk_cache is not None else k_cache
    beta_buf = beta_cache if beta_cache is not None else g_cache
    options = {}
    if pdl:
        options.update(USE_GDC=True, launch_pdl=True)
    if maxnreg is not None:
        options["maxnreg"] = maxnreg
    gdn_replayssm_conv_fused_kernel[grid](
        x,
        conv_weight,
        conv_state,
        conv_window,
        window_indices,
        tickets,
        scratch,
        a,
        b,
        A_log,
        dt_bias,
        out,
        checkpoint_state,
        d_cache,
        k_cache,
        g_cache,
        rawv_buf,
        rawk_buf,
        beta_buf,
        query_start_loc,
        ssm_state_indices,
        replay_indices,
        write_pos,
        cache_base,
        is_flush,
        scale,
        x.stride(0),
        conv_weight.stride(0),
        conv_weight.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        stride_win_slot,
        stride_win_dim,
        window_indices.stride(0),
        a.stride(0),
        b.stride(0),
        out.stride(0),
        checkpoint_state.stride(0),
        d_cache.stride(0),
        k_cache.stride(0),
        g_cache.stride(0),
        rawv_buf.stride(0),
        rawk_buf.stride(0),
        beta_buf.stride(0),
        query_start_loc.stride(0),
        ssm_state_indices.stride(0),
        replay_indices.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        BS=BS,
        BC=BC,
        NK=nk,
        BKT=BK // nk,
        NT=32 * num_warps,
        SCR_W=2 * K + BV,
        MAX_CACHE_LEN=max_cache_len,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        STORE_EXACT_RAW=store_exact_raw,
        STORE_D_RESIDUAL=store_residual,
        STORE_K_RESIDUAL=store_residual,
        NULL_BLOCK_ID=null_block_id,
        DOT_PRECISION=dot_precision,
        PREFETCH=prefetch,
        REDUNDANT_QK=redundant_qk,
        num_warps=num_warps,
        num_stages=num_stages,
        **options,
    )
    return out
