# SPDX-License-Identifier: Apache-2.0
"""ReplaySSM GDN verify with the causal conv1d update (width 4, SiLU) in its prologue.

One launch replaces ``causal_conv1d_update`` followed by the circular verify kernel
(``gdn_replayssm_spec_circular_kernel``, verify launch) for the BF16-checkpoint verify of
7 linear draft tokens at head dim 128. Every byte it writes (verify output, rings, conv
state, conv window) matches the two stock kernels: the conv output is rounded to the
activation dtype before the recurrence, exactly as the stock conv stores it.

Each program recomputes the q and k conv of its key head; every program of a key-head
group reads the old q/k conv state, so the last one to take a ticket writes the new state.
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
def _conv_silu(
    x_row,
    stride_x_t: tl.constexpr,
    cs_row,
    stride_cs_dim: tl.constexpr,
    stride_cs_tok: tl.constexpr,
    conv_w,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    tok,
    ch,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    # Operation order of _causal_conv1d_update_kernel: activation-dtype products summed in fp32.
    s0 = tl.load(cs_row + ch * stride_cs_dim)
    s1 = tl.load(cs_row + ch * stride_cs_dim + stride_cs_tok)
    s2 = tl.load(cs_row + ch * stride_cs_dim + 2 * stride_cs_tok)
    acc = tl.zeros([ROWS, COLS], dtype=tl.float32)
    for j in tl.static_range(4):
        src = tok + (j - 3)
        xj = tl.load(
            x_row + tl.maximum(src, 0) * stride_x_t + ch, mask=src >= 0, other=0.0
        )
        tap = tl.where(
            src >= 0, xj, tl.where(src == -1, s2, tl.where(src == -2, s1, s0))
        )
        w = tl.load(conv_w + ch * stride_w_dim + j * stride_w_width)
        acc += tap * w
    acc = acc / (1 + tl.exp(-acc))
    return acc.to(x_row.dtype.element_ty)


@triton.jit
def _window_values(
    x_row,
    stride_x_t: tl.constexpr,
    cs_row,
    stride_cs_dim: tl.constexpr,
    stride_cs_tok: tl.constexpr,
    ch0,
    NCH: tl.constexpr,
    T: tl.constexpr,
    FLAT: tl.constexpr,
):
    # Window position p of a channel holds old state column p + 1 (p < 2) or token p - 2.
    f = tl.arange(0, FLAT)
    c = f // (T + 2)
    p = f % (T + 2)
    m = f < NCH * (T + 2)
    old = tl.load(
        cs_row + (ch0 + c) * stride_cs_dim + (p + 1) * stride_cs_tok,
        mask=m & (p < 2),
        other=0.0,
    )
    new = tl.load(
        x_row + tl.maximum(p - 2, 0) * stride_x_t + ch0 + c,
        mask=m & (p >= 2),
        other=0.0,
    )
    return tl.where(p < 2, old, new)


@triton.jit
def _store_window(
    win_row, stride_win_dim: tl.constexpr, ch0, vals, NCH: tl.constexpr, T: tl.constexpr, FLAT: tl.constexpr
):
    f = tl.arange(0, FLAT)
    c = f // (T + 2)
    p = f % (T + 2)
    tl.store(win_row + (ch0 + c) * stride_win_dim + p, vals, mask=f < NCH * (T + 2))


@triton.jit
def _new_state_values(
    x_row, stride_x_t: tl.constexpr, ch0, NCH: tl.constexpr, T: tl.constexpr, FLAT: tl.constexpr
):
    g = tl.arange(0, FLAT)
    return tl.load(
        x_row + (T - 3 + g % 3) * stride_x_t + ch0 + g // 3, mask=g < NCH * 3
    )


@triton.jit
def _store_state(
    cs_row,
    stride_cs_dim: tl.constexpr,
    stride_cs_tok: tl.constexpr,
    ch0,
    vals,
    NCH: tl.constexpr,
    FLAT: tl.constexpr,
):
    g = tl.arange(0, FLAT)
    tl.store(
        cs_row + (ch0 + g // 3) * stride_cs_dim + (g % 3) * stride_cs_tok,
        vals,
        mask=g < NCH * 3,
    )


@triton.jit
def _conv_writes(
    x_row,
    stride_x_t: tl.constexpr,
    cs_row,
    stride_cs_dim: tl.constexpr,
    stride_cs_tok: tl.constexpr,
    win_row,
    stride_win_dim: tl.constexpr,
    ch0,
    NCH: tl.constexpr,
    T: tl.constexpr,
    WIN_FLAT: tl.constexpr,
    ST_FLAT: tl.constexpr,
):
    win = _window_values(
        x_row, stride_x_t, cs_row, stride_cs_dim, stride_cs_tok, ch0, NCH, T, WIN_FLAT
    )
    st = _new_state_values(x_row, stride_x_t, ch0, NCH, T, ST_FLAT)
    # Every thread's old-state read precedes the in-place state update.
    tl.debug_barrier()
    _store_window(win_row, stride_win_dim, ch0, win, NCH, T, WIN_FLAT)
    _store_state(cs_row, stride_cs_dim, stride_cs_tok, ch0, st, NCH, ST_FLAT)


@triton.jit
def gdn_replayssm_conv_fused_kernel(
    x,  # [total_tokens, >= conv_dim] pre-conv rows, channels q | k | v
    conv_w,  # [conv_dim, 4]
    conv_state,  # [slots, conv_dim, 3]
    conv_window,  # [slots, conv_dim, T + 2] physical window
    window_indices,  # [B]
    tickets,  # [>= B, H] int32, zero between launches
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
    BV: tl.constexpr,
    BS: tl.constexpr,
    BC: tl.constexpr,
    BKT: tl.constexpr,
    T: tl.constexpr,
    NT: tl.constexpr,
    WIN_FLAT_QK: tl.constexpr,
    WIN_FLAT_V: tl.constexpr,
    ST_FLAT_QK: tl.constexpr,
    ST_FLAT_V: tl.constexpr,
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

    ch_q = i_h * K
    ch_k = H * K + i_h * K
    ch_v = 2 * H * K + i_hv * V + i_v * BV
    x_row = x + bos * stride_x_t
    cs_row = conv_state + state_idx * stride_cs_slot
    win_idx = tl.load(window_indices + i_n * stride_window_indices).to(tl.int64)
    win_row = conv_window + win_idx * stride_win_slot
    write_k = (i_v == 0) and (i_hv == i_h * (HV // H))

    if PREFETCH:
        # Inputs the prerequisite grid does not write: conv weights / state, checkpoint, key ring.
        lines = tl.arange(0, 8) * 64
        _prefetch_l2(conv_w + ch_q * stride_w_dim + tl.minimum(lines, K * 4 - 1))
        _prefetch_l2(conv_w + ch_k * stride_w_dim + tl.minimum(lines, K * 4 - 1))
        _prefetch_l2(conv_w + ch_v * stride_w_dim + tl.minimum(lines, BV * 4 - 1))
        _prefetch_l2(cs_row + ch_q * stride_cs_dim + tl.minimum(lines, K * 3 - 1))
        _prefetch_l2(cs_row + ch_k * stride_cs_dim + tl.minimum(lines, K * 3 - 1))
        _prefetch_l2(cs_row + ch_v * stride_cs_dim + tl.minimum(lines, BV * 3 - 1))
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

    b_is_flush = tl.load(is_flush_flags + replay_idx) != 0
    if b_is_flush:
        # The stock verify launch skips flush rows; only the conv writes remain.
        if USE_GDC:
            tl.extra.cuda.gdc_wait()
        _conv_writes(
            x_row, stride_x_t, cs_row, stride_cs_dim, stride_cs_tok, win_row,
            stride_win_dim, ch_v, BV, T, WIN_FLAT_V, ST_FLAT_V,
        )
        if write_k:
            _conv_writes(
                x_row, stride_x_t, cs_row, stride_cs_dim, stride_cs_tok, win_row,
                stride_win_dim, ch_q, K, T, WIN_FLAT_QK, ST_FLAT_QK,
            )
            _conv_writes(
                x_row, stride_x_t, cs_row, stride_cs_dim, stride_cs_tok, win_row,
                stride_win_dim, ch_k, K, T, WIN_FLAT_QK, ST_FLAT_QK,
            )
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

    # The prerequisite writes the pre-conv rows and the gates.
    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    a_s = tl.load(a + (bos + o_s_safe) * stride_a_t + i_hv, mask=mask_s, other=0.0).to(
        tl.float32
    )
    b_s = tl.load(b + (bos + o_s_safe) * stride_b_t + i_hv, mask=mask_s, other=0.0).to(
        tl.float32
    )
    x_g = a_s + dt_bias_val
    softplus_x = tl.where(x_g <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x_g)), x_g)
    g_s = tl.where(mask_s, -tl.exp(A_log_val) * softplus_x, 0.0)
    beta_s = tl.where(mask_s, 1.0 / (1.0 + tl.exp(-b_s)), 0.0)
    G_s = tl.cumsum(g_s, axis=0)
    expG_s = tl.exp(G_s)

    # Conv of the program's channels; padding token rows repeat token 0 like o_s_safe loads.
    o_kt0 = tl.arange(0, BKT)
    tok_r = o_s_safe[:, None]
    q_c0 = _conv_silu(
        x_row, stride_x_t, cs_row, stride_cs_dim, stride_cs_tok, conv_w, stride_w_dim,
        stride_w_width, tok_r, (ch_q + o_kt0)[None, :], BS, BKT,
    )
    k_c0 = _conv_silu(
        x_row, stride_x_t, cs_row, stride_cs_dim, stride_cs_tok, conv_w, stride_w_dim,
        stride_w_width, tok_r, (ch_k + o_kt0)[None, :], BS, BKT,
    )
    q_c1 = _conv_silu(
        x_row, stride_x_t, cs_row, stride_cs_dim, stride_cs_tok, conv_w, stride_w_dim,
        stride_w_width, tok_r, (ch_q + BKT + o_kt0)[None, :], BS, BKT,
    )
    k_c1 = _conv_silu(
        x_row, stride_x_t, cs_row, stride_cs_dim, stride_cs_tok, conv_w, stride_w_dim,
        stride_w_width, tok_r, (ch_k + BKT + o_kt0)[None, :], BS, BKT,
    )
    v_c = _conv_silu(
        x_row, stride_x_t, cs_row, stride_cs_dim, stride_cs_tok, conv_w, stride_w_dim,
        stride_w_width, o_s_safe[None, :], (ch_v + tl.arange(0, BV))[:, None], BV, BS,
    )

    # Ticket after this program's last old q/k state read; NT lanes keep one owner per element.
    o_t = tl.arange(0, NT)
    p_ticket = tickets + i_n * H + i_h + o_t * 0
    if write_k:
        win_q = _window_values(
            x_row, stride_x_t, cs_row, stride_cs_dim, stride_cs_tok, ch_q, K, T, WIN_FLAT_QK
        )
        win_k = _window_values(
            x_row, stride_x_t, cs_row, stride_cs_dim, stride_cs_tok, ch_k, K, T, WIN_FLAT_QK
        )
        tl.debug_barrier()
        ticket = tl.atomic_add(
            p_ticket, tl.where(o_t == 0, 1, 0), mask=o_t == 0, sem="release", scope="gpu"
        )
        _store_window(win_row, stride_win_dim, ch_q, win_q, K, T, WIN_FLAT_QK)
        _store_window(win_row, stride_win_dim, ch_k, win_k, K, T, WIN_FLAT_QK)
    else:
        tl.debug_barrier()
        ticket = tl.atomic_add(
            p_ticket, tl.where(o_t == 0, 1, 0), mask=o_t == 0, sem="release", scope="gpu"
        )

    if USE_QK_L2NORM_IN_KERNEL:
        qnorm_acc = tl.zeros([BS], dtype=tl.float32)
        knorm_acc = tl.zeros([BS], dtype=tl.float32)
        qn = tl.where(mask_s[:, None], q_c0.to(tl.float32), 0.0)
        knn = tl.where(mask_s[:, None], k_c0.to(tl.float32), 0.0)
        qnorm_acc += tl.sum(qn * qn, axis=1)
        knorm_acc += tl.sum(knn * knn, axis=1)
        qn = tl.where(mask_s[:, None], q_c1.to(tl.float32), 0.0)
        knn = tl.where(mask_s[:, None], k_c1.to(tl.float32), 0.0)
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

    for kk in tl.static_range(2):
        if kk == 0:
            q_conv = q_c0
            k_conv = k_c0
        else:
            q_conv = q_c1
            k_conv = k_c1
        o_kt = kk * BKT + tl.arange(0, BKT)
        mask_kt = o_kt < K
        q_tile = q_conv.to(tl.float32)
        k_tile = k_conv.to(tl.float32)
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

    v_tile = tl.where(mask_v[:, None] & mask_s[None, :], v_c.to(tl.float32), 0.0)
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

    _conv_writes(
        x_row, stride_x_t, cs_row, stride_cs_dim, stride_cs_tok, win_row,
        stride_win_dim, ch_v, BV, T, WIN_FLAT_V, ST_FLAT_V,
    )
    # The store follows the ticket's data dependency; all readers released before it.
    last = tl.sum(tl.where(o_t == 0, ticket, 0), axis=0) == (HV // H) * (V // BV) - 1
    if last:
        st_q = _new_state_values(x_row, stride_x_t, ch_q, K, T, ST_FLAT_QK)
        st_k = _new_state_values(x_row, stride_x_t, ch_k, K, T, ST_FLAT_QK)
        _store_state(cs_row, stride_cs_dim, stride_cs_tok, ch_q, st_q, K, ST_FLAT_QK)
        _store_state(cs_row, stride_cs_dim, stride_cs_tok, ch_k, st_k, K, ST_FLAT_QK)
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
    bs_min: int = 4,
    dot_precision: str = "tf32" if is_tf32_supported else "ieee",
    pdl: bool = False,
    maxnreg: int | None = None,
    prefetch: bool = True,
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
    assert conv_weight.stride() == (4, 1) and conv_state.stride()[1:] == (3, 1)
    assert max_cache_len & (max_cache_len - 1) == 0 and d_cache.shape[2] == max_cache_len
    if scale is None:
        scale = K**-0.5
    if is_flush.dtype != torch.int8:
        is_flush = is_flush.to(torch.int8)

    BV = block_v
    BS = max(bs_min, triton.next_power_of_2(max_spec_len))
    BC = max(16, triton.next_power_of_2(max_cache_len))
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
    grid = (triton.cdiv(V, BV), B, HV)
    gdn_replayssm_conv_fused_kernel[grid](
        x,
        conv_weight,
        conv_state,
        conv_window,
        window_indices,
        tickets,
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
        BV=BV,
        BS=BS,
        BC=BC,
        BKT=K // 2,
        T=max_spec_len,
        NT=32 * num_warps,
        WIN_FLAT_QK=triton.next_power_of_2(K * (max_spec_len + 2)),
        WIN_FLAT_V=triton.next_power_of_2(BV * (max_spec_len + 2)),
        ST_FLAT_QK=triton.next_power_of_2(K * 3),
        ST_FLAT_V=triton.next_power_of_2(BV * 3),
        MAX_CACHE_LEN=max_cache_len,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        STORE_EXACT_RAW=store_exact_raw,
        STORE_D_RESIDUAL=store_residual,
        STORE_K_RESIDUAL=store_residual,
        NULL_BLOCK_ID=null_block_id,
        DOT_PRECISION=dot_precision,
        PREFETCH=prefetch,
        num_warps=num_warps,
        num_stages=num_stages,
        **options,
    )
    return out
