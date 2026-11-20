"""CuTe DSL KDA conv-MTP verify, persistent (SGLANG_OPT_KDA_MTP_VERIFY_PERSIST).

Same arithmetic, operation for operation, as the register-resident ReplaySSM path of
kda_decode_mtp_v2.kda_decode_mtp_kernel (narrow 256-thread block, num_spec >= 1,
CACHE_RING), so outputs, rings and conv scratch are bit-identical. What changes is
the schedule: at the D4 shape (24 heads x 77 requests on 212 SMs) the per-block
kernel runs 4.36 lock-stepped waves of two blocks per SM, and each wave waits for
its 64 KB state tiles while the SM does nothing else. Here each block is persistent
over (request, head) items and, as soon as the current item's state sits in
registers, cp.asyncs the next item's state, conv windows, conv weights and
activations into the shared-memory stages, so those loads overlap the current
item's recurrence. Phase 1 reads its inputs from shared memory.
"""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass.cute.nvgpu import cpasync

from sglang.kernels.jit.cute_aot_cache import get_jit_cache
from sglang.kernels.ops.attention.kda_decode_mtp_v2 import (
    BLOCK_THREADS_NARROW,
    KERNEL_WIDTH,
    NUM_STATE_STAGES,
    P1_NUM_JOBS,
    P1_V_WARP_DIVISOR,
    TILE_K,
    VEC_SIZE,
    WARP_SIZE,
    _clock,
    _cp_async16,
    _cute_tensor,
    _fits_32bit_stride,
    _issue_state_tile,
    _p2_lanes,
    _qk_smem,
    _smid,
    _state_tile_v,
    _tensor_layout_key,
)

HEAD_DIM = TILE_K
BLOCK_THREADS = BLOCK_THREADS_NARROW


@cute.jit
def _issue_inputs(
    tidx: cutlass.Int32,
    sCW: cute.Tensor,
    sW: cute.Tensor,
    sDtb: cute.Tensor,
    sOnormWeight: cute.Tensor,
    sX: cute.Tensor,
    cs_q: cute.Tensor,
    cs_k: cute.Tensor,
    cs_v: cute.Tensor,
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    dt_bias: cute.Tensor,
    onorm_weight: cute.Tensor,
    x_q: cute.Tensor,
    x_k: cute.Tensor,
    x_v: cute.Tensor,
    g: cute.Tensor,
    onorm_g: cute.Tensor,
    slot_n: cutlass.Int64,
    hv_n: cutlass.Int32,
    bos_n: cutlass.Int32,
    ntok_n: cutlass.Int32,
    LO: cutlass.Constexpr[int],
    HI: cutlass.Constexpr[int],
    T_LOOP: cutlass.Constexpr[int],
    APPLY_ONORM: cutlass.Constexpr[bool],
):
    """cp.async one item's phase-1 input chunks [LO, HI) into the smem stages (no commit).

    Chunk order: conv windows, conv weights, dt_bias, onorm weight, then the
    activations (q, k, v, g, onorm gate) of T_LOOP token rows.
    """
    CW_CHUNKS = 3 * (KERNEL_WIDTH - 1) * (HEAD_DIM // 8)
    W_CHUNKS = 3 * HEAD_DIM
    P0_CHUNKS = CW_CHUNKS + W_CHUNKS + (2 if APPLY_ONORM else 1) * (HEAD_DIM // 4)
    head_n = hv_n * HEAD_DIM
    for it in cutlass.range_constexpr((HI - LO + BLOCK_THREADS - 1) // BLOCK_THREADS):
        c = tidx + it * BLOCK_THREADS + LO
        if c < HI:
            if c < CW_CHUNKS:
                j = c // ((KERNEL_WIDTH - 1) * (HEAD_DIM // 8))
                w = (c // (HEAD_DIM // 8)) % (KERNEL_WIDTH - 1)
                ch = (c % (HEAD_DIM // 8)) * 8
                dst = sCW.iterator + cute.crd2idx((j, w, ch), sCW.layout)
                if j == 0:
                    _cp_async16(
                        dst,
                        cs_q.iterator
                        + cute.crd2idx((slot_n, head_n + ch, w), cs_q.layout),
                    )
                elif j == 1:
                    _cp_async16(
                        dst,
                        cs_k.iterator
                        + cute.crd2idx((slot_n, head_n + ch, w), cs_k.layout),
                    )
                else:
                    _cp_async16(
                        dst,
                        cs_v.iterator
                        + cute.crd2idx((slot_n, head_n + ch, w), cs_v.layout),
                    )
            elif c < CW_CHUNKS + W_CHUNKS:
                cw = c - CW_CHUNKS
                j = cw // HEAD_DIM
                ch = cw % HEAD_DIM
                dst = sW.iterator + cute.crd2idx((j, ch, 0), sW.layout)
                if j == 0:
                    _cp_async16(
                        dst, w_q.iterator + cute.crd2idx((head_n + ch, 0), w_q.layout)
                    )
                elif j == 1:
                    _cp_async16(
                        dst, w_k.iterator + cute.crd2idx((head_n + ch, 0), w_k.layout)
                    )
                else:
                    _cp_async16(
                        dst, w_v.iterator + cute.crd2idx((head_n + ch, 0), w_v.layout)
                    )
            elif c < CW_CHUNKS + W_CHUNKS + HEAD_DIM // 4:
                ch = (c - CW_CHUNKS - W_CHUNKS) * 4
                _cp_async16(
                    sDtb.iterator + ch,
                    dt_bias.iterator + cute.crd2idx((head_n + ch,), dt_bias.layout),
                )
            elif c < P0_CHUNKS:
                if cutlass.const_expr(APPLY_ONORM):
                    ch = (c - CW_CHUNKS - W_CHUNKS - HEAD_DIM // 4) * 4
                    _cp_async16(
                        sOnormWeight.iterator + ch,
                        onorm_weight.iterator
                        + cute.crd2idx((ch,), onorm_weight.layout),
                    )
            else:
                cx = c - P0_CHUNKS
                j = cx // (T_LOOP * (HEAD_DIM // 8))
                t = (cx // (HEAD_DIM // 8)) % T_LOOP
                ch = (cx % (HEAD_DIM // 8)) * 8
                tok = bos_n + cutlass.min(t, ntok_n - 1)
                dst = sX.iterator + cute.crd2idx((j, t, ch), sX.layout)
                if j == 0:
                    _cp_async16(
                        dst, x_q.iterator + cute.crd2idx((0, tok, hv_n, ch), x_q.layout)
                    )
                elif j == 1:
                    _cp_async16(
                        dst, x_k.iterator + cute.crd2idx((0, tok, hv_n, ch), x_k.layout)
                    )
                elif j == 2:
                    _cp_async16(
                        dst, x_v.iterator + cute.crd2idx((0, tok, hv_n, ch), x_v.layout)
                    )
                elif j == 3:
                    _cp_async16(
                        dst, g.iterator + cute.crd2idx((0, tok, hv_n, ch), g.layout)
                    )
                else:
                    if cutlass.const_expr(APPLY_ONORM):
                        _cp_async16(
                            dst,
                            onorm_g.iterator
                            + cute.crd2idx((0, tok, hv_n, ch), onorm_g.layout),
                        )


@cute.jit
def _issue_state(
    state_g2s_copy: cute.TiledCopy,
    thr_state_copy,
    h0: cute.Tensor,
    sState: cute.Tensor,
    slot_n: cutlass.Int64,
    hv_n: cutlass.Int32,
    TILE_V: cutlass.Constexpr[int],
    STATE_STAGES: cutlass.Constexpr[int],
):
    gTiles = cute.local_tile(
        h0[(slot_n, hv_n, None, None)], (TILE_V, TILE_K), (None, 0)
    )
    for i_v in cutlass.range_constexpr(STATE_STAGES):
        _issue_state_tile(
            state_g2s_copy, thr_state_copy, gTiles, sState, i_v, STATE_STAGES
        )


@cute.kernel
def kda_mtp_persist_kernel(
    state_g2s_copy: cute.TiledCopy,
    h0: cute.Tensor,
    x_q: cute.Tensor,
    x_k: cute.Tensor,
    x_v: cute.Tensor,
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    cs_q: cute.Tensor,
    cs_k: cute.Tensor,
    cs_v: cute.Tensor,
    A_log: cute.Tensor,
    g: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    o: cute.Tensor,
    intermediate_state_indices: cute.Tensor,
    intermediate_conv_q: cute.Tensor,
    intermediate_conv_k: cute.Tensor,
    intermediate_conv_v: cute.Tensor,
    ring_rawv: cute.Tensor,
    ring_rawk: cute.Tensor,
    ring_g: cute.Tensor,
    ring_beta: cute.Tensor,
    onorm_g: cute.Tensor,
    onorm_weight: cute.Tensor,
    smem_qk_layout: cute.Layout,
    smem_state_layout: cute.Layout,
    ssm_state_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    prof: cute.Tensor,
    scale: cutlass.Constexpr[float],
    NUM_SPEC: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    N_ITEMS: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    APPLY_ONORM: cutlass.Constexpr[bool],
    onorm_eps: cutlass.Constexpr[float],
    PROF: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    in_warp_tid = tidx % WARP_SIZE
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    item0, _, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    T_LOOP = 1 + NUM_SPEC
    NUM_WARPS = BLOCK_THREADS // WARP_SIZE
    P1_V_WARPS = max(1, NUM_WARPS // P1_V_WARP_DIVISOR)
    P1_JOB_WARPS = (NUM_WARPS - P1_V_WARPS) // P1_NUM_JOBS
    P1_QKG_WARPS = P1_NUM_JOBS * P1_JOB_WARPS
    V_CH_PER_THREAD = TILE_K // (P1_V_WARPS * WARP_SIZE)
    P2_LANES_K = _p2_lanes(block_threads=BLOCK_THREADS, num_spec=NUM_SPEC)
    P2_ROWS_LANE = WARP_SIZE // P2_LANES_K
    P2_VEC = TILE_K // P2_LANES_K
    TILE_V = _state_tile_v(block_threads=BLOCK_THREADS, num_spec=NUM_SPEC)
    NUM_V_TILES = HEAD_DIM // TILE_V
    STATE_STAGES = min(NUM_STATE_STAGES, NUM_V_TILES)
    NUM_V_ROWS = TILE_V // NUM_WARPS
    P2_BATCHES = NUM_V_ROWS // P2_ROWS_LANE
    TILES_PER_PASS = NUM_V_TILES
    CW_CHUNKS = 3 * (KERNEL_WIDTH - 1) * (HEAD_DIM // 8)
    W_CHUNKS = 3 * HEAD_DIM
    P0_CHUNKS = CW_CHUNKS + W_CHUNKS + (2 if APPLY_ONORM else 1) * (HEAD_DIM // 4)
    X_TENSORS = 5 if APPLY_ONORM else 4
    X_CHUNKS = X_TENSORS * T_LOOP * (HEAD_DIM // 8)
    IN_CHUNKS = P0_CHUNKS + X_CHUNKS

    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sK = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sG = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_LOOP,)), 16)
    sVall = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T_LOOP * HEAD_DIM,)), 16
    )
    sState = smem.allocate_tensor(cutlass.Float32, smem_state_layout, 16)
    sCW = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout(
            (3, KERNEL_WIDTH - 1, HEAD_DIM),
            stride=((KERNEL_WIDTH - 1) * HEAD_DIM, HEAD_DIM, 1),
        ),
        16,
    )
    sW = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout(
            (3, HEAD_DIM, KERNEL_WIDTH),
            stride=(HEAD_DIM * KERNEL_WIDTH, KERNEL_WIDTH, 1),
        ),
        16,
    )
    sDtb = smem.allocate_tensor(cutlass.Float32, cute.make_layout((HEAD_DIM,)), 16)
    # q, k, v, g, onorm gate: row t holds token bos + min(t, n_tok - 1).
    sX = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout(
            (5, T_LOOP, HEAD_DIM), stride=(T_LOOP * HEAD_DIM, HEAD_DIM, 1)
        ),
        16,
    )
    sBraw = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T_LOOP,)), 16)
    if cutlass.const_expr(APPLY_ONORM):
        sOall = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T_LOOP * HEAD_DIM,)), 16
        )
        sOnormGate = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T_LOOP * HEAD_DIM,)), 16
        )
        sOnormWeight = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((HEAD_DIM,)), 16
        )
    else:
        sOall = sVall
        sOnormGate = sVall
        sOnormWeight = sDtb
    r_q = cute.make_rmem_tensor(
        cute.make_layout((P2_VEC,), stride=(1,)), cutlass.Float32
    )
    r_k = cute.make_rmem_tensor(
        cute.make_layout((P2_VEC,), stride=(1,)), cutlass.Float32
    )
    r_decay = cute.make_rmem_tensor(
        cute.make_layout((P2_VEC,), stride=(1,)), cutlass.Float32
    )
    R_STATE_ELEMS = max(
        TILES_PER_PASS * P2_BATCHES * P2_VEC, 2 * (KERNEL_WIDTH - 1) * VEC_SIZE
    )
    r_state = cute.make_rmem_tensor(
        cute.make_layout((R_STATE_ELEMS,), stride=(1,)), cutlass.Float32
    )
    r_wq = cute.make_rmem_tensor(
        cute.make_layout((KERNEL_WIDTH * VEC_SIZE,), stride=(1,)), cutlass.Float32
    )
    thr_state_copy = state_g2s_copy.get_slice(tidx)
    p1_job = warp_idx % P1_NUM_JOBS
    p1_par = warp_idx // P1_NUM_JOBS
    k_grp = in_warp_tid % P2_LANES_K
    row_grp = in_warp_tid // P2_LANES_K

    # ---- first item: static inputs + state before the grid-dependency wait ----
    first_hv = item0 % H
    first_n = item0 // H
    first_slot = cutlass.Int64(ssm_state_indices[first_n])
    first_bos = cu_seqlens[first_n]
    first_ntok = cu_seqlens[first_n + 1] - first_bos
    if first_slot >= 0:
        _issue_inputs(
            tidx,
            sCW,
            sW,
            sDtb,
            sOnormWeight,
            sX,
            cs_q,
            cs_k,
            cs_v,
            w_q,
            w_k,
            w_v,
            dt_bias,
            onorm_weight,
            x_q,
            x_k,
            x_v,
            g,
            onorm_g,
            first_slot,
            first_hv,
            first_bos,
            first_ntok,
            0,
            P0_CHUNKS,
            T_LOOP,
            APPLY_ONORM,
        )
        _issue_state(
            state_g2s_copy,
            thr_state_copy,
            h0,
            sState,
            first_slot,
            first_hv,
            TILE_V,
            STATE_STAGES,
        )
    cute.arch.griddepcontrol_wait()
    # The grid is at most two blocks per SM, all resident: dependents cannot take
    # an SM before these blocks exit, and they wait for this grid before reading.
    cute.arch.griddepcontrol_launch_dependents()
    if first_slot >= 0:
        _issue_inputs(
            tidx,
            sCW,
            sW,
            sDtb,
            sOnormWeight,
            sX,
            cs_q,
            cs_k,
            cs_v,
            w_q,
            w_k,
            w_v,
            dt_bias,
            onorm_weight,
            x_q,
            x_k,
            x_v,
            g,
            onorm_g,
            first_slot,
            first_hv,
            first_bos,
            first_ntok,
            P0_CHUNKS,
            IN_CHUNKS,
            T_LOOP,
            APPLY_ONORM,
        )
        cute.arch.cp_async_commit_group()
        if tidx < T_LOOP:
            sBraw[tidx] = cutlass.Float32(
                beta[0, first_bos + cutlass.min(tidx, first_ntok - 1), first_hv]
            )

    for item in cutlass.range(item0, N_ITEMS, grid_x):
        i_hv = item % H
        i_n = item // H
        head_off = i_hv * HEAD_DIM
        slot = cutlass.Int64(ssm_state_indices[i_n])
        bos = cu_seqlens[i_n]
        n_tok = cu_seqlens[i_n + 1] - bos
        scratch_row = intermediate_state_indices[i_n]
        nxt = item + grid_x
        has_next = nxt < N_ITEMS
        nxt_c = cutlass.min(nxt, N_ITEMS - 1)
        hv_n = nxt_c % H
        n_n = nxt_c // H
        slot_n = cutlass.Int64(ssm_state_indices[n_n])
        bos_n = cu_seqlens[n_n]
        ntok_n = cu_seqlens[n_n + 1] - bos_n
        pre_next = cutlass.Int32(0)
        if has_next:
            if slot_n >= 0:
                pre_next = cutlass.Int32(1)
        prof_row = cutlass.Int64(item) * 8
        if cutlass.const_expr(PROF):
            if tidx == 0:
                prof[prof_row + 0] = _clock()
                prof[prof_row + 7] = _smid()
        r_bn = cutlass.Float32(0.0)
        if slot < 0:
            # CUDA-graph padding row: zero output, no state, prefetch the next item.
            for i_t in cutlass.range_constexpr(T_LOOP):
                if tidx < HEAD_DIM:
                    o[0, bos + i_t, i_hv, tidx] = cutlass.BFloat16(0.0)
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            if pre_next == 1:
                _issue_inputs(
                    tidx,
                    sCW,
                    sW,
                    sDtb,
                    sOnormWeight,
                    sX,
                    cs_q,
                    cs_k,
                    cs_v,
                    w_q,
                    w_k,
                    w_v,
                    dt_bias,
                    onorm_weight,
                    x_q,
                    x_k,
                    x_v,
                    g,
                    onorm_g,
                    slot_n,
                    hv_n,
                    bos_n,
                    ntok_n,
                    0,
                    IN_CHUNKS,
                    T_LOOP,
                    APPLY_ONORM,
                )
                _issue_state(
                    state_g2s_copy,
                    thr_state_copy,
                    h0,
                    sState,
                    slot_n,
                    hv_n,
                    TILE_V,
                    STATE_STAGES,
                )
                if tidx < T_LOOP:
                    sBraw[tidx] = cutlass.Float32(
                        beta[0, bos_n + cutlass.min(tidx, ntok_n - 1), hv_n]
                    )
        else:
            r_exp_A = cutlass.Float32(0.0)
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            if cutlass.const_expr(PROF):
                if tidx == 0:
                    prof[prof_row + 1] = _clock()
            # ---------------- phase 1 (from the smem stages) ----------------
            if warp_idx < P1_QKG_WARPS:
                if p1_job == 0:
                    for i in range(VEC_SIZE):
                        k_idx = i * 32 + in_warp_tid
                        for w in range(KERNEL_WIDTH - 1):
                            r_state[w * VEC_SIZE + i] = cutlass.Float32(
                                sCW[0, w, k_idx]
                            )
                        for w in range(KERNEL_WIDTH):
                            r_wq[w * VEC_SIZE + i] = sW[0, k_idx, w]
                elif p1_job == 1:
                    for i in range(VEC_SIZE):
                        k_idx = i * 32 + in_warp_tid
                        for w in range(KERNEL_WIDTH - 1):
                            r_state[
                                (KERNEL_WIDTH - 1) * VEC_SIZE + w * VEC_SIZE + i
                            ] = cutlass.Float32(sCW[1, w, k_idx])
                        for w in range(KERNEL_WIDTH):
                            r_wq[w * VEC_SIZE + i] = sW[1, k_idx, w]
                if p1_job == 2:
                    r_exp_A = cute.math.exp(cutlass.Float32(A_log[i_hv]), fastmath=True)
                    for i in range(VEC_SIZE):
                        r_state[i] = sDtb[i * WARP_SIZE + in_warp_tid]
                if p1_job == 0:
                    for _pi in cutlass.range_constexpr(P1_JOB_WARPS - 1):
                        if _pi < cutlass.min(p1_par, n_tok):
                            for i in range(VEC_SIZE):
                                _xn = cutlass.Float32(sX[0, _pi, i * 32 + in_warp_tid])
                                r_state[0 * VEC_SIZE + i] = r_state[1 * VEC_SIZE + i]
                                r_state[1 * VEC_SIZE + i] = r_state[2 * VEC_SIZE + i]
                                r_state[2 * VEC_SIZE + i] = _xn
                elif p1_job == 1:
                    for _pi in cutlass.range_constexpr(P1_JOB_WARPS - 1):
                        if _pi < cutlass.min(p1_par, n_tok):
                            for i in range(VEC_SIZE):
                                _xn = cutlass.Float32(sX[1, _pi, i * 32 + in_warp_tid])
                                r_state[
                                    (KERNEL_WIDTH - 1) * VEC_SIZE + 0 * VEC_SIZE + i
                                ] = r_state[
                                    (KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i
                                ]
                                r_state[
                                    (KERNEL_WIDTH - 1) * VEC_SIZE + 1 * VEC_SIZE + i
                                ] = r_state[
                                    (KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i
                                ]
                                r_state[
                                    (KERNEL_WIDTH - 1) * VEC_SIZE + 2 * VEC_SIZE + i
                                ] = _xn
                for _job in cutlass.range_constexpr(P1_NUM_JOBS):
                    if p1_job == _job:
                        for producer_step in cutlass.range_constexpr(
                            (T_LOOP + P1_JOB_WARPS - 1) // P1_JOB_WARPS
                        ):
                            i_t = p1_par + producer_step * P1_JOB_WARPS
                            if (
                                cutlass.const_expr(T_LOOP % P1_JOB_WARPS == 0)
                                or i_t < T_LOOP
                            ):
                                tok_i = cutlass.min(i_t, n_tok - 1)
                                if cutlass.const_expr(_job == 0):
                                    for i_pair in range(VEC_SIZE // 2):
                                        i0 = i_pair * 2
                                        i1 = i_pair * 2 + 1
                                        k_idx0 = i0 * 32 + in_warp_tid
                                        k_idx1 = i1 * 32 + in_warp_tid
                                        r_conv_0 = 0.0
                                        r_conv_1 = 0.0
                                        for w in range(KERNEL_WIDTH - 1):
                                            r_conv_0 += (
                                                r_state[w * VEC_SIZE + i0]
                                                * r_wq[w * VEC_SIZE + i0]
                                            )
                                            r_conv_1 += (
                                                r_state[w * VEC_SIZE + i1]
                                                * r_wq[w * VEC_SIZE + i1]
                                            )
                                        r_xq_0 = cutlass.Float32(sX[0, tok_i, k_idx0])
                                        r_xq_1 = cutlass.Float32(sX[0, tok_i, k_idx1])
                                        _cwq_last_0 = r_wq[
                                            (KERNEL_WIDTH - 1) * VEC_SIZE + i0
                                        ]
                                        _cwq_last_1 = r_wq[
                                            (KERNEL_WIDTH - 1) * VEC_SIZE + i1
                                        ]
                                        r_conv_0 += r_xq_0 * _cwq_last_0
                                        r_conv_1 += r_xq_1 * _cwq_last_1
                                        e0 = cute.math.exp(-r_conv_0, fastmath=True)
                                        e1 = cute.math.exp(-r_conv_1, fastmath=True)
                                        sig_0 = cute.arch.rcp_approx(
                                            cutlass.Float32(1.0) + e0
                                        )
                                        sig_1 = cute.arch.rcp_approx(
                                            cutlass.Float32(1.0) + e1
                                        )
                                        r_q[i0] = r_conv_0 * sig_0
                                        r_q[i1] = r_conv_1 * sig_1
                                        r_state[0 * VEC_SIZE + i0] = r_state[
                                            1 * VEC_SIZE + i0
                                        ]
                                        r_state[0 * VEC_SIZE + i1] = r_state[
                                            1 * VEC_SIZE + i1
                                        ]
                                        r_state[1 * VEC_SIZE + i0] = r_state[
                                            2 * VEC_SIZE + i0
                                        ]
                                        r_state[1 * VEC_SIZE + i1] = r_state[
                                            2 * VEC_SIZE + i1
                                        ]
                                        r_state[2 * VEC_SIZE + i0] = r_xq_0
                                        r_state[2 * VEC_SIZE + i1] = r_xq_1
                                    sum_q = 0.0
                                    for i in range(VEC_SIZE):
                                        sum_q += r_q[i] * r_q[i]
                                    for offset in [16, 8, 4, 2, 1]:
                                        sum_q += cute.arch.shuffle_sync_bfly(
                                            sum_q,
                                            offset=offset,
                                            mask=-1,
                                            mask_and_clamp=31,
                                        )
                                    rnorm_q_scaled = (
                                        cute.math.rsqrt(sum_q + 1e-06, fastmath=True)
                                        * scale
                                    )
                                    for i in range(VEC_SIZE):
                                        r_q[i] = r_q[i] * rnorm_q_scaled
                                    for i in range(VEC_SIZE):
                                        k_idx = i * WARP_SIZE + in_warp_tid
                                        qk_grp = k_idx % P2_LANES_K
                                        qk_j = k_idx // P2_LANES_K
                                        sQ[i_t, qk_grp, qk_j // 4, qk_j % 4] = r_q[i]
                                elif cutlass.const_expr(_job == 1):
                                    r_b_raw = cutlass.Float32(0.0)
                                    if in_warp_tid == 0:
                                        r_b_raw = sBraw[tok_i]
                                    for i in range(VEC_SIZE):
                                        k_idx = i * 32 + in_warp_tid
                                        r_conv = (
                                            r_state[
                                                (KERNEL_WIDTH - 1) * VEC_SIZE
                                                + 0 * VEC_SIZE
                                                + i
                                            ]
                                            * r_wq[0 * VEC_SIZE + i]
                                        )
                                        r_conv += (
                                            r_state[
                                                (KERNEL_WIDTH - 1) * VEC_SIZE
                                                + 1 * VEC_SIZE
                                                + i
                                            ]
                                            * r_wq[1 * VEC_SIZE + i]
                                        )
                                        r_conv += (
                                            r_state[
                                                (KERNEL_WIDTH - 1) * VEC_SIZE
                                                + 2 * VEC_SIZE
                                                + i
                                            ]
                                            * r_wq[2 * VEC_SIZE + i]
                                        )
                                        r_xk = cutlass.Float32(sX[1, tok_i, k_idx])
                                        r_conv += (
                                            r_xk
                                            * r_wq[(KERNEL_WIDTH - 1) * VEC_SIZE + i]
                                        )
                                        r_conv = r_conv * cute.arch.rcp_approx(
                                            cutlass.Float32(1.0)
                                            + cute.math.exp(-r_conv, fastmath=True)
                                        )
                                        r_k[i] = r_conv
                                        r_state[
                                            (KERNEL_WIDTH - 1) * VEC_SIZE
                                            + 0 * VEC_SIZE
                                            + i
                                        ] = r_state[
                                            (KERNEL_WIDTH - 1) * VEC_SIZE
                                            + 1 * VEC_SIZE
                                            + i
                                        ]
                                        r_state[
                                            (KERNEL_WIDTH - 1) * VEC_SIZE
                                            + 1 * VEC_SIZE
                                            + i
                                        ] = r_state[
                                            (KERNEL_WIDTH - 1) * VEC_SIZE
                                            + 2 * VEC_SIZE
                                            + i
                                        ]
                                        r_state[
                                            (KERNEL_WIDTH - 1) * VEC_SIZE
                                            + 2 * VEC_SIZE
                                            + i
                                        ] = r_xk
                                    for i in range(VEC_SIZE):
                                        k_idx = i * WARP_SIZE + in_warp_tid
                                        ring_rawk[slot, i_hv, i_t, k_idx] = (
                                            cutlass.BFloat16(r_k[i])
                                        )
                                    sum_k = 0.0
                                    for i in range(VEC_SIZE):
                                        sum_k += r_k[i] * r_k[i]
                                    for offset in [16, 8, 4, 2, 1]:
                                        sum_k += cute.arch.shuffle_sync_bfly(
                                            sum_k,
                                            offset=offset,
                                            mask=-1,
                                            mask_and_clamp=31,
                                        )
                                    rnorm_k = cute.math.rsqrt(
                                        sum_k + 1e-06, fastmath=True
                                    )
                                    for i in range(VEC_SIZE):
                                        r_k[i] = r_k[i] * rnorm_k
                                    for i in range(VEC_SIZE):
                                        k_idx = i * WARP_SIZE + in_warp_tid
                                        qk_grp = k_idx % P2_LANES_K
                                        qk_j = k_idx // P2_LANES_K
                                        sK[i_t, qk_grp, qk_j // 4, qk_j % 4] = r_k[i]
                                    if in_warp_tid == 0:
                                        sBeta[i_t] = cute.arch.rcp_approx(
                                            cutlass.Float32(1.0)
                                            + cute.math.exp(-r_b_raw, fastmath=True)
                                        )
                                        ring_beta[slot, i_hv, i_t] = sBeta[i_t]
                                else:
                                    for i in range(VEC_SIZE):
                                        r_q[i] = cutlass.Float32(
                                            sX[3, tok_i, i * 32 + in_warp_tid]
                                        )
                                    for i in range(VEC_SIZE):
                                        k_idx = i * 32 + in_warp_tid
                                        r_g_raw = r_q[i] + r_state[i]
                                        exp_A_x = r_exp_A * r_g_raw
                                        sigmoid_val = cute.arch.rcp_approx(
                                            cutlass.Float32(1.0)
                                            + cute.math.exp(-exp_A_x, fastmath=True)
                                        )
                                        r_gk = lower_bound * sigmoid_val
                                        qk_grp = k_idx % P2_LANES_K
                                        qk_j = k_idx // P2_LANES_K
                                        sG[i_t, qk_grp, qk_j // 4, qk_j % 4] = (
                                            cute.math.exp(r_gk, fastmath=True)
                                        )
                                        ring_g[slot, i_hv, i_t, k_idx] = r_gk
                                if cutlass.const_expr(_job == 0):
                                    for i in range(VEC_SIZE):
                                        k_idx = i * 32 + in_warp_tid
                                        for w in range(KERNEL_WIDTH - 1):
                                            intermediate_conv_q[
                                                scratch_row, i_t, head_off + k_idx, w
                                            ] = cutlass.BFloat16(
                                                r_state[w * VEC_SIZE + i]
                                            )
                                elif cutlass.const_expr(_job == 1):
                                    for i in range(VEC_SIZE):
                                        k_idx = i * 32 + in_warp_tid
                                        for w in range(KERNEL_WIDTH - 1):
                                            intermediate_conv_k[
                                                scratch_row, i_t, head_off + k_idx, w
                                            ] = cutlass.BFloat16(
                                                r_state[
                                                    (KERNEL_WIDTH - 1) * VEC_SIZE
                                                    + w * VEC_SIZE
                                                    + i
                                                ]
                                            )
                                if cutlass.const_expr(_job == 0):
                                    for _a in range(P1_JOB_WARPS - 1):
                                        _nx_i = cutlass.min(i_t + 1 + _a, T_LOOP - 1)
                                        for i in range(VEC_SIZE):
                                            _xn = cutlass.Float32(
                                                sX[0, _nx_i, i * 32 + in_warp_tid]
                                            )
                                            r_state[0 * VEC_SIZE + i] = r_state[
                                                1 * VEC_SIZE + i
                                            ]
                                            r_state[1 * VEC_SIZE + i] = r_state[
                                                2 * VEC_SIZE + i
                                            ]
                                            r_state[2 * VEC_SIZE + i] = _xn
                                elif cutlass.const_expr(_job == 1):
                                    for _a in range(P1_JOB_WARPS - 1):
                                        _nx_i = cutlass.min(i_t + 1 + _a, T_LOOP - 1)
                                        for i in range(VEC_SIZE):
                                            _xn = cutlass.Float32(
                                                sX[1, _nx_i, i * 32 + in_warp_tid]
                                            )
                                            r_state[
                                                (KERNEL_WIDTH - 1) * VEC_SIZE
                                                + 0 * VEC_SIZE
                                                + i
                                            ] = r_state[
                                                (KERNEL_WIDTH - 1) * VEC_SIZE
                                                + 1 * VEC_SIZE
                                                + i
                                            ]
                                            r_state[
                                                (KERNEL_WIDTH - 1) * VEC_SIZE
                                                + 1 * VEC_SIZE
                                                + i
                                            ] = r_state[
                                                (KERNEL_WIDTH - 1) * VEC_SIZE
                                                + 2 * VEC_SIZE
                                                + i
                                            ]
                                            r_state[
                                                (KERNEL_WIDTH - 1) * VEC_SIZE
                                                + 2 * VEC_SIZE
                                                + i
                                            ] = _xn
            else:
                for _c in range(V_CH_PER_THREAD):
                    _v_idx = (tidx - P1_QKG_WARPS * 32) + _c * (
                        HEAD_DIM // V_CH_PER_THREAD
                    )
                    _csv0 = cutlass.Float32(sCW[2, 0, _v_idx])
                    _csv1 = cutlass.Float32(sCW[2, 1, _v_idx])
                    _csv2 = cutlass.Float32(sCW[2, 2, _v_idx])
                    _wv = [sW[2, _v_idx, w] for w in range(KERNEL_WIDTH)]
                    _win = [_csv0, _csv1, _csv2]
                    for _t in cutlass.range_constexpr(T_LOOP):
                        _win.append(
                            cutlass.Float32(sX[2, cutlass.min(_t, n_tok - 1), _v_idx])
                        )
                    for _t in cutlass.range_constexpr(T_LOOP):
                        _vconv = _win[_t] * _wv[0]
                        for _w in cutlass.range_constexpr(1, KERNEL_WIDTH):
                            _vconv += _win[_t + _w] * _wv[_w]
                        _vconv = _vconv * cute.arch.rcp_approx(
                            cutlass.Float32(1.0) + cute.math.exp(-_vconv, fastmath=True)
                        )
                        sVall[_t * HEAD_DIM + _v_idx] = _vconv
                        if cutlass.const_expr(APPLY_ONORM):
                            gate_raw = cutlass.Float32(
                                sX[4, cutlass.min(_t, n_tok - 1), _v_idx]
                            )
                            sOnormGate[_t * HEAD_DIM + _v_idx] = cute.arch.rcp_approx(
                                cutlass.Float32(1.0)
                                + cute.math.exp(-gate_raw, fastmath=True)
                            )
                        ring_rawv[slot, i_hv, _t, _v_idx] = cutlass.BFloat16(_vconv)
                        for _w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
                            intermediate_conv_v[
                                scratch_row, _t, head_off + _v_idx, _w
                            ] = cutlass.BFloat16(_win[_t + 1 + _w])

            # ---------------- state -> registers, then prefetch the next item ----
            for i_l in cutlass.range_constexpr(TILES_PER_PASS):
                for b in range(P2_BATCHES):
                    _st = (i_l * P2_BATCHES + b) * P2_VEC
                    _row = warp_idx * NUM_V_ROWS + b * P2_ROWS_LANE + row_grp
                    for j in range(P2_VEC):
                        r_state[_st + j] = cutlass.Float32(
                            sState[_row, j * P2_LANES_K + k_grp, i_l % STATE_STAGES]
                        )
            # Phase-1 outputs published; the input stages and sState are free.
            cute.arch.barrier()
            if cutlass.const_expr(PROF):
                if tidx == 0:
                    prof[prof_row + 3] = _clock()
            if pre_next == 1:
                _issue_inputs(
                    tidx,
                    sCW,
                    sW,
                    sDtb,
                    sOnormWeight,
                    sX,
                    cs_q,
                    cs_k,
                    cs_v,
                    w_q,
                    w_k,
                    w_v,
                    dt_bias,
                    onorm_weight,
                    x_q,
                    x_k,
                    x_v,
                    g,
                    onorm_g,
                    slot_n,
                    hv_n,
                    bos_n,
                    ntok_n,
                    0,
                    IN_CHUNKS,
                    T_LOOP,
                    APPLY_ONORM,
                )
                _issue_state(
                    state_g2s_copy,
                    thr_state_copy,
                    h0,
                    sState,
                    slot_n,
                    hv_n,
                    TILE_V,
                    STATE_STAGES,
                )
                if tidx < T_LOOP:
                    r_bn = cutlass.Float32(
                        beta[0, bos_n + cutlass.min(tidx, ntok_n - 1), hv_n]
                    )

            # ---------------- phase 2: the token chain on resident state ----------
            r_beta_val = cutlass.Float32(0.0)
            for i_t in cutlass.range(T_LOOP):
                r_beta_val = sBeta[i_t]
                for j in range(P2_VEC):
                    r_q[j] = sQ[i_t, k_grp, j // 4, j % 4]
                    r_k[j] = sK[i_t, k_grp, j // 4, j % 4]
                    r_decay[j] = sG[i_t, k_grp, j // 4, j % 4]
                for i_l in range(TILES_PER_PASS):
                    v_base = i_l * TILE_V
                    for b in range(P2_BATCHES):
                        _st = (i_l * P2_BATCHES + b) * P2_VEC
                        v_row = warp_idx * NUM_V_ROWS + b * P2_ROWS_LANE + row_grp
                        r_v = sVall[i_t * HEAD_DIM + v_base + v_row]
                        shk_1 = 0.0
                        shk_2 = 0.0
                        for jp in range(P2_VEC // 2):
                            _p = jp * 2
                            r_state[_st + _p], r_state[_st + _p + 1] = (
                                cute.arch.mul_packed_f32x2(
                                    (r_decay[_p], r_decay[_p + 1]),
                                    (r_state[_st + _p], r_state[_st + _p + 1]),
                                )
                            )
                            shk_1, shk_2 = cute.arch.fma_packed_f32x2(
                                src_a=(r_state[_st + _p], r_state[_st + _p + 1]),
                                src_b=(r_k[_p], r_k[_p + 1]),
                                src_c=(shk_1, shk_2),
                            )
                        shk = shk_1 + shk_2
                        for offset in [16, 8, 4, 2, 1]:
                            if cutlass.const_expr(offset < P2_LANES_K):
                                shk += cute.arch.shuffle_sync_bfly(
                                    shk, offset=offset, mask=-1, mask_and_clamp=31
                                )
                        vnb = (r_v - shk) * r_beta_val
                        shq_1 = 0.0
                        shq_2 = 0.0
                        for jp in range(P2_VEC // 2):
                            _p = jp * 2
                            r_state[_st + _p], r_state[_st + _p + 1] = (
                                cute.arch.fma_packed_f32x2(
                                    src_a=(vnb, vnb),
                                    src_b=(r_k[_p], r_k[_p + 1]),
                                    src_c=(r_state[_st + _p], r_state[_st + _p + 1]),
                                )
                            )
                            shq_1, shq_2 = cute.arch.fma_packed_f32x2(
                                src_a=(r_state[_st + _p], r_state[_st + _p + 1]),
                                src_b=(r_q[_p], r_q[_p + 1]),
                                src_c=(shq_1, shq_2),
                            )
                        shq = shq_1 + shq_2
                        for offset in [16, 8, 4, 2, 1]:
                            if cutlass.const_expr(offset < P2_LANES_K):
                                shq += cute.arch.shuffle_sync_bfly(
                                    shq, offset=offset, mask=-1, mask_and_clamp=31
                                )
                        if k_grp == 0 and i_t < n_tok:
                            if cutlass.const_expr(APPLY_ONORM):
                                sOall[i_t * HEAD_DIM + v_base + v_row] = shq
                            else:
                                o[0, bos + i_t, i_hv, v_base + v_row] = (
                                    cutlass.BFloat16(shq)
                                )
            if cutlass.const_expr(PROF):
                if tidx == 0:
                    prof[prof_row + 4] = _clock()
            if cutlass.const_expr(APPLY_ONORM):
                cute.arch.barrier()
                for i_t in cutlass.range(warp_idx, T_LOOP, NUM_WARPS):
                    sumsq = cutlass.Float32(0.0)
                    for i in range(VEC_SIZE):
                        _o = sOall[i_t * HEAD_DIM + i * 32 + in_warp_tid]
                        sumsq += _o * _o
                    for offset in [16, 8, 4, 2, 1]:
                        sumsq += cute.arch.shuffle_sync_bfly(
                            sumsq, offset=offset, mask=-1, mask_and_clamp=31
                        )
                    rms = cute.math.rsqrt(
                        sumsq / cutlass.Float32(HEAD_DIM) + onorm_eps, fastmath=True
                    )
                    for i in range(VEC_SIZE):
                        v_idx = i * 32 + in_warp_tid
                        raw_o = sOall[i_t * HEAD_DIM + v_idx]
                        gate = sOnormGate[i_t * HEAD_DIM + v_idx]
                        weight = sOnormWeight[v_idx]
                        if i_t < n_tok:
                            o[0, bos + i_t, i_hv, v_idx] = cutlass.BFloat16(
                                raw_o * rms * weight * gate
                            )
            # Next item's raw beta (its loads were issued with the prefetch); phase 1
            # of this item read sBraw before the barrier above.
            if pre_next == 1:
                if tidx < T_LOOP:
                    sBraw[tidx] = r_bn
            if cutlass.const_expr(PROF):
                if tidx == 0:
                    prof[prof_row + 5] = _clock()
    cute.arch.cp_async_wait_group(0)


@cute.jit
def _run_persist(
    h0: cute.Tensor,
    x_q: cute.Tensor,
    x_k: cute.Tensor,
    x_v: cute.Tensor,
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    cs_q: cute.Tensor,
    cs_k: cute.Tensor,
    cs_v: cute.Tensor,
    A_log: cute.Tensor,
    g: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    out: cute.Tensor,
    intermediate_state_indices: cute.Tensor,
    intermediate_conv_q: cute.Tensor,
    intermediate_conv_k: cute.Tensor,
    intermediate_conv_v: cute.Tensor,
    ssm_state_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    ring_rawv: cute.Tensor,
    ring_rawk: cute.Tensor,
    ring_g: cute.Tensor,
    ring_beta: cute.Tensor,
    onorm_g: cute.Tensor,
    onorm_weight: cute.Tensor,
    prof: cute.Tensor,
    scale: cutlass.Constexpr[float],
    H: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    NUM_SPEC: cutlass.Constexpr[int],
    GRID: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    APPLY_ONORM: cutlass.Constexpr[bool],
    onorm_eps: cutlass.Constexpr[float],
    PROF: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    p2_lanes = _p2_lanes(block_threads=BLOCK_THREADS, num_spec=NUM_SPEC)
    p2_vec = TILE_K // p2_lanes
    qk_stride, qk_elems = _qk_smem(block_threads=BLOCK_THREADS, num_spec=NUM_SPEC)
    smem_qk_layout = cute.make_layout(
        (1 + NUM_SPEC, p2_lanes, p2_vec // 4, 4), stride=qk_stride
    )
    smem_state_stride = TILE_K + p2_lanes
    TILE_V = _state_tile_v(block_threads=BLOCK_THREADS, num_spec=NUM_SPEC)
    state_stages = min(NUM_STATE_STAGES, TILE_K // TILE_V)
    state_stage_elems = TILE_V * smem_state_stride
    smem_state_layout = cute.make_layout(
        (TILE_V, TILE_K, state_stages),
        stride=(smem_state_stride, 1, state_stage_elems),
    )
    state_copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    state_g2s_copy = cute.make_tiled_copy_tv(
        state_copy_atom,
        thr_layout=cute.make_layout(
            (TILE_V, BLOCK_THREADS // TILE_V),
            stride=(BLOCK_THREADS // TILE_V, 1),
        ),
        val_layout=cute.make_layout((1, 4)),
    )
    t_loop = 1 + NUM_SPEC
    smem_bytes = (
        (3 * t_loop * qk_elems + t_loop + t_loop * TILE_K) * 4
        + state_stages * state_stage_elems * 4
        + (t_loop * TILE_K * 4 if APPLY_ONORM else 0)
        + ((t_loop + 1) * TILE_K * 4 if APPLY_ONORM else 0)
        + 3 * (KERNEL_WIDTH - 1) * TILE_K * 2
        + 3 * TILE_K * KERNEL_WIDTH * 4
        + TILE_K * 4
        + 5 * t_loop * TILE_K * 2
        + 16 * ((t_loop * 4 + 15) // 16)
        + 512
    )
    kda_mtp_persist_kernel(
        state_g2s_copy,
        h0,
        x_q,
        x_k,
        x_v,
        w_q,
        w_k,
        w_v,
        cs_q,
        cs_k,
        cs_v,
        A_log,
        g,
        dt_bias,
        beta,
        out,
        intermediate_state_indices,
        intermediate_conv_q,
        intermediate_conv_k,
        intermediate_conv_v,
        ring_rawv,
        ring_rawk,
        ring_g,
        ring_beta,
        onorm_g,
        onorm_weight,
        smem_qk_layout,
        smem_state_layout,
        ssm_state_indices,
        cu_seqlens,
        prof,
        scale,
        NUM_SPEC,
        H,
        H * N,
        lower_bound,
        APPLY_ONORM,
        onorm_eps,
        PROF,
    ).launch(
        grid=(GRID, 1, 1),
        block=[BLOCK_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
        use_pdl=True,
        max_number_threads=(BLOCK_THREADS, 1, 1),
        min_blocks_per_mp=2,
    )


_PERSIST_COMPILED = get_jit_cache(
    "kimi_k3_kda_mtp_verify_persist",
    source_paths=(__file__,),
    enable_tvm_ffi=False,
)


def run_persist(
    args, *, H, N, num_spec, scale, lower_bound, apply_onorm, onorm_eps, prof
):
    """args: the v2 wrapper's tensor tuple (state ... onorm_weight) plus the prof buffer."""
    import torch

    num_sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    grid = min(H * N, 2 * num_sms)
    fits_32bit = tuple(_fits_32bit_stride(tensor) for tensor in args)
    key = (
        torch.cuda.get_device_capability(),
        H,
        N,
        num_spec,
        grid,
        apply_onorm,
        float(onorm_eps) if apply_onorm else 0.0,
        prof,
        float(scale),
        float(lower_bound),
        *(_tensor_layout_key(tensor, fits) for tensor, fits in zip(args, fits_32bit)),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = _PERSIST_COMPILED[key] if key in _PERSIST_COMPILED else None
    if compiled is None:
        cute_args = tuple(
            _cute_tensor(tensor, dynamic=False, fits_32bit_stride=fits)
            for tensor, fits in zip(args, fits_32bit)
        )
        compiled = cute.compile(
            _run_persist,
            *cute_args,
            scale=float(scale),
            H=H,
            N=N,
            NUM_SPEC=num_spec,
            GRID=grid,
            lower_bound=float(lower_bound),
            APPLY_ONORM=apply_onorm,
            onorm_eps=float(onorm_eps) if apply_onorm else 0.0,
            PROF=prof,
            stream=stream,
        )
        _PERSIST_COMPILED[key] = compiled
    compiled(
        *(
            _cute_tensor(tensor, fits_32bit_stride=fits)
            for tensor, fits in zip(args, fits_32bit)
        ),
        stream,
    )
