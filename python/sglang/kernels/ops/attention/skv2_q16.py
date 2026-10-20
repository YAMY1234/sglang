# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Derived from NVIDIA/cutlass v4.8.0 examples/python/CuTeDSL/cute_ext/blackwell/attention/
# (gqa_decode_simple.py, gqa_decode_opt.py; BSD-3-Clause): warp-specialized split-KV decode structure,
# rewritten for paged FP8 KV.

# Modified by NVIDIA: paged FP8 verify with explicit local reduction.
# Modified by NVIDIA: stock split-KV rounding (page splits, one FP8 P rounding, bf16 partials).
# Modified by NVIDIA: 16-head GQA groups with query rows on TMEM lanes (S = Q K^T, O = P V accumulated in TMEM).

"""Paged FP8 q7 split attention for 16 query heads per KV head (TP2 target verify), B requests per launch.

The 112 query rows (7 positions x 16 heads) of one request sit on the TMEM lanes of a 128-row MMA, so every
softmax thread owns one row: row maxima and sums need no cross-thread reduction, and the output accumulates
in TMEM across the split's 128-token blocks (rescaled in place when the running maximum moves). Rounding
follows sglang split_kv_verify: the stock page split (SPLITS per request), the raw running maximum, one e4m3
rounding of P x 256, bf16 partials x (V_SCALE / 256), so the stock merge kernel reduces the partials unchanged.
Only fp32 association differs.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute import experimental as ex
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.experimental import primitives as prim


@cute.experimental.jit
def load_keys(
    k: cute.Tensor,
    table: cute.Tensor,
    sk: cute.Tensor,
    barrier: cute.Pointer,
    first_page: cutlass.Int32,
    stage: cutlass.Int32,
    tile: cutlass.Constexpr[int],
    length: cutlass.Int32,
    req: cutlass.Int32,
):
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(barrier, tile * 256)
    for half in cutlass.range_constexpr(tile // 64):
        page_index = first_page + half
        page = cutlass.Int32(0)
        if page_index < cute.ceil_div(length, 64):
            page = table[req, page_index]
        source = cute.local_tile(k, (64, 256), (page, 0))
        dest = cute.local_tile(sk[None, None, stage], (64, 256), (half, 0))
        ex.tma_load(source, dest, barrier.value, update_expect_tx=False)


@cute.experimental.jit
def load_values(
    v: cute.Tensor,
    table: cute.Tensor,
    sv: cute.Tensor,
    barrier: cute.Pointer,
    first_page: cutlass.Int32,
    stage: cutlass.Int32,
    tile: cutlass.Constexpr[int],
    length: cutlass.Int32,
    req: cutlass.Int32,
):
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(barrier, tile * 256)
    for half in cutlass.range_constexpr(tile // 64):
        page_index = first_page + half
        page = cutlass.Int32(0)
        if page_index < cute.ceil_div(length, 64):
            page = table[req, page_index]
        source = cute.local_tile(v, (256, 64), (0, page))
        dest = cute.local_tile(sv[None, None, stage], (256, 64), (0, half))
        ex.tma_load(source, dest, barrier.value, update_expect_tx=False)


class Q16SplitDecode:
    """Warps 0-7: two softmax warpgroups (thread = query row; warpgroup g owns S columns g * 64.. and O half g),
    warp 8: both MMAs, warp 9: TMA loads."""

    def __init__(self, heads=16, qlen=7, splits=148, stages=2, batch=1, mode=0, trace=False):
        assert heads * qlen <= 128 and (heads * qlen) % 8 == 0 and stages in (1, 2)
        # Timing ablations only (wrong outputs): 1 no O rescale, 2 no softmax math, 4 no P V MMA, 8 no Q K MMA,
        # 16 no tail masking.
        self.mode = mode
        # Debug: CTA (0, 0) writes clock64 stamps per role and block into the trace tensor.
        self.trace = trace
        self.heads, self.qlen, self.splits, self.stages, self.batch = (
            heads,
            qlen,
            splits,
            stages,
            batch,
        )
        self.rows = heads * qlen
        self.tile = 128
        self.half = 64
        self.slots = 2

    @cute.experimental.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        table: cute.Tensor,
        lengths: cute.Tensor,
        partial: cute.Tensor,
        maxima: cute.Tensor,
        denom: cute.Tensor,
        trace: cute.Tensor,
        qk_scale: cutlass.Float32,
        out_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        dtype = cutlass.Float8E4M3FN
        # S = Q K^T: A = Q (rows x d, K-major), B = K tile (tokens x d, K-major); rows land on TMEM lanes.
        kq = cute.make_tiled_mma(
            tcgen05.MmaFP8Op(
                dtype,
                cutlass.Float32,
                (128, self.tile),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.K,
            )
        )
        # O_half += P V_half: A = P (rows x tokens, K-major), B = V half (128 d x tokens, d contiguous = MN-major).
        vp = cute.make_tiled_mma(
            tcgen05.MmaFP8Op(
                dtype,
                cutlass.Float32,
                (128, 128),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.MN,
            )
        )
        self.kernel(
            q, k, v, table, lengths, partial, maxima, denom, trace, qk_scale, out_scale, kq, vp
        ).launch(grid=(self.splits, 1, self.batch), block=(320, 1, 1), stream=stream)

    @cute.experimental.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        table: cute.Tensor,
        lengths: cute.Tensor,
        partial: cute.Tensor,
        maxima: cute.Tensor,
        denom: cute.Tensor,
        trace: cute.Tensor,
        qk_scale: cutlass.Float32,
        out_scale: cutlass.Float32,
        kq: cute.TiledMma,
        vp: cute.TiledMma,
    ):
        tid, _, _ = cute.arch.thread_idx()
        split, _, req = cute.arch.block_idx()
        traced = cutlass.Boolean(False)
        if cutlass.const_expr(self.trace):
            if split == 0:
                if req == 0:
                    traced = cutlass.Boolean(True)
            if traced:
                if tid == 0:
                    trace[7 * 16 + 0] = cute.arch.clock64()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        local_tid = tid % 128
        rows, half = self.rows, self.half
        dtype, f32 = cutlass.Float8E4M3FN, cutlass.Float32
        k_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, dtype
        )
        v_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.MN_SW128, dtype
        )
        sq = ex.allocate(
            dtype,
            cutlass.AddressSpace.smem,
            cute.tile_to_shape(k_atom, (128, 256), order=(1, 0)),
            alignment=128,
        )
        sk = ex.allocate(
            dtype,
            cutlass.AddressSpace.smem,
            cute.tile_to_shape(k_atom, (self.tile, 256, self.stages), order=(1, 0, 2)),
            alignment=128,
        )
        sv = ex.allocate(
            dtype,
            cutlass.AddressSpace.smem,
            cute.tile_to_shape(v_atom, (256, self.tile, self.stages), order=(0, 1, 2)),
            alignment=128,
        )
        sp = ex.allocate(
            dtype,
            cutlass.AddressSpace.smem,
            cute.tile_to_shape(k_atom, (128, self.tile, self.slots), order=(1, 0, 2)),
            alignment=128,
        )
        # Per-row block maxima of each warpgroup, double-buffered by S slot; denominators of warpgroup 1.
        xmax = ex.allocate(
            f32,
            cutlass.AddressSpace.smem,
            cute.make_layout((128, 2, self.slots), stride=(1, 128, 256)),
            alignment=128,
        )
        xden = ex.allocate(
            f32,
            cutlass.AddressSpace.smem,
            cute.make_layout(128),
            alignment=128,
        )
        nbar = 1 + 4 * self.stages + 4 * self.slots + 1
        barriers = ex.allocate(
            cutlass.Int64,
            cutlass.AddressSpace.smem,
            cute.make_layout(nbar),
            alignment=8,
        )
        bq = barriers.iterator
        bk = bq + 1
        bv = bk + self.stages
        bkfree = bv + self.stages
        bvfree = bkfree + self.stages
        bs = bvfree + self.stages
        bsfree = bs + self.slots
        bp = bsfree + self.slots
        bpfree = bp + self.slots
        bo = bpfree + self.slots
        if tid < nbar:
            count = cutlass.Int32(1)
            # S slot released and P slot written by every softmax thread of both warpgroups.
            if tid >= 1 + 4 * self.stages + self.slots:
                if tid < 1 + 4 * self.stages + 3 * self.slots:
                    count = cutlass.Int32(256)
            cute.arch.mbarrier_init(barriers.iterator + tid, count)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        ts = ex.allocate(
            f32,
            cutlass.AddressSpace.tmem,
            ex.make_tmem_layout_acc(kq, (128, self.tile, 256), self.slots),
            alignment=32,
        )
        to = ex.allocate(
            f32,
            cutlass.AddressSpace.tmem,
            ex.make_tmem_layout_acc(vp, (128, 128, self.tile), 2),
            alignment=32,
        )
        ld_half = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(half)), f32)
        st_half = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(half)), f32)
        s_half0 = cute.local_tile(ts[(None, None), 0, 0, 0], (128, half), (0, 0))
        hlayout = (
            tcgen05.make_tmem_copy(ld_half, s_half0)
            .get_slice(local_tid)
            .partition_D(s_half0)
            .layout
        )
        o_sub0 = cute.local_tile(to[(None, None), 0, 0, 0], (128, half), (0, 0))
        olayout = (
            tcgen05.make_tmem_copy(ld_half, o_sub0)
            .get_slice(local_tid)
            .partition_D(o_sub0)
            .layout
        )
        # Stock split rule: pages_per_split = cdiv(cdiv(length, 64), SPLITS); a split covers whole pages.
        length = lengths[req]
        pps = cute.ceil_div(cute.ceil_div(length, 64), self.splits)
        begin_tok = split * pps * 64
        end_tok = begin_tok + pps * 64
        if end_tok > length:
            end_tok = length
        nblk = cutlass.Int32(0)
        if begin_tok < end_tok:
            nblk = cute.ceil_div(end_tok - begin_tok, self.tile)
        begin_page = split * pps
        prow = (req * self.splits + split) * rows
        cute.arch.sync_threads()
        if warp < 8:
            for wg in cutlass.range_constexpr(2):
                if warp // 4 == wg:
                    row = local_tid
                    # Last token this row may see: inside the split and causal within the verify positions.
                    limit = length - self.qlen + row // self.heads
                    if limit > end_tok - 1:
                        limit = end_tok - 1
                    vals = cute.make_rmem_tensor((half, 1), f32)
                    sview = cute.make_tensor(vals.iterator, hlayout)
                    # O is handled in 64-column pieces through the same registers once P is in SMEM.
                    oview = cute.make_tensor(vals.iterator, olayout)
                    packed = cute.make_rmem_tensor((half, 1), dtype)
                    o_plane = to[(None, None), 0, 0, wg]
                    running_m = f32(-f32.inf)
                    running_l = f32(0)
                    for it in cutlass.range(0, nblk):
                        slot = it % self.slots
                        cute.arch.mbarrier_wait(bs + slot, phase=(it // self.slots) % 2)
                        if cutlass.const_expr(self.trace):
                            if traced:
                                if tid == 0:
                                    if it < 16:
                                        trace[4 * 16 + it] = cute.arch.clock64()
                        s_half = cute.local_tile(ts[(None, None), 0, 0, slot], (128, half), (0, wg))
                        copy_s = tcgen05.make_tmem_copy(ld_half, s_half).get_slice(row)
                        ex.partition_and_copy(copy_s, s_half, sview)
                        cute.arch.fence_view_async_tmem_load()
                        if cutlass.const_expr(self.trace):
                            if traced:
                                if tid == 0:
                                    if it < 16:
                                        trace[8 * 16 + it] = cute.arch.clock64()
                        cute.arch.mbarrier_arrive(bsfree + slot)
                        last = limit - (begin_tok + it * self.tile + wg * half)
                        if cutlass.const_expr(not self.mode & 16):
                            if last < half - 1:
                                # One scalar select per column; a conditional store into vals would make every
                                # branch carry the whole fragment.
                                for j in cutlass.range_constexpr(half):
                                    bias = f32(0)
                                    if j > last:
                                        bias = f32(-f32.inf)
                                    vals[j, 0] = vals[j, 0] + bias
                        # Raw block maximum in four independent chains, then the other warpgroup's half.
                        m0 = vals[0, 0]
                        m1 = vals[1, 0]
                        m2 = vals[2, 0]
                        m3 = vals[3, 0]
                        for j in cutlass.range_constexpr(1, half // 4):
                            m0 = cute.arch.fmax(m0, vals[4 * j, 0])
                            m1 = cute.arch.fmax(m1, vals[4 * j + 1, 0])
                            m2 = cute.arch.fmax(m2, vals[4 * j + 2, 0])
                            m3 = cute.arch.fmax(m3, vals[4 * j + 3, 0])
                        mine = cute.arch.fmax(cute.arch.fmax(m0, m1), cute.arch.fmax(m2, m3))
                        if cutlass.const_expr(self.trace):
                            if traced:
                                if tid == 0:
                                    if it < 16:
                                        trace[9 * 16 + it] = cute.arch.clock64()
                        xmax[row, wg, slot] = mine
                        prim.barrier_cta_sync(1, thread_count=256)
                        if cutlass.const_expr(self.trace):
                            if traced:
                                if tid == 0:
                                    if it < 16:
                                        trace[10 * 16 + it] = cute.arch.clock64()
                        other = xmax[row, 1 - wg, slot]
                        # qk_scale > 0, so scaling the raw maximum equals the maximum of the scaled scores.
                        block_max = cute.arch.fmax(mine, other) * qk_scale
                        raw = cute.arch.fmax(running_m, block_max)
                        # Stock keeps the raw running max (-inf for a row with no valid token yet) and
                        # exponentiates against 0 instead.
                        safe = raw
                        if raw == -f32.inf:
                            safe = f32(0)
                        alpha = cute.math.exp2(running_m - safe, fastmath=True)
                        t0 = f32(0)
                        t1 = f32(0)
                        t2 = f32(0)
                        t3 = f32(0)
                        if cutlass.const_expr(not self.mode & 2):
                            for j in cutlass.range_constexpr(half // 4):
                                p0 = cute.math.exp2(vals[4 * j, 0] * qk_scale - safe, fastmath=True)
                                p1 = cute.math.exp2(vals[4 * j + 1, 0] * qk_scale - safe, fastmath=True)
                                p2 = cute.math.exp2(vals[4 * j + 2, 0] * qk_scale - safe, fastmath=True)
                                p3 = cute.math.exp2(vals[4 * j + 3, 0] * qk_scale - safe, fastmath=True)
                                t0 += p0
                                t1 += p1
                                t2 += p2
                                t3 += p3
                                vals[4 * j, 0] = p0 * 256
                                vals[4 * j + 1, 0] = p1 * 256
                                vals[4 * j + 2, 0] = p2 * 256
                                vals[4 * j + 3, 0] = p3 * 256
                        running_l = running_l * alpha + ((t0 + t1) + (t2 + t3))
                        running_m = raw
                        if cutlass.const_expr(self.trace):
                            if traced:
                                if tid == 0:
                                    if it < 16:
                                        trace[11 * 16 + it] = cute.arch.clock64()
                        packed[None, 0].store(vals[None, 0].load().to(dtype))
                        if cutlass.const_expr(self.trace):
                            if traced:
                                if tid == 0:
                                    if it < 16:
                                        trace[12 * 16 + it] = cute.arch.clock64()
                        cute.arch.mbarrier_wait(bpfree + slot, phase=(it // self.slots + 1) % 2)
                        if cutlass.const_expr(self.trace):
                            if traced:
                                if tid == 0:
                                    if it < 16:
                                        trace[13 * 16 + it] = cute.arch.clock64()
                        cute.autovec_copy(
                            packed[None, 0], cute.local_tile(sp[row, None, slot], (half,), (wg,))
                        )
                        if it == 0:
                            # O accumulates across blocks; start it at zero (TMEM is not initialized).
                            for j in cutlass.range_constexpr(half):
                                vals[j, 0] = f32(0)
                            for h in cutlass.range_constexpr(128 // half):
                                o_sub = cute.local_tile(o_plane, (128, half), (0, h))
                                store_o = tcgen05.make_tmem_copy(st_half, o_sub).get_slice(row)
                                ex.partition_and_copy(store_o, oview, o_sub)
                            cute.arch.fence_view_async_tmem_store()
                        else:
                            need = cutlass.Boolean(False)
                            if row < rows:
                                if alpha != f32(1.0):
                                    need = cutlass.Boolean(True)
                            if cutlass.const_expr(self.mode & 1):
                                need = cutlass.Boolean(False)
                            if cute.arch.vote_any_sync(need):
                                # The previous block's P V must have landed before the rescale.
                                cute.arch.mbarrier_wait(bo, phase=(it - 1) % 2)
                                for h in cutlass.range_constexpr(128 // half):
                                    o_sub = cute.local_tile(o_plane, (128, half), (0, h))
                                    load_o = tcgen05.make_tmem_copy(ld_half, o_sub).get_slice(row)
                                    ex.partition_and_copy(load_o, o_sub, oview)
                                    cute.arch.fence_view_async_tmem_load()
                                    for j in cutlass.range_constexpr(half):
                                        vals[j, 0] = vals[j, 0] * alpha
                                    store_o = tcgen05.make_tmem_copy(st_half, o_sub).get_slice(row)
                                    ex.partition_and_copy(store_o, oview, o_sub)
                                cute.arch.fence_view_async_tmem_store()
                        if cutlass.const_expr(self.trace):
                            if traced:
                                if tid == 0:
                                    if it < 16:
                                        trace[14 * 16 + it] = cute.arch.clock64()
                        cute.arch.fence_view_async_shared()
                        if cutlass.const_expr(self.trace):
                            if traced:
                                if tid == 0:
                                    if it < 16:
                                        trace[5 * 16 + it] = cute.arch.clock64()
                        cute.arch.mbarrier_arrive(bp + slot)
                    if nblk > 0:
                        # bo completes once per block; take the last two completions in order.
                        if nblk > 1:
                            cute.arch.mbarrier_wait(bo, phase=(nblk - 2) % 2)
                        cute.arch.mbarrier_wait(bo, phase=(nblk - 1) % 2)
                        if cutlass.const_expr(self.trace):
                            if traced:
                                if tid == 0:
                                    trace[6 * 16 + 0] = cute.arch.clock64()
                        if cutlass.const_expr(wg == 1):
                            xden[row] = running_l
                        prim.barrier_cta_sync(2, thread_count=256)
                        out16 = cute.make_rmem_tensor((half,), cutlass.BFloat16)
                        for h in cutlass.range_constexpr(128 // half):
                            o_sub = cute.local_tile(o_plane, (128, half), (0, h))
                            load_o = tcgen05.make_tmem_copy(ld_half, o_sub).get_slice(row)
                            ex.partition_and_copy(load_o, o_sub, oview)
                            cute.arch.fence_view_async_tmem_load()
                            out16.store((vals[None, 0].load() * out_scale).to(cutlass.BFloat16))
                            if row < rows:
                                cute.autovec_copy(
                                    out16,
                                    cute.local_tile(
                                        partial[prow + row, None], (half,), (wg * 2 + h,)
                                    ),
                                )
                        if row < rows:
                            if cutlass.const_expr(wg == 0):
                                maxima[prow + row] = running_m
                                denom[prow + row] = running_l + xden[row]
                        if cutlass.const_expr(self.trace):
                            if traced:
                                if tid == 0:
                                    trace[6 * 16 + 1] = cute.arch.clock64()
        elif warp == 8:
            q_frag = kq.get_slice(0).partition_A(sq)
            atom_kq = cute.make_mma_atom(kq.op)
            atom_vp = cute.make_mma_atom(vp.op)
            atom_vp.set(tcgen05.Field.ACCUMULATE, True)
            cute.arch.mbarrier_wait(bq, phase=0)
            if cutlass.const_expr(self.trace):
                if traced:
                    if tid == 256:
                        trace[7 * 16 + 1] = cute.arch.clock64()
            for it in cutlass.range(0, nblk + 1):
                if it < nblk:
                    slot, stage = it % self.slots, it % self.stages
                    cute.arch.mbarrier_wait(
                        bsfree + slot, phase=(it // self.slots + 1) % 2
                    )
                    cute.arch.mbarrier_wait(bk + stage, phase=(it // self.stages) % 2)
                    if cutlass.const_expr(self.trace):
                        if traced:
                            if tid == 256:
                                if it < 16:
                                    trace[2 * 16 + it] = cute.arch.clock64()
                    ktile = cute.local_tile(sk[None, None, stage], (self.tile, 256), (0, 0))
                    k_frag = kq.get_slice(0).partition_B(ktile)
                    atom_kq.set(tcgen05.Field.ACCUMULATE, False)
                    for kk in cutlass.range_constexpr(
                        0 if self.mode & 8 else cute.size(k_frag, mode=[2])
                    ):
                        ex.dot(
                            atom_kq,
                            q_frag[None, None, kk],
                            k_frag[None, None, kk],
                            ts[None, None, None, slot],
                        )
                        atom_kq.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(bs + slot, None, tcgen05.CtaGroup.ONE)
                        tcgen05.commit(bkfree + stage, None, tcgen05.CtaGroup.ONE)
                if it > 0:
                    prev = it - 1
                    slot, stage = prev % self.slots, prev % self.stages
                    cute.arch.mbarrier_wait(bp + slot, phase=(prev // self.slots) % 2)
                    cute.arch.mbarrier_wait(bv + stage, phase=(prev // self.stages) % 2)
                    if cutlass.const_expr(self.trace):
                        if traced:
                            if tid == 256:
                                if prev < 16:
                                    trace[3 * 16 + prev] = cute.arch.clock64()
                    p_frag = vp.get_slice(0).partition_A(sp[None, None, slot])
                    for dm in cutlass.range_constexpr(2):
                        vtile = cute.local_tile(
                            sv[None, None, stage], (128, self.tile), (dm, 0)
                        )
                        v_frag = vp.get_slice(0).partition_B(vtile)
                        for kk in cutlass.range_constexpr(
                            0 if self.mode & 4 else cute.size(v_frag, mode=[2])
                        ):
                            ex.dot(
                                atom_vp,
                                p_frag[None, None, kk],
                                v_frag[None, None, kk],
                                to[None, None, None, dm],
                            )
                    with cute.arch.elect_one():
                        tcgen05.commit(bpfree + slot, None, tcgen05.CtaGroup.ONE)
                        tcgen05.commit(bvfree + stage, None, tcgen05.CtaGroup.ONE)
                        tcgen05.commit(bo, None, tcgen05.CtaGroup.ONE)
        elif warp == 9:
            # sq rows past this request's are never written: they only feed the padding lanes' S, P and O rows.
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bq, rows * 256)
            ex.tma_load(
                cute.local_tile(q, (rows, 256), (req, 0)),
                cute.local_tile(sq, (rows, 256), (0, 0)),
                bq.value,
                update_expect_tx=False,
            )
            for stage in cutlass.range_constexpr(self.stages):
                if stage < nblk:
                    load_keys(
                        k,
                        table,
                        sk,
                        bk + stage,
                        begin_page + stage * (self.tile // 64),
                        stage,
                        self.tile,
                        length,
                        req,
                    )
            for it in cutlass.range(0, nblk):
                stage = it % self.stages
                cute.arch.mbarrier_wait(
                    bvfree + stage, phase=(it // self.stages + 1) % 2
                )
                if cutlass.const_expr(self.trace):
                    if traced:
                        if tid == 288:
                            if it < 16:
                                trace[0 * 16 + it] = cute.arch.clock64()
                load_values(
                    v,
                    table,
                    sv,
                    bv + stage,
                    begin_page + it * (self.tile // 64),
                    stage,
                    self.tile,
                    length,
                    req,
                )
                if it + self.stages < nblk:
                    cute.arch.mbarrier_wait(
                        bkfree + stage, phase=(it // self.stages) % 2
                    )
                    if cutlass.const_expr(self.trace):
                        if traced:
                            if tid == 288:
                                if it < 16:
                                    trace[1 * 16 + it] = cute.arch.clock64()
                    load_keys(
                        k,
                        table,
                        sk,
                        bk + stage,
                        begin_page + (it + self.stages) * (self.tile // 64),
                        stage,
                        self.tile,
                        length,
                        req,
                    )


import torch
import triton
from cutlass.cute.runtime import from_dlpack

from sglang.kernels.ops.attention.split_kv_verify import _merge_attention


class Q16SplitPlan:
    """Reusable storage and a compiled producer for one fixed input layout; merge is the stock kernel."""

    def __init__(self, query, kv, table, lengths, *, splits=None, stages=2, mode=0, trace=False):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Paged split-KV storage must be prepared before graph capture"
            )
        self.batch = lengths.numel()
        self.heads = query.shape[-2]
        self.qlen = query.shape[0] // self.batch
        self.splits = splits or 148 // self.batch
        self.rows = self.qlen * self.heads
        self.partial = torch.empty(
            (self.batch, self.splits, self.rows, 256),
            dtype=torch.bfloat16,
            device=query.device,
        )
        self.maximum = torch.empty(
            (self.batch, self.splits, self.rows), dtype=torch.float32, device=query.device
        )
        self.denom = torch.empty_like(self.maximum)
        self.trace = torch.zeros(16 * 16, dtype=torch.int64, device=query.device)
        kernel = Q16SplitDecode(
            self.heads,
            self.qlen,
            self.splits,
            stages=stages,
            batch=self.batch,
            mode=mode,
            trace=trace,
        )
        views = self._views(query, kv, table, lengths)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self.compiled = cute.compile(
            kernel, *views, cutlass.Float32(1.0), cutlass.Float32(1.0), stream
        )

    def _views(self, query, kv, table, lengths):
        k, v = kv
        tensors = (
            query.view(-1, 256),
            k.view(-1, 256),
            v.view(-1, 256).T,
            table,
            lengths,
        )
        own = (self.partial.view(-1, 256), self.maximum.view(-1), self.denom.view(-1), self.trace)
        # Own buffers are 256-byte aligned allocations; 16 lets the epilogue store 16-byte vectors.
        return tuple(from_dlpack(t) for t in tensors) + tuple(
            from_dlpack(t, assumed_align=16) for t in own
        )

    def run(self, query, kv, table, lengths, *, qk_scale, v_scale, out):
        views = self._views(query, kv, table, lengths)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        # Same fp32 constants as the stock kernel's constexpr products.
        self.compiled(
            *views,
            cutlass.Float32(qk_scale * 1.4426950408889634),
            cutlass.Float32(v_scale / 256.0),
            stream,
        )
        _merge_attention[(self.rows, triton.cdiv(256, 64), self.batch)](
            self.partial,
            self.maximum,
            self.denom,
            lengths,
            out,
            self.rows,
            self.splits,
            triton.next_power_of_2(self.splits),
            64,
            128,
            False,
            True,
            True,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return out


def skv2_q16_verify(query, kv, table, lengths, plan, *, qk_scale, v_scale, out):
    """q7 split attention for the plan's B requests (16 heads per KV head); same contract as split_kv_verify."""
    return plan.run(
        query, kv, table, lengths, qk_scale=qk_scale, v_scale=v_scale, out=out
    )
