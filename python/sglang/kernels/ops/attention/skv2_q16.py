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
    """Warps 0-3: softmax + O rescale + epilogue (thread = query row); warp 4: both MMAs; warp 5: TMA loads."""

    def __init__(self, heads=16, qlen=7, splits=148, stages=2, batch=1):
        assert heads * qlen <= 128 and (heads * qlen) % 8 == 0 and stages in (1, 2)
        self.heads, self.qlen, self.splits, self.stages, self.batch = (
            heads,
            qlen,
            splits,
            stages,
            batch,
        )
        self.rows = heads * qlen
        self.tile = 128
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
            q, k, v, table, lengths, partial, maxima, denom, qk_scale, out_scale, kq, vp
        ).launch(grid=(self.splits, 1, self.batch), block=(192, 1, 1), stream=stream)

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
        qk_scale: cutlass.Float32,
        out_scale: cutlass.Float32,
        kq: cute.TiledMma,
        vp: cute.TiledMma,
    ):
        tid, _, _ = cute.arch.thread_idx()
        split, _, req = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        local_tid = tid % 128
        rows = self.rows
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
            # S slot released and P slot written by every softmax thread.
            if tid >= 1 + 4 * self.stages + self.slots:
                if tid < 1 + 4 * self.stages + 3 * self.slots:
                    count = cutlass.Int32(128)
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
        ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(128)), f32)
        st_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(128)), f32)
        s_plane0 = ts[(None, None), 0, 0, 0]
        rlayout = (
            tcgen05.make_tmem_copy(ld_atom, s_plane0)
            .get_slice(local_tid)
            .partition_D(s_plane0)
            .layout
        )
        o_plane0 = to[(None, None), 0, 0, 0]
        olayout = (
            tcgen05.make_tmem_copy(ld_atom, o_plane0)
            .get_slice(local_tid)
            .partition_D(o_plane0)
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
        if warp < 4:
            row = local_tid
            # Last token this row may see: inside the split and causal within the 7 verify positions.
            limit = length - self.qlen + row // self.heads
            if limit > end_tok - 1:
                limit = end_tok - 1
            vals = cute.make_rmem_tensor((128, 1), f32)
            sview = cute.make_tensor(vals.iterator, rlayout)
            oview = cute.make_tensor(vals.iterator, olayout)
            packed = cute.make_rmem_tensor((128, 1), dtype)
            running_m = f32(-f32.inf)
            running_l = f32(0)
            for it in cutlass.range(0, nblk):
                slot = it % self.slots
                cute.arch.mbarrier_wait(bs + slot, phase=(it // self.slots) % 2)
                current_s = ts[(None, None), 0, 0, slot]
                copy_s = tcgen05.make_tmem_copy(ld_atom, current_s).get_slice(local_tid)
                ex.partition_and_copy(copy_s, current_s, sview)
                cute.arch.fence_view_async_tmem_load()
                cute.arch.mbarrier_arrive(bsfree + slot)
                last = limit - (begin_tok + it * self.tile)
                block_max = f32(-f32.inf)
                for j in cutlass.range_constexpr(self.tile):
                    score = vals[j, 0] * qk_scale
                    if j > last:
                        score = -f32.inf
                    vals[j, 0] = score
                    block_max = cute.arch.fmax(block_max, score)
                raw = cute.arch.fmax(running_m, block_max)
                # Stock keeps the raw running max (-inf for a row with no valid token yet) and
                # exponentiates against 0 instead.
                safe = raw
                if raw == -f32.inf:
                    safe = f32(0)
                alpha = cute.math.exp2(running_m - safe, fastmath=True)
                total = f32(0)
                for j in cutlass.range_constexpr(self.tile):
                    probability = cute.math.exp2(vals[j, 0] - safe, fastmath=True)
                    total += probability
                    vals[j, 0] = probability * 256
                running_l = running_l * alpha + total
                running_m = raw
                packed[None, 0].store(vals[None, 0].load().to(dtype))
                cute.arch.mbarrier_wait(
                    bpfree + slot, phase=(it // self.slots + 1) % 2
                )
                cute.autovec_copy(packed[None, 0], sp[local_tid, None, slot])
                if it == 0:
                    # O accumulates across blocks; start it at zero (TMEM is not initialized).
                    for j in cutlass.range_constexpr(128):
                        vals[j, 0] = f32(0)
                    for dm in cutlass.range_constexpr(2):
                        o_plane = to[(None, None), 0, 0, dm]
                        store_o = tcgen05.make_tmem_copy(st_atom, o_plane).get_slice(
                            local_tid
                        )
                        ex.partition_and_copy(store_o, oview, o_plane)
                    cute.arch.fence_view_async_tmem_store()
                else:
                    need = cutlass.Boolean(False)
                    if row < rows:
                        if alpha != f32(1.0):
                            need = cutlass.Boolean(True)
                    if cute.arch.vote_any_sync(need):
                        # The previous block's P V must have landed before the rescale.
                        cute.arch.mbarrier_wait(bo, phase=(it - 1) % 2)
                        for dm in cutlass.range_constexpr(2):
                            o_plane = to[(None, None), 0, 0, dm]
                            load_o = tcgen05.make_tmem_copy(ld_atom, o_plane).get_slice(
                                local_tid
                            )
                            ex.partition_and_copy(load_o, o_plane, oview)
                            cute.arch.fence_view_async_tmem_load()
                            for j in cutlass.range_constexpr(128):
                                vals[j, 0] = vals[j, 0] * alpha
                            store_o = tcgen05.make_tmem_copy(st_atom, o_plane).get_slice(
                                local_tid
                            )
                            ex.partition_and_copy(store_o, oview, o_plane)
                        cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(bp + slot)
            if nblk > 0:
                # bo completes once per block; take the last two completions in order.
                if nblk > 1:
                    cute.arch.mbarrier_wait(bo, phase=(nblk - 2) % 2)
                cute.arch.mbarrier_wait(bo, phase=(nblk - 1) % 2)
                out16 = cute.make_rmem_tensor((128,), cutlass.BFloat16)
                for dm in cutlass.range_constexpr(2):
                    o_plane = to[(None, None), 0, 0, dm]
                    load_o = tcgen05.make_tmem_copy(ld_atom, o_plane).get_slice(local_tid)
                    ex.partition_and_copy(load_o, o_plane, oview)
                    cute.arch.fence_view_async_tmem_load()
                    out16.store((vals[None, 0].load() * out_scale).to(cutlass.BFloat16))
                    if row < rows:
                        dst = cute.local_tile(partial[prow + row, None], (128,), (dm,))
                        cute.autovec_copy(out16, dst)
                if row < rows:
                    maxima[prow + row] = running_m
                    denom[prow + row] = running_l
        elif warp == 4:
            q_frag = kq.get_slice(0).partition_A(sq)
            atom_kq = cute.make_mma_atom(kq.op)
            atom_vp = cute.make_mma_atom(vp.op)
            atom_vp.set(tcgen05.Field.ACCUMULATE, True)
            cute.arch.mbarrier_wait(bq, phase=0)
            for it in cutlass.range(0, nblk + 1):
                if it < nblk:
                    slot, stage = it % self.slots, it % self.stages
                    cute.arch.mbarrier_wait(
                        bsfree + slot, phase=(it // self.slots + 1) % 2
                    )
                    cute.arch.mbarrier_wait(bk + stage, phase=(it // self.stages) % 2)
                    ktile = cute.local_tile(sk[None, None, stage], (self.tile, 256), (0, 0))
                    k_frag = kq.get_slice(0).partition_B(ktile)
                    atom_kq.set(tcgen05.Field.ACCUMULATE, False)
                    for kk in cutlass.range_constexpr(cute.size(k_frag, mode=[2])):
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
                    p_frag = vp.get_slice(0).partition_A(sp[None, None, slot])
                    for dm in cutlass.range_constexpr(2):
                        vtile = cute.local_tile(
                            sv[None, None, stage], (128, self.tile), (dm, 0)
                        )
                        v_frag = vp.get_slice(0).partition_B(vtile)
                        for kk in cutlass.range_constexpr(cute.size(v_frag, mode=[2])):
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
        elif warp == 5:
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

    def __init__(self, query, kv, table, lengths, *, splits=None, stages=2):
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
        kernel = Q16SplitDecode(
            self.heads, self.qlen, self.splits, stages=stages, batch=self.batch
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
            self.partial.view(-1, 256),
            self.maximum.view(-1),
            self.denom.view(-1),
        )
        return tuple(from_dlpack(t) for t in tensors)

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
