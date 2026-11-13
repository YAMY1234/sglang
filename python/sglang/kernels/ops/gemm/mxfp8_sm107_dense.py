# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
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
"""SM107 MXFP8 persistent dense GEMM with programmatic dependent launch (PDL).

``__call__`` and ``kernel_impl`` are copied from FlashInfer 0.7.3's
``gemm/kernels/dense_blockscaled_gemm_sm107.py`` (``Sm107BlockScaledPersistentDenseGemmKernel``,
itself ported from CUTLASS's Rubin ``dense_blockscaled_gemm_persistent.py``; source sha256
``SOURCE_SHA256`` below) with three changes:

* the kernel is launched with PDL (``use_pdl=True``);
* the TMA warp issues the first tile's first ``num_ab_stage`` weight stages (and their scale
  factors) before ``griddepcontrol.wait`` and the activation stages after it, so the weight
  fetch overlaps the previous kernel; optionally it also L2-prefetches the rest of that tile's
  weights before the wait;
* every CTA calls ``griddepcontrol.launch_dependents`` once its last load is issued;
* optional L2 eviction policies on the weight and activation TMA loads.

Everything else (tiles, scheduler, MMA, epilogue, numerics) is the stock kernel's, so the output
is bitwise identical to FlashInfer's ``mm_mxfp8(backend="cute-dsl")`` at the same tactic.
"""

import hashlib
import inspect
import logging
from functools import lru_cache
from typing import Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.arch import griddepcontrol_launch_dependents, griddepcontrol_wait
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils.gemm.sm100 import (
    epilogue_smem_copy_and_partition,
    epilogue_tmem_copy_and_partition,
    transform_partitioned_tensor_layout,
)
from flashinfer.gemm.kernels import dense_blockscaled_gemm_sm107 as _src
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100 import _l2_cache_policy
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm107 import (  # noqa: F401
    CollectorOp,
    OperandMajorMode,
    Optional,
    Sm107BlockScaledPersistentDenseGemmKernel,
    Type,
    Union,
    blockscaled_utils,
    cpasync,
    cuda,
    epilogue_tma_store_with_alpha,
    pipeline,
    pipeline_init_arrive,
    pipeline_init_wait,
    sm100_utils,
    sm107_utils,
    tcgen05,
    utils,
)

logger = logging.getLogger(__name__)

# sha256 of the FlashInfer source the methods below were copied from.
SOURCE_SHA256 = "d2fec3852c7a3467c0d1ca4f492bdfe0e420e700fedf487c060058beb932852f"


def source_matches() -> bool:
    """Whether the installed FlashInfer SM107 kernel is the one this copy was made from."""
    with open(inspect.getsourcefile(_src), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest() == SOURCE_SHA256


@dsl_user_op
def _fence_acq_rel_gpu(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], "fence.acq_rel.gpu;", "", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def _split_barrier(ptr: cute.Pointer, count: int, *, loc=None, ip=None) -> None:
    """Reusable gpu-scope barrier of ``count`` arrivals on one u32 word (bit 31 = phase,
    low bits = arrivals). The last arrival resets the count and flips the phase; the
    others spin until the phase they arrived in has flipped."""
    addr = ptr.toint(loc=loc, ip=ip).ir_value()
    flip = (1 << 31) - count
    llvm.inline_asm(
        None,
        [addr],
        "{\n"
        ".reg .u32 old_v, cnt_v, ph_v, cur_v, add_v;\n"
        ".reg .pred last_p, same_p;\n"
        "atom.add.acq_rel.gpu.global.u32 old_v, [$0], 1;\n"
        "and.b32 cnt_v, old_v, 2147483647;\n"
        "shr.u32 ph_v, old_v, 31;\n"
        f"setp.eq.u32 last_p, cnt_v, {count - 1};\n"
        f"mov.u32 add_v, {flip};\n"
        "@last_p red.add.release.gpu.global.u32 [$0], add_v;\n"
        "@last_p bra.uni SPLIT_DONE_${:uid};\n"
        "SPLIT_WAIT_${:uid}:\n"
        "nanosleep.u32 32;\n"
        "ld.acquire.gpu.global.u32 cur_v, [$0];\n"
        "shr.u32 cur_v, cur_v, 31;\n"
        "setp.eq.u32 same_p, cur_v, ph_v;\n"
        "@same_p bra.uni SPLIT_WAIT_${:uid};\n"
        "SPLIT_DONE_${:uid}:\n"
        "}\n",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _replace_last(stride, value):
    return (*stride[:-1], value) if isinstance(stride, tuple) else value


def _split_k_sf_layout(layout, rest_mn_stride, batch_stride):
    """``tile_atom_to_shape_SF((MN, K/S, S))`` re-strided onto the full-K scale tensor: the last
    (rest) stride of the MN mode steps over all K atoms, the batch mode over K/S atoms."""
    stride = layout.stride
    return cute.make_layout(
        layout.shape,
        stride=(
            _replace_last(stride[0], rest_mn_stride),
            stride[1],
            _replace_last(stride[2], batch_stride),
        ),
    )


# Barrier words per split-K GEMM: word (n_tile * _BAR_STRIDE + cta_m_tile) * _BAR_PAD, one word
# per 128-byte line so the spinning CTAs of different tiles do not share a line.
_BAR_STRIDE = 64
_BAR_PAD = 32


class Sm107PdlMxfp8GemmKernel(Sm107BlockScaledPersistentDenseGemmKernel):
    def __init__(
        self,
        sf_vec_size: int,
        mma_inst_shape_mnk: Tuple[int, int, int],
        mma_tiler_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        weights_are_b: bool = True,
        pre_wait_l2_prefetch: bool = False,
        w_l2_policy: Optional[str] = None,
        x_l2_policy: Optional[str] = None,
        split_k: int = 1,
        use_pdl: bool = True,
        split_barrier: bool = True,
    ):
        super().__init__(sf_vec_size, mma_inst_shape_mnk, mma_tiler_mnk, cluster_shape_mn)
        # Diagnostics only: launch without PDL / skip the split-K barrier (wrong results).
        self.use_pdl = use_pdl
        self.split_barrier = split_barrier
        # K split over the batch mode: S CTAs (pairs) share an output tile, each sums
        # K/S, and they reduce-scatter FP32 partials through a global workspace.
        self.split_k = split_k
        # The weight operand is the kernel's B unless A and B are swapped.
        self.weights_are_b = weights_are_b
        self.pre_wait_l2_prefetch = pre_wait_l2_prefetch
        # L2 eviction policy ("evict_first" / "evict_last" / None) of the weight and
        # activation TMA loads.
        self.w_l2_policy = w_l2_policy
        self.x_l2_policy = x_l2_policy

    @cute.jit
    def wrapper(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sf_m: cutlass.Int64,
        sf_n: cutlass.Int64,
        sf_k: cutlass.Int64,
        l: cutlass.Constexpr,  # noqa: E741
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        alpha_tensor: cute.Tensor,
        ws_tensor: cute.Tensor,
        bar_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        current_stream,
        swap_ab: cutlass.Constexpr = False,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Tensor entry point (the parent's wrapper + wrapper_ptrs, MXFP8 only), plus the
        split-K workspace and barrier tensors."""
        m = cute.size(mA, mode=[0])
        k = cute.size(mA, mode=[1])
        n = cute.size(mB, mode=[0])
        c_layout = utils.LayoutEnum.ROW_MAJOR
        if cutlass.const_expr(swap_ab):
            c_layout = utils.LayoutEnum.COL_MAJOR
        self(
            mA.iterator,
            mB.iterator,
            a_sf_ptr,
            b_sf_ptr,
            mC.iterator,
            alpha_tensor,
            ws_tensor,
            bar_tensor,
            (OperandMajorMode.K, OperandMajorMode.K, c_layout),
            (cutlass.Int32(m), cutlass.Int32(n), cutlass.Int32(k), l),
            max_active_clusters,
            current_stream,
            epilogue_op,
            cutlass.Int64(0),
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_bkeep: Optional[cute.TiledMma],
        tiled_mma_breuse: Optional[cute.TiledMma],
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        tCtSFA_layout: cute.Layout,
        tCtSFB_layout: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        alpha: cute.Tensor,
        mW_mnl: cute.Tensor,
        bar: cute.Tensor,
    ):
        self.kernel_impl(
            tiled_mma,
            tiled_mma_bkeep,
            tiled_mma_breuse,
            tiled_mma_sfb,
            tma_atom_a,
            mA_mkl,
            tma_atom_b,
            mB_nkl,
            tma_atom_sfa,
            mSFA_mkl,
            tma_atom_sfb,
            mSFB_nkl,
            tma_atom_c,
            mC_mnl,
            cluster_layout_vmnk,
            cluster_layout_sfb_vmnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
            tCtSFA_layout,
            tCtSFB_layout,
            c_smem_layout_staged,
            epi_tile,
            tile_sched_params,
            epilogue_op,
            self.cluster_shape_mn,
            self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1,
            self.is_a_mcast,
            self.is_b_mcast,
            alpha,
            mW_mnl,
            bar,
        )

    @cute.jit
    def _epilogue_split_k(
        self,
        epi_tidx: cutlass.Int32,
        warp_idx: cutlass.Int32,
        tma_atom_c: cute.CopyAtom,
        tCtAcc_base: cute.Tensor,
        sC: cute.Tensor,
        tCgC_base: cute.Tensor,
        mW_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        num_tiles_executed: cutlass.Int32,
        epilogue_op: cutlass.Constexpr,
        alpha_value,
        cta_tile_coord,
        mma_tile_coord_mnl,
        acc_consumer_state: pipeline.PipelineState,
        acc_pipeline: pipeline.PipelineAsync,
        c_pipeline: pipeline.PipelineTmaStore,
        bar: cute.Tensor,
    ) -> pipeline.PipelineState:
        """Split-K epilogue: reduce-scatter over the S splits of this CTA's output tile.

        Subtile c belongs to split c % S. Each split stores its FP32 partials of the
        subtiles it does not own into its slice of the workspace, meets the other splits
        at a gpu-scope barrier, then sums every owned subtile over splits 0..S-1 in that
        order (its own partial from TMEM, the others from the workspace) and writes BF16
        through the stock TMA store. The fixed order makes the result deterministic.
        """
        split_k = self.split_k
        split = cta_tile_coord[2]
        tCgC = transform_partitioned_tensor_layout(tCgC_base)
        tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)

        tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
            self, epi_tidx, tCtAcc, tCgC, epi_tile, self.use_2cta_instrs
        )
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
        tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(
            self, tiled_copy_t2r, tTR_rC, epi_tidx, sC
        )
        tTR_rPart = cute.make_rmem_tensor(tTR_rAcc.shape, self.acc_dtype)
        tTR_rSum = cute.make_rmem_tensor(tTR_rAcc.shape, self.acc_dtype)

        tCgC_epi = cute.flat_divide(tCgC, epi_tile)
        bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(tCgC_epi, 0, 2),
        )
        # The output has one batch: the split index is not a C coordinate.
        bSG_gC = bSG_gC_partitioned[
            (None, None, None, mma_tile_coord_mnl[0], mma_tile_coord_mnl[1], 0)
        ]
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

        # Workspace scratch: one block of [frag / 4][128 threads][4] FP32 per (CTA tile, split,
        # subtile), so every warp access is 512 contiguous bytes and the addressing is static.
        # A thread reads back exactly the fragment slots it (or the same thread of another
        # split) wrote, so the order of a fragment's registers does not matter.
        frag = cute.size(tTR_rAcc)
        assert frag % 4 == 0, f"split-K workspace needs 4-float fragments, got {frag}"
        ws_frag_layout = cute.make_layout((4, frag // 4), stride=(1, 4 * 128))
        reg_flat_layout = cute.make_layout((4, frag // 4), stride=(1, 4))
        acc_flat = cute.make_tensor(tTR_rAcc.iterator, reg_flat_layout)
        part_flat = cute.make_tensor(tTR_rPart.iterator, reg_flat_layout)
        ws_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.acc_dtype, num_bits_per_copy=128
        )
        # CTA rows of the padded tile grid (a partial last MMA tile still has all its CTAs).
        cta_rows = (
            (cute.size(mW_mnl, mode=[0]) + self.mma_tiler[0] - 1) // self.mma_tiler[0]
        ) * (self.mma_tiler[0] // self.cta_tile_shape_mnk[0])
        ws_group = cta_tile_coord[1] * cta_rows + cta_tile_coord[0]

        epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.epilog_sync_bar_id,
            num_threads=32 * len(self.epilogue_warp_id),
        )

        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]
        acc_pipeline.consumer_wait(acc_consumer_state)
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

        # 1. Partials of the subtiles the other splits own.
        for subtile_idx in range(subtile_cnt):
            if subtile_idx % split_k != split:
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, subtile_idx)], tTR_rAcc)
                ws_off = cute.assume(
                    ((ws_group * split_k + split) * subtile_cnt + subtile_idx) * frag * 128
                    + epi_tidx * 4,
                    4,
                )
                cute.copy(ws_atom, acc_flat, cute.make_tensor(mW_mnl.iterator + ws_off, ws_frag_layout))

        # 2. Meet the other splits of this tile.
        _fence_acq_rel_gpu()
        epilog_sync_barrier.arrive_and_wait()
        if epi_tidx == 0:
            bar_idx = (cta_tile_coord[1] * _BAR_STRIDE + cta_tile_coord[0]) * _BAR_PAD
            if cutlass.const_expr(self.split_barrier):
                _split_barrier(bar.iterator + bar_idx, split_k)
        epilog_sync_barrier.arrive_and_wait()
        _fence_acq_rel_gpu()

        # 3. Owned subtiles: sum splits 0..S-1 in order, convert, TMA store.
        num_prev_subtiles = num_tiles_executed * subtile_cnt
        for subtile_idx in range(subtile_cnt):
            if subtile_idx % split_k == split:
                for j in range(split_k):
                    if j == split:
                        cute.copy(
                            tiled_copy_t2r, tTR_tAcc[(None, None, None, subtile_idx)], tTR_rPart
                        )
                    else:
                        ws_off = cute.assume(
                            ((ws_group * split_k + j) * subtile_cnt + subtile_idx) * frag * 128
                            + epi_tidx * 4,
                            4,
                        )
                        cute.copy(
                            ws_atom,
                            cute.make_tensor(mW_mnl.iterator + ws_off, ws_frag_layout),
                            part_flat,
                        )
                    if j == 0:
                        tTR_rSum.store(tTR_rPart.load())
                    else:
                        tTR_rSum.store(tTR_rSum.load() + tTR_rPart.load())

                acc_vec = tiled_copy_r2s.retile(tTR_rSum).load()
                acc_vec = epilogue_op((acc_vec * alpha_value).to(self.c_dtype))
                tRS_rC.store(acc_vec)

                c_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
                cute.arch.fence_proxy("async.shared", space="cta")
                epilog_sync_barrier.arrive_and_wait()
                if warp_idx == self.epilogue_warp_id[0]:
                    cute.copy(
                        tma_atom_c,
                        bSG_sC[(None, c_buffer)],
                        bSG_gC[(None, subtile_idx)],
                    )
                    c_pipeline.producer_commit()
                    c_pipeline.producer_acquire()
                epilog_sync_barrier.arrive_and_wait()

        epilog_sync_barrier.arrive_and_wait()
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_consumer_state)
        acc_consumer_state.advance()
        return acc_consumer_state

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha: cute.Tensor,
        ws_tensor: cute.Tensor,
        bar_tensor: cute.Tensor,
        layouts: cutlass.Constexpr[Tuple[OperandMajorMode, OperandMajorMode, utils.LayoutEnum]],
        problem_mnkl: Tuple[int, int, int, int],
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        c_ld: cutlass.Int64 = cutlass.Int64(0),
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a_tensor: Input tensor A
        :type a_tensor: cute.Tensor
        :param b_tensor: Input tensor B
        :type b_tensor: cute.Tensor
        :param sfa_tensor: Scale factor tensor A
        :type sfa_tensor: cute.Tensor
        :param sfb_tensor: Scale factor tensor B
        :type sfb_tensor: cute.Tensor
        :param c_tensor: Output tensor C
        :type c_tensor: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """

        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a_ptr.value_type
        self.b_dtype: Type[cutlass.Numeric] = b_ptr.value_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_ptr.value_type
        self.c_dtype: Type[cutlass.Numeric] = c_ptr.value_type

        m, n, k, l = problem_mnkl
        self.a_major_mode, self.b_major_mode, self.c_layout = layouts

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        a_layout = cute.make_ordered_layout((m, cute.assume(k, 32), l), order=(0, 1, 2))
        if cutlass.const_expr(self.a_major_mode == OperandMajorMode.K):
            a_layout = cute.make_ordered_layout((cute.assume(m, 32), k, l), order=(1, 0, 2))
        b_layout = cute.make_ordered_layout((n, cute.assume(k, 32), l), order=(0, 1, 2))
        if cutlass.const_expr(self.b_major_mode == OperandMajorMode.K):
            b_layout = cute.make_ordered_layout((cute.assume(n, 32), k, l), order=(1, 0, 2))
        split_k = self.split_k
        ks = k
        if cutlass.const_expr(split_k > 1):
            # Split K over the batch mode: slice s is columns [s * K/S, (s + 1) * K/S).
            ks = cute.assume(k // split_k, 128)
            a_layout = cute.make_layout((cute.assume(m, 32), ks, split_k), stride=(k, 1, ks))
            b_layout = cute.make_layout((cute.assume(n, 32), ks, split_k), stride=(k, 1, ks))
        # c supports strided output for uGPU shared buffers.
        # c_ld: leading dimension (0 = use default contiguous layout).
        if cutlass.const_expr(self.c_layout == utils.LayoutEnum.ROW_MAJOR):
            actual_c_ld = c_ld + (n - c_ld) * (c_ld == 0)
            c_out_layout = cute.make_layout(
                (m, cute.assume(n, 32), l), stride=(actual_c_ld, 1, m * actual_c_ld)
            )
            # Each split's slice spans the workspace's padded rows, so the rows of a
            # partial last M tile never reach the next split's slice.
            w_layout = cute.make_layout(
                (m, cute.assume(n, 32), split_k),
                stride=(n, 1, cute.size(ws_tensor, mode=[0]) // split_k),
            )
        else:
            actual_c_ld = c_ld + (m - c_ld) * (c_ld == 0)
            c_out_layout = cute.make_layout(
                (cute.assume(m, 32), n, l), stride=(1, actual_c_ld, n * actual_c_ld)
            )
            w_layout = cute.make_layout(
                (cute.assume(m, 32), n, split_k), stride=(1, m, m * n)
            )
        # FP32 partials of every split, in C's orientation (split-K only).
        w_tensor = cute.make_tensor(ws_tensor.iterator, w_layout)
        a_tensor = cute.make_tensor(a_ptr, a_layout)
        b_tensor = cute.make_tensor(b_ptr, b_layout)
        c_tensor = cute.make_tensor(c_ptr, c_out_layout)

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, self.sf_vec_size)
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, self.sf_vec_size)
        if cutlass.const_expr(split_k > 1):
            # The canonical (rows, K/S, S) scale layout with the strides of the full-K
            # tensor: Rest_MN steps over all K atoms, the batch mode over K/S of them.
            rk = (k + self.sf_vec_size * 4 - 1) // (self.sf_vec_size * 4)
            rks = ks // (self.sf_vec_size * 4)
            sfa_layout = _split_k_sf_layout(sfa_layout, rk * 512, rks * 512)
            sfb_layout = _split_k_sf_layout(sfb_layout, rk * 512, rks * 512)
            sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

        atom_layout_mnk = (1, 1, 1)
        permutation_mnk = self._get_mma_permutation_mnk()

        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=atom_layout_mnk,
            permutation_mnk=permutation_mnk,
        )

        tiled_mma.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma.set(tcgen05.Field.NEGATE_B, False)

        # For 2CTA blockscaled kernels, SFB needs to be replicated across peer CTAs.
        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )

        tiled_mma_sfb.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_B, False)

        tiled_mma_bkeep = None
        tiled_mma_breuse = None
        if cutlass.const_expr(self.enable_breuse):
            # The 'FILL' collector operation indicates that B data should be kept
            # for reuse in subsequent operations.
            tiled_mma_bkeep = sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD,
                b_collector_op=CollectorOp.FILL,
                atom_layout_mnk=atom_layout_mnk,
                permutation_mnk=permutation_mnk,
            )
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_B, False)

            # The 'LASTUSE' collector operation indicates that this is the last use
            # of the B data that was kept from the previous operation.
            tiled_mma_breuse = sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD,
                b_collector_op=CollectorOp.LASTUSE,
                atom_layout_mnk=atom_layout_mnk,
                permutation_mnk=permutation_mnk,
            )

            tiled_mma_breuse.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_breuse.set(tcgen05.Field.NEGATE_B, False)

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFA
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)

            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )
            # Use right multiplication for ScaledBasis (3 * x instead of x * 3)
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1],
                tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb_new_layout = cute.make_layout(new_shape, stride=new_stride)
            tma_tensor_sfb = cute.make_tensor(tma_tensor_sfb.iterator, tma_tensor_sfb_new_layout)

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # Setup TMA store for C
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_tensor,
            epi_smem_layout,
            self.epi_tile,
        )

        # Compute grid size
        sched_c = c_tensor
        if cutlass.const_expr(split_k > 1):
            sched_c = w_tensor
        self.tile_sched_params, grid = self._compute_grid(
            sched_c,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            max_active_clusters,
            self.swizzle_size,
            self.raster_order,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_bkeep,
            tiled_mma_breuse,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.tCtSFA_layout,
            self.tCtSFB_layout,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
            alpha,
            w_tensor,
            bar_tensor,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )
        return

    @cute.jit
    def kernel_impl(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_bkeep: Optional[cute.TiledMma],
        tiled_mma_breuse: Optional[cute.TiledMma],
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        tCtSFA_layout: cute.Layout,
        tCtSFB_layout: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        cluster_shape_mn: Tuple[int, int],
        num_tma_producer: int,
        is_a_mcast: bool,
        is_b_mcast: bool,
        alpha: cute.Tensor,
        mW_mnl: cute.Tensor,
        bar: cute.Tensor,
    ):
        """
        GPU device kernel implementation performing the Persistent batched GEMM computation.
        """
        alpha_value = alpha[0].to(self.c_dtype)

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            arch=self.arch,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/SFA/SFB/C
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        #
        # Compute multicast mask for A/B/SFA/SFB buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(is_a_mcast or is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #  TMA-load SFA partition_S/D
        sfa_cta_layout = a_cta_layout
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA-load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=cluster_shape_mn)

        #
        # Construct the scheduler
        #
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        #
        # Specialized TMA load warp
        #
        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            # Optional L2 eviction policies for the weight and activation streams.
            w_cache_policy = None
            x_cache_policy = None
            if cutlass.const_expr(self.w_l2_policy is not None):
                w_cache_policy = _l2_cache_policy(self.w_l2_policy)
            if cutlass.const_expr(self.x_l2_policy is not None):
                x_cache_policy = _l2_cache_policy(self.x_l2_policy)
            a_cache_policy = x_cache_policy
            b_cache_policy = w_cache_policy
            if cutlass.const_expr(not self.weights_are_b):
                a_cache_policy = w_cache_policy
                b_cache_policy = x_cache_policy

            # PDL: the weight operand does not depend on the previous kernel, so the
            # first tile's first stages of weights (and their scale factors) are
            # issued before griddepcontrol.wait; activations follow the wait.
            # Those stages are acquired here and skipped by the acquire below.
            num_pre = cutlass.Int32(0)
            if work_tile.is_valid_tile:
                pre_tile_coord = work_tile.tile_idx
                pre_mma_coord = (
                    pre_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    pre_tile_coord[1],
                    pre_tile_coord[2],
                )
                pre_slice_n = pre_mma_coord[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    pre_slice_n = pre_mma_coord[1] // 2
                num_pre = cutlass.Int32(self.num_ab_stage)
                if k_tile_cnt < num_pre:
                    num_pre = k_tile_cnt
                pre_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage
                )
                for k_tile in cutlass.range(0, num_pre, 1, unroll=1):
                    ab_pipeline.producer_acquire(pre_state)
                    if cutlass.const_expr(self.weights_are_b):
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, pre_mma_coord[1], pre_state.count, pre_mma_coord[2])],
                            tBsB[(None, pre_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(pre_state),
                            cache_policy=b_cache_policy,
                            mcast_mask=b_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB[(None, pre_slice_n, pre_state.count, pre_mma_coord[2])],
                            tBsSFB[(None, pre_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(pre_state),
                            cache_policy=b_cache_policy,
                            mcast_mask=sfb_full_mcast_mask,
                        )
                    else:
                        cute.copy(
                            tma_atom_a,
                            tAgA[(None, pre_mma_coord[0], pre_state.count, pre_mma_coord[2])],
                            tAsA[(None, pre_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(pre_state),
                            cache_policy=a_cache_policy,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA[(None, pre_mma_coord[0], pre_state.count, pre_mma_coord[2])],
                            tAsSFA[(None, pre_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(pre_state),
                            cache_policy=a_cache_policy,
                            mcast_mask=sfa_full_mcast_mask,
                        )
                    pre_state.advance()
                if cutlass.const_expr(self.pre_wait_l2_prefetch):
                    for k_tile in cutlass.range(num_pre, k_tile_cnt, 1, unroll=1):
                        if cutlass.const_expr(self.weights_are_b):
                            cute.prefetch(
                                tma_atom_b,
                                tBgB[(None, pre_mma_coord[1], k_tile, pre_mma_coord[2])],
                            )
                        else:
                            cute.prefetch(
                                tma_atom_a,
                                tAgA[(None, pre_mma_coord[0], k_tile, pre_mma_coord[2])],
                            )
            griddepcontrol_wait()

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                # ((atom_v, rest_v), RestK)
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                # ((atom_v, rest_v), RestK)
                tAgSFA_slice = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]

                # Apply SFB slicing hack when cta_tile_shape_n=64
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2

                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    if k_tile < num_pre:
                        # Stage acquired and weights issued before the wait.
                        if cutlass.const_expr(self.weights_are_b):
                            cute.copy(
                                tma_atom_a,
                                tAgA_slice[(None, ab_producer_state.count)],
                                tAsA[(None, ab_producer_state.index)],
                                tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                                cache_policy=a_cache_policy,
                                mcast_mask=a_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_slice[(None, ab_producer_state.count)],
                                tAsSFA[(None, ab_producer_state.index)],
                                tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                                cache_policy=a_cache_policy,
                                mcast_mask=sfa_full_mcast_mask,
                            )
                        else:
                            cute.copy(
                                tma_atom_b,
                                tBgB_slice[(None, ab_producer_state.count)],
                                tBsB[(None, ab_producer_state.index)],
                                tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                                cache_policy=b_cache_policy,
                                mcast_mask=b_full_mcast_mask,
                            )
                            cute.copy(
                                tma_atom_sfb,
                                tBgSFB_slice[(None, ab_producer_state.count)],
                                tBsSFB[(None, ab_producer_state.index)],
                                tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                                cache_policy=b_cache_policy,
                                mcast_mask=sfb_full_mcast_mask,
                            )
                    else:
                        # Conditionally wait for AB buffer empty
                        ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)

                        # TMA load A/B/SFA/SFB
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, ab_producer_state.count)],
                            tAsA[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            cache_policy=a_cache_policy,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_slice[(None, ab_producer_state.count)],
                            tBsB[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            cache_policy=b_cache_policy,
                            mcast_mask=b_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_slice[(None, ab_producer_state.count)],
                            tAsSFA[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            cache_policy=a_cache_policy,
                            mcast_mask=sfa_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB_slice[(None, ab_producer_state.count)],
                            tBsSFB[(None, ab_producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                            cache_policy=b_cache_policy,
                            mcast_mask=sfb_full_mcast_mask,
                        )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                num_pre = cutlass.Int32(0)
                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Every load is issued: let the next kernel launch.
            griddepcontrol_launch_dependents()

            #
            # Wait A/B buffer empty
            #
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator/SFA/SFB tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # Make accumulator tmem tensor
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_M, MMA_K)
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            # (MMA, MMA_N, MMA_K)
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            #
            # Partition for S2T copy of SFA/SFB
            #
            sfa_s2t_bundle = self._mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            sfb_s2t_bundle = self._mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            #
            # Persistent tile scheduling loop
            #
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Get accumulator stage index
                acc_stage_index = acc_producer_state.index

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                # Apply TMEM pointer offset hack when cta_tile_shape_n=192 or 64
                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] in {64, 192}):
                    # If this is an ODD tile, shift the TMEM start address for
                    # cta_tile_shape_n=192 or 64 case by two words (ignores first 64 columns of SFB)
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                #
                # Mma mainloop
                #
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        if cutlass.const_expr(not self._is_interleaved_utccp()):
                            # Unless UTCCPs are to be interleaved, all SFA/SFB for all
                            # k_blocks are copied before MMA are executed
                            self._mainloop_s2t_copies(
                                ab_consumer_state.index, sfa_s2t_bundle, sfb_s2t_bundle
                            )

                        # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for k_block in cutlass.range(num_kblocks, unroll_full=True):
                            if cutlass.const_expr(
                                self.enable_breuse
                                and cute.size(tCtAcc.layout, mode=[1]) == 2
                                and cute.size(tCtAcc.layout, mode=[2]) == 1
                            ):
                                tCtAcc_bkeep = tCtAcc[(None, 0, 0)]
                                tCtAcc_breuse = tCtAcc[(None, 1, 0)]

                                a_kblk_crd_keep = (
                                    None,
                                    0,
                                    k_block,
                                    ab_consumer_state.index,
                                )
                                a_kblk_crd_reuse = (
                                    None,
                                    1,
                                    k_block,
                                    ab_consumer_state.index,
                                )
                                b_kblk_crd = (None, 0, k_block, ab_consumer_state.index)

                                sfa_kblk_crd_keep = (None, 0, k_block)
                                sfa_kblk_crd_reuse = (None, 1, k_block)
                                sfb_kblk_crd = (None, 0, k_block)

                                if cutlass.const_expr(self._is_interleaved_utccp()):
                                    self._mainloop_s2t_interleaved_copies(
                                        k_block,
                                        ab_consumer_state.index,
                                        sfa_s2t_bundle,
                                        sfb_s2t_bundle,
                                    )

                                # Keep
                                tiled_mma_bkeep.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or k_block != 0,
                                )
                                cute.gemm(
                                    tiled_mma_bkeep,
                                    tCtAcc_bkeep,
                                    [tCrA[a_kblk_crd_keep], tCtSFA[sfa_kblk_crd_keep]],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_bkeep,
                                )
                                # Reuse
                                tiled_mma_breuse.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or k_block != 0,
                                )
                                cute.gemm(
                                    tiled_mma_breuse,
                                    tCtAcc_breuse,
                                    [
                                        tCrA[a_kblk_crd_reuse],
                                        tCtSFA[sfa_kblk_crd_reuse],
                                    ],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_breuse,
                                )
                            else:
                                kblk_crd = (
                                    None,
                                    None,
                                    k_block,
                                    ab_consumer_state.index,
                                )
                                sf_kblk_crd = (None, None, k_block)

                                tiled_mma.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or k_block != 0,
                                )
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    [tCrA[kblk_crd], tCtSFA[sf_kblk_crd]],
                                    [tCrB[kblk_crd], tCtSFB_mma[sf_kblk_crd]],
                                    tCtAcc,
                                )

                        # Async arrive AB buffer empty
                        ab_pipeline.consumer_release(ab_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                #
                # Async arrive accumulator buffer full
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)
        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilogue_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage, producer_group=c_producer_group
            )

            #
            # Persistent tile scheduling loop
            #
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

                if cutlass.const_expr(self.split_k > 1):
                    acc_consumer_state = self._epilogue_split_k(
                        tidx,
                        warp_idx,
                        tma_atom_c,
                        tCtAcc,
                        sC,
                        tCgC,
                        mW_mnl,
                        epi_tile,
                        tile_sched.num_tiles_executed,
                        epilogue_op,
                        alpha_value,
                        cur_tile_coord,
                        mma_tile_coord_mnl,
                        acc_consumer_state,
                        acc_pipeline,
                        c_pipeline,
                        bar,
                    )
                else:
                    # flashinfer local fix (b53cfd6a7, not in TRT-LLM): apply alpha in
                    # Float32 before the c_dtype conversion to avoid fp16 overflow.
                    acc_consumer_state = epilogue_tma_store_with_alpha(
                        self,
                        tidx,
                        warp_idx,
                        tma_atom_c,
                        tCtAcc,
                        sC,
                        tCgC,
                        epi_tile,
                        tile_sched.num_tiles_executed,
                        epilogue_op,
                        alpha_value,
                        mma_tile_coord_mnl,
                        acc_consumer_state,
                        acc_pipeline,
                        c_pipeline,
                    )

            c_pipeline.producer_tail()

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            tmem.free(acc_tmem_ptr)



_KERNEL_CACHE: dict = {}
_WORKSPACE: dict = {}
# Split-K workspaces cover this many output rows per split (the decode's largest verify batch);
# a partial last M tile writes its padded rows inside its own split's slice.
SPLIT_K_MAX_M = 2048


def _compile(kernel, swap_ab, c_dtype, max_active_clusters, sf_m, sf_n, sf_k):
    """cute.compile of ``kernel.wrapper`` with symbolic M, N, K (one binary for every shape),
    following FlashInfer's _make_blockscaled_gemm_compile_fn plus the split-K tensors."""
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_ptr

    sym_m, sym_k, sym_n = cute.sym_int(), cute.sym_int(), cute.sym_int()
    a_fake = make_fake_compact_tensor(
        cutlass.Float8E4M3FN, (sym_m, sym_k), stride_order=(1, 0), assumed_align=16
    )
    b_fake = make_fake_compact_tensor(
        cutlass.Float8E4M3FN, (sym_n, sym_k), stride_order=(1, 0), assumed_align=16
    )
    if swap_ab:
        c_fake = make_fake_compact_tensor(c_dtype, (sym_n, sym_m), stride_order=(0, 1), assumed_align=16)
    else:
        c_fake = make_fake_compact_tensor(c_dtype, (sym_m, sym_n), stride_order=(1, 0), assumed_align=16)
    a_sf_ptr = make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, 16)
    b_sf_ptr = make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, 16)
    alpha_fake = make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
    ws_fake = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(),), assumed_align=16)
    bar_fake = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)
    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel.wrapper,
        a_fake,
        b_fake,
        c_fake,
        sf_m,
        sf_n,
        sf_k,
        1,
        a_sf_ptr,
        b_sf_ptr,
        alpha_fake,
        ws_fake,
        bar_fake,
        max_active_clusters,
        stream_fake,
        swap_ab,
        options="--opt-level 2 --enable-tvm-ffi",
    )


@lru_cache(maxsize=None)
def _max_active_clusters(cluster_size: int) -> int:
    from flashinfer.cute_dsl.utils import get_max_active_clusters

    return get_max_active_clusters(cluster_size)


def split_k_fits(m: int, n: int, tactic, split_k: int) -> bool:
    """Whether every (tile, split) work unit of a split-K launch gets its own resident
    cluster: the splits of a tile wait for each other, so a second round would deadlock."""
    mma_tiler_mn, cluster_shape_mn, swap_ab, mma_inst_shape_m = tactic
    if swap_ab or split_k < 2:
        return split_k == 1
    # The workspace stores have no column predicate: whole N tiles only.
    if n % mma_tiler_mn[1]:
        return False
    tiles_m = -(-m // mma_tiler_mn[0])
    tiles_n = -(-n // mma_tiler_mn[1])
    cta_m = mma_tiler_mn[0] // (2 if mma_inst_shape_m == 256 else 1)
    if -(-m // cta_m) > _BAR_STRIDE or tiles_n > _BAR_STRIDE or m > SPLIT_K_MAX_M:
        return False
    clusters = (
        -(-tiles_m // (cluster_shape_mn[0] // (2 if mma_inst_shape_m == 256 else 1)))
        * -(-tiles_n // cluster_shape_mn[1])
        * split_k
    )
    return clusters <= _max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1])


def _workspace(device, n: int, split_k: int):
    key = (device, n, split_k)
    ws = _WORKSPACE.get(key)
    if ws is None:
        ws = (
            torch.empty(split_k * SPLIT_K_MAX_M * n, dtype=torch.float32, device=device),
            torch.zeros(_BAR_STRIDE * _BAR_STRIDE * _BAR_PAD, dtype=torch.int32, device=device),
        )
        _WORKSPACE[key] = ws
    return ws


def mxfp8_sm107_gemm(
    a: torch.Tensor,
    a_sf: torch.Tensor,
    weight: torch.Tensor,
    weight_sf: torch.Tensor,
    out: torch.Tensor,
    tactic,
    split_k: int = 1,
    pre_wait_l2_prefetch: bool = False,
    w_l2_policy: Optional[str] = None,
    x_l2_policy: Optional[str] = None,
    use_pdl: bool = True,
    split_barrier: bool = True,
) -> torch.Tensor:
    """out[M, N] = a[M, K] @ weight[N, K]^T with MXFP8 operands and 128x4-swizzled UE8M0 scales
    (the layout of FlashInfer's mm_mxfp8 cute-dsl backend). ``tactic`` is an SM107 mm_mxfp8 tactic
    (mma_tiler_mn, cluster_shape_mn, swap_ab, mma_inst_shape_m). ``split_k`` > 1 needs
    ``split_k_fits`` and K divisible by 128 * split_k."""
    from flashinfer.cute_dsl.utils import torch_to_cutlass_dtype
    from flashinfer.gemm.gemm_mm_fp4_cute_dsl import _prepare_alpha_for_launch
    from flashinfer.gemm.kernels.utils import (
        _SM107_MXFP8_MMA_INST_SHAPE_K,
        _SM107_MXFP8_MMA_TILER_K,
    )

    mma_tiler_mn, cluster_shape_mn, swap_ab, mma_inst_shape_m = tactic
    m, real_k = a.shape
    n = weight.shape[0]
    if swap_ab:
        kernel_m, kernel_n = n, m
        kernel_a, kernel_b, kernel_a_sf, kernel_b_sf = weight, a, weight_sf, a_sf
    else:
        kernel_m, kernel_n = m, n
        kernel_a, kernel_b, kernel_a_sf, kernel_b_sf = a, weight, a_sf, weight_sf
    sf_m = (kernel_m + 127) // 128
    sf_n = (kernel_n + 127) // 128
    sf_k = (real_k // 32 + 3) // 4
    key = (
        a.device.index,
        tuple(mma_tiler_mn),
        tuple(cluster_shape_mn),
        swap_ab,
        mma_inst_shape_m,
        out.dtype,
        split_k,
        pre_wait_l2_prefetch,
        w_l2_policy,
        x_l2_policy,
        use_pdl,
        split_barrier,
    )
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        kernel = Sm107PdlMxfp8GemmKernel(
            32,
            (mma_inst_shape_m, mma_tiler_mn[1], _SM107_MXFP8_MMA_INST_SHAPE_K),
            (mma_tiler_mn[0], mma_tiler_mn[1], _SM107_MXFP8_MMA_TILER_K),
            tuple(cluster_shape_mn),
            weights_are_b=not swap_ab,
            pre_wait_l2_prefetch=pre_wait_l2_prefetch,
            w_l2_policy=w_l2_policy,
            x_l2_policy=x_l2_policy,
            split_k=split_k,
            use_pdl=use_pdl,
            split_barrier=split_barrier,
        )
        compiled = _compile(
            kernel,
            swap_ab,
            torch_to_cutlass_dtype(out.dtype),
            _max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]),
            sf_m,
            sf_n,
            sf_k,
        )
        _KERNEL_CACHE[key] = compiled
    if split_k > 1:
        ws, bar = _workspace(a.device, n, split_k)
    else:
        ws, bar = _workspace(a.device, 1, 1)
    launch_out = out.as_strided(out.shape, (1, out.shape[0])) if swap_ab else out
    compiled(
        kernel_a,
        kernel_b,
        launch_out,
        sf_m,
        sf_n,
        sf_k,
        kernel_a_sf.data_ptr(),
        kernel_b_sf.data_ptr(),
        _prepare_alpha_for_launch(None, a.device),
        ws,
        bar,
    )
    return out


# Decode table: (N, K) -> [(max M, tactic, split_k), ...], ascending max M; the first entry whose max M
# covers the call's M is used, and an M above the last entry keeps FlashInfer's mm_mxfp8. Entries are
# the shapes and M ranges where this PDL copy beat FlashInfer's autotuned SM107 tactic on VR200 in
# 40-layer CUDA-graph chains of real weights; the tactics are that sweep's best per M. split_k > 1 lost
# at every measured shape (its reduction's L2 round trips cost more than the shorter main loop), so
# no entry uses it.
DECODE_TABLE: dict = {
    # (q_a | kv) projection, 1792 x 5120: wins up to M 512, not at 576.
    (1792, 5120): [
        (256, ((256, 64), (2, 1), False, 256), 1),
        (512, ((256, 64), (2, 4), False, 256), 1),
    ],
    # q_b projection, 32768 x 1280: wins at every measured M (256-1152).
    (32768, 1280): [
        (256, ((256, 192), (2, 4), False, 256), 1),
        (512, ((256, 192), (2, 4), False, 256), 1),
        (576, ((256, 192), (2, 1), True, 256), 1),
        (1024, ((256, 256), (2, 1), False, 256), 1),
        (1152, ((256, 256), (4, 1), True, 256), 1),
    ],
}

_LOGGED: set = set()


@lru_cache(maxsize=None)
def _source_ok() -> bool:
    ok = source_matches()
    if not ok:
        logger.warning(
            "SM107 MXFP8 dense table: the installed FlashInfer SM107 kernel differs from the "
            "copied source (sha256 %s); keeping mm_mxfp8.",
            SOURCE_SHA256,
        )
    return ok


def decode_linear(
    q_input: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """The tabled GEMM for an MXFP8 linear, or None when (N, K, M) is not tabled."""
    entries = DECODE_TABLE.get(tuple(weight.shape))
    if not entries or out_dtype != torch.bfloat16 or q_input.dim() != 2:
        return None
    m = q_input.shape[0]
    for max_m, tactic, split_k in entries:
        if m <= max_m:
            break
    else:
        return None
    if not _source_ok():
        return None
    n = weight.shape[0]
    if split_k > 1 and not split_k_fits(m, n, tactic, split_k):
        split_k = 1
    out = torch.empty((m, n), dtype=out_dtype, device=q_input.device)
    mxfp8_sm107_gemm(q_input, x_scale, weight, weight_scale, out, tactic, split_k=split_k)
    key = (n, weight.shape[1], tactic, split_k)
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(
            "SM107 MXFP8 dense table: N=%d K=%d M=%d tactic=%s split_k=%d (PDL)",
            n,
            weight.shape[1],
            m,
            tactic,
            split_k,
        )
    return out
