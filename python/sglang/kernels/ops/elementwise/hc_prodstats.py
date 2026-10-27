"""M4 GDN producer statistics feeding the native split16 HC down/up pair."""

import torch
import triton
import triton.language as tl
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute import experimental as cute_ext

from sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue import (
    SplitKDenseGemmKernel, SplitKTactic, _from_dlpack_dynamic, _to_cute_swap,
)

_AB_BUFFER_ALIGN_BYTES = 1024
_MBARRIER_BYTES = 8
_TMEM_POINTER_BYTES = 4
_MAILBOX_ALIGN_BYTES = 128
_GATE_TILE_ALIGN_BYTES = 16


@triton.jit
def _sum_three_combine_stats(Partial, Output, Residual, Gates, Combined, Stats):
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    tile = tl.program_id(0)
    row = tile // 10
    column = tile % 10 * 256 + tl.arange(0, 256)
    index = row * 2560 + column
    x0 = tl.load(Partial + index)
    x1 = tl.load(Partial + 10240 + index)
    x2 = tl.load(Partial + 20480 + index)
    block = ((x0 + x1) + x2).to(tl.bfloat16)
    tl.store(Output + index, block)
    for branch in tl.static_range(4):
        total = tl.full((), 0.0, tl.float32)
        for split in tl.static_range(8):
            total += tl.load(Gates + (row * 8 + split) * 4 + branch)
        gate = 2.0 / (1.0 + tl.exp(-total * 0.25))
        offset = row * 10240 + branch * 2560 + column
        residual = tl.load(Residual + offset).to(tl.float32)
        combined = tl.fma(gate, block.to(tl.float32), residual).to(tl.bfloat16)
        tl.store(Combined + offset, combined)
        value = combined.to(tl.float32)
        tl.store(Stats + (row * 4 + branch) * 10 + tile % 10,
                 tl.sum(value * value, 0))


def combine_stats(partial, output, residual, gates):
    combined = torch.empty((4, 10240), dtype=torch.bfloat16, device=partial.device)
    stats = torch.empty((4, 4, 10), dtype=torch.float32, device=partial.device)
    _sum_three_combine_stats[(40,)](
        partial, output, residual, gates, combined, stats,
        num_warps=4, enable_fp_fusion=False, launch_pdl=True,
    )
    return combined, stats


class HcProdStatsDown(SplitKDenseGemmKernel):
    def __init__(self, eps):
        super().__init__(tactic=SplitKTactic(64, 8, 16, 6), use_pdl=True,
                         has_bias=False, epilogue_mode="silu", epilogue_scale=0.25)
        self.eps = eps

    @cute.experimental.jit
    def __call__(self, weights: cute.Tensor, combined: cute.Tensor,
                 out: cute.Tensor, stats: cute.Tensor, norm_weight: cute.Tensor,
                 normalized: cute.Tensor, stream: cuda.CUstream):
        self.kernel(weights, combined, out, out, out, out,
                    stats, norm_weight, normalized).launch(
            grid=(5, 16, 1), block=(256, 1, 1), cluster=(1, 16, 1),
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream, use_pdl=True,
        )

    @cute.experimental.jit
    def normalize_fill(self, combined, stats, weight, normalized, sb, inv,
                       rank, output_tile, norm_ready):
        tid = cute.arch.thread_idx()[0] - 96
        warp = tid // 32
        lane = tid % 32
        branch = rank // 4
        base = rank * 640
        # Four warp leaders merge the ten producer statistics in fixed order.
        if warp < 4 and lane == 0:
            total = stats[warp, branch, 0]
            for part in cutlass.range_constexpr(1, 10):
                total = total + stats[warp, branch, part]
            inv[warp] = cmath.rsqrt(total * (1.0 / 2560.0) + self.eps, fastmath=True)
        cute.arch.barrier(barrier_id=6, number_of_threads=160)
        for j in cutlass.range_constexpr(4):
            k = tid * 4 + j
            w = weight[base + k].to(cutlass.Float32)
            for row in cutlass.range_constexpr(4):
                x = combined[row, base + k, 0].to(cutlass.Float32)
                value = (x * inv[row] * (1.0 + w)).to(cutlass.BFloat16)
                sb[(row, k % 16), 0, ((k % 128) // 16 % 4, (k % 128) // 64), k // 128] = value
                if output_tile == 0:
                    normalized[row, base + k] = value
            for row in cutlass.range_constexpr(4, 8):
                sb[(row, k % 16), 0, ((k % 128) // 16 % 4, (k % 128) // 64), k // 128] = cutlass.BFloat16(0.0)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier(barrier_id=6, number_of_threads=160)
        if tid == 0:
            cute.arch.mbarrier_arrive(norm_ready)

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major
        mB: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major
        mC: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major
        mBias: cute.Tensor,  # Broadcast bias; dead when has_bias=False
        mX: cute.Tensor,  # Gate activation; dead unless epilogue_mode="gate"
        mOut: cute.Tensor,
        stats: cute.Tensor,
        norm_weight: cute.Tensor,
        normalized: cute.Tensor,
    ):
        """Allocate storage and dispatch the specialized warps."""
        stages = self.num_ab_stage

        ab_dtype = mA.element_type
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        sA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )

        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, self.mma_tiler_mn, acc_stage=1
        )
        c_tiler_mn = (self.cta_m, self.cta_n)

        bar_full = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_empty = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_tma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_mma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_tmem_alloc = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        tmem_base_ptr = cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_TMEM_POINTER_BYTES,
        ).iterator

        if cutlass.const_expr(self.split_k > 1):
            mailbox = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(self.mailbox_elements),
                alignment=_MAILBOX_ALIGN_BYTES,
            )
            bar_reduce = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_MBARRIER_BYTES,
            ).iterator
        else:
            # Dummy operands for the compile-time-elided reduction.
            mailbox = sA
            bar_reduce = bar_mma_epilog

        if cutlass.const_expr(self.epilogue_mode == "gate"):
            gate_tile = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout((self.cta_m, self.cta_n)),
                alignment=_GATE_TILE_ALIGN_BYTES,
            )
        else:
            gate_tile = mailbox

        inputs_ready = cute_ext.allocate(cutlass.Int64, cute.AddressSpace.smem,
            cute.make_layout(1), alignment=8).iterator
        norm_ready = cute_ext.allocate(cutlass.Int64, cute.AddressSpace.smem,
            cute.make_layout(1), alignment=8).iterator
        inv = cute_ext.allocate(cutlass.Float32, cute.AddressSpace.smem,
            cute.make_layout(4), alignment=16)

        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(inputs_ready, 1)
                cute.arch.mbarrier_init(norm_ready, 1)
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)

                if cutlass.const_expr(self.split_k > 1):
                    # Owner arrival plus peer transaction-byte credits.
                    cute.arch.mbarrier_init(bar_reduce, 1)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            # Publish peer barriers before cross-CTA stores.
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        # Host validation guarantees an equal, tail-free K partition.
        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count

        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        # Keep native weight prefetch and cluster MMA. The B warp publishes
        # five normalized activation stages prepared by the otherwise idle
        # warp plus the epilogue warps.
        if warp_idx == 0:
            self.dma_warp(
                bar_full, bar_empty, bar_tma_epilog,
                cute.local_tile(mA, (self.cta_m, self.cta_k), (bidx, None, l_idx)),
                sA, cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                k_tile_start, k_tile_count, True)
        elif warp_idx == 1:
            cute.arch.griddepcontrol_wait()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(inputs_ready)
            cute.arch.mbarrier_wait(norm_ready, 0)
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(5):
                    cute.arch.mbarrier_arrive(bar_full + stage)
        elif warp_idx == 2:
            self.mma_warp(
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2]),
                k_tile_count,
            )
        elif warp_idx >= 3:
            cute.arch.mbarrier_wait(inputs_ready, 0)
            self.normalize_fill(mB, stats, norm_weight, normalized, sB, inv,
                split_rank, bidx, norm_ready)
        if warp_idx >= 4:
            self.epilog_warp(
                bar_tma_epilog,
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                cute.local_tile(mC, c_tiler_mn, (bidx, n_idx, l_idx)),
                cute.local_tile(mBias, c_tiler_mn, (bidx, n_idx, l_idx)),
                cute.arch.thread_idx()[0] - 128,
                mC.element_type,
                utils.LayoutEnum.from_tensor(mC),
                mailbox,
                bar_reduce,
                split_rank,
                gate_tile,
                mX,
                mOut,
                bidx,
                n_idx,
            )


@cute.experimental.jit
def _bmm_prodstats(op: cutlass.Constexpr, a: cute.Tensor, b: cute.Tensor,
                   c: cute.Tensor, stats: cute.Tensor, weight: cute.Tensor,
                   normalized: cute.Tensor, stream: cuda.CUstream):
    op(cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
       cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
       cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0])),
       stats, weight, normalized, stream)


_down_cache = {}


def run_prodstats_down(combined, w_down, out, normalized, norm_stats):
    stats, weight, eps = norm_stats
    args = _to_cute_swap(combined, w_down.T, out, None)
    tensors = (*args[:3], _from_dlpack_dynamic(stats, 2),
               _from_dlpack_dynamic(weight.detach(), 0),
               _from_dlpack_dynamic(normalized, 1))
    stream = cuda.CUstream(torch.cuda.current_stream(combined.device).cuda_stream)
    key = (combined.device, eps)
    compiled = _down_cache.get(key)
    if compiled is None:
        compiled = cute_ext.compile(_bmm_prodstats, HcProdStatsDown(eps), *tensors, stream)
        _down_cache[key] = compiled
    compiled(*tensors, stream)
