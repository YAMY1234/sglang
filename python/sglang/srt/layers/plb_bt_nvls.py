# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from flashinfer/comm/mnnvl_cutedsl/kernel_bt/device_kernels.py
# (_SharedOnlyPublishDeviceKernel, _VectorFinalizeUnicastDeviceKernel,
# _OwnerReduceMulticastDeviceKernel, _MaterializeRMSNormDeviceKernel; FlashInfer 0.7.1
# as shipped in the sglang-nightly-dev-cu13-20261015 image) for SGLang's PERF-LB NVLS
# route.

"""The BT route's reduction on NVLS multimem.ld_reduce (SGLANG_PLB_BT_NVLS).

A call of stage s keeps BT's three launches and its stage rotation. The publish kernel
writes this rank's partial rows (the all-reduce input, or the MoE finalize) into its own
copy of the prenorm mailbox's previous generation X = (s + 2) % 3. After its grid
dependency (the local publish) the reduce kernel's first CTA tells every rank so through
one word of that rank's contribution mailbox [s][this rank][0]; each CTA waits for the
words of all ranks, sums the ranks' rows through the prenorm mailbox's multicast mapping
(multimem.ld_reduce), adds the residual to that BF16 sum and multicasts the BF16 prenorm
to [s] as FlashInfer's OwnerReduceMulticast does. The norm kernel is FlashInfer's; it also
returns the flag words and this rank's rows of X to the -0 sentinel. No buffer or
multicast object is allocated; every slot a call touches is back at the sentinel when its
norm kernel completes.
"""

from __future__ import annotations

import logging
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op
from flashinfer.comm.mnnvl_cutedsl.cute_dsl_primitives import (
    NEGATIVE_ZERO_BF16_PAIR,
    VEC_BF16,
    WARP_SIZE,
    bf16x8_to_packed_u32x4,
    fragment_has_negative_zero,
    load_global_bf16_as_f32,
    load_global_u32x4,
    load_global_u32x4_address,
    load_volatile_u32,
    packed_u32x4_to_bf16x8,
    sanitize_negative_zero_u32x4,
    stmc_bf16x8,
    store_global_u32,
    store_global_u32_address,
    store_global_u32x4,
    store_lamport_sentinel_u32x4,
)
from flashinfer.comm.mnnvl_cutedsl.kernel_bt.device_kernels import (
    ACTIVE_STAGE,
    LAMPORT_GENERATIONS,
    NEXT_STAGE,
    _block_sum,
)

logger = logging.getLogger(__name__)

# 16-byte vectors a thread: FlashInfer's BT all-reduce publish (128 threads, 1 vector)
# and VectorFinalize (128 threads, 8 elements); the reduce kernel uses the same split.
PUBLISH_THREADS = 128
REDUCE_THREADS = 128


def _inline(result_type, operands, text, constraints, *, loc, ip):
    return llvm.inline_asm(
        result_type,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_reduce_bf16x8(
    address: Int64, *, acc_f32: cutlass.Constexpr[bool] = True, loc=None, ip=None
):
    """The sum over ranks of eight BF16s at a multicast address, as BF16; .acc::f32 asks
    the switch for FP32 intermediate accumulation."""
    precision = ".acc::f32" if acc_f32 else ""
    loaded = _inline(
        llvm.StructType.get_literal([T.i32()] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        f"multimem.ld_reduce.relaxed.sys.global.add{precision}.v4.bf16x2 "
        "{$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [
            llvm.extractvalue(T.i32(), loaded, [index], loc=loc, ip=ip)
            for index in range(4)
        ],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@dsl_user_op
def _store_release_u32(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    _inline(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "st.release.sys.global.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _store_relaxed_u32(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    _inline(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "st.relaxed.sys.global.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _load_acquire_u32(
    address: Int64, *, relaxed: cutlass.Constexpr[bool] = False, loc=None, ip=None
) -> Uint32:
    sem = "relaxed" if relaxed else "acquire"
    return Uint32(
        _inline(
            T.i32(),
            [address.ir_value(loc=loc, ip=ip)],
            f"ld.{sem}.sys.global.u32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _load_global_s32(address: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _inline(
            T.i32(),
            [address.ir_value(loc=loc, ip=ip)],
            "ld.global.s32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


def _flag_offset(stage, source_rank, tp_size: int, local_capacity: int, hidden: int):
    """Byte offset of the flag word of ``source_rank`` in a contribution mailbox: the
    first word of row [stage][source_rank][0], which an NVLS call does not otherwise use."""
    return (
        (Int64(stage) * tp_size + Int64(source_rank)) * local_capacity * hidden
    ) * 2


class PlbNvlsPublishDeviceKernel:
    """FlashInfer's BT publish (``finalize``: VectorFinalize's arithmetic) writing this
    rank's rows into its own prenorm mailbox generation (s + 2) % 3."""

    def __init__(
        self,
        *,
        hidden_size: int,
        capacity_m: int,
        enable_pdl: bool,
        early_loads: bool,
        finalize: bool,
        top_k: int = 1,
        routed_scaling_factor: float = 1.0,
        include_shared_expert: bool = False,
        early_trigger: bool = False,
    ) -> None:
        self.hidden_size = hidden_size
        self.capacity_m = capacity_m
        self.enable_pdl = enable_pdl
        self.early_loads = early_loads
        self.early_trigger = early_trigger
        self.finalize = finalize
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.include_shared_expert = include_shared_expert
        self.threads = PUBLISH_THREADS
        self.fragments = hidden_size // VEC_BF16
        self.ctas_per_token = math.ceil(self.fragments / self.threads)
        self.metadata = top_k if finalize else 1

    @cute.jit
    def __call__(
        self,
        rows: Int64,
        expert_weights: Int64,
        permuted_indices: Int64,
        shared_output: Int64,
        stage_state: cute.Tensor,
        prenorm_mailbox: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            rows, expert_weights, permuted_indices, shared_output, stage_state, prenorm_mailbox
        ).launch(
            grid=(m * self.ctas_per_token, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.metadata * 8,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.jit
    def _stage_metadata(self, expert_weights, permuted_indices, indices, weights, token):
        tidx, _, _ = cute.arch.thread_idx()
        item = Int32(tidx)
        while item < Int32(self.top_k):
            offset = Int64(token) * self.top_k + Int64(item)
            routed_index = _load_global_s32(permuted_indices + offset * 4)
            weight = load_global_bf16_as_f32(expert_weights + offset * 2)
            if cutlass.const_expr(self.routed_scaling_factor != 1.0):
                weight = weight * Float32(self.routed_scaling_factor)
            if routed_index == Int32(-1):
                # Vector loads are unpredicated; row zero is safe and its weight is zero.
                routed_index = Int32(0)
                weight = Float32(0.0)
            cute.arch.store((indices + item).llvm_ptr, routed_index)
            cute.arch.store((weights + item).llvm_ptr, weight)
            item = item + self.threads
        cute.arch.barrier()

    @cute.kernel
    def kernel(
        self,
        rows: Int64,
        expert_weights: Int64,
        permuted_indices: Int64,
        shared_output: Int64,
        stage_state: cute.Tensor,
        prenorm_mailbox: Int64,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        token = block // self.ctas_per_token
        fragment = (block % self.ctas_per_token) * self.threads + tidx
        row_offset = Int64(token) * self.hidden_size + Int64(fragment) * VEC_BF16
        if cutlass.const_expr(self.enable_pdl and self.early_trigger):
            # The reduce kernel reads this grid's rows and ACTIVE_STAGE only after its wait.
            cute.arch.griddepcontrol_launch_dependents()

        smem = cutlass.utils.SmemAllocator()
        indices = smem.allocate_array(Int32, self.metadata)
        weights = smem.allocate_array(Float32, self.metadata)
        shared_values = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), BFloat16)
        shared_values.fill(BFloat16(0.0))

        # Under speculative decoding the producer of these rows may be the kernel right
        # before this one (alternating draft / verify graphs): read them after the wait.
        if cutlass.const_expr(self.finalize and self.early_loads):
            self._stage_metadata(expert_weights, permuted_indices, indices, weights, token)
            if cutlass.const_expr(self.include_shared_expert):
                if fragment < self.fragments:
                    shared_values.store(
                        packed_u32x4_to_bf16x8(
                            load_global_u32x4_address(shared_output + row_offset * 2)
                        )
                    )
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.finalize and not self.early_loads):
            self._stage_metadata(expert_weights, permuted_indices, indices, weights, token)
            if cutlass.const_expr(self.include_shared_expert):
                if fragment < self.fragments:
                    shared_values.store(
                        packed_u32x4_to_bf16x8(
                            load_global_u32x4_address(shared_output + row_offset * 2)
                        )
                    )

        stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)
        data_stage = (stage + Uint32(LAMPORT_GENERATIONS - 1)) % Uint32(LAMPORT_GENERATIONS)
        if fragment < self.fragments:
            packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
            if cutlass.const_expr(self.finalize):
                accumulator = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
                accumulator.fill(Float32(0.0))
                for k in cutlass.range_constexpr(self.top_k):
                    routed_index = cute.arch.load((indices + k).llvm_ptr, Int32)
                    weight = cute.arch.load((weights + k).llvm_ptr, Float32)
                    source_values = packed_u32x4_to_bf16x8(
                        load_global_u32x4_address(
                            rows
                            + (
                                Int64(routed_index) * self.hidden_size
                                + Int64(fragment) * VEC_BF16
                            )
                            * 2
                        )
                    ).to(Float32)
                    if weight != Float32(0.0):
                        accumulator.store(accumulator.load() + source_values * weight)
                result = accumulator.load()
                if cutlass.const_expr(self.include_shared_expert):
                    result = result + shared_values.load().to(Float32)
                packed.store(bf16x8_to_packed_u32x4(result.to(BFloat16)))
            else:
                packed.store(load_global_u32x4_address(rows + row_offset * 2))
            store_global_u32x4(
                prenorm_mailbox
                + (Int64(data_stage) * self.capacity_m * self.hidden_size + row_offset) * 2,
                packed.load(),
            )

        if block == 0 and tidx == 0:
            store_global_u32(stage_state.iterator + ACTIVE_STAGE, stage)
        if cutlass.const_expr(self.enable_pdl and not self.early_trigger):
            cute.arch.barrier()
            cute.arch.griddepcontrol_launch_dependents()


class PlbNvlsReduceDeviceKernel:
    """FlashInfer's OwnerReduceMulticast with the ranks' sum read through the multicast
    mapping, after a flag word from every rank."""

    def __init__(
        self,
        *,
        hidden_size: int,
        tp_size: int,
        rank: int,
        capacity_m: int,
        local_capacity: int,
        add_residual: bool,
        enable_pdl: bool,
        early_loads: bool,
        acc_f32: bool = True,
        relaxed_flags: bool = False,
        # Timing experiments only: relaxed flag ops; no cross-rank wait at all (outputs not valid).
        skip_flags: bool = False,
    ) -> None:
        self.relaxed_flags = relaxed_flags
        self.skip_flags = skip_flags
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.rank = rank
        self.capacity_m = capacity_m
        self.local_capacity = local_capacity
        self.add_residual = add_residual
        self.enable_pdl = enable_pdl
        self.early_loads = early_loads
        self.acc_f32 = acc_f32
        self.threads = REDUCE_THREADS
        self.fragments = hidden_size // VEC_BF16
        self.ctas_per_token = math.ceil(self.fragments / self.threads)

    @cute.jit
    def __call__(
        self,
        residual_source: Int64,
        stage_state: cute.Tensor,
        prenorm_multicast: Int64,
        contribution_mailbox: Int64,
        contribution_peers: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        local_tokens = (m + Int32(self.tp_size - 1)) // Int32(self.tp_size)
        self.kernel(
            residual_source,
            stage_state,
            prenorm_multicast,
            contribution_mailbox,
            contribution_peers,
            m,
        ).launch(
            grid=(local_tokens * self.ctas_per_token, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        residual_source: Int64,
        stage_state: cute.Tensor,
        prenorm_multicast: Int64,
        contribution_mailbox: Int64,
        contribution_peers: cute.Tensor,
        m: Int32,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        local_token = block // self.ctas_per_token
        fragment = (block % self.ctas_per_token) * self.threads + tidx
        token = local_token * self.tp_size + self.rank
        active = token < m and fragment < self.fragments
        row_offset = Int64(token) * self.hidden_size + Int64(fragment) * VEC_BF16

        residual = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), BFloat16)
        residual.fill(BFloat16(0.0))
        if cutlass.const_expr(self.add_residual and self.early_loads):
            if active:
                residual.store(
                    packed_u32x4_to_bf16x8(
                        load_global_u32x4_address(residual_source + row_offset * 2)
                    )
                )
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.add_residual and not self.early_loads):
            if active:
                residual.store(
                    packed_u32x4_to_bf16x8(
                        load_global_u32x4_address(residual_source + row_offset * 2)
                    )
                )

        # The publish grid, and so this rank's rows, completed before the wait returned.
        stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
        data_stage = (stage + Uint32(LAMPORT_GENERATIONS - 1)) % Uint32(LAMPORT_GENERATIONS)
        if cutlass.const_expr(not self.skip_flags):
            if block == 0 and tidx < self.tp_size:
                peer = cute.arch.load((contribution_peers.iterator + tidx).llvm_ptr, Int64)
                flag_out = peer + _flag_offset(
                    stage, self.rank, self.tp_size, self.local_capacity, self.hidden_size
                )
                if cutlass.const_expr(self.relaxed_flags):
                    _store_relaxed_u32(flag_out, Uint32(0))
                else:
                    _store_release_u32(flag_out, Uint32(0))
            # Every CTA waits for every rank, block 0 included even without a token: the
            # norm kernel resets the words once this grid completed.
            if tidx < self.tp_size:
                flag = contribution_mailbox + _flag_offset(
                    stage, tidx, self.tp_size, self.local_capacity, self.hidden_size
                )
                word = _load_acquire_u32(flag, relaxed=self.relaxed_flags)
                while word == Uint32(NEGATIVE_ZERO_BF16_PAIR):
                    word = _load_acquire_u32(flag, relaxed=self.relaxed_flags)
        cute.arch.barrier()

        if active:
            total = packed_u32x4_to_bf16x8(
                _ld_reduce_bf16x8(
                    prenorm_multicast
                    + (Int64(data_stage) * self.capacity_m * self.hidden_size + row_offset)
                    * 2,
                    acc_f32=self.acc_f32,
                )
            ).to(Float32)
            if cutlass.const_expr(self.add_residual):
                total = total + residual.load().to(Float32)
            stmc_bf16x8(
                prenorm_multicast
                + (Int64(stage) * self.capacity_m * self.hidden_size + row_offset) * 2,
                sanitize_negative_zero_u32x4(bf16x8_to_packed_u32x4(total.to(BFloat16))),
            )

        if block == 0 and tidx == 0:
            store_global_u32(
                stage_state.iterator + NEXT_STAGE,
                (stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
            )
        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


class PlbNvlsMaterializeRMSNormDeviceKernel:
    """FlashInfer's MaterializeRMSNorm (same outputs) for an NVLS call: it also returns the
    flag words and this rank's rows of the NVLS generation to the -0 sentinel."""

    def __init__(
        self,
        *,
        hidden_size: int,
        tp_size: int,
        capacity_m: int,
        local_capacity: int,
        threads: int,
        rms_epsilon: float,
        weight_bias: float,
        write_residual_output: bool,
        enable_pdl: bool,
        apply_rms_norm: bool = True,
    ) -> None:
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.capacity_m = capacity_m
        self.local_capacity = local_capacity
        self.threads = threads
        self.rms_epsilon = rms_epsilon
        self.weight_bias = weight_bias
        self.write_residual_output = write_residual_output
        self.enable_pdl = enable_pdl
        self.apply_rms_norm = apply_rms_norm
        self.fragments = hidden_size // VEC_BF16
        self.trips = math.ceil(self.fragments / threads)
        self.warps = threads // WARP_SIZE

    @cute.jit
    def __call__(
        self,
        prenorm_mailbox: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        gamma: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            prenorm_mailbox,
            residual_output,
            norm_output,
            gamma,
            stage_state,
            contribution_mailbox,
        ).launch(
            grid=(m, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.warps * 4,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        prenorm_mailbox: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        gamma: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox: Int64,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()

        gamma_fragments = cute.make_rmem_tensor(
            cute.make_layout((self.trips, VEC_BF16), stride=(VEC_BF16, 1)), BFloat16
        )
        prenorm_fragments = cute.make_rmem_tensor(
            cute.make_layout((self.trips, VEC_BF16), stride=(VEC_BF16, 1)), BFloat16
        )
        gamma_fragments.fill(BFloat16(0.0))
        prenorm_fragments.fill(BFloat16(0.0))
        if cutlass.const_expr(self.apply_rms_norm):
            for trip in cutlass.range_constexpr(self.trips):
                fragment = tidx + trip * self.threads
                if fragment < self.fragments:
                    gamma_pointer = cute.make_ptr(
                        BFloat16,
                        (gamma.iterator + Int64(fragment) * VEC_BF16).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    gamma_fragments[trip, None].store(
                        packed_u32x4_to_bf16x8(load_global_u32x4(gamma_pointer))
                    )

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
        thread_sum = Float32(0.0)
        for trip in cutlass.range_constexpr(self.trips):
            fragment = tidx + trip * self.threads
            if fragment < self.fragments:
                mailbox_offset = (
                    Int64(stage) * self.capacity_m + Int64(token)
                ) * self.hidden_size + Int64(fragment) * VEC_BF16
                mailbox_pointer = cute.make_ptr(
                    BFloat16,
                    (prenorm_mailbox.iterator + mailbox_offset).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                packed = load_global_u32x4(mailbox_pointer, volatile=True)
                while fragment_has_negative_zero(packed):
                    packed = load_global_u32x4(mailbox_pointer, volatile=True)
                prenorm = packed_u32x4_to_bf16x8(packed)
                if cutlass.const_expr(self.apply_rms_norm):
                    prenorm_fragments[trip, None].store(prenorm)

                if cutlass.const_expr(self.write_residual_output):
                    output_offset = (
                        Int64(token) * self.hidden_size + Int64(fragment) * VEC_BF16
                    )
                    store_global_u32x4(
                        Int64((residual_output.iterator + output_offset).toint()),
                        packed,
                    )

                if cutlass.const_expr(self.apply_rms_norm):
                    values = prenorm.to(Float32)
                    thread_sum = thread_sum + (values * values).reduce(
                        cute.ReductionOp.ADD,
                        init_val=Float32(0.0),
                        reduction_profile=0,
                    )

        smem = cutlass.utils.SmemAllocator()
        warp_sums = smem.allocate_array(Float32, self.warps)
        if cutlass.const_expr(self.apply_rms_norm):
            full_sum = _block_sum(thread_sum, warp_sums, self.warps)
            inverse_rms = cute.math.rsqrt(
                full_sum / Float32(self.hidden_size) + Float32(self.rms_epsilon),
                fastmath=True,
            )
            for trip in cutlass.range_constexpr(self.trips):
                fragment = tidx + trip * self.threads
                if fragment < self.fragments:
                    gamma_value = gamma_fragments[trip, None].load().to(Float32)
                    if cutlass.const_expr(self.weight_bias != 0.0):
                        gamma_value = gamma_value + Float32(self.weight_bias)
                    result = (
                        prenorm_fragments[trip, None].load().to(Float32)
                        * inverse_rms
                        * gamma_value
                    ).to(BFloat16)
                    output_offset = (
                        Int64(token) * self.hidden_size + Int64(fragment) * VEC_BF16
                    )
                    store_global_u32x4(
                        Int64((norm_output.iterator + output_offset).toint()),
                        bf16x8_to_packed_u32x4(result),
                    )

        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        if token == 0 and tidx < self.tp_size:
            # Every CTA of the reduce kernel saw these words before that grid completed.
            store_global_u32_address(
                contribution_mailbox
                + _flag_offset(
                    stage, tidx, self.tp_size, self.local_capacity, self.hidden_size
                ),
                Uint32(NEGATIVE_ZERO_BF16_PAIR),
            )
        data_stage = (stage + Uint32(LAMPORT_GENERATIONS - 1)) % Uint32(LAMPORT_GENERATIONS)
        for trip in cutlass.range_constexpr(self.trips):
            fragment = tidx + trip * self.threads
            if fragment < self.fragments:
                row_offset = Int64(token) * self.hidden_size + Int64(fragment) * VEC_BF16
                store_lamport_sentinel_u32x4(
                    Int64(
                        (
                            prenorm_mailbox.iterator
                            + Int64(stage) * self.capacity_m * self.hidden_size
                            + row_offset
                        ).toint()
                    )
                )
                # This rank's partial row: every owner's ld_reduce of it returned before
                # the prenorm row read above was multicast.
                store_lamport_sentinel_u32x4(
                    Int64(
                        (
                            prenorm_mailbox.iterator
                            + Int64(data_stage) * self.capacity_m * self.hidden_size
                            + row_offset
                        ).toint()
                    )
                )


def bt_nvls_unsupported(workspace, *, lamport: int, fp8_out: bool) -> str | None:
    """Why ``workspace``'s BT route cannot take the NVLS reduction, or None."""
    from flashinfer.comm.mnnvl_cutedsl import ProtocolKind

    protocol = workspace._protocols.get(ProtocolKind.BT)
    if protocol is None:
        return "no BT route"
    if lamport:
        return f"SGLANG_PLB_MAT_LAMPORT={lamport} (the NVLS norm keeps FlashInfer's order)"
    if fp8_out:
        return "SGLANG_PLB_AR_FP8_OUT (the NVLS norm writes no E4M3 rows)"
    if protocol.hidden_size % (VEC_BF16 * PUBLISH_THREADS):
        return f"hidden {protocol.hidden_size} not a multiple of {VEC_BF16 * PUBLISH_THREADS}"
    if protocol.state.prenorm_mailbox.multicast_address is None:
        return "the prenorm mailbox has no multicast mapping"
    if protocol.state.contribution_mailbox.peer_addresses is None:
        return "the contribution mailbox has no peer mapping"
    return None


class BTNvls:
    """The NVLS kernels for ``workspace``'s BT route (an MNNVLCuteDSLAllReduceFusionWorkspace).

    Calls of at least ``min_rows`` rows that the profile routes to BT run the NVLS publish,
    reduce and norm on the BT protocol's own state; other calls are untouched.
    ``early_loads``: the residual, top-k metadata and shared rows may be read before the
    grid-dependency wait (SGLang allows it without speculative decoding).
    """

    def __init__(
        self,
        workspace,
        *,
        min_rows: int,
        early_loads: bool,
        acc_f32: bool = True,
        relaxed_flags: bool = False,
        early_trigger: bool = False,
        skip_flags: bool = False,
    ) -> None:
        from flashinfer.comm.mnnvl_cutedsl import ProtocolKind
        from flashinfer.comm.mnnvl_cutedsl.runtime import current_cu_stream

        self._bt_kind = ProtocolKind.BT
        self.protocol = protocol = workspace._protocols[ProtocolKind.BT]
        self.profile = workspace.profile
        self.min_rows = int(min_rows)
        self.early_loads = bool(early_loads)
        self.hidden_size = protocol.hidden_size
        state = protocol.state
        self.prenorm_multicast = int(state.prenorm_mailbox.multicast_address)
        self.contribution_peers = state.contribution_mailbox.peer_addresses
        stream = current_cu_stream()
        tunings = (*protocol.all_reduce_kernels, *protocol.finalize_kernels)
        pdl = {tuning.collective.enable_pdl for tuning in tunings}
        publish = dict(
            hidden_size=protocol.hidden_size,
            capacity_m=protocol.capacity_m,
            early_loads=self.early_loads,
            early_trigger=early_trigger,
        )
        self.publish_kernels = {
            (finalize, enable_pdl): self._compile_publish(
                PlbNvlsPublishDeviceKernel(
                    **publish,
                    enable_pdl=enable_pdl,
                    finalize=finalize,
                    top_k=protocol.top_k,
                    routed_scaling_factor=protocol.routed_scaling_factor,
                    include_shared_expert=protocol.include_shared_expert,
                ),
                stream,
            )
            for finalize in (False, True)
            for enable_pdl in pdl
        }
        self.reduce_kernels = {
            enable_pdl: self._compile_reduce(
                PlbNvlsReduceDeviceKernel(
                    hidden_size=protocol.hidden_size,
                    tp_size=protocol.tp_size,
                    rank=protocol.rank,
                    capacity_m=protocol.capacity_m,
                    local_capacity=protocol.local_capacity,
                    add_residual=protocol.add_residual,
                    enable_pdl=enable_pdl,
                    early_loads=self.early_loads,
                    acc_f32=acc_f32,
                    relaxed_flags=relaxed_flags,
                    skip_flags=skip_flags,
                ),
                stream,
            )
            for enable_pdl in pdl
        }
        self.norm_kernels = {
            key: self._compile_norm(*key, stream)
            for key in {
                (tuning.collective.rms_threads, tuning.collective.enable_pdl)
                for tuning in tunings
            }
        }
        self._first_call = {"all-reduce": True, "finalize": True}

    def _compile_publish(self, kernel, stream):
        from cutlass.cute.runtime import make_fake_compact_tensor

        return cute.compile(
            kernel,
            *(Int64(0) for _ in range(4)),
            make_fake_compact_tensor(Int32, (2,), assumed_align=4),
            Int64(0),
            Int32(self.protocol.capacity_m),
            stream,
        )

    def _compile_reduce(self, kernel, stream):
        from cutlass.cute.runtime import make_fake_compact_tensor

        return cute.compile(
            kernel,
            Int64(0),
            make_fake_compact_tensor(Int32, (2,), assumed_align=4),
            Int64(0),
            Int64(0),
            make_fake_compact_tensor(Int64, (self.protocol.tp_size,), assumed_align=8),
            Int32(self.protocol.capacity_m),
            stream,
        )

    def _compile_norm(self, threads: int, enable_pdl: bool, stream):
        from cutlass.cute.runtime import make_fake_compact_tensor
        from flashinfer.comm.mnnvl_cutedsl.runtime import make_fake_dynamic_compact_tensor

        protocol = self.protocol
        hidden = protocol.hidden_size
        kernel = PlbNvlsMaterializeRMSNormDeviceKernel(
            hidden_size=hidden,
            tp_size=protocol.tp_size,
            capacity_m=protocol.capacity_m,
            local_capacity=protocol.local_capacity,
            threads=threads,
            rms_epsilon=protocol.rms_epsilon,
            weight_bias=protocol.weight_bias,
            write_residual_output=protocol.write_residual_output,
            enable_pdl=enable_pdl,
            apply_rms_norm=protocol.apply_rms_norm,
        )
        rows = make_fake_dynamic_compact_tensor(
            BFloat16, alignment=16, divisibility=hidden
        )
        return cute.compile(
            kernel,
            make_fake_compact_tensor(
                BFloat16,
                (LAMPORT_GENERATIONS * protocol.capacity_m * hidden,),
                assumed_align=16,
            ),
            rows,
            rows,
            make_fake_compact_tensor(BFloat16, (hidden,), assumed_align=16),
            make_fake_compact_tensor(Int32, (2,), assumed_align=4),
            Int64(0),
            Int32(protocol.capacity_m),
            stream,
        )

    def describe(self) -> str:
        return (
            f"min rows {self.min_rows}, capacity {self.protocol.capacity_m}, "
            f"early loads {self.early_loads}, multicast reused: the BT prenorm mailbox's"
        )

    def takes_all_reduce(self, m: int) -> bool:
        return (
            m >= self.min_rows
            and self.profile.all_reduce_routes.select(m).protocol is self._bt_kind
        )

    def takes_finalize(self, m: int) -> bool:
        return (
            m >= self.min_rows
            and self.profile.finalize_routes.select(m).protocol is self._bt_kind
        )

    def all_reduce(
        self,
        *,
        local_contribution: torch.Tensor,
        residual_source: torch.Tensor | None,
        gamma: torch.Tensor,
        norm_output: torch.Tensor,
        residual_output: torch.Tensor,
        m: int,
    ) -> None:
        collective = self.profile.all_reduce_routes.select(m).preset.collective
        self._note_first("all-reduce", m)
        self._launch(
            collective,
            finalize=False,
            rows=local_contribution,
            residual_source=residual_source,
            gamma=gamma,
            norm_output=norm_output,
            residual_output=residual_output,
            m=m,
        )

    def finalize(
        self,
        *,
        routed_output: torch.Tensor,
        expert_weights: torch.Tensor,
        permuted_indices: torch.Tensor,
        shared_output: torch.Tensor | None,
        residual_source: torch.Tensor | None,
        gamma: torch.Tensor,
        norm_output: torch.Tensor,
        residual_output: torch.Tensor,
        m: int,
    ) -> None:
        if self.protocol.include_shared_expert != (shared_output is not None):
            raise ValueError("shared_output does not match the compiled workspace")
        collective = self.profile.finalize_routes.select(m).preset.collective
        self._note_first("finalize", m)
        self._launch(
            collective,
            finalize=True,
            rows=routed_output,
            expert_weights=expert_weights,
            permuted_indices=permuted_indices,
            shared_output=shared_output,
            residual_source=residual_source,
            gamma=gamma,
            norm_output=norm_output,
            residual_output=residual_output,
            m=m,
        )

    def _note_first(self, op: str, m: int) -> None:
        if self._first_call[op]:
            self._first_call[op] = False
            logger.info(
                "SGLANG_PLB_BT_NVLS=%d: first %s on NVLS multimem.ld_reduce (m %d)",
                self.min_rows,
                op,
                m,
            )

    def _launch(
        self,
        collective,
        *,
        finalize: bool,
        rows: torch.Tensor,
        residual_source: torch.Tensor | None,
        gamma: torch.Tensor,
        norm_output: torch.Tensor,
        residual_output: torch.Tensor,
        m: int,
        expert_weights: torch.Tensor | None = None,
        permuted_indices: torch.Tensor | None = None,
        shared_output: torch.Tensor | None = None,
    ) -> None:
        from flashinfer.comm.mnnvl_cutedsl.runtime import (
            current_cu_stream,
            to_cute,
            to_cute_dynamic,
        )

        protocol = self.protocol
        if not 1 <= m <= protocol.capacity_m:
            raise ValueError(f"m must be in [1, {protocol.capacity_m}]")
        if protocol.add_residual != (residual_source is not None):
            raise ValueError("residual_source does not match the compiled workspace")
        state = protocol.state
        hidden = self.hidden_size
        stream = current_cu_stream()
        stage_state = to_cute(state.stage_state, 4)
        contribution = Int64(state.contribution_mailbox.tensor.data_ptr())

        def address(tensor):
            return Int64(0 if tensor is None else tensor.data_ptr())

        self.publish_kernels[(finalize, collective.enable_pdl)](
            address(rows),
            address(expert_weights),
            address(permuted_indices),
            address(shared_output),
            stage_state,
            Int64(state.prenorm_mailbox.tensor.data_ptr()),
            Int32(m),
            stream,
        )
        self.reduce_kernels[collective.enable_pdl](
            address(residual_source),
            stage_state,
            Int64(self.prenorm_multicast),
            contribution,
            to_cute(self.contribution_peers, 8),
            Int32(m),
            stream,
        )
        self.norm_kernels[(collective.rms_threads, collective.enable_pdl)](
            to_cute(state.prenorm_mailbox.tensor.flatten(), 16),
            to_cute_dynamic(residual_output.flatten(), 16, divisibility=hidden),
            to_cute_dynamic(norm_output.flatten(), 16, divisibility=hidden),
            to_cute(gamma, 16),
            stage_state,
            contribution,
            Int32(m),
            stream,
        )
