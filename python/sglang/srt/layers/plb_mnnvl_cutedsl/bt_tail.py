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
# (_OwnerReduceMulticastDeviceKernel, _MaterializeRMSNormDeviceKernel; FlashInfer
# 0.7.1 as shipped in the sglang-nightly-dev-cu13-20261015 image) for SGLang's
# PERF-LB knobs.

"""The BT tail (reduce + MaterializeRMSNorm), bit for bit FlashInfer's outputs.

``early_trigger`` (reduce): the CTA lets the norm kernel launch right after its own
grid-dependency wait instead of after the multicast. Norm: ``early_dependents`` lets
the dependents launch at entry; ``poll_before_wait`` polls the prenorm mailbox before
the grid-dependency wait, which moves to just before the sentinel reset (both:
TensorRT-LLM's rmsNormLamport order).
``fp8_out`` (norm): the norm rows are also written as E4M3 under a static per-tensor
scale, byte for byte SGLang's ``static_quant_fp8`` of them.
"""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from flashinfer.comm.mnnvl_cutedsl.cute_dsl_primitives import (
    VEC_BF16,
    WARP_SIZE,
    bf16x8_to_packed_u32x4,
    fragment_has_negative_zero,
    load_global_u32x4,
    load_volatile_u32,
    packed_u32x4_to_bf16x8,
    sanitize_negative_zero_u32x4,
    stmc_bf16x8,
    store_global_u32,
    store_global_u32x2,
    store_global_u32x4,
    store_lamport_sentinel_u32x4,
)
from flashinfer.comm.mnnvl_cutedsl.kernel_bt.device_kernels import (
    ACTIVE_STAGE,
    LAMPORT_GENERATIONS,
    NEXT_STAGE,
    _block_sum,
)

# static_quant_fp8's clamp bounds (float8_e4m3fn) as PTX float literals
E4M3_MAX = 448.0
_E4M3_MAX_PTX = "0f43E00000"
_E4M3_MIN_PTX = "0fC3E00000"


@dsl_user_op
def _load_global_f32(address: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [address.ir_value(loc=loc, ip=ip)],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _reciprocal_div_full(value: Float32, *, loc=None, ip=None) -> Float32:
    # _static_quant_fp8's 1.0 / scale: Triton lowers an fp32 division to div.full.f32.
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [value.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .f32 one;\n\tmov.f32 one, 0f3F800000;\n\tdiv.full.f32 $0, one, $1;\n\t}",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16x4_to_e4m3x4(
    low: Uint32, high: Uint32, inverse_scale: Float32, *, loc=None, ip=None
) -> Uint32:
    """Four BF16 (two packed words, element 0 lowest) as four E4M3 bytes:
    min(max(x * inverse_scale, -448), 448) rounded to nearest even."""
    lines = [
        ".reg .b16 h0, h1, h2, h3, q0, q1;",
        ".reg .f32 f0, f1, f2, f3, lo, hi;",
        f"mov.f32 lo, {_E4M3_MIN_PTX};",
        f"mov.f32 hi, {_E4M3_MAX_PTX};",
        "mov.b32 {h0, h1}, $1;",
        "mov.b32 {h2, h3}, $2;",
    ]
    for i in range(4):
        lines += [
            f"cvt.f32.bf16 f{i}, h{i};",
            f"mul.rn.f32 f{i}, f{i}, $3;",
            f"max.f32 f{i}, f{i}, lo;",
            f"min.f32 f{i}, f{i}, hi;",
        ]
    # cvt d, a, b puts a in the high byte
    lines += [
        "cvt.rn.satfinite.e4m3x2.f32 q0, f1, f0;",
        "cvt.rn.satfinite.e4m3x2.f32 q1, f3, f2;",
        "mov.b32 $0, {q0, q1};",
    ]
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                low.ir_value(loc=loc, ip=ip),
                high.ir_value(loc=loc, ip=ip),
                inverse_scale.ir_value(loc=loc, ip=ip),
            ],
            "{\n\t" + "\n\t".join(lines) + "\n\t}",
            "=r,r,r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class PlbOwnerReduceMulticastDeviceKernel:
    """FlashInfer's call signature (it takes the place of the tail's reduce)."""

    def __init__(
        self,
        *,
        hidden_size: int,
        tp_size: int,
        rank: int,
        capacity_m: int,
        local_capacity: int,
        threads: int,
        add_residual: bool,
        enable_pdl: bool,
        early_trigger: bool = False,
    ) -> None:
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.rank = rank
        self.capacity_m = capacity_m
        self.local_capacity = local_capacity
        self.threads = threads
        self.add_residual = add_residual
        self.enable_pdl = enable_pdl
        self.early_trigger = early_trigger
        self.fragments = hidden_size // VEC_BF16
        self.ctas_per_token = math.ceil(self.fragments / threads)

    @cute.jit
    def __call__(
        self,
        contribution_mailbox: cute.Tensor,
        residual_source: cute.Tensor,
        stage_state: cute.Tensor,
        prenorm_mailbox_multicast_address: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        local_tokens = (m + Int32(self.tp_size - 1)) // Int32(self.tp_size)
        self.kernel(
            contribution_mailbox,
            residual_source,
            stage_state,
            prenorm_mailbox_multicast_address,
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
        contribution_mailbox: cute.Tensor,
        residual_source: cute.Tensor,
        stage_state: cute.Tensor,
        prenorm_mailbox_multicast_address: Int64,
        m: Int32,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        local_token = block // self.ctas_per_token
        fragment = (block % self.ctas_per_token) * self.threads + tidx
        token = local_token * self.tp_size + self.rank
        active = token < m and fragment < self.fragments

        if cutlass.const_expr(self.add_residual):
            residual = cute.make_rmem_tensor(
                cute.make_layout((VEC_BF16,)),
                BFloat16,
            )
            residual.fill(BFloat16(0.0))
            if active:
                residual_offset = (
                    Int64(token) * self.hidden_size + Int64(fragment) * VEC_BF16
                )
                residual_pointer = cute.make_ptr(
                    BFloat16,
                    (residual_source.iterator + residual_offset).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                residual.store(
                    packed_u32x4_to_bf16x8(load_global_u32x4(residual_pointer))
                )

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.early_trigger):
                # The publish kernel (ACTIVE_STAGE's writer) has completed; the norm
                # kernel reads nothing else of this grid but the polled mailbox rows.
                cute.arch.griddepcontrol_launch_dependents()

        stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
        rank_values = cute.make_rmem_tensor(
            cute.make_layout((self.tp_size, 4)),
            Uint32,
        )
        rank_values.fill(Uint32(0))
        dirty = active
        while dirty:
            dirty = False
            for source_rank in cutlass.range_constexpr(self.tp_size):
                source_offset = (
                    (Int64(stage) * self.tp_size + source_rank) * self.local_capacity
                    + Int64(local_token)
                ) * self.hidden_size + Int64(fragment) * VEC_BF16
                source_pointer = cute.make_ptr(
                    BFloat16,
                    (contribution_mailbox.iterator + source_offset).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                packed = load_global_u32x4(source_pointer, volatile=True)
                dirty = dirty | fragment_has_negative_zero(packed)
                for word in cutlass.range_constexpr(4):
                    rank_values[source_rank, word] = packed[word]
        if active:
            reduced = cute.make_rmem_tensor(
                cute.make_layout((VEC_BF16,)),
                Float32,
            )
            reduced.fill(Float32(0.0))
            for source_rank in cutlass.range_constexpr(self.tp_size):
                packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                for word in cutlass.range_constexpr(4):
                    packed[word] = rank_values[source_rank, word]
                reduced.store(
                    reduced.load() + packed_u32x4_to_bf16x8(packed.load()).to(Float32)
                )

            prenorm = reduced.load()
            if cutlass.const_expr(self.add_residual):
                prenorm = prenorm + residual.load().to(Float32)
            prenorm_bf16 = prenorm.to(BFloat16)
            output_offset = (
                Int64(stage) * self.capacity_m + Int64(token)
            ) * self.hidden_size + Int64(fragment) * VEC_BF16
            stmc_bf16x8(
                prenorm_mailbox_multicast_address + output_offset * 2,
                sanitize_negative_zero_u32x4(bf16x8_to_packed_u32x4(prenorm_bf16)),
            )

        if block == 0 and tidx == 0:
            store_global_u32(
                stage_state.iterator + NEXT_STAGE,
                (stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
            )
        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl and not self.early_trigger):
            cute.arch.griddepcontrol_launch_dependents()

        if active:
            for source_rank in cutlass.range_constexpr(self.tp_size):
                source_offset = (
                    (Int64(stage) * self.tp_size + source_rank) * self.local_capacity
                    + Int64(local_token)
                ) * self.hidden_size + Int64(fragment) * VEC_BF16
                store_lamport_sentinel_u32x4(
                    Int64((contribution_mailbox.iterator + source_offset).toint())
                )


class PlbMaterializeRMSNormDeviceKernel:
    """FlashInfer's call signature (it takes the place of the tail's rms_norm)."""

    def __init__(
        self,
        *,
        hidden_size: int,
        capacity_m: int,
        threads: int,
        rms_epsilon: float,
        weight_bias: float,
        write_residual_output: bool,
        enable_pdl: bool,
        apply_rms_norm: bool = True,
        early_dependents: bool = False,
        after_wait_dependents: bool = False,
        poll_before_wait: bool = False,
        fp8_out: bool = False,
    ) -> None:
        if fp8_out and not apply_rms_norm:
            raise ValueError("the E4M3 copy is of the norm, which is compiled out")
        self.hidden_size = hidden_size
        self.capacity_m = capacity_m
        self.threads = threads
        self.rms_epsilon = rms_epsilon
        self.weight_bias = weight_bias
        self.write_residual_output = write_residual_output
        self.enable_pdl = enable_pdl
        self.apply_rms_norm = apply_rms_norm
        self.early_dependents = early_dependents
        self.after_wait_dependents = after_wait_dependents
        self.poll_before_wait = poll_before_wait
        self.fp8_out = fp8_out
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
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            prenorm_mailbox,
            residual_output,
            norm_output,
            gamma,
            stage_state,
            Int64(0),
            Int64(0),
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
        fp8_output: Int64,
        fp8_scale: Int64,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()

        if cutlass.const_expr(self.enable_pdl and self.early_dependents):
            # Dependents may launch now; their own wait still covers this grid.
            cute.arch.griddepcontrol_launch_dependents()

        gamma_fragments = cute.make_rmem_tensor(
            cute.make_layout(
                (self.trips, VEC_BF16),
                stride=(VEC_BF16, 1),
            ),
            BFloat16,
        )
        prenorm_fragments = cute.make_rmem_tensor(
            cute.make_layout(
                (self.trips, VEC_BF16),
                stride=(VEC_BF16, 1),
            ),
            BFloat16,
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

        inverse_scale = Float32(0.0)
        if cutlass.const_expr(self.fp8_out):
            inverse_scale = _reciprocal_div_full(_load_global_f32(fp8_scale))

        if cutlass.const_expr(self.enable_pdl and not self.poll_before_wait):
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.enable_pdl and self.after_wait_dependents):
            # E6v2: the producer is complete, so releasing the projection here cannot
            # make this grid resident while it polls. The projection's own PDL wait
            # still protects the E4M3 rows materialized below.
            cute.arch.griddepcontrol_launch_dependents()

        # poll_before_wait: the publish kernel that wrote ACTIVE_STAGE completed before
        # the reduce kernel let this grid launch; a mailbox row is final once no word
        # holds the -0 sentinel.
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
                    packed_result = bf16x8_to_packed_u32x4(result)
                    store_global_u32x4(
                        Int64((norm_output.iterator + output_offset).toint()),
                        packed_result,
                    )
                    if cutlass.const_expr(self.fp8_out):
                        quantized = cute.make_rmem_tensor(
                            cute.make_layout((2,)), Uint32
                        )
                        quantized[0] = _bf16x4_to_e4m3x4(
                            packed_result[0], packed_result[1], inverse_scale
                        )
                        quantized[1] = _bf16x4_to_e4m3x4(
                            packed_result[2], packed_result[3], inverse_scale
                        )
                        # one byte an element
                        store_global_u32x2(fp8_output + output_offset, quantized.load())

        if cutlass.const_expr(self.enable_pdl and self.poll_before_wait):
            # The reduce kernel (and so everything before it) completes before the
            # reset and before this grid's completion.
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(
            not self.early_dependents and not self.after_wait_dependents
        ):
            cute.arch.barrier()
            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_launch_dependents()

        for trip in cutlass.range_constexpr(self.trips):
            fragment = tidx + trip * self.threads
            if fragment < self.fragments:
                mailbox_offset = (
                    Int64(stage) * self.capacity_m + Int64(token)
                ) * self.hidden_size + Int64(fragment) * VEC_BF16
                store_lamport_sentinel_u32x4(
                    Int64((prenorm_mailbox.iterator + mailbox_offset).toint())
                )


class PlbMaterializeRMSNormFp8DeviceKernel(PlbMaterializeRMSNormDeviceKernel):
    """With the E4M3 rows' address and the scale's address (an fp32 scalar)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, fp8_out=True)

    @cute.jit
    def __call__(
        self,
        prenorm_mailbox: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        gamma: cute.Tensor,
        stage_state: cute.Tensor,
        fp8_output: Int64,
        fp8_scale: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            prenorm_mailbox,
            residual_output,
            norm_output,
            gamma,
            stage_state,
            fp8_output,
            fp8_scale,
        ).launch(
            grid=(m, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.warps * 4,
            stream=stream,
            use_pdl=self.enable_pdl,
        )
