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
# _OwnerReduceMulticastDeviceKernel, _MaterializeRMSNormDeviceKernel; FlashInfer
# 0.7.1 as shipped in the sglang-nightly-dev-cu13-20261015 image) for SGLang's
# PERF-LP E4M3 payload.

"""The BT route with an E4M3 Lamport payload (TensorRT-LLM's TRTLLM_MNNVL_TWOSHOT_FP8).

A payload row is H E4M3 bytes followed by H / 256 FP32 scales, one per 256 consecutive
elements (one warp: 32 threads x 8 elements). A block is stored as
satfinite_rn(x * RN(448 / amax)) under the scale RN(amax * (1 / 448)), amax its max |x|
floored at 1e-30. A data word equal to the Lamport sentinel (FP32 -0, the E4M3 bytes
+0, +0, +0, -0) is stored as +0 and scales are positive, so every word of an unwritten
row reads as the sentinel. The publish kernels send the token's owner the row quantized;
the owner's own rows stay BF16 in a local buffer. The owner sums the ranks in rank order
in FP32 (its own row exact, the others dequantized), quantizes the sum the same way and
multicasts it. The norm kernel adds the residual to BF16(E4M3 x scale) (BF16, nearest
even), writes that as the residual output and normalizes it as FlashInfer's
MaterializeRMSNorm does.
"""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op
from flashinfer.comm.mnnvl_cutedsl.cute_dsl_primitives import (
    VEC_BF16,
    WARP_SIZE,
    bf16x8_to_packed_u32x4,
    load_global_bf16_as_f32,
    load_global_u32x4,
    load_global_u32x4_address,
    load_volatile_u32,
    packed_u32x4_to_bf16x8,
    store_global_u32,
    store_global_u32_address,
    store_global_u32x2,
    store_global_u32x4,
)
from flashinfer.comm.mnnvl_cutedsl.kernel_bt.device_kernels import (
    ACTIVE_STAGE,
    LAMPORT_GENERATIONS,
    NEXT_STAGE,
    _block_sum,
)

from sglang.srt.layers.plb_mnnvl_cutedsl.bt_tail import (
    _bf16x4_to_e4m3x4,
    _load_global_f32,
    _reciprocal_div_full,
)

SCALE_BLOCK = WARP_SIZE * VEC_BF16
# FP32 -0.0, as E4M3 bytes +0, +0, +0, -0
E4M3_SENTINEL = 0x80000000
# FP32 bits of 1e-30 (the amax floor), 448 and RN(1 / 448)
_AMAX_FLOOR_PTX = "0f0DA24260"
_E4M3_MAX_PTX = "0f43E00000"
_INVERSE_E4M3_MAX_PTX = "0f3B124925"


def e4m3_row_bytes(hidden_size: int) -> int:
    """H E4M3 bytes then the row's FP32 scales."""
    return hidden_size + hidden_size // SCALE_BLOCK * 4


def _inline(result_type, operands, lines, constraints, *, side_effects, loc, ip):
    return llvm.inline_asm(
        result_type,
        operands,
        "{\n\t" + "\n\t".join(lines) + "\n\t}",
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _values(dtype, wrap, struct_value, count, loc, ip):
    return tuple(
        wrap(llvm.extractvalue(dtype, struct_value, [index], loc=loc, ip=ip))
        for index in range(count)
    )


@dsl_user_op
def _abs_max_bf16x8(
    w0: Uint32, w1: Uint32, w2: Uint32, w3: Uint32, *, loc=None, ip=None
) -> Float32:
    """max |x| of the eight BF16 values in four words, element 0 lowest."""
    lines = [
        ".reg .b16 " + ", ".join(f"h{i}" for i in range(8)) + ";",
        ".reg .f32 m, t;",
    ]
    for word in range(4):
        lines.append(f"mov.b32 {{h{2 * word}, h{2 * word + 1}}}, ${word + 1};")
    lines += ["cvt.f32.bf16 m, h0;", "abs.f32 m, m;"]
    for element in range(1, 8):
        lines += [f"cvt.f32.bf16 t, h{element};", "abs.f32 t, t;", "max.f32 m, m, t;"]
    lines.append("mov.f32 $0, m;")
    return Float32(
        _inline(
            T.f32(),
            [w.ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
            lines,
            "=f,r,r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _abs_max_f32x8(
    a0: Float32,
    a1: Float32,
    a2: Float32,
    a3: Float32,
    a4: Float32,
    a5: Float32,
    a6: Float32,
    a7: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    lines = [".reg .f32 m, t;", "abs.f32 m, $1;"]
    for element in range(2, 9):
        lines += [f"abs.f32 t, ${element};", "max.f32 m, m, t;"]
    lines.append("mov.f32 $0, m;")
    return Float32(
        _inline(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip)
                for a in (a0, a1, a2, a3, a4, a5, a6, a7)
            ],
            lines,
            "=f,f,f,f,f,f,f,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _warp_quant_scale(local_max: Float32, *, loc=None, ip=None):
    """(RN(448 / amax), RN(amax * (1 / 448))), amax the warp's max floored at 1e-30.
    All 32 lanes must call it."""
    lines = [".reg .f32 m, t, c;", "mov.f32 m, $2;"]
    for offset in (16, 8, 4, 2, 1):
        lines += [
            f"shfl.sync.bfly.b32 t, m, {offset}, 0x1f, 0xffffffff;",
            "max.f32 m, m, t;",
        ]
    lines += [
        f"mov.f32 c, {_AMAX_FLOOR_PTX};",
        "max.f32 m, m, c;",
        f"mov.f32 c, {_E4M3_MAX_PTX};",
        "div.rn.f32 $0, c, m;",
        f"mov.f32 c, {_INVERSE_E4M3_MAX_PTX};",
        "mul.rn.f32 $1, m, c;",
    ]
    result = _inline(
        llvm.StructType.get_literal([T.f32()] * 2),
        [Float32(local_max).ir_value(loc=loc, ip=ip)],
        lines,
        "=f,=f,f",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return _values(T.f32(), Float32, result, 2, loc, ip)


# cvt d, a, b puts a in the high byte; a word equal to the sentinel becomes +0.
_E4M3_PACK = [
    "cvt.rn.satfinite.e4m3x2.f32 q0, f1, f0;",
    "cvt.rn.satfinite.e4m3x2.f32 q1, f3, f2;",
    "mov.b32 w, {q0, q1};",
    "setp.eq.b32 p, w, 0x80000000;",
    "selp.b32 $0, 0, w, p;",
]


@dsl_user_op
def _e4m3x4_of_bf16x4(
    low: Uint32, high: Uint32, inverse: Float32, *, loc=None, ip=None
) -> Uint32:
    """Four BF16 (two words) times ``inverse`` as four E4M3 bytes, element 0 lowest."""
    lines = [
        ".reg .b16 h0, h1, h2, h3, q0, q1;",
        ".reg .f32 f0, f1, f2, f3;",
        ".reg .b32 w;",
        ".reg .pred p;",
        "mov.b32 {h0, h1}, $1;",
        "mov.b32 {h2, h3}, $2;",
    ]
    for element in range(4):
        lines += [
            f"cvt.f32.bf16 f{element}, h{element};",
            f"mul.rn.f32 f{element}, f{element}, $3;",
        ]
    return Uint32(
        _inline(
            T.i32(),
            [
                low.ir_value(loc=loc, ip=ip),
                high.ir_value(loc=loc, ip=ip),
                Float32(inverse).ir_value(loc=loc, ip=ip),
            ],
            lines + _E4M3_PACK,
            "=r,r,r,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _e4m3x4_of_f32x4(
    a0: Float32,
    a1: Float32,
    a2: Float32,
    a3: Float32,
    inverse: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    lines = [
        ".reg .b16 q0, q1;",
        ".reg .f32 f0, f1, f2, f3;",
        ".reg .b32 w;",
        ".reg .pred p;",
    ]
    for element in range(4):
        lines.append(f"mul.rn.f32 f{element}, ${element + 1}, $5;")
    return Uint32(
        _inline(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip) for a in (a0, a1, a2, a3, inverse)],
            lines + _E4M3_PACK,
            "=r,f,f,f,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


_E4M3_UNPACK = [
    "mov.b32 {lo, hi}, $8;",
    "cvt.rn.f16x2.e4m3x2 h01, lo;",
    "cvt.rn.f16x2.e4m3x2 h23, hi;",
    "mov.b32 {e0, e1}, h01;",
    "mov.b32 {e2, e3}, h23;",
]


@dsl_user_op
def _add_e4m3x4(
    a0: Float32,
    a1: Float32,
    a2: Float32,
    a3: Float32,
    word: Uint32,
    scale: Float32,
    *,
    loc=None,
    ip=None,
):
    """a_i + RN(e4m3_i x scale) (each RN), the four E4M3 of ``word``, element 0 lowest."""
    lines = [
        ".reg .b16 lo, hi, e0, e1, e2, e3;",
        ".reg .b32 h01, h23;",
        ".reg .f32 f0, f1, f2, f3;",
        *_E4M3_UNPACK,
    ]
    for element in range(4):
        lines += [
            f"cvt.f32.f16 f{element}, e{element};",
            f"mul.rn.f32 f{element}, f{element}, $9;",
            f"add.rn.f32 ${element}, ${element + 4}, f{element};",
        ]
    result = _inline(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Float32(a).ir_value(loc=loc, ip=ip) for a in (a0, a1, a2, a3)]
        + [word.ir_value(loc=loc, ip=ip), Float32(scale).ir_value(loc=loc, ip=ip)],
        lines,
        "=f,=f,=f,=f,f,f,f,f,r,f",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _values(T.f32(), Float32, result, 4, loc, ip)


@dsl_user_op
def _add_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _inline(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            ["add.rn.f32 $0, $1, $2;"],
            "=f,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f32_bits(value: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _inline(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            ["mov.b32 $0, $1;"],
            "=r,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bits_f32(word: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _inline(
            T.f32(),
            [word.ir_value(loc=loc, ip=ip)],
            ["mov.b32 $0, $1;"],
            "=f,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _dequant_add_bf16x8(
    payload, r0: Uint32, r1: Uint32, r2: Uint32, r3: Uint32, *, loc=None, ip=None
):
    """BF16(BF16(e4m3 x scale) + residual) for the eight E4M3 of payload[0:2] under the
    FP32 scale bits payload[2], residual as four BF16x2 words; four BF16x2 words out."""
    names = ", ".join
    lines = [
        ".reg .b16 lo, hi, "
        + names(f"e{i}" for i in range(8))
        + ", "
        + names(f"r{i}" for i in range(8))
        + ";",
        ".reg .b32 h01, h23;",
        ".reg .f32 s, "
        + names(f"f{i}" for i in range(8))
        + ", "
        + names(f"g{i}" for i in range(8))
        + ";",
        "mov.b32 s, $6;",
    ]
    for word in range(2):
        lines += [
            f"mov.b32 {{lo, hi}}, ${4 + word};",
            "cvt.rn.f16x2.e4m3x2 h01, lo;",
            "cvt.rn.f16x2.e4m3x2 h23, hi;",
            f"mov.b32 {{e{4 * word}, e{4 * word + 1}}}, h01;",
            f"mov.b32 {{e{4 * word + 2}, e{4 * word + 3}}}, h23;",
        ]
    for word in range(4):
        lines.append(f"mov.b32 {{r{2 * word}, r{2 * word + 1}}}, ${7 + word};")
    for element in range(8):
        lines += [
            f"cvt.f32.f16 f{element}, e{element};",
            f"mul.rn.f32 f{element}, f{element}, s;",
            f"cvt.rn.bf16.f32 e{element}, f{element};",
            f"cvt.f32.bf16 f{element}, e{element};",
            f"cvt.f32.bf16 g{element}, r{element};",
            f"add.rn.f32 f{element}, f{element}, g{element};",
        ]
    for word in range(4):
        # cvt d, a, b puts a in the high half
        lines.append(f"cvt.rn.bf16x2.f32 ${word}, f{2 * word + 1}, f{2 * word};")
    result = _inline(
        llvm.StructType.get_literal([T.i32()] * 4),
        [payload[index].ir_value(loc=loc, ip=ip) for index in range(3)]
        + [w.ir_value(loc=loc, ip=ip) for w in (r0, r1, r2, r3)],
        lines,
        "=r,=r,=r,=r,r,r,r,r,r,r,r",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [
            llvm.extractvalue(T.i32(), result, [index], loc=loc, ip=ip)
            for index in range(4)
        ],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@dsl_user_op
def _load_volatile_e4m3(
    data_address: Int64, scale_address: Int64, *, loc=None, ip=None
):
    """A thread's two E4M3 data words and its block's scale word."""
    result = _inline(
        llvm.StructType.get_literal([T.i32()] * 3),
        [data_address.ir_value(loc=loc, ip=ip), scale_address.ir_value(loc=loc, ip=ip)],
        [
            "ld.volatile.global.v2.u32 {$0, $1}, [$3];",
            "ld.volatile.global.u32 $2, [$4];",
        ],
        "=r,=r,=r,l,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([3], T.i32(), loc=loc),
        [
            llvm.extractvalue(T.i32(), result, [index], loc=loc, ip=ip)
            for index in range(3)
        ],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 3, Uint32)


@cute.jit
def _e4m3_pending(payload):
    return (
        (payload[0] == Uint32(E4M3_SENTINEL))
        | (payload[1] == Uint32(E4M3_SENTINEL))
        | (payload[2] == Uint32(E4M3_SENTINEL))
    )


@dsl_user_op
def _store_u32x2(
    address: Int64, low: Uint32, high: Uint32, *, loc=None, ip=None
) -> None:
    _inline(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            low.ir_value(loc=loc, ip=ip),
            high.ir_value(loc=loc, ip=ip),
        ],
        ["st.global.v2.u32 [$0], {$1, $2};"],
        "l,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _multicast_u32x2(
    address: Int64, low: Uint32, high: Uint32, *, loc=None, ip=None
) -> None:
    _inline(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            low.ir_value(loc=loc, ip=ip),
            high.ir_value(loc=loc, ip=ip),
        ],
        [
            ".reg .f32 a, b;",
            "mov.b32 a, $1;",
            "mov.b32 b, $2;",
            "multimem.st.relaxed.sys.global.v2.f32 [$0], {a, b};",
        ],
        "l,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _multicast_u32(address: Int64, word: Uint32, *, loc=None, ip=None) -> None:
    _inline(
        None,
        [address.ir_value(loc=loc, ip=ip), word.ir_value(loc=loc, ip=ip)],
        [
            ".reg .f32 a;",
            "mov.b32 a, $1;",
            "multimem.st.relaxed.sys.global.f32 [$0], a;",
        ],
        "l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _store_sentinel_u32x2(address: Int64, *, loc=None, ip=None) -> None:
    _inline(
        None,
        [address.ir_value(loc=loc, ip=ip)],
        [
            ".reg .b32 s;",
            "mov.b32 s, 0x80000000;",
            "st.global.v2.u32 [$0], {s, s};",
        ],
        "l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


def _check_scale_blocks(hidden_size: int, threads: int) -> None:
    # A scale block must be exactly one warp's 32 vectors, all in or all out of the row.
    if hidden_size % SCALE_BLOCK or threads % WARP_SIZE:
        raise ValueError(
            f"E4M3 payload needs hidden % {SCALE_BLOCK} == 0 and threads % "
            f"{WARP_SIZE} == 0, got hidden {hidden_size}, threads {threads}"
        )


class PlpSharedOnlyPublishE4M3DeviceKernel:
    """FlashInfer's SharedOnlyPublish with the E4M3 payload; rows this rank owns go to
    ``own_rows`` ([generation][local token][H] BF16) unquantized."""

    def __init__(
        self,
        *,
        hidden_size: int,
        tp_size: int,
        rank: int,
        local_capacity: int,
        threads: int,
        vectors_per_thread: int,
        enable_pdl: bool,
    ) -> None:
        _check_scale_blocks(hidden_size, threads)
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.rank = rank
        self.local_capacity = local_capacity
        self.threads = threads
        self.vectors_per_thread = vectors_per_thread
        self.enable_pdl = enable_pdl
        self.fragments = hidden_size // VEC_BF16
        self.fragments_per_cta = threads * vectors_per_thread
        self.ctas_per_token = math.ceil(self.fragments / self.fragments_per_cta)
        self.row_bytes = e4m3_row_bytes(hidden_size)

    @cute.jit
    def __call__(
        self,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_peer_addresses: cute.Tensor,
        own_rows: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            shared_output,
            stage_state,
            contribution_mailbox_peer_addresses,
            own_rows,
        ).launch(
            grid=(m * self.ctas_per_token, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_peer_addresses: cute.Tensor,
        own_rows: Int64,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        token = block // self.ctas_per_token
        fragment_base = (block % self.ctas_per_token) * self.fragments_per_cta + tidx

        inputs = cute.make_rmem_tensor(
            cute.make_layout(
                (self.vectors_per_thread, 4),
                stride=(4, 1),
            ),
            Uint32,
        )
        inputs.fill(Uint32(0))

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        for trip in cutlass.range_constexpr(self.vectors_per_thread):
            fragment = fragment_base + trip * self.threads
            if fragment < self.fragments:
                source_offset = (
                    Int64(token) * self.hidden_size + Int64(fragment) * VEC_BF16
                )
                source_pointer = cute.make_ptr(
                    BFloat16,
                    (shared_output.iterator + source_offset).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                packed = load_global_u32x4(source_pointer)
                for word in cutlass.range_constexpr(4):
                    inputs[trip, word] = packed[word]

        stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)
        destination_rank = token % self.tp_size
        local_token = token // self.tp_size
        if destination_rank == self.rank:
            own_row = own_rows + (
                Int64(stage) * self.local_capacity + Int64(local_token)
            ) * (self.hidden_size * 2)
            for trip in cutlass.range_constexpr(self.vectors_per_thread):
                own_fragment = fragment_base + trip * self.threads
                if own_fragment < self.fragments:
                    own_words = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                    for word in cutlass.range_constexpr(4):
                        own_words[word] = inputs[trip, word]
                    store_global_u32x4(
                        own_row + Int64(own_fragment) * 16, own_words.load()
                    )
        else:
            destination_base = cute.arch.load(
                (
                    contribution_mailbox_peer_addresses.iterator + destination_rank
                ).llvm_ptr,
                Int64,
            )
            row = (
                destination_base
                + (
                    (Int64(stage) * self.tp_size + self.rank) * self.local_capacity
                    + Int64(local_token)
                )
                * self.row_bytes
            )
            for trip in cutlass.range_constexpr(self.vectors_per_thread):
                peer_fragment = fragment_base + trip * self.threads
                if peer_fragment < self.fragments:
                    inverse, scale = _warp_quant_scale(
                        _abs_max_bf16x8(
                            inputs[trip, 0],
                            inputs[trip, 1],
                            inputs[trip, 2],
                            inputs[trip, 3],
                        )
                    )
                    _store_u32x2(
                        row + Int64(peer_fragment) * 8,
                        _e4m3x4_of_bf16x4(inputs[trip, 0], inputs[trip, 1], inverse),
                        _e4m3x4_of_bf16x4(inputs[trip, 2], inputs[trip, 3], inverse),
                    )
                    if lane == 0:
                        store_global_u32_address(
                            row
                            + Int64(self.hidden_size)
                            + Int64(peer_fragment // WARP_SIZE) * 4,
                            _f32_bits(scale),
                        )

        if block == 0 and tidx == 0:
            store_global_u32(stage_state.iterator + ACTIVE_STAGE, stage)
        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


class PlpVectorFinalizeUnicastE4M3DeviceKernel:
    """FlashInfer's VectorFinalizeUnicast (8 elements a thread) with the E4M3 payload;
    rows this rank owns go to ``own_rows`` unquantized."""

    def __init__(
        self,
        *,
        hidden_size: int,
        top_k: int,
        tp_size: int,
        rank: int,
        local_capacity: int,
        threads: int,
        routed_scaling_factor: float,
        include_shared_expert: bool,
        load_shared_expert_before_pdl: bool,
        enable_pdl: bool,
        prefetch_group: int,
    ) -> None:
        _check_scale_blocks(hidden_size, threads)
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.tp_size = tp_size
        self.rank = rank
        self.local_capacity = local_capacity
        self.threads = threads
        self.routed_scaling_factor = routed_scaling_factor
        self.include_shared_expert = include_shared_expert
        self.load_shared_expert_before_pdl = load_shared_expert_before_pdl
        self.enable_pdl = enable_pdl
        self.prefetch_group = prefetch_group
        self.prefetch_groups = (top_k + prefetch_group - 1) // prefetch_group
        self.words_per_fragment = VEC_BF16 // 2
        self.fragments = hidden_size // VEC_BF16
        self.ctas_per_token = math.ceil(self.fragments / threads)
        self.row_bytes = e4m3_row_bytes(hidden_size)

    @cute.jit
    def __call__(
        self,
        routed_output: cute.Tensor,
        expert_weights: cute.Tensor,
        permuted_indices: cute.Tensor,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_peer_addresses: cute.Tensor,
        own_rows: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            routed_output,
            expert_weights,
            permuted_indices,
            shared_output,
            stage_state,
            contribution_mailbox_peer_addresses,
            own_rows,
        ).launch(
            grid=(m * self.ctas_per_token, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.top_k * 8,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        routed_output: cute.Tensor,
        expert_weights: cute.Tensor,
        permuted_indices: cute.Tensor,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_peer_addresses: cute.Tensor,
        own_rows: Int64,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        token = block // self.ctas_per_token
        fragment = (block % self.ctas_per_token) * self.threads + tidx

        smem = cutlass.utils.SmemAllocator()
        staged_indices = smem.allocate_array(Int32, self.top_k)
        staged_weights = smem.allocate_array(Float32, self.top_k)
        metadata_index = Int32(tidx)
        while metadata_index < Int32(self.top_k):
            metadata_offset = Int64(token) * self.top_k + Int64(metadata_index)
            routed_index = cute.arch.load(
                (permuted_indices.iterator + metadata_offset).llvm_ptr,
                Int32,
            )
            weight = load_global_bf16_as_f32(
                Int64((expert_weights.iterator + metadata_offset).toint())
            )
            if cutlass.const_expr(self.routed_scaling_factor != 1.0):
                weight = weight * Float32(self.routed_scaling_factor)
            if routed_index == Int32(-1):
                # Vector loads are unpredicated; row zero is safe and its weight is zero.
                routed_index = Int32(0)
                weight = Float32(0.0)
            cute.arch.store(
                (staged_indices + metadata_index).llvm_ptr,
                routed_index,
            )
            cute.arch.store(
                (staged_weights + metadata_index).llvm_ptr,
                weight,
            )
            metadata_index = metadata_index + self.threads
        cute.arch.barrier()

        if cutlass.const_expr(self.include_shared_expert):
            shared_values = cute.make_rmem_tensor(
                cute.make_layout((VEC_BF16,)),
                BFloat16,
            )
            shared_values.fill(BFloat16(0.0))
            if cutlass.const_expr(self.load_shared_expert_before_pdl):
                if fragment < self.fragments:
                    shared_offset = (
                        Int64(token) * self.hidden_size + Int64(fragment) * VEC_BF16
                    )
                    shared_pointer = cute.make_ptr(
                        BFloat16,
                        (shared_output.iterator + shared_offset).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    shared_values.store(
                        packed_u32x4_to_bf16x8(load_global_u32x4(shared_pointer))
                    )

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        if cutlass.const_expr(
            self.include_shared_expert and not self.load_shared_expert_before_pdl
        ):
            if fragment < self.fragments:
                shared_offset = (
                    Int64(token) * self.hidden_size + Int64(fragment) * VEC_BF16
                )
                shared_pointer = cute.make_ptr(
                    BFloat16,
                    (shared_output.iterator + shared_offset).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                shared_values.store(
                    packed_u32x4_to_bf16x8(load_global_u32x4(shared_pointer))
                )

        stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)
        if fragment < self.fragments:
            accumulator = cute.make_rmem_tensor(
                cute.make_layout((VEC_BF16,)),
                Float32,
            )
            accumulator.fill(Float32(0.0))
            if cutlass.const_expr(self.prefetch_group == 1):
                for k in cutlass.range_constexpr(self.top_k):
                    routed_index = cute.arch.load(
                        (staged_indices + k).llvm_ptr,
                        Int32,
                    )
                    weight = cute.arch.load(
                        (staged_weights + k).llvm_ptr,
                        Float32,
                    )
                    source_offset = (
                        Int64(routed_index) * self.hidden_size
                        + Int64(fragment) * VEC_BF16
                    )
                    source_pointer = cute.make_ptr(
                        BFloat16,
                        (routed_output.iterator + source_offset).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    source_values = packed_u32x4_to_bf16x8(
                        load_global_u32x4(source_pointer)
                    ).to(Float32)
                    if weight != Float32(0.0):
                        accumulator.store(accumulator.load() + source_values * weight)
            else:
                inputs = cute.make_rmem_tensor(
                    cute.make_layout((self.prefetch_group, self.words_per_fragment)),
                    Uint32,
                )
                inputs.fill(Uint32(0))
                for group in cutlass.range_constexpr(self.prefetch_groups):
                    for item in cutlass.range_constexpr(self.prefetch_group):
                        k = group * self.prefetch_group + item
                        if cutlass.const_expr(k < self.top_k):
                            routed_index = cute.arch.load(
                                (staged_indices + k).llvm_ptr,
                                Int32,
                            )
                            source_offset = (
                                Int64(routed_index) * self.hidden_size
                                + Int64(fragment) * VEC_BF16
                            )
                            source_pointer = cute.make_ptr(
                                BFloat16,
                                (routed_output.iterator + source_offset).llvm_ptr,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            source = load_global_u32x4(source_pointer)
                            for word in cutlass.range_constexpr(
                                self.words_per_fragment
                            ):
                                inputs[item, word] = source[word]
                    for item in cutlass.range_constexpr(self.prefetch_group):
                        k = group * self.prefetch_group + item
                        if cutlass.const_expr(k < self.top_k):
                            source = cute.make_rmem_tensor(
                                cute.make_layout((self.words_per_fragment,)),
                                Uint32,
                            )
                            for word in cutlass.range_constexpr(
                                self.words_per_fragment
                            ):
                                source[word] = inputs[item, word]
                            weight = cute.arch.load(
                                (staged_weights + k).llvm_ptr,
                                Float32,
                            )
                            if weight != Float32(0.0):
                                accumulator.store(
                                    accumulator.load()
                                    + packed_u32x4_to_bf16x8(source.load()).to(Float32)
                                    * weight
                                )

            result = accumulator.load()
            if cutlass.const_expr(self.include_shared_expert):
                result = result + shared_values.load().to(Float32)
            # Staged in registers: a TensorSSA's element reads are cached per region.
            contribution = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
            contribution.store(bf16x8_to_packed_u32x4(result.to(BFloat16)))
            destination_rank = token % self.tp_size
            local_token = token // self.tp_size
            if destination_rank == self.rank:
                store_global_u32x4(
                    own_rows
                    + (
                        (Int64(stage) * self.local_capacity + Int64(local_token))
                        * self.hidden_size
                        + Int64(fragment) * VEC_BF16
                    )
                    * 2,
                    contribution.load(),
                )
            else:
                destination_base = cute.arch.load(
                    (
                        contribution_mailbox_peer_addresses.iterator + destination_rank
                    ).llvm_ptr,
                    Int64,
                )
                row = (
                    destination_base
                    + (
                        (Int64(stage) * self.tp_size + self.rank) * self.local_capacity
                        + Int64(local_token)
                    )
                    * self.row_bytes
                )
                inverse, scale = _warp_quant_scale(
                    _abs_max_bf16x8(
                        contribution[0],
                        contribution[1],
                        contribution[2],
                        contribution[3],
                    )
                )
                _store_u32x2(
                    row + Int64(fragment) * 8,
                    _e4m3x4_of_bf16x4(contribution[0], contribution[1], inverse),
                    _e4m3x4_of_bf16x4(contribution[2], contribution[3], inverse),
                )
                if lane == 0:
                    store_global_u32_address(
                        row
                        + Int64(self.hidden_size)
                        + Int64(fragment // WARP_SIZE) * 4,
                        _f32_bits(scale),
                    )

        if block == 0 and tidx == 0:
            store_global_u32(stage_state.iterator + ACTIVE_STAGE, stage)
        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


class PlpOwnerReduceMulticastE4M3DeviceKernel:
    """FlashInfer's OwnerReduceMulticast on the E4M3 payload: the FP32 rank-order sum
    (own row exact) quantized and multicast; the residual is added by the norm kernel."""

    def __init__(
        self,
        *,
        hidden_size: int,
        tp_size: int,
        rank: int,
        capacity_m: int,
        local_capacity: int,
        threads: int,
        enable_pdl: bool,
        early_trigger: bool = False,
    ) -> None:
        _check_scale_blocks(hidden_size, threads)
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.rank = rank
        self.capacity_m = capacity_m
        self.local_capacity = local_capacity
        self.threads = threads
        self.enable_pdl = enable_pdl
        self.early_trigger = early_trigger
        self.fragments = hidden_size // VEC_BF16
        self.ctas_per_token = math.ceil(self.fragments / threads)
        self.row_bytes = e4m3_row_bytes(hidden_size)

    @cute.jit
    def __call__(
        self,
        contribution_mailbox: Int64,
        own_rows: Int64,
        stage_state: cute.Tensor,
        prenorm_mailbox_multicast_address: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        local_tokens = (m + Int32(self.tp_size - 1)) // Int32(self.tp_size)
        self.kernel(
            contribution_mailbox,
            own_rows,
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
        contribution_mailbox: Int64,
        own_rows: Int64,
        stage_state: cute.Tensor,
        prenorm_mailbox_multicast_address: Int64,
        m: Int32,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        local_token = block // self.ctas_per_token
        fragment = (block % self.ctas_per_token) * self.threads + tidx
        token = local_token * self.tp_size + self.rank
        active = token < m and fragment < self.fragments

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.early_trigger):
                # The publish kernel (ACTIVE_STAGE's writer) has completed; the norm
                # kernel reads nothing else of this grid but the polled mailbox rows.
                cute.arch.griddepcontrol_launch_dependents()

        stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
        rank_words = cute.make_rmem_tensor(
            cute.make_layout((self.tp_size, 3)),
            Uint32,
        )
        rank_words.fill(Uint32(0))
        dirty = active
        while dirty:
            dirty = False
            for source_rank in cutlass.range_constexpr(self.tp_size):
                if cutlass.const_expr(source_rank != self.rank):
                    source_row = (
                        contribution_mailbox
                        + (
                            (Int64(stage) * self.tp_size + source_rank)
                            * self.local_capacity
                            + Int64(local_token)
                        )
                        * self.row_bytes
                    )
                    payload = _load_volatile_e4m3(
                        source_row + Int64(fragment) * 8,
                        source_row
                        + Int64(self.hidden_size)
                        + Int64(fragment // WARP_SIZE) * 4,
                    )
                    dirty = dirty | _e4m3_pending(payload)
                    for word in cutlass.range_constexpr(3):
                        rank_words[source_rank, word] = payload[word]
        if active:
            own_values = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
            own_values.store(
                packed_u32x4_to_bf16x8(
                    load_global_u32x4_address(
                        own_rows
                        + (
                            (Int64(stage) * self.local_capacity + Int64(local_token))
                            * self.hidden_size
                            + Int64(fragment) * VEC_BF16
                        )
                        * 2
                    )
                ).to(Float32)
            )
            reduced = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
            reduced.fill(Float32(0.0))
            for source_rank in cutlass.range_constexpr(self.tp_size):
                if cutlass.const_expr(source_rank == self.rank):
                    for element in cutlass.range_constexpr(VEC_BF16):
                        reduced[element] = _add_rn(
                            reduced[element], own_values[element]
                        )
                else:
                    source_scale = _bits_f32(rank_words[source_rank, 2])
                    for half in cutlass.range_constexpr(2):
                        sums = _add_e4m3x4(
                            reduced[4 * half],
                            reduced[4 * half + 1],
                            reduced[4 * half + 2],
                            reduced[4 * half + 3],
                            rank_words[source_rank, half],
                            source_scale,
                        )
                        for element in cutlass.range_constexpr(4):
                            reduced[4 * half + element] = sums[element]

            inverse, scale = _warp_quant_scale(
                _abs_max_f32x8(
                    reduced[0],
                    reduced[1],
                    reduced[2],
                    reduced[3],
                    reduced[4],
                    reduced[5],
                    reduced[6],
                    reduced[7],
                )
            )
            output_row = (
                prenorm_mailbox_multicast_address
                + (Int64(stage) * self.capacity_m + Int64(token)) * self.row_bytes
            )
            _multicast_u32x2(
                output_row + Int64(fragment) * 8,
                _e4m3x4_of_f32x4(
                    reduced[0], reduced[1], reduced[2], reduced[3], inverse
                ),
                _e4m3x4_of_f32x4(
                    reduced[4], reduced[5], reduced[6], reduced[7], inverse
                ),
            )
            if lane == 0:
                _multicast_u32(
                    output_row
                    + Int64(self.hidden_size)
                    + Int64(fragment // WARP_SIZE) * 4,
                    _f32_bits(scale),
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
                if cutlass.const_expr(source_rank != self.rank):
                    reset_row = (
                        contribution_mailbox
                        + (
                            (Int64(stage) * self.tp_size + source_rank)
                            * self.local_capacity
                            + Int64(local_token)
                        )
                        * self.row_bytes
                    )
                    _store_sentinel_u32x2(reset_row + Int64(fragment) * 8)
                    if lane == 0:
                        store_global_u32_address(
                            reset_row
                            + Int64(self.hidden_size)
                            + Int64(fragment // WARP_SIZE) * 4,
                            Uint32(E4M3_SENTINEL),
                        )


class PlpMaterializeRMSNormE4M3DeviceKernel:
    """FlashInfer's MaterializeRMSNorm on the E4M3 payload, with the residual add and
    PERF-LB's launch orders (``early_dependents``, ``poll_before_wait``) and E4M3 copy
    (``fp8_out``)."""

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
        early_dependents: bool = False,
        poll_before_wait: bool = False,
        fp8_out: bool = False,
    ) -> None:
        _check_scale_blocks(hidden_size, threads)
        self.hidden_size = hidden_size
        self.capacity_m = capacity_m
        self.threads = threads
        self.rms_epsilon = rms_epsilon
        self.weight_bias = weight_bias
        self.write_residual_output = write_residual_output
        self.enable_pdl = enable_pdl
        self.early_dependents = early_dependents
        self.poll_before_wait = poll_before_wait
        self.fp8_out = fp8_out
        self.fragments = hidden_size // VEC_BF16
        self.trips = math.ceil(self.fragments / threads)
        self.warps = threads // WARP_SIZE
        self.row_bytes = e4m3_row_bytes(hidden_size)

    @cute.jit
    def __call__(
        self,
        prenorm_mailbox: Int64,
        residual_source: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        gamma: cute.Tensor,
        stage_state: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            prenorm_mailbox,
            residual_source,
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
        prenorm_mailbox: Int64,
        residual_source: cute.Tensor,
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
        residual_words = cute.make_rmem_tensor(
            cute.make_layout((self.trips, 4), stride=(4, 1)),
            Uint32,
        )
        gamma_fragments.fill(BFloat16(0.0))
        prenorm_fragments.fill(BFloat16(0.0))
        residual_words.fill(Uint32(0))
        # The reduce kernel let this grid launch after its own wait, so the residual's
        # writer has completed (it read the residual there before its wait).
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
                residual_offset = (
                    Int64(token) * self.hidden_size + Int64(fragment) * VEC_BF16
                )
                residual_pointer = cute.make_ptr(
                    BFloat16,
                    (residual_source.iterator + residual_offset).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                residual_packed = load_global_u32x4(residual_pointer)
                for word in cutlass.range_constexpr(4):
                    residual_words[trip, word] = residual_packed[word]

        inverse_scale = Float32(0.0)
        if cutlass.const_expr(self.fp8_out):
            inverse_scale = _reciprocal_div_full(_load_global_f32(fp8_scale))

        if cutlass.const_expr(self.enable_pdl and not self.poll_before_wait):
            cute.arch.griddepcontrol_wait()

        stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
        row = (
            prenorm_mailbox
            + (Int64(stage) * self.capacity_m + Int64(token)) * self.row_bytes
        )
        thread_sum = Float32(0.0)
        for trip in cutlass.range_constexpr(self.trips):
            fragment = tidx + trip * self.threads
            if fragment < self.fragments:
                data_address = row + Int64(fragment) * 8
                scale_address = (
                    row + Int64(self.hidden_size) + Int64(fragment // WARP_SIZE) * 4
                )
                payload = _load_volatile_e4m3(data_address, scale_address)
                while _e4m3_pending(payload):
                    payload = _load_volatile_e4m3(data_address, scale_address)
                packed = _dequant_add_bf16x8(
                    payload,
                    residual_words[trip, 0],
                    residual_words[trip, 1],
                    residual_words[trip, 2],
                    residual_words[trip, 3],
                )
                prenorm = packed_u32x4_to_bf16x8(packed)
                prenorm_fragments[trip, None].store(prenorm)

                if cutlass.const_expr(self.write_residual_output):
                    output_offset = (
                        Int64(token) * self.hidden_size + Int64(fragment) * VEC_BF16
                    )
                    store_global_u32x4(
                        Int64((residual_output.iterator + output_offset).toint()),
                        packed,
                    )

                values = prenorm.to(Float32)
                thread_sum = thread_sum + (values * values).reduce(
                    cute.ReductionOp.ADD,
                    init_val=Float32(0.0),
                    reduction_profile=0,
                )

        smem = cutlass.utils.SmemAllocator()
        warp_sums = smem.allocate_array(Float32, self.warps)
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
                    quantized = cute.make_rmem_tensor(cute.make_layout((2,)), Uint32)
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
        if cutlass.const_expr(not self.early_dependents):
            cute.arch.barrier()
            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_launch_dependents()

        # _block_sum's barriers order every thread's polls before these resets.
        for trip in cutlass.range_constexpr(self.trips):
            fragment = tidx + trip * self.threads
            if fragment < self.fragments:
                _store_sentinel_u32x2(row + Int64(fragment) * 8)
                if fragment % WARP_SIZE == 0:
                    store_global_u32_address(
                        row
                        + Int64(self.hidden_size)
                        + Int64(fragment // WARP_SIZE) * 4,
                        Uint32(E4M3_SENTINEL),
                    )


class PlpMaterializeRMSNormE4M3Fp8DeviceKernel(PlpMaterializeRMSNormE4M3DeviceKernel):
    """With the E4M3 rows' address and the scale's address (an fp32 scalar)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, fp8_out=True)

    @cute.jit
    def __call__(
        self,
        prenorm_mailbox: Int64,
        residual_source: cute.Tensor,
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
            residual_source,
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
