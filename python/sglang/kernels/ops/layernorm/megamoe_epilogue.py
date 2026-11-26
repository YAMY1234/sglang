"""MegaMoE epilogue at the next layer's input: shared-expert add, residual add, RMSNorm.

``megamoe_shared_add_rmsnorm`` takes a MegaMoE layer's reduced routed output and
its shared-expert output, adds them as ``routed += shared`` does, then runs the
next layer's residual add and input RMSNorm in the arithmetic, reduction order
and schedule of RMSNorm's JIT fused add (``elementwise/rmsnorm.cuh``, FUSED_ADD),
so the residual and the normalized rows equal the unfused chain's bit for bit.
One CTA per token, one 32-byte vector per thread
(``kernels/jit/csrc/elementwise/megamoe_epilogue.cuh``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.layernorm.norm import _schedule_rmsnorm

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# One 32-byte BF16 vector per thread.
_VEC_ELEMS = 16


@cache_once
def _jit_module(hidden: int) -> Module:
    args = make_cpp_args(hidden, is_arch_support_pdl())
    return load_jit(
        "megamoe_epilogue",
        *args,
        cuda_files=["elementwise/megamoe_epilogue.cuh"],
        cuda_wrappers=[("megamoe_epilogue", f"MegaMoeEpilogueKernel<{args}>::run")],
    )


@cache_once
def _jit_fused_add_max_tokens(hidden: int) -> Optional[int]:
    """The token count up to which RMSNorm's JIT fused add on BF16 rows of this
    size runs the schedule this kernel reproduces (32-byte vectors, one row per
    CTA, one vector per thread, whole warps), or None if it never does."""
    decode, _, threshold = _schedule_rmsnorm(hidden, 2, True)
    threads = hidden // _VEC_ELEMS
    reproduced = (
        decode.vec_bytes == 2 * _VEC_ELEMS
        and decode.rows_per_cta == 1
        and decode.threads_per_row == threads
        and hidden % _VEC_ELEMS == 0
        and threads % 32 == 0
        and threads <= 1024
    )
    return threshold if reproduced else None


def _rows_aligned(*tensors: torch.Tensor) -> bool:
    return all(
        t.dtype == torch.bfloat16
        and t.stride(-1) == 1
        and t.data_ptr() % (2 * _VEC_ELEMS) == 0
        and all(s % _VEC_ELEMS == 0 for s in t.stride()[:-1])
        for t in tensors
    )


def can_use_megamoe_shared_add_rmsnorm(
    routed: torch.Tensor,
    shared: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    """Whether the fused kernel takes these operands."""
    if routed.dim() != 2:
        return False
    num_tokens, hidden = routed.shape
    max_tokens = _jit_fused_add_max_tokens(hidden)
    return (
        max_tokens is not None
        and 0 < num_tokens <= max_tokens
        and shared.shape == residual.shape == (num_tokens, hidden)
        and weight.shape == (hidden,)
        and weight.is_contiguous()
        and _rows_aligned(routed, shared, residual, weight)
    )


def megamoe_shared_add_rmsnorm(
    routed: torch.Tensor,
    shared: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """``x = routed + shared``, then the fused add + RMSNorm: ``residual += x`` in
    place, and the normalized rows are returned."""
    num_tokens, hidden = routed.shape
    out = residual.new_empty((num_tokens, hidden))
    _jit_module(hidden).megamoe_epilogue(routed, shared, residual, weight, out, eps)
    return out
