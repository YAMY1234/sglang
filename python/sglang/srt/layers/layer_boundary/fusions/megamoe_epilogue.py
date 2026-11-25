"""FlashInfer MegaMOE epilogue handoff, consumed at the next layer's input norm.

A MegaMOE layer on the SM107 kernel with the fused epilogue hands its reduced
routed rows and its shared-expert output to the next attention input, whose one
kernel adds them, adds the residual and applies the input RMSNorm. Every other
consumer completes the handoff with the unfused shared add first.
"""

from __future__ import annotations

import functools
import logging

import torch

from sglang.kernels.fused_op import get_fused_op_backend
from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.layers.layer_boundary.output import DeferredFinalize
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class MegaMoeEpilogueHandoff(DeferredFinalize, frozen=True):
    """``routed``: [T, H] BF16, the MegaMOE top-k reduce output this layer owns.
    ``shared``: [T, H], the shared-expert output."""

    routed: torch.Tensor
    shared: torch.Tensor

    def complete(self) -> torch.Tensor:
        # The unfused layer's in-place shared add.
        return self.routed.add_(self.shared)


class MegaMoeEpilogueFusion:
    """One layer's stage fusion. Its FFN exit may hand off a decode or verify
    batch's MegaMOE output, except at the stack's end; its attention input
    takes such a handoff in the fused kernel when the norm matches it."""

    def __init__(self, *, defers_finalize: bool) -> None:
        self.defers_finalize = defers_finalize

    def attn_input_fusions(self, plan) -> tuple:
        return (functools.partial(_shared_add_norm, plan),)

    def ffn_input_fusions(self, plan) -> tuple:
        return ()

    def can_defer_all_reduce(self, plan, forward_batch: ForwardBatch) -> bool:
        return False

    def can_defer_finalize(self, plan, forward_batch: ForwardBatch) -> bool:
        mode = forward_batch.forward_mode
        return (
            self.defers_finalize
            and not plan.terminal
            and (mode.is_decode() or mode.is_target_verify())
        )


def _shared_add_norm(plan, owed, residual, forward_batch, post_residual_addition):
    """Run the fused kernel; None (nothing touched) when it does not take the
    batch, so the handoff completes unfused and the ordinary add + norm runs."""
    from sglang.kernels.ops.moe.megamoe_epilogue import (
        can_use_megamoe_shared_add_rmsnorm,
        megamoe_shared_add_rmsnorm,
    )

    if not isinstance(owed, MegaMoeEpilogueHandoff):
        return None
    norm = plan.norm
    if not (
        post_residual_addition is None
        and _matches_fused_norm(norm)
        and can_use_megamoe_shared_add_rmsnorm(
            owed.routed, owed.shared, residual, norm.weight
        )
    ):
        _log_once("completed unfused: the input norm or operands do not match it")
        return None
    _log_once("shared add + residual add + RMSNorm in one kernel")
    hidden_states = megamoe_shared_add_rmsnorm(
        owed.routed, owed.shared, residual, norm.weight, norm.variance_epsilon
    )
    return hidden_states, residual


def _matches_fused_norm(norm) -> bool:
    """Whether ``norm(x, residual)`` on BF16 runs RMSNorm's JIT fused add, whose
    rounding the fused kernel reproduces."""
    return (
        type(norm) is RMSNorm
        and norm.has_weight
        and not norm.cast_x_before_out_mul
        and not norm.fp32_residual
        and norm.variance_size_override is None
        and norm.override_orig_dtype is None
        and norm._quant_linear is None
        and norm.use_jit_rmsnorm
        and norm.weight.dtype == torch.bfloat16
        and not norm.is_torch_compile
        and get_fused_op_backend() is None
        and not is_batch_invariant_mode_enabled()
    )


@functools.cache
def _log_once(what: str) -> None:
    logger.info("FlashInfer MegaMOE fused epilogue: %s", what)
