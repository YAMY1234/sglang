"""MoE routing plus FlashInfer MegaMoE NVFP4 input staging in one launch.

At decode, route_warp, the padded-row id mask and FlashInfer's fused MegaMoE input
stager are three launches between the router GEMM and the MegaMoE kernel. This op
runs them as one: the first CTAs route one token per warp with route_warp's code
and write the routes to the router outputs and to the MegaMoE workspace; the others
quantize one BF16 row each into the workspace with the operation sequence of
FlashInfer's NVFP4 stager and mask the workspace's route tail. Routes and staged
NVFP4 rows and scales are bit-identical to the unfused chain. Wired into serving
through sglang.srt.layers.moe.route_quant_handoff.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.moe import moe_route_radix, moe_route_warp

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module(
    num_experts: int, topk: int, dtype: torch.dtype, bias_dtype: torch.dtype
) -> Module:
    args = make_cpp_args(num_experts, topk, dtype, bias_dtype, is_arch_support_pdl())
    # Default flags only, as route_warp's module: the routing half must compile to the same code.
    return load_jit(
        "moe_route_warp_nvfp4_stage",
        *args,
        cuda_files=["moe/route_warp_nvfp4_stage.cuh"],
        cuda_wrappers=[
            (
                "run",
                f"moe::route_warp_nvfp4_stage::RouteWarpNvfp4StageKernel<{args}>::run",
            )
        ],
    )


def covered(scores: torch.Tensor, bias: torch.Tensor, topk: int) -> bool:
    """Single-group sigmoid routing that moe_fused_gate serves with route_warp
    (not route_radix): row-major fp32/bf16 scores and a contiguous fp32 bias."""
    return (
        scores.dim() == 2
        and scores.stride(1) == 1
        and bias.dtype == torch.float32
        and moe_route_warp.covered(scores.dtype, scores.shape[1], topk, "sigmoid")
        and moe_route_warp.bias_layout_ok(bias)
        and not moe_route_radix.covered(scores, bias, topk)
    )


def route_nvfp4_stage(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    *,
    renormalize: bool,
    routed_scaling_factor: Optional[float],
    apply_routed_scaling_factor_on_output: bool,
    num_token_non_padded: Optional[torch.Tensor],
    x: torch.Tensor,
    x_q: torch.Tensor,
    x_sf: torch.Tensor,
    stage_ids: torch.Tensor,
    stage_weights: torch.Tensor,
    norm_const: float,
    scale_multiplier: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns the router's (weights [T, topk] fp32, ids [T, topk] int32), rows past
    ``num_token_non_padded`` as zero and -1. Also writes the workspace: ``x_q``
    [capacity, H / 2] and ``x_sf`` [capacity, H / 16] (uint8 views) rows [0, T),
    ``stage_ids`` / ``stage_weights`` [capacity, topk] rows [0, T), and -1 into
    ``stage_ids`` rows [T, capacity). ``norm_const`` and ``scale_multiplier`` are
    the fp32 values the stager receives. Caller must have checked covered()."""
    num_tokens = scores.shape[0]
    out_w = torch.empty((num_tokens, topk), dtype=torch.float32, device=scores.device)
    out_i = torch.empty((num_tokens, topk), dtype=torch.int32, device=scores.device)
    _jit_module(scores.shape[1], topk, scores.dtype, bias.dtype).run(
        scores,
        bias,
        out_w,
        out_i,
        num_token_non_padded,
        x,
        x_q,
        x_sf,
        stage_ids,
        stage_weights,
        float(1.0 if routed_scaling_factor is None else routed_scaling_factor),
        bool(renormalize),
        bool(apply_routed_scaling_factor_on_output),
        float(norm_const),
        float(scale_multiplier),
    )
    return out_w, out_i
