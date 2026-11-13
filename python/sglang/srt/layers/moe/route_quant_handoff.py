"""Attempt-and-verify handoff for the fused K3 MoE-front prep launch.

At decode batch sizes the chain between the K3 fused-front GEMM and the
trtllm-gen SiTU MoE op is three tiny back-to-back kernels on the critical
path — route_radix (top-16 of 896), the triton ``(id << 16) | bf16(weight)``
pack, and per_token_group_quant (mxfp8, ``[T, 3584]``) — about 7.5 us busy
plus two extra launches per MoE layer, each leaving the SMs near idle. The
fused kernel (kernels/ops/moe/moe_route_quant_fused.py) runs all three in one
launch: routing CTAs and one quant CTA per token, concurrently.

The inputs live in different modules (router logits reach the router through
TopK, activations through the MoE runner), so the fusion is wired as a
consume-once stash instead of new signatures:

    KimiK3MoE._forward_routed*      stage(x) before self.topk, clear() after
                                    the experts call
    biased_grouped_topk_gpu         try_route_quant_fused() replaces the
                                    moe_fused_gate call on a hit
    Mxfp4MoEMethod.apply (situ)     take(x) skips the quant and the pack

FlashInfer's SM107 MegaMoE takes the same bracket from DeepseekV2MoE
(SGLANG_OPT_FLASHINFER_MEGAMOE_FUSED_ROUTE_QUANT): the single-group router
calls try_route_stage_megamoe(), which routes and writes the NVFP4 inputs and
routes straight into the layer's MegaMoE workspace, and FlashInfer's input
staging skips its launch for that workspace (take_megamoe_staged).

Every step is fallback-safe: if the routing dispatch never consumes the staged
request (different model, uncovered shape, triton fallback) or the runner's
``take`` misses (activations re-viewed or copied), the unfused chain runs as
before. The stage/clear bracket in the model layer guarantees a published
entry can never leak into another layer whose allocator reused the same
activation address.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import msgspec
import torch


class _Handoff(msgspec.Struct):
    # staged by the model layer: the activation rows the runner will quantize
    request_x: Optional[torch.Tensor] = None
    request_layer: Optional[Any] = None
    num_token_non_padded: Optional[torch.Tensor] = None
    fused_routing_ok: bool = False
    padded_rows_masked: bool = False
    # published by the routing dispatch, keyed by the staged activations
    produced_x: Optional[torch.Tensor] = None
    packed: Optional[torch.Tensor] = None
    prepared_ids: Optional[torch.Tensor] = None
    routing_metadata: Optional[Any] = None
    da_body: Optional[Any] = None
    x_q: Optional[torch.Tensor] = None
    x_s: Optional[torch.Tensor] = None
    # published by the MegaMoE route + stage launch: what it wrote into which workspace
    megamoe_x: Optional[torch.Tensor] = None
    megamoe_ids: Optional[torch.Tensor] = None
    megamoe_workspace: Optional[Any] = None


_handoff = _Handoff()
_megamoe_route_stage_logged = False


def stage(x: torch.Tensor, layer: Optional[Any] = None) -> None:
    """Publish the routed activations for the upcoming topk call. Caller pairs
    this with clear() after the experts call (try/finally)."""
    clear()
    _handoff.request_x = x
    _handoff.request_layer = layer


def clear() -> None:
    global _handoff
    _handoff = _Handoff()


def bind_select_experts(
    num_token_non_padded: Optional[torch.Tensor], fused_routing_ok: bool
) -> None:
    if _handoff.request_x is None:
        return
    _handoff.num_token_non_padded = num_token_non_padded
    _handoff.fused_routing_ok = fused_routing_ok


def take_padded_rows_masked() -> bool:
    masked = _handoff.padded_rows_masked
    _handoff.padded_rows_masked = False
    return masked


def try_route_quant_fused(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    num_fused_shared_experts: int,
    renormalize: bool,
    routed_scaling_factor: Optional[float],
    apply_routed_scaling_factor_on_output: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fused replacement for the ungrouped-sigmoid moe_fused_gate call when a
    staged request covers it. Returns (weights, ids) on a hit, None otherwise
    (caller falls through to the unfused router)."""
    x = _handoff.request_x
    if x is None or num_fused_shared_experts != 0:
        return None
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    from sglang.kernels.ops.moe import kimi_k3_fused_routing, moe_route_quant_fused

    layer = _handoff.request_layer
    if (
        layer is not None
        and _handoff.fused_routing_ok
        and kimi_k3_fused_routing.covered(gating_output, correction_bias, topk, x)
        and kimi_k3_fused_routing.available()
    ):
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            select_trtllm_routed_body,
        )

        weights_dtype = kimi_k3_fused_routing.expert_weights_dtype(x.shape[0])
        body = select_trtllm_routed_body(layer, x, topk, weights_dtype)
        if body is not None:
            routed = kimi_k3_fused_routing.fused_routing(
                gating_output,
                correction_bias,
                x,
                topk,
                renormalize=renormalize,
                routed_scaling_factor=routed_scaling_factor,
                apply_scale=apply_routed_scaling_factor_on_output,
                num_token_non_padded=_handoff.num_token_non_padded,
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
                num_local_experts=layer.num_local_experts,
                tile_tokens=body.tile_n,
                weights_dtype=weights_dtype,
            )
            _handoff.padded_rows_masked = True
            _handoff.prepared_ids = routed.ids
            _handoff.routing_metadata = routed.metadata
            _handoff.da_body = body
            return _publish(
                x, routed.weights, routed.ids, routed.ids, routed.x_q, routed.x_s
            )

    if (
        not moe_route_quant_fused.covered(gating_output, correction_bias, topk, x)
        or not moe_route_quant_fused.available()
    ):
        return None

    weights, ids, packed, x_q, x_s = moe_route_quant_fused.route_quant_fused(
        gating_output,
        correction_bias,
        x,
        topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_scale=apply_routed_scaling_factor_on_output,
    )
    return _publish(x, weights, ids, packed, x_q, x_s)


def _publish(x, weights, ids, packed, x_q, x_s):
    _handoff.request_x = None
    _handoff.produced_x = x
    _handoff.packed = packed
    _handoff.x_q = x_q
    _handoff.x_s = x_s
    return weights, ids


def try_route_stage_megamoe(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    renormalize: bool,
    routed_scaling_factor: Optional[float],
    apply_routed_scaling_factor_on_output: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Single-group sigmoid routing of the staged FlashInfer MegaMoE layer with its
    NVFP4 input staging in the same launch. Returns (weights, ids) on a hit, with
    padded rows masked; None otherwise (caller runs the unfused router)."""
    x = _handoff.request_x
    if x is None or x is not hidden_states or not _handoff.fused_routing_ok:
        return None

    from sglang.kernels.ops.moe import moe_route_nvfp4_stage
    from sglang.srt.layers.moe.flashinfer_megamoe import flashinfer_megamoe_input_stage

    bias = correction_bias.to(torch.float32)
    if not moe_route_nvfp4_stage.covered(gating_output, bias, topk):
        return None
    stage = flashinfer_megamoe_input_stage(_handoff.request_layer, x)
    if stage is None:
        return None
    weights, ids = moe_route_nvfp4_stage.route_nvfp4_stage(
        gating_output,
        bias,
        topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
        num_token_non_padded=_handoff.num_token_non_padded,
        x=x,
        x_q=stage.x_q,
        x_sf=stage.x_sf,
        stage_ids=stage.topk_idx,
        stage_weights=stage.topk_weights,
        norm_const=stage.norm_const,
        scale_multiplier=stage.scale_multiplier,
    )
    _handoff.request_x = None
    _handoff.padded_rows_masked = True
    _handoff.megamoe_x = x
    _handoff.megamoe_ids = ids
    _handoff.megamoe_workspace = stage.workspace
    global _megamoe_route_stage_logged
    if not _megamoe_route_stage_logged:
        _megamoe_route_stage_logged = True
        from sglang.srt.utils import print_info_once

        print_info_once(
            "FlashInfer MegaMOE: routing and NVFP4 input staging run as one launch "
            f"(first: {x.shape[0]} tokens, workspace capacity {stage.topk_idx.shape[0]})."
        )
    return weights, ids


def take_megamoe_staged(
    x: torch.Tensor, topk_ids: torch.Tensor, workspace: Any
) -> bool:
    """Consume the route + stage launch's record: True when it staged these exact
    rows and routes into ``workspace``. Any call ends the record."""
    staged_x, staged_ids = _handoff.megamoe_x, _handoff.megamoe_ids
    hit = (
        staged_x is not None
        and _handoff.megamoe_workspace is workspace
        and staged_x.data_ptr() == x.data_ptr()
        and staged_x.shape == x.shape
        and staged_x.stride() == x.stride()
        and staged_ids.data_ptr() == topk_ids.data_ptr()
        and staged_ids.shape == topk_ids.shape
    )
    _handoff.megamoe_x = None
    _handoff.megamoe_ids = None
    _handoff.megamoe_workspace = None
    if staged_x is not None and not hit:
        from sglang.srt.utils import print_warning_once

        print_warning_once(
            "FlashInfer MegaMOE: the router staged inputs this forward does not use "
            "(other rows, routes or workspace); FlashInfer stages them again."
        )
    return hit


def take_prepared_routing(ids: torch.Tensor) -> Optional[Tuple[Any, Any]]:
    if _handoff.prepared_ids is not ids:
        return None
    out = (_handoff.routing_metadata, _handoff.da_body)
    _handoff.prepared_ids = None
    _handoff.routing_metadata = None
    _handoff.da_body = None
    return out


def take(
    x: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Consume the published (packed_topk, x_q, x_s int32) for these exact
    activation rows, or None. Storage identity is verified so a re-viewed or
    copied tensor simply misses."""
    produced = _handoff.produced_x
    if produced is None:
        return None
    if (
        produced.data_ptr() != x.data_ptr()
        or produced.shape != x.shape
        or produced.dtype != x.dtype
        or produced.stride() != x.stride()
    ):
        return None
    out = (_handoff.packed, _handoff.x_q, _handoff.x_s)
    _handoff.produced_x = None
    _handoff.packed = None
    _handoff.x_q = None
    _handoff.x_s = None
    return out
