"""ROUTE_TAIL: one router/quant/metadata grid and the original native body."""
import logging

import torch

from .native_body import NativeFp4Body
from .resolve_native import resolve_native_tactic

logger = logging.getLogger(__name__)


def _native_kwargs(mlp, front, defer_finalize):
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        get_activation_type,
        trtllm_moe_enable_pdl,
    )
    from sglang.srt.layers.moe.utils import RoutingMethodType

    experts = mlp.experts
    cfg = experts.quant_method.moe_runner_config
    return dict(
        topk_ids=(front.ids, front.weights), routing_bias=None,
        hidden_states=front.packed,
        hidden_states_scale=front.sf.view(torch.float8_e4m3fn),
        gemm1_weights=experts.w13_weight.data,
        gemm1_weights_scale=experts.w13_weight_scale.data.view(torch.float8_e4m3fn),
        gemm1_bias=None, gemm1_alpha=getattr(experts, "gemm1_alpha", None),
        gemm1_beta=getattr(experts, "gemm1_beta", None),
        gemm1_clamp_limit=getattr(experts, "gemm1_clamp_limit", None),
        gemm2_weights=experts.w2_weight.data,
        gemm2_weights_scale=experts.w2_weight_scale.data.view(torch.float8_e4m3fn),
        gemm2_bias=None, output1_scale_scalar=experts.g1_scale_c.data,
        output1_scale_gate_scalar=experts.g1_alphas.data,
        output2_scale_scalar=experts.g2_alphas.data,
        per_token_scale=None, num_experts=experts.num_experts, top_k=10,
        n_group=0, topk_group=0,
        intermediate_size=experts.intermediate_size_per_partition,
        local_expert_offset=0, local_num_experts=experts.num_local_experts,
        routed_scaling_factor=None,
        routing_method_type=getattr(experts, "routing_method_type", RoutingMethodType.Default),
        do_finalize=not defer_finalize,
        activation_type=get_activation_type(cfg.activation, is_gated=cfg.is_gated),
        enable_pdl=trtllm_moe_enable_pdl(front.m),
        tune_max_num_tokens=front.m, norm_topk_prob=True,
        num_fused_shared_experts=0,
    )


def routed(mlp, x, *, defer_finalize=False):
    from .front import Front

    if x.shape[0] != 4:
        raise ValueError("ROUTE_TAIL v2 is M4 only; M8/M12 use the parent path")
    experts = mlp.experts
    if experts.trtllm_padded_hidden_size is not None:
        raise ValueError("ROUTE_TAIL keeps the original unpadded H2560 prepared weights")
    if mlp.is_nextn or tuple(mlp.gate.weight.shape) != (512, 2560):
        raise ValueError("ROUTE_TAIL is target FN NVFP4 only")
    cfg = mlp.topk.topk_config
    if not (cfg.top_k == 10 and cfg.renormalize and cfg.scoring_func == "softmax"
            and cfg.correction_bias is None and not cfg.use_grouped_topk):
        raise ValueError("ROUTE_TAIL requires native FN top10 renormalized softmax")
    if not hasattr(mlp, "_q38c_route_tail_bodies"):
        mlp._q38c_route_tail_bodies = {}
    key = (x.shape[0], bool(defer_finalize))
    pair = mlp._q38c_route_tail_bodies.get(key)
    gs = experts.w13_input_scale_quant.reshape(1)
    if pair is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("ROUTE_TAIL must prepare its native body during graph warmup")
        front = Front(x, mlp.gate.weight, gs)
        kwargs = _native_kwargs(mlp, front, defer_finalize)
        tactic = resolve_native_tactic(kwargs, routing_logits=front.logits)
        body = NativeFp4Body(kwargs, tactic=tactic)
        # The native TMA gather ABI includes a guard row and histogram scratch.
        # Initialize unused capacity once; every live map/pad is rewritten by tail.
        body.slot.permuted_idx_to_token_idx.fill_(-1)
        body.slot.expert_count_histogram.zero_()
        front.prepare(body.metadata, body.tactic[0], x, mlp.gate.weight, gs)
        pair = (front, body)
        mlp._q38c_route_tail_bodies[key] = pair
        logger.info("Q38C_ROUTE_TAIL layer=%s M=%s tactic=%s defer=%s native body",
                    mlp.layer_id, x.shape[0], tactic, defer_finalize)
    front, body = pair
    front.run(x, mlp.gate.weight, gs)
    scales = front.sf.view(torch.float8_e4m3fn)
    return (body.run_sglang_deferred(front.packed, scales) if defer_finalize
            else body.run(front.packed, scales))
