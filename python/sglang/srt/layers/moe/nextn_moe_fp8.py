"""The NextN (MTP / EAGLE draft) MoE on an FP8 copy of its BF16 routed experts.

``SGLANG_NEXTN_MOE_FP8`` (default off) makes the draft layer's routed experts run on the
FlashInfer TRT-LLM FP8 MoE instead of the TRT-LLM BF16 MoE, for checkpoints whose NextN
experts are BF16 (e.g. the GLM-5.2 / DeepSeek NVFP4 checkpoints, whose MTP layer is not
quantized). At low row counts that MoE streams its expert weights from HBM, so the FP8 copy
halves the bytes the draft step reads per pass.

* ``mxfp8`` (or ``1``): E4M3 weights and activations with one UE8M0 scale per 32 along K
  (``Fp8QuantizationType.MxFp8``); the faster one at 1-8 rows on GB300 (T90a).
* ``block``: DeepSeek-style FP8, E4M3 weights with one fp32 scale per 128 x 128
  block, activations quantized per 1 x 128 group (``trtllm_fp8_block_scale_moe``,
  ``Fp8QuantizationType.DeepSeekFp8``).

The router, its top-k (the same routing logits, correction bias, groups and routed scaling
factor are handed to the same TRT-LLM routing), the shared expert and everything else stay
as they are; only the routed experts' GEMMs change, so the draft's proposals may change
(acceptance), never the verified tokens. The copy is made from the canonical BF16 weights at
``process_weights_after_loading`` (before the BF16 block layout); the BF16 weights stay (the
copy adds half their size). Served on the FlashInfer TRT-LLM MoE runner only.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)

ENV = "SGLANG_NEXTN_MOE_FP8"
# The draft MoE's finalize + shared-expert add as SGLang's deferred finalize (one fused kernel
# or the next fused collective) instead of the TRT-LLM finalize and a separate add.
DEFER_ENV = "SGLANG_NEXTN_MOE_DEFER_FINALIZE"
FP8_MAX = 448.0
BLOCK = 128
MX_BLOCK = 32
# the attribute the NextN model sets on its experts layer (FusedMoE) to request the copy,
# and the one holding the built copy (a plain attribute: not a registered submodule)
REQUEST_ATTR = "nextn_moe_fp8_mode"
COPY_ATTR = "_nextn_moe_fp8"


def nextn_moe_fp8_mode() -> Optional[str]:
    """``block`` / ``mxfp8`` from ``SGLANG_NEXTN_MOE_FP8``, or None (off)."""
    v = os.environ.get(ENV, "").strip().lower()
    if v in ("", "0", "off", "false", "none", "bf16"):
        return None
    if v in ("1", "on", "true", "mxfp8"):
        return "mxfp8"
    if v in ("block", "fp8"):
        return "block"
    raise ValueError(f"{ENV}={v!r}: expected block, mxfp8 or unset")


def defer_finalize_requested() -> bool:
    return os.environ.get(DEFER_ENV, "").strip().lower() in ("1", "on", "true")


def unsupported_reason(experts: Module, dtype_check: bool = True) -> Optional[str]:
    """Why ``experts`` (a FusedMoE) cannot take the FP8 copy (or, without ``dtype_check``,
    the deferred finalize), or None."""
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    qm = getattr(experts, "quant_method", None)
    if not isinstance(qm, UnquantizedFusedMoEMethod):
        return f"quant method {type(qm).__name__} is not the unquantized MoE method"
    if not qm.use_flashinfer_trtllm_moe:
        return "the MoE runner is not flashinfer_trtllm"
    runner = getattr(qm, "runner", None)
    backend = getattr(runner, "runner_backend", None)
    if backend is None or not backend.is_flashinfer_trtllm():
        return "the MoE runner is not flashinfer_trtllm (routed runners are not served)"
    cfg = experts.moe_runner_config
    if not cfg.is_gated or cfg.activation != "silu":
        return f"activation {cfg.activation} (gated={cfg.is_gated}) is not SwiGLU"
    if getattr(qm, "with_bias", False):
        return "expert biases are not served"
    if cfg.num_fused_shared_experts:
        return "fused shared experts are not served"
    if cfg.routing_method_type is None:
        return "no routing method type"
    if not dtype_check:
        return None
    if experts.w13_weight.dtype != torch.bfloat16:
        return f"expert weights are {experts.w13_weight.dtype}, not bfloat16"
    hidden, inter = experts.hidden_size, experts.intermediate_size_per_partition
    if hidden % BLOCK or inter % BLOCK:
        return f"hidden {hidden} / intermediate {inter} are not multiples of {BLOCK}"
    return None


def quantize_block_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``w`` [E, N, K] BF16 -> (E4M3 [E, N, K], fp32 dequant scales [E, N/128, K/128]):
    one scale per 128 x 128 block, amax / 448 (the DeepSeek FP8 ``weight_scale_inv``)."""
    e, n, k = w.shape
    assert n % BLOCK == 0 and k % BLOCK == 0, (n, k)
    q = torch.empty(e, n, k, dtype=torch.float8_e4m3fn, device=w.device)
    s = torch.empty(e, n // BLOCK, k // BLOCK, dtype=torch.float32, device=w.device)
    for i in range(e):  # one expert at a time: fp32 temporaries of one expert only
        x = w[i].float().view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
        amax = x.abs().amax(dim=(1, 3))
        scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
        q[i] = (
            (x / scale[:, None, :, None])
            .clamp_(-FP8_MAX, FP8_MAX)
            .to(torch.float8_e4m3fn)
            .view(n, k)
        )
        s[i] = scale
    return q, s


def quantize_mxfp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``w`` [E, N, K] BF16 -> (E4M3 [E, N, K], UE8M0 scales uint8 [E, N, K/32]): one
    power-of-two scale per 32 along K, rounded up so that amax / scale <= 448."""
    e, n, k = w.shape
    assert k % MX_BLOCK == 0, k
    q = torch.empty(e, n, k, dtype=torch.float8_e4m3fn, device=w.device)
    s = torch.empty(e, n, k // MX_BLOCK, dtype=torch.uint8, device=w.device)
    for i in range(e):
        x = w[i].float().view(n, k // MX_BLOCK, MX_BLOCK)
        amax = x.abs().amax(dim=-1)
        exp = torch.ceil(torch.log2(amax / FP8_MAX))
        exp = torch.where(amax > 0, exp, torch.full_like(exp, -127.0)).clamp_(-127, 127)
        q[i] = (
            (x / torch.exp2(exp)[..., None])
            .clamp_(-FP8_MAX, FP8_MAX)
            .to(torch.float8_e4m3fn)
            .view(n, k)
        )
        s[i] = (exp + 127).to(torch.uint8)
    return q, s


class NextnMoeFp8Copy(Module):
    """The FP8 copy of one experts layer's weights (and scales) in the runner's layout."""

    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode


def build_copy(layer: Module, mode: str) -> NextnMoeFp8Copy:
    """The FP8 copy of ``layer``'s canonical BF16 ``w13_weight`` ([gate; up] rows) and
    ``w2_weight``, prepared for the FlashInfer TRT-LLM FP8 MoE."""
    w13, w2 = layer.w13_weight.data, layer.w2_weight.data
    copy = NextnMoeFp8Copy(mode)
    if mode == "block":
        w13_q, w13_s = quantize_block_fp8(w13)
        w2_q, w2_s = quantize_block_fp8(w2)
        # DeepSeekFp8 runs on the canonical (MajorK, unshuffled) layout, as the FP8
        # block-quant checkpoints do (fp8.py: process_weights_after_loading_block_quant)
        for name, t in (
            ("w13_weight", w13_q),
            ("w2_weight", w2_q),
            ("w13_weight_scale_inv", w13_s),
            ("w2_weight_scale_inv", w2_s),
        ):
            copy.register_parameter(name, Parameter(t, requires_grad=False))
    elif mode == "mxfp8":
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            align_mxfp8_moe_weights_for_flashinfer_trtllm,
        )

        w13_q, w13_s = quantize_mxfp8(w13)
        w2_q, w2_s = quantize_mxfp8(w2)
        for name, t in (
            ("w13_weight", w13_q),
            ("w2_weight", w2_q),
            ("w13_weight_scale_inv", w13_s),
            ("w2_weight_scale_inv", w2_s),
        ):
            copy.register_parameter(name, Parameter(t, requires_grad=False))
        # the gated row reorder, the row shuffles and the scale interleave of the MXFP8
        # checkpoints (fp8.py: _process_mxfp8_moe_weights)
        copy.moe_runner_config = layer.moe_runner_config
        align_mxfp8_moe_weights_for_flashinfer_trtllm(copy)
    else:
        raise ValueError(mode)
    return copy


def maybe_build(layer: Module) -> None:
    """At ``process_weights_after_loading`` (canonical BF16 weights): build the FP8 copy
    when the NextN model requested it on this layer."""
    mode = getattr(layer, REQUEST_ATTR, None)
    if not mode or getattr(layer, COPY_ATTR, None) is not None:
        return
    why = unsupported_reason(layer)
    if why:
        logger.warning("%s=%s declined: %s; the draft MoE stays BF16", ENV, mode, why)
        return
    copy = build_copy(layer, mode)
    layer.__dict__[COPY_ATTR] = copy  # not a registered submodule
    torch.cuda.empty_cache()
    mib = sum(p.numel() * p.element_size() for p in copy.parameters()) / 2**20
    logger.info(
        "%s=%s: draft MoE routed experts on an FP8 copy (%s, %d experts, copy %.1f MiB); "
        "FlashInfer TRT-LLM FP8 MoE",
        ENV,
        mode,
        "E4M3 + fp32 scale per 128x128" if mode == "block" else "MXFP8 E4M3 + UE8M0 per 32",
        layer.num_local_experts,
        mib,
    )


def quant_info(layer: Module):
    """The FP8 MoE payload of ``layer``'s copy, or None (no copy: the BF16 MoE)."""
    copy = getattr(layer, COPY_ATTR, None)
    if copy is None:
        return None
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmFp8MoeQuantInfo,
        get_activation_type,
    )

    cfg = layer.moe_runner_config
    return FlashInferTrtllmFp8MoeQuantInfo(
        w13_weight=copy.w13_weight,
        w2_weight=copy.w2_weight,
        global_num_experts=layer.num_experts,
        local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
        local_num_experts=layer.num_local_experts,
        intermediate_size=copy.w2_weight.shape[2],
        # the BF16 MoE's routing (runner_config.routing_method_type, as
        # fused_experts_none_to_flashinfer_trtllm_bf16 passes it)
        routing_method_type=int(cfg.routing_method_type),
        block_quant=True,
        use_mxfp8=copy.mode == "mxfp8",
        weight_block_k=BLOCK if copy.mode == "block" else MX_BLOCK,
        w13_weight_scale_inv=copy.w13_weight_scale_inv,
        w2_weight_scale_inv=copy.w2_weight_scale_inv,
        activation_type=get_activation_type(cfg.activation, is_gated=cfg.is_gated),
    )


def run(layer: Module, runner, dispatch_output):
    """The routed experts of ``layer`` on its FP8 copy through ``runner`` (the layer's
    FlashInfer TRT-LLM runner), or None (no copy: the BF16 MoE runs)."""
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    info = quant_info(layer)
    if info is None:
        return None
    topk = dispatch_output.topk_output
    # The BF16 MoE routes every row from the logits (padded rows included) and ignores
    # num_token_non_padded; the FP8 runner would materialize the top-k for it instead.
    if (
        TopKOutputChecker.format_is_bypassed(topk)
        and topk.num_token_non_padded is not None
        and topk.expert_location_dispatch_info is None
    ):
        dispatch_output = dispatch_output._replace(
            topk_output=topk._replace(num_token_non_padded=None)
        )
    return runner.run(dispatch_output, info)
