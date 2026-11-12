"""Source-qualified gate and NVFP4 input production into caller-owned buffers."""

from __future__ import annotations

import os

import torch

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    NumericOperation,
    RouterQuantIO,
    RouterQuantSignature,
    ScalarType,
)
from sglang.srt.layers.moe.quantized_input import Nvfp4RoutedInputPreQuant

_SOURCE_ENABLED = os.environ.get("SGLANG_Q38_ROUTER_FUSED", "0") == "1"
_SOURCE_WIDE = os.environ.get("SGLANG_Q38_ROUTER_WIDE", "0") == "1"
_SOURCE_K4 = os.environ.get("SGLANG_Q38_ROUTER_K4", "0") == "1"
_SOURCE_LAYERS = os.environ.get("SGLANG_Q38_ROUTER_LAYERS", "all")


def _source_eligible(producer):
    if not _SOURCE_ENABLED:
        return False
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import get_moe_a2a_backend, get_moe_runner_backend
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptNvFp4FusedMoEMethod,
    )
    from sglang.srt.runtime_context import get_lora

    experts = producer.experts
    weight = producer.gate.weight
    scale = getattr(experts, "w13_input_scale_quant", None)
    backend = get_moe_runner_backend()
    return (
        envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.get()
        and not producer.is_nextn
        and not get_lora().enable_lora
        and get_moe_a2a_backend().is_none()
        and type(experts) is FusedMoE
        and isinstance(experts.quant_method, ModelOptNvFp4FusedMoEMethod)
        and backend is not None
        and backend.is_flashinfer_trtllm()
        and not experts.quant_method.quant_config.use_per_token_activation
        and getattr(experts, "trtllm_padded_hidden_size", None) is None
        and not producer.enable_shared_expert_fusion
        and weight.is_cuda
        and weight.dtype == torch.bfloat16
        and weight.ndim == 2
        and all(size > 0 for size in weight.shape)
        and weight.is_contiguous()
        and scale is not None
        and scale.is_cuda
        and scale.dtype == torch.float32
        and scale.shape in ((), (1,))
        and scale.is_contiguous()
        and scale.device == weight.device
        and (
            _SOURCE_LAYERS == "all"
            or producer.layer_id in frozenset(int(v) for v in _SOURCE_LAYERS.split(","))
        )
    )


def _signature(producer, rows, variant):
    return RouterQuantSignature(
        NumericOperation.ROUTER_QUANT,
        rows,
        producer.gate.weight.shape[1],
        producer.gate.weight.shape[0],
        ScalarType.BF16,
        variant,
    )


def numeric_signatures(producer, widths):
    """Publish actual dimensions only when the selected provider accepts them."""
    from sglang.srt.kernel_plugins.nvidia.numeric_binding import (
        signature_capability_supported,
    )
    from sglang.srt.runtime_context import get_parallel

    signatures = (
        tuple(
            signature
            for rows in widths
            if type(rows) is int and rows > 0
            for variant in (0, 1, 2)
            if signature_capability_supported(
                signature := _signature(producer, rows, variant),
                spec=getattr(producer, "_numeric_capability_spec", None),
                tp_size=producer.tp_size,
                ep_size=get_parallel().moe_ep_size,
                wide_enabled=int(_SOURCE_WIDE),
                k4_enabled=int(_SOURCE_K4),
            )
        )
        if _source_eligible(producer)
        else ()
    )
    producer._numeric_router_quant_signatures = frozenset(signatures)
    return signatures


def produce(producer, hidden):
    """Run before the caller's existing side-stream event, or leave gate stock."""
    from sglang.srt.kernel_plugins.nvidia.numeric_binding import numeric_operator

    if not _SOURCE_ENABLED:
        return None
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        is_in_tc_piecewise_cuda_graph,
    )

    if (
        not _SOURCE_ENABLED
        or hidden.ndim != 2
        or torch.compiler.is_compiling()
        or is_in_tc_piecewise_cuda_graph()
    ):
        return None
    rows = hidden.shape[0]
    declared = getattr(producer, "_numeric_router_quant_signatures", ())
    signature = next((s for s in declared if s.rows == rows), None)
    if signature is None or not _source_eligible(producer):
        return None
    weight = producer.gate.weight
    scale = producer.experts.w13_input_scale_quant
    if scale.ndim == 0:
        scale = scale.view(1)
    if not (
        hidden.shape == (rows, signature.hidden_size)
        and hidden.dtype == torch.bfloat16
        and hidden.is_cuda
        and hidden.device == weight.device
        and hidden.is_contiguous()
        and hidden.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
        and scale.data_ptr() % 4 == 0
    ):
        return None
    run = numeric_operator(signature, prepared=True)
    if run is None:
        return None

    # Match the source wrapper's per-call outputs, including graph private pools.
    logits = torch.empty(
        (rows, signature.expert_count), dtype=torch.bfloat16, device=hidden.device
    )
    packed = torch.empty(
        (rows, signature.hidden_size // 2), dtype=torch.uint8, device=hidden.device
    )
    scales = torch.empty(
        (rows, signature.hidden_size // 16),
        dtype=torch.float8_e4m3fn,
        device=hidden.device,
    )
    run(RouterQuantIO(hidden, weight, scale, logits, packed, scales))
    return logits, Nvfp4RoutedInputPreQuant(packed, scales)
