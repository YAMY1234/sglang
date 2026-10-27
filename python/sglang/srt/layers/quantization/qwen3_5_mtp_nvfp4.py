"""Opt-in NVFP4 routed experts for the Qwen3.5 MTP draft.

The explicit ``qwen3_5_mtp_nvfp4`` draft quantization uses the calibrated
TensorRT-LLM activation scales and retains a BF16 expert twin for the plugin's
one- and two-token fused path. Other draft modules remain in their source dtype.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.layers.quantization.nvfp4_online import (
    ModelOptNvFp4OnlineFusedMoEMethod,
    _ModelOptFp4OnlineConfig,
)

logger = logging.getLogger(__name__)

NVFP4_BLOCK_SIZE = 16
NVFP4_GLOBAL_RANGE = 6.0 * 448.0
E2M1_MAX = 6.0
FC1_INPUT_AMAX = 256.0
FC2_INPUT_AMAX = 2048.0


def _e2m1_codes(magnitude: torch.Tensor) -> torch.Tensor:
    """Round magnitudes in [0, 6] to E2M1 codes with round-to-nearest-even."""
    code = (magnitude > 0.25).to(torch.uint8)
    for threshold, inclusive in (
        (0.75, True),
        (1.25, False),
        (1.75, True),
        (2.5, False),
        (3.5, True),
        (5.0, False),
    ):
        code += (magnitude >= threshold if inclusive else magnitude > threshold).to(
            torch.uint8
        )
    return code


def quantize_qwen3_5_mtp_nvfp4(
    weight: torch.Tensor, weight_scale_2: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D expert weight with the calibrated no-MSE NVFP4 recipe."""
    if weight.ndim != 2 or weight.shape[1] % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            "Qwen3.5 MTP NVFP4 requires a 2D weight whose K is divisible by "
            f"{NVFP4_BLOCK_SIZE}, got {tuple(weight.shape)}."
        )
    if not weight.is_floating_point():
        raise ValueError(
            "Qwen3.5 MTP NVFP4 requires floating-point source expert weights, "
            f"got {weight.dtype}."
        )

    rows, cols = weight.shape
    source = weight.float()
    if weight_scale_2 is None:
        weight_scale_2 = (
            source.abs().nan_to_num().amax() / NVFP4_GLOBAL_RANGE
        ).clamp_min(1e-30)
    else:
        weight_scale_2 = weight_scale_2.to(
            device=source.device, dtype=torch.float32
        ).reshape(())

    blocks = source.view(rows, cols // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    scales = (
        (blocks.abs().amax(dim=-1) / E2M1_MAX / weight_scale_2)
        .clamp(max=448.0)
        .to(torch.float8_e4m3fn)
    )
    effective_scale = scales.float() * weight_scale_2
    scaled = torch.where(
        effective_scale[..., None] > 0,
        blocks / effective_scale[..., None].clamp_min(1e-30),
        torch.zeros_like(blocks),
    )
    codes = _e2m1_codes(scaled.abs().clamp(max=E2M1_MAX))
    codes = (codes | ((scaled < 0).to(torch.uint8) << 3)).view(rows, cols)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, scales.contiguous(), weight_scale_2.reshape(())


def _copy_bf16_expert_slice(
    layer: torch.nn.Module,
    loaded_weight: torch.Tensor,
    shard_id: str,
    expert_id: int,
) -> None:
    """Copy this TP rank's checkpoint-order expert slice into the BF16 twin."""
    local_expert = layer._map_global_expert_id_to_local_expert_id(expert_id)
    if local_expert < 0:
        return
    intermediate = layer.draft_bf16_w2.shape[-1]
    start = layer.moe_tp_rank * intermediate
    source = loaded_weight.to(device=layer.draft_bf16_w13.device, dtype=torch.bfloat16)
    with torch.no_grad():
        if shard_id == "w2":
            layer.draft_bf16_w2[local_expert].copy_(
                source[:, start : start + intermediate]
            )
            return
        row = 0 if shard_id == "w1" else intermediate
        layer.draft_bf16_w13[local_expert, row : row + intermediate].copy_(
            source[start : start + intermediate]
        )


class ModelOptQwen3_5MtpNvFp4FusedMoEMethod(ModelOptNvFp4OnlineFusedMoEMethod):
    """Load-time MTP expert conversion with calibrated static input scales."""

    _quantize_weight_nvfp4 = staticmethod(quantize_qwen3_5_mtp_nvfp4)
    _logged = False

    def create_weights(self, layer: torch.nn.Module, **kwargs) -> None:
        layer.draft_bf16_w13 = None
        layer.draft_bf16_w2 = None
        if layer.moe_ep_size == 1:
            intermediate = kwargs["intermediate_size_per_partition"]
            hidden = kwargs["hidden_size"]
            layer.draft_bf16_w13 = torch.empty(
                kwargs["num_experts"],
                2 * intermediate,
                hidden,
                dtype=torch.bfloat16,
            )
            layer.draft_bf16_w2 = torch.empty(
                kwargs["num_experts"],
                hidden,
                intermediate,
                dtype=torch.bfloat16,
            )
        super().create_weights(layer, **kwargs)

    def prepare_weight_loader(self, layer, weight_loader):
        quantizing_loader = super().prepare_weight_loader(layer, weight_loader)
        if layer.draft_bf16_w13 is None:
            return quantizing_loader

        def loader(param, loaded_weight, weight_name, shard_id, expert_id):
            if (
                shard_id in ("w1", "w2", "w3")
                and expert_id is not None
                and loaded_weight.is_floating_point()
            ):
                _copy_bf16_expert_slice(layer, loaded_weight, shard_id, expert_id)
            return quantizing_loader(
                param,
                loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )

        return loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        with torch.no_grad():
            layer.w13_input_scale.fill_(FC1_INPUT_AMAX / NVFP4_GLOBAL_RANGE)
            layer.w2_input_scale.fill_(FC2_INPUT_AMAX / NVFP4_GLOBAL_RANGE)
        super().process_weights_after_loading(layer)
        if not type(self)._logged:
            type(self)._logged = True
            logger.info(
                "Qwen3.5 MTP draft experts use NVFP4 with calibrated input "
                "amax %.1f / %.1f; BF16 twin %s.",
                FC1_INPUT_AMAX,
                FC2_INPUT_AMAX,
                "retained" if layer.draft_bf16_w13 is not None else "not retained",
            )


class Qwen3_5MtpNvFp4Config(_ModelOptFp4OnlineConfig):
    """Quantize only the Qwen3.5 MTP draft's routed experts."""

    @classmethod
    def get_name(cls) -> str:
        return "qwen3_5_mtp_nvfp4"

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, FusedMoE):
            return ModelOptQwen3_5MtpNvFp4FusedMoEMethod(self, prefix)
        return None
