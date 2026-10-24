"""PERF-LP: the Qwen3.5 MTP draft's routed experts in NVFP4 with TRT-LLM's recipe.

TRT-LLM's ``TRTLLM_Q35_MTP_MOE_QUANT=nvfp4`` (``modules/q35_mtp_moe_quant.py``) quantizes the checkpoint's BF16 MTP
experts at load time to the target experts' format and runs them on the target's NVFP4 MoE path. This module
reproduces that recipe (``SGLANG_PERFLP_DRAFT_MOE_NVFP4=1``); only the draft's routed experts change, so the target's
outputs are untouched and only the draft proposals (the real acceptance length) can move:

* weights: ModelOpt NVFP4, E2M1 values (round to nearest, ties to the even code), one E4M3 block scale per 16
  K-elements ``e4m3(block_amax / 6 / ws2)``, one FP32 ``ws2 = amax / (6 * 448)`` per expert tensor taken over the
  whole expert before TP sharding, w1 / w3 sharing ``max(amax)``; no MSE search;
* activations: NVFP4 with static global scales ``amax / (6 * 448)`` for the FC1 / FC2 inputs, from TRT's calibrated
  amax (256, 2048; ``SGLANG_PERFLP_DRAFT_MOE_NVFP4_AMAX`` overrides).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.nvfp4_online import (
    ModelOptNvFp4OnlineFusedMoEMethod,
    _ModelOptFp4OnlineConfig,
)

logger = logging.getLogger(__name__)

NVFP4_BLOCK = 16
# ModelOpt / TRT-LLM global scale convention: amax maps to E2M1 max (6) x E4M3 max (448).
_NVFP4_RANGE = 6.0 * 448.0
_E2M1_MAX = 6.0
# TRT-LLM's TRTLLM_Q35_MTP_MOE_NVFP4_AMAX defaults (FC1 input, SwiGLU output), calibrated on the BF16 MTP layer.
DEFAULT_ACT_AMAX = (256.0, 2048.0)


def draft_act_amax() -> Tuple[float, float]:
    v = envs.SGLANG_PERFLP_DRAFT_MOE_NVFP4_AMAX.get().strip()
    if not v:
        return DEFAULT_ACT_AMAX
    parts = [float(t) for t in v.split(",")]
    if len(parts) != 2 or not all(p > 0 for p in parts):
        raise ValueError(
            f"SGLANG_PERFLP_DRAFT_MOE_NVFP4_AMAX={v!r}: expected '<fc1>,<fc2>' (positive)"
        )
    return parts[0], parts[1]


def _e2m1_codes(mag: torch.Tensor) -> torch.Tensor:
    """Magnitudes in [0, 6] -> E2M1 magnitude codes 0..7 (uint8), nearest with ties to the even code, as
    cvt.rn.satfinite.e2m1x2 rounds (TRT-LLM's ``_e2m1_codes``)."""
    code = (mag > 0.25).to(torch.uint8)
    for thr, inclusive in (
        (0.75, True),
        (1.25, False),
        (1.75, True),
        (2.5, False),
        (3.5, True),
        (5.0, False),
    ):
        code += (mag >= thr if inclusive else mag > thr).to(torch.uint8)
    return code


def quantize_nvfp4_trt(
    weight: torch.Tensor, weight_scale_2: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``weight`` [N, K] -> (packed E2M1 uint8 [N, K / 2], E4M3 block scales [N, K / 16], FP32 ``weight_scale_2``
    0-d); TRT-LLM's ``quantize_nvfp4`` with MSE off. Element ``2j`` goes to the low nibble of byte ``j``; a
    negative value that rounds to zero keeps its sign (code 8)."""
    n, k = weight.shape
    if k % NVFP4_BLOCK != 0:
        raise ValueError(f"NVFP4 needs K % 16 == 0, got {tuple(weight.shape)}")
    wf = weight.float()
    if weight_scale_2 is None:
        ws2 = (wf.abs().amax() / _NVFP4_RANGE).clamp_min(1e-30).to(torch.float32)
    else:
        ws2 = weight_scale_2.to(device=wf.device, dtype=torch.float32).reshape(())
    blocks = wf.view(n, k // NVFP4_BLOCK, NVFP4_BLOCK)
    sf = (
        (blocks.abs().amax(dim=-1) / _E2M1_MAX / ws2)
        .clamp(max=448.0)
        .to(torch.float8_e4m3fn)
    )
    eff = sf.float() * ws2
    scaled = torch.where(
        eff[..., None] > 0,
        blocks / eff[..., None].clamp_min(1e-30),
        torch.zeros_like(blocks),
    )
    code = _e2m1_codes(scaled.abs().clamp(max=_E2M1_MAX))
    code = (code | ((scaled < 0).to(torch.uint8) << 3)).view(n, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, sf.contiguous(), ws2.reshape(())


def dequantize_nvfp4(
    packed: torch.Tensor, scales: torch.Tensor, weight_scale_2: torch.Tensor
) -> torch.Tensor:
    """FP32 [N, K] from ``quantize_nvfp4_trt``'s output (tests)."""
    n, half = packed.shape
    k = 2 * half
    code = torch.stack([packed & 0x0F, packed >> 4], dim=-1).view(n, k).long()
    grid = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=packed.device
    )
    val = grid[code & 7] * torch.where(code >= 8, -1.0, 1.0)
    scale = scales.view(torch.float8_e4m3fn).float() * weight_scale_2.float()
    return (val.view(n, k // NVFP4_BLOCK, NVFP4_BLOCK) * scale[..., None]).view(n, k)


# FusedMoE's weight loader takes the ModelOpt branch on "ModelOpt" in the method's class name.
class ModelOptNvFp4PerfLpDraftMoEMethod(ModelOptNvFp4OnlineFusedMoEMethod):
    """The draft experts' NVFP4 method: the online loader with TRT-LLM's quantizer, and TRT-LLM's static input
    scales."""

    _quantize_weight_nvfp4 = staticmethod(quantize_nvfp4_trt)
    _logged = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        fc1_amax, fc2_amax = draft_act_amax()
        # The BF16 checkpoint has no input scales for these experts: install TRT-LLM's.
        with torch.no_grad():
            layer.w13_input_scale.fill_(fc1_amax / _NVFP4_RANGE)
            layer.w2_input_scale.fill_(fc2_amax / _NVFP4_RANGE)
        super().process_weights_after_loading(layer)
        if not ModelOptNvFp4PerfLpDraftMoEMethod._logged:
            ModelOptNvFp4PerfLpDraftMoEMethod._logged = True
            logger.info(
                "PERF-LP draft MoE NVFP4 (%s): %d local experts quantized at load time, "
                "TRT recipe, input amax fc1 %.1f fc2 %.1f",
                self.layer_log_name,
                layer.num_local_experts,
                fc1_amax,
                fc2_amax,
            )


class PerfLpDraftNvFp4Config(_ModelOptFp4OnlineConfig):
    """Quant config handed to the MTP draft's routed experts only (the draft's other modules stay BF16)."""

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, FusedMoE):
            return ModelOptNvFp4PerfLpDraftMoEMethod(self, prefix)
        return None
