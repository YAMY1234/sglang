"""K3.1 prefill routed experts as NVFP4 W4A4 on FlashInfer's CuTe DSL fused MoE (SGLANG_K31_PREFILL_MOE_A4W4).

The checkpoint's MXFP4 experts are repacked at load without changing a value: the E2M1 nibbles are kept and each
UE8M0 group-32 scale 2^e becomes two E4M3 group-16 scales 2^(e - g), with one global 2^g per expert folded into the
GEMM alphas. Tokens and the FC1 output are quantized to NVFP4 with global scale 1.0 (NVIDIA's NVFP4 R3 recipe).
"""

from __future__ import annotations

import logging

import msgspec
import torch

logger = logging.getLogger(__name__)

_SF_VEC = 16
# E4M3 holds 2^b exactly for b in [-9, 8]: normals down to 2^-6, subnormals 2^-7..2^-9.
_E4M3_POW2_MIN, _E4M3_POW2_MAX = -9, 8
# CuteDSL's fused gated GEMM1 reads [up, gate] in alternating 64-row groups.
_W13_INTERLEAVE_ROWS = 64


class CuteDslA4W4Weights(msgspec.Struct, frozen=True, kw_only=True):
    # [E, 2I, H/2] packed E2M1; rows [up, gate] interleaved in 64-row groups
    w13: torch.Tensor
    w13_sf: torch.Tensor  # E4M3 group-16 scales of w13 in CuteDSL's MMA layout
    w1_alpha: torch.Tensor  # [E] fp32: 2^g of w13 (token global scale is 1.0)
    w2: torch.Tensor  # [E, H, I/2] packed E2M1
    w2_sf: torch.Tensor  # E4M3 group-16 scales of w2 in the MMA layout
    w2_alpha: torch.Tensor  # [E] fp32: 2^g of w2 (FC1 output global scale is 1.0)
    unit_scale: torch.Tensor  # [1] fp32 1.0: token quant and FC2-input global scale
    hidden: int
    inter: int
    num_experts: int
    beta: float  # SiTU gate tanh scale (gemm1_alpha)
    lbeta: float  # SiTU up tanh scale (gemm1_clamp_limit)


def _e4m3_pow2(exp: torch.Tensor) -> torch.Tensor:
    """E4M3 bytes of 2^exp, exp in [-9, 8]."""
    normal = (exp + 7) << 3
    subnormal = torch.bitwise_left_shift(
        torch.ones_like(exp), (exp + 9).clamp(min=0, max=2)
    )
    return torch.where(exp >= -6, normal, subnormal).to(torch.uint8)


def _nvfp4_scales(
    weight: torch.Tensor, scale_ue8m0: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """UE8M0 [E, N, K/32] -> (E4M3 bytes [E, N, K/16], per-expert global exponent [E]), the same block values."""
    exp = scale_ue8m0.view(torch.uint8).to(torch.int16) - 127
    g = exp.flatten(1).amax(dim=1) - _E4M3_POW2_MAX
    block = exp - g[:, None, None]
    under = block < _E4M3_POW2_MIN
    if bool(under.any()):
        # A block below the E4M3 range is exact only if all its nibbles are +-0.
        nibbles = weight.view(torch.uint8).unflatten(-1, (-1, 16))
        nonzero = ((nibbles & 0x77) != 0).any(dim=-1)
        if bool((under & nonzero).any()):
            raise ValueError(
                "MXFP4 scale span exceeds E4M3: repacking to NVFP4 would change values"
            )
        block = block.clamp(min=_E4M3_POW2_MIN)
    return _e4m3_pow2(block).repeat_interleave(2, dim=-1), g


def _up_gate_interleave_rows(inter: int, device: torch.device) -> torch.Tensor:
    """Source row per output row: checkpoint [gate, up] -> [up, gate] in alternating 64-row groups."""
    group = torch.arange(2 * inter, device=device) // _W13_INTERLEAVE_ROWS
    offset = torch.arange(2 * inter, device=device) % _W13_INTERLEAVE_ROWS
    half_row = (group // 2) * _W13_INTERLEAVE_ROWS + offset
    return torch.where(group % 2 == 0, inter + half_row, half_row)


def _mma_sf(sf: torch.Tensor, num_experts: int, m: int, k: int) -> torch.Tensor:
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    swizzled = (
        swizzle_blockscale(sf.view(torch.float8_e4m3fn))
        .contiguous()
        .view(torch.uint8)
        .reshape(-1)
    )
    return convert_sf_to_mma_layout(
        swizzled, m=m, k=k, num_groups=num_experts, sf_vec_size=_SF_VEC
    )


def prepare_weights(
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    *,
    beta: float,
    lbeta: float,
) -> CuteDslA4W4Weights:
    """Checkpoint-order MXFP4 experts (w13 rows: gate [0, I), up [I, 2I)) -> this path's NVFP4 layout."""
    e, two_i, half_h = w13.shape
    inter, hidden = two_i // 2, half_h * 2
    if inter % _W13_INTERLEAVE_ROWS or tuple(w2.shape) != (e, hidden, inter // 2):
        raise ValueError(
            f"unsupported MoE shape: w13 {tuple(w13.shape)}, w2 {tuple(w2.shape)}"
        )
    sf13, g13 = _nvfp4_scales(w13, s13)
    sf2, g2 = _nvfp4_scales(w2, s2)
    rows = _up_gate_interleave_rows(inter, w13.device)
    return CuteDslA4W4Weights(
        w13=w13.view(torch.uint8).index_select(1, rows),
        w13_sf=_mma_sf(sf13.index_select(1, rows), e, two_i, hidden),
        w1_alpha=torch.exp2(g13.to(torch.float32)),
        w2=w2.view(torch.uint8).contiguous(),
        w2_sf=_mma_sf(sf2, e, hidden, inter),
        w2_alpha=torch.exp2(g2.to(torch.float32)),
        unit_scale=torch.ones(1, dtype=torch.float32, device=w13.device),
        hidden=hidden,
        inter=inter,
        num_experts=e,
        beta=float(beta),
        lbeta=float(lbeta),
    )


def run(
    weights: CuteDslA4W4Weights,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    output: torch.Tensor,
) -> torch.Tensor:
    """Routed experts for BF16 tokens x [T, H] into output [T, H] (BF16; the finalize adds with atomics)."""
    from flashinfer import cute_dsl_fused_moe
    from flashinfer.tllm_enums import ActivationType

    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    num_tokens = x.shape[0]
    x_fp4, x_sf = fp4_quantize(
        x, weights.unit_scale, sf_vec_size=_SF_VEC, is_sf_swizzled_layout=False
    )
    return cute_dsl_fused_moe(
        x_fp4.reshape(num_tokens, weights.hidden // 2),
        x_sf.view(torch.float8_e4m3fn).reshape(num_tokens, weights.hidden // _SF_VEC),
        topk_ids.to(torch.int32),
        topk_weights,
        weights.w13,
        weights.w13_sf,
        weights.w1_alpha,
        weights.unit_scale,
        weights.w2,
        weights.w2_sf,
        weights.w2_alpha,
        num_experts=weights.num_experts,
        top_k=topk_ids.shape[1],
        moe_output=output,
        activation_type=ActivationType.Swiglu.value,
        situ_beta=weights.beta,
        situ_linear_beta=weights.lbeta,
        quant_mode="w4a4",
    )


_tuned_max_tokens = 0


def ensure_tuned(weights: CuteDslA4W4Weights, *, top_k: int, max_tokens: int) -> None:
    """Tune the kernel tactics for every token bucket up to max_tokens once per process (all layers share them);
    the default tactic is far slower at prefill sizes and the K3 model has no EXTEND autotune pass."""
    global _tuned_max_tokens
    if max_tokens <= _tuned_max_tokens:
        return
    from flashinfer.autotuner import autotune as fi_autotune

    device = weights.w13.device
    gen = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        max_tokens, weights.hidden, device=device, dtype=torch.bfloat16, generator=gen
    )
    ids = (
        torch.rand(max_tokens, weights.num_experts, device=device, generator=gen)
        .topk(top_k, dim=1)
        .indices
    )
    w = torch.full((max_tokens, top_k), 1.0 / top_k, device=device, dtype=torch.float32)
    out = torch.empty(max_tokens, weights.hidden, device=device, dtype=torch.bfloat16)
    with fi_autotune(True):
        run(weights, x, ids, w, output=out)
    torch.cuda.synchronize(device)
    _tuned_max_tokens = max_tokens
    logger.info(
        "SGLANG_K31_PREFILL_MOE_A4W4: tuned the CuTe DSL MoE up to %d tokens",
        max_tokens,
    )
