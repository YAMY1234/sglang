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


class _TacticTable(msgspec.Struct, frozen=True, kw_only=True):
    max_tokens: int  # largest tuned bucket; larger calls take the tuner's own lookup
    tactics: dict  # tuning bucket -> (tile_size, gemm1 tactic, gemm2 tactic)


class _Workspace(msgspec.Struct, frozen=True, kw_only=True):
    # moe_sort outputs and the NVFP4 FC1 output, flat, sized for max_tokens at the largest tile; one per process
    tile_expert: torch.Tensor
    tile_limit: torch.Tensor
    expanded_to_permuted: torch.Tensor
    permuted_to_expanded: torch.Tensor
    total_padded: torch.Tensor
    non_exiting_tiles: torch.Tensor
    fc1: torch.Tensor
    fc1_sf: torch.Tensor


_table: _TacticTable | None = None
_workspace: _Workspace | None = None


def _quantize_tokens(weights: CuteDslA4W4Weights, x: torch.Tensor):
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    num_tokens = x.shape[0]
    x_fp4, x_sf = fp4_quantize(
        x, weights.unit_scale, sf_vec_size=_SF_VEC, is_sf_swizzled_layout=False
    )
    return x_fp4.reshape(num_tokens, weights.hidden // 2), x_sf.view(
        torch.float8_e4m3fn
    ).reshape(num_tokens, weights.hidden // _SF_VEC)


def _run_tuned(
    weights: CuteDslA4W4Weights,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    output: torch.Tensor,
) -> torch.Tensor:
    """FlashInfer's functional API: per call it builds a runner, looks the tactic up and allocates its buffers."""
    from flashinfer import cute_dsl_fused_moe
    from flashinfer.tllm_enums import ActivationType

    x_fp4, x_sf = _quantize_tokens(weights, x)
    return cute_dsl_fused_moe(
        x_fp4,
        x_sf,
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


def _get_workspace(weights: CuteDslA4W4Weights, top_k: int) -> _Workspace:
    """Allocated by ensure_tuned at load, before the KV pool is sized, so the pool sizing accounts for it."""
    global _workspace
    if _workspace is None:
        from flashinfer.fused_moe.cute_dsl.moe_utils import (
            get_max_num_permuted_tokens,
            get_max_num_tiles,
        )

        assert _table is not None
        e, t = weights.num_experts, _table.max_tokens
        tiles = {s for s, _, _ in _table.tactics.values()}
        num_tiles = max(get_max_num_tiles(t, top_k, e, s) for s in tiles)
        permuted = max(get_max_num_permuted_tokens(t, top_k, e, s) for s in tiles)
        new = lambda n, dtype=torch.int32: torch.empty(
            n, dtype=dtype, device=weights.w13.device
        )
        _workspace = _Workspace(
            tile_expert=new(num_tiles),
            tile_limit=new(num_tiles),
            expanded_to_permuted=new(t * top_k),
            permuted_to_expanded=new(permuted),
            total_padded=new(1),
            non_exiting_tiles=new(1),
            fc1=new(permuted * weights.inter // 2, torch.uint8),
            fc1_sf=new(permuted * _fc1_sf_cols(weights.inter), torch.uint8),
        )
    return _workspace


def _fc1_sf_cols(inter: int) -> int:
    # FlashInfer's FC1 scale atom: 4 rows x 4 scales (64 columns) per 128-row tile column
    return 4 * ((inter + 4 * _SF_VEC - 1) // (4 * _SF_VEC))


def run(
    weights: CuteDslA4W4Weights,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    output: torch.Tensor,
) -> torch.Tensor:
    """Routed experts for BF16 tokens x [T, H] into output [T, H] (BF16; the finalize adds with atomics).

    The same kernels and tactic as FlashInfer's functional API, minus its per-call host work: the tactic comes from
    the table resolved after tuning, and moe_sort / FC1 write into views of one persistent workspace."""
    from flashinfer.fused_moe.cute_dsl.fused_moe import _moe_core_impl
    from flashinfer.fused_moe.cute_dsl.moe_utils import (
        get_max_num_permuted_tokens,
        get_max_num_tiles,
    )
    from flashinfer.fused_moe.cute_dsl.tuner import _extract_tactic_params
    from flashinfer.fused_moe.utils import map_to_hybrid_bucket
    from flashinfer.tllm_enums import ActivationType

    num_tokens, top_k = topk_ids.shape
    tactic = None
    if _table is not None and num_tokens <= _table.max_tokens:
        tactic = _table.tactics.get(map_to_hybrid_bucket(num_tokens, _table.max_tokens))
    if tactic is None:
        return _run_tuned(weights, x, topk_ids, topk_weights, output=output)
    p = _extract_tactic_params(tactic)
    tile, e = p["tile_size"], weights.num_experts
    ws = _get_workspace(weights, top_k)
    num_tiles = get_max_num_tiles(num_tokens, top_k, e, tile)
    permuted = get_max_num_permuted_tokens(num_tokens, top_k, e, tile)
    x_fp4, x_sf = _quantize_tokens(weights, x)
    return _moe_core_impl(
        x=x_fp4,
        x_sf=x_sf,
        token_selected_experts=topk_ids.to(torch.int32),
        token_final_scales=topk_weights,
        w1_weight=weights.w13,
        w1_weight_sf=weights.w13_sf,
        w1_alpha=weights.w1_alpha,
        fc2_input_scale=weights.unit_scale,
        w2_weight=weights.w2,
        w2_weight_sf=weights.w2_sf,
        w2_alpha=weights.w2_alpha,
        num_experts=e,
        top_k=top_k,
        num_local_experts=e,
        tile_size=tile,
        gemm1_mma_tiler_mn=p["gemm1_mma_tiler_mn"],
        gemm1_cluster_shape_mn=p["gemm1_cluster_shape_mn"],
        gemm2_mma_tiler_mn=p["gemm2_mma_tiler_mn"],
        gemm2_cluster_shape_mn=p["gemm2_cluster_shape_mn"],
        gemm1_mma_tiler=p["gemm1_mma_tiler"],
        gemm1_mma_inst_shape=p["gemm1_mma_inst_shape"],
        gemm2_mma_tiler=p["gemm2_mma_tiler"],
        gemm2_mma_inst_shape=p["gemm2_mma_inst_shape"],
        moe_sort_buffers=dict(
            out_tile_idx_to_expert_idx=ws.tile_expert[:num_tiles],
            out_tile_idx_to_mn_limit=ws.tile_limit[:num_tiles],
            out_expanded_idx_to_permuted_idx=ws.expanded_to_permuted[
                : num_tokens * top_k
            ].view(num_tokens, top_k),
            out_permuted_idx_to_expanded_idx=ws.permuted_to_expanded[:permuted],
            out_total_num_padded_tokens=ws.total_padded,
            out_num_non_exiting_tiles=ws.non_exiting_tiles,
        ),
        gemm1_out=ws.fc1[: permuted * weights.inter // 2].view(
            permuted, weights.inter // 2
        ),
        gemm1_out_scale=ws.fc1_sf[: permuted * _fc1_sf_cols(weights.inter)].view(
            32, 4, permuted // 128, 4, _fc1_sf_cols(weights.inter) // 4, 1
        ),
        moe_output=output,
        output_dtype=torch.bfloat16,
        # The output memset stays on this stream: FlashInfer's aux stream comes from torch's round-robin stream
        # pool, so in a server it can alias a busy stream whose queued work would then gate GEMM2.
        use_async_memset=False,
        use_fused_finalize=True,
        enable_pdl=True,
        activation_type=ActivationType.Swiglu.value,
        situ_beta=weights.beta,
        situ_linear_beta=weights.lbeta,
    )


def _resolve_tactics(op_name: str, max_tokens: int) -> _TacticTable:
    """The tuner's winners for this op, by token bucket (they are shared by every layer of the process)."""
    from flashinfer.autotuner import AutoTuner

    tactics = {
        key.nearest_profile[0][0]: value[0]
        for key, value in AutoTuner.get().profiling_cache.items()
        if key.custom_op == op_name and value[0] != -1
    }
    return _TacticTable(max_tokens=max(tactics, default=0), tactics=tactics)


_tuned_max_tokens = 0


def ensure_tuned(weights: CuteDslA4W4Weights, *, top_k: int, max_tokens: int) -> None:
    """Tune the kernel tactics for every token bucket up to max_tokens once per process (all layers share them);
    the default tactic is far slower at prefill sizes and the K3 model has no EXTEND autotune pass."""
    global _tuned_max_tokens, _table
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
        _run_tuned(weights, x, ids, w, output=out)
    torch.cuda.synchronize(device)
    del x, ids, w, out
    _tuned_max_tokens = max_tokens
    _table = _resolve_tactics("CuteDslFusedMoE::run_moe_w4a4::Situ", max_tokens)
    workspace_bytes = 0
    if _table.tactics:
        ws = _get_workspace(weights, top_k)
        workspace_bytes = sum(
            getattr(ws, f).nbytes for f in _Workspace.__struct_fields__
        )
    logger.info(
        "SGLANG_K31_PREFILL_MOE_A4W4: tuned the CuTe DSL MoE up to %d tokens; "
        "%d bucket tactics up to %d tokens resolved for the direct path; "
        "workspace %.3f GB",
        max_tokens,
        len(_table.tactics),
        _table.max_tokens,
        workspace_bytes / 1e9,
    )
