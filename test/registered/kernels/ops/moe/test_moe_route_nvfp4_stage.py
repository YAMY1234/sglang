"""Routing plus FlashInfer MegaMoE NVFP4 input staging in one launch, against the
unfused chain it replaces: the route_warp router (moe_fused_gate), the padded-row id
mask, and FlashInfer's compute capability 10.7 NVFP4 stager into the same buffers."""

import sys

import pytest
import torch

from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.kernels.ops.moe.moe_route_nvfp4_stage import covered, route_nvfp4_stage
from sglang.srt.runtime_context import get_platform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or not get_platform().is_blackwell,
    reason="E2M1 conversions and FlashInfer's CuTe DSL stager need SM100+",
)

DEVICE = "cuda"
# The MegaMoE workspace's NVFP4 storage dtype.
FP4 = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
STALE_ID, STALE_WEIGHT = 7, 3.0


def _stager():
    staging = pytest.importorskip(
        "flashinfer.moe_ep.backends.mega.kernel.sm107.nvfp4_nvfp4_bf16_cutedsl.staging"
    )
    return staging.stage_mega_moe_inputs


def _workspace(capacity: int, hidden: int, topk: int):
    """Buffers shaped and typed as the workspace's, holding a previous batch."""
    return (
        torch.full(
            (capacity, hidden // 2), 0x5A, device=DEVICE, dtype=torch.uint8
        ).view(FP4),
        torch.full(
            (capacity, hidden // 16), 0x3C, device=DEVICE, dtype=torch.uint8
        ).view(torch.float8_e4m3fn),
        torch.full((capacity, topk), STALE_ID, device=DEVICE, dtype=torch.int32),
        torch.full((capacity, topk), STALE_WEIGHT, device=DEVICE, dtype=torch.float32),
    )


def _inputs(num_tokens: int, num_experts: int, hidden: int, dtype, seed: int = 0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    scores = (
        torch.randn((num_tokens, num_experts), device=DEVICE, generator=g) * 2
    ).to(dtype)
    bias = torch.randn(num_experts, device=DEVICE, generator=g) * 0.1
    x = torch.randn((num_tokens, hidden), device=DEVICE, generator=g) * 3
    x[:, :16] = 0  # an all-zero NVFP4 block
    return scores, bias, x.to(torch.bfloat16)


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "num_tokens,num_non_padded", [(1, None), (5, 3), (64, None), (200, 129)]
)
@pytest.mark.parametrize(
    "renormalize,scale,apply_scale", [(True, 2.5, True), (False, 1.0, False)]
)
def test_matches_unfused_chain(
    dtype, num_tokens, num_non_padded, renormalize, scale, apply_scale
) -> None:
    num_experts, topk, hidden, capacity = 256, 8, 6144, 256
    norm_const = 0.75
    scores, bias, x = _inputs(num_tokens, num_experts, hidden, dtype)
    assert covered(scores, bias, topk)
    n = num_tokens if num_non_padded is None else num_non_padded
    padded = (
        None
        if num_non_padded is None
        else torch.tensor([num_non_padded], device=DEVICE, dtype=torch.int32)
    )

    # Unfused: the router, the padded-row id mask, FlashInfer's stager.
    ref_w, ref_i = moe_fused_gate(
        scores,
        bias,
        topk,
        scoring_func="sigmoid",
        renormalize=renormalize,
        routed_scaling_factor=scale,
        apply_routed_scaling_factor_on_output=apply_scale,
    )
    ref_i[n:] = -1
    ref_ws = _workspace(capacity, hidden, topk)
    _stager()(
        x, ref_w, ref_i, *ref_ws, input_norm_const=norm_const, num_experts=num_experts
    )

    ws = _workspace(capacity, hidden, topk)
    weights, ids = route_nvfp4_stage(
        scores,
        bias,
        topk,
        renormalize=renormalize,
        routed_scaling_factor=scale,
        apply_routed_scaling_factor_on_output=apply_scale,
        num_token_non_padded=padded,
        x=x,
        x_q=_bytes(ws[0]),
        x_sf=_bytes(ws[1]),
        stage_ids=ws[2],
        stage_weights=ws[3],
        norm_const=norm_const,
        scale_multiplier=norm_const / 6.0,
    )
    torch.cuda.synchronize()

    assert torch.equal(ids, ref_i)
    assert torch.equal(weights[:n].view(torch.int32), ref_w[:n].view(torch.int32))
    # Padded rows route nowhere, with zero weight.
    assert not weights[n:].any()
    assert torch.equal(_bytes(ws[0])[:num_tokens], _bytes(ref_ws[0])[:num_tokens])
    assert torch.equal(_bytes(ws[1])[:num_tokens], _bytes(ref_ws[1])[:num_tokens])
    # The workspace routes, the masked tail included.
    assert torch.equal(ws[2], ref_ws[2])
    assert (ws[2][num_tokens:] == -1).all()
    assert torch.equal(ws[3][:n].view(torch.int32), ref_ws[3][:n].view(torch.int32))
    assert not ws[3][n:num_tokens].any()


def test_covers_only_the_route_warp_dispatch() -> None:
    bias = torch.zeros(256, device=DEVICE)
    scores = torch.zeros((4, 256), device=DEVICE)
    assert covered(scores, bias, 8)
    assert not covered(scores.half(), bias, 8)
    assert not covered(scores, bias.bfloat16(), 8)
    assert not covered(scores.t().contiguous().t(), bias, 8)
    # Kimi-K3 routing (896 experts, top-16) is not route_warp's.
    k3 = torch.zeros((4, 896), device=DEVICE)
    assert not covered(k3, torch.zeros(896, device=DEVICE), 16)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
