"""MegaMoE epilogue (shared-expert add, residual add, input RMSNorm in one kernel)
against the unfused chain: the in-place BF16 shared add, then the JIT fused-add RMSNorm."""

import sys

import pytest
import torch

from sglang.kernels.ops.layernorm.megamoe_epilogue import (
    can_use_megamoe_shared_add_rmsnorm,
    megamoe_shared_add_rmsnorm,
)
from sglang.kernels.ops.layernorm.norm import fused_add_rmsnorm
from sglang.srt.runtime_context import get_platform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or not get_platform().is_blackwell,
    reason="the kernel reproduces the fused-add RMSNorm's 32-byte-vector schedule",
)

DEVICE = "cuda"
EPS = 1e-6


def _inputs(num_tokens: int, hidden: int, seed: int = 0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    def randn(scale: float) -> torch.Tensor:
        x = torch.randn((num_tokens, hidden), device=DEVICE, generator=g) * scale
        return x.to(torch.bfloat16)

    routed, shared, residual = randn(1.0), randn(0.5), randn(4.0)
    weight = (torch.rand(hidden, device=DEVICE, generator=g) + 0.5).to(torch.bfloat16)
    return routed, shared, residual, weight


def _unfused(routed, shared, residual, weight):
    hidden_states = routed.clone().add_(shared)
    residual = residual.clone()
    fused_add_rmsnorm(hidden_states, residual, weight, EPS)
    return hidden_states, residual


def _assert_bitwise(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("hidden", [4096, 6144, 7168])
@pytest.mark.parametrize("num_tokens", [1, 3, 16, 64, 255, 512])
def test_matches_unfused_chain(hidden: int, num_tokens: int) -> None:
    routed, shared, residual, weight = _inputs(num_tokens, hidden)
    expected_out, expected_residual = _unfused(routed, shared, residual, weight)
    routed_before, shared_before = routed.clone(), shared.clone()

    assert can_use_megamoe_shared_add_rmsnorm(routed, shared, residual, weight)
    out = megamoe_shared_add_rmsnorm(routed, shared, residual, weight, EPS)

    _assert_bitwise(out, expected_out)
    _assert_bitwise(residual, expected_residual)
    # The handoff's inputs stay intact; only the residual is updated in place.
    _assert_bitwise(routed, routed_before)
    _assert_bitwise(shared, shared_before)


def test_strided_rows_match_unfused_chain() -> None:
    hidden, num_tokens = 6144, 7
    routed, shared, residual, weight = _inputs(num_tokens, hidden, seed=1)
    wide = torch.zeros((num_tokens, 2 * hidden), device=DEVICE, dtype=torch.bfloat16)
    strided_residual = wide[:, hidden:]
    strided_residual.copy_(residual)
    expected_out, expected_residual = _unfused(routed, shared, residual, weight)

    assert can_use_megamoe_shared_add_rmsnorm(routed, shared, strided_residual, weight)
    out = megamoe_shared_add_rmsnorm(routed, shared, strided_residual, weight, EPS)

    _assert_bitwise(out, expected_out)
    _assert_bitwise(strided_residual, expected_residual)
    assert not wide[:, :hidden].any()


def test_declines_operands_outside_the_reproduced_schedule() -> None:
    routed, shared, residual, weight = _inputs(8, 6144)
    assert can_use_megamoe_shared_add_rmsnorm(routed, shared, residual, weight)
    # Past the decode threshold the fused add switches to its prefill schedule.
    big = _inputs(513, 6144)
    assert not can_use_megamoe_shared_add_rmsnorm(*big)
    # A row longer than the fused add's one-vector-per-thread decode schedule.
    assert not can_use_megamoe_shared_add_rmsnorm(*_inputs(8, 12288))
    assert not can_use_megamoe_shared_add_rmsnorm(
        routed[:0], shared[:0], residual[:0], weight
    )
    # A row start off the 32-byte vector.
    unaligned = torch.empty((8, 6144 + 8), device=DEVICE, dtype=torch.bfloat16)[:, 8:]
    assert not can_use_megamoe_shared_add_rmsnorm(routed, shared, unaligned, weight)
    assert not can_use_megamoe_shared_add_rmsnorm(
        routed.half(), shared, residual, weight
    )
    assert not can_use_megamoe_shared_add_rmsnorm(
        routed, shared, residual, weight.float()
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
