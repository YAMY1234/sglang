"""Separate HC gate preparation and combine, with optional next-branch norm.

The decode specialization accepts four branches of 2560 BF16 values. Gate
partials belong to each invocation, so overlapping sublayers never reuse them.
"""

import os

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.environ import envs

_HC_CLUSTER_M4ONLY = envs.SGLANG_Q38C_HC_CLUSTER_M4ONLY.get()
_HC_M4_VARIANT = int(os.environ.get("SGLANG_Q38C_HC_M4_COMBINE", "0"))
if _HC_M4_VARIANT not in (0, 1, 2, 3):
    raise ValueError("HC M4 combine variant must be 0, 1, 2 or 3")


@cache_once
def _jit_module(parts: int = 2):
    if parts not in (1, 2, 4):
        raise ValueError("HC combine parts must be 1, 2 or 4")
    m4only = str(_HC_CLUSTER_M4ONLY).lower()
    return load_jit(
        f"hc_combine_decode_p{parts}_m4only{int(_HC_CLUSTER_M4ONLY)}_v{_HC_M4_VARIANT}",
        cuda_files=["elementwise/hc_combine_decode.cuh"],
        cuda_wrappers=[
            ("gate", "HcCombineDecode::gate"),
            ("apply", "HcCombineDecode::apply"),
            ("apply_norm", f"HcCombineNormDecode<{parts}, {m4only}, {_HC_M4_VARIANT}>::run"),
            ("shared_apply_norm", f"HcSharedCombineNormDecode<{parts}, {m4only}>::run"),
        ],
    )


def hc_combine_gate(normed: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    partials = torch.empty(
        (normed.shape[0], 8, 4), device=normed.device, dtype=torch.float32
    )
    _jit_module().gate(normed, weight, partials)
    return partials


def hc_combine_apply(
    block: torch.Tensor, residual: torch.Tensor, partials: torch.Tensor
) -> torch.Tensor:
    combined = torch.empty_like(residual)
    _jit_module().apply(block, residual, partials, combined)
    return combined


def hc_combine_apply_norm(
    block: torch.Tensor,
    residual: torch.Tensor,
    partials: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    parts: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined = torch.empty_like(residual)
    normalized = torch.empty_like(residual)
    _jit_module(parts).apply_norm(
        block, residual, partials, weight, combined, normalized, eps
    )
    return combined, normalized


def hc_shared_combine_apply_norm(hidden, shared, routed, residual, partials,
                                 shared_weight, norm_weight, eps, parts=2):
    combined = torch.empty_like(residual)
    normalized = torch.empty_like(residual)
    _jit_module(parts).shared_apply_norm(
        hidden, shared, routed, residual, partials, shared_weight,
        norm_weight, combined, normalized, eps
    )
    return combined, normalized
