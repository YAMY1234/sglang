"""Separate HC gate preparation and combine, with optional next-branch norm.

The decode specialization accepts four branches of 2560 BF16 values. Gate
partials belong to each invocation, so overlapping sublayers never reuse them.
"""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _jit_module():
    return load_jit(
        "hc_combine_decode",
        cuda_files=["elementwise/hc_combine_decode.cuh"],
        cuda_wrappers=[
            ("gate", "HcCombineDecode::gate"),
            ("apply", "HcCombineDecode::apply"),
            ("apply_norm", "HcCombineNormDecode<2>::run"),
        ],
    )


def hc_combine_gate(normed: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    partials = torch.empty(
        (normed.shape[0], 8, 4), device=normed.device, dtype=torch.float32
    )
    from sglang.kernels.plugin_kernels.nvidia.numeric import (
        HcGateIO,
        HcSignature,
        NumericOperation,
        ScalarType,
    )
    from sglang.srt.kernel_plugins.nvidia.numeric_binding import numeric_operator

    streams = weight.shape[0]
    hidden = normed.shape[1] // streams
    run = numeric_operator(
        HcSignature(
            NumericOperation.HC_GATE, normed.shape[0], streams, hidden, ScalarType.BF16
        ),
        prepared=(
            normed.is_cuda
            and normed.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and weight.device == normed.device
            and normed.shape[1] == streams * hidden
            and weight.shape == (streams, normed.shape[1])
            and normed.is_contiguous()
            and weight.is_contiguous()
        ),
    )
    if run is not None:
        run(HcGateIO(normed, weight, streams, hidden, partials))
        return partials
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
) -> tuple[torch.Tensor, torch.Tensor]:
    combined = torch.empty_like(residual)
    normalized = torch.empty_like(residual)
    from sglang.kernels.plugin_kernels.nvidia.numeric import HcApplyNormIO

    run = _numeric_hc_apply_norm(block, residual, partials, weight)
    if run is not None:
        run(
            HcApplyNormIO(
                block, residual, partials, weight, eps, 2, combined, normalized
            )
        )
        return combined, normalized
    _jit_module().apply_norm(
        block, residual, partials, weight, combined, normalized, eps
    )
    return combined, normalized


def _numeric_hc_apply_norm(block, residual, partials, weight):
    from sglang.kernels.plugin_kernels.nvidia.numeric import (
        HcSignature,
        NumericOperation,
        ScalarType,
    )
    from sglang.srt.kernel_plugins.nvidia.numeric_binding import numeric_operator

    hidden = block.shape[1]
    streams = residual.shape[1] // hidden
    return numeric_operator(
        HcSignature(
            NumericOperation.HC_APPLY_NORM,
            block.shape[0],
            streams,
            hidden,
            ScalarType.BF16,
        ),
        prepared=(
            block.is_cuda
            and block.dtype == residual.dtype == torch.bfloat16
            and residual.shape[1] == streams * hidden
            and residual.shape[0] == block.shape[0]
            and partials.shape == (block.shape[0], 8, streams)
            and partials.dtype == torch.float32
            and weight.dtype == torch.bfloat16
            and weight.numel() == residual.shape[1]
            and all(t.device == block.device for t in (residual, partials, weight))
            and all(t.is_contiguous() for t in (block, residual, partials, weight))
        ),
    )
