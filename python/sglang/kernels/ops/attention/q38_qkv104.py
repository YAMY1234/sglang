"""FN QKV with the mature full-K, 104-CTA TGV tactic.

Tactic 12 uses 128 output channels by 16 token rows, one CTA, six stages.
For N=13312 and M<=16 its grid is 104 CTAs. The original DMA_A weight
prefetch, DMA_B activation PDL wait, MMA and BF16 epilogue are unchanged.
"""

import torch

from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import _run_tgv
from sglang.srt.utils.common import direct_register_custom_op


def _qkv104_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


def _qkv104_run(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out = _qkv104_fake(x, weight)
    return _run_tgv(x, weight.t(), None, out, pdl=True, tactic=12)


direct_register_custom_op(
    op_name="q38c_qkv104",
    op_func=_qkv104_run,
    mutates_args=[],
    fake_impl=_qkv104_fake,
)


def qkv104(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.ops.sglang.q38c_qkv104(x, weight)
