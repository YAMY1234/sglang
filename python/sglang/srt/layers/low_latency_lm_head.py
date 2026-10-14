"""Low-M BF16 LM head on FlashInfer ``mm_bf16`` ``cute-dsl`` (SM100 / SM103).

With SGLANG_OPT_DSV41_LOW_LATENCY_LM_HEAD=1 the DSV4.1 LM head (which the DSpark
draft shares) runs FlashInfer's standalone low-M split-K kernels for 1 to
LOW_LATENCY_LM_HEAD_MAX_ROWS BF16 rows instead of cuBLAS split-K plus its reduce;
on the TP4 shard (N = 32320, K = 5120) at M = 5-6 that is about 48 us against
55 us. Same BF16 output; only the K summation order differs. Ported from muLLM
models/deepseek_v41/nvidia/low_latency_lm_head.py (commits b695361a, fbec1cdf).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)

LOW_LATENCY_LM_HEAD_BACKEND = "cute-dsl"
LOW_LATENCY_LM_HEAD_MAX_ROWS = 16
# One call per FlashInfer token bucket (powers of two) tunes 1-16 rows.
_TUNING_ROWS = tuple(1 << i for i in range(LOW_LATENCY_LM_HEAD_MAX_ROWS.bit_length()))


@lru_cache(maxsize=1)
def _low_latency_lm_head_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        return False
    try:
        from flashinfer import mm_bf16

        return bool(
            mm_bf16.is_backend_supported(LOW_LATENCY_LM_HEAD_BACKEND, major * 10 + minor)
        )
    except Exception:
        return False


@lru_cache(maxsize=1)
def _use_pdl() -> bool:
    from sglang.kernels.jit.utils.arch import is_arch_support_pdl

    return is_arch_support_pdl()


def mark_low_latency_lm_head(lm_head: nn.Module) -> bool:
    """Tag an unquantized BF16 head for the low-M path; the tag is read per call."""
    if lm_head.weight.dtype != torch.bfloat16 or not _low_latency_lm_head_supported():
        logger.warning(
            "SGLANG_OPT_DSV41_LOW_LATENCY_LM_HEAD is set but the LM head weight is "
            f"{lm_head.weight.dtype} or FlashInfer cute-dsl mm_bf16 is unavailable; "
            "keeping the default LM head GEMM."
        )
        return False
    lm_head.low_latency_lm_head = True
    return True


def maybe_low_latency_lm_head(
    hidden_states: torch.Tensor, lm_head: nn.Module
) -> Optional[torch.Tensor]:
    """BF16 logits of a tagged head at 1-16 contiguous BF16 rows, else None."""
    if not getattr(lm_head, "low_latency_lm_head", False):
        return None
    weight = lm_head.weight
    if not (
        hidden_states.dim() == 2
        and 1 <= hidden_states.shape[0] <= LOW_LATENCY_LM_HEAD_MAX_ROWS
        and hidden_states.dtype == weight.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and weight.is_contiguous()
    ):
        return None
    from flashinfer import mm_bf16

    return mm_bf16(
        hidden_states, weight.t(), pdl=_use_pdl(), backend=LOW_LATENCY_LM_HEAD_BACKEND
    )


def autotune_low_latency_lm_head(lm_head: nn.Module) -> None:
    """Call inside the FlashInfer autotune context: tunes buckets 1-16, same calls on every rank."""
    if not getattr(lm_head, "low_latency_lm_head", False):
        return
    weight = lm_head.weight
    for rows in _TUNING_ROWS:
        x = torch.zeros((rows, weight.shape[1]), dtype=weight.dtype, device=weight.device)
        maybe_low_latency_lm_head(x, lm_head)
    logger.info(
        f"Tuned the low-latency LM head (N={weight.shape[0]}, K={weight.shape[1]}) "
        f"at 1-{LOW_LATENCY_LM_HEAD_MAX_ROWS} rows."
    )
