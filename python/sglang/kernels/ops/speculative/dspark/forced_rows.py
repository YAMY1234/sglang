from __future__ import annotations

import torch
import triton
import triton.language as tl

# Measured on VR200 at 636-1060 rows of a 129,280 vocab: smaller blocks are CTA-launch bound.
_BLOCK = 32768


@triton.jit
def _fill_forced_rows_kernel(
    logits_ptr,
    logits_stride,
    forced_ptr,
    forced_stride0,
    forced_stride1,
    width,
    vocab_start,
    vocab,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    token = tl.load(forced_ptr + (row // width) * forced_stride0 + (row % width) * forced_stride1)
    if token >= 0:
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        value = tl.where(offs == token - vocab_start, 0.0, float("-inf"))
        tl.store(
            logits_ptr + row.to(tl.int64) * logits_stride + offs,
            value.to(logits_ptr.dtype.element_ty),
            mask=offs < vocab,
        )


def fill_forced_rows(
    logits: torch.Tensor, forced_tokens: torch.Tensor, vocab_start: int = 0
) -> None:
    """One-hot logits rows [0, forced_tokens.numel()) whose forced token (row-major in
    forced_tokens [bs, width]) is >= 0: 0 at the token when it lies in this vocab shard,
    -inf elsewhere. Rows without a forced token are not read or written."""
    rows = forced_tokens.numel()
    if rows == 0:
        return
    vocab = logits.shape[1]
    _fill_forced_rows_kernel[(rows, triton.cdiv(vocab, _BLOCK))](
        logits,
        logits.stride(0),
        forced_tokens,
        forced_tokens.stride(0),
        forced_tokens.stride(1),
        forced_tokens.shape[1],
        vocab_start,
        vocab,
        BLOCK=_BLOCK,
        num_warps=8,
    )
