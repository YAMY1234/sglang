"""Integer metadata for the next EAGLE draft-extend; no sampling changes."""

import triton
import triton.language as tl


@triton.jit
def draft_extend_metadata_kernel(
    accept_lens,
    num_correct_drafts,
    select_index,
    N: tl.constexpr,
    DRAFT_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(accept_lens + row, row < N, other=1)
    tl.store(num_correct_drafts + row, length - 1, row < N)
    index = row.to(tl.int64) * DRAFT_WIDTH + length.to(tl.int64) - 1
    tl.store(select_index + row, index, row < N)
