"""Small-batch draft input staging with one CUDA launch.

This changes only copies into existing graph input buffers. Capture, metadata
planning, stream ordering, and graph output ownership remain in the caller.
"""

import triton
import triton.language as tl


@triton.jit
def _copy_draft_inputs(
    Seq,
    Loc,
    Pos,
    Prob,
    Token,
    Req,
    Hidden,
    DSeq,
    DLoc,
    DPos,
    DProb,
    DToken,
    DReq,
    DHidden,
    STEPS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x = tl.arange(0, BLOCK)
    # Integer casts match copy_ into the graph buffer's destination dtype.
    tl.store(DSeq + row, tl.load(Seq + row))
    tl.store(DPos + row, tl.load(Pos + row))
    tl.store(DProb + row, tl.load(Prob + row))
    tl.store(DToken + row, tl.load(Token + row))
    tl.store(DReq + row, tl.load(Req + row))
    loc = tl.load(Loc + row * STEPS + x, x < STEPS, other=0)
    tl.store(DLoc + row * STEPS + x, loc, x < STEPS)
    hidden = tl.load(Hidden + row * WIDTH + x, x < WIDTH, other=0)
    tl.store(DHidden + row * WIDTH + x, hidden, x < WIDTH)


def try_copy_draft_inputs(buffers, batch, steps):
    """Return False for layouts outside the measured top-k=1, small-batch path."""
    bs = batch.batch_size
    hidden = batch.spec_info.hidden_states
    if not 0 < bs <= 8 or hidden is None or buffers.hidden_states is None:
        return False
    if hidden.ndim != 2 or hidden.shape[0] != bs:
        return False
    width = hidden.shape[1]
    if not 0 < width <= 16384 or buffers.hidden_states.shape[1] != width:
        return False
    src = (
        batch.seq_lens,
        batch.out_cache_loc,
        batch.positions,
        batch.spec_info.topk_p,
        batch.spec_info.topk_index,
        batch.req_pool_indices,
        hidden,
    )
    dst = (
        buffers.seq_lens,
        buffers.out_cache_loc,
        buffers.positions,
        buffers.topk_p,
        buffers.topk_index,
        buffers.req_pool_indices,
        buffers.hidden_states,
    )
    sizes = (bs, bs * steps, bs, bs, bs, bs, bs * width)
    if any(t.numel() != n for t, n in zip(src, sizes)):
        return False
    if any(t.numel() < n for t, n in zip(dst, sizes)):
        return False
    if any(not t.is_cuda or not t.is_contiguous() for t in (*src, *dst)):
        return False
    if any(t.device != hidden.device for t in (*src, *dst)):
        return False
    # Do not introduce floating-point conversion semantics beyond exact copies.
    if src[3].dtype != dst[3].dtype or hidden.dtype != buffers.hidden_states.dtype:
        return False
    _copy_draft_inputs[(bs,)](
        *src,
        *dst,
        STEPS=steps,
        WIDTH=width,
        BLOCK=triton.next_power_of_2(max(width, steps)),
        num_warps=4,
    )
    return True
