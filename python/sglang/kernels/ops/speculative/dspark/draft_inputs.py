"""The DSpark draft forward's per-step inputs in one launch: the bonus token into column 0 of
the draft block ids, and the first ``query_token_num`` columns of the verify window's
positions and cache locations as contiguous rows. Same values as the three torch copies."""

import torch
import triton
import triton.language as tl


@triton.jit
def _draft_inputs_kernel(
    bonus_ptr,
    block_ids_ptr,
    block_ids_stride,
    positions_ptr,
    positions_stride,
    loc_ptr,
    loc_stride,
    positions_out_ptr,
    loc_out_ptr,
    Q: tl.constexpr,
    BLOCK: tl.constexpr,
    WRITE_BONUS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if WRITE_BONUS:
        tl.store(block_ids_ptr + row * block_ids_stride, tl.load(bonus_ptr + row))
    cols = tl.arange(0, BLOCK)
    mask = cols < Q
    pos = tl.load(positions_ptr + row * positions_stride + cols, mask=mask)
    tl.store(positions_out_ptr + row * Q + cols, pos, mask=mask)
    loc = tl.load(loc_ptr + row * loc_stride + cols, mask=mask)
    tl.store(loc_out_ptr + row * Q + cols, loc, mask=mask)


def draft_inputs(
    bonus: torch.Tensor,
    draft_block_ids: torch.Tensor,
    positions_2d: torch.Tensor,
    verify_cache_loc_2d: torch.Tensor,
    query_token_num: int,
    write_bonus: bool,
):
    """Returns (draft_positions, draft_cache_loc), each [bs * query_token_num]; writes
    ``bonus`` into ``draft_block_ids[:, 0]`` when ``write_bonus``."""
    bs = draft_block_ids.shape[0]
    q = query_token_num
    positions_out = torch.empty(bs * q, dtype=positions_2d.dtype, device=positions_2d.device)
    loc_out = torch.empty(
        bs * q, dtype=verify_cache_loc_2d.dtype, device=verify_cache_loc_2d.device
    )
    if bs:
        positions_2d = positions_2d if positions_2d.stride(1) == 1 else positions_2d.contiguous()
        if verify_cache_loc_2d.stride(1) != 1:
            verify_cache_loc_2d = verify_cache_loc_2d.contiguous()
        bonus = bonus.contiguous()
        _draft_inputs_kernel[(bs,)](
            bonus,
            draft_block_ids,
            draft_block_ids.stride(0),
            positions_2d,
            positions_2d.stride(0),
            verify_cache_loc_2d,
            verify_cache_loc_2d.stride(0),
            positions_out,
            loc_out,
            Q=q,
            BLOCK=triton.next_power_of_2(q),
            WRITE_BONUS=write_bonus,
            num_warps=1,
        )
    return positions_out, loc_out
