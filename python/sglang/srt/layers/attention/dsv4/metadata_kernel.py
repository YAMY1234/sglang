from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _expand_prefill_casually_ragged_kernel(
    seq_lens_ptr,
    extend_seq_lens_ptr,
    extend_start_loc_ptr,
    req_pool_indices_ptr,
    seq_lens_casual_ptr,
    req_pool_indices_repeated_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    req_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    extend_len = tl.load(extend_seq_lens_ptr + req_id).to(tl.int32)
    start_loc = tl.load(extend_start_loc_ptr + req_id).to(tl.int64)
    seq_len = tl.load(seq_lens_ptr + req_id).to(tl.int32)
    req_pool_index = tl.load(req_pool_indices_ptr + req_id)

    mask = offsets < extend_len
    out_offsets = start_loc + offsets.to(tl.int64)
    seq_lens_casual = seq_len - extend_len + 1 + offsets.to(tl.int32)

    tl.store(seq_lens_casual_ptr + out_offsets, seq_lens_casual, mask=mask)
    tl.store(req_pool_indices_repeated_ptr + out_offsets, req_pool_index, mask=mask)


@triton.jit
def _expand_prefill_casually_fixed_kernel(
    seq_lens_ptr,
    req_pool_indices_ptr,
    seq_lens_casual_ptr,
    req_pool_indices_repeated_ptr,
    num_tokens,
    tokens_per_bs: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_mask = offsets < num_tokens

    req_id = offsets // tokens_per_bs
    local_offsets = offsets - req_id * tokens_per_bs
    seq_lens = tl.load(seq_lens_ptr + req_id, mask=valid_mask, other=0).to(tl.int32)
    req_pool_indices = tl.load(req_pool_indices_ptr + req_id, mask=valid_mask, other=0)

    seq_lens_casual = seq_lens - tokens_per_bs + 1 + local_offsets.to(tl.int32)
    tl.store(seq_lens_casual_ptr + offsets, seq_lens_casual, mask=valid_mask)
    tl.store(req_pool_indices_repeated_ptr + offsets, req_pool_indices, mask=valid_mask)


@triton.jit
def _fill_expand_prefill_padding_kernel(
    req_pool_indices_ptr,
    seq_lens_casual_ptr,
    req_pool_indices_repeated_ptr,
    num_tokens,
    pad_size,
    pad_req_index,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < pad_size
    pad_req_pool_index = tl.load(req_pool_indices_ptr + pad_req_index)

    tl.store(seq_lens_casual_ptr + num_tokens + offsets, 1, mask=mask)
    tl.store(
        req_pool_indices_repeated_ptr + num_tokens + offsets,
        pad_req_pool_index,
        mask=mask,
    )


def expand_prefill_casually_ragged(
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
    req_pool_indices: torch.Tensor,
    num_tokens: int,
    padded_num_tokens: Optional[int],
    max_extend_len: int,
    pad_req_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    total_tokens = (
        padded_num_tokens
        if padded_num_tokens is not None and padded_num_tokens > num_tokens
        else num_tokens
    )
    seq_lens_casual = torch.empty(total_tokens, dtype=torch.int32, device=seq_lens.device)
    req_pool_indices_repeated = torch.empty(
        total_tokens, dtype=req_pool_indices.dtype, device=req_pool_indices.device
    )
    if total_tokens == 0:
        return seq_lens_casual, req_pool_indices_repeated

    bs = seq_lens.shape[0]
    pad_size = total_tokens - num_tokens
    BLOCK_SIZE = 256
    real_blocks = triton.cdiv(max(max_extend_len, 1), BLOCK_SIZE)
    _expand_prefill_casually_ragged_kernel[(bs, real_blocks)](
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
        req_pool_indices,
        seq_lens_casual,
        req_pool_indices_repeated,
        BLOCK_SIZE,
    )
    if pad_size > 0:
        _fill_expand_prefill_padding_kernel[(triton.cdiv(pad_size, BLOCK_SIZE),)](
            req_pool_indices,
            seq_lens_casual,
            req_pool_indices_repeated,
            num_tokens,
            pad_size,
            pad_req_index,
            BLOCK_SIZE,
        )
    return seq_lens_casual, req_pool_indices_repeated


def expand_prefill_casually_fixed_length(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    tokens_per_bs: int,
    num_tokens: int,
    padded_num_tokens: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    total_tokens = (
        padded_num_tokens
        if padded_num_tokens is not None and padded_num_tokens > num_tokens
        else num_tokens
    )
    seq_lens_casual = torch.empty(total_tokens, dtype=torch.int32, device=seq_lens.device)
    req_pool_indices_repeated = torch.empty(
        total_tokens, dtype=req_pool_indices.dtype, device=req_pool_indices.device
    )
    if total_tokens == 0:
        return seq_lens_casual, req_pool_indices_repeated

    assert tokens_per_bs > 0
    assert num_tokens % tokens_per_bs == 0

    BLOCK_SIZE = 256
    _expand_prefill_casually_fixed_kernel[(triton.cdiv(num_tokens, BLOCK_SIZE),)](
        seq_lens,
        req_pool_indices,
        seq_lens_casual,
        req_pool_indices_repeated,
        num_tokens,
        tokens_per_bs,
        BLOCK_SIZE,
    )

    pad_size = total_tokens - num_tokens
    if pad_size > 0:
        pad_req_index = num_tokens // tokens_per_bs - 1
        _fill_expand_prefill_padding_kernel[(triton.cdiv(pad_size, BLOCK_SIZE),)](
            req_pool_indices,
            seq_lens_casual,
            req_pool_indices_repeated,
            num_tokens,
            pad_size,
            pad_req_index,
            BLOCK_SIZE,
        )

    return seq_lens_casual, req_pool_indices_repeated


@triton.jit
def _init_compressed_attn_metadata_kernel(
    seq_lens_ptr,
    positions_ptr,
    raw_out_loc_ptr,
    page_table_ptr,
    c4_out_loc_ptr,
    c4_positions_ptr,
    c4_seq_lens_raw_ptr,
    c4_seq_lens_clamp1_ptr,
    c128_out_loc_ptr,
    c128_positions_ptr,
    c128_seq_lens_clamp1_ptr,
    c128_page_indices_ptr,
    bs,
    max_pages,
    page_size: tl.constexpr,
    c128_max_seq_len: tl.constexpr,
    c128_page_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COMPUTE_PAGE_INDICES: tl.constexpr,
):
    batch_id = tl.program_id(0)
    if batch_id >= bs:
        return

    seq_len = tl.load(seq_lens_ptr + batch_id)
    position = tl.load(positions_ptr + batch_id)
    raw_out_loc = tl.load(raw_out_loc_ptr + batch_id)

    c4_should_compress = (seq_len % 4) == 0
    c4_out_loc = tl.where(c4_should_compress, raw_out_loc // 4, 0)
    c4_positions = position & (~3)
    c4_seq_lens_raw = seq_len // 4
    c4_seq_lens_clamp1 = tl.maximum(c4_seq_lens_raw, 1)

    tl.store(c4_out_loc_ptr + batch_id, c4_out_loc)
    tl.store(c4_positions_ptr + batch_id, c4_positions)
    tl.store(c4_seq_lens_raw_ptr + batch_id, c4_seq_lens_raw)
    tl.store(c4_seq_lens_clamp1_ptr + batch_id, c4_seq_lens_clamp1)

    c128_should_compress = (seq_len % 128) == 0
    c128_out_loc = tl.where(c128_should_compress, raw_out_loc // 128, 0)
    c128_positions = position & (~127)
    c128_seq_lens_raw = seq_len // 128
    c128_seq_lens_clamp1 = tl.maximum(c128_seq_lens_raw, 1)

    tl.store(c128_out_loc_ptr + batch_id, c128_out_loc)
    tl.store(c128_positions_ptr + batch_id, c128_positions)
    tl.store(c128_seq_lens_clamp1_ptr + batch_id, c128_seq_lens_clamp1)

    if COMPUTE_PAGE_INDICES:
        page_indices_base = batch_id * c128_max_seq_len
        for block_start in range(0, c128_max_seq_len, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < c128_max_seq_len

            page_idx = offsets // c128_page_size
            offset_in_page = offsets % c128_page_size

            page_mask = mask & (page_idx < max_pages)
            page_table_vals = tl.load(
                page_table_ptr + batch_id * max_pages + page_idx,
                mask=page_mask,
                other=0,
            )

            c_page_indices_vals = page_table_vals * c128_page_size + offset_in_page

            valid_mask = offsets < c128_seq_lens_raw
            c_page_indices_vals = tl.where(valid_mask, c_page_indices_vals, -1)

            tl.store(
                c128_page_indices_ptr + page_indices_base + offsets,
                c_page_indices_vals,
                mask=mask,
            )


def _init_compressed_attn_metadata_triton(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    raw_out_loc: torch.Tensor,
    page_table: Optional[torch.Tensor] = None,
    page_size: int = 0,
    compute_page_indices: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    bs = seq_lens.shape[0]
    device = seq_lens.device

    c4_out_loc = torch.empty(bs, dtype=torch.int32, device=device)
    c4_positions = torch.empty(bs, dtype=torch.int32, device=device)
    c4_seq_lens_raw = torch.empty(bs, dtype=torch.int32, device=device)
    c4_seq_lens_clamp1 = torch.empty(bs, dtype=torch.int32, device=device)

    c128_out_loc = torch.empty(bs, dtype=torch.int32, device=device)
    c128_positions = torch.empty(bs, dtype=torch.int32, device=device)
    c128_seq_lens_clamp1 = torch.empty(bs, dtype=torch.int32, device=device)

    if compute_page_indices:
        assert (
            page_table is not None
        ), "page_table required when compute_page_indices=True"
        assert page_size > 0, "page_size required when compute_page_indices=True"
        max_pages = page_table.shape[1]
        c128_page_size = page_size // 128
        c128_max_seq_len = c128_page_size * max_pages
        c128_page_indices = torch.empty(
            bs, c128_max_seq_len, dtype=torch.int32, device=device
        )
        BLOCK_SIZE = triton.next_power_of_2(max(c128_page_size, 64))
    else:
        max_pages = 0
        c128_page_size = 1
        c128_max_seq_len = 0
        c128_page_indices = None
        BLOCK_SIZE = 64
        if page_table is None:
            page_table = torch.empty(0, dtype=torch.int32, device=device)

    grid = (bs,)
    _init_compressed_attn_metadata_kernel[grid](
        seq_lens,
        positions,
        raw_out_loc,
        page_table,
        c4_out_loc,
        c4_positions,
        c4_seq_lens_raw,
        c4_seq_lens_clamp1,
        c128_out_loc,
        c128_positions,
        c128_seq_lens_clamp1,
        (
            c128_page_indices
            if c128_page_indices is not None
            else torch.empty(0, dtype=torch.int32, device=device)
        ),
        bs,
        max_pages,
        page_size if page_size > 0 else 128,
        c128_max_seq_len,
        c128_page_size,
        BLOCK_SIZE,
        compute_page_indices,
    )

    return (
        c4_out_loc,
        c4_positions,
        c4_seq_lens_raw,
        c4_seq_lens_clamp1,
        c128_out_loc,
        c128_positions,
        c128_seq_lens_clamp1,
        c128_page_indices,
    )


def init_compression_metadata(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    raw_out_loc: torch.Tensor,
    page_table: Optional[torch.Tensor] = None,
    page_size: int = 0,
    compute_page_indices: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    return _init_compressed_attn_metadata_triton(
        seq_lens,
        positions,
        raw_out_loc,
        page_table,
        page_size,
        compute_page_indices,
    )
