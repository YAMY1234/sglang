"""One-launch replacements for the torch-op chains that build DeepSeek-V4 attention
metadata every step. Each function returns tensors bitwise equal to the chain it
replaces (same values, dtypes and shapes); tests compare them against those chains."""

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.attn_metadata_kernels import (
    PageTablePositionsResult,
)


@triton.jit
def _expand_same_length_kernel(
    SEQ,
    REQ,
    SEQ_EXT,
    CASUAL,
    REQ_REP,
    num_rows,
    QO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < num_rows
    r = i // QO
    j = i % QO
    ext = tl.load(SEQ + r, mask=m, other=0) + QO
    tl.store(SEQ_EXT + r, ext, mask=m & (j == 0))
    tl.store(CASUAL + i, ext + (j - QO + 1), mask=m)
    tl.store(REQ_REP + i, tl.load(REQ + r, mask=m, other=0), mask=m)


def expand_same_length(
    seq_lens: torch.Tensor, req_pool_indices: torch.Tensor, qo_len: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``seq_lens + qo_len`` and its expansion to ``qo_len`` causal rows per request:
    what target verify builds with ``seq_lens + qo_len`` followed by
    ``expand_extend_with_same_length``."""
    bs = seq_lens.shape[0]
    rows = bs * qo_len
    seq_ext = torch.empty_like(seq_lens)
    casual = torch.empty(
        rows,
        dtype=torch.promote_types(seq_lens.dtype, torch.int32),
        device=seq_lens.device,
    )
    req_rep = torch.empty(
        rows, dtype=req_pool_indices.dtype, device=req_pool_indices.device
    )
    if rows:
        block = 256
        _expand_same_length_kernel[(triton.cdiv(rows, block),)](
            seq_lens, req_pool_indices, seq_ext, casual, req_rep, rows, qo_len, block
        )
    return seq_ext, casual, req_rep


@triton.jit
def _page_table_positions_2d_kernel(
    REQ_TO_TOKEN,
    REQS,
    LENS,
    OUT_LENS,
    POS,
    PAGES,
    SWA,
    stride,
    num_pages,
    PAGE_SHIFT: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    if tile == 0:
        length = tl.load(LENS + row).to(tl.int32)
        tl.store(OUT_LENS + row, length)
        tl.store(POS + row, length - 1)
        tl.store(SWA + row, tl.minimum(length, WINDOW))
    req = tl.load(REQS + row).to(tl.int64)
    p = tile * BLOCK + tl.arange(0, BLOCK)
    m = p < num_pages
    slot = tl.load(
        REQ_TO_TOKEN + req * stride + p.to(tl.int64) * (1 << PAGE_SHIFT),
        mask=m,
        other=0,
    )
    # Arithmetic shift is floor division, as torch's // is.
    tl.store(
        PAGES + row.to(tl.int64) * num_pages + p, (slot >> PAGE_SHIFT).to(tl.int32), m
    )


def build_page_table_positions_2d(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices_repeated: torch.Tensor,
    seq_lens_casual: torch.Tensor,
    max_seq_len: int,
    page_size: int,
    swa_window: int,
) -> PageTablePositionsResult:
    """``build_page_table_positions`` with the pages of a row spread over the grid
    (the one-program-per-row kernel walks a long context's pages serially)."""
    assert page_size > 0 and page_size & (page_size - 1) == 0, page_size
    assert req_to_token.stride(1) == 1
    rows = seq_lens_casual.shape[0]
    pages = triton.cdiv(max_seq_len, page_size)
    kw = dict(device=seq_lens_casual.device, dtype=torch.int32)
    lengths, positions, swa = [torch.empty(rows, **kw) for _ in range(3)]
    table = torch.empty((rows, pages), **kw)
    block = 256
    if rows:
        _page_table_positions_2d_kernel[(rows, max(triton.cdiv(pages, block), 1))](
            req_to_token,
            req_pool_indices_repeated,
            seq_lens_casual,
            lengths,
            positions,
            table,
            swa,
            req_to_token.stride(0),
            pages,
            page_size.bit_length() - 1,
            swa_window,
            block,
        )
    return PageTablePositionsResult(
        seq_lens_casual=lengths,
        positions_casual=positions,
        page_table=table,
        swa_topk_lengths=swa,
    )


@triton.jit
def _gather_int32_kernel(MAP, LOC, OUT, num_loc, num_out, map_size, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = i < num_loc
    loc = tl.load(LOC + i, mask=live, other=0).to(tl.int64)
    # torch indexing wraps negative indices.
    loc = tl.where(loc < 0, loc + map_size, loc)
    val = tl.load(MAP + loc, mask=live, other=0).to(tl.int32)
    tl.store(OUT + i, tl.where(live, val, 0), mask=i < num_out)


def gather_int32(
    mapping: torch.Tensor, loc: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``mapping[loc].to(torch.int32)``; with ``out``, written to ``out[:len(loc)]``
    and ``out[len(loc):]`` zeroed (the CUDA-graph SWA write-location buffer)."""
    n = loc.shape[0]
    if out is None:
        out = torch.empty(n, dtype=torch.int32, device=loc.device)
    assert out.dtype == torch.int32 and out.shape[0] >= n and out.is_contiguous()
    total = out.shape[0]
    if total:
        block = 512
        _gather_int32_kernel[(triton.cdiv(total, block),)](
            mapping, loc, out, n, total, mapping.shape[0], block
        )
    return out[:n]


@triton.jit
def _expand_index_pair_kernel(
    PT,
    OUT_A,
    OUT_B,
    n_pages,
    pt_stride,
    BPP_A: tl.constexpr,
    BPP_B: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    p = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = p < n_pages
    page = tl.load(PT + row * pt_stride + p, mask=m, other=0).to(tl.int64)
    ka = tl.arange(0, BPP_A)
    tl.store(
        OUT_A + row * (n_pages * BPP_A) + p[:, None] * BPP_A + ka[None, :],
        (page[:, None] * BPP_A + ka[None, :]).to(tl.int32),
        mask=m[:, None],
    )
    kb = tl.arange(0, BPP_B)
    tl.store(
        OUT_B + row * (n_pages * BPP_B) + p[:, None] * BPP_B + kb[None, :],
        (page[:, None] * BPP_B + kb[None, :]).to(tl.int32),
        mask=m[:, None],
    )


def expand_index_page_table_pair(
    page_table: torch.Tensor, bpp_a: int, bpp_b: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Both low-ratio indexer block tables (``expand_index_page_table`` at two
    blocks-per-page) from one read of ``page_table``."""
    assert page_table.dim() == 2 and page_table.stride(1) == 1
    for b in (bpp_a, bpp_b):
        assert b > 1 and b & (b - 1) == 0, b
    bs, n = page_table.shape
    kw = dict(dtype=torch.int32, device=page_table.device)
    out_a = torch.empty((bs, n * bpp_a), **kw)
    out_b = torch.empty((bs, n * bpp_b), **kw)
    if bs and n:
        block = 128
        _expand_index_pair_kernel[(bs, triton.cdiv(n, block))](
            page_table, out_a, out_b, n, page_table.stride(0), bpp_a, bpp_b, block
        )
    return out_a, out_b


_ROW_GROUP_CACHE: Dict[Tuple[torch.device, int], torch.Tensor] = {}


def row_group_indices(
    num_rows: int, group: int, device: torch.device, max_rows: int
) -> Optional[torch.Tensor]:
    """``arange(num_rows, int32) // group`` as a view of a table built once per
    (device, group). None while a CUDA graph is capturing and the table is not
    built yet: capture does not run the fill, so the caller builds its own."""
    key = (torch.device(device), group)
    table = _ROW_GROUP_CACHE.get(key)
    if table is None or table.shape[0] < num_rows:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return None
        size = max(num_rows, max_rows)
        table = torch.arange(size, dtype=torch.int32, device=device) // group
        _ROW_GROUP_CACHE[key] = table
    return table[:num_rows]
