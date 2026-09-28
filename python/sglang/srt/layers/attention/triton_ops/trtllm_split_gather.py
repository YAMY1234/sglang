"""Bit-preserving gather of a TRTLLM decode split's Q and metadata."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_split_inputs(
    Q, Pages, Seq, Order, QOut, PagesOut, SeqOut,
    N: tl.constexpr, QLEN: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr,
    PAGE_COLS: tl.constexpr,
    QS0: tl.constexpr, QS1: tl.constexpr, QS2: tl.constexpr, QS3: tl.constexpr,
    PS0: tl.constexpr, PS1: tl.constexpr, SS0: tl.constexpr, IS0: tl.constexpr,
    Q_BLOCKS: tl.constexpr, PAGE_BLOCKS: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    qrow: tl.constexpr = QLEN * HEADS * DIM
    if pid < Q_BLOCKS:
        offset = pid * BLOCK + lane
        row = offset // qrow
        col = offset % qrow
        valid = offset < N * qrow
        source = tl.load(Order + row * IS0, valid, other=0)
        ptr = (Q + source * QS0 + (col // (HEADS * DIM)) * QS1
               + (col // DIM % HEADS) * QS2 + (col % DIM) * QS3)
        query_value = tl.load(ptr, valid, other=0)
        tl.store(QOut + offset, query_value, valid)
    elif pid < Q_BLOCKS + PAGE_BLOCKS:
        offset = (pid - Q_BLOCKS) * BLOCK + lane
        row = offset // PAGE_COLS
        col = offset % PAGE_COLS
        valid = offset < N * PAGE_COLS
        source = tl.load(Order + row * IS0, valid, other=0)
        page_value = tl.load(Pages + source * PS0 + col * PS1, valid, other=0)
        tl.store(PagesOut + offset, page_value, valid)
    else:
        row = (pid - Q_BLOCKS - PAGE_BLOCKS) * BLOCK + lane
        valid = row < N
        source = tl.load(Order + row * IS0, valid, other=0)
        seq_value = tl.load(Seq + source * SS0, valid, other=0)
        tl.store(SeqOut + row, seq_value, valid)


def gather_split_inputs(query, block_tables, seq_lens, indices):
    """Equivalent to three index_select(0, indices), with one GPU launch.

    Query is [requests, q_len, heads, head_dim]. Integer views of identical
    element width preserve every FP8/BF16 bit, including special values.
    All shapes/strides are host metadata: capture does not read GPU scalars.
    Outputs own their storage, as in index_select; no layer-global aliasing.
    """
    assert query.is_cuda and query.ndim == 4
    assert block_tables.ndim == 2 and seq_lens.ndim == indices.ndim == 1
    assert query.device == block_tables.device == seq_lens.device == indices.device
    assert query.shape[0] == block_tables.shape[0] == seq_lens.shape[0]
    assert block_tables.shape[1] > 0
    assert indices.dtype in (torch.int32, torch.int64)
    n = indices.numel()
    qout = torch.empty((n, *query.shape[1:]), dtype=query.dtype, device=query.device)
    pout = torch.empty((n, block_tables.shape[1]), dtype=block_tables.dtype, device=query.device)
    sout = torch.empty((n,), dtype=seq_lens.dtype, device=query.device)
    if not n:
        return qout, pout, sout
    raw_dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[query.element_size()]
    block = 1024
    q_blocks = triton.cdiv(qout.numel(), block)
    page_blocks = triton.cdiv(pout.numel(), block)
    _gather_split_inputs[(q_blocks + page_blocks + triton.cdiv(n, block),)](
        query.view(raw_dtype), block_tables, seq_lens, indices,
        qout.view(raw_dtype), pout, sout,
        n, *query.shape[1:], block_tables.shape[1],
        *query.stride(), *block_tables.stride(), seq_lens.stride(0), indices.stride(0),
        q_blocks, page_blocks, block, num_warps=4,
    )
    return qout, pout, sout
