"""Shared-window sparse MLA for a draft block's sliding-window layers.

A request's draft block rows sit at consecutive positions, so consecutive rows'
windows ``[max(p - window_left, floor), p]`` differ only by a few keys at each end.
Rows are taken ``group`` at a time: the keys every row of a group attends (the
common window) are read once for the whole group, with the group's rows folded
into the head dimension, and each row's few remaining keys (its edge) are read
on their own. The two partial softmaxes are merged with their LSEs. Keys follow
the draft KV ring layout of ``mla_window_sparse_indices`` (no DCP).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _shared_window_indices_kernel(
    REQ_POOL_INDICES,
    POSITIONS,
    MIN_POS,
    COMMON,
    COMMON_COUNTS,
    EDGE,
    EDGE_COUNTS,
    ROWS_PER_REQ: tl.constexpr,
    GROUP: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    RING_ROWS_PER_REQ: tl.constexpr,
    COMMON_K: tl.constexpr,
    EDGE_K: tl.constexpr,
    BLOCK_COMMON: tl.constexpr,
    BLOCK_EDGE: tl.constexpr,
):
    group = tl.program_id(0)
    groups_per_req = ROWS_PER_REQ // GROUP
    request = group // groups_per_req
    first_row = request * ROWS_PER_REQ + (group % groups_per_req) * GROUP
    slot = tl.load(REQ_POOL_INDICES + request).to(tl.int64)
    floor = tl.maximum(tl.load(MIN_POS + slot).to(tl.int64), 0)
    ring = 1 + slot * RING_ROWS_PER_REQ
    first = tl.load(POSITIONS + first_row).to(tl.int64)
    last = tl.load(POSITIONS + first_row + GROUP - 1).to(tl.int64)
    # Every row of the group attends [common_lo, first].
    common_lo = tl.maximum(last - WINDOW_LEFT, floor)
    offs = tl.arange(0, BLOCK_COMMON)
    key = common_lo + offs.to(tl.int64)
    valid = (offs < COMMON_K) & (key <= first)
    rows = ring + key % RING_ROWS_PER_REQ
    tl.store(
        COMMON + group * COMMON_K + offs,
        tl.where(valid, rows, -1).to(tl.int32),
        mask=offs < COMMON_K,
    )
    tl.store(COMMON_COUNTS + group, tl.sum(valid.to(tl.int32), axis=0))
    # A row's edge: its older keys before common_lo, then its keys after first.
    edge_offs = tl.arange(0, BLOCK_EDGE)
    for k in tl.static_range(GROUP):
        row = first_row + k
        position = tl.load(POSITIONS + row).to(tl.int64)
        lo = tl.maximum(position - WINDOW_LEFT, floor)
        older = tl.maximum(common_lo - lo, 0)
        key = tl.where(
            edge_offs < older, lo + edge_offs, first + 1 + (edge_offs - older)
        )
        valid = (edge_offs < EDGE_K) & (key <= position) & (key >= lo)
        tl.store(
            EDGE + row * EDGE_K + edge_offs,
            tl.where(valid, ring + key % RING_ROWS_PER_REQ, -1).to(tl.int32),
            mask=edge_offs < EDGE_K,
        )
        tl.store(EDGE_COUNTS + row, tl.sum(valid.to(tl.int32), axis=0))


def shared_window_top_k(window_left: int, group: int) -> tuple[int, int]:
    """Index-list widths, padded to 4 for trtllm-gen sparse MLA: the common
    window's (at most ``window_left + 2 - group`` keys) and an edge's (at most
    ``group - 1``)."""
    common = window_left + 2 - group
    return -(-common // 4) * 4, max(4, -(-(group - 1) // 4) * 4)


def mla_shared_window_indices(
    *,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    min_pos: torch.Tensor,
    window_left: int,
    ring_rows_per_req: int,
    group: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ring-row index lists (int32, -1 padded) and counts of each group's common
    window, [bs * rows_per_req / group, common_k], and each row's edge,
    [bs * rows_per_req, edge_k]. ``positions`` are request-major and consecutive
    within a request's rows; ``min_pos`` floors each request's window."""
    bs = req_pool_indices.shape[0]
    rows = positions.shape[0]
    assert bs > 0 and rows % bs == 0
    rows_per_req = rows // bs
    assert rows_per_req % group == 0, (rows_per_req, group)
    common_k, edge_k = shared_window_top_k(window_left, group)
    groups = rows // group
    device = positions.device
    common = torch.empty((groups, common_k), dtype=torch.int32, device=device)
    common_counts = torch.empty(groups, dtype=torch.int32, device=device)
    edge = torch.empty((rows, edge_k), dtype=torch.int32, device=device)
    edge_counts = torch.empty(rows, dtype=torch.int32, device=device)
    _shared_window_indices_kernel[(groups,)](
        req_pool_indices,
        positions,
        min_pos,
        common,
        common_counts,
        edge,
        edge_counts,
        ROWS_PER_REQ=rows_per_req,
        GROUP=group,
        WINDOW_LEFT=window_left,
        RING_ROWS_PER_REQ=ring_rows_per_req,
        COMMON_K=common_k,
        EDGE_K=edge_k,
        BLOCK_COMMON=triton.next_power_of_2(common_k),
        BLOCK_EDGE=triton.next_power_of_2(edge_k),
        num_warps=4,
    )
    return common, common_counts, edge, edge_counts


@triton.jit
def _merge_shared_window_kernel(
    COMMON_OUT,
    common_row_stride,
    COMMON_LSE,
    common_lse_stride,
    EDGE_OUT,
    EDGE_LSE,
    OUT,
    HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    DIM: tl.constexpr,
    BASE2: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    group = row // GROUP
    # The group's sequence holds its rows' heads one row after another.
    common_head = (row % GROUP) * HEADS + head
    lse_c = tl.load(COMMON_LSE + group * common_lse_stride + common_head)
    lse_e = tl.load(EDGE_LSE + row * HEADS + head)
    top = tl.maximum(lse_c, lse_e)
    if BASE2:
        w_c = tl.exp2(lse_c - top)
        w_e = tl.exp2(lse_e - top)
    else:
        w_c = tl.exp(lse_c - top)
        w_e = tl.exp(lse_e - top)
    dim = tl.arange(0, DIM)
    out_c = tl.load(
        COMMON_OUT + group * common_row_stride + common_head * DIM + dim
    ).to(tl.float32)
    out_e = tl.load(EDGE_OUT + (row * HEADS + head) * DIM + dim).to(tl.float32)
    # An empty side has lse -inf and weight 0; both empty (no key at all) gives 0.
    total = w_c + w_e
    merged = tl.where(total > 0.0, (out_c * w_c + out_e * w_e) / total, 0.0)
    tl.store(OUT + (row * HEADS + head) * DIM + dim, merged.to(OUT.dtype.element_ty))


def merge_shared_window(
    common_out: torch.Tensor,
    common_lse: torch.Tensor,
    edge_out: torch.Tensor,
    edge_lse: torch.Tensor,
    *,
    group: int,
    base2: bool,
) -> torch.Tensor:
    """softmax-merge each row's common-window and edge attention, [rows, heads, dim].

    ``common_out`` is [groups, kernel_heads, dim] with row ``k`` of a group in
    heads ``k * heads .. (k + 1) * heads``; ``common_lse`` [groups, kernel_heads];
    ``edge_out`` [rows, heads, dim] and ``edge_lse`` [rows, heads], contiguous.
    LSEs are log2 sums when ``base2``, else natural.
    """
    rows, heads, dim = edge_out.shape
    assert common_out.stride(-1) == 1 and common_out.stride(-2) == dim
    assert edge_out.is_contiguous() and edge_lse.is_contiguous()
    assert common_lse.stride(-1) == 1 and dim & (dim - 1) == 0
    out = torch.empty_like(edge_out)
    if rows:
        _merge_shared_window_kernel[(rows, heads)](
            common_out,
            common_out.stride(0),
            common_lse,
            common_lse.stride(0),
            edge_out,
            edge_lse,
            out,
            HEADS=heads,
            GROUP=group,
            DIM=dim,
            BASE2=base2,
            num_warps=4,
        )
    return out


@triton.jit
def _merge_edges_kernel(
    COMMON_OUT,
    common_row_stride,
    COMMON_LSE,
    common_lse_stride,
    QUERY,
    KV,
    kv_row_stride,
    EDGE,
    EDGE_COUNTS,
    OUT,
    scale_log2,
    HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    DIM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    EDGE_K: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    group = row // GROUP
    common_head = (row % GROUP) * HEADS + head
    top = tl.load(COMMON_LSE + group * common_lse_stride + common_head)
    dim = tl.arange(0, DIM)
    acc = tl.load(COMMON_OUT + group * common_row_stride + common_head * DIM + dim)
    acc = acc.to(tl.float32)
    total = tl.where(top == float("-inf"), 0.0, 1.0)
    acc = acc * total
    qdim = tl.arange(0, BLOCK_HEAD_DIM)
    query = tl.load(
        QUERY + (row * HEADS + head) * HEAD_DIM + qdim, mask=qdim < HEAD_DIM, other=0.0
    ).to(tl.float32)
    count = tl.load(EDGE_COUNTS + row)
    # Online softmax over the row's few edge keys, in the kernel's log2 units;
    # a key's value is the first DIM entries of its cache row.
    for e in tl.static_range(EDGE_K):
        if e < count:
            key_row = KV + tl.load(EDGE + row * EDGE_K + e).to(tl.int64) * kv_row_stride
            key = tl.load(key_row + qdim, mask=qdim < HEAD_DIM, other=0.0).to(
                tl.float32
            )
            score = tl.sum(query * key, axis=0) * scale_log2
            new_top = tl.maximum(top, score)
            carry = tl.exp2(top - new_top)
            weight = tl.exp2(score - new_top)
            value = tl.load(key_row + dim).to(tl.float32)
            acc = acc * carry + weight * value
            total = total * carry + weight
            top = new_top
    out = tl.where(total > 0.0, acc / total, 0.0)
    tl.store(OUT + (row * HEADS + head) * DIM + dim, out.to(OUT.dtype.element_ty))


def merge_shared_window_edges(
    common_out: torch.Tensor,
    common_lse: torch.Tensor,
    query: torch.Tensor,
    kv_rows: torch.Tensor,
    edge: torch.Tensor,
    edge_counts: torch.Tensor,
    *,
    group: int,
    softmax_scale: float,
) -> torch.Tensor:
    """``merge_shared_window`` with each row's edge attention computed here, from
    ``query`` [rows, heads, head_dim] and the cache rows ``kv_rows`` [N, head_dim]
    (a key's value is its first ``dim`` entries) instead of a second sparse call.
    ``common_lse`` is log2-based; ``softmax_scale`` is the kernel's bmm1 scale."""
    rows, heads, head_dim = query.shape
    dim = common_out.shape[-1]
    assert query.is_contiguous() and kv_rows.stride(-1) == 1
    assert common_out.stride(-1) == 1 and common_out.stride(-2) == dim
    assert edge.shape[0] == rows and edge_counts.shape == (rows,)
    out = torch.empty((rows, heads, dim), dtype=torch.bfloat16, device=query.device)
    if rows:
        _merge_edges_kernel[(rows, heads)](
            common_out,
            common_out.stride(0),
            common_lse,
            common_lse.stride(0),
            query,
            kv_rows,
            kv_rows.stride(0),
            edge,
            edge_counts,
            out,
            softmax_scale * 1.4426950408889634,
            HEADS=heads,
            GROUP=group,
            DIM=dim,
            HEAD_DIM=head_dim,
            BLOCK_HEAD_DIM=triton.next_power_of_2(head_dim),
            EDGE_K=edge.shape[1],
            num_warps=4,
        )
    return out
