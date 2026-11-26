"""mla_window_sparse_indices: per query row, the flat paged-cache rows of the
window keys [p - window_left, p] that one DCP rank owns, against a Python
reference of the cyclic DCP layout (position pos on rank pos % N at local slot
req_to_token[pos] // N, page through v2p when given). The union over ranks
must be exactly the window, each key once."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.mla_shared_window import (
    merge_shared_window,
    mla_shared_window_indices,
)
from sglang.kernels.ops.attention.mla_window_indices import (
    mla_window_sparse_indices,
    mla_window_top_k,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE = 64


def _layout(seq_lens, dcp, with_v2p, seed):
    """req_to_token of page-aligned requests in a shuffled pool, so slot % dcp
    cycles with position, and a shuffled virtual->physical page table."""
    g = torch.Generator().manual_seed(seed)
    cycle = PAGE * dcp
    runs = [-(-s // cycle) for s in seq_lens]
    order = torch.randperm(sum(runs) + 1, generator=g)[: len(seq_lens)] + 1
    req_to_token = torch.full(
        (len(seq_lens) + 3, max(seq_lens) + 16), -1, dtype=torch.int32
    )
    pool_idx = torch.randperm(req_to_token.shape[0], generator=g)[: len(seq_lens)]
    base = 0
    for i, s in enumerate(seq_lens):
        base = int(order[i]) * cycle * 64
        req_to_token[pool_idx[i], :s] = torch.arange(base, base + s, dtype=torch.int32)
    num_pages = int(req_to_token.max()) // dcp // PAGE + 1
    v2p = torch.randperm(num_pages, generator=g).to(torch.int64) if with_v2p else None
    return req_to_token, pool_idx, v2p


def _reference(req_to_token, pool_idx, positions, q_len, window_left, dcp, rank, v2p):
    rows = []
    for r, p in enumerate(positions.tolist()):
        req = int(pool_idx[r // q_len])
        keys = []
        for pos in range(max(0, p - window_left), p + 1):
            if pos % dcp != rank:
                continue
            slot = int(req_to_token[req, pos]) // dcp
            page = slot // PAGE
            if v2p is not None:
                page = int(v2p[page])
            keys.append(page * PAGE + slot % PAGE)
        rows.append(keys)
    return rows


@pytest.mark.parametrize("dcp", [1, 8])
@pytest.mark.parametrize("window_left", [4095, 127])
@pytest.mark.parametrize("with_v2p", [False, True])
def test_window_indices(dcp, window_left, with_v2p):
    q_len = 8
    ctx = [0, 5, window_left - 3, window_left, window_left + 1, 3 * window_left + 77]
    seq_lens = [c + q_len for c in ctx]
    req_to_token, pool_idx, v2p = _layout(
        seq_lens, dcp, with_v2p, seed=dcp + window_left
    )
    positions = torch.tensor(
        [c + i for c in ctx for i in range(q_len)], dtype=torch.int64
    )
    top_k = mla_window_top_k(window_left, dcp)
    union = [[] for _ in positions]
    for rank in range(dcp):
        indices, counts = mla_window_sparse_indices(
            req_to_token=req_to_token.cuda(),
            req_pool_indices=pool_idx.cuda(),
            positions=positions.cuda(),
            window_left=window_left,
            dcp_size=dcp,
            dcp_rank=rank,
            page_size=PAGE,
            v2p=None if v2p is None else v2p.cuda(),
        )
        assert indices.shape == (positions.numel(), top_k)
        ref = _reference(
            req_to_token, pool_idx, positions, q_len, window_left, dcp, rank, v2p
        )
        indices, counts = indices.cpu(), counts.cpu()
        for r, keys in enumerate(ref):
            assert int(counts[r]) == len(keys), (rank, r)
            assert indices[r, : len(keys)].tolist() == keys, (rank, r)
            assert (indices[r, len(keys) :] == -1).all(), (rank, r)
            union[r] += [(rank, k) for k in keys]
    for r, p in enumerate(positions.tolist()):
        assert len(union[r]) == min(p, window_left) + 1


def _ring_case(q_len, window_left, ring_rows, seed):
    """Requests at the start of a sequence, inside the window, past it, and with a
    floor (min_pos) inside their window, over a draft KV ring."""
    g = torch.Generator().manual_seed(seed)
    anchors = torch.tensor(
        [0, 3, window_left - 2, window_left, 5 * window_left + 7, 9 * window_left + 1]
    )
    slots = torch.randperm(len(anchors) + 3, generator=g)[: len(anchors)]
    min_pos = torch.zeros(len(anchors) + 3, dtype=torch.int32)
    min_pos[slots[4]] = int(anchors[4])
    min_pos[slots[5]] = int(anchors[5]) - window_left // 2
    positions = (anchors[:, None] + torch.arange(q_len)).flatten()
    return slots.cuda(), positions.cuda(), min_pos.cuda()


@pytest.mark.parametrize("group", [2, 3, 6])
@pytest.mark.parametrize("window_left", [1024, 127])
def test_shared_window_rows_read_their_window(group, window_left):
    """A row's group common window plus its edge is its own window, each key once."""
    q_len, ring_rows = 6, window_left + 7
    slots, positions, min_pos = _ring_case(q_len, window_left, ring_rows, group)
    ref, ref_counts = mla_window_sparse_indices(
        req_to_token=torch.zeros(1, 1, dtype=torch.int32, device="cuda"),
        req_pool_indices=slots,
        positions=positions,
        window_left=window_left,
        dcp_size=1,
        dcp_rank=0,
        page_size=PAGE,
        ring_rows_per_req=ring_rows,
        min_pos=min_pos,
    )
    common, common_counts, edge, edge_counts = mla_shared_window_indices(
        req_pool_indices=slots,
        positions=positions,
        min_pos=min_pos,
        window_left=window_left,
        ring_rows_per_req=ring_rows,
        group=group,
    )
    assert common.shape[0] == positions.numel() // group
    assert (edge_counts <= group - 1).all()
    ref, ref_counts = ref.cpu(), ref_counts.cpu()
    common, common_counts = common.cpu(), common_counts.cpu()
    edge, edge_counts = edge.cpu(), edge_counts.cpu()
    for r in range(positions.numel()):
        c, e = int(common_counts[r // group]), int(edge_counts[r])
        keys = common[r // group, :c].tolist() + edge[r, :e].tolist()
        assert sorted(keys) == sorted(ref[r, : ref_counts[r]].tolist()), r
        assert len(set(keys)) == len(keys) and c >= 1, r
        assert (common[r // group, c:] == -1).all() and (edge[r, e:] == -1).all()


@pytest.mark.parametrize("base2", [True, False])
def test_merge_shared_window_matches_softmax_merge(base2):
    gen = torch.Generator(device="cuda").manual_seed(int(base2))
    groups, group, heads, dim = 7, 2, 24, 512
    rows = groups * group
    common_out = torch.randn(groups, group * heads, dim, device="cuda", generator=gen)
    common_lse = torch.randn(groups, group * heads, device="cuda", generator=gen) * 4
    edge_out = torch.randn(rows, heads, dim, device="cuda", generator=gen)
    edge_lse = torch.randn(rows, heads, device="cuda", generator=gen) * 4
    # An empty edge carries lse -inf.
    edge_lse[3] = -torch.inf
    out = merge_shared_window(
        common_out.bfloat16(),
        common_lse,
        edge_out.bfloat16(),
        edge_lse,
        group=group,
        base2=base2,
    )
    exp = torch.exp2 if base2 else torch.exp
    c_out = common_out.bfloat16().float().view(rows, heads, dim)
    c_lse = common_lse.view(rows, heads)
    top = torch.maximum(c_lse, edge_lse)
    w_c, w_e = exp(c_lse - top)[..., None], exp(edge_lse - top)[..., None]
    ref = (c_out * w_c + edge_out.bfloat16().float() * w_e) / (w_c + w_e)
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out[3].float(), c_out[3], rtol=0, atol=0)


@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 10, reason="trtllm-gen sparse MLA"
)
@pytest.mark.parametrize("edges_in_merge", [True, False])
def test_shared_window_attention_matches_per_row(edges_in_merge):
    """The shared-window draft attention equals the per-row window attention up to
    the BF16 rounding of the two partial outputs it merges."""
    from sglang.srt.layers.attention.draft_shared_window import (
        shared_window_indices,
        shared_window_sparse_attention,
    )
    from sglang.srt.layers.attention.dsa.dsa_dcp import dcp_trtllm_sparse_attention
    from sglang.srt.layers.attention.trtllm_mla_backend import (
        make_persistent_multi_ctas_kv_counter_buffer,
    )
    from sglang.srt.mem_cache.layout.paged_view import paged_row_view

    q_len, heads, head_dim, window_left, ring_rows, group = 6, 24, 576, 127, 134, 2
    slots, positions, min_pos = _ring_case(q_len, window_left, ring_rows, 11)
    bs, rows = slots.numel(), positions.numel()
    torch.manual_seed(0)
    cache_rows = -(-(1 + int(slots.max() + 1) * ring_rows) // PAGE) * PAGE
    kv = (torch.randn(cache_rows, 1, head_dim, device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    query = (torch.randn(bs, q_len, heads, head_dim, device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    kwargs = dict(
        kv_cache=paged_row_view(kv, PAGE).unsqueeze(1),
        workspace_buffer=torch.zeros(128 << 20, dtype=torch.int8, device="cuda"),
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        max_seq_len=4 * window_left,
        bmm1_scale=192**-0.5,
        backend="trtllm-gen",
        skip_softmax_threshold_scale_factor=None,
        multi_ctas_kv_counter_buffer=make_persistent_multi_ctas_kv_counter_buffer(
            torch.device("cuda"), 128, max_batch_size=rows
        ),
    )
    indptr = torch.arange(rows + 1, dtype=torch.int32, device="cuda")
    indices, counts = mla_window_sparse_indices(
        req_to_token=torch.zeros(1, 1, dtype=torch.int32, device="cuda"),
        req_pool_indices=slots,
        positions=positions,
        window_left=window_left,
        dcp_size=1,
        dcp_rank=0,
        page_size=PAGE,
        ring_rows_per_req=ring_rows,
        min_pos=min_pos,
    )
    ref, _ = dcp_trtllm_sparse_attention(
        query=query.view(rows, 1, heads, head_dim),
        block_tables=indices.unsqueeze(1),
        seq_lens=counts,
        cu_seqlens_q=indptr,
        sparse_mla_top_k=indices.shape[1],
        need_lse=True,
        **kwargs,
    )
    out = shared_window_sparse_attention(
        query=query,
        indices=shared_window_indices(
            req_pool_indices=slots,
            positions=positions,
            min_pos=min_pos,
            window_left=window_left,
            ring_rows_per_req=ring_rows,
            group=group,
        ),
        group=group,
        row_indptr=indptr,
        edges_in_merge=edges_in_merge,
        **kwargs,
    )
    torch.testing.assert_close(
        out.float(), ref.view(rows, heads, -1).float(), rtol=2e-2, atol=2e-2
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
