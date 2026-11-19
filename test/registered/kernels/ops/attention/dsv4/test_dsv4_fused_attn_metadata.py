"""The fused DSV4 attention-metadata kernels are bitwise the torch chains they replace."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.attn_metadata_kernels import (
    build_page_table_positions,
    build_page_table_positions_triton,
)
from sglang.kernels.ops.attention.dsv4.fused_attn_metadata import (
    build_page_table_positions_2d,
    expand_index_page_table_pair,
    expand_same_length,
    first_row_offsets,
    gather_int32,
    prefill_k_slots_fused,
    row_group_indices,
)
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.kvcache.dsv4_request_window_copy import commit_packed_rows
from sglang.kernels.ops.attention.dsv4.metadata_kernel import (
    build_low_ratio_metadata,
)
from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    _low_ratio_compression_metadata,
    _low_ratio_sparse_buffers,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.metadata import expand_index_page_table
from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import prefill_k_slots
from sglang.srt.mem_cache.dsv41_request_window import _first_row_offsets
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(not is_cuda(), reason="CUDA Triton kernels")

DEVICE = "cuda"
PAGE_SIZE = 256
INDEX_PAGE_SIZE = 64
TOPK = 512
SWA_WINDOW = 128
NUM_SLOTS = 1 << 22


def _same(a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.dtype == b.dtype and a.shape == b.shape, (a.dtype, b.dtype, a.shape, b.shape)
    assert torch.equal(a, b)


def _req_to_token(num_reqs: int, width: int, gen) -> torch.Tensor:
    # Built on the device: a host-side fill of up to 1M x 1024 takes minutes.
    device_gen = torch.Generator(device=DEVICE).manual_seed(
        int(torch.randint(0, 1 << 31, (1,), generator=gen))
    )
    return torch.randint(
        0, NUM_SLOTS, (num_reqs, width), generator=device_gen, dtype=torch.int32, device=DEVICE
    )


def _verify_rows(bs: int, qo: int, gen):
    seq_lens = torch.randint(1, 200_000, (bs,), generator=gen, dtype=torch.int32)
    req = torch.randperm(1024, generator=gen)[:bs].to(torch.int64)
    return seq_lens.to(DEVICE), req.to(DEVICE)


def _prefill_rows(total: int, gen):
    """Causal rows of a mixed batch: random prefixes, extends summing to total."""
    cuts = torch.sort(torch.randint(1, total, (7,), generator=gen)).values.tolist()
    extends = [b - a for a, b in zip([0] + cuts, cuts + [total]) if b > a]
    prefixes = torch.randint(0, 120_000, (len(extends),), generator=gen).tolist()
    prefixes[0] = 0
    casual = torch.cat(
        [torch.arange(p + 1, p + e + 1, dtype=torch.int32) for p, e in zip(prefixes, extends)]
    )
    req = torch.cat(
        [torch.full((e,), i * 3 + 1, dtype=torch.int64) for i, e in enumerate(extends)]
    )
    return casual.to(DEVICE), req.to(DEVICE)


@pytest.mark.parametrize("bs", [1, 2, 7, 64, 118, 160])
@pytest.mark.parametrize("seq_dtype", [torch.int32, torch.int64])
def test_expand_same_length(bs, seq_dtype):
    gen = torch.Generator().manual_seed(bs)
    qo = 4
    seq_lens, req = _verify_rows(bs, qo, gen)
    seq_lens = seq_lens.to(seq_dtype)
    backend = SimpleNamespace(cuda_int32_kwargs=dict(device=DEVICE, dtype=torch.int32))
    ref_ext = seq_lens + qo
    ref_casual, ref_req = DeepseekV4AttnBackend.expand_extend_with_same_length(
        backend, bs=bs, qo_len=qo, seq_lens=ref_ext, req_pool_indices=req
    )
    ext, casual, req_rep = expand_same_length(
        seq_lens=seq_lens, req_pool_indices=req, qo_len=qo
    )
    _same(ext, ref_ext)
    _same(casual, ref_casual)
    _same(req_rep, ref_req)


@pytest.mark.parametrize(
    "rows_kind,size,max_seq_len",
    [
        ("verify", 1, 262144),
        ("verify", 118, 262144),
        ("verify", 160, 1048576),
        ("prefill", 3000, 131072),
        ("prefill", 65536, 131072),
    ],
)
def test_page_table_and_low_ratio(rows_kind, size, max_seq_len):
    gen = torch.Generator().manual_seed(size)
    if rows_kind == "verify":
        seq_lens, req = _verify_rows(size, 4, gen)
        _, casual, req_rep = expand_same_length(
            seq_lens=seq_lens, req_pool_indices=req, qo_len=4
        )
    else:
        casual, req_rep = _prefill_rows(size, gen)
    req_to_token = _req_to_token(1024, max_seq_len, gen)
    req_to_token[3, :PAGE_SIZE] = -5  # torch floor-divides negatives
    if rows_kind == "prefill":
        req_rep[:2] = 3
    kw = dict(
        req_to_token=req_to_token,
        req_pool_indices_repeated=req_rep,
        seq_lens_casual=casual,
        max_seq_len=max_seq_len,
        page_size=PAGE_SIZE,
        swa_window=SWA_WINDOW,
    )
    ref = build_page_table_positions(**kw)
    old = build_page_table_positions_triton(**kw)
    new = build_page_table_positions_2d(**kw)
    for name in ("seq_lens_casual", "positions_casual", "page_table", "swa_topk_lengths"):
        _same(getattr(new, name), getattr(ref, name))
        if rows_kind == "verify" or name != "page_table":
            _same(getattr(new, name), getattr(old, name))

    rows = casual.shape[0]
    out_loc = torch.randint(-1, NUM_SLOTS, (rows,), generator=gen).to(DEVICE)
    fused = build_low_ratio_metadata(new.seq_lens_casual, out_loc, TOPK)
    for ratio, base in ((1, 0), (2, 4)):
        loc, lens = _low_ratio_compression_metadata(ratio, new.seq_lens_casual, out_loc)
        sparse, pages, _ = _low_ratio_sparse_buffers(lens, TOPK, rows_kind == "prefill")
        for got, want in zip(fused[base : base + 4], (loc, lens, sparse, pages)):
            _same(got, want)

    table = new.page_table
    c1, c2 = expand_index_page_table_pair(table, PAGE_SIZE // INDEX_PAGE_SIZE, PAGE_SIZE // 2 // INDEX_PAGE_SIZE)
    for got, ratio in ((c1, 1), (c2, 2)):
        want = expand_index_page_table(
            table,
            full_page_size=PAGE_SIZE,
            compress_ratio=ratio,
            index_page_size=INDEX_PAGE_SIZE,
        )
        _same(got, want)


@pytest.mark.parametrize("n,pad", [(1, 0), (472, 0), (472, 168), (65536, 0)])
def test_gather_int32(n, pad):
    gen = torch.Generator().manual_seed(n + pad)
    mapping = torch.randint(0, 1 << 20, (NUM_SLOTS,), device=DEVICE)
    loc = torch.randint(0, NUM_SLOTS, (n,), generator=gen).to(DEVICE)
    loc[0] = -1  # torch wraps negative indices
    ref = mapping[loc].to(torch.int32)
    _same(gather_int32(mapping=mapping, loc=loc), ref)
    buf = torch.full((n + pad,), 7, dtype=torch.int32, device=DEVICE)
    got = gather_int32(mapping=mapping, loc=loc, out=buf)
    _same(got, ref)
    assert got.data_ptr() == buf.data_ptr()
    assert torch.all(buf[n:] == 0)


@pytest.mark.parametrize("group", [1, 4])
def test_row_group_indices(group):
    for n in (96, 472, 640, 29039):
        ref = torch.arange(n, dtype=torch.int32, device=DEVICE) // group
        _same(row_group_indices(num_rows=n, group=group, device=DEVICE, max_rows=1024), ref)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cached = row_group_indices(num_rows=500, group=group, device=DEVICE, max_rows=1)
        missing = row_group_indices(num_rows=1 << 16, group=group, device=DEVICE, max_rows=1)
    assert cached is not None and missing is None


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("seq_lens", [[1], [3, 0, 7], [110003, 97411, 121577, 1, 2, 88901, 131011, 4]])
def test_prefill_k_slots(ratio, seq_lens):
    gen = torch.Generator().manual_seed(len(seq_lens) * 10 + ratio)
    req_to_token = _req_to_token(64, 131072 + 64, gen)
    req_to_token[5, :8] = -3  # torch floor-divides negatives
    req = torch.randperm(64, generator=gen)[: len(seq_lens)].to(DEVICE)
    req[0] = 5
    kw = dict(req_to_token=req_to_token, req_pool_indices=req, seq_lens_cpu=seq_lens, ratio=ratio)
    with envs.SGLANG_OPT_DSV4_FUSED_ATTN_METADATA.override(False):
        ref = prefill_k_slots(**kw)
    _same(prefill_k_slots_fused(**kw), ref)


@pytest.mark.parametrize("n", [1, 5, 1023, 1024, 1025, 32768, 65537])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_first_row_offsets(n, dtype):
    gen = torch.Generator().manual_seed(n)
    runs = torch.randint(1, max(2, n // 3), (n,), generator=gen)
    req = torch.repeat_interleave(torch.randint(0, 9, (n,), generator=gen), runs)[:n]
    req = req.to(dtype).to(DEVICE)
    with envs.SGLANG_OPT_DSV4_FUSED_ATTN_METADATA.override(False):
        ref = _first_row_offsets(req)
    for got, want in zip(first_row_offsets(req), ref):
        _same(got, want)


@pytest.mark.parametrize("num_rows", [1, 300, 32768])
def test_commit_packed_rows_persistent(num_rows):
    gen = torch.Generator().manual_seed(num_rows)
    layout, page, capacity = KVLayout.V41_NEW, 256, 256
    pages = triton_cdiv(num_rows + 1, page)
    stride = page * layout.bytes_per_token
    src = torch.randint(0, 256, (pages, stride), generator=gen, dtype=torch.uint8).to(DEVICE)
    num_reqs = triton_cdiv(num_rows, capacity)
    window = SimpleNamespace(
        write_loc=torch.randperm(pages * page, generator=gen)[:num_rows].to(DEVICE),
        req=(torch.arange(num_rows) // capacity).to(DEVICE),
        pos=torch.arange(num_rows).to(DEVICE),
        commit_mask=(torch.rand(num_rows, generator=gen) < 0.05).to(DEVICE),
    )
    dst_pages = triton_cdiv(num_reqs * capacity, page)
    outs = []
    for persistent in (False, True):
        dst = torch.zeros((dst_pages, stride), dtype=torch.uint8, device=DEVICE)
        tags = torch.full((num_reqs * capacity,), -1, dtype=torch.int64, device=DEVICE)
        with envs.SGLANG_OPT_DSV41_WINDOW_COMMIT_PERSISTENT.override(persistent):
            commit_packed_rows(src, dst, tags, window, page_size=page, capacity=capacity, layout=layout)
        outs.append((dst, tags))
    # Each (req, pos % capacity) is distinct, so no two kept rows share a target.
    _same(outs[1][0], outs[0][0])
    _same(outs[1][1], outs[0][1])


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
