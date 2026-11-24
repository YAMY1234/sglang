"""DFLASH draft KV ring: the rows the worker writes (DFlashDraftKVRing.locs, kept
under the pool's DCP owner rule) are exactly the rows the window kernel reads in
ring mode, the window floor holds, and an extend keeps only the tail a later
window can read."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.mla_window_indices import mla_window_sparse_indices
from sglang.srt.speculative.dflash_draft_ring import (
    DFlashDraftKVRing,
    DraftKVRingStaging,
    draft_kv_ring_pool_rows,
    draft_kv_ring_rows_per_req,
    draft_kv_ring_unsupported_reason,
    dspark_draft_kv_ring_rows_per_req,
    dspark_draft_kv_ring_unsupported_reason,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

WINDOW_LEFT = 1023
BLOCK = 8


def _ring(dcp, req_pool_rows, device, dcp_rank=0):
    return DFlashDraftKVRing(
        rows_per_req=draft_kv_ring_rows_per_req(
            window_left=WINDOW_LEFT, block_size=BLOCK, dcp_size=dcp
        ),
        dcp_size=dcp,
        req_pool_rows=req_pool_rows,
        device=device,
        window_left=WINDOW_LEFT,
        dcp_rank=dcp_rank,
    )


@pytest.mark.parametrize("dcp", [1, 8])
def test_window_reads_the_rows_the_ring_writes(dcp):
    device = "cuda"
    req_pool_rows = 9
    ring = _ring(dcp, req_pool_rows, device)
    pool_rows = draft_kv_ring_pool_rows(
        req_pool_rows=req_pool_rows, rows_per_req=ring.rows_per_req, page_size=64
    )
    req_pool_indices = torch.tensor([3, 7, 1], dtype=torch.int32, device=device)
    prefix = torch.tensor([5000, 700, 3], dtype=torch.int64, device=device)
    ring.min_pos[1] = 2  # request 2: a 2-token prefix hit
    ring.min_pos[3] = 4100
    positions = (prefix.unsqueeze(1) + torch.arange(BLOCK, device=device)).reshape(-1)
    for rank in range(dcp):
        indices, counts = mla_window_sparse_indices(
            req_to_token=torch.empty((1, 1), dtype=torch.int32, device=device),
            req_pool_indices=req_pool_indices,
            positions=positions,
            window_left=WINDOW_LEFT,
            dcp_size=dcp,
            dcp_rank=rank,
            page_size=64,
            ring_rows_per_req=ring.rows_per_req,
            min_pos=ring.min_pos,
        )
        for row, p in enumerate(positions.tolist()):
            req = int(req_pool_indices[row // BLOCK])
            lo = max(p - WINDOW_LEFT, 0, int(ring.min_pos[req]))
            keys = torch.arange(lo, p + 1, device=device)
            locs = ring.locs(torch.full_like(keys, req), keys)
            mine = locs % dcp == rank
            expected = (locs[mine] // dcp).tolist()
            got = indices[row, : int(counts[row])].tolist()
            assert got == expected, (rank, row)
            assert all(0 < r < pool_rows for r in got)


@pytest.mark.parametrize("dcp", [1, 8])
def test_window_and_block_never_share_a_row(dcp):
    ring = _ring(dcp, 4, "cpu")
    p = 123457
    # The block's rows and every row a query in that block can read.
    keys = torch.arange(p - WINDOW_LEFT, p + BLOCK)
    locs = ring.locs(torch.full_like(keys, 2), keys)
    assert locs.unique().numel() == keys.numel()
    assert ring.span >= WINDOW_LEFT + 1 + BLOCK


def test_extend_keeps_only_the_readable_tail():
    ring = _ring(8, 4, "cpu")
    lens = torch.tensor([3000, 40])
    prefix = torch.tensor([0, 100])
    positions = torch.cat(
        [torch.arange(int(a), int(a + n)) for a, n in zip(prefix, lens)]
    )
    locs = ring.extend_locs(
        req_pool_indices=torch.tensor([1, 2]),
        positions=positions,
        extend_lens=lens,
        seq_ends=prefix + lens,
    )
    first, second = locs[:3000], locs[3000:]
    kept = first != 0
    assert int(kept.sum()) == ring.span
    assert bool(kept[-ring.span :].all())
    assert first[kept].unique().numel() == ring.span
    assert bool((second != 0).all())


@pytest.mark.parametrize("seq_len", [5000, 300])
@pytest.mark.parametrize("decode_dcp", [8, 1])
def test_pd_window_rows_pair_prefill_with_each_decode_rank(seq_len, decode_dcp):
    """PD: a DCP-1 prefill's window rows, paired positionally with each decode
    rank's (sink pairs dropped as the transfer does), cover every window position
    exactly once per DCP group, each landing on the row that rank's window reads.
    A DCP-1 decode ring (SGLANG_K31_DRAFT_NO_DCP) takes the whole window."""
    prefill = _ring(1, 4, "cpu")
    src = prefill.transfer_rows(2, seq_len)
    window = list(range(max(0, seq_len - WINDOW_LEFT - 1), seq_len))
    assert len(src) == len(window) and (src != 0).all()
    covered = []
    for rank in range(decode_dcp):
        decode = _ring(decode_dcp, 6, "cpu", dcp_rank=rank)
        dst = decode.transfer_rows(5, seq_len)
        assert len(dst) == len(src)
        for p, a, b in zip(window, src, dst):
            if a and b:
                covered.append(p)
                assert p % decode_dcp == rank
                assert a == int(prefill.locs(torch.tensor([2]), torch.tensor([p])))
                assert b * decode_dcp + rank == int(
                    decode.locs(torch.tensor([5]), torch.tensor([p]))
                )
    assert sorted(covered) == window


def test_decode_floor_reset_and_unsupported_drafts():
    """PD decode resets a slot's floor to its prompt end at preallocation; the
    ring is resolved off for drafts and backends it does not fit."""
    from unittest import mock

    from sglang.srt.speculative import dflash_utils

    ring = _ring(8, 4, "cpu")
    ring.min_pos[2] = 7
    ring.reset_floor(2, 900)
    assert ring.min_pos.tolist() == [0, 0, 900, 0]

    def reason(mla=True, layers=("sliding_attention",) * 5, backend="trtllm_mla", window=None):
        with mock.patch.object(
            dflash_utils, "is_dflash_mla_draft_config", return_value=mla
        ), mock.patch.object(
            dflash_utils, "get_dflash_layer_types", return_value=list(layers)
        ):
            return draft_kv_ring_unsupported_reason(
                object(), draft_backend=backend, draft_window_size=window
            )

    assert reason() is None
    assert reason(backend="cutedsl_mla") is None
    assert reason(mla=False) is not None
    assert reason(layers=("sliding_attention", "full_attention")) is not None
    assert reason(backend="flashinfer") is not None
    assert reason(window=4096) is not None


# DSpark: a page-aligned ring (sink page, whole pages per request) read through a
# paged trtllm_mha block table.
DSPARK_WINDOW_LEFT = 2047
DSPARK_BLOCK = 6
PAGE = 64


def _dspark_ring(req_pool_rows, device):
    return DFlashDraftKVRing(
        rows_per_req=dspark_draft_kv_ring_rows_per_req(
            window_left=DSPARK_WINDOW_LEFT, block_size=DSPARK_BLOCK, page_size=PAGE
        ),
        dcp_size=1,
        req_pool_rows=req_pool_rows,
        device=device,
        window_left=DSPARK_WINDOW_LEFT,
        sink_rows=PAGE,
        page_size=PAGE,
    )


def test_dspark_page_table_reads_the_rows_the_ring_writes():
    """For every request, the block table's page of each window or block position
    is the page of the row the ring writes it to, at the same offset; the rows are
    distinct and inside the request's ring."""
    ring = _dspark_ring(9, "cpu")
    assert ring.rows_per_req % PAGE == 0
    req_pool_indices = torch.tensor([3, 7, 1, 0], dtype=torch.int32)
    seq_lens = [9000, 2054, 70, 3 * ring.rows_per_req + 5]
    width = (max(seq_lens) + DSPARK_BLOCK) // PAGE + 2
    table = torch.full((4, width), -1, dtype=torch.int32)
    ring.page_table_into(req_pool_indices, table)
    for b, seq_len in enumerate(seq_lens):
        req = int(req_pool_indices[b])
        keys = torch.arange(
            max(0, seq_len - DSPARK_WINDOW_LEFT - 1), seq_len + DSPARK_BLOCK
        )
        locs = ring.locs(torch.full_like(keys, req), keys)
        assert torch.equal(locs // PAGE, table[b, keys // PAGE].to(torch.int64))
        assert torch.equal(locs % PAGE, keys % PAGE)
        assert locs.unique().numel() == keys.numel()
        lo = PAGE + req * ring.rows_per_req
        assert bool(((locs >= lo) & (locs < lo + ring.rows_per_req)).all())
    pool_rows = draft_kv_ring_pool_rows(
        req_pool_rows=9, rows_per_req=ring.rows_per_req, page_size=PAGE, sink_rows=PAGE
    )
    assert pool_rows == PAGE + 9 * ring.rows_per_req
    # A second batch reuses the cached columns.
    table2 = torch.full((2, width), -1, dtype=torch.int32)
    ring.page_table_into(req_pool_indices[2:], table2)
    assert torch.equal(table2, table[2:])


@pytest.mark.parametrize("dspark", [False, True])
def test_fused_block_locs_match_locs(dspark):
    """The one-launch block locs equal ``locs`` for a sink row and a sink page."""
    ring = _dspark_ring(9, "cuda") if dspark else _ring(8, 9, "cuda")
    ring.min_pos[3], ring.min_pos[7] = 8000, 2053
    block = DSPARK_BLOCK if dspark else BLOCK
    req_pool_indices = torch.tensor([3, 7, 1, 0], dtype=torch.int64, device="cuda")
    starts = torch.tensor([9000, 2054, 70, 123457], device="cuda")
    positions = (starts.unsqueeze(1) + torch.arange(block, device="cuda")).reshape(-1)
    expected = ring.locs(req_pool_indices.repeat_interleave(block), positions)
    ring._fused_block_locs = True
    assert torch.equal(ring.block_locs(req_pool_indices, positions, block), expected)


def test_dspark_window_starts_at_the_floor():
    """A rebased ring with a prefix floor: the reader's window (window_seq_lens and
    the block table) covers exactly the positions from max(floor, seq_len - window)
    on, at the rows the ring writes; prefill's PD rows pair with decode's absolute
    landing rows, and unstage rotates them to where the decode ring reads them."""
    ring = _dspark_ring(9, "cpu")
    req_pool_indices = torch.tensor([3, 7, 1], dtype=torch.int32)
    seq_lens = torch.tensor([9000, 2054, 3 * ring.rows_per_req + 5])
    floors = [8300, 1000, 2 * ring.rows_per_req + 77]
    reqs = [
        type("R", (), {"rid": f"r{b}", "kv": type("K", (), {"req_pool_idx": int(r)})})()
        for b, r in enumerate(req_pool_indices.tolist())
    ]
    ring.record_floors(reqs, floors)
    lens = ring.window_seq_lens(req_pool_indices, seq_lens)
    assert lens.tolist() == [int(n) - f for n, f in zip(seq_lens, floors)]
    width = int(lens.max() + DSPARK_BLOCK) // PAGE + 2
    table = torch.full((3, width), -1, dtype=torch.int32)
    ring.page_table_into(req_pool_indices, table)
    for b in range(3):
        req, floor = int(req_pool_indices[b]), floors[b]
        logical = torch.arange(
            max(0, int(lens[b]) - DSPARK_WINDOW_LEFT - 1), int(lens[b]) + DSPARK_BLOCK
        )
        read = table[b, logical // PAGE].to(torch.int64) * PAGE + logical % PAGE
        assert torch.equal(
            read, ring.locs(torch.full_like(logical, req), logical + floor)
        )
        assert read.unique().numel() == logical.numel()

    # PD: prefill sends from its floor; decode lands at absolute rows, then rotates.
    n, req = int(seq_lens[0]), int(req_pool_indices[0])
    src = torch.from_numpy(ring.transfer_rows(req, n)).long()
    window = torch.arange(n - DSPARK_WINDOW_LEFT - 1, n)
    assert torch.equal(src, ring.locs(torch.full_like(window, req), window))

    class _Pool:
        def __init__(self, rows):
            self.bufs = [torch.zeros((rows, 2, 8), dtype=torch.int32)]

        def _pd_registerable_tensors(self):
            return self.bufs

    decode = _dspark_ring(6, "cpu")
    decode.pool = _Pool(PAGE + 6 * decode.rows_per_req)
    decode.reset_floor(5, n)
    dst = torch.from_numpy(decode.transfer_rows(5, n)).long()
    decode.pool.bufs[0][dst] = window.view(-1, 1, 1).int().expand(-1, 2, 8)
    decode.min_pos[5] = floors[0]  # prefill's floor row lands
    decode.mark_unstage(5)
    decode.flush_unstage()
    kept = window[window >= floors[0]]
    got = decode.pool.bufs[0][decode.locs(torch.full_like(kept, 5), kept)]
    assert torch.equal(got[:, 0, 0].long(), kept)


def test_dspark_extend_and_pd_rows():
    """A long chunk keeps only its readable tail; prefill's and decode's window
    rows pair positionally and each is the row its side's ring reads."""
    ring = _dspark_ring(4, "cpu")
    n = 3 * ring.rows_per_req + 777
    positions = torch.arange(n)
    locs = ring.extend_locs(
        req_pool_indices=torch.tensor([2]),
        positions=positions,
        extend_lens=torch.tensor([n]),
        seq_ends=torch.tensor([n]),
    )
    kept = locs != 0
    assert int(kept.sum()) == ring.span and bool(kept[-ring.span :].all())
    assert locs[kept].unique().numel() == ring.span
    decode = _dspark_ring(6, "cpu")
    src = ring.transfer_rows(2, n)
    dst = decode.transfer_rows(5, n)
    window = torch.arange(n - DSPARK_WINDOW_LEFT - 1, n)
    assert torch.equal(
        torch.from_numpy(src).long(), ring.locs(torch.full_like(window, 2), window)
    )
    assert torch.equal(
        torch.from_numpy(dst).long(), decode.locs(torch.full_like(window, 5), window)
    )


def test_dspark_staging_unstage_copies_this_ranks_heads():
    """A head-sharded decode ring lands whole rows in staging and copies its own
    head columns into the ring rows of the marked requests only."""

    class _Pool:
        page_size = PAGE

        def __init__(self, rows, heads):
            self.bufs = [
                torch.zeros((rows, heads, 64), dtype=torch.uint8) for _ in range(4)
            ]

        def _pd_registerable_tensors(self):
            return self.bufs

    ring = _dspark_ring(3, "cpu")
    rows = PAGE + 3 * ring.rows_per_req
    pool = _Pool(rows, 4)
    ring.pool = pool
    staging = DraftKVRingStaging(pool=pool, num_rows=rows, total_kv_heads=16)
    ring.attach_staging(staging, head_start=8, num_heads=4)
    for i, buf in enumerate(staging.ring_buffers()):
        buf.copy_(torch.randint(0, 255, buf.shape, dtype=torch.uint8))
    ring.mark_unstage(1)
    ring.flush_unstage()
    r1 = slice(PAGE + ring.rows_per_req, PAGE + 2 * ring.rows_per_req)
    for src, dst in zip(staging.ring_buffers(), pool.bufs):
        assert torch.equal(dst[r1], src[r1][:, 8:12])
        assert int(dst[: r1.start].sum()) == 0 and int(dst[r1.stop :].sum()) == 0
    _, _, item_lens = ring.ring_buf_infos()
    assert item_lens == [16 * 64] * 4


def test_dspark_staging_slots():
    """With staging slots a decode lands each request's PD rows in the slot its
    preallocation took; unstage copies (and rotates) them into the request's ring
    rows, then frees the slot once the copy has run; a request that leaves before
    its unstage gives its slot back and is skipped."""

    class _Pool:
        page_size = PAGE

        def __init__(self, rows, heads):
            self.bufs = [
                torch.zeros((rows, heads, 8), dtype=torch.int32, device="cuda")
                for _ in range(2)
            ]

        def _pd_registerable_tensors(self):
            return self.bufs

    ring = _dspark_ring(6, "cuda")
    ring.pool = _Pool(PAGE + 6 * ring.rows_per_req, 2)
    staging_rows = draft_kv_ring_pool_rows(
        req_pool_rows=2, rows_per_req=ring.rows_per_req, page_size=PAGE, sink_rows=PAGE
    )
    staging = DraftKVRingStaging(
        pool=ring.pool, num_rows=staging_rows, total_kv_heads=4
    )
    ring.attach_staging(staging, head_start=2, num_heads=2, slots=2)
    n = 3 * ring.rows_per_req + 9
    floors = {3: n - 700, 1: 0}  # request 3's floor inside its window
    window = torch.arange(n - DSPARK_WINDOW_LEFT - 1, n, device="cuda")
    for req in (3, 1):
        ring.reset_floor(req, n)
        dst = torch.from_numpy(ring.transfer_rows(req, n)).long().cuda()
        assert bool(((dst >= PAGE) & (dst < staging_rows)).all())
        for buf in staging.ring_buffers():
            buf[dst] = (window * 10 + req).int().view(-1, 1, 1).expand(-1, 4, 8)
        ring.min_pos[req] = floors[req]  # prefill's floor row lands
    assert not ring.staging_slot_free()
    with pytest.raises(RuntimeError):
        ring.reset_floor(4, n)
    ring.mark_unstage(3)
    ring.mark_unstage(1)
    ring.flush_unstage()
    torch.cuda.synchronize()
    for req in (3, 1):
        kept = window[window >= floors[req]]
        got = ring.pool.bufs[0][ring.locs(torch.full_like(kept, req), kept)]
        assert torch.equal(got[:, 0, 0].long(), kept * 10 + req)
    assert ring.staging_slot_free() and len(ring._free_slots) == 2
    ring.reset_floor(4, n)
    ring.release_staging_slot(4)
    ring.mark_unstage(4)
    ring.flush_unstage()
    assert len(ring._free_slots) == 2 and not ring._slot_of


def test_dspark_unsupported_drafts():
    from unittest import mock

    from sglang.srt.speculative import dflash_utils

    def reason(mla=False, layers=("sliding_attention",) * 5, backend="trtllm_mha"):
        with (
            mock.patch.object(
                dflash_utils, "is_dflash_mla_draft_config", return_value=mla
            ),
            mock.patch.object(
                dflash_utils, "get_dflash_layer_types", return_value=list(layers)
            ),
        ):
            return dspark_draft_kv_ring_unsupported_reason(
                object(), draft_backend=backend
            )

    assert reason() is None
    assert reason(mla=True) is not None
    assert reason(layers=("sliding_attention", "full_attention")) is not None
    assert reason(backend="flashinfer") is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
