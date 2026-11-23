"""Multi-GPU test of the DSA DCP top-k merge (``DcpTopKMerge``).

Every rank holds the token-interleaved shard of the same global score rows,
runs the v2 candidate epilogue on it and merges through NVLink multicast. Each
rank's result must be exactly its share of the global top-k (score descending,
then position ascending) as widened slots, -1 after them, also when a graph
replays the merge on three alternating inputs until the 8-bit epoch tag wraps,
so a read of a stale call's entries would return another input's keys. Unlike
the one-GPU DSA DCP tests it needs 8 real ranks with NVLink multicast.

Usage::

    python test/registered/kernels/ops/attention/test_dsa_dcp_topk_merge.py
    python test/registered/kernels/ops/attention/test_dsa_dcp_topk_merge.py -k periodic
"""

from __future__ import annotations

import atexit
from typing import Optional
import logging
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_candidates_v2
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_dcp_sharded import DcpTopKMerge
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=60, stage="extra-b", runner_config="8-gpu-h200")

TOPK = 2048
PAGE_SIZE = 64
MAX_ROWS = 16
SEEDS = (1234, 1235, 1236)  # inputs the graph alternates between
GRAPH_CHECKS, REPLAYS_PER_CHECK = 5, 20  # 3 * 100 calls per row slot > 255 tags


@cache_once
def _init_once() -> DcpTopKMerge:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.cuda.set_stream(torch.cuda.Stream())
    dist.init_process_group(backend="gloo")
    ps._WORLD = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=ps._WORLD)
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    return DcpTopKMerge(ps._WORLD, MAX_ROWS, TOPK)


_HOT = {"skew": (383, 900, 383, 382), "skew2": (124, 900, 900, 124)}


def _rows(kind: str, seed: int) -> list[torch.Tensor]:
    """The same global score rows on every rank."""
    g = torch.Generator().manual_seed(seed)
    if kind == "random":
        return [torch.randn(131072, generator=g)]
    if kind == "random2":  # two rows, as the skewed kinds
        return [torch.randn(65536, generator=g), torch.randn(65532, generator=g)]
    if kind.startswith("stale"):
        # stale{1,2}_{old,new}: ranks 1 (and 2) hold 900 keys at 30+U(0,1) (old) or at 10.3+U
        # (10.6+U for two ranks; new), the others 600 at 10+U. Both fail on ranks 1 (and 2), and
        # the merged threshold, so the other ranks' share, differs between old and new.
        two, new = kind[5] == "2", kind.endswith("new")
        rows = []
        for shift in (0, 4):
            x = torch.randn(65536, generator=g)
            pos = torch.arange(65536)
            for r in range(4):
                hot = r == 1 or (two and r == 2)
                n_hot, base = (900, (10.6 if two else 10.3) if new else 30.0) if hot else (600, 10.0)
                at = pos[pos % 4 == r][torch.randperm(16384, generator=g)[:n_hot]]
                x[at] = base + torch.rand(n_hot, generator=g)
            rows.append(x[shift:])
        return rows
    if kind == "dense_top":
        # many distinct scores near the global threshold: overflows the v2
        # merge's 10-bit threshold bin, not the local 12-bit ones
        return [
            10 + 0.3 * torch.randn(131072, generator=g),
            10 + 0.3 * torch.randn(65536, generator=g),
        ]
    if kind == "ties":
        return [
            torch.randint(0, 200, (32768 - 100 * i,), generator=g).float() / 8
            for i in range(4)
        ]
    if kind == "periodic":
        # one rank owns the whole top-k: its candidates cannot certify it
        rows = []
        for n in (65536, 20000):
            base = torch.randn(8, generator=g)
            base[3] += 10
            # distinct scores spread over many of v2's coarse bins, so the local
            # top-k stays exact and the merge alone is under test
            rows.append(
                base.repeat(n // 8 + 1)[:n] + 4.0 * torch.randperm(n, generator=g) / n
            )
        return rows
    if kind in _HOT:
        # DCP4: rank r holds _HOT[kind][r] of the global top-2048 (skew: 900 on rank 1, more
        # than 768 candidates; skew2: 900 on ranks 1 and 2)
        rows = []
        for shift in (0, 4):
            x = torch.randn(65536, generator=g)
            pos = torch.arange(65536)
            for r, n_hot in enumerate(_HOT[kind]):
                hot = pos[pos % 4 == r][torch.randperm(16384, generator=g)[:n_hot]]
                x[hot] = 10 + torch.rand(n_hot, generator=g)
            rows.append(x[shift:])
        return rows
    if kind == "short":
        lengths = [1, 3, 7, 9, 300, 2047, 2048, 2049, 2055, 3000, 5000]
        return [torch.randn(n, generator=g) for n in lengths]
    if kind == "long":  # local rows past the register paths of the local top-k
        return [
            torch.randn(200000, generator=g),
            1 + 2 * torch.randn(160000, generator=g),
            torch.randn(140001, generator=g),
        ]
    raise ValueError(kind)


def _all_ranks_ok(ok: bool) -> bool:
    """Agree on a check across ranks, so that a failure on one rank fails the
    test everywhere instead of leaving the others waiting on its pushes."""
    flag = torch.tensor([int(ok)])
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _keys(scores: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    b = scores.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    ordered = torch.where(b >= 0x80000000, 0xFFFFFFFF - b, b + 0x80000000)
    return ((ordered - 0x80000000) << 32) | (0xFFFFFFFF - pos.to(torch.int64))


class _Case:
    """One input set: this rank's shard of the rows and its expected results."""

    def __init__(self, kind: str, seed: int, rank: int, world: int, span: int, glob: Optional[list] = None):
        device = torch.device("cuda", torch.cuda.current_device())
        glob = _rows(kind, seed) if glob is None else glob
        self.rows = len(glob)
        lens = [x.numel() for x in glob]
        self.local_lens = torch.tensor(
            [max((n - rank + world - 1) // world, 0) for n in lens],
            dtype=torch.int32,
            device=device,
        )
        width = (max(int(self.local_lens.max()), 1) + 11) // 4 * 4
        self.logits = torch.full((self.rows, width), float("-inf"), device=device)
        for i, x in enumerate(glob):
            local = x[rank::world]
            self.logits[i, : local.numel()] = local.to(device)
        self.global_lens = torch.tensor(lens, dtype=torch.int32, device=device)
        pages = (max(lens) + span - 1) // span
        g = torch.Generator().manual_seed(seed + 99)
        self.page_table = (
            torch.stack(
                [
                    torch.randperm(4 * pages, generator=g)[:pages]
                    for _ in range(self.rows)
                ]
            )
            .to(torch.int32)
            .to(device)
        )

        self.expect, self.expect_local = [], []
        for i, x in enumerate(glob):
            local_pos = torch.arange(x.numel())[rank::world]
            local_keys = _keys(x[rank::world], local_pos)
            local_top = local_pos[
                torch.topk(local_keys, min(TOPK, local_pos.numel())).indices
            ]
            self.expect_local.append(sorted(local_top.tolist()))
            pos = torch.arange(x.numel())
            top = pos[torch.topk(_keys(x, pos), min(TOPK, x.numel())).indices]
            mine = top[top % world == rank]
            pt = self.page_table[i].cpu().long()
            slots = pt[mine // span] * span + mine % span
            self.expect.append(sorted(slots.tolist()))

        self.candidates = torch.empty(
            (self.rows, TOPK, 2), dtype=torch.int32, device=device
        )
        self.out = torch.empty((self.rows, TOPK), dtype=torch.int32, device=device)

    def run(self, merge: DcpTopKMerge, world: int, rank: int, span: int) -> None:
        topk_transform_candidates_v2(
            self.logits.clone(),
            self.local_lens,
            out_candidates=self.candidates,
            world=world,
            rank=rank,
        )
        merge(self.candidates, self.global_lens, self.page_table, self.out, span)

    def run_pushed(self, merge: DcpTopKMerge, span: int) -> None:
        merge.prepare_step(self.local_lens, layer_id=0)  # the step's LPT row map, if any
        self.run_pushed_only(merge, span)

    def run_pushed_only(self, merge: DcpTopKMerge, span: int) -> None:
        self.local, self.lengths = merge.merge_pushed(
            self.logits.clone(), self.local_lens, self.global_lens, self.page_table, self.out, span
        )

    def pushed_failures(self, rank: int, world: int) -> list[str]:
        bad = []
        for i in range(self.rows):
            n = len(self.expect[i])
            if sorted(self.out[i, :n].tolist()) != self.expect[i]:
                bad.append(f"rank {rank} row {i}: merged slots differ")
            if not bool((self.out[i, n:] == -1).all()):
                bad.append(f"rank {rank} row {i}: padding is not -1")
            if int(self.lengths[i]) != n or sorted(self.local[i, :n].tolist()) != sorted(
                s // world for s in self.expect[i]
            ):
                bad.append(f"rank {rank} row {i}: localized rows differ")
        return bad

    def failures(self, rank: int) -> list[str]:
        bad = []
        for i in range(self.rows):
            got_local = self.candidates[i, :, 1]
            if sorted(got_local[got_local >= 0].tolist()) != self.expect_local[i]:
                bad.append(f"rank {rank} row {i}: local candidates are not exact")
            n = len(self.expect[i])
            if sorted(self.out[i, :n].tolist()) != self.expect[i]:
                bad.append(f"rank {rank} row {i}: merged slots differ")
            if not bool((self.out[i, n:] == -1).all()):
                bad.append(f"rank {rank} row {i}: padding is not -1")
        return bad


@pytest.mark.parametrize("kind", ["random", "dense_top", "ties", "periodic", "short"])
@torch.inference_mode()
def test_dcp_topk_merge(kind: str) -> None:
    merge = _init_once()
    world, rank = ps._WORLD.world_size, ps._WORLD.rank_in_group
    if world != 8:
        pytest.skip("the merge needs 8 ranks")
    if merge.buffer is None:
        pytest.skip("multimem multicast unavailable")
    span = PAGE_SIZE * world
    cases = [_Case(kind, seed, rank, world, span) for seed in SEEDS]

    for case in cases:
        case.run(merge, world, rank, span)
        torch.cuda.synchronize()
        bad = case.failures(rank)
        assert _all_ranks_ok(not bad), bad or "failed on another rank"
        fell_back = merge.fallback[: case.rows].tolist()
        if kind == "periodic":
            assert all(fell_back), fell_back
        elif kind in ("random", "short"):
            assert not any(fell_back), fell_back

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for case in cases:
            case.run(merge, world, rank, span)
    for _ in range(GRAPH_CHECKS):
        for _ in range(REPLAYS_PER_CHECK):
            graph.replay()
        torch.cuda.synchronize()
        bad = [msg for case in cases for msg in case.failures(rank)]
        assert _all_ranks_ok(not bad), bad or "failed on another rank"


_PUSH_MERGES: dict[int, DcpTopKMerge] = {}


def _push_merge(cand_k: int) -> DcpTopKMerge:
    if cand_k not in _PUSH_MERGES:
        with envs.SGLANG_DCP4_PUSH_CAND_K.override(cand_k):
            _PUSH_MERGES[cand_k] = DcpTopKMerge(ps._WORLD, MAX_ROWS, TOPK, push=True)
    return _PUSH_MERGES[cand_k]


def _expected_failures(kind: str, cand_k: int) -> tuple[Optional[bool], Optional[int]]:
    """(every row falls back, ranks failing the certificate per row); None: not fixed."""
    if cand_k == TOPK:
        return False, 0
    if 4 * cand_k == TOPK:  # every candidate is in the top-k: each rank with more keys fails
        return (None, None) if kind == "short" else (True, 4)
    if kind == "periodic":  # rank 3 holds the whole top-k; ranks with a high phase may fail too
        return True, None
    if kind in _HOT:
        n = sum(h > cand_k for h in _HOT[kind])
        return n > 0, n
    return (False, 0) if kind in ("random", "short") else (None, None)


@pytest.mark.parametrize("cand_k", [512, 768, 1024, 2048])
@pytest.mark.parametrize(
    "kind", ["random", "dense_top", "ties", "periodic", "skew", "skew2", "short"]
)
@torch.inference_mode()
def test_dcp4_topk_merge_pushed(kind: str, cand_k: int) -> None:
    """DCP4 push merge: exact for every candidate count (SGLANG_DCP4_PUSH_CAND_K)
    with one (skew, periodic), two (skew2) or all four (512) ranks failing the
    certificate; at 2048 a rank holding the whole top-k (periodic) needs no fallback."""
    _init_once()
    world, rank = ps._WORLD.world_size, ps._WORLD.rank_in_group
    if world != 4:
        pytest.skip("the DCP4 push merge needs 4 ranks")
    merge = _push_merge(cand_k)
    if merge.buffer is None:
        pytest.skip("multimem multicast unavailable")
    assert merge.cand_k == cand_k
    span = PAGE_SIZE * world
    all_fall_back, n_fail = _expected_failures(kind, cand_k)
    for seed in SEEDS:
        case = _Case(kind, seed, rank, world, span)
        published = int(merge.stats[1])
        case.run_pushed(merge, span)
        torch.cuda.synchronize()
        bad = case.pushed_failures(rank, world)
        assert _all_ranks_ok(not bad), bad or "failed on another rank"
        fell_back = merge.fallback[: case.rows].tolist()
        if all_fall_back is not None:
            assert all(fell_back) if all_fall_back else not any(fell_back), fell_back
        if n_fail is None:
            continue
        total = torch.tensor([int(merge.stats[1]) - published])
        dist.all_reduce(total)
        assert int(total) == n_fail * case.rows, (int(total), n_fail, case.rows)


_WIDE_MERGES: dict[int, DcpTopKMerge] = {}


def _wide_merge(warps: int) -> DcpTopKMerge:
    if warps not in _WIDE_MERGES:
        with envs.SGLANG_DCP4_PUSH_CAND_K.override(TOPK), envs.SGLANG_DCP4_PUSH_MERGE_WARPS.override(warps):
            _WIDE_MERGES[warps] = DcpTopKMerge(ps._WORLD, MAX_ROWS, TOPK, push=True)
    return _WIDE_MERGES[warps]


def _rerun_merge(merge: DcpTopKMerge, case: "_Case", rank: int, span: int, **kw) -> tuple:
    """The merge again on the candidates the last call pushed (its row counters stepped back)."""
    from sglang.kernels.ops.attention.dsa.dcp_topk_merge import dcp_topk_merge_pushed

    rows = case.rows
    merge.counters[:rows] -= 1
    out, local = torch.empty_like(case.out), torch.empty_like(case.out)
    lengths = torch.empty(rows, dtype=torch.int32, device=case.out.device)
    dcp_topk_merge_pushed(
        case.logits.clone(), case.local_lens, case.global_lens, case.page_table,
        buffer=merge.buffer, peer_ptrs=merge.peer_ptrs, counters=merge.counters,
        out_slots=out, out_local=local, out_lengths=lengths,
        out_fallback=torch.empty(rows, dtype=torch.int32, device=out.device),
        rank=rank, span=span, cand_k=TOPK, **kw,
    )
    torch.cuda.synchronize()
    return out, local, lengths


@pytest.mark.parametrize("warps", [2, 4, 8])
@pytest.mark.parametrize(
    "kind", ["random", "dense_top", "ties", "periodic", "skew", "skew2", "short"]
)
@torch.inference_mode()
def test_dcp4_topk_merge_pushed_wide(kind: str, warps: int) -> None:
    """SGLANG_DCP4_PUSH_MERGE_WARPS at cand_k == top-k: on the same pushed candidates, the
    same slots, localized rows and lengths as the one-warp merge, element for element; exact
    sets; and stable under CUDA graph replay."""
    _init_once()
    world, rank = ps._WORLD.world_size, ps._WORLD.rank_in_group
    if world != 4:
        pytest.skip("the DCP4 push merge needs 4 ranks")
    wide = _wide_merge(warps)
    if wide.buffer is None:
        pytest.skip("multimem multicast unavailable")
    assert wide.merge_warps == warps
    span = PAGE_SIZE * world
    for seed in SEEDS:
        case = _Case(kind, seed, rank, world, span)
        case.run_pushed(wide, span)
        torch.cuda.synchronize()
        bad = case.pushed_failures(rank, world)
        assert not any(wide.fallback[: case.rows].tolist())
        got = (case.out.clone(), case.local.clone(), case.lengths.clone())
        dist.barrier()
        ref = _rerun_merge(wide, case, rank, span, warps_per_rank=0)
        same = all(torch.equal(x, y) for x, y in zip(got, ref))
        assert _all_ranks_ok(not bad and same), bad or ("outputs differ from the one-warp merge" if not same else "failed on another rank")

    case = _Case(kind, SEEDS[0], rank, world, span)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        case.run_pushed(wide, span)
    for _ in range(GRAPH_CHECKS):
        for _ in range(REPLAYS_PER_CHECK):
            graph.replay()
        torch.cuda.synchronize()
        bad = case.pushed_failures(rank, world)
        assert _all_ranks_ok(not bad), bad or "failed on another rank"


_SAMPLED_MERGES: dict[tuple[int, int, bool], DcpTopKMerge] = {}


def _sampled_merge(cand_k: int, min_len: int, lpt: bool = False) -> DcpTopKMerge:
    if (cand_k, min_len, lpt) not in _SAMPLED_MERGES:
        with (
            envs.SGLANG_DCP4_PUSH_CAND_K.override(cand_k),
            envs.SGLANG_DSA_DCP_TOPK_PUSH_SAMPLED.override(True),
            envs.SGLANG_DSA_DCP_TOPK_PUSH_SAMPLED_MIN_LEN.override(min_len),
            envs.SGLANG_DSA_DCP_TOPK_PUSH_LPT.override(lpt),
        ):
            _SAMPLED_MERGES[cand_k, min_len, lpt] = DcpTopKMerge(ps._WORLD, MAX_ROWS, TOPK, push=True)
    return _SAMPLED_MERGES[cand_k, min_len, lpt]


@pytest.mark.parametrize("lpt", [False, True])
@pytest.mark.parametrize("min_len", [0, 32768])
@pytest.mark.parametrize("cand_k", [768, 2048])
@pytest.mark.parametrize(
    "kind", ["random", "long", "dense_top", "ties", "periodic", "skew", "short"]
)
@torch.inference_mode()
def test_dcp4_topk_merge_pushed_sampled(kind: str, cand_k: int, min_len: int, lpt: bool) -> None:
    """SGLANG_DSA_DCP_TOPK_PUSH_SAMPLED: the sampled local top-k pushes the exact local
    candidate sets, so the merge is exact and fails its certificate on the same rows as
    with the exact local top-k; Gaussian rows need no exact redo of the local top-k.
    min_len 32768 keeps the 32768-key local rows of `random` exact and samples `long`;
    lpt (longest row first) changes only which block runs which row."""
    _init_once()
    world, rank = ps._WORLD.world_size, ps._WORLD.rank_in_group
    if world != 4:
        pytest.skip("the DCP4 push merge needs 4 ranks")
    sampled, exact = _sampled_merge(cand_k, min_len, lpt), _push_merge(cand_k)
    if sampled.buffer is None:
        pytest.skip("multimem multicast unavailable")
    assert sampled.sampled and not exact.sampled and sampled.sampled_min_len == min_len
    assert sampled.lpt == lpt and (sampled.lpt_row_map is not None) == lpt
    span = PAGE_SIZE * world
    for seed in SEEDS:
        case = _Case(kind, seed, rank, world, span)
        case.run_pushed(exact, span)
        torch.cuda.synchronize()
        exact_fell_back = exact.fallback[: case.rows].tolist()
        redone = int(sampled.sampled_fallbacks)
        case.run_pushed(sampled, span)
        torch.cuda.synchronize()
        bad = case.pushed_failures(rank, world)
        if sampled.fallback[: case.rows].tolist() != exact_fell_back:
            bad.append(f"rank {rank}: certificate failures differ from the exact local top-k")
        if kind in ("random", "long") and int(sampled.sampled_fallbacks) != redone:
            bad.append(f"rank {rank}: the sampled local top-k redid a Gaussian row")
        assert _all_ranks_ok(not bad), bad or "failed on another rank"


@pytest.mark.parametrize("owner", [False, True])
@torch.inference_mode()
def test_dcp4_topk_merge_pushed_lpt_forwards(owner: bool) -> None:
    """LPT across forwards as the indexer drives it (prepare_step in every DSA layer, the map built
    at the lowest layer id, 0 as in the other tests that share this merge): forwards of 2, 11, 3 and
    1 rows, each over three layers, with
    an idle forward between; with SGLANG_DSA_DCP_TOPK_LPT_CHECK every launch's map must be its own
    lengths' order, and every merge is exact. A forward that skips the map's layer launches on a
    map built for other rows: plain order, still exact."""
    _init_once()
    world, rank = ps._WORLD.world_size, ps._WORLD.rank_in_group
    if world != 4:
        pytest.skip("the DCP4 push merge needs 4 ranks")
    merge = _owner_merge(4, sampled=True, lpt=True) if owner else _sampled_merge(2048, 0, True)
    if merge.buffer is None:
        pytest.skip("multimem multicast unavailable")
    assert merge.owner_merge == owner
    span = PAGE_SIZE * world
    bad = []
    with envs.SGLANG_DSA_DCP_TOPK_LPT_CHECK.override(True):
        for kind, seed in (("random2", 1234), ("short", 1235), ("long", 1236), ("random", 1237)):
            case = _Case(kind, seed, rank, world, span)
            for layer in (0, 4, 8):  # one forward: the map is built at layer 0 only
                merge.prepare_step(case.local_lens, layer)
                case.run_pushed_only(merge, span)
                torch.cuda.synchronize()
                bad += [f"{kind} layer {layer}: {b}" for b in case.pushed_failures(rank, world)]
            merge.prepare_step(case.local_lens[:0], 0)  # an idle forward: nothing to map
    # a forward that skips layer 0 after a 1-row forward: its map is for 1 row, so plain order
    case = _Case("random2", 1238, rank, world, span)
    merge.prepare_step(case.local_lens, 4)
    case.run_pushed_only(merge, span)
    torch.cuda.synchronize()
    bad += [f"skipped map: {b}" for b in case.pushed_failures(rank, world)]
    assert _all_ranks_ok(not bad), bad or "failed on another rank"


_VARIANT_MERGES: dict[tuple, DcpTopKMerge] = {}


def _owner_merge(warps: int, sampled: bool = False, lpt: bool = False, owner: bool = True, max_rows: int = MAX_ROWS) -> DcpTopKMerge:
    """cand_k == top-k push merge with `warps` warps per rank; owner: SGLANG_DCP4_PUSH_OWNER_MERGE."""
    key = (warps, sampled, lpt, owner, max_rows)
    if key not in _VARIANT_MERGES:
        with (
            envs.SGLANG_DCP4_PUSH_CAND_K.override(TOPK),
            envs.SGLANG_DCP4_PUSH_MERGE_WARPS.override(warps),
            envs.SGLANG_DSA_DCP_TOPK_PUSH_SAMPLED.override(sampled),
            envs.SGLANG_DSA_DCP_TOPK_PUSH_SAMPLED_MIN_LEN.override(0),
            envs.SGLANG_DSA_DCP_TOPK_PUSH_LPT.override(lpt),
            envs.SGLANG_DCP4_PUSH_OWNER_MERGE.override(owner),
        ):
            _VARIANT_MERGES[key] = DcpTopKMerge(ps._WORLD, max_rows, TOPK, push=True)
    return _VARIANT_MERGES[key]


@pytest.mark.parametrize("variant", ["exact", "sampled", "sampled_lpt"])
@pytest.mark.parametrize(
    "kind", ["random", "dense_top", "ties", "periodic", "skew", "skew2", "short", "long"]
)
@torch.inference_mode()
def test_dcp4_topk_merge_pushed_owner(kind: str, variant: str) -> None:
    """SGLANG_DCP4_PUSH_OWNER_MERGE: each row's candidates go only to its owner (row % 4), which
    sends t* to the others. Every rank's slots, localized rows and lengths equal the multicast
    merge's element for element on the same inputs (same local top-k), and are exact. Then three
    inputs, one CUDA graph each, replayed in turn for 300 calls per row slot (past the 255 tags):
    a non-owner reading a stale t* would write another input's keys."""
    _init_once()
    world, rank = ps._WORLD.world_size, ps._WORLD.rank_in_group
    if world != 4:
        pytest.skip("the DCP4 push merge needs 4 ranks")
    sampled, lpt = variant != "exact", variant == "sampled_lpt"
    owner, multicast = _owner_merge(4, sampled, lpt), _owner_merge(4, sampled, lpt, owner=False)
    if owner.buffer is None:
        pytest.skip("multimem multicast unavailable")
    assert owner.owner_merge and not multicast.owner_merge
    span = PAGE_SIZE * world
    cases = [_Case(kind, seed, rank, world, span) for seed in SEEDS]
    for case in cases:
        case.run_pushed(multicast, span)
        torch.cuda.synchronize()
        ref = (case.out.clone(), case.local.clone(), case.lengths.clone())
        case.run_pushed(owner, span)
        torch.cuda.synchronize()
        bad = case.pushed_failures(rank, world)
        same = all(torch.equal(x, y) for x, y in zip((case.out, case.local, case.lengths), ref))
        assert _all_ranks_ok(not bad and same), bad or ("outputs differ from the multicast merge" if not same else "failed on another rank")

    graphs = []
    for case in cases:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            case.run_pushed(owner, span)
        graphs.append(graph)
    for i in range(300):
        graphs[i % 3].replay()
        if i % 25 == 24 or i >= 297:
            torch.cuda.synchronize()
            bad = cases[i % 3].pushed_failures(rank, world)
            assert _all_ranks_ok(not bad), bad or "failed on another rank"


@pytest.mark.parametrize("rows", [1, 3, 5, 500, 600])
@torch.inference_mode()
def test_dcp4_topk_merge_pushed_owner_rows(rows: int) -> None:
    """Owner mode with ranks that own no row (1-3 rows), 500 rows (more merge blocks than one GPU
    holds: owners first) and 600 (more rows waiting for t* than resident blocks: the multicast
    merge instead); exact either way, local rows of 1-3000 keys."""
    _init_once()
    world, rank = ps._WORLD.world_size, ps._WORLD.rank_in_group
    if world != 4:
        pytest.skip("the DCP4 push merge needs 4 ranks")
    owner = _owner_merge(4, max_rows=1024)
    if owner.buffer is None:
        pytest.skip("multimem multicast unavailable")
    span = PAGE_SIZE * world
    g = torch.Generator().manual_seed(rows)
    lens = torch.randint(1, 12000, (rows,), generator=g).tolist()
    case = _Case("", 0, rank, world, span, glob=[torch.randn(n, generator=g) for n in lens])
    for _ in range(3):
        case.run_pushed(owner, span)
    torch.cuda.synchronize()
    bad = case.pushed_failures(rank, world)
    assert _all_ranks_ok(not bad), bad or "failed on another rank"


@pytest.mark.parametrize("failing", ["stale1", "stale2"])
@torch.inference_mode()
def test_dcp4_topk_merge_pushed_stale_slot(failing: str) -> None:
    """A rank's full slot is written only when it fails; the next failure of the same
    rank(s) on the same row slot 510 calls later (same buffer parity and 8-bit tag) must
    not let the other ranks read the old call's keys."""
    _init_once()
    world, rank = ps._WORLD.world_size, ps._WORLD.rank_in_group
    if world != 4:
        pytest.skip("the DCP4 push merge needs 4 ranks")
    merge = _push_merge(768)
    if merge.buffer is None:
        pytest.skip("multimem multicast unavailable")
    span = PAGE_SIZE * world
    first, idle, again = (_Case(k, seed, rank, world, span) for k, seed in
                          ((f"{failing}_old", 1234), ("random2", 1235), (f"{failing}_new", 1236)))
    assert first.rows == idle.rows == again.rows
    calls = int(merge.counters[0])
    first.run_pushed(merge, span)
    while int(merge.counters[0]) < calls + 510:
        idle.run_pushed(merge, span)
    published = int(merge.stats[1])
    again.run_pushed(merge, span)
    torch.cuda.synchronize()
    bad = again.pushed_failures(rank, world)
    assert _all_ranks_ok(not bad), bad or "failed on another rank"
    total = torch.tensor([int(merge.stats[1]) - published])
    dist.all_reduce(total)
    assert int(total) == (2 if failing == "stale2" else 1) * again.rows, int(total)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4, 8))
