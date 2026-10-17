"""The sharded DSpark Markov chain kernel draws MarkovSampleStep's tokens and keeps its
step logits bit for bit when the bias products sum exactly: one rank and four ranks
(spawned, symmetric memory), one to 64 rows (direct exchange, rank-local fold and the
wide kernel), greedy, sampled and mixed temperatures, BF16 and FP32 base logits,
local-shard and full-vocab base logits, three launches (epoch tags and both staging
parities) and CUDA graph replay with the noise forked onto its side stream. Rows at
temperature 0 leave their step logits untouched. Long graph replays of mixed row counts
and launch counters crossing 2**32 exercise the epoch-tag reuse, and a rank whose peer
never arrives stops at its deadline instead of hanging.
"""

import os
import socket
import subprocess
import sys
import time

import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

VOCAB = 129_280  # DeepSeek-V4.1
RANK = 256
STEPS = 5
MAX_BS = 8
KEY = 0x1234_5678_9ABC_DEF
TEMPERATURE_SETS = ((0.0,) * 8, (1.0,) * 8, (0.7, 1.3, 1.0, 0.0, 0.5, 2.0, 0.9, 1.1))
# Arbitrary; far above any launch, so only a missing peer reaches it.
TIMEOUT_S = 20.0
# Graph replays of the long test: LONG_LAUNCHES launches per replay.
LONG_LAUNCHES = 64
LONG_REPLAYS = 1100
LONG_ROWS = (8, 1, 3, 8, 2)
# The wide kernel (9-64 rows at TP4).
WIDE_BS = 64
WIDE_ROWS = (9, 16, 32, 64)
WIDE_GRAPH_ROWS = (64, 9, 1, 8, 33)
WIDE_REPLAYS = 14_000
# Per-CTA launch counters (dspark_sharded_markov.cuh).
NARROW_COUNT_WORDS = slice(256, None)
WIDE_COUNT_WORDS = slice(1280, None)
ARRIVAL_WORDS = slice(128, 136)


def _gpu_ready() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in (
        (10, 0),
        (10, 3),
    )


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100/SM103 GPU")


def _ternary(g: torch.Generator, device, *shape: int) -> torch.Tensor:
    """Sparse {-1, 0, 1} BF16: |bias| <= 256, exact in BF16 in any summation order."""
    values = torch.randint(-1, 2, shape, generator=g)
    values *= torch.rand(shape, generator=g) < 0.25
    return values.to(device, torch.bfloat16)


def _reference(w1, w2, base, anchor, temperatures, offset, cache):
    """MarkovSampleStep per step over the full vocab, as the unsharded draft step runs it."""
    from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep

    prev, tokens = anchor.long(), []
    for step in range(base.shape[1]):
        bias = F.linear(w1[prev], w2)
        prev = MarkovSampleStep.execute(
            base_logits=base[:, step],
            bias=bias,
            temperatures=temperatures,
            key=KEY,
            offset=offset,
            step=step,
            step_dtype=torch.float32,
            logits_out=None if cache is None else cache[: base.shape[0], step],
            has_sampled_rows=None if cache is not None else False,
        )
        tokens.append(prev)
    return torch.stack(tokens, 1)


def _assert_cache(cache, want_cache, temperatures, rows: int, what) -> None:
    """Rows that keep logits match; temperature-0 rows stay zero."""
    keeps = temperatures[:rows] != 0
    assert torch.equal(cache[:rows][keeps], want_cache[:rows][keeps]), what
    assert not cache[:rows][~keeps].any(), what


def _make_sampler(group, rank, world, w1, w2, max_bs=MAX_BS, keeps_logits=True):
    from sglang.srt.speculative.dspark_components.dspark_sharded_markov import (
        DSparkShardedMarkovSampler,
    )

    return DSparkShardedMarkovSampler(
        embed_weight=w1,
        head_weight=w2,
        vocab_size=VOCAB,
        num_steps=STEPS,
        cache_dtype=torch.float32,
        group=group,
        rank=rank,
        world=world,
        max_bs=max_bs,
        timeout_s=TIMEOUT_S,
        keeps_logits=keeps_logits,
    )


def _check_equal(device, group, rank: int, world: int, base_dtype, seed: int):
    g = torch.Generator().manual_seed(seed)
    w1, w2 = _ternary(g, device, VOCAB, RANK), _ternary(g, device, VOCAB, RANK)
    sampler = _make_sampler(group, rank, world, w1, w2)
    # A greedy draft's sampler has no logit staging; it serves greedy blocks only.
    greedy_sampler = _make_sampler(group, rank, world, w1, w2, keeps_logits=False)
    shard = slice(rank * VOCAB // world, (rank + 1) * VOCAB // world)
    offset = torch.tensor([3], dtype=torch.int64, device=device)
    for rows in (1, 2, 4, 8):
        base = (torch.randn(rows, STEPS, VOCAB, generator=g) * 3).to(device, base_dtype)
        anchor = torch.randint(0, VOCAB, (rows,), generator=g).to(device)
        for temps in TEMPERATURE_SETS:
            temperatures = torch.tensor(temps, device=device)
            for keep in (True, False):  # sampled draft vs greedy draft (no logits)
                want_cache = (
                    torch.zeros(MAX_BS, STEPS, VOCAB, device=device) if keep else None
                )
                want = _reference(
                    w1, w2, base, anchor, temperatures[:rows], offset, want_cache
                )
                for chain in (sampler,) if keep else (sampler, greedy_sampler):
                    for local in (True, False):
                        inputs = base[:, :, shard].contiguous() if local else base
                        for _ in range(3):
                            cache = (
                                torch.zeros(MAX_BS, STEPS, VOCAB, device=device)
                                if keep
                                else None
                            )
                            tokens = torch.full(
                                (MAX_BS, STEPS), -7, dtype=torch.int64, device=device
                            )
                            if keep:
                                chain.launch_noise(rows=rows, key=KEY, offset=offset)
                            chain.sample(
                                base_logits=inputs,
                                anchor_tokens=anchor,
                                temperatures=temperatures,
                                logits_cache=cache,
                                draft_tokens=tokens,
                            )
                            torch.cuda.synchronize()
                            what = (rows, temps, keep, local, chain is sampler)
                            assert torch.equal(tokens[:rows], want), what
                            if keep:
                                _assert_cache(
                                    cache, want_cache, temperatures, rows, what
                                )

    _check_forked_graph(device, sampler, w1, w2, shard, base_dtype, g)
    assert int(sampler.control[ARRIVAL_WORDS].abs().sum()) == 0  # arrivals reset


def _check_forked_graph(device, sampler, w1, w2, shard, base_dtype, g):
    """CUDA graph: noise forked onto the side stream, replayed with new offsets."""
    rows = 2
    base = (torch.randn(rows, STEPS, VOCAB, generator=g) * 3).to(device, base_dtype)
    anchor = torch.randint(0, VOCAB, (rows,), generator=g).to(device)
    temperatures = torch.tensor(TEMPERATURE_SETS[2], device=device)
    offset = torch.tensor([3], dtype=torch.int64, device=device)
    cache = torch.zeros(MAX_BS, STEPS, VOCAB, device=device)
    tokens = torch.full((MAX_BS, STEPS), -7, dtype=torch.int64, device=device)

    def forked_sample():
        sampler.fork_noise(rows=rows, key=KEY, offset=offset)
        sampler.join_noise()
        sampler.sample(
            base_logits=base[:, :, shard].contiguous(),
            anchor_tokens=anchor,
            temperatures=temperatures,
            logits_cache=cache,
            draft_tokens=tokens,
        )

    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        forked_sample()
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forked_sample()
    for step_offset in (5, 6, 1 << 33):
        offset.fill_(step_offset)
        graph.replay()
        want_cache = torch.zeros_like(cache)
        want = _reference(w1, w2, base, anchor, temperatures[:rows], offset, want_cache)
        torch.cuda.synchronize()
        assert torch.equal(tokens[:rows], want), step_offset
        _assert_cache(cache, want_cache, temperatures, rows, step_offset)


def _check_long_replay(device, group, rank: int, world: int, seed: int):
    """Graphs of mixed row counts share the launch counters; > 2 tag periods of
    launches, then a fresh sampler whose launch counters cross 2**32."""
    g = torch.Generator().manual_seed(seed)
    w1, w2 = _ternary(g, device, VOCAB, RANK), _ternary(g, device, VOCAB, RANK)
    shard = slice(rank * VOCAB // world, (rank + 1) * VOCAB // world)
    base = (torch.randn(MAX_BS, STEPS, VOCAB, generator=g) * 3).to(
        device, torch.bfloat16
    )
    local = base[:, :, shard].contiguous()
    anchor = torch.randint(0, VOCAB, (MAX_BS,), generator=g).to(device)
    temperatures = torch.tensor(TEMPERATURE_SETS[2], device=device)
    offset = torch.tensor([11], dtype=torch.int64, device=device)
    want_cache = torch.zeros(MAX_BS, STEPS, VOCAB, device=device)
    want_sampled = _reference(w1, w2, base, anchor, temperatures, offset, want_cache)
    want_greedy = _reference(w1, w2, base, anchor, temperatures, offset, None)

    sampler = _make_sampler(group, rank, world, w1, w2)
    cache = torch.zeros(MAX_BS, STEPS, VOCAB, device=device)
    tokens = torch.full(
        (LONG_LAUNCHES, MAX_BS, STEPS), -7, dtype=torch.int64, device=device
    )

    def launches():
        for i in range(LONG_LAUNCHES):
            rows, keep = LONG_ROWS[i % len(LONG_ROWS)], i % 2 == 0
            if keep:
                sampler.launch_noise(rows=rows, key=KEY, offset=offset)
            sampler.sample(
                base_logits=local[:rows],
                anchor_tokens=anchor[:rows],
                temperatures=temperatures,
                logits_cache=cache if keep else None,
                draft_tokens=tokens[i],
            )

    def check(what):
        torch.cuda.synchronize()
        for i in range(LONG_LAUNCHES):
            rows, keep = LONG_ROWS[i % len(LONG_ROWS)], i % 2 == 0
            want = want_sampled if keep else want_greedy
            assert torch.equal(tokens[i, :rows], want[:rows]), (what, i)
        tokens.fill_(-7)

    launches()
    check("eager")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launches()
    for replay in range(LONG_REPLAYS):
        graph.replay()
        if replay % 100 == 99:
            check(replay)
    _assert_cache(cache, want_cache, temperatures, MAX_BS, "long")
    del graph

    # A fresh mailbox and counters three launches short of 2**32.
    sampler = _make_sampler(group, rank, world, w1, w2)
    sampler.control[NARROW_COUNT_WORDS] = -3
    for launch in range(6):
        for rows in (1, MAX_BS):  # the direct exchange and the fold
            out = torch.full((MAX_BS, STEPS), -7, dtype=torch.int64, device=device)
            cache.zero_()
            sampler.launch_noise(rows=rows, key=KEY, offset=offset)
            sampler.sample(
                base_logits=local[:rows],
                anchor_tokens=anchor[:rows],
                temperatures=temperatures,
                logits_cache=cache,
                draft_tokens=out,
            )
            torch.cuda.synchronize()
            assert torch.equal(out[:rows], want_sampled[:rows]), ("wrap", launch, rows)
            _assert_cache(cache, want_cache, temperatures, rows, ("wrap", launch))


def _check_wide(device, group, rank: int, world: int, seed: int):
    """9-64 rows on the wide kernel: tokens and kept logits, a graph replay of mixed
    row counts across all three kernels, and a wide counter wrap."""
    g = torch.Generator().manual_seed(seed)
    w1, w2 = _ternary(g, device, VOCAB, RANK), _ternary(g, device, VOCAB, RANK)
    sampler = _make_sampler(group, rank, world, w1, w2, max_bs=WIDE_BS)
    assert sampler.wide_rows == WIDE_BS and sampler.max_rows == WIDE_BS
    shard = slice(rank * VOCAB // world, (rank + 1) * VOCAB // world)
    offset = torch.tensor([5], dtype=torch.int64, device=device)
    mixed = torch.tensor(TEMPERATURE_SETS[2] * (WIDE_BS // 8), device=device)
    for rows in WIDE_ROWS:
        base = (torch.randn(rows, STEPS, VOCAB, generator=g) * 3).to(
            device, torch.bfloat16
        )
        anchor = torch.randint(0, VOCAB, (rows,), generator=g).to(device)
        for temperatures in (torch.zeros(WIDE_BS, device=device), mixed):
            for keep in (True, False):
                want_cache = (
                    torch.zeros(WIDE_BS, STEPS, VOCAB, device=device) if keep else None
                )
                want = _reference(
                    w1, w2, base, anchor, temperatures[:rows], offset, want_cache
                )
                for local in (True, False):
                    inputs = base[:, :, shard].contiguous() if local else base
                    for _ in range(3):
                        cache = (
                            torch.zeros(WIDE_BS, STEPS, VOCAB, device=device)
                            if keep
                            else None
                        )
                        tokens = torch.full(
                            (WIDE_BS, STEPS), -7, dtype=torch.int64, device=device
                        )
                        if keep:
                            sampler.launch_noise(rows=rows, key=KEY, offset=offset)
                        sampler.sample(
                            base_logits=inputs,
                            anchor_tokens=anchor,
                            temperatures=temperatures,
                            logits_cache=cache,
                            draft_tokens=tokens,
                        )
                        torch.cuda.synchronize()
                        what = ("wide", rows, keep, local, bool(temperatures.any()))
                        assert torch.equal(tokens[:rows], want), what
                        if keep:
                            _assert_cache(cache, want_cache, temperatures, rows, what)

    # One graph per row count; replays interleave the direct, narrow and wide kernels.
    base = (torch.randn(WIDE_BS, STEPS, VOCAB, generator=g) * 3).to(
        device, torch.bfloat16
    )
    local = base[:, :, shard].contiguous()
    anchor = torch.randint(0, VOCAB, (WIDE_BS,), generator=g).to(device)
    want_cache = torch.zeros(WIDE_BS, STEPS, VOCAB, device=device)
    want = _reference(w1, w2, base, anchor, mixed, offset, want_cache)
    cache = torch.zeros(WIDE_BS, STEPS, VOCAB, device=device)
    outs = [
        torch.full((rows, STEPS), -7, dtype=torch.int64, device=device)
        for rows in WIDE_GRAPH_ROWS
    ]

    def launch(rows, out):
        sampler.launch_noise(rows=rows, key=KEY, offset=offset)
        sampler.sample(
            base_logits=local[:rows],
            anchor_tokens=anchor[:rows],
            temperatures=mixed,
            logits_cache=cache,
            draft_tokens=out,
        )

    graphs = []
    for rows, out in zip(WIDE_GRAPH_ROWS, outs):
        launch(rows, out)  # warm-up outside capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(rows, out)
        graphs.append(graph)
    for replay in range(WIDE_REPLAYS):
        for graph in graphs:
            graph.replay()
        if replay % 2000 == 1999:
            torch.cuda.synchronize()
            for rows, out in zip(WIDE_GRAPH_ROWS, outs):
                assert torch.equal(out, want[:rows]), (replay, rows)
                out.fill_(-7)
    _assert_cache(cache, want_cache, mixed, WIDE_BS, "wide graph")
    del graphs

    # A fresh sampler whose wide counters are three launches short of 2**32.
    sampler = _make_sampler(group, rank, world, w1, w2, max_bs=WIDE_BS)
    sampler.wide_control[WIDE_COUNT_WORDS] = -3
    for launch_index in range(6):
        out = torch.full((WIDE_BS, STEPS), -7, dtype=torch.int64, device=device)
        cache.zero_()
        launch(WIDE_BS, out)
        torch.cuda.synchronize()
        assert torch.equal(out, want), ("wide wrap", launch_index)
        _assert_cache(cache, want_cache, mixed, WIDE_BS, ("wide wrap", launch_index))


@pytest.mark.parametrize("base_dtype", [torch.bfloat16, torch.float32])
def test_one_rank_matches_markov_sample_step(base_dtype):
    _check_equal(torch.device("cuda"), None, 0, 1, base_dtype, seed=1)


def test_one_rank_long_replay_and_counter_wrap():
    _check_long_replay(torch.device("cuda"), None, 0, 1, seed=3)


def _missing_peer_child(rows: int) -> None:
    """Rank 0 of two whose peer never launches: its waits must hit the deadline."""
    from sglang.srt.speculative.dspark_components.dspark_sharded_markov import (
        DSparkShardedMarkovSampler,
    )

    device = torch.device("cuda")
    g = torch.Generator().manual_seed(5)
    w1, w2 = _ternary(g, device, VOCAB, RANK), _ternary(g, device, VOCAB, RANK)
    sampler = DSparkShardedMarkovSampler(
        embed_weight=w1,
        head_weight=w2,
        vocab_size=VOCAB,
        num_steps=STEPS,
        cache_dtype=torch.float32,
        group=None,
        rank=0,
        world=2,
        max_bs=MAX_BS,
        timeout_s=1.0,
        keeps_logits=False,
    )
    absent_peer = torch.zeros_like(sampler._peer_buffer)
    sampler.peers = torch.tensor(
        [sampler._peer_buffer.data_ptr(), absent_peer.data_ptr()],
        dtype=torch.int64,
        device=device,
    )
    tokens = torch.zeros((MAX_BS, STEPS), dtype=torch.int64, device=device)
    sampler.sample(
        base_logits=torch.zeros(rows, STEPS, VOCAB // 2, device=device),
        anchor_tokens=torch.zeros(rows, dtype=torch.int64, device=device),
        temperatures=torch.zeros(MAX_BS, device=device),
        logits_cache=None,
        draft_tokens=tokens,
    )
    print("launched", flush=True)
    torch.cuda.synchronize()


@pytest.mark.parametrize("rows", [1, 4])  # the direct exchange and the fold
def test_missing_peer_stops_at_the_deadline(rows):
    module = os.path.splitext(os.path.basename(__file__))[0]
    start = time.monotonic()
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import {module}; {module}._missing_peer_child({rows})",
        ],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = child.stdout + child.stderr
    # The kernel launched, then the trap failed the process instead of a hang.
    assert child.returncode != 0 and "launched" in child.stdout, output
    # JIT builds are cached by the earlier tests; the wait itself is one second.
    assert time.monotonic() - start < 300, output


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _four_rank_worker(rank: int, world: int, port: int, mode: str) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world)
    group = dist.new_group(list(range(world)), backend="gloo")
    try:
        if mode == "long":
            _check_long_replay(device, group, rank, world, seed=9)
        elif mode == "wide":
            _check_wide(device, group, rank, world, seed=11)
        else:
            for base_dtype in (torch.bfloat16, torch.float32):
                _check_equal(device, group, rank, world, base_dtype, seed=7)
        dist.barrier(group=group)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
def test_four_ranks_match_markov_sample_step():
    torch.multiprocessing.spawn(
        _four_rank_worker, args=(4, _free_port(), "equal"), nprocs=4, join=True
    )


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
def test_four_ranks_long_replay_and_counter_wrap():
    torch.multiprocessing.spawn(
        _four_rank_worker, args=(4, _free_port(), "long"), nprocs=4, join=True
    )


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
def test_four_ranks_wide_rows_match_markov_sample_step():
    torch.multiprocessing.spawn(
        _four_rank_worker, args=(4, _free_port(), "wide"), nprocs=4, join=True
    )
