"""The vocab-sharded argmax exchange returns the stock greedy verify's ids bit for bit:
row_argmax (AcceptGreedy's fused argmax) and torch.argmax of the gathered FP32 logits,
at one rank and four spawned ranks (symmetric memory), BF16 and FP32 shards, 1 to 384
rows. Ties go to the smaller index across shards, chunks and whole rows, -0 ties +0,
NaN follows torch.argmax, and LM-head padding columns past the vocabulary never win.
Graph replays of mixed row counts over > 2 * 32767 launches and a launch-counter wrap
reuse the mailbox safely, the sink's side-stream fork joins inside a CUDA graph, and a
rank whose peer never arrives stops at its deadline instead of hanging, in the exchange
and in the flag-gated logits gather.
"""

import os
import re
import socket
import subprocess
import sys
import time

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

SHARD = 32_320  # DeepSeek-V4.1 vocab / TP4
# A vocabulary that leaves 40 padding columns in the last shard.
PADDING = 40
MAX_ROWS = 384
ROWS = (1, 6, 24, 48, 96, 384)
LAUNCHES_PER_GRAPH = 10
GRAPH_ROWS = (6, 384, 24)
REPLAYS = 2400
# Arbitrary; far above any launch, so only a missing peer reaches it.
TIMEOUT_S = 20.0
# Short so that the ranks a skipped gather leaves waiting trap quickly.
GATHER_DEADLINE_S = 2.0


def _gpu_ready() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in (
        (10, 0),
        (10, 3),
    )


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100/SM103 GPU")


def _make_exchange(group, rank, world, device):
    from sglang.srt.speculative.dspark_components.dspark_local_argmax import (
        LocalArgmaxExchange,
    )

    return LocalArgmaxExchange(
        max_rows=MAX_ROWS,
        group=group,
        rank=rank,
        world=world,
        device=device,
        timeout_s=TIMEOUT_S,
    )


def _logits(rows: int, world: int, vocab: int, dtype, seed: int, device):
    """Full padded logits [rows, world * SHARD], identical on every rank, with forced
    ties; padding columns hold the row's largest value."""
    g = torch.Generator().manual_seed(seed)
    full = (torch.randn(rows, world * SHARD, generator=g) * 4).to(dtype)
    top = full.float().max().item() + 1
    special = (
        (0, 5, (world - 1) * SHARD + 7),  # tie across shards
        (2, 2023, 2024, SHARD - 1),  # ties across chunks
        (3, vocab - 1),  # the last vocabulary column
    )
    for row, *cols in special:
        if row < rows:
            full[row, cols] = top
    if rows > 1:
        full[1].fill_(-3.0)  # whole-row tie
    if rows > 4:
        full[4].fill_(-1.0)
        full[4, 3] = -0.0  # ties the +0 of the next shard
        full[4, min(SHARD + 3, vocab - 1)] = 0.0
    if rows > 5:
        full[5].fill_(float("-inf"))
    full[:, vocab:] = top + 1
    return full.to(device)


def _check_rows(exchange, rank: int, world: int, device):
    from sglang.kernels.ops.speculative.row_argmax import row_argmax

    shard = slice(rank * SHARD, (rank + 1) * SHARD)
    for vocab in (world * SHARD, world * SHARD - PADDING):
        for dtype in (torch.bfloat16, torch.float32):
            for rows in ROWS:
                full = _logits(rows, world, vocab, dtype, seed=rows, device=device)
                gathered = full.float()[:, :vocab]
                want = gathered.argmax(dim=1)
                # The stock greedy verify's argmax (AcceptGreedy, fused) agrees.
                assert torch.equal(row_argmax(gathered.contiguous()), want)
                local = full[:, shard].contiguous()
                for _ in range(3):  # both parities and a reused one
                    ids = exchange(local, vocab_size=vocab)
                    torch.cuda.synchronize(device)
                    assert torch.equal(ids, want), (vocab, dtype, rows)


def _check_nan(exchange, rank: int, world: int, device):
    """NaN beats every number and the first NaN wins, as in torch.argmax."""
    vocab = world * SHARD
    # Identical on every rank: the shards must come from one logits tensor.
    full = torch.randn(4, vocab, generator=torch.Generator().manual_seed(3)).to(device)
    full[0, [vocab - 2, 17]] = float("nan")
    full[1, [SHARD * (world - 1) + 1]] = -float("nan")
    full[2, [SHARD - 1, 3]] = torch.tensor([float("nan"), -float("nan")], device=device)
    want = full.argmax(dim=1)
    local = full[:, rank * SHARD : (rank + 1) * SHARD].contiguous()
    ids = exchange(local, vocab_size=vocab)
    torch.cuda.synchronize(device)
    assert torch.equal(ids, want), (ids, want)


def _check_graph_replay(exchange, rank: int, world: int, device):
    """Graphs of mixed row counts share the mailbox and the launch counters."""
    vocab = world * SHARD
    shard = slice(rank * SHARD, (rank + 1) * SHARD)
    inputs, wants, outs = [], [], []
    for i, rows in enumerate(GRAPH_ROWS):
        full = _logits(rows, world, vocab, torch.bfloat16, seed=100 + i, device=device)
        inputs.append(full[:, shard].contiguous())
        wants.append(full.float().argmax(dim=1))
        outs.append(
            torch.full((LAUNCHES_PER_GRAPH, rows), -1, dtype=torch.int64, device=device)
        )
    graphs = []
    for local, out in zip(inputs, outs):
        exchange(local, vocab_size=vocab)  # warm-up outside capture
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for k in range(LAUNCHES_PER_GRAPH):
                out[k].copy_(exchange(local, vocab_size=vocab))
        graphs.append(graph)
    for replay in range(REPLAYS):
        for graph in graphs:
            graph.replay()
        if replay % 400 == 399:
            torch.cuda.synchronize(device)
            for out, want in zip(outs, wants):
                assert torch.equal(out, want.expand_as(out)), replay
                out.fill_(-1)
    del graphs
    return inputs, wants


def _check_sink_graph(exchange, rank: int, world: int, device):
    """The sink's ids feed the main stream while a predicated full-logits copy runs on
    its stream; the copy lands only while the flag is set."""
    from sglang.kernels.ops.speculative.dspark.local_argmax import predicated_copy_
    from sglang.srt.speculative.dspark_components.dspark_local_argmax import (
        LocalArgmaxSink,
    )

    vocab = world * SHARD
    sink = LocalArgmaxSink(exchange=exchange, device=device)
    full = _logits(24, world, vocab, torch.bfloat16, seed=7, device=device)
    local = full[:, rank * SHARD : (rank + 1) * SHARD].contiguous()
    want = full.float().argmax(dim=1)
    ids_out = torch.full_like(want, -1)
    copy_out = torch.full(local.shape, -7.0, device=device)

    def step():
        sink.fork(local, vocab_size=vocab)
        predicated_copy_(copy_out, local, sink.full_logits_flag)
        sink.unfork()
        ids_out.copy_(sink.ids)
        sink.join()

    step()  # warm-up outside capture
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for needed in (False, True, False):
        copy_out.fill_(-7.0)
        ids_out.fill_(-1)
        sink.set_full_logits_needed(needed)
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.equal(ids_out, want), needed
        if needed:
            assert torch.equal(copy_out, local.float())
        else:
            assert torch.all(copy_out == -7.0)
    del graph


def _check(device, group, rank: int, world: int):
    exchange = _make_exchange(group, rank, world, device)
    _check_rows(exchange, rank, world, device)
    _check_nan(exchange, rank, world, device)
    inputs, wants = _check_graph_replay(exchange, rank, world, device)
    _check_sink_graph(exchange, rank, world, device)

    # A fresh mailbox and launch counters three launches short of 2**32.
    exchange = _make_exchange(group, rank, world, device)
    exchange.counters.fill_(-3)
    for launch in range(6):
        local, want = inputs[launch % len(inputs)], wants[launch % len(wants)]
        ids = exchange(local, vocab_size=world * SHARD)
        torch.cuda.synchronize(device)
        assert torch.equal(ids, want), ("wrap", launch)


def test_one_rank_matches_stock_argmax():
    _check(torch.device("cuda"), None, 0, 1)


def _missing_peer_child() -> None:
    """Rank 0 of two whose peer never launches: its waits must hit the deadline."""
    from sglang.srt.speculative.dspark_components.dspark_local_argmax import (
        LocalArgmaxExchange,
    )

    device = torch.device("cuda")
    exchange = LocalArgmaxExchange(
        max_rows=MAX_ROWS, group=None, rank=0, world=2, device=device, timeout_s=1.0
    )
    absent_peer = torch.zeros_like(exchange._mailbox)
    exchange.peers = torch.tensor(
        [exchange._mailbox.data_ptr(), absent_peer.data_ptr()],
        dtype=torch.int64,
        device=device,
    )
    local = torch.zeros(6, SHARD, dtype=torch.bfloat16, device=device)
    exchange(local, vocab_size=2 * SHARD)
    print("launched", flush=True)
    torch.cuda.synchronize()


def test_missing_peer_stops_at_the_deadline():
    module = os.path.splitext(os.path.basename(__file__))[0]
    start = time.monotonic()
    child = subprocess.run(
        [sys.executable, "-c", f"import {module}; {module}._missing_peer_child()"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
        text=True,
        timeout=600,
    )
    output = child.stdout + child.stderr
    # The kernel launched, then the trap failed the process instead of a hang.
    assert child.returncode != 0 and "launched" in child.stdout, output
    # The JIT build is cached by the earlier test; the wait itself is one second.
    assert time.monotonic() - start < 300, output


def _flag_split_worker(rank: int, world: int, port: int) -> None:
    """The last rank reads a zero enable flag and skips the gather; the others enter
    its barriers, which must trap at the deadline."""
    import torch.distributed as dist

    from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
        all_gather_inner,
        create_state,
    )

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world)
    group = dist.new_group(list(range(world)), backend="gloo")
    state = create_state(
        group=group, rank_in_group=rank, max_tokens=8, hidden_size=world * 64
    )
    shard = torch.zeros(8, 64, dtype=torch.bfloat16, device=device)
    enable = torch.full((1,), int(rank < world - 1), dtype=torch.int32, device=device)
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    all_gather_inner(
        state,
        shard,
        tp_hidden_dim=world * 64,
        enable=enable,
        timeout_ns=int(GATHER_DEADLINE_S * 1e9),
    )
    # After the JIT build, which runs before the launch.
    start = time.monotonic()
    try:
        torch.cuda.synchronize(device)
    except RuntimeError as error:
        print(f"trapped after {time.monotonic() - start:.2f} s: {error}", flush=True)
        raise
    print(f"rank {rank} finished", flush=True)
    # Keep the skipping rank's signal pad mapped until the others fail.
    time.sleep(10 * GATHER_DEADLINE_S)


def _flag_split_main() -> None:
    torch.multiprocessing.spawn(
        _flag_split_worker, args=(4, _free_port()), nprocs=4, join=True
    )


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
def test_gather_flag_disagreement_stops_at_the_deadline():
    module = os.path.splitext(os.path.basename(__file__))[0]
    child = subprocess.run(
        [sys.executable, "-c", f"import {module}; {module}._flag_split_main()"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = child.stdout + child.stderr
    assert child.returncode != 0 and "rank 3 finished" in child.stdout, output
    waits = [float(m) for m in re.findall(r"trapped after ([0-9.]+) s", output)]
    # A wait of about the deadline: not a fast fault, and not a hang.
    assert waits and all(
        0.9 * GATHER_DEADLINE_S <= wait < GATHER_DEADLINE_S + 5 for wait in waits
    ), output


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _four_rank_worker(rank: int, world: int, port: int) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world)
    group = dist.new_group(list(range(world)), backend="gloo")
    try:
        _check(device, group, rank, world)
        dist.barrier(group=group)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
def test_four_ranks_match_stock_argmax():
    torch.multiprocessing.spawn(
        _four_rank_worker, args=(4, _free_port()), nprocs=4, join=True
    )
