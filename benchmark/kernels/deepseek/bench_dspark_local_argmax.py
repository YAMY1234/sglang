"""Microbenchmark: DSpark target-verify greedy argmax at TP4, gathered vs exchanged.

  stock     multimem all-gather of the [rows, 32320] BF16 shards, FP32 copy, row_argmax
  exchange  LocalArgmaxExchange: per-chunk keys posted to every rank, reduced in the
            same cooperative launch

One process per GPU. For every row count, each arm is a CUDA graph of --inner launches;
replays alternate the arms (order reversed every replay) on new random logits, are timed
with CUDA events on rank 0, and the arms' ids must match. The logits stay L2-hot, as they
are in serving (the LM head writes them right before).

  python benchmark/kernels/deepseek/bench_dspark_local_argmax.py --world 4 --rows 6,24,48,96,384
"""

import argparse
import os
import socket
import statistics

import torch
import torch.distributed as dist

SHARD = 32_320


def _quantiles(values):
    ordered = sorted(values)

    def pick(q):
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return statistics.median(ordered), pick(0.1), pick(0.9)


def _worker(rank: int, args) -> None:
    from sglang.kernels.ops.speculative.row_argmax import row_argmax
    from sglang.srt.distributed.device_communicators import triton_symm_mem_ag as ag
    from sglang.srt.speculative.dspark_components.dspark_local_argmax import (
        LocalArgmaxExchange,
    )

    world = args.world
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.port)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world)
    cpu_group = dist.new_group(list(range(world)), backend="gloo")
    rows_list = [int(r) for r in args.rows.split(",")]
    max_rows = max(rows_list)
    state = ag.create_state(
        group=dist.group.WORLD,
        rank_in_group=rank,
        max_tokens=max_rows,
        hidden_size=SHARD * world,
        device=device,
    )
    exchange = LocalArgmaxExchange(
        max_rows=max_rows,
        group=cpu_group,
        rank=rank,
        world=world,
        device=device,
        timeout_s=args.timeout_s,
    )
    local = torch.zeros(max_rows, SHARD, dtype=torch.bfloat16, device=device)
    if rank == 0:
        print(
            f"# {torch.cuda.get_device_name(device)} world={world} exchange_ctas={exchange.ctas} "
            f"inner={args.inner} replays={args.replays}",
            flush=True,
        )

    for rows in rows_list:
        shard = local[:rows]
        out = {
            "stock": torch.zeros(rows, dtype=torch.int64, device=device),
            "exchange": None,
        }

        def stock():
            for _ in range(args.inner):
                full = ag.all_gather_inner(
                    state, shard, tp_hidden_dim=SHARD * world, safe=False
                )
                out["stock"].copy_(row_argmax(full.float()))

        def new():
            for _ in range(args.inner):
                out["exchange"] = exchange(shard, vocab_size=SHARD * world)

        graphs = {}
        for name, body in (("stock", stock), ("exchange", new)):
            body()
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                body()
            graphs[name] = graph
        times = {name: [] for name in graphs}
        events = {
            name: (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for name in graphs
        }
        mismatches = torch.zeros((), dtype=torch.int64, device=device)
        names = list(graphs)
        for replay in range(args.warmup + args.replays):
            gen = torch.Generator(device=device).manual_seed(replay * 64 + rank)
            shard.copy_(torch.randn(shard.shape, generator=gen, device=device) * 4)
            for name in names if replay % 2 == 0 else names[::-1]:
                torch.cuda.synchronize(device)
                dist.barrier(group=cpu_group)
                start, end = events[name]
                start.record()
                graphs[name].replay()
                end.record()
                if replay >= args.warmup:
                    torch.cuda.synchronize(device)
                    times[name].append(start.elapsed_time(end) * 1000.0 / args.inner)
            mismatches += (out["stock"] != out["exchange"]).any().long()
        if rank == 0:
            for name in names:
                median, p10, p90 = _quantiles(times[name])
                print(
                    f"arm={name} rows={rows} median_us={median:.2f} p10_us={p10:.2f} p90_us={p90:.2f} "
                    f"replays={len(times[name])} id_mismatch_replays={int(mismatches)}",
                    flush=True,
                )
        assert int(mismatches) == 0, (
            f"rank {rank}: arms disagree in {int(mismatches)} replays (rows {rows})"
        )
        del graphs
    dist.barrier(group=cpu_group)
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--rows", default="6,24,48,96,384")
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--inner", type=int, default=20, help="launches per graph replay"
    )
    parser.add_argument("--timeout-s", type=float, default=20.0)
    args = parser.parse_args()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        args.port = s.getsockname()[1]
    torch.multiprocessing.spawn(_worker, args=(args,), nprocs=args.world, join=True)


if __name__ == "__main__":
    main()
