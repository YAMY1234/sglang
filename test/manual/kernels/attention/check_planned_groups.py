# SPDX-License-Identifier: Apache-2.0
"""GPU byte check of production grouped FMHA, including ragged group sizes."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import flashinfer.decode as fd
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 7, 8, 12, 16, 24, 32])
    parser.add_argument("--qlens", type=int, nargs="+", default=[1, 7])
    parser.add_argument("--banks", type=int, default=2)
    parser.add_argument("--calls", type=int, default=16)
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument(
        "--arms", nargs="+", choices=("sequential", "concurrent"),
        default=["sequential", "concurrent"],
    )
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("production_groups", args.source)
    groups = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(groups)
    torch.manual_seed(20261016)
    lengths = json.loads(args.vectors.read_text())["lengths"]
    counts = [(length + 63) // 64 for length in lengths]
    pages = sum(counts)
    table = torch.zeros((32, max(counts)), dtype=torch.int32, device="cuda")
    permutation = torch.randperm(pages, dtype=torch.int32, device="cuda")
    offset = 0
    for row, count in enumerate(counts):
        table[row, :count] = permutation[offset:offset + count]
        offset += count
    seq = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    banks = [tuple((torch.randn(pages, 2, 64, 256, device="cuda", dtype=torch.bfloat16) * .5).to(torch.float8_e4m3fn)
                   for _ in range(2)) for _ in range(args.banks)]
    workspaces = [torch.zeros(512 * 1024**2, dtype=torch.uint8, device="cuda") for _ in range(4)]
    counters = [torch.zeros(8 * 1024**2, dtype=torch.uint8, device="cuda") for _ in range(4)]
    streams = tuple(torch.cuda.Stream() for _ in range(4))
    results = []
    for batch in args.batches:
        for qlen in args.qlens:
            query = (torch.randn(batch * qlen, 32, 256, device="cuda") * .5).to(torch.float8_e4m3fn)
            plan = groups.DecodeGroupPlans().get(block_tables=table[:batch], seq_lens=seq[:batch])
            outputs = {arm: [torch.empty_like(query, dtype=torch.bfloat16) for _ in banks]
                       for arm in args.arms}
            active_kv = banks[0]

            def native(q, blocks, lens, out, group=0):
                fd.trtllm_batch_decode_with_kv_cache(
                    query=q, kv_cache=active_kv, workspace_buffer=workspaces[group],
                    block_tables=blocks, seq_lens=lens, max_seq_len=262144,
                    bmm1_scale=1 / 16, bmm2_scale=1., window_left=-1,
                    out=out, out_dtype=torch.bfloat16, q_len_per_req=qlen,
                    multi_ctas_kv_counter_buffer=counters[group])

            def run(arm, bank):
                nonlocal active_kv
                active_kv = banks[bank]
                kwargs = dict(query=query, plan=plan, num_splits=min(4, batch),
                              q_len_per_req=qlen, out=outputs[arm][bank],
                              out_dtype=torch.bfloat16, run_group=native)
                if arm == "concurrent":
                    groups.run_planned_concurrent_groups(streams=streams, **kwargs)
                else:
                    groups.run_planned_groups(**kwargs)

            graphs = []
            for arm in outputs:
                for bank in range(args.banks):
                    run(arm, bank)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for call in range(args.calls):
                        run(arm, call % args.banks)
                graphs.append(graph)
            for _ in range(args.replays):
                for graph in graphs:
                    graph.replay()
            torch.cuda.synchronize()
            passed = (
                all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                    for a, b in zip(outputs["sequential"], outputs["concurrent"]))
                if len(outputs) == 2 else None
            )
            row = {"batch": batch, "qlen": qlen, "banks": args.banks,
                   "graph_replays": args.replays, "calls_per_graph": args.calls, "byte_pass": passed}
            results.append(row)
            print(json.dumps(row), flush=True)
            if passed is False:
                raise RuntimeError("production groups byte mismatch")
            del graphs
    args.output.write_text(json.dumps({
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(args.vectors.read_bytes()).hexdigest(),
        "arms": args.arms, "results": results,
        "verdict": "PASS" if len(outputs) == 2 else "EXECUTED"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
