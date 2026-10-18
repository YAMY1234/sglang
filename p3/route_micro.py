"""Matched P-role forward microbenchmark; launch with route_micro.sh.

--factory module:function constructs one P model per TP rank and returns a
workload with prepare(prefix_tokens, extend_tokens, seed), forward(), join(),
and counters() methods. prepare restores identical input/checkpoint state,
outside timing, after route selection; join drains publication. counters is a
monotonic mapping containing 'fallbacks'. The factory owns frozen model args,
TP initialization, prefix/cache setup, P0_EVENTS and CUDA timeline recording.
No HTTP, D model, or model reload belongs inside forward().

Each paired ABBA block restores the same seed. Thirty blocks per comparison
and shape; 0n/1 for P5, also 0n/0 and 0/1 for P3. Full work includes the final
publication join. Report raw device and host samples; never infer acceptance
from synthetic model-free timings.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import time

SHAPES = ((16384, 1024), (32768, 2048), (0, 4096))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--stage", choices=("p5", "p3"), default="p5")
    parser.add_argument("--blocks", type=int, default=30)
    args = parser.parse_args()
    if args.blocks != 30:
        parser.error("pre-registered measurement requires 30 ABBA blocks")
    if args.stage == "p3":
        os.environ['SGLANG_GDN_PD_SHALLOW_FULLN'] = '1'
        os.environ['TWINSTAR_PD_EMITTER_GRAPH'] = '1'
    import torch
    from sglang.srt.mem_cache.gdn_prefill_agg_contract import set_pd_micro_route

    module, function = args.factory.split(":", 1)
    workload = getattr(importlib.import_module(module), function)()
    pairs = (("0n", "1"),) if args.stage == "p5" else (("0n", "0"), ("0", "1"), ("0n", "1"))
    args.out.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    (args.out / f"device-r{rank}.json").write_text(json.dumps(dict(
        name=device.name, capability=[device.major, device.minor], torch=torch.__version__,
        cuda=torch.version.cuda, stage=args.stage, pairs=pairs), indent=2))
    with (args.out / f"samples-r{rank}.jsonl").open("w") as output:
        for prefix, extend in SHAPES:
            # Warm both route graphs, emitter caches, kernels and exact-tail.
            for route in dict.fromkeys(route for pair in pairs for route in pair):
                set_pd_micro_route(route)
                # The second changing-input replay also completes the emitter
                # rebind byte guard before any scored ABBA sample.
                for seed in (42, 43):
                    workload.prepare(prefix, extend, seed)
                    workload.forward()
                    workload.join()
                    torch.cuda.synchronize()
            for pair in pairs:
                for block in range(args.blocks):
                    for order, route in enumerate((*pair, *reversed(pair))):
                        set_pd_micro_route(route)
                        workload.prepare(prefix, extend, 1000 + block)
                        torch.cuda.synchronize()
                        before = dict(workload.counters())
                        if "fallbacks" not in before:
                            raise RuntimeError("workload must expose fallback counters")
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        began = time.perf_counter_ns()
                        workload.forward()
                        workload.join()
                        end.record()
                        end.synchronize()
                        host_ms = (time.perf_counter_ns() - began) / 1e6
                        after = dict(workload.counters())
                        if after["fallbacks"] != before["fallbacks"]:
                            raise RuntimeError("P-route fallback in measured forward")
                        row = dict(rank=rank, prefix=prefix, extend=extend, pair=pair,
                                   block=block, order=order, route=route,
                                   gpu_ms=start.elapsed_time(end), host_ms=host_ms,
                                   before=before, after=after)
                        output.write(json.dumps(row) + "\n")
                        output.flush()
                        print("P0_EVENTS " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
