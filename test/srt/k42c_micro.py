"""K4.2c micro + bitwise check (docs/223 §3.9): host cost of the full-N publication launch, fast bind off vs on.

Builds the served-shape pool (k42_module_bitwise.build_pool), the compact full-N workspace and the AGG whole-prefix
graph exactly as gdn_prefill_agg_contract.prewarm does (FullNWorkspace(pool, 16, chunk) + PrefillBatchGraph(
include_tail=False, workspace, shared=workspace.shared)), snapshots prompt-end states into the workspace like
BatchCollector.add, then per rep and per variant (from identical pool bytes):
  off: PDBatchPublication(agg).submit + start_after_forward with SGLANG_GDN_K42_FAST_BIND off (per-layer bind)
  on : same with the fast bind (states proven resident, per-layer bind and record_stream skipped)
Host time = perf_counter around start_after_forward (the critical host segment at the end of a full-N forward).
Every pool tensor is compared bitwise between off and on. Exit 0 iff all equal and the fast path actually engaged.

  PYTHONPATH=python python3 test/srt/k42c_micro.py --heads 24 --rows 1 --reps 30 --out k42c.json
"""
import argparse
import json
import os
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k42_module_bitwise as mb  # noqa: E402  (sets the served recipe env before sglang imports)

import torch  # noqa: E402

from sglang.srt.mem_cache import gdn_factored_pool as gp  # noqa: E402
from sglang.srt.mem_cache import gdn_pd_publication as pubmod  # noqa: E402
from sglang.srt.mem_cache.gdn_fulln_workspace import FullNWorkspace  # noqa: E402
from sglang.srt.mem_cache.gdn_prefill_batch_graph import PrefillBatchGraph  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--rows", default="1")
    ap.add_argument("--tracked", action="store_true")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--out", default="k42c.json")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()
    pool, _ = mb.build_pool(a.heads, tmp)
    ws = FullNWorkspace(pool, 16, a.chunk)
    graph = PrefillBatchGraph(include_tail=False, workspace=ws, shared=ws.shared)
    policy = (gp.ORTH_METHOD, gp.ORTH_WARPS_OVERRIDE, gp.factorize_dense)
    graph.prewarm(pool, eager=gp.factorize_layers, policy=policy)
    result = dict(heads=a.heads, tracked=a.tracked, reps=a.reps, cases=[])
    ok = True
    for rows in [int(r) for r in a.rows.split(",")]:
        base = mb.snapshot(pool)
        times = {"off": [], "on": []}
        equal = True
        engaged = False
        for rep in range(a.reps):
            src, plan, controls, _ = mb.make_rep(pool, rows, a.tracked, seed=1000 + rep)
            finals = {}
            for variant in ("off", "on"):
                mb.restore(pool, base)
                states = []
                for i in range(len(mb.GDN_LAYERS)):
                    tracked = None if src["tracked"] is None else src["tracked"][i]
                    states.append(ws.snapshot(i, src["normal"][i], tracked, 4096))
                torch.cuda.synchronize()
                pubmod._FAST_BIND = variant == "on"
                publication = pubmod.PDBatchPublication(pool, pubmod.CudaPublicationRuntime(pool.a.device), agg=True)
                publication.submit(graph, plan, states, controls, eager=gp.factorize_layers, policy=policy)
                t0 = time.perf_counter()
                publication.start_after_forward()
                t1 = time.perf_counter()
                publication.join()
                torch.cuda.synchronize()
                if rep:  # first rep warms the Python paths
                    times[variant].append((t1 - t0) * 1e6)
                if variant == "on":
                    engaged = engaged or getattr(publication, "_fast_bind_logged", False)
                finals[variant] = mb.snapshot(pool)
            for key in finals["off"]:
                if not torch.equal(finals["off"][key], finals["on"][key]):
                    equal = False
        pubmod._FAST_BIND = False
        case = dict(rows=rows, bitwise=equal, fast_path_engaged=engaged,
                    host_us_off=statistics.median(times["off"]), host_us_on=statistics.median(times["on"]),
                    host_us_off_p90=sorted(times["off"])[int(0.9 * len(times["off"]))],
                    host_us_on_p90=sorted(times["on"])[int(0.9 * len(times["on"]))])
        case["saving_us"] = case["host_us_off"] - case["host_us_on"]
        result["cases"].append(case)
        ok = ok and equal and engaged
        print(json.dumps(case))
        mb.restore(pool, base)
    result["pass"] = ok
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print("PASS" if ok else "FAIL", a.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
