"""docs/167 P4 / V3: Lightning decode throughput at C1 and C16 for production, reference and stock, same allocation.

Each arm is one server start (stage2.server with the arm's profile; stock = no release); for each concurrency C the
driver keeps C requests in flight for `--seconds` (twice), counting completion tokens.  Output: one JSON per arm
and a summary with production / reference and production / stock ratios.  Not a numerical gate.
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from lightning_sgl_stage2 import now, request, save, server

PROMPT = ("Write a detailed, step by step explanation of how a transformer language model generates text, "
          "covering tokenisation, embeddings, attention, feed-forward layers and sampling. Be thorough.")


def load(endpoint, concurrency, seconds, max_new):
    stop = time.monotonic() + seconds
    counts = [0] * concurrency
    requests_done = [0] * concurrency
    errors = []

    def worker(i):
        while time.monotonic() < stop:
            try:
                r = request(endpoint, "/generate", {"text": PROMPT, "sampling_params": {
                    "temperature": 0, "max_new_tokens": max_new, "ignore_eos": True}}, timeout=600)
                counts[i] += int(r["meta_info"].get("completion_tokens", 0))
                requests_done[i] += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")
                return
    t0 = time.monotonic()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0
    return dict(concurrency=concurrency, seconds=wall, completion_tokens=sum(counts), requests=sum(requests_done),
                tokens_per_second=sum(counts) / wall if wall else None, errors=errors[:5])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--duet", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--arms", default="production,reference,stock")
    ap.add_argument("--concurrencies", default="1,16")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-running-requests", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = dict(cell="p4-lightning-throughput", started_at=now(), job_id=os.environ.get("SLURM_JOB_ID"), arms={})
    port = 31350
    for arm in args.arms.split(","):
        sargs = SimpleNamespace(model=args.model, duet=args.duet, gpu=args.gpu,
                                max_running_requests=args.max_running_requests,
                                duet_numerics="production" if arm == "production" else "reference")
        mode = "stock" if arm == "stock" else "duet"
        result = dict(arm=arm, mode=mode, status="running", runs=[])
        try:
            with server(sargs, mode, args.out / f"{arm}-server", port=port) as (endpoint, meta):
                result["ready_seconds"] = meta.get("ready_seconds")
                # warm-up
                request(endpoint, "/generate", {"text": PROMPT, "sampling_params": {"temperature": 0, "max_new_tokens": 32}})
                for c in (int(x) for x in args.concurrencies.split(",")):
                    for rep in range(args.repeats):
                        run = load(endpoint, c, args.seconds, args.max_new)
                        run["repeat"] = rep
                        result["runs"].append(run)
                        print(f"{arm} C{c} rep{rep}: {run['tokens_per_second']:.1f} tok/s ({run['completion_tokens']} tokens)", flush=True)
                        save(args.out / f"{arm}.json", result)
            result["status"] = "pass" if all(not r["errors"] for r in result["runs"]) else "fail"
        except Exception as exc:  # noqa: BLE001
            result.update(status="fail", error=f"{type(exc).__name__}: {exc}")
        save(args.out / f"{arm}.json", result)
        summary["arms"][arm] = result
        port += 1
    rates = {}
    for arm, res in summary["arms"].items():
        for c in sorted({r["concurrency"] for r in res.get("runs", [])}):
            vals = [r["tokens_per_second"] for r in res["runs"] if r["concurrency"] == c and r["tokens_per_second"]]
            rates[(arm, c)] = sum(vals) / len(vals) if vals else None
    ratios = {}
    for c in sorted({c for _, c in rates}):
        p, r, s = rates.get(("production", c)), rates.get(("reference", c)), rates.get(("stock", c))
        ratios[f"C{c}"] = dict(production=p, reference=r, stock=s,
                               production_over_reference=(p / r) if p and r else None,
                               production_over_stock=(p / s) if p and s else None,
                               reference_over_stock=(r / s) if r and s else None)
    summary["ratios"] = ratios
    summary["finished_at"] = now()
    save(args.out / "p4-throughput.json", summary)
    print(json.dumps(ratios, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
