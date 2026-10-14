"""CPU-only TTFT decomposition of a Lightning C8 service pair (docs/209).

Server prefills are serialized (chunked prefill off, overlap off, no radix),
so a request whose prompt arrived while another request's prefill ran cannot
start before that prefill's first token. For each in-window request:
  wait    = max(arrival, latest other first token in (arrival, own first token - 5 ms)) - arrival
  service = own first token - that start (own prefill + boundary/tail + sampling + delivery)
The 5 ms guard keeps co-batched (B2) neighbours from counting as blockers.
Decode stall = overlap of a request's [first token, end] with other requests'
reconstructed [start, first token] prefill intervals. Server histograms are
read independently as a cross-check; their cohorts differ and are not merged.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

ARMS = ("original-off", "factored")
GUARD_NS = 5_000_000


def ols(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    residual = sum((y - my - slope * (x - mx)) ** 2 for x, y in zip(xs, ys))
    total = sum((y - my) ** 2 for y in ys)
    return dict(intercept_s=my - slope * mx, us_per_token=slope * 1e6, r2=1 - residual / total, n=len(xs))


def arm_report(root, read):
    rows = [json.loads(line) for line in read(root / "C8/request-timeseries.jsonl").splitlines()]
    rows = [r for r in rows if r["successful"]]
    arrive = [r["request_start_ns"] for r in rows]
    first = [r["first_token_ns_derived"] for r in rows]
    end = [r["request_end_ns"] for r in rows]
    start, blockers = [], []
    for i in range(len(rows)):
        ahead = [first[j] for j in range(len(rows))
                 if j != i and arrive[i] < first[j] < first[i] - GUARD_NS]
        start.append(max([arrive[i], *ahead]))
        blockers.append(len(ahead))
    cohort = [i for i, r in enumerate(rows) if r["in_official_cohort"]]
    ttft = [(first[i] - arrive[i]) / 1e9 for i in cohort]
    wait = [(start[i] - arrive[i]) / 1e9 for i in cohort]
    service = [(first[i] - start[i]) / 1e9 for i in cohort]
    length = [rows[i]["input_length"] for i in cohort]
    free = [k for k, i in enumerate(cohort) if blockers[i] == 0]
    middle = sorted(range(len(cohort)), key=lambda k: ttft[k])[len(cohort) // 2 - 5:len(cohort) // 2 + 5]
    stall, decode, tokens = 0.0, 0.0, 0
    for i in cohort:
        if rows[i]["output_length"] <= 1:
            continue
        a, b = first[i], end[i]
        stall += sum(max(0, min(first[j], b) - max(start[j], a)) for j in range(len(rows)) if j != i) / 1e9
        decode += (b - a) / 1e9
        tokens += rows[i]["output_length"] - 1
    exported = json.loads(read(root / "C8/rep1/aiperf_artifacts/server_metrics_export.json"))
    phase = exported["summary"]["phase_time_ranges"]["profiling"]
    server = {}
    for name in ("sglang:queue_time_seconds", "sglang:per_stage_req_latency_seconds",
                 "sglang:time_to_first_token_seconds", "sglang:inter_token_latency_seconds"):
        for series in exported["metrics"][name]["series"]:
            label = series["labels"].get("stage", "all")
            bins = [t for t in series.get("timeslices") or []
                    if phase["start_ns"] <= t["start_ns"] and t["end_ns"] <= phase["end_ns"]]
            count = sum(t["count"] for t in bins)
            if count:
                server[f"{name}[{label}]"] = dict(count=count, mean_s=sum(t["sum"] for t in bins) / count)
    return dict(
        requests=len(cohort),
        ttft=dict(p50_s=statistics.median(ttft), mean_s=statistics.fmean(ttft)),
        wait=dict(p50_s=statistics.median(wait), mean_s=statistics.fmean(wait),
                  fraction_waited=sum(w > 0 for w in wait) / len(wait)),
        service=dict(p50_s=statistics.median(service), mean_s=statistics.fmean(service)),
        median_decile=dict(ttft_s=statistics.fmean(ttft[k] for k in middle),
                           wait_s=statistics.fmean(wait[k] for k in middle),
                           service_s=statistics.fmean(service[k] for k in middle),
                           input_tokens=statistics.fmean(length[k] for k in middle)),
        input_tokens=dict(mean=statistics.fmean(length), p50=statistics.median(length)),
        service_fit_unblocked=ols([length[k] for k in free], [service[k] for k in free]),
        prefill_occupancy_of_900s=sum(service) / 900,
        decode=dict(stall_fraction=stall / decode, ms_per_token_with_stalls=decode / tokens * 1e3,
                    ms_per_token_without_stalls=(decode - stall) / tokens * 1e3),
        server_histograms=server)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="service pair directory with original-off/ and factored/")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    hashes = {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path.relative_to(args.root))] = hashlib.sha256(raw).hexdigest()
        return raw.decode()

    result = {arm: arm_report(args.root / arm, read) for arm in ARMS}
    result["inputs_sha256"] = hashes
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for arm in ARMS:
        r = result[arm]
        f = r["service_fit_unblocked"]
        print(f"{arm}: TTFT50 {r['ttft']['p50_s']:.3f} | wait mean {r['wait']['mean_s']:.3f} "
              f"({r['wait']['fraction_waited']:.0%} waited) | service mean {r['service']['mean_s']:.3f} | "
              f"fit {f['intercept_s']:.4f}s + {f['us_per_token']:.2f}us/tok R2 {f['r2']:.4f} n={f['n']} | "
              f"prefill occupancy {r['prefill_occupancy_of_900s']:.1%} | decode stall {r['decode']['stall_fraction']:.1%}")


if __name__ == "__main__":
    main()
