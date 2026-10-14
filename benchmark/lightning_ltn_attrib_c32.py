"""docs/209 §10: per-request wait/service/decode-stall split of a C32 pair from raw profile_export.jsonl.

Same method as lightning_ltn_attrib_window.py (serialized prefill, 5 ms co-batch guard), but reads
the raw AIPerf export because the original arm of 4487159 has no request-timeseries (reducer
rejected it for one client transport error). Errored/cancelled records are excluded.
"""
import json, statistics, sys
from pathlib import Path
root = Path(sys.argv[1])
def ols(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    b = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / sum((x-mx)**2 for x in xs)
    return my - b*mx, b
out = {}
for arm in ("original-off", "factored"):
    art = root/arm/"C32/rep1/aiperf_artifacts"
    start = json.loads((art/"server_metrics_export.json").read_text())["summary"]["phase_time_ranges"]["profiling"]["start_ns"]
    rows = []
    for line in (art/"profile_export.jsonl").open():
        r = json.loads(line); m = r["metadata"]
        if r.get("error") or m.get("was_cancelled"): continue
        a = m["request_start_ns"]; e = m["request_end_ns"]; t = r["metrics"]["time_to_first_token"]["value"]*1e6
        rows.append(dict(a=a, f=a+int(t), e=e, L=r["metrics"]["input_sequence_length"]["value"],
                         O=r["metrics"]["output_sequence_length"]["value"],
                         cohort=m["benchmark_phase"] == "profiling" and start <= e <= start+900*10**9))
    F = [r["f"] for r in rows]
    for i, r in enumerate(rows):
        ahead = [F[j] for j in range(len(rows)) if j != i and r["a"] < F[j] < r["f"]-5_000_000]
        r["s"] = max([r["a"], *ahead]); r["nb"] = len(ahead)
    c = [r for r in rows if r["cohort"]]
    ttft = [(r["f"]-r["a"])/1e9 for r in c]; wait = [(r["s"]-r["a"])/1e9 for r in c]; svc = [(r["f"]-r["s"])/1e9 for r in c]
    free = [r for r in c if r["nb"] == 0]
    b0, b1 = ols([r["L"] for r in free], [(r["f"]-r["s"])/1e9 for r in free])
    stall = dec = tok = 0
    for r in c:
        if r["O"] <= 1: continue
        stall += sum(max(0, min(q["f"], r["e"]) - max(q["s"], r["f"])) for q in rows if q is not r)/1e9
        dec += (r["e"]-r["f"])/1e9; tok += r["O"]-1
    out[arm] = dict(n=len(c), ttft_p50=statistics.median(ttft), wait_mean=statistics.fmean(wait), waited=sum(w > 0 for w in wait)/len(wait),
                    svc_mean=statistics.fmean(svc), fit_intercept_s=b0, fit_us_per_token=b1*1e6, n_free=len(free),
                    prefill_occupancy=sum(svc)/900, decode_stall=stall/dec, decode_ms_tok_excl=(dec-stall)/tok*1e3, decode_ms_tok_incl=dec/tok*1e3,
                    input_mean=statistics.fmean(r["L"] for r in c), output_mean=statistics.fmean(r["O"] for r in c))
    print(arm, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[arm].items()})
Path(sys.argv[2]).write_text(json.dumps(out, indent=2)+"\n")
