"""docs/209 §9.9: in-window clean decode step by batch (Decode-log intervals with no Prefill in between).

step_ms = 1000 * running-req / gen throughput of the 40-step log interval. Usage: ROOT OUT.json"""
import json, re, sys, datetime, statistics
from zoneinfo import ZoneInfo
from pathlib import Path
root = Path(sys.argv[1]); tz = ZoneInfo("America/Los_Angeles"); out = {}
micro = {"original-off": {1: 2.240573, 8: 2.550896, 32: 3.517497}, "factored": {1: 2.374472, 8: 3.002473, 32: 5.008242}}
bins = {"1-4": (1, 5), "5-8": (5, 9), "9-16": (9, 17), "17-24": (17, 25), "25-32": (25, 33)}
for arm in micro:
    start = json.loads((root/arm/"C32/rep1/aiperf_artifacts/server_metrics_export.json").read_text())["summary"]["phase_time_ranges"]["profiling"]["start_ns"]/1e9
    clean, prev_prefill = {k: [] for k in bins}, True
    for line in (root/arm/"server.log").open():
        m = re.match(r"\[(\S+ \S+)\] (Decode|Prefill) batch", line)
        if not m: continue
        t = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz).timestamp()
        if m.group(2) == "Prefill": prev_prefill = True; continue
        b = int(re.search(r"#running-req: (\d+)", line).group(1)); thr = float(re.search(r"gen throughput \(token/s\): ([\d.]+)", line).group(1))
        if start+1 <= t < start+899 and not prev_prefill and thr > 0:
            for k, (lo, hi) in bins.items():
                if lo <= b < hi: clean[k].append((b, 1000*b/thr))
        prev_prefill = False
    out[arm] = {k: dict(n=len(v), mean_batch=statistics.fmean(x for x, _ in v), step_ms_p50=statistics.median(y for _, y in v)) for k, v in clean.items() if v}
for k in bins:
    o, f = out["original-off"].get(k), out["factored"].get(k)
    if o and f: print(f"B {k:6s} orig n={o['n']:4d} step p50 {o['step_ms_p50']:.3f} ms | fix n={f['n']:4d} step p50 {f['step_ms_p50']:.3f} ms | ratio {f['step_ms_p50']/o['step_ms_p50']:.3f}")
Path(sys.argv[2]).write_text(json.dumps(out, indent=2)+"\n")
