"""docs/209 §9.9: C32 decode GPU-time budget and closed-loop split of the E2E delta.

Usage: ROOT c32-step.json window-c32.json OUT.json (outputs of the two companion scripts)."""
import json, re, sys, datetime, statistics
from zoneinfo import ZoneInfo
from pathlib import Path
root = Path(sys.argv[1]); tz = ZoneInfo("America/Los_Angeles")
step = json.loads(Path(sys.argv[2]).read_text())
bins = {"1-4": (1, 5), "5-8": (5, 9), "9-16": (9, 17), "17-24": (17, 25), "25-32": (25, 33)}
def binof(b): return next(k for k, (lo, hi) in bins.items() if lo <= b < hi)
res = {}
for arm in ("original-off", "factored"):
    art = root/arm/"C32/rep1/aiperf_artifacts"
    start = json.loads((art/"server_metrics_export.json").read_text())["summary"]["phase_time_ranges"]["profiling"]["start_ns"]
    # decode GPU time from log lines x 40 steps x in-window clean step by batch bin (own arm)
    lines = []
    for line in (root/arm/"server.log").open():
        m = re.match(r"\[(\S+ \S+)\] Decode batch", line)
        if not m: continue
        t = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz).timestamp()
        if start/1e9+1 <= t < start/1e9+899: lines.append(int(re.search(r"#running-req: (\d+)", line).group(1)))
    own = {k: v["step_ms_p50"] for k, v in step[arm].items()}
    orig = {k: v["step_ms_p50"] for k, v in step["original-off"].items()}
    dec_s = sum(40*own[binof(b)] for b in lines)/1000
    dec_at_orig_s = sum(40*orig[binof(b)] for b in lines)/1000
    # per-token extra step cost seen by a request = time-weighted over tokens: weight bins by batch*steps
    w = {k: 0 for k in bins}
    for b in lines: w[binof(b)] += b*40
    tok = sum(w.values())
    dstep_ms = sum(w[k]*(own[k]-orig[k]) for k in bins)/tok
    rows = []
    for line in (art/"profile_export.jsonl").open():
        r = json.loads(line); m = r["metadata"]
        if r.get("error") or m["benchmark_phase"] != "profiling": continue
        if not start <= m["request_end_ns"] <= start+900*10**9: continue
        t = r["metrics"]["time_to_first_token"]["value"]/1e3; e2e = (m["request_end_ns"]-m["request_start_ns"])/1e9
        rows.append(dict(ttft=t, dec=e2e-t, e2e=e2e, O=r["metrics"]["output_sequence_length"]["value"]))
    res[arm] = dict(n=len(rows), cycle_s=32*900/len(rows), ttft_mean=statistics.fmean(r["ttft"] for r in rows),
                    decode_mean=statistics.fmean(r["dec"] for r in rows), e2e_mean=statistics.fmean(r["e2e"] for r in rows),
                    out_mean=statistics.fmean(r["O"] for r in rows), decode_gpu_s=dec_s, decode_gpu_at_orig_step_s=dec_at_orig_s,
                    token_weighted_dstep_ms=dstep_ms)
    print(arm, {k: round(v, 4) for k, v in res[arm].items()})
o, f = res["original-off"], res["factored"]
w = json.loads(Path(sys.argv[3]).read_text())
d = dict(dcycle=f["cycle_s"]-o["cycle_s"], de2e=f["e2e_mean"]-o["e2e_mean"], dttft=f["ttft_mean"]-o["ttft_mean"],
         dwait=w["factored"]["wait_mean"]-w["original-off"]["wait_mean"], dsvc=w["factored"]["svc_mean"]-w["original-off"]["svc_mean"],
         ddecode=f["decode_mean"]-o["decode_mean"], dstep_per_req=f["out_mean"]*f["token_weighted_dstep_ms"]/1000)
d["dstall_and_other"] = d["ddecode"]-d["dstep_per_req"]
d["critical_fraction"] = d["dcycle"]/d["de2e"]
for k in ("dwait", "dsvc", "dstall_and_other", "dstep_per_req"):
    d["share_"+k] = d[k]/d["de2e"]
print({k: round(v, 4) for k, v in d.items()})
Path(sys.argv[4]).write_text(json.dumps(dict(arms=res, delta=d), indent=2)+"\n")
