"""Summarize prefill benchmark runs (run_arm.sh output in RESULTS/<run>/). For each measured phase (closed / poisson):
client new tok/s, hit, latency p50/p90 (pfab_client summary), and the same window from the server's rank-0 Prefill batch
lines (PP0 for PP8, TP0 for TEP8): new tok/s, hit, share of lines with #queue-req >= 1, lines. Then the server_args
diff between two runs' first steps.  usage: pfab_parse.py RESULTS_DIR RUN [RUN ...] [--args-diff RUN_A RUN_B]
"""
import datetime as dt
import glob
import json
import os
import re
import sys

LINE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?: ([^\]]*))?\] Prefill batch.*?#new-token: (\d+), #cached-token: (\d+)"
                  r".*?#queue-req: (\d+)")


def leader_lines(step):
    rows = []
    for f in sorted(glob.glob(f"{step}/prefill-*.log")):
        for line in open(f, errors="replace"):
            m = LINE.match(line)
            if m:
                rows.append((m.group(1), (m.group(2) or "").strip(), int(m.group(3)), int(m.group(4)), int(m.group(5))))
    tags = {}
    for r in rows:
        if re.search(r"\b(PP0|TP0)\b", r[1]) or r[1] == "":
            tags[r[1]] = tags.get(r[1], 0) + 1
    tag = max(tags, key=tags.get) if tags else None
    return tag, [r for r in rows if r[1] == tag]


def phases(step):
    out = {}
    for line in open(f"{step}/phases.tsv") if os.path.exists(f"{step}/phases.tsv") else []:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 4 and p[1] == "start":
            out[p[0]] = dt.datetime.strptime(p[3], "%Y-%m-%d %H:%M:%S")
    return out


def summarize(run_dir):
    res = []
    for step in sorted(glob.glob(f"{run_dir}/s*"), key=lambda s: int(s.rsplit("s", 1)[1])):
        variant = open(f"{step}/START").read().split()[0] if os.path.exists(f"{step}/START") else "?"
        boot = dict(l.rstrip("\n").split("\t") for l in open(f"{step}/boot.tsv")) if os.path.exists(f"{step}/boot.tsv") else {}
        tag, rows = leader_lines(step)
        starts = phases(step)
        for sf in sorted(glob.glob(f"{step}/*.jsonl.sum.json")):
            s = json.load(open(sf))
            if s["mode"] not in ("closed", "poisson"):
                continue
            ph = os.path.basename(sf)[: -len(".jsonl.sum.json")]
            a = s["args"]
            w0 = starts.get(ph)
            srv = {}
            if w0:
                lo, hi = w0 + dt.timedelta(seconds=a["lead"]), w0 + dt.timedelta(seconds=a["lead"] + a["window"])
                win = [r for r in rows if lo <= dt.datetime.strptime(r[0], "%Y-%m-%d %H:%M:%S") < hi]
                n, c = sum(r[2] for r in win), sum(r[3] for r in win)
                srv = {"srv_new_tok_s": n / a["window"], "srv_hit": c / (n + c) if n + c else None,
                       "queue_ge1": sum(r[4] >= 1 for r in win) / len(win) if win else None, "lines": len(win)}
            res.append({"run": os.path.basename(run_dir), "step": os.path.basename(step), "variant": variant,
                        "boot_s": boot.get("boot_s"), "phase": ph, "mode": s["mode"], "conc": a.get("concurrency"),
                        "rate": a.get("rate"), "window": a["window"], "new_tok_s": s["new_tok_s"], "hit": s["hit"],
                        "failed": s["failed"], "done": s["done_in_window"], "lat_p50": s["lat_p50"], "lat_p90": s["lat_p90"],
                        "tag": tag, **srv})
    return res


def server_args(run_dir):
    for f in sorted(glob.glob(f"{run_dir}/s1/server-key-*.log")):
        for line in open(f, errors="replace"):
            m = re.search(r"server_args=ServerArgs\((.*)\)\s*$", line)
            if m:
                return dict(re.findall(r"(\w+)=('[^']*'|\[[^\]]*\]|\{[^}]*\}|[^,]+)", m.group(1)))
    return {}


if __name__ == "__main__":
    argv = sys.argv[1:]
    diff = None
    if "--args-diff" in argv:
        i = argv.index("--args-diff")
        diff, argv = argv[i + 1:i + 3], argv[:i]
    results, runs = argv[0], argv[1:]
    for run in runs:
        for r in summarize(f"{results}/{run}"):
            print(json.dumps(r))
    if diff:
        a, b = server_args(f"{results}/{diff[0]}"), server_args(f"{results}/{diff[1]}")
        keys = sorted(set(a) | set(b))
        print(f"== server_args: {len(keys)} fields, {sum(a.get(k) == b.get(k) for k in keys)} identical")
        for k in keys:
            if a.get(k) != b.get(k):
                print(f"  {k:48s} {diff[0]}={str(a.get(k))[:40]:40s} {diff[1]}={str(b.get(k))[:40]}")
