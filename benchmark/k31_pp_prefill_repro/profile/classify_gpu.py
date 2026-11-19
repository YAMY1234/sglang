#!/usr/bin/env python3
"""Per-rank GPU breakdown from one node's nsys SQLite (stdlib only; read-only).

For each CUDA device in the report: a common steady window (trimmed), an exclusive sweep of resident activity into
  classes in CLASSES order (moe, compute, fused TP collectives, NCCL p2p / all-gather / all-reduce, barriers), then
  copy_only (only memcpy/memset resident) and idle (nothing resident).
Earlier classes win, so each ns is counted once and the parts sum to the window. TEP8 MegaMoE fuses dispatch + expert
GEMM + combine into one kernel, so its "moe" includes its EP communication. Also reports inclusive unions per comm class (overlapped + exposed) and per-step values = window part / steps,
where steps = count of an anchor kernel (regex, per device) / anchor occurrences per forward.
usage: classify_gpu.py SQLITE --out OUT.json [--trim-s 0.5] [--anchor REGEX] [--anchor-per-step N] [--names-out F]
       [--timeline-out F --timeline-s 3.0]
"""
import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict

# Ordered (class, regex on lowercase kernel name); first match wins. Sweep priority = list order (earlier wins), so a
# ns where MoE and a collective are both resident counts as MoE. Kernel names checked on VR200 traces of both arms.
CLASSES = [
    ("moe", r"mega_moe|bmm_mxe4m3_mxe2m1|bmm_bfloat16_mxe2m1|finalizekernel|routingindices|mask_topk_ids"),
    ("compute", None),  # every kernel not matched by any other class
    ("tp_fused_comm", r"gemm_reduce_scatter|direct_ag"),  # SP collective fused into GEMM-RS / attn-res all-gather
    ("nccl_p2p", r"nccl.*sendrecv"),
    ("nccl_allgather", r"nccl.*allgather"),
    ("nccl_allreduce", r"nccl.*(allreduce|reducescatter)"),
    ("barrier", r"multimem_barrier|nccl"),
]


def cls(name, comm_re, moe_re):
    n = name.lower()
    for c, r in comm_re:
        if r is not None and r.search(n):
            return c
    return "compute"


def table_exists(db, t):
    return db.execute("select 1 from sqlite_master where name=?", (t,)).fetchone() is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trim-s", type=float, default=0.5)
    ap.add_argument("--anchor", default=None)
    ap.add_argument("--anchor-per-step", type=float, default=1.0)
    ap.add_argument("--names-out", default=None)
    ap.add_argument("--timeline-out", default=None)
    ap.add_argument("--timeline-s", type=float, default=3.0)
    ap.add_argument("--timeline-start-frac", type=float, default=0.4)
    ap.add_argument("--timeline-start-utc-ns", type=int, default=None, help="absolute start (overrides frac)")
    a = ap.parse_args()
    comm_re = [(c, re.compile(r) if r else None) for c, r in CLASSES]
    moe_re = None
    db = sqlite3.connect(f"file:{a.sqlite}?mode=ro", uri=True)
    s = {i: v for i, v in db.execute("select id, value from StringIds")}
    acts = defaultdict(list)  # device -> (start, end, class, name)
    names = defaultdict(Counter)
    pid_of = {}
    for st, en, dev, nm, gpid in db.execute(
            "select start, end, deviceId, coalesce(demangledName, shortName), globalPid from CUPTI_ACTIVITY_KIND_KERNEL"):
        n = s.get(nm, str(nm))
        c = cls(n, comm_re, moe_re)
        acts[dev].append((st, en, c, n))
        names[dev][(c, n[:160])] += 1
        pid_of[dev] = gpid >> 24 & 0xFFFFFF
    for t, kind in (("CUPTI_ACTIVITY_KIND_MEMCPY", "copy_only"), ("CUPTI_ACTIVITY_KIND_MEMSET", "copy_only")):
        if table_exists(db, t):
            for st, en, dev in db.execute(f"select start, end, deviceId from {t}"):
                acts[dev].append((st, en, kind, kind))
    if not acts:
        raise SystemExit("no GPU activity")
    lo = max(min(x[0] for x in v) for v in acts.values()) + int(a.trim_s * 1e9)
    hi = min(max(x[1] for x in v) for v in acts.values()) - int(a.trim_s * 1e9)
    prio = {c: i for i, (c, _) in enumerate(CLASSES)}
    prio["copy_only"] = 50
    out = {"sqlite": a.sqlite, "window_ns": [lo, hi], "window_s": (hi - lo) / 1e9, "devices": {}}
    if table_exists(db, "TARGET_INFO_SESSION_START_TIME"):
        r = db.execute("select * from TARGET_INFO_SESSION_START_TIME").fetchone()
        cols = [c[1] for c in db.execute("pragma table_info(TARGET_INFO_SESSION_START_TIME)")]
        out["session_start"] = dict(zip(cols, r))
    tl_lo = lo + int((hi - lo) * a.timeline_start_frac)
    if a.timeline_start_utc_ns is not None and "session_start" in out:
        ss = out["session_start"]
        tl_lo = a.timeline_start_utc_ns - int(ss["utcEpochNs"])  # trace timestamps are ns since session start
    tl_hi = tl_lo + int(a.timeline_s * 1e9)
    timeline = {}
    anchor = re.compile(a.anchor) if a.anchor else None
    for dev, v in sorted(acts.items()):
        pts, incl = [], defaultdict(list)
        n_anchor = 0
        for st, en, c, n in v:
            st2, en2 = max(st, lo), min(en, hi)
            if anchor and lo <= st < hi and c != "copy_only" and anchor.search(n):
                n_anchor += 1
            if en2 <= st2:
                continue
            pts += [(st2, 1, c), (en2, -1, c)]
            incl[c].append((st2, en2))
        pts.sort()
        active, acc, last, segs = Counter(), Counter(), lo, []
        for t, sg, c in pts:
            if t > last:
                lab = min((k for k, x in active.items() if x), key=prio.get, default="idle")
                acc[lab] += t - last
                if a.timeline_out and t > tl_lo and last < tl_hi:
                    s0, s1 = max(last, tl_lo), min(t, tl_hi)
                    if segs and segs[-1][2] == lab and segs[-1][1] == s0 - tl_lo:
                        segs[-1][1] = s1 - tl_lo
                    else:
                        segs.append([s0 - tl_lo, s1 - tl_lo, lab])
                last = t
            active[c] += sg
        if last < hi:
            acc["idle"] += hi - last
        inc = {}
        for c, iv in incl.items():
            tot, end = 0, None
            for x0, x1 in sorted(iv):
                if end is None or x0 >= end:
                    tot += x1 - x0
                    end = x1
                elif x1 > end:
                    tot += x1 - end
                    end = x1
            inc[c] = tot / 1e6
        steps = n_anchor / a.anchor_per_step if anchor else None
        d = {"pid": pid_of.get(dev), "exclusive_ms": {k: x / 1e6 for k, x in acc.items()},
             "inclusive_union_ms": inc, "anchor_count": n_anchor, "steps": steps}
        if steps:
            d["per_step_ms"] = {k: x / 1e6 / steps for k, x in acc.items()}
            d["step_ms"] = (hi - lo) / 1e6 / steps
        out["devices"][str(dev)] = d
        if a.timeline_out:
            timeline[str(dev)] = segs
    json.dump(out, open(a.out, "w"), indent=1)
    if a.names_out:
        json.dump({str(d): [[c, n, k] for (c, n), k in cnt.most_common(400)] for d, cnt in names.items()},
                  open(a.names_out, "w"), indent=0)
    if a.timeline_out:
        json.dump({"t0_ns": tl_lo, "len_s": a.timeline_s, "session_start": out.get("session_start"), "devices": timeline}, open(a.timeline_out, "w"))
    for dev, d in out["devices"].items():
        print(dev, d["anchor_count"], {k: round(x, 1) for k, x in sorted(d["exclusive_ms"].items())})


if __name__ == "__main__":
    main()
