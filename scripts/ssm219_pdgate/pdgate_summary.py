"""docs/222 §15 PD gate summary: the agg gate's gate-summary.json schema (docs/219 gate_summary.py: kl / gate / gsm, same
statistics and the 8.x bistable block) plus what PD adds: the role split of the scored steps, the activation lines of the
candidate's switches (docs/219 §8 rule 7b: none -> void) and one verdict per candidate.
    python3 pdgate_summary.py <out dir>      -> <out>/gate-summary.json and <out>/summary.json (same content)
"""
import json
import re
import statistics
import sys
from pathlib import Path

INFO = ("DUET controls", "Factored GDN pool allocated", "R3A_PREFILL_MIXED_ACTIVE", "K31_MIXED_EIGH_ANY_BATCH_ACTIVE",
        "K31_FUSED_ACTIVE", "PD full-N deferred publication enabled", "FACTORED_TRUNC_CUDA_ACTIVE",
        "GDN_FACTORED_BOOKKEEPING_FOLD_ACTIVE", "GDN_FACTORED_STEP_VARIANT_ACTIVE", "PREFIX_RESTORE_FUSED_ACTIVE",
        "Flash-Next D boundary")


def records(out, arm, ref):
    by = {}
    for role in ("P", "D"):
        f = out / f"rec-{arm}-score.{role}.jsonl"
        if f.exists():
            for s in f.read_text().splitlines():
                r = json.loads(s)
                by.setdefault(r["window"], {})[r["step"]] = r["kl_" + ref]
    return by


def per_window(by):
    return [statistics.fmean(by[w].values()) for w in sorted(by)]


def mse(x):
    return statistics.fmean(x), statistics.stdev(x) / len(x) ** 0.5


def scan(path, patterns):
    hits = {p: [] for p in patterns}
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            for p in patterns:
                if re.search(p, line):
                    hits[p].append(line.strip()[:240])
    return hits


def main(out):
    out = Path(out)
    plan = json.loads((out / "plan.json").read_text())
    ref, nw, cont = plan["ref"]["name"], plan["windows"]["n"], plan["windows"]["cont"]
    phases = plan["phases"]
    kl, complete = {}, {}
    for arm in phases:
        by = records(out, arm, ref)
        complete[arm] = len(by) == nw and all(len(v) == cont for v in by.values())
        if complete[arm]:
            kl[arm] = per_window(by)
    gsm = {}
    for arm in phases:
        f = out / f"gsm-{arm}.json"
        if f.exists():
            gsm[arm] = json.loads(f.read_text())
    res = dict(kl={a: dict(zip(("mean", "se"), mse(v))) for a, v in kl.items()}, gate={},
               gsm={a: g["correct"] for a, g in gsm.items()}, pd={}, activation={}, lines={}, verdict={},
               reference=plan["ref"], windows=plan["windows"], expectation_notes=plan["expectation_notes"])
    # PD role split (score() already refused anything but one scored record per (window, step) and step 0 on its owner)
    for arm in phases:
        meta = out / f"score-meta-{arm}.json"
        smoke = out / f"smoke-{arm}.json"
        calls = {}
        for role in ("P", "D"):
            f = out / f"calls-{arm}.{role}.jsonl"
            if f.exists():
                for s in f.read_text().splitlines():
                    c = json.loads(s)
                    key = f"{role}:{'step0' if c['step'] == 0 else 'decode'}:{'scored' if c['scored'] else 'forced-only'}"
                    calls[key] = calls.get(key, 0) + 1
        m = json.loads(meta.read_text()) if meta.exists() else None
        res["pd"][arm] = dict(step0_owner=plan["arms"][arm]["step0_owner"], complete=complete[arm], calls=calls,
                              by_role=m and m["by_role"], step0_by_role=m and m["step0_by_role"],
                              P_cached_max=m and max(v["P_cached"] or 0 for v in m["windows"].values()),
                              smoke=json.loads(smoke.read_text())["D_meta"]["completion_tokens"] if smoke.exists() else None)
        res["lines"][arm] = {role: {p: dict(n=len(h), first=h[0] if h else None)
                                    for p, h in scan(out / f"srv-{arm}-{role}.log", INFO).items()} for role in ("P", "D")}
    # activation: every expected line on the candidate's role; lines marked forbid_in_before absent from `before`
    for arm, roles in plan["expectations"].items():
        act = res["activation"].setdefault(arm, {"ok": True, "items": []})
        for role, items in roles.items():
            got = scan(out / f"srv-{arm}-{role}.log", [it["regex"] for it in items])
            bef = scan(out / f"srv-before-{role}.log", [it["regex"] for it in items if it.get("forbid_in_before")])
            for it in items:
                n, nb = len(got[it["regex"]]), len(bef.get(it["regex"], []))
                ok = n >= 1 and (not it.get("forbid_in_before") or nb == 0)
                act["items"].append(dict(role=role, key=it["key"], kind=it["kind"], regex=it["regex"], n=n, n_before=nb,
                                         first=got[it["regex"]][0] if n else None, ok=ok))
                act["ok"] &= ok
    for c in [a for a in phases if a != "before"]:
        g = {}
        if c in kl and "before" in kl:
            d = [x - y for x, y in zip(kl[c], kl["before"])]
            m, se = mse(d)
            order = sorted(range(len(d)), key=lambda i: -abs(d[i]))
            rest = [d[i] for i in range(len(d)) if i != order[0]]
            m1, se1 = mse(rest)
            bistable = dict(median=statistics.median(d), top3_windows=order[:3],
                            top3_share=sum(d[i] for i in order[:3]) / sum(d) if sum(d) else None,
                            excl_window=order[0], excl_mean=m1, excl_se=se1, excl_z=m1 / se1 if se1 else None)
            g.update(kl_minus_before=m, se=se, z=m / se if se else None, kl_pass=m <= 2 * se, bistable=bistable)
        if c in gsm and "before" in gsm:
            dd = [int(p["correct"]) - int(q["correct"]) for p, q in zip(gsm[c]["records"], gsm["before"]["records"])]
            gm, gse = mse(dd)
            g.update(gsm_diff_pp=100 * gm, gsm_se_pp=100 * gse, gsm_pass=gm >= -2 * gse)
        res["gate"][c] = g
        act_ok = res["activation"].get(c, {"ok": True})["ok"]
        if "kl_pass" not in g or "gsm_pass" not in g or not complete.get("before") or not complete.get(c):
            v = "INCOMPLETE"
        elif not act_ok:
            v = "VOID (activation line missing or present in before)"
        else:
            v = "PASS" if g["kl_pass"] and g["gsm_pass"] else "FAIL"
        res["verdict"][c] = v
    text = json.dumps(res, indent=1)
    (out / "gate-summary.json").write_text(text)
    (out / "summary.json").write_text(text)
    print(text)
    print("VERDICT " + "; ".join(f"{c}: {v} (KL {res['gate'][c].get('kl_minus_before', float('nan')):+.5f} "
                                 f"+- {res['gate'][c].get('se', float('nan')):.5f}, z {res['gate'][c].get('z') or float('nan'):+.2f}; "
                                 f"GSM {res['gsm'].get(c)} vs {res['gsm'].get('before')})"
                                 for c, v in res["verdict"].items()))


if __name__ == "__main__":
    main(sys.argv[1])
