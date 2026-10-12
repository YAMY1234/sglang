"""Verdict of a served-vs-reference parity run (step 3 of validate_fn_parity_job.sh and validate_hooked_parity_job.sh).

    python -m benchmark.duet_eval.parity_verdict <out dir> [--manifest M --tokens T] [--write FILE]

Reads <out>/ref/reference.json (two passes of the reference DUET over the guard windows), <out>/ref/reference-base.json (one pass of
the reference base) and <out>/nll-<duet|stock>-<seq|conc16>.json (the served arms scored by forced_nll.py, one request at a time and
16 in flight); writes <out>/verdict.json (or --write FILE) and prints the table and "PARITY PASS" / "PARITY FAIL".

Gate, per serving mode (seq, conc16): the served DUET - served base difference against the reference DUET - base difference.
Per window i, d_i = (served DUET_i - served base_i) - (reference DUET_i, mean of the two passes, - reference base_i).
  PASS (absolute)     |served mean difference - reference mean difference| <= threshold for each reference DUET pass, threshold =
                      max(0.002, reference noise), reference noise = |DUET pass 1 - DUET pass 2| (mean NLL, nats);
  PASS (statistical)  otherwise, the mean of d_i within 2 SE of zero (SE over the windows).
The difference cancels what moves both served arms of one job together: the served path's run-to-run offset is ~0.002 nats on these
windows (twinstar docs/66, FN-BF16 v3.5 parity: the unchanged base arm missed the old gate in a rerun by as much as the DUET arm
in the first run), the size of the threshold.  The per-arm numbers (each served arm against its own reference: within the
threshold -- the base arm within 0.002 -- or within 2 SE) are computed, printed and written beside the gate as information
(verdict.json "duet-seq", ..., "per_arm_rule"; "per_arm_rule_passed").  The verdict reads only the score files, so it can be
re-run on an earlier run's directory (--write elsewhere keeps that run's verdict.json).

--manifest / --tokens (Flash-Next): first check that the served windows (forced_nll.windows of the token stream) are the guard
manifest's cases, the windows the reference scored.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

MODES = ("seq", "conc16")
FLOOR = 0.002          # nats: the guard's nll_decision threshold (the reference noise raises it)


def _mean_se(values):
    return st.mean(values), st.stdev(values) / len(values) ** 0.5


def _label(absolute, statistical):
    return "PASS (absolute)" if absolute else ("PASS (statistical)" if statistical else "FAIL")


def load(out):
    out = Path(out)
    ref = json.loads((out / "ref/reference.json").read_text())["passes"]
    base = json.loads((out / "ref/reference-base.json").read_text())["passes"][0]
    served = {(arm, mode): json.loads((out / f"nll-{arm}-{mode}.json").read_text()) for arm in ("duet", "stock") for mode in MODES}
    return ref, base, served


def verdict(ref, base, served):
    """ref: the reference DUET passes, base: the reference base pass ({"mean_nll", "per_window"}); served[(arm, mode)]: forced_nll
    output ({"mean_nll", "per_window_nll"}).  Returns the verdict dict (module docstring)."""
    if len(ref) != 2:
        raise ValueError(f"expected two reference DUET passes, got {len(ref)}")
    n = len(base["per_window"])
    if any(len(p["per_window"]) != n for p in ref) or any(len(s["per_window_nll"]) != n for s in served.values()):
        raise ValueError("the reference and served score files cover different numbers of windows")
    noise = abs(ref[0]["mean_nll"] - ref[1]["mean_nll"])
    thr = max(FLOOR, noise)
    out = {"gate": "served DUET - served base against reference DUET - base, per serving mode",
           "reference_noise": noise, "threshold": thr, "reference_duet": [p["mean_nll"] for p in ref],
           "reference_base": base["mean_nll"], "reference_difference": [p["mean_nll"] - base["mean_nll"] for p in ref]}
    # per arm: information, not part of the gate
    per_arm_ok = True
    for arm, refs in (("duet", ref), ("stock", [base])):
        for mode in MODES:
            s = served[(arm, mode)]
            d = [abs(s["mean_nll"] - r["mean_nll"]) for r in refs]
            w = max(abs(a - b) for r in refs for a, b in zip(s["per_window_nll"], r["per_window"]))
            paired = [a - st.mean(r["per_window"][i] for r in refs) for i, a in enumerate(s["per_window_nll"])]
            m, se = _mean_se(paired)
            absolute = max(d) <= (thr if arm == "duet" else FLOOR)
            statistical = abs(m) <= 2 * se
            per_arm_ok &= absolute or statistical
            out[f"{arm}-{mode}"] = {"served": s["mean_nll"], "abs_delta": d, "max_window_abs_delta": w, "paired_mean": m,
                                    "paired_se": se, "absolute": absolute, "statistical": statistical,
                                    "per_arm_rule": _label(absolute, statistical)}
    out["per_arm_rule_passed"] = per_arm_ok
    # the gate: the served difference against the reference difference
    ref_window = [st.mean(p["per_window"][i] for p in ref) - base["per_window"][i] for i in range(n)]
    ok = True
    for mode in MODES:
        sd, sb = served[("duet", mode)], served[("stock", mode)]
        served_diff = sd["mean_nll"] - sb["mean_nll"]
        d = [abs(served_diff - r) for r in out["reference_difference"]]
        dd = [a - b - r for a, b, r in zip(sd["per_window_nll"], sb["per_window_nll"], ref_window)]
        m, se = _mean_se(dd)
        absolute = max(d) <= thr
        statistical = abs(m) <= 2 * se
        ok &= absolute or statistical
        out[f"difference-{mode}"] = {"served": served_diff, "reference": out["reference_difference"], "abs_delta": d,
                                     "max_window_abs_delta": max(abs(x) for x in dd), "paired_mean": m, "paired_se": se,
                                     "absolute": absolute, "statistical": statistical, "passed": absolute or statistical,
                                     "verdict": _label(absolute, statistical)}
    out["passed"] = ok
    return out


def report(v):
    lines = ["per arm, served vs reference (information, not the gate):"]
    for arm in ("duet", "stock"):
        for mode in MODES:
            r = v[f"{arm}-{mode}"]
            against = " / ".join("%.6f" % x for x in (v["reference_duet"] if arm == "duet" else [v["reference_base"]]))
            lines.append(f"  {arm:5s} {mode:6s} served {r['served']:.6f}  vs reference {against}  |d| {max(r['abs_delta']):.6f}  "
                         f"paired {r['paired_mean']:+.6f} +- {r['paired_se']:.6f}  max window |d| {r['max_window_abs_delta']:.4f}  "
                         f"[{r['per_arm_rule']}]")
    lines.append("gate, served DUET - served base vs reference DUET - base:")
    for mode in MODES:
        r = v[f"difference-{mode}"]
        against = " / ".join("%+.6f" % x for x in r["reference"])
        lines.append(f"  {mode:6s} served {r['served']:+.6f}  vs reference {against}  |d| {max(r['abs_delta']):.6f}  "
                     f"paired {r['paired_mean']:+.6f} +- {r['paired_se']:.6f}  max window |d| {r['max_window_abs_delta']:.4f}  "
                     f"{r['verdict']}")
    lines.append(f"reference DUET passes {v['reference_duet'][0]:.6f} / {v['reference_duet'][1]:.6f} (noise {v['reference_noise']:.6f}, "
                 f"threshold {v['threshold']:.6f}); reference base {v['reference_base']:.6f}")
    lines.append("PARITY " + ("PASS" if v["passed"] else "FAIL"))
    return "\n".join(lines)


def check_windows(manifest, tokens, n=32, prefix=3584, cont=256):
    """The served windows (forced_nll.windows) are the guard manifest's first n cases."""
    from benchmark.duet_eval.forced_nll import windows
    cases = json.loads(Path(manifest).read_text())["cases"][:n]
    x = windows(tokens)
    for i, c in enumerate(cases):
        if list(map(int, x[i][:prefix])) != c["input_ids"] or list(map(int, x[i][prefix:prefix + cont])) != c["targets"]:
            raise SystemExit(f"window {i}: the served windows are not the manifest's ({manifest})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="the parity job's output directory")
    ap.add_argument("--manifest", default=None, help="Flash-Next: the guard's admission manifest (checked with --tokens)")
    ap.add_argument("--tokens", default=None, help="Flash-Next: the eval token stream the served windows were drawn from")
    ap.add_argument("--write", type=Path, default=None, help="verdict file (default <out>/verdict.json)")
    args = ap.parse_args(argv)
    if bool(args.manifest) != bool(args.tokens):
        ap.error("--manifest and --tokens go together")
    if args.manifest:
        check_windows(args.manifest, args.tokens)
    v = verdict(*load(args.out))
    print(report(v), flush=True)
    path = args.write or args.out / "verdict.json"
    path.write_text(json.dumps(v, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
