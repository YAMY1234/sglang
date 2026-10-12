"""Paired report over lm-eval sample logs of two arms served identically (run.py, one rep per process).

    python -m benchmark.duet_eval.lmeval.report_lmeval --arms stock=OUT/stock duet=OUT/duet [--md] [--out report.json]

Cells: gpqa_se and mmlu_se (OpenAI simple-evals prompt and extraction, se_utils.py; mmlu_se/<subject> with --all-cells), aime24,
aime25, aime (both years), math500, ifeval (prompt-level strict, lm-eval's headline metric; prompt-level loose beside it -- the
instruction-level metrics are per-instruction lists in the sample logs and are not reported), lcb (pass@1, per difficulty),
humaneval_plus (pass@1 on the base + plus tests) and humaneval (base tests) from humaneval_grade.py's verdicts, longbench_v2
(accuracy; longbench_v2/<easy|hard|short|medium|long> as the official leaderboard breaks it down),
ruler@<length> (mean over the RULER tasks of the run: 13 with the job's default RULER_TASKS=all, the 8 NIAH tasks with
RULER_TASKS=retrieval; *.anymatch beside it),
aa_lcr (the verdicts of the --aa-lcr-judge judge, aa_lcr_judge.py) with aa_lcr/judge-<tag> per judge.  Runs before 2026-10-04 also
hold lm-eval's gpqa_diamond / mmlu tasks; those cells were retired with the code that read them (fork a81c871a34 still has it).
Repetitions: the protocol's first N per exam (tasks.EXAMS reps; rep 0..N-1, pre-registered): a run's extra repetitions are
dropped and the report prints, per arm and exam, the repetitions read, dropped and missing; --reps overrides it.
Score per (task, doc_id, rep) -- simple-evals docs by their se_id; the paired delta averages the per-question difference
of the per-question means over the repetitions common to both arms, SE = sd / sqrt(questions).  Each sample line is joined with
its sidecar record (model.py) on the sha1 of the templated prompt and of the reply; the record gives truncation, token counts and
the sampling seed.  Checks over the paired samples: the same prompt (lm-eval prompt_hash) and the same sampling seed on both arms
(a seed depends on the pair seed, the repetition, the prompt and its occurrence count in the lm-eval process)."""
import argparse
import collections
import glob
import json
import math
import os
import re
import statistics

from benchmark.duet_eval.lmeval import aa_lcr_judge, tasks
from benchmark.duet_eval.lmeval.common import first_string, rep_dirs, response_of, sample_files, sha1

SE_TASKS = tuple(tasks.EXAMS[e]["static"]["task"] for e in tasks.SE_EXAMS)    # paired on the doc's se_id
RETIRED = ("gpqa_diamond", "mmlu_flan_cot_zeroshot")   # lm-eval's GPQA / MMLU in runs before 2026-10-04: not scored (module docstring)
PER_REP_PROMPTS = ("duet_gpqa_se",)                 # simple-evals permutes GPQA's options per repetition: one prompt per (doc, rep)


def ref_key(task, doc, rep):
    """Key of a prompt in a --write-prompts / --check-prompts reference."""
    return f"{doc}@{rep}" if task in PER_REP_PROMPTS else str(doc)


def cells_of(task, line):
    """(cell, metric) pairs this sample contributes to; the first is the primary metric of the task."""
    base = task[len("duet_"):] if task.startswith("duet_") else task
    if base.startswith(RETIRED):
        return []
    if base in ("aime24", "aime25"):
        return [(base, "exact_match"), ("aime", "exact_match")]
    if base == "math500":
        return [("math500", "math_verify"), ("math500/strict_em", "exact_match")]
    if base == "lcb":
        return [("lcb", "pass@1"), (f"lcb/{line.get('doc', {}).get('difficulty', '?')}", "pass@1")]
    if base == "ifeval":
        return [("ifeval", "prompt_level_strict_acc"), ("ifeval/prompt_loose", "prompt_level_loose_acc")]
    if base == "aa_lcr":
        return [("aa_lcr/answered", "answered")]          # the score: the judge cells added in load_arm
    if base == "humaneval_plus":                          # the verdicts humaneval_grade.py wrote into the line (none before grading)
        return [("humaneval_plus", "humaneval_plus"), ("humaneval", "humaneval")]
    if base == "longbench_v2":                            # the official breakdown: difficulty and length class of the question
        doc = line.get("doc", {})
        return [("longbench_v2", "accuracy"), (f"longbench_v2/{doc.get('difficulty', '?')}", "accuracy"),
                (f"longbench_v2/{doc.get('length', '?')}", "accuracy")]
    if base == "gpqa_se":
        return [("gpqa_se", "accuracy")]
    if base == "mmlu_se":
        return [("mmlu_se", "accuracy"), (f"mmlu_se/{line.get('doc', {}).get('Subject', '?')}", "accuracy")]
    ruler = base[len("ruler_"):] if base.startswith("ruler_") else base      # installed names: niah_*, ruler_vt, ruler_cwe, ...
    if ruler in tasks.RULER_TASKS:
        lengths = [k for k, v in line.items() if k.isdigit() and v != -1]
        family = "ruler_niah" if ruler.startswith("niah") else "ruler_other"       # retrieval (8 NIAH tasks) vs cwe/fwe/vt/qa
        cells = [(f"{family}@{L}", L) for L in lengths] + [(f"ruler@{L}", L) for L in lengths] + [(f"ruler/{ruler}@{L}", L) for L in lengths]
        # *.anymatch: the QA tasks scored as RULER itself scores them (a reply is right if it contains ANY of the reference
        # answers); lm-eval's string_match_part scores the share of the references it contains, so of SQuAD's three annotator
        # answers "Seljuk Turks" earns 1/3 and "the Seljuk Turks" 2/3.  The other tasks keep their official score in these cells.
        qa = ruler.startswith("qa_")
        if qa:
            reply = re.sub(r"[\x00-\x1f]", "\n", response_of(line).strip()).strip().lower()
            line["anymatch"] = float(any(r.lower() in reply for r in line["doc"]["outputs"]))
            cells += [(f"ruler/{ruler}.anymatch@{L}", "anymatch") for L in lengths]
        cells += [(f"{family}.anymatch@{L}", "anymatch" if qa else L) for L in lengths]
        cells += [(f"ruler.anymatch@{L}", "anymatch" if qa else L) for L in lengths]
        return cells
    metric = next((m for m in line.get("metrics", []) if isinstance(line.get(m), (int, float))), None)
    return [(base, metric)] if metric else []


def rep_caps(spec):
    """{exam: repetitions the report reads (None = every one)}: tasks.EXAMS reps, overridden by 'exam=N,...' or 'all'."""
    caps = {exam: spec_["reps"] for exam, spec_ in tasks.EXAMS.items()}
    if spec == "all":
        return dict.fromkeys(caps)
    for item in filter(None, (spec or "").split(",")):
        exam, _, n = item.partition("=")
        if exam not in caps or not n.isdigit() or int(n) < 1:
            raise SystemExit(f"--reps {item!r}: expected <exam>=<N >= 1> with exam in {sorted(caps)}, or 'all'")
        caps[exam] = int(n)
    return caps


def match_sidecar(recs, taken, line):
    """Index of the sidecar record of this sample line among `recs` (the records of its prompt), or None.  The record whose
    reply is the line's (answer_sha1) is the line's own even when a process sent the same prompt more than once; sidecars
    without answer_sha1 fall back to the records in order.  `taken`: indices already given to lines of the same filter."""
    answer = sha1(response_of(line))
    free = [i for i in range(len(recs)) if i not in taken]
    for i in free:
        if recs[i].get("answer_sha1") == answer:
            return i
    return free[0] if free else None


def sample_key(task, line, rep):
    """(task@lengths, doc key, rep): the pairing key of a sample line.  simple-evals docs by se_id (lm-eval renumbers doc_id
    inside a shard); RULER lines carry their context length."""
    lens = [k for k, v in line.items() if k.isdigit() and v != -1]
    doc_key = line["doc"]["se_id"] if task in SE_TASKS else line["doc_id"]
    return task + "".join(f"@{L}" for L in lens), doc_key, rep


def read_sidecars(rep_dir):
    """prompt sha1 -> sidecar records of one repetition: rep<k>/<kind>/sidecar.jsonl (one per process) and rep<k>/sidecar.jsonl
    (runs before 2026-10-02)."""
    side = collections.defaultdict(list)
    for sc in [os.path.join(rep_dir, "sidecar.jsonl")] + sorted(glob.glob(os.path.join(rep_dir, "*", "sidecar.jsonl"))):
        if not os.path.exists(sc):
            continue
        with open(sc, encoding="utf-8") as fh:
            for raw in fh:
                r = json.loads(raw)
                side[r.get("prompt_sha1")].append(r)
    return side


def load_arm(root, aa_lcr_primary="fn", caps=None):
    """One arm directory -> dict:
      scores[cell][(task, doc_key, rep)] = value; aux[cell] = arm-level lists (sidecar flags, token counts);
      prompts / seeds[(task@lengths, doc_key, rep)] = lm-eval's prompt_hash / (sampling seed, occurrence) of the sidecar record;
      reps[exam] = {"used": set, "dropped": set} -- with caps, repetitions k >= caps[exam] are dropped (first N per exam).
    AA-LCR: aa_lcr/judge-<tag> per judge file of a rep (aa_lcr_judge.py), and aa_lcr = the judge `aa_lcr_primary`."""
    caps = caps or {}
    scores = collections.defaultdict(dict)
    prompts, seeds = {}, {}
    aux = collections.defaultdict(lambda: {"length_truncated": [], "no_think_end": [], "completion_tokens": [], "unmatched": 0,
                                           "no_answer": []})
    reps = collections.defaultdict(lambda: {"used": set(), "dropped": set()})
    ungraded = collections.Counter()                     # exam -> sample lines its external grader has not scored yet
    for rep, rep_dir in rep_dirs(root):
        side = read_sidecars(rep_dir)
        taken = collections.defaultdict(set)
        lcr = {tag: aa_lcr_judge.load_judged(rep_dir, tag) for tag in aa_lcr_judge.judge_tags(rep_dir)}
        for task, path in sample_files(rep_dir):
            exam = tasks.exam_of_task(task)
            if exam is not None and caps.get(exam) is not None and rep >= caps[exam]:
                reps[exam]["dropped"].add(rep)
                continue
            with open(path, encoding="utf-8") as fh:
                for raw in fh:
                    line = json.loads(raw)
                    pairs = cells_of(task, line)
                    if not pairs:
                        continue
                    if task == "duet_humaneval_plus" and "humaneval_plus" not in line:
                        ungraded["humaneval_plus"] += 1
                    pair_key = sample_key(task, line, rep)
                    doc_key = pair_key[1]
                    prompts[pair_key] = line.get("prompt_hash")
                    if task == "duet_aa_lcr":
                        for tag, verdicts in lcr.items():
                            rec_j = verdicts.get((str(line["doc_id"]), sha1(response_of(line))))
                            if rec_j is not None:        # an unjudged reply has no score (the cell then covers fewer samples)
                                line = dict(line, **{f"judge_{tag}": float(rec_j["verdict"] == "CORRECT")})
                                pairs = pairs + [(f"aa_lcr/judge-{tag}", f"judge_{tag}")] + ([("aa_lcr", f"judge_{tag}")] if tag == aa_lcr_primary else [])
                    prompt = first_string(line.get("arguments"))
                    key = sha1(prompt) if prompt is not None else None
                    recs = side.get(key, [])
                    slot = (key, line.get("filter"))                 # one sample line per filter, one sidecar record per response
                    i = match_sidecar(recs, taken[slot], line)
                    rec = None if i is None else recs[i]
                    if rec is not None:
                        taken[slot].add(i)
                        seeds[pair_key] = (rec.get("seed"), rec.get("occurrence"))
                    filtered = line.get("filtered_resps") or [None]
                    no_answer = (filtered[0] == "[invalid]" or line.get("no_boxed") == 1 or line.get("code_extracted") == 0   # no answer / no code
                                 or line.get("no_answer") == 1)
                    for cell, metric in pairs:
                        value = line.get(metric)
                        if not isinstance(value, (int, float)):
                            continue
                        scores[cell][(task, doc_key, rep)] = float(value)
                        reps[exam or task]["used"].add(rep)      # a repetition counts as read only once a sample of it is scored
                        a = aux[cell]
                        a["no_answer"].append(bool(no_answer))
                        if rec is None:
                            a["unmatched"] += 1
                        else:
                            a["length_truncated"].append(bool(rec.get("length_truncated")))
                            a["no_think_end"].append(bool(rec.get("no_think_end")))
                            if rec.get("completion_tokens") is not None:
                                a["completion_tokens"].append(rec["completion_tokens"])
    return {"scores": scores, "aux": aux, "prompts": prompts, "seeds": seeds, "reps": reps, "ungraded": ungraded}


def merge_arm(into, extra):
    """Merge a load_arm result (the arm's RULER-only runs) into the arm's."""
    into["prompts"].update(extra["prompts"])
    into["seeds"].update(extra["seeds"])
    for exam, r in extra["reps"].items():
        into["reps"][exam]["used"] |= r["used"]
        into["reps"][exam]["dropped"] |= r["dropped"]
    for cell in extra["scores"]:
        into["scores"][cell].update(extra["scores"][cell])
        for k, v in extra["aux"][cell].items():
            if isinstance(v, list):
                into["aux"][cell][k].extend(v)
            else:
                into["aux"][cell][k] += v


def ranges(reps):
    """{0, 1, 2, 5} -> '0-2,5'."""
    out, run = [], []
    for k in sorted(reps):
        if run and k != run[-1] + 1:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
            run = []
        run.append(k)
    if run:
        out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(out) or "none"


def seed_check(base, other):
    """Paired samples whose sidecar records carry different sampling seeds (the arms then drew different random streams)."""
    common = set(base["seeds"]) & set(other["seeds"])
    bad = sorted((k for k in common if base["seeds"][k][0] != other["seeds"][k][0]), key=str)
    no_record = len((set(base["prompts"]) & set(other["prompts"])) - common)
    return {"paired_with_records": len(common), "seed_mismatch": len(bad), "paired_without_record": no_record,
            "examples": [{"sample": list(map(str, k)), "base": base["seeds"][k], "other": other["seeds"][k]} for k in bad[:5]]}


def paired(base, other):
    common = sorted(set(base) & set(other))
    by_doc = collections.defaultdict(list)
    for k in common:
        by_doc[k[:2]].append(other[k] - base[k])
    diffs = [statistics.mean(v) for v in by_doc.values()]
    if not diffs:
        return {"n_docs": 0, "n_samples": 0, "delta": None, "se": None}
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    return {"n_docs": len(diffs), "n_samples": len(common), "delta": 100 * statistics.mean(diffs),
            "se": 100 * sd / math.sqrt(len(diffs)), "reps_per_doc": len(common) / len(diffs),
            "base_common": 100 * statistics.mean(base[k] for k in common), "other_common": 100 * statistics.mean(other[k] for k in common)}


def pct(flags):
    return 100 * sum(flags) / len(flags) if flags else None


def fmt(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True, help="name=dir; the first is the baseline")
    ap.add_argument("--ruler", nargs="*", default=[], help="name=dir of the RULER-only runs (same arm names), merged into the arms")
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--aa-lcr-judge", default="fn", help="the AA-LCR judge whose verdicts are the cell aa_lcr (aa_lcr_judge.py --tag)")
    ap.add_argument("--all-cells", action="store_true", help="also list per-subject / per-RULER-task cells")
    ap.add_argument("--out", default=None, help="json with every number")
    ap.add_argument("--write-prompts", default=None, help="write {task: {doc_id: prompt_hash}} of the baseline arm (a reference for reruns)")
    ap.add_argument("--check-prompts", default=None, help="reference written by --write-prompts: every arm's prompts must match it")
    ap.add_argument("--reps", default=None,
                    help="repetitions read per exam: default the protocol's (tasks.EXAMS reps; a run's extra repetitions are dropped), "
                         "'exam=N,...' overrides those exams, 'all' reads every repetition")
    args = ap.parse_args()
    caps = rep_caps(args.reps)
    arms = [a.split("=", 1) for a in args.arms]
    data = {name: load_arm(path, args.aa_lcr_judge, caps) for name, path in arms}
    for extra in args.ruler:
        name, path = extra.split("=", 1)
        if not os.path.isdir(path):
            print(f"WARNING: --ruler {extra}: no such directory, not merged", flush=True)
        else:
            merge_arm(data[name], load_arm(path, args.aa_lcr_judge, caps))
    base_name = arms[0][0]
    for name, d in data.items():
        for exam, n in d["ungraded"].items():
            print(f"WARNING: {name}: {n} {exam} samples are not graded yet (humaneval_grade.py over both arms) -- not in the table",
                  flush=True)
    rep_record = {name: {exam: {"cap": caps.get(exam), "used": sorted(r["used"]), "dropped": sorted(r["dropped"]),
                                "missing": sorted(set(range(caps.get(exam) or 0)) - r["used"])}
                         for exam, r in sorted(d["reps"].items())} for name, d in data.items()}
    for name, by_exam in rep_record.items():
        print(f"repetitions read, {name} (first N per exam; --reps overrides): " + "; ".join(
            f"{exam} {ranges(r['used'])}" + (f" (dropped {ranges(r['dropped'])}: beyond {r['cap']})" if r["dropped"] else "")
            + (f" (missing {ranges(r['missing'])} of the first {r['cap']})" if r["missing"] else "")
            for exam, r in by_exam.items()), flush=True)
    if args.write_prompts:
        ref = collections.defaultdict(dict)
        for (task, doc, rep), h in data[base_name]["prompts"].items():
            ref[task][ref_key(task, doc, rep)] = h
        with open(args.write_prompts, "w", encoding="utf-8") as fh:
            json.dump(ref, fh, indent=0, sort_keys=True)
        print(f"wrote {sum(len(v) for v in ref.values())} prompt hashes of {len(ref)} tasks to {args.write_prompts}", flush=True)
    if args.check_prompts:
        ref = json.load(open(args.check_prompts))
        for name, d in data.items():
            keyed = [(task, ref_key(task, doc, rep), h) for (task, doc, rep), h in d["prompts"].items()]     # one entry per sample
            seen = [k for k in keyed if k[1] in ref.get(k[0], {})]
            bad = [k for k in seen if ref[k[0]][k[1]] != k[2]]
            missing = len({k[:2] for k in keyed} - {(t, d) for t in ref for d in ref[t]})
            print(f"prompt reference check, {name}: {len(seen)} samples of reference questions, {len(bad)} with a different prompt, "
                  f"{missing} questions not in the reference" + (f"; e.g. {bad[:3]}" if bad else ""), flush=True)
    checks, seed_checks = {}, {}
    for name, d in data.items():           # paired samples must be the same prompt and seed (cross-job ablation pairing)
        if name == base_name:
            continue
        ref, prompts = data[base_name]["prompts"], d["prompts"]
        common = set(ref) & set(prompts)
        bad = sorted(k for k in common if ref[k] != prompts[k])
        checks[name] = {"common_samples": len(common), "prompt_hash_mismatch": len(bad), "examples": [list(map(str, k)) for k in bad[:5]]}
        if bad:
            print(f"WARNING: {name} vs {base_name}: {len(bad)} of {len(common)} paired samples have a different prompt "
                  f"(lm-eval prompt_hash), e.g. {bad[:3]} -- these arms are not comparable", flush=True)
        sc = seed_checks[name] = seed_check(data[base_name], d)
        print(f"seed check, {name} vs {base_name}: {sc['paired_with_records']} paired samples with a sidecar record on both arms, "
              f"{sc['seed_mismatch']} with different sampling seeds, {sc['paired_without_record']} without a record on an arm", flush=True)
        if sc["seed_mismatch"]:
            print(f"WARNING: {name} vs {base_name}: {sc['seed_mismatch']} paired samples were drawn with different seeds, e.g. "
                  + "; ".join(f"{'/'.join(e['sample'])}: {e['base']} vs {e['other']}" for e in sc["examples"][:3])
                  + " -- (seed, occurrence); a different occurrence means the arms grouped the prompt's repeats differently "
                  "(SE_SHARD, LIMIT) or one arm re-sent a request", flush=True)
    cells = sorted({c for d in data.values() for c in d["scores"]}, key=lambda c: ("/" in c, c))
    table, out = [], {}
    for cell in cells:
        if "/" in cell and not args.all_cells and not cell.startswith(("ifeval/", "lcb/", "aa_lcr/", "longbench_v2/")):
            continue
        row = {"cell": cell}
        for name, d in data.items():
            scores = d["scores"]
            vals = list(scores[cell].values())
            docs = {k[:2] for k in scores[cell]}
            a = d["aux"][cell]
            row[name] = {"n_docs": len(docs), "n_samples": len(vals), "mean": 100 * statistics.mean(vals) if vals else None,
                         "length_truncated_pct": pct(a["length_truncated"]), "no_think_end_pct": pct(a["no_think_end"]),
                         "no_answer_pct": pct(a["no_answer"]),
                         "completion_tokens_mean": statistics.mean(a["completion_tokens"]) if a["completion_tokens"] else None,
                         "unmatched_sidecar": a["unmatched"]}
            if name != base_name:
                row[name]["paired"] = paired(data[base_name]["scores"][cell], scores[cell])
        table.append(row)
        out[cell] = row
    names = [n for n, _ in arms]
    if args.md:
        print("| 任务 | 题数 × 共同轮次 | " + " | ".join(names) + " |")
        print("|---|---|" + "---|" * len(names))
    for row in table:
        b = row[base_name]
        # every arm is shown on the repetitions it shares with the baseline (the first paired arm's, for the baseline column);
        # an arm with more repetitions than that adds its own all-repetition mean
        first = next((row[n]["paired"] for n in names if n != base_name and row[n]["paired"].get("delta") is not None), None)
        parts = []
        for name in names:
            r = row[name]
            p = r.get("paired") if name != base_name else first
            common_mean = None if p is None or p.get("delta") is None else (p["other_common"] if name != base_name else p["base_common"])
            s = fmt(r["mean"]) if common_mean is None else fmt(common_mean)
            if name != base_name and common_mean is not None:
                s += f" ({p['delta']:+.1f} ± {p['se']:.1f}, n={p['n_docs']})"
            arm_reps = r["n_samples"] / r["n_docs"] if r["n_docs"] else 0
            if common_mean is not None and p and arm_reps > p["reps_per_doc"] + 1e-9:
                s += f" · 全部 {arm_reps:.0f} 轮 {fmt(r['mean'])}"
            s += f" · 截断 {fmt(r['length_truncated_pct'])}%"
            if r["no_think_end_pct"]:
                s += f" · 思考未闭合 {fmt(r['no_think_end_pct'])}%"
            if r["no_answer_pct"]:
                s += f" · 无答案 {fmt(r['no_answer_pct'])}%"
            if r["completion_tokens_mean"] is not None:
                s += f" · {r['completion_tokens_mean']:.0f} tok"
            parts.append(s)
        reps = (f"{first['reps_per_doc']:.0f}" if first else f"{b['n_samples'] / b['n_docs']:.0f}") if b["n_docs"] else "0"
        if args.md:
            print(f"| {row['cell']} | {b['n_docs']} × {reps} | " + " | ".join(parts) + " |")
        else:
            print(f"{row['cell']:<28} {b['n_docs']:>5} x {reps:<3} " + "  ||  ".join(parts))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"baseline": base_name, "arms": dict(arms), "repetitions": rep_record, "prompt_check": checks,
                       "seed_check": seed_checks, "cells": out}, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
