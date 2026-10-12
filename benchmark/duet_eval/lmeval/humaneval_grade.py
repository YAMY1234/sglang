"""Grade HumanEval+ generations in the lm-eval sample logs with EvalPlus itself (evalplus 26d6d00bb1fd, evalplus_utils.py):
each reply (the text after the thinking block, as model.py stores it) is sanitized by evalplus.sanitize.sanitize(reply,
entrypoint) -- what evalplus.codegen does with a chat model's reply -- and every sanitized solution of every listed directory is
run by ONE evalplus.evaluate.evaluate call (dataset humaneval = HumanEval+ v0.1.10, base and plus tests, EvalPlus's default time
limits max(4 s, 4 x ground-truth time) per test; the address-space cap per checker raised from 4 to 8 GiB, MEMORY_LIMIT_BYTES).  One call for both arms of a pairing, with the samples of a
task interleaved across the listed directories, so the two arms are graded by the same process, ground truth and load.  Adds
to each sample line of samples_duet_humaneval_plus_*.jsonl: `solution` (the sanitized code), `base_status` / `plus_status`
(EvalPlus's pass / fail / timeout), `humaneval` (1 = passes the base tests: HumanEval), `humaneval_plus` (1 = passes base and plus
tests: HumanEval+), `no_answer` (1 = the sanitized solution does not define the task's entry point -- a reporting flag only: EvalPlus
grades every solution, and its sanitizer returns the reply's first line when it finds no code) and `humaneval_grade` (the grading
record's path); the original
file is kept once as <name>.orig.  --passes 2 grades the same solutions a second time and reports how many verdicts differ (the
grader's own run-to-run noise: timeouts under load); the sample lines keep the first pass.  --canonical checks the grader on the
canonical solutions (every task must pass the base and plus tests).

    python -m benchmark.duet_eval.lmeval.humaneval_grade --out OUT/humaneval-grade OUT/stock OUT/duet [--parallel N] [--passes 2]
    python -m benchmark.duet_eval.lmeval.humaneval_grade --out DIR --canonical

EvalPlus reads its data and caches the ground truth under $XDG_CACHE_HOME/evalplus, which this process sets to
$DUET_DATA_ROOT/evalplus-cache (prepare_data.sh puts HumanEvalPlus-v0.1.10.jsonl there; the first grading, or --canonical, writes the
ground truth: the canonical solutions' outputs and run times, from which EvalPlus derives the time limits); the grader needs evalplus and tree-sitter (requirements-evalplus.txt),
not lm-eval or a GPU.  It executes model-written code: run it inside the exam container on a compute node, not on a login node."""
import argparse
import ast
import collections
import glob
import hashlib
import json
import os
import shutil
import sys
import time

from benchmark.duet_eval.lmeval.common import response_of
from benchmark.duet_eval.lmeval.evalplus_utils import EVALPLUS_CACHE, N_DOCS

EVALPLUS_COMMIT = "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"
# before evalplus is imported (evalplus.data.utils.CACHE_DIR): every grading reads the same data file and ground truth
os.environ["XDG_CACHE_HOME"] = EVALPLUS_CACHE
# EvalPlus caps each checker process's address space at EVALPLUS_MAX_MEMORY_BYTES (default 4 GiB), and the checker inherits the
# address space of evaluate's process, 4.44 GiB in the exam container (the ground truth of the large plus inputs; twinstar docs/90):
# under the default the canonical solutions of HumanEval/15 and /130 fail on their million-element inputs.  8 GiB leaves a solution
# 3.5 GiB above what it inherits; every canonical solution then passes except HumanEval/32 (KNOWN_ORACLE_FAILURES).
MEMORY_LIMIT_BYTES = 8 * 2**30
os.environ["EVALPLUS_MAX_MEMORY_BYTES"] = str(MEMORY_LIMIT_BYTES)
# HumanEval/32 (find_zero): its canonical solution fails its own plus oracle |poly(xs, x)| <= 1e-4 on inputs with large
# coefficients, at any memory limit (reported upstream: github.com/evalplus/evalplus/issues/253); kept, as upstream scores it
KNOWN_ORACLE_FAILURES = ("HumanEval/32",)


def evalplus_record():
    """The installed EvalPlus: version and the commit pip recorded (direct_url.json) -- must be EVALPLUS_COMMIT."""
    from importlib.metadata import distribution
    dist = distribution("evalplus")
    commit = None
    try:
        commit = json.loads(dist.read_text("direct_url.json") or "{}").get("vcs_info", {}).get("commit_id")
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {"version": dist.version, "commit": commit}


def defines_entry_point(code, name):
    """True when the code parses and defines `name` at the top level (the function EvalPlus calls)."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in tree.body)


def sample_files(dirs):
    files = []
    for d in dirs:
        files += [d] if d.endswith(".jsonl") else glob.glob(os.path.join(d, "**", "samples_duet_humaneval_plus_*.jsonl"), recursive=True)
    return sorted(set(files))


def rep_of(path):
    for part in reversed(path.split(os.sep)):
        if part.startswith("rep") and part[3:].isdigit():
            return int(part[3:])
    return -1


def run_evaluate(samples_path, result_path, parallel):
    from evalplus.evaluate import evaluate
    if os.path.exists(result_path):
        os.remove(result_path)
    evaluate(dataset="humaneval", samples=samples_path, parallel=parallel, output_file=result_path)
    with open(result_path, encoding="utf-8") as fh:
        return json.load(fh)["eval"]


def verdicts(entries, results):
    """entries [(task_id, solution)] in file order -> [(base_status, plus_status)]: EvalPlus lists each task's results in the
    order its samples appear (completion_id)."""
    seen = collections.Counter()
    out = []
    for task_id, solution in entries:
        r = results[task_id][seen[task_id]]
        seen[task_id] += 1
        if r["solution"] != solution:
            raise SystemExit(f"EvalPlus result order mismatch at {task_id} #{seen[task_id]}")
        out.append((r["base_status"], r["plus_status"]))
    return out


def grade(dirs, out, parallel, passes, partial=False):
    from evalplus.data import get_human_eval_plus
    from evalplus.sanitize import sanitize
    problems = get_human_eval_plus()
    files = sample_files(dirs)
    if not files:
        raise SystemExit(f"no samples_duet_humaneval_plus_*.jsonl below {dirs}")
    lines = {f: [json.loads(l) for l in open(f, encoding="utf-8")] for f in files}
    order = {d: i for i, d in enumerate(dirs)}
    owner = {f: next((order[d] for d in dirs if f.startswith(d.rstrip("/") + "/") or f == d), 0) for f in files}
    items = []                                                   # (doc_id, rep, directory index, file, line index)
    for f, ls in lines.items():
        for i, d in enumerate(ls):
            items.append((d["doc_id"], rep_of(f), owner[f], f, i))
    items.sort()                                                 # a task's samples together, the arms interleaved
    entries = []
    for doc_id, rep, k, f, i in items:
        d = lines[f][i]
        task_id = d["doc"]["task_id"]
        entry_point = problems[task_id]["entry_point"]
        solution = sanitize(response_of(d), entrypoint=entry_point)
        d["solution"] = solution
        d["no_answer"] = int(not defines_entry_point(solution, entry_point))
        entries.append((task_id, solution))
    missing = sorted(set(problems) - {t for t, _ in entries})
    if missing and not partial:
        raise SystemExit(f"{len(missing)} HumanEval+ tasks have no sample (e.g. {missing[:3]}): grade complete runs only")
    os.makedirs(out, exist_ok=True)
    samples_path = os.path.join(out, "samples.jsonl")
    with open(samples_path, "w", encoding="utf-8") as fh:
        # --partial (smoke tests): evalplus.evaluate needs every task, so a task without a sample gets an empty solution at the end
        for task_id, solution in entries + [(t, "") for t in missing]:
            fh.write(json.dumps({"task_id": task_id, "solution": solution}) + "\n")
    t0 = time.time()
    runs = [verdicts(entries, run_evaluate(samples_path, os.path.join(out, f"eval_results.pass{p + 1}.json"), parallel))
            for p in range(passes)]
    first = runs[0]
    by_dir = collections.defaultdict(lambda: collections.Counter())
    for (doc_id, rep, k, f, i), (base, plus) in zip(items, first):
        d = lines[f][i]
        d.update(base_status=base, plus_status=plus, humaneval=int(base == "pass"), humaneval_plus=int(base == "pass" and plus == "pass"),
                 humaneval_grade=os.path.abspath(out))
        c = by_dir[dirs[k]]
        c["samples"] += 1
        c["humaneval"] += d["humaneval"]
        c["humaneval_plus"] += d["humaneval_plus"]
        c["no_answer"] += d["no_answer"]
        c[f"base_{base}"] += 1
        c[f"plus_{plus}"] += 1
    flips = collections.Counter()
    for other in runs[1:]:
        for (doc_id, rep, k, f, i), a, b in zip(items, first, other):
            if (a[0] == "pass" and a[1] == "pass") != (b[0] == "pass" and b[1] == "pass"):
                flips[dirs[k]] += 1
    for f, ls in lines.items():
        backup = f + ".orig"
        if not os.path.exists(backup):
            shutil.copy(f, backup)
        tmp = f + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for d in ls:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        os.replace(tmp, f)
    gt = sorted(glob.glob(os.path.join(os.environ["XDG_CACHE_HOME"], "evalplus", "*.pkl")))
    record = {"evalplus": evalplus_record(), "expected_commit": EVALPLUS_COMMIT, "memory_limit_bytes": MEMORY_LIMIT_BYTES,
              "dirs": dirs, "files": files, "passes": passes,
              "parallel": parallel, "cpus": len(os.sched_getaffinity(0)), "seconds": round(time.time() - t0, 1),
              "ground_truth": {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest() for p in gt},
              "per_dir": {d: dict(c) for d, c in by_dir.items()},
              "verdict_flips_between_passes": {d: flips[d] for d in dirs} if passes > 1 else None}
    with open(os.path.join(out, "grade.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)
    for d in dirs:
        c = by_dir[d]
        n = c["samples"] or 1
        print(f"{d}: {c['samples']} samples, HumanEval+ {100 * c['humaneval_plus'] / n:.1f}%, HumanEval {100 * c['humaneval'] / n:.1f}%, "
              f"no code {c['no_answer']}, base timeouts {c['base_timeout']}, plus timeouts {c['plus_timeout']}"
              + (f", verdicts differing in pass 2: {flips[d]}" if passes > 1 else ""), flush=True)
    print(f"HUMANEVAL GRADE DONE ({time.strftime('%H:%M')}): {out}/grade.json", flush=True)


def canonical(out, parallel):
    """Every canonical solution, written as a chat reply (prose + a python code block), through the same sanitizer and evaluator."""
    from evalplus.data import get_human_eval_plus
    from evalplus.sanitize import sanitize
    problems = get_human_eval_plus()
    if len(problems) != N_DOCS:
        raise SystemExit(f"HumanEval+ holds {len(problems)} tasks, {N_DOCS} expected")
    entries = []
    for task_id, p in problems.items():
        reply = f"Here is the solution.\n\n```python\n{p['prompt']}{p['canonical_solution']}\n```\n\nIt passes the tests."
        entries.append((task_id, sanitize(reply, entrypoint=p["entry_point"])))
    os.makedirs(out, exist_ok=True)
    samples_path = os.path.join(out, "canonical.jsonl")
    with open(samples_path, "w", encoding="utf-8") as fh:
        for task_id, solution in entries:
            fh.write(json.dumps({"task_id": task_id, "solution": solution}) + "\n")
    v = verdicts(entries, run_evaluate(samples_path, os.path.join(out, "canonical.eval_results.json"), parallel))
    bad = [(t, b, p) for (t, _), (b, p) in zip(entries, v) if not (b == "pass" and p == "pass")]
    print(f"canonical solutions: {len(entries) - len(bad)} of {len(entries)} pass base + plus tests; failing: {bad}; "
          f"memory limit {MEMORY_LIMIT_BYTES / 2**30:.0f} GiB; evalplus {evalplus_record()}", flush=True)
    ok = all(t in KNOWN_ORACLE_FAILURES and b == "pass" for t, b, _ in bad)
    print("HUMANEVAL CANONICAL " + ("PASS" if not bad else f"PASS except the known oracle failures {list(KNOWN_ORACLE_FAILURES)}" if ok
                                    else "FAIL"), flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", help="arm directories (rep*/ inside) or sample files; the arms of a pairing in ONE call")
    ap.add_argument("--out", required=True, help="directory for the grading record (samples.jsonl, eval_results.pass*.json, grade.json)")
    ap.add_argument("--parallel", type=int, default=max(1, len(os.sched_getaffinity(0)) // 2),
                    help="EvalPlus worker processes (default: half the CPUs this process may use; EvalPlus warns against oversubscription)")
    ap.add_argument("--passes", type=int, default=1, help="grade the same solutions this many times (the first pass is recorded)")
    ap.add_argument("--canonical", action="store_true", help="check the grader on the canonical solutions")
    ap.add_argument("--partial", action="store_true",
                    help="smoke tests only: grade runs that lack some tasks (each missing task is padded with an empty solution, not reported)")
    a = ap.parse_args()
    if a.canonical:
        sys.exit(0 if canonical(a.out, a.parallel) else 1)
    if not a.dirs:
        ap.error("give the arm directories to grade")
    grade([os.path.abspath(d) for d in a.dirs], a.out, a.parallel, a.passes, partial=a.partial)


if __name__ == "__main__":
    main()
