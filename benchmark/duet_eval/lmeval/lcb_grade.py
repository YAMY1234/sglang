"""Grade LiveCodeBench generations in the lm-eval sample logs with the official LiveCodeBench grader (vendored under lcb/):
every public + private test case, 6 s per test, in parallel processes.  Adds `pass@1` (0/1), `tests_passed`, `tests_total`
and `code` to each sample line of samples_duet_lcb_*.jsonl (the original file is kept once as <name>.orig).

    python -m benchmark.duet_eval.lmeval.lcb_grade --tests $DUET_DATA_ROOT/lcb/lcb_v6_2408_2505_tests.pkl \\
        OUT/duet OUT/stock [--processes 48] [--timeout 6]
"""
import argparse
import glob
import json
import os
import pickle
import shutil

from benchmark.duet_eval.lmeval.lcb.compute_code_generation_metrics import codegen_metrics
from benchmark.duet_eval.lmeval.lcb_utils import extract_code


def grade_file(path, tests, processes, timeout):
    lines = [json.loads(l) for l in open(path, encoding="utf-8")]
    if lines and all("pass@1" in d for d in lines):
        print(f"{path}: already graded ({len(lines)} samples)")
        return
    samples, gens = [], []
    for d in lines:
        qid = d["doc"]["question_id"]
        samples.append({"input_output": tests[qid]["input_output"]})
        gens.append([extract_code(d["resps"][0][0]) if d.get("resps") else ""])
    metrics, results, metadata = codegen_metrics(samples, gens, k_list=[1], num_process_evaluate=processes, timeout=timeout)
    for i, d in enumerate(lines):
        graded = results[i][0]                                  # list over tests: 1 pass, 0 fail, -1 / -2 timeout / error
        d["graded_list"] = graded
        d["tests_total"] = len(graded)
        d["tests_passed"] = sum(1 for g in graded if g == 1)
        d["pass@1"] = int(len(graded) > 0 and all(g == 1 for g in graded))
        d["code"] = gens[i][0]
    backup = path + ".orig"
    if not os.path.exists(backup):
        shutil.copy(path, backup)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for d in lines:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    n = len(lines)
    print(f"{path}: {n} samples, pass@1 {100 * sum(d['pass@1'] for d in lines) / n:.1f}%, no code {sum(1 for d in lines if not d['code'].strip())}, "
          f"official pass@1 {metrics.get('pass@1', float('nan')):.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", required=True)
    ap.add_argument("dirs", nargs="+", help="arm directories (rep*/ inside) or sample files")
    ap.add_argument("--processes", type=int, default=max(4, (os.cpu_count() or 8) // 2))
    ap.add_argument("--timeout", type=int, default=6)
    a = ap.parse_args()
    tests = pickle.load(open(a.tests, "rb"))
    files = []
    for d in a.dirs:
        files += [d] if d.endswith(".jsonl") else glob.glob(os.path.join(d, "**", "samples_duet_lcb_*.jsonl"), recursive=True)
    for path in sorted(files):
        grade_file(path, tests, a.processes, a.timeout)


if __name__ == "__main__":
    main()
