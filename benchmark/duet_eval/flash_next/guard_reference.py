"""Reference side of the Flash-Next parity gate (validate_fn_parity_job.sh): the twinstar reference implementation scores the
32 guard windows of the admission manifest (prefix 3584, then 256 forced targets) and records the NLL of every position.

In-fork copy of twinstar `twinstar/bench/nvfp4_duet_guard.py` @ e91d2a7a6 (the accuracy-serving branch, retired
2026-10-07 and kept as an archive tag; file sha256 1629ae3ff1f2...): `save`, `identity`, `check_cases`,
`score_summary` and `reference_pass` unchanged; `reference` and `main` reduced to the one cell the parity job runs (`--cell nll`:
the DUET passes, the reference tree's source hashes and the release spec in <out>/reference.json, as the original writes them).
The other cells, the serving side and the verdicts of the original are not copied.

    PYTHONPATH=<twinstar checkout> python guard_reference.py reference --model BASE --release REL --manifest MANIFEST \
        --reference-root <twinstar checkout> --cell nll --repeats 2 --out OUT/ref

Needs torch and, for `reference`, the twinstar repository; `reference_pass(ts, cases)` takes any model with twinstar's
`generate_batch(prompts, n, forced=, logits_out=)` contract (the job's base pass imports it from this file)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    temp.replace(path)


def identity(case):
    import struct
    tokens = case["input_ids"] + case["targets"]
    return hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()


def check_cases(manifest):
    cases = manifest["cases"][:32]
    if (len(cases) != 32 or [c["window"] for c in cases] != list(range(32))
            or any(len(c["input_ids"]) != 3584 or len(c["targets"]) != 256 for c in cases)):
        raise ValueError("admission requires all 32 x 256 registered positions")
    return cases


def score_summary(cases, rows):
    expected = [(c["window"], s, t) for c in cases for s, t in enumerate(c["targets"])]
    if [(r["window"], r["step"], r["token"]) for r in rows] != expected:
        raise ValueError("scoring positions incomplete, duplicated or reordered")
    if any(not math.isfinite(r["nll"]) for r in rows):
        raise ValueError("nonfinite NLL")
    return dict(complete=True, window_hashes=[identity(c) for c in cases], rows=rows,
                mean_nll=statistics.fmean(r["nll"] for r in rows),
                per_window=[statistics.fmean(r["nll"] for r in rows if r["window"] == c["window"]) for c in cases])


def reference_pass(ts, cases):
    import torch
    rows, timings = [], []
    for case in cases:
        started = time.monotonic()
        class Scores:
            step = 0

            def append(self, logits):
                token = case["targets"][self.step]
                value = -torch.log_softmax(logits[0].float(), -1)[token].item()
                rows.append(dict(window=case["window"], step=self.step, token=token, nll=value))
                self.step += 1

        scores = Scores()
        outputs = ts.generate_batch([case["input_ids"]], len(case["targets"]), forced=[case["targets"]], logits_out=scores)
        if outputs != [case["targets"]] or scores.step != len(case["targets"]):
            raise RuntimeError("reference forcing incomplete")
        timings.append(dict(window=case["window"], seconds=time.monotonic()-started))
        print("REFERENCE", case["window"], scores.step, timings[-1], flush=True)
    return {**score_summary(cases, rows), "window_timings": timings}


def reference(args, manifest):
    import torch
    import twinstar
    from twinstar.models.ckpt import twinstar_from_spec

    root = Path(twinstar.__file__).resolve().parent.parent
    if root != Path(args.reference_root).resolve():
        raise RuntimeError(f"reference imported from unexpected tree {root}")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    ts = twinstar_from_spec(args.model, "all", "all", args.release)
    cases = check_cases(manifest)
    result = dict(complete=False, model=args.model, release=args.release, reference_root=str(root), passes=[])
    files = ["release/run_duet.py", "release/hf_duet.py", "twinstar/models/ckpt.py", "twinstar/models/nvfp4.py",
             "twinstar/models/qwen4_exp.py", "twinstar/duet/spec.py", "twinstar/duet/state.py", "twinstar/duet/latentfmt.py", "twinstar/duet/gdnfactor.py", "twinstar/duet/latent.py"]
    result["source_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
    result["spec"] = json.loads((Path(args.release) / "spec.json").read_text())
    for repeat in range(args.repeats):
        result["passes"].append(reference_pass(ts, cases))
        save(args.out / "reference.json", result)
    result["complete"] = True
    save(args.out / "reference.json", result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["reference"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference-root", default="/ref")
    ap.add_argument("--cell", choices=["nll"], default="nll")
    ap.add_argument("--repeats", type=int, choices=[1, 2], default=2)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reference(args, json.loads(args.manifest.read_text()))


if __name__ == "__main__":
    main()
