"""One-node I4 E2: matched GSM prompts, B1/B8 prefill, old/new N1 and M1."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from lightning_dense_bridge_qualify import atomic, http, worker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default="original,current,batched")
    ap.add_argument("--skip-numerical", action="store_true")
    ap.add_argument("--shadow-check", action="store_true",
                    help="Unscored candidate shadow prefill after N1, before all-shape warmup")
    ap.add_argument("--require-variable-n1", action="store_true",
                    help="All three arms wait for current/new N1 equality before timing")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    source = Path(plan["factored_source"])
    for name, digest in plan["source_file_sha256"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
    os.environ.update(HF_HOME="/work/hf-cache", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      HF_MODULES_CACHE=str(args.out / "hf-modules"))
    sys.path[:0] = plan["helper_paths"]
    from lightning_factored_service_pair import prepare_client
    preflight = args.out / "preflight"
    preflight.mkdir(exist_ok=True)
    prepare_client(plan, preflight, "gsm")
    from transformers import AutoTokenizer
    from twinstar.eval_thinking import load_gsm8k
    from kimi_ckpt_guard import gsm_prompt_ids
    tok = AutoTokenizer.from_pretrained(plan["model"], trust_remote_code=True)
    questions = load_gsm8k()[:200]
    prompts = [gsm_prompt_ids(tok, q["question"]) for q in questions]
    assert len(prompts) == 200
    atomic(args.out / "prompts.json", dict(ids=[q["id"] for q in questions], input_ids=prompts,
        sha256=[hashlib.sha256(json.dumps(p, separators=(",", ":")).encode()).hexdigest() for p in prompts]))

    def n1_gate():
        report = {}
        for kind, count in (("n1", 123), ("variable-n1", 24)):
            a, b = [json.loads((args.out/arm/(kind+".json")).read_text())
                    for arm in ("current", "batched")]
            assert a['complete'] and b['complete']
            lhs = [r['output_ids'] for x in a['rows'] for r in x['responses']]
            rhs = [r['output_ids'] for x in b['rows'] for r in x['responses']]
            assert len(lhs) == len(rhs) == count
            assert all(len(x) == 65 for x in lhs+rhs)
            report[kind] = dict(exact=sum(x == y for x, y in zip(lhs, rhs)), total=count)
        passed = all(x['exact'] == x['total'] for x in report.values())
        atomic(args.out/'prefill-admission.json', dict(passed=passed, checks=report))
        if not passed:
            raise RuntimeError('N1 failed; prefill performance stage was not admitted')

    admission = threading.Barrier(3, action=n1_gate, timeout=300) if args.require_variable_n1 else None
    if admission is not None:
        assert set(args.arms.split(',')) == {'original', 'current', 'batched'}

    def run_arm(item):
        label, arm, gpu, batched = item
        local = dict(plan, batched_extend=batched, instrument_forward=True)
        root = args.out / label

        def evaluate(url, _, root, env):
            def generate(inputs, count=1):
                return http(url, "/generate", dict(input_ids=inputs,
                    sampling_params=dict(temperature=0, max_new_tokens=count, ignore_eos=True)))

            probe = root / "probe"
            probe.mkdir(exist_ok=True)
            n1 = []
            for repeat in range(3):
                for batch in (1, 8, 32):
                    n1.append(dict(round=repeat, B=batch, responses=generate([[1000]*128]*batch, 65)))
                    atomic(root / "n1.json", dict(complete=False, rows=n1))
            atomic(root / "n1.json", dict(complete=True, rows=n1))
            # Variable lengths and nontrivial prompts, then slot reuse. Keep separate
            # from the legacy 123-request N1 so that neither gate is diluted.
            variable = [p[:n] for p, n in zip(prompts[:8], (1, 3, 17, 31, 63, 64, 95, 127))]
            rich = []
            for inputs in (variable, prompts[:8], list(reversed(variable))):
                rich.append(dict(input_ids=inputs, responses=generate(inputs, 65)))
                atomic(root / "variable-n1.json", dict(complete=False, rows=rich))
            atomic(root / "variable-n1.json", dict(complete=True, rows=rich))
            if args.shadow_check and batched:
                atomic(probe / "i4-diagnostic.json", dict(mode="shadow", tag="fixed-shadow"))
                shadow_response = generate(variable, 65)
                (probe / "i4-diagnostic.json").unlink()
                atomic(root / "shadow-n1.json", dict(input_ids=variable, responses=shadow_response))
            if admission is not None:
                admission.wait()
            samples = []
            for batch in (1, 8):
                batches = [prompts[i:i+batch] for i in range(0, 200, batch)]
                for inputs in batches:  # all scored lengths warm before measurement
                    generate(inputs)
                for repeat in range(2):
                    order = list(range(len(batches)))
                    if repeat:
                        order.reverse()
                    for index in order:
                        tag = f"B{batch}-r{repeat}-p{index}"
                        atomic(probe / "trigger.json", dict(tag=tag, mode="events",
                            all_forward_modes=True, synchronize_prefill=True))
                        wall = time.monotonic()
                        response = generate(batches[index])
                        elapsed = time.monotonic()-wall
                        (probe / "trigger.json").unlink()
                        samples.append(dict(tag=tag, B=batch, prompt_index=index, round=repeat,
                            endpoint_wall_ms=elapsed*1000, responses=response))
                    atomic(root / "prefill.json", dict(complete=False, rows=samples))
                print("I4_PREFILL", label, batch, flush=True)
            time.sleep(1)
            atomic(root / "prefill.json", dict(complete=True, rows=samples))
        worker(local, arm, gpu, root, micro=False, gsm=False, evaluate=evaluate)

    def numerical():
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="3", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   TRITON_CACHE_DIR=str(args.out / "numerical-triton"), NVIDIA_TF32_OVERRIDE="0")
        failed = []
        for command, logfile in [
            ([sys.executable, str(source / "test/manual/duet/test_lightning_batched_extend.py"),
              "--model", plan["model"], "--release", plan["release"], "--out", str(args.out/"extend-numerical.json")], "extend-test.log"),
            ([sys.executable, str(source / "benchmark/lightning_factored_mgs_sweep.py"),
              "--iterations", "3", "--out", str(args.out / "m1")], "m1.log"),
        ]:
            with (args.out / logfile).open("w") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                failed.append(f"{logfile}: exit {result.returncode}")
        if failed:
            raise RuntimeError("; ".join(failed))

    failures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        items = (
            ("original", "original-off", 0, False), ("current", "factored", 1, False),
            ("batched", "factored", 2, True))
        selected = args.arms.split(",")
        assert selected and set(selected) <= {item[0] for item in items}
        futures = {pool.submit(run_arm, item): item[0] for item in items if item[0] in selected}
        if not args.skip_numerical:
            futures[pool.submit(numerical)] = "numerical"
        for future in concurrent.futures.as_completed(futures):
            label = futures[future]
            try:
                future.result()
            except Exception as error:
                failures[label] = repr(error)
            atomic(args.out / "collection.json", dict(complete=False, failures=failures))
    atomic(args.out / "collection.json", dict(complete=not failures, failures=failures))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
