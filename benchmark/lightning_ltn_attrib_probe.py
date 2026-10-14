"""docs/209: long-prompt B1 prefill attribution for the C8 4484543 service arms.

Diagnostic only. Each arm is launched by the frozen qualify worker with the
E3 plan, then its execution env/argv is checked against the served
configuration.json before any timing. Prompts are B1, the shape 98% of window
prefills had; plain wall timing precedes the i4 stage probe.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from lightning_dense_bridge_qualify import atomic, http, worker

LENGTHS = (512, 2048, 8192, 16384, 24576, 32000)
EQUALITY = (2048, 16384)
# Keys that legitimately differ from the served arm: placement, caches, probe dir,
# the source path of an ablation export, and the ablation switch itself.
ALLOWED = {"CUDA_VISIBLE_DEVICES", "HF_MODULES_CACHE", "SGLANG_CACHE_DIR", "LTNBRIDGE_PROBE_DIR",
           "PYTHONPATH", "SGLANG_LTN_BOUNDARY_CODEC"}


def prompt(length):
    rng = random.Random(20261014 + length)
    return [rng.randrange(1000, 120000) for _ in range(length)]


def config_diff(execution, served, source, served_source):
    env, ref = execution["env"], served["env"]
    keys = sorted(set(env) | set(ref))
    diff = {k: [ref.get(k), env.get(k)] for k in keys if env.get(k) != ref.get(k)}
    unexpected = sorted(set(diff) - ALLOWED)
    if "PYTHONPATH" in diff and env["PYTHONPATH"].replace(source, served_source) != ref["PYTHONPATH"]:
        unexpected.append("PYTHONPATH")
    argv, ref_argv = list(execution["argv"][1:]), list(served["argv"][1:])
    for a in (argv, ref_argv):
        a[a.index("--port") + 1] = "PORT"
    if argv != ref_argv:
        unexpected.append("argv")
    return dict(diff=diff, unexpected=unexpected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--served-config", type=Path, required=True)
    ap.add_argument("--arm", choices=("original-off", "factored"), required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--source")
    ap.add_argument("--sha")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    served = json.loads(args.served_config.read_text())
    if args.source:
        plan.update(factored_source=args.source, factored_sha=args.sha)
    prompts = {length: prompt(length) for length in LENGTHS}
    digest = hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest()

    def evaluate(url, plan, root, env):
        execution = json.loads((root / "execution.json").read_text())
        check = config_diff(execution, served, execution["source"], served["source"])
        atomic(root / "config-diff.json", check)
        if check["unexpected"]:
            raise RuntimeError(f"arm differs from served configuration: {check['unexpected']}")
        probe = root / "probe"
        probe.mkdir(exist_ok=True)
        rows = []

        def generate(ids, new_tokens=1):
            start = time.perf_counter()
            response = http(url, "/generate", dict(input_ids=ids, sampling_params=dict(
                temperature=0, max_new_tokens=new_tokens, ignore_eos=True)))
            return (time.perf_counter() - start) * 1000, response

        for length in LENGTHS:  # compile/autotune every shape outside the scored passes
            generate(prompts[length])
        for mode, reps in (("plain", 3), ("i4-stages", 2)):
            for rep in range(reps):
                for length in LENGTHS:
                    tag = f"{mode}-L{length}-r{rep}"
                    if mode != "plain":
                        atomic(probe / "trigger.json", dict(tag=tag, mode=mode))
                    wall_ms, response = generate(prompts[length])
                    if mode != "plain":
                        (probe / "trigger.json").unlink()
                    rows.append(dict(tag=tag, mode=mode, rep=rep, length=length, wall_ms=wall_ms,
                                     meta_info=response.get("meta_info")))
                    atomic(root / "timing.json", dict(complete=False, prompts_sha256=digest, rows=rows))
        equality = {}
        for length in EQUALITY:
            _, response = generate(prompts[length], 32)
            equality[length] = dict(output_ids=response.get("output_ids"), text=response.get("text"))
        atomic(root / "equality.json", equality)
        atomic(root / "timing.json", dict(complete=True, prompts_sha256=digest, rows=rows))

    worker(dict(plan, instrument_forward=True), args.arm, args.gpu, args.out,
           micro=False, gsm=False, evaluate=evaluate)


if __name__ == "__main__":
    main()
