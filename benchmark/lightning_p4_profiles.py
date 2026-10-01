"""docs/167 P4: Lightning DUET under both numerics profiles in one 4-GPU allocation.

GPU0  engine worker, reference profile   -> duet, duet2
GPU1  control worker                     -> stock1, stock2, off, off2
GPU2  engine worker, production profile  -> duetp, duetp2   (--duet-allow-unvalidated-profile)
GPU3  options worker (accuracy-first smoke, reference profile)
reference1 / reference2 are copied from a completed cell (--reuse-reference, e.g. 940205's guard dir): the torch
reference does not depend on the serving tree, and the windows file is checked to be the same object.
Then benchmark/lightning_sgl_decision.py runs twice (candidates duet,duet2 and duetp,duetp2).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GUARD = HERE / "lightning_sgl_guard.py"
DECISION = HERE / "lightning_sgl_decision.py"


def now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def copy_reference(source, out, windows):
    wanted = json.loads(Path(windows).read_text())["windows"]
    identity = [(w["id"], w["prompt_sha256"], w["target"]) for w in wanted]
    for name in ("reference1", "reference2"):
        cell = json.loads((Path(source) / (name + ".json")).read_text())
        if not cell.get("complete") or len(cell["windows"]) != 32:
            raise ValueError(f"{name}: reused reference cell incomplete")
        if [(w["id"], w["prompt_sha256"], w["target"]) for w in cell["windows"]] != identity:
            raise ValueError(f"{name}: reused reference cell was measured on different windows")
        shutil.copy(Path(source) / (name + ".json"), out / (name + ".json"))
    return {"source": str(source), "reference_sha": cell.get("reference_sha"),
            "windows_sha256": hashlib.sha256(Path(windows).read_bytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--duet", required=True)
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reuse-reference", type=Path, required=True, help="dir with reference1.json / reference2.json")
    ap.add_argument("--reference-repo", default="/reference")
    ap.add_argument("--production-max-running-requests", type=int, default=2,
                    help="kept equal to the reference arm for the NLL cells (fairness); throughput uses its own job")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    record = dict(cell="p4-lightning-profiles", status="running", started_at=now(), job_id=os.environ.get("SLURM_JOB_ID"))
    start = time.monotonic()
    record["reused_reference"] = copy_reference(args.reuse_reference, args.out, args.windows)
    lanes = [
        ("engine", 0, ["--duet-numerics", "reference", "--cell-prefix", "duet"]),
        ("control", 1, []),
        ("engine", 2, ["--duet-numerics", "production", "--cell-prefix", "duetp"]),
        ("options", 3, ["--duet-numerics", "reference"]),
    ]
    procs, logs = [], []
    try:
        for mode, gpu, extra in lanes:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["TWINSTAR_DEVICES"] = "cuda:0"
            env["PYTHONPATH"] = args.reference_repo + os.pathsep + env.get("PYTHONPATH", "")
            for key in ("LATENT_OFF", "STATE_OFF", "TWINSTAR_STATE_TRUNCATION", "TWINSTAR_STATE_OVERSAMPLE", "TWINSTAR_STATE_POWER"):
                env.pop(key, None)
            cmd = [sys.executable, str(GUARD), "--worker", mode, "--gpu", str(gpu), "--model", args.model,
                   "--duet", args.duet, "--windows", str(args.windows), "--out", str(args.out),
                   "--reference-repo", args.reference_repo, *extra]
            name = f"{mode}-{'production' if 'production' in extra else 'reference'}" if mode == "engine" else mode
            log = (args.out / f"{name}.log").open("w"); logs.append(log)
            procs.append((name, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))
            print(f"started {name} on GPU{gpu}", flush=True)
        while any(p.poll() is None for _, p in procs):
            for name, p in procs:
                if p.poll() not in (None, 0):
                    raise RuntimeError(f"{name} worker exited {p.returncode}; see {args.out}/{name}.log")
            time.sleep(5)
        for name, p in procs:
            if p.returncode:
                raise RuntimeError(f"{name} worker exited {p.returncode}")
        decisions = {}
        for profile, names in (("reference", "duet,duet2"), ("production", "duetp,duetp2")):
            report = f"variance-decision-{profile}.json"
            rc = subprocess.run([sys.executable, str(DECISION), "--out", str(args.out), "--candidate-names", names,
                                 "--report", report], capture_output=True, text=True)
            decisions[profile] = dict(returncode=rc.returncode, report=report, stdout=rc.stdout[-2000:], stderr=rc.stderr[-2000:])
            print(f"decision[{profile}] rc={rc.returncode}: {rc.stdout.strip()[:400]}", flush=True)
        record["decisions"] = decisions
        record["status"] = "pass" if all(d["returncode"] == 0 for d in decisions.values()) else "fail"
    except Exception as exc:  # noqa: BLE001 -- record the failure, then stop the lanes
        record.update(status="fail", error=f"{type(exc).__name__}: {exc}")
        for _, p in procs:
            if p.poll() is None:
                p.terminate()
        for _, p in procs:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()
    finally:
        for log in logs:
            log.close()
        record.update(finished_at=now(), wall_seconds=time.monotonic() - start)
        (args.out / "p4-profiles.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({k: record.get(k) for k in ("status", "wall_seconds", "error")}), flush=True)
    return int(record["status"] != "pass")


if __name__ == "__main__":
    sys.exit(main())
