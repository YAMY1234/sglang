"""Same-build native servers; sealed docs/197 GSM200 evaluator, one arm/node."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import gsm_endpoint

family = os.environ["FAMILY"]
arm = "original" if os.environ["SLURM_PROCID"] == "0" else "factored"
root = Path(os.environ["OUT"])
out = root / arm
out.mkdir(parents=True, exist_ok=True)
bundle = Path(os.environ["KDA_BUNDLE"])
args = json.loads((bundle / "base-argv.json").read_text())
if "--port" not in args:
    args += ["--port", "30500"]
if arm == "factored":
    args += ["--duet-release", os.environ["KDA_RELEASE"]]
    os.environ["SGLANG_KDA_FACTORED_STATE"] = "r=32,w=32"
else:
    os.environ["SGLANG_KDA_FACTORED_STATE"] = "0"
model = args[args.index("--model-path") + 1]
cmd = [sys.executable, "-m", "sglang.launch_server", *args]
receipt = {
    "family": family,
    "arm": arm,
    "argv": cmd,
    "node": os.environ["SLURMD_NODENAME"],
    "job": os.environ["SLURM_JOB_ID"],
    "source_manifest": os.environ["KDA_SOURCE_MANIFEST"],
    "evaluator_sha256": hashlib.sha256(
        Path(gsm_endpoint.__file__).read_bytes()
    ).hexdigest(),
    "env": {
        k: v
        for k, v in os.environ.items()
        if k.startswith(("SGLANG_", "NVIDIA_", "HF_", "PYTHONPATH"))
    },
}
(out / "configuration.json").write_text(json.dumps(receipt, indent=2))
url = "http://127.0.0.1:30500"
start = time.monotonic()
with (out / "server.log").open("w") as log:
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    try:
        while time.monotonic() - start < 850:
            if proc.poll() is not None:
                raise RuntimeError(f"{arm} server exited {proc.returncode}")
            try:
                with urllib.request.urlopen(url + "/health", timeout=5):
                    break
            except OSError:
                time.sleep(2)
        else:
            raise TimeoutError(f"{arm} startup")
        receipt["health_seconds"] = time.monotonic() - start
        receipt["ready_time_ns"] = time.time_ns()
        (out / "ready.json").write_text(json.dumps(receipt, indent=2))
        other = root / ("factored" if arm == "original" else "original") / "ready.json"
        until = time.monotonic() + 850
        while not other.exists():
            if proc.poll() is not None or time.monotonic() >= until:
                raise RuntimeError("paired service did not become ready")
            time.sleep(1)
        # Both native servers are healthy before either evaluator starts.
        gsm_endpoint.run(url, model, out / "gsm200.json")
        receipt["complete"] = True
        receipt["total_seconds"] = time.monotonic() - start
        (out / "complete.json").write_text(json.dumps(receipt, indent=2))
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
