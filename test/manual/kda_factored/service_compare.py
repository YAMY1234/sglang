"""Same-release dense+W32 versus factors: native end-to-end decode outputs.

Run one arm on each paired node. This is an argmax/output-logprob comparison,
not a teacher-forced full-logits BF16 test or a formal performance window.
"""

import concurrent.futures
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

arm = "dense_w32" if os.environ["SLURM_PROCID"] == "0" else "factored"
root = Path(os.environ["OUT"])
out = root / arm
out.mkdir(parents=True, exist_ok=True)
base = json.loads((Path(os.environ["KDA_BUNDLE"]) / "base-argv.json").read_text())
if "--port" not in base:
    base += ["--port", "30500"]
base += ["--duet-release", os.environ["KDA_RELEASE"]]
os.environ["SGLANG_KDA_FACTORED_STATE"] = (
    "0" if arm == "dense_w32" else "r=32,w=32,dtype=fp32"
)
command = [sys.executable, "-m", "sglang.launch_server", *base]
receipt = {
    "arm": arm,
    "family": os.environ["FAMILY"],
    "node": os.environ["SLURMD_NODENAME"],
    "job": os.environ["SLURM_JOB_ID"],
    "source_manifest": os.environ["KDA_SOURCE_MANIFEST"],
    "argv": command,
    "factor_option": os.environ["SGLANG_KDA_FACTORED_STATE"],
}
(out / "configuration.json").write_text(json.dumps(receipt, indent=2))
url = "http://127.0.0.1:30500"


def request(path, body=None, timeout=180):
    req = urllib.request.Request(
        url + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status if path == "/health" else json.load(response)


start = time.monotonic()
with (out / "server.log").open("w") as log:
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    try:
        while time.monotonic() - start < 1000:
            if process.poll() is not None:
                raise RuntimeError(f"{arm} server exited {process.returncode}")
            try:
                request("/health", timeout=5)
                break
            except OSError:
                time.sleep(2)
        else:
            raise TimeoutError(arm + " startup")
        receipt["health_seconds"] = time.monotonic() - start
        (out / "ready.json").write_text(json.dumps(receipt, indent=2))
        other = (
            root / ("factored" if arm == "dense_w32" else "dense_w32") / "ready.json"
        )
        until = time.monotonic() + 1000
        while not other.exists():
            if process.poll() is not None or time.monotonic() >= until:
                raise RuntimeError("paired service not ready")
            time.sleep(1)
        for batch in (1, 8, 32):
            prompts = [
                f"Question {i}: A shop has {17 + i} red balls and {8 + i} blue balls. It sells 9 balls. How many balls remain? Explain the arithmetic clearly. Answer:"
                for i in range(batch)
            ]
            payloads = [
                {
                    "text": text,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 64,
                        "ignore_eos": True,
                    },
                    "return_logprob": True,
                    "top_logprobs_num": 5,
                }
                for text in prompts
            ]
            (out / f"B{batch}-input.json").write_text(json.dumps(payloads, indent=2))
            began = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as executor:
                answers = list(
                    executor.map(lambda p: request("/generate", p), payloads)
                )
            assert all(x["meta_info"]["completion_tokens"] == 64 for x in answers)
            row = {
                "batch": batch,
                "input_sha256": hashlib.sha256(
                    json.dumps(payloads, sort_keys=True).encode()
                ).hexdigest(),
                "elapsed_seconds": time.monotonic() - began,
                "answers": answers,
            }
            (out / f"B{batch}.json").write_text(json.dumps(row, indent=2))
            print(
                json.dumps({k: v for k, v in row.items() if k != "answers"}), flush=True
            )
        receipt["complete"] = True
        (out / "complete.json").write_text(json.dumps(receipt, indent=2))
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
