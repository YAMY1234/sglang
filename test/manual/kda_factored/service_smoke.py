"""Weight-loaded native service admission; no formal performance claims."""

import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

out = Path(os.environ["OUT"])
out.mkdir(parents=True, exist_ok=True)
family = os.environ.get("FAMILY", "glm")
bundle = "inputs-glm-seed35-r6" if family == "glm" else "inputs-v35-mem065-r1"
args = json.loads(
    Path("/work/results/seed-serve-20261011", bundle, "base-argv.json").read_text()
)
if "--port" not in args:
    args += ["--port", "30500"]
release_root = Path("/work/hf-cache/duet-releases")
release = next(
    release_root.glob(
        "Latent-Voyager--GLM-5.3-Flash-SEED/*/spec.json"
        if family == "glm"
        else "Latent-Voyager--Kimi-Linear-48B-A3B-Instruct-SEED/*/spec.json"
    )
).parent
args += ["--duet-release", str(release)]
cmd = [sys.executable, "-m", "sglang.launch_server", *args]
(out / "argv.json").write_text(json.dumps(cmd, indent=2))
url = "http://127.0.0.1:30500"
start = time.monotonic()


def request(path, body=None):
    req = urllib.request.Request(
        url + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response) if path != "/health" else response.status


with (out / "server.log").open("w") as log:
    process = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    try:
        while time.monotonic() - start < 750:
            if process.poll() is not None:
                raise RuntimeError(
                    f"server exited {process.returncode}; see server.log"
                )
            try:
                request("/health")
                break
            except OSError:
                time.sleep(2)
        else:
            raise TimeoutError("service admission health deadline 750s")
        ready = time.monotonic() - start
        print(json.dumps({"family": family, "health_seconds": ready}), flush=True)
        (out / "server_info.json").write_text(
            json.dumps(request("/get_server_info"), indent=2)
        )
        rows = []
        for batch in (1, 8, 32):

            def generate(i):
                return request(
                    "/generate",
                    {
                        "text": f"Question {i}: Explain why the sky is blue in simple terms. Answer:",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 64,
                            "ignore_eos": True,
                        },
                    },
                )

            t = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as executor:
                replies = list(executor.map(generate, range(batch)))
            elapsed = time.monotonic() - t
            (out / f"B{batch}.json").write_text(json.dumps(replies, indent=2))
            assert all(x["meta_info"]["completion_tokens"] == 64 for x in replies)
            rows.append({"B": batch, "tokens": 64, "elapsed_seconds": elapsed})
            print(json.dumps(rows[-1]), flush=True)
        receipt = {
            "service_smoke": "PASS",
            "family": family,
            "health_seconds": ready,
            "batches": rows,
            "formal_window": False,
        }
        (out / "service_smoke.json").write_text(json.dumps(receipt, indent=2))
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
