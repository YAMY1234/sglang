"""Stock Lightning HTTP smoke in the unmodified 20260913 SGLang image.

Run only inside an authorized GPU allocation. All artifacts go to --out.
This checks service behavior, not DUET parity or accuracy.
"""

import argparse
import concurrent.futures
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def request(base, path, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw else {"status": response.status}


def validate_generation(result):
    meta = result["meta_info"]
    assert isinstance(result["text"], str) and result["text"].strip(), result
    assert meta["completion_tokens"] == 16, meta
    scores = meta["output_token_logprobs"]
    assert len(scores) == 16, meta
    assert all(math.isfinite(row[0]) and row[0] <= 1e-5 for row in scores), scores
    assert all(isinstance(row[1], int) and row[1] >= 0 for row in scores), scores
    return [row[1] for row in scores]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=31333)
    parser.add_argument("--startup-timeout", type=int, default=900)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model_config = Path(args.model, "config.json").read_bytes()
    cmd = [
        sys.executable, "-m", "sglang.launch_server", "--model-path", args.model,
        "--served-model-name", "lightning-stock", "--host", "127.0.0.1",
        "--port", str(args.port), "--tp-size", "1", "--dtype", "bfloat16",
        "--trust-remote-code", "--context-length", "8192",
        "--max-total-tokens", "8192", "--max-running-requests", "4",
        "--mem-fraction-static", "0.7", "--disable-cuda-graph",
        "--disable-radix-cache", "--chunked-prefill-size", "-1",
    ]
    record = {
        "cell": "stock-lightning-http", "status": "running", "started_at": now(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"), "command": cmd,
        "model_config_sha256": hashlib.sha256(model_config).hexdigest(),
        "model_config": json.loads(model_config),
        "versions": {p: importlib.metadata.version(p) for p in ["sglang", "torch", "transformers"]},
        "scope": "stock serving only; DUET, NLL guard and GSM8K pending stage 2",
    }
    record["hardware"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,uuid", "--format=csv,noheader"], text=True
    ).strip().splitlines()
    base = f"http://127.0.0.1:{args.port}"
    proc = None
    begin = time.monotonic()
    try:
        with (args.out / "server.log").open("w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited {proc.returncode}; inspect server.log")
                try:
                    request(base, "/health", timeout=3)
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("server health deadline expired")
                    time.sleep(2)
            record["ready_at"] = now()
            record["startup_seconds"] = time.monotonic() - begin
            record["models"] = request(base, "/v1/models")
            assert any(m["id"] == "lightning-stock" for m in record["models"]["data"])
            payloads = [
                {"text": text, "sampling_params": {"temperature": 0, "max_new_tokens": 16, "ignore_eos": True},
                 "return_logprob": True, "logprob_start_len": 0}
                for text in ["The capital of France is", "The sum of one and two is"]
            ]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                outputs = list(pool.map(lambda p: request(base, "/generate", p), payloads))
            ids = [validate_generation(o) for o in outputs]
            repeat = request(base, "/generate", payloads[0])
            repeat_ids = validate_generation(repeat)
            record["generate"] = outputs
            record["repeat"] = repeat
            record["repeat_ids_equal"] = ids[0] == repeat_ids
            assert record["repeat_ids_equal"], "request reuse changed greedy output"
            record["completion"] = request(base, "/v1/completions", {
                "model": "lightning-stock", "prompt": "The capital of France is",
                "temperature": 0, "max_tokens": 16, "ignore_eos": True,
            })
            assert record["completion"]["choices"][0]["text"].strip()
            assert record["completion"]["usage"]["completion_tokens"] > 0
            request(base, "/health", timeout=10)
            assert proc.poll() is None
            record["status"] = "pass"
    except Exception as exc:
        record["status"] = "fail"
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        record["finished_at"] = now()
        record["wall_seconds"] = time.monotonic() - begin
        path = args.out / "result.json"
        path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: record.get(k) for k in ["status", "job_id", "startup_seconds", "wall_seconds", "error"]}), flush=True)
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
