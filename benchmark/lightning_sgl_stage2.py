"""Lightning DUET GPU cells. Each attempt writes its own auditable JSON."""

import argparse
import datetime
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def request(endpoint, path, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        endpoint + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw else {"status": response.status}


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def server(args, mode, directory, port=31334):
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for key in ("SGLANG_DUET_ENABLED", "TWINSTAR_LIGHTNING_DUET", "TWINSTAR_LIGHTNING_DUET_DIR"):
        env.pop(key, None)
    if mode == "duet":
        env["SGLANG_DUET_DIR"] = args.duet
    else:
        # DIR itself enables DUET; an inherited path must not activate off/stock.
        env.pop("SGLANG_DUET_DIR", None)
        env.pop("TWINSTAR_LIGHTNING_DUET_DIR", None)
    if mode == "stock":
        env.pop("SGLANG_EXTERNAL_MODEL_PACKAGE", None)
    else:
        env["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "sglang.srt.models.lightning_duet"
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--served-model-name",
        "lightning",
        "--random-seed",
        "20260929",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tp-size",
        "1",
        "--watchdog-timeout",
        "1800",
        "--dtype",
        "bfloat16",
        "--trust-remote-code",
        "--context-length",
        "8192",
        "--max-total-tokens",
        "16384",
        "--max-running-requests",
        "2",
        "--max-mamba-cache-size",
        "8",
        "--mem-fraction-static",
        "0.7",
        "--disable-cuda-graph",
        "--disable-overlap-schedule",
        "--disable-radix-cache",
        "--chunked-prefill-size",
        "-1",
        "--skip-server-warmup",
    ]
    if mode == "duet":
        cmd.extend(getattr(args, "duet_cli", []))
        cmd.extend(["--duet-release", args.duet])
    log = (directory / "server.log").open("w")
    begin = time.monotonic()
    proc = subprocess.Popen(
        cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    metadata = {"mode": mode, "command": cmd, "started_at": now(), "pid": proc.pid}
    endpoint = f"http://127.0.0.1:{port}"
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"{mode} server exited {proc.returncode}; see {directory}/server.log"
                )
            try:
                request(endpoint, "/health", timeout=2)
                break
            except (OSError, urllib.error.URLError):
                if time.monotonic() - begin > 600:
                    raise TimeoutError("server did not become healthy within 600s")
                time.sleep(2)
        metadata["ready_seconds"] = time.monotonic() - begin
        metadata["models"] = request(endpoint, "/v1/models")
        if not any(row["id"] == "lightning" for row in metadata["models"]["data"]):
            raise AssertionError("served model missing from /v1/models")
        save(directory / "server.json", metadata)
        yield endpoint, metadata
        request(endpoint, "/health", timeout=10)
        if proc.poll() is not None:
            raise RuntimeError("server exited before final health check")
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        log.close()
        metadata.update(
            finished_at=now(),
            wall_seconds=time.monotonic() - begin,
            teardown_returncode=proc.returncode,
        )
        save(directory / "server.json", metadata)


def load_cell(args, out):
    with server(args, "duet", out, port=getattr(args, "port", 31334)) as (
        endpoint,
        metadata,
    ):
        outputs = []
        for prompt in [
            "The capital of France is",
            "The sum of one and two is",
            "The capital of France is",
        ]:
            output = request(
                endpoint,
                "/generate",
                {
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 33,
                        "ignore_eos": True,
                    },
                    "return_logprob": True,
                    "logprob_start_len": -1,
                },
            )
            scores = output["meta_info"]["output_token_logprobs"]
            if len(scores) != 33 or not all(math.isfinite(row[0]) for row in scores):
                raise AssertionError(
                    "load smoke did not complete 33 tokens with finite logprobs"
                )
            outputs.append(output)
            save(out / "partial.json", {"outputs": outputs})
        token_ids = lambda output: [
            row[1] for row in output["meta_info"]["output_token_logprobs"]
        ]
        if token_ids(outputs[0]) != token_ids(outputs[2]):
            raise AssertionError("slot reuse changed greedy output")
        completion = request(
            endpoint,
            "/v1/completions",
            {
                "model": "lightning",
                "prompt": "One plus two is",
                "temperature": 0,
                "max_tokens": 4,
            },
        )
        if not completion["choices"][0]["text"].strip():
            raise AssertionError("empty OpenAI completion")
        return {
            "server": metadata,
            "outputs": outputs,
            "completion": completion,
            "slot_reuse_ids_equal": True,
            "prune_boundaries_exercised": [16, 32],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=["load"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--duet", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    record = {
        "cell": args.cell,
        "status": "running",
        "started_at": now(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    begin = time.monotonic()
    try:
        record.update(load_cell(args, args.out))
        record["status"] = "pass"
    except Exception as exc:
        record.update(status="fail", error=f"{type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
    finally:
        record.update(finished_at=now(), wall_seconds=time.monotonic() - begin)
        save(args.out / "result.json", record)
        print(
            json.dumps(
                {
                    k: record.get(k)
                    for k in ("cell", "status", "job_id", "wall_seconds", "error")
                }
            ),
            flush=True,
        )
    return int(record["status"] != "pass")


if __name__ == "__main__":
    sys.exit(main())
