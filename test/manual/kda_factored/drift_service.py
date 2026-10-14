"""Generate 48 native dense replies while test-only KDA shadows record 1024 steps."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

root = Path(os.environ["OUT"])
root.mkdir(parents=True, exist_ok=True)
source = Path(os.environ["KDA_SOURCE"])
args = json.loads((Path(os.environ["KDA_BUNDLE"]) / "base-argv.json").read_text())


def option(name, value):
    if name in args:
        args[args.index(name) + 1] = str(value)
    else:
        args.extend([name, str(value)])


option("--port", 30500)
option("--max-running-requests", 48)
option(
    "--max-mamba-cache-size",
    max(64, int(args[args.index("--max-mamba-cache-size") + 1])),
)
option("--mamba-ssm-dtype", "float32")
args += ["--disable-cuda-graph"]
env = dict(os.environ)
env.update(
    KDA_DRIFT_SHADOW="1",
    KDA_DRIFT_ROOT=str(root),
    KDA_DRIFT_COMPONENTS=str(Path(env["KDA_RELEASE"]) / "duet_components.safetensors"),
    SGLANG_KDA_FACTORED_STATE="0",
)
env["PYTHONPATH"] = (
    str(source / "test/manual/kda_factored/shadow_site")
    + ":"
    + str(source / "test/manual/kda_factored")
    + ":"
    + env["PYTHONPATH"]
)
command = [sys.executable, "-m", "sglang.launch_server", *args]
topics = [
    "the water cycle",
    "photosynthesis",
    "the history of navigation",
    "prime numbers",
    "how computers represent text",
    "the causes of ocean tides",
    "urban public transport",
    "how libraries organize knowledge",
]
styles = [
    "a detailed tutorial",
    "a dialogue between a teacher and a student",
    "an illustrated explanation using words only",
    "a historical essay",
    "a sequence of worked examples",
    "a careful comparison of common misconceptions",
]
prompts = [
    f"Write {style} about {topic}. Develop the explanation over at least 1500 words, with concrete examples and connected paragraphs. Begin the response now."
    for style in styles
    for topic in topics
]
body = {
    "text": prompts,
    "sampling_params": {"temperature": 0, "max_new_tokens": 1025, "ignore_eos": True},
    "return_logprob": True,
    "top_logprobs_num": 1,
}
(root / "inputs.json").write_text(json.dumps(body, indent=2))
receipt = {
    "family": os.environ["FAMILY"],
    "job": os.environ["SLURM_JOB_ID"],
    "source_commit": os.environ["KDA_SOURCE_COMMIT"],
    "command": command,
    "input_sha256": hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()
    ).hexdigest(),
    "reference": "original model, exact dense FP32 recurrent state, no DUET periodic projection",
    "scope": "48 self-generated replies; first output comes from prefill, next 1024 from decode; shadow state/output drift only",
}
(root / "configuration.json").write_text(json.dumps(receipt, indent=2))
url = "http://127.0.0.1:30500"
start = time.monotonic()
with (root / "server.log").open("w") as log:
    process = subprocess.Popen(
        command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    try:
        while time.monotonic() - start < 1300:
            if process.poll() is not None:
                raise RuntimeError(f"server exit {process.returncode}")
            try:
                with urllib.request.urlopen(url + "/health", timeout=5):
                    break
            except OSError:
                time.sleep(2)
        else:
            raise TimeoutError("health timeout")
        receipt["health_seconds"] = time.monotonic() - start
        print(json.dumps(receipt), flush=True)
        (root / "armed").touch()
        request = urllib.request.Request(
            url + "/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t = time.monotonic()
        with urllib.request.urlopen(request, timeout=1800) as response:
            replies = json.load(response)
        (root / "replies.json").write_text(json.dumps(replies, indent=2))
        assert len(replies) == 48
        assert all(x["meta_info"]["completion_tokens"] == 1025 for x in replies)
        receipt["generation_seconds"] = time.monotonic() - t
        expected = 20 if os.environ["FAMILY"] == "kimi" else 34 * 4
        complete = list(root.glob("tp*/layer*/complete.json"))
        assert len(complete) == expected, (len(complete), expected)
        receipt["complete"] = True
        receipt["layer_rank_traces"] = expected
        (root / "complete.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
