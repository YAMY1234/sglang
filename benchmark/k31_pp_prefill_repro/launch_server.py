"""Exec one node's `sglang.launch_server` for a prefill arm from arms.json (written by make_arms.py).

usage: launch_server.py ARMS_JSON ARM NODE_RANK DIST_INIT_ADDR PORT
env:   MODEL_PATH (required), DRAFT_MODEL_PATH (required; replaces @DRAFT_MODEL_PATH@), NNODES (default 2)
       NSYS=<nsys binary> and NSYS_SESSION=<name>: run under `nsys launch` (an interactive session that records
       nothing until `nsys start --session=<name>`, see profile/nsys_window.sh). Server argv and env are unchanged.
Bools: true -> --flag, false -> omitted; lists -> several values.
"""
import json
import os
import sys

arms_json, arm_name, rank, addr, port = sys.argv[1:6]
arm = json.load(open(arms_json))[arm_name]
model, draft = os.environ["MODEL_PATH"], os.environ["DRAFT_MODEL_PATH"]
argv = ["python3", "-m", "sglang.launch_server", "--model-path", model, "--host", "0.0.0.0", "--port", port,
        "--dist-init-addr", addr, "--nnodes", os.environ.get("NNODES", "2"), "--node-rank", rank,
        "--disaggregation-mode", "prefill"]
for k, v in arm["args"].items():
    if v is False or v is None:
        continue
    argv.append(f"--{k}")
    if v is True:
        continue
    vals = v if isinstance(v, list) else [v]
    argv += [str(x).replace("@DRAFT_MODEL_PATH@", draft) for x in vals]
env = {**os.environ, **arm["env"]}
print("LAUNCH", json.dumps({"arm": arm_name, "argv": argv, "env": arm["env"]}), flush=True)
nsys = os.environ.get("NSYS")
if nsys:
    pre = [nsys, "launch", f"--session-new={os.environ['NSYS_SESSION']}", "--trace=cuda,nvtx,osrt"]
    print("NSYS_LAUNCH", json.dumps(pre), flush=True)
    argv = pre + argv
os.execvpe(argv[0], argv, env)
