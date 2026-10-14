"""Run the sealed docs/197 worker with a same-rack pre-client pair barrier."""

import ast
import hashlib
import json
import os
import time
from pathlib import Path

bundle = Path(os.environ["KDA_BUNDLE"])
path = bundle / "perf_runner.py"
source = path.read_bytes()
tree = ast.parse(source, filename=str(path))
cut = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Try))
namespace = {"__name__": "kdaf_sealed_perf", "__file__": str(path)}
exec(  # noqa: S102 — execute only the pinned, sealed local driver definitions
    compile(ast.Module(body=tree.body[:cut], type_ignores=[]), str(path), "exec"),
    namespace,
)
arm = "original" if os.environ["SLURM_PROCID"] == "0" else "seed"
out = Path(os.environ["KDA_OUT"])
pair = Path(os.environ["KDA_PAIR_ROOT"])
pair.mkdir(parents=True, exist_ok=True)
receipt = {
    "arm": arm,
    "node": os.environ["SLURMD_NODENAME"],
    "job": os.environ["SLURM_JOB_ID"],
    "source_manifest": os.environ["KDA_SOURCE_MANIFEST"],
    "source_commit": os.environ["KDA_SOURCE_COMMIT"],
    "sealed_driver_sha256": hashlib.sha256(source).hexdigest(),
    "sealed_worker_ast_sha256": hashlib.sha256(
        ast.dump(
            next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "worker"
            )
        ).encode()
    ).hexdigest(),
    "adaptation": "single arm per node; barrier before flush/cache + client; frozen worker/client/reducer function bodies unchanged",
}
original_http = namespace["http"]


def paired_http(url, path, payload=None):
    if path == "/flush_cache":
        mine = pair / f"{arm}-ready.json"
        ready = dict(receipt, ready_ns=time.time_ns())
        temporary = mine.with_suffix(".tmp")
        temporary.write_text(json.dumps(ready, indent=2))
        temporary.replace(mine)
        other = pair / (
            "seed-ready.json" if arm == "original" else "original-ready.json"
        )
        deadline = time.monotonic() + 850
        while not other.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("paired C32 server did not finish model warmup")
            time.sleep(1)
        peer = json.loads(other.read_text())
        assert peer["source_manifest"] == ready["source_manifest"]
        assert peer["node"] != ready["node"]
        assert peer["node"].split("-T")[0] == ready["node"].split("-T")[0], (
            peer,
            ready,
        )
        receipt["client_barrier_release_ns"] = time.time_ns()
        (out / "actual-build.json").write_text(json.dumps(receipt, indent=2))
    return original_http(url, path, payload)


namespace["http"] = paired_http
rows = namespace["worker"]((0, arm))
assert len(rows) == 1 and rows[0]["C"] == 32 and rows[0]["complete"]
namespace["write"](out / "cells.json", rows)
namespace["write"](
    out / "complete.json",
    {
        "complete": True,
        "window_seconds": 900,
        "arm": arm,
        "source_commit": os.environ["KDA_SOURCE_COMMIT"],
        "measurement_gate": "REVIEW_REQUIRED"
        if list(out.rglob("*review-required*"))
        or list(out.rglob("control-failures.jsonl"))
        else "EVIDENCE_COMPLETE",
        "legacy_build_and_state_labels": "sealed raw reducer labels are historical; actual-build and native pool allocation logs are authoritative",
    },
)
