#!/usr/bin/env python3
"""Execute the frozen core serializer and endpoint with CPU-only mocks.
No model imports, accelerator discovery, torch, CUDA, or server processes.
"""

import argparse
import ast
import asyncio
import datetime
import json
import pathlib
import subprocess
import time
from types import SimpleNamespace
from typing import Optional

REVISION = "7a841e4189445d108085423c3d6054bd9a7b486c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def source(path):
        return subprocess.check_output(
            ["git", "-C", args.sglang_root, "show", f"{REVISION}:{path}"], text=True
        )

    snapshot = ast.parse(source("python/sglang/srt/managers/load_snapshot.py"))
    cls = next(
        n
        for n in snapshot.body
        if isinstance(n, ast.ClassDef) and n.name == "LoadSnapshot"
    )
    attrs = [n for n in cls.body if isinstance(n, ast.AnnAssign)]
    core_keys = tuple(
        n.target.id
        for n in attrs
        if n.target.id
        not in {"memory", "speculative", "lora", "disaggregation", "queues"}
    )
    # Keep defaults and the exact frozen to_dict method; replace only the msgspec
    # base with object so constructing snapshots needs no third-party imports.
    body = [ast.Assign(targets=[n.target], value=n.value) for n in attrs]
    body += [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.Assign))]
    mock_cls = ast.ClassDef(
        name="FrozenCore", bases=[], keywords=[], body=body, decorator_list=[]
    )
    env = {"Optional": Optional, "_CORE_KEYS": core_keys}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[mock_cls], type_ignores=[])),
            "frozen-load-snapshot-core",
            "exec",
        ),
        env,
    )
    snapshots = []
    for rank, score in enumerate([17, 43]):
        obj = env["FrozenCore"]()
        obj.dp_rank = rank
        obj.timestamp = 1791675600.0
        obj.num_total_tokens = score
        obj.num_used_tokens = score
        obj.num_active_tokens = score
        obj.max_total_num_tokens = 1000
        snapshots.append(obj)

    class Tokenizer:
        async def get_loads(self, **kwargs):
            assert kwargs == {"include": ["core"], "dp_rank": None}
            return snapshots

    class Clock:
        @staticmethod
        def now(tz):
            return datetime.datetime(2026, 10, 10, 21, 0, 0, tzinfo=tz)

    endpoint = ast.parse(source("python/sglang/srt/entrypoints/v1_loads.py"))
    functions = [
        n
        for n in endpoint.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name in {"get_loads", "_num_accelerators_per_dp_rank"}
    ]
    for n in functions:
        n.decorator_list = []
    env.update(
        {
            "time": time,
            "datetime": Clock,
            "timezone": datetime.timezone,
            "Depends": lambda f: None,
            "_get_tokenizer_manager": lambda: None,
            "_accelerator_name": lambda: None,
            "__version__": "synthetic-cpu-fixture",
            "get_parallel": lambda: SimpleNamespace(
                tp_size=2, pp_size=1, dp_size=2, enable_dp_attention=True
            ),
        }
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
            "frozen-v1-loads",
            "exec",
        ),
        env,
    )
    payload = asyncio.run(
        env["get_loads"](include="core", tokenizer_manager=Tokenizer())
    )
    assert (
        "aggregate" not in payload
        and sum(x["num_total_tokens"] for x in payload["loads"]) == 60
    )
    pathlib.Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"PASS frozen {REVISION} endpoint and LoadSnapshot.to_dict(core): loads[] scores [17,43], total 60; old aggregate-only lookup absent"
    )


if __name__ == "__main__":
    main()
