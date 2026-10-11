#!/usr/bin/env python3
"""Run real msgpack/SHM and tokenizer/endpoint bodies without torch or a GPU.

Only runtime-context/environment/network imports and accelerator metadata are
stubbed. Production slot classes, get_loads method and endpoint body execute.
--revision verifies the same tests against the frozen git object.
"""


# ruff: noqa: S102 -- execute trusted repository AST bodies for GPU-free checks.

import argparse
import ast
import asyncio
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("--revision")
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]


def source(path):
    if args.revision:
        return subprocess.check_output(
            ["git", "show", f"{args.revision}:{path}"], cwd=root, text=True
        )
    return (root / path).read_text()


path = "python/sglang/srt/managers/load_snapshot.py"
tree = ast.parse(source(path))
tree.body = [
    n
    for n in tree.body
    if not (isinstance(n, ast.ImportFrom) and (n.module or "").startswith("sglang."))
]
module = types.ModuleType("sglang.srt.managers.load_snapshot")
module.__dict__.update(
    get_parallel=lambda: SimpleNamespace(
        dp_size=1, tp_size=2, pp_size=1, enable_dp_attention=False
    ),
    get_serving=lambda: None,
    envs=None,
    is_zmq_endpoint_ipv6=lambda _: False,
)
sys.modules[module.__name__] = module
exec(compile(tree, path, "exec"), module.__dict__)
ns = module.__dict__

# Execute the requested unit tests against the actual writer/reader classes.
test_ns = {"__name__": "cpu_load_recovery_tests"}
exec(
    (root / "test/registered/unit/managers/test_load_snapshot_recovery.py").read_text(),
    test_ns,
)
suite = unittest.defaultTestLoader.loadTestsFromTestCase(
    test_ns["TestLoadSnapshotRecovery"]
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
    sys.exit(1)

control = ast.parse(source("python/sglang/srt/managers/tokenizer_control_mixin.py"))
cls = next(
    n
    for n in control.body
    if isinstance(n, ast.ClassDef) and n.name == "TokenizerControlMixin"
)
method = next(
    n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "get_loads"
)
# Avoid importing tokenizer generation/device dependencies; retain its real body.
for arg in method.args.args:
    arg.annotation = None
method.returns = None
exec(
    compile(ast.Module(body=[method], type_ignores=[]), "tokenizer.get_loads", "exec"),
    ns,
)
endpoint = ast.parse(source("python/sglang/srt/entrypoints/v1_loads.py"))
fn = next(
    n
    for n in endpoint.body
    if isinstance(n, ast.AsyncFunctionDef) and n.name == "get_loads"
)
fn.decorator_list = []
fn.args.defaults = [ast.Constant(None) for _ in fn.args.defaults]
for arg in fn.args.args:
    arg.annotation = None
ns.update(
    __version__="cpu-frozen-contract",
    _accelerator_name=lambda: "CPU stub",
    _num_accelerators_per_dp_rank=lambda *a: 2,
)
exec("import time\nfrom datetime import datetime, timezone", ns)
endpoint_ns = dict(ns)
exec(
    compile(
        ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
        "v1_loads.get_loads",
        "exec",
    ),
    endpoint_ns,
)

with tempfile.TemporaryDirectory() as directory:
    path = str(Path(directory) / "slots")
    writer = ns["ShmLoadSnapshotWriter"](path, 1, 0)
    reader = ns["ShmLoadSnapshotReader"](path, 1)

    class Manager:
        elastic_worker_count = 1
        load_snapshot_reader = reader
        metrics_collector = None
        get_loads = ns["get_loads"]

        def auto_create_handle_loop(self):
            pass

    try:
        for mode in ["prefill", "decode"]:
            for tokens, running in [(128, 3), (0, 0)]:
                Path(path).unlink()
                writer.write(
                    ns["LoadSnapshot"](
                        dp_rank=0,
                        num_total_tokens=tokens,
                        num_running_reqs=running,
                        disaggregation=ns["DisaggregationMetrics"](mode=mode),
                    )
                )
                for include in ["core", "all"]:
                    payload = asyncio.run(
                        endpoint_ns["get_loads"](
                            include=include, tokenizer_manager=Manager()
                        )
                    )
                    assert len(payload["loads"]) == 1, payload
                    assert payload["loads"][0]["num_total_tokens"] == tokens
                    assert payload["loads"][0]["num_running_reqs"] == running
        print(
            "PASS TP2/DP1 prefill+decode, busy+idle, include=core/all: real tokenizer and endpoint bodies return one rank after unlink/republication"
        )
    finally:
        reader.close()
        writer.close()
