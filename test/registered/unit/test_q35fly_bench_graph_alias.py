"""CPU-only contracts for the bench graph alias; imports no torch/CUDA."""
import ast
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang/srt/model_executor/runner_backend"
E = "SGLANG_R14_KSMALL_E6V2"
F = "SGLANG_Q35FLY_MKA_HOP_FP8"
LABELS = ("N0", "N1", "N2", "N3")


def load_helper(variants, alias):
    fake_env = types.ModuleType("sglang.srt.environ")
    fake_env.envs = types.SimpleNamespace(
        SGLANG_TRTL17_PAD_AB=types.SimpleNamespace(get=lambda: ())
    )
    with patch.dict(sys.modules, {"sglang.srt.environ": fake_env}):
        with patch.dict(os.environ, {
            "Q35MF_AB_VARIANTS": json.dumps(variants),
            "Q35MF_AB_PATTERN": ",".join(variants),
            "Q35MF_AB_BENCH_ALIAS_GRAPH": str(alias),
        }):
            spec = importlib.util.spec_from_file_location("alias_test_ab", SOURCE / "ab_variants.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module


def backend_methods(helper):
    tree = ast.parse((SOURCE / "full_cuda_graph_backend.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "FullCudaGraphBackend")
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body += [n for n in cls.body if isinstance(n, ast.FunctionDef)
             and n.name in ("capture_one", "cleanup")]
    ns = {"ab_variants": helper, "logger": logging.getLogger(__name__)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 str(SOURCE / "full_cuda_graph_backend.py"), "exec"), ns)
    return ns


class Graph:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


class Recorder:
    def __init__(self):
        self._graphs, self._outputs = {}, {}
        self._output_buffer, self._pool = object(), object()
        self.calls = []

    def _capture_graph(self, key, fn, hook):
        self.calls.append((key, os.environ.get(E), os.environ.get(F)))
        fn()
        self._graphs[key], self._outputs[key] = Graph(), object()


class TestBenchGraphAlias(unittest.TestCase):
    def test_alias_default_and_restore(self):
        variants = {name: {E: "0", F: "0"} for name in LABELS}
        for alias in (0, 1):
            with self.subTest(alias=alias), patch.dict(os.environ, {E: "1"}):
                os.environ.pop(F, None)
                ab = load_helper(variants, alias)
                methods, record = backend_methods(ab), Recorder()
                shapes = [(phase, b) for phase in ("verify", "draft", "extend") for b in (1, 2, 8)]
                for shape in shapes:
                    methods["capture_one"](record, shape, lambda: None)
                    assert os.environ[E] == "1" and F not in os.environ
                    graphs = [record._graphs[(name, shape)] for name in LABELS]
                    outputs = [record._outputs[(name, shape)] for name in LABELS]
                    self.assertEqual(len({id(g) for g in graphs}), 1 if alias else 4)
                    self.assertEqual(len({id(g) for g in outputs}), 1 if alias else 4)
                self.assertEqual(len(record.calls), len(shapes) * (1 if alias else 4))
                self.assertTrue(all(e == f == "0" for _, e, f in record.calls))
                graphs = list(record._graphs.values())
                methods["cleanup"](record)
                self.assertTrue(all(g.resets == 1 for g in graphs))
                self.assertFalse(record._graphs or record._outputs)
                self.assertIsNone(record._pool)

    def test_restore_after_capture_exception(self):
        for alias in (0, 1):
            with self.subTest(alias=alias), patch.dict(os.environ, {E: "1", F: "1"}):
                ab = load_helper({n: {E: "0", F: "0"} for n in LABELS}, alias)
                def fail():
                    raise RuntimeError("capture failed")
                with self.assertRaisesRegex(RuntimeError, "capture failed"):
                    backend_methods(ab)["capture_one"](Recorder(), ("verify", 1), fail)
                self.assertEqual((os.environ[E], os.environ[F]), ("1", "1"))

    def test_reject_nonnull_alias(self):
        for variants in ({}, {"N0": {}}, {"N0": {F: "1"}},
                         {"N0": {F: "0"}, "N1": {E: "0"}}):
            with self.subTest(variants=variants):
                ab = load_helper(variants, 1)
                with self.assertRaises(ValueError):
                    backend_methods(ab)["capture_one"](Recorder(), 1, lambda: None)

    def test_default_candidate_and_no_variant_capture(self):
        variants = {n: {F: "1" if i in (1, 2) else "0"} for i, n in enumerate(LABELS)}
        ab, record = load_helper(variants, 0), Recorder()
        backend_methods(ab)["capture_one"](record, 1, lambda: None)
        self.assertEqual([x[2] for x in record.calls], ["0", "1", "1", "0"])
        ab, record = load_helper({}, 0), Recorder()
        backend_methods(ab)["capture_one"](record, 1, lambda: None)
        self.assertEqual(len(record.calls), 1)
        self.assertEqual(record.calls[0][0], 1)


if __name__ == "__main__":
    unittest.main()
