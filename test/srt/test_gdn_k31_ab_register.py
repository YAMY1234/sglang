"""K4.2 ab_measure register file (docs/219 §8): wiring, idempotence, latch values. CPU only, no torch."""
import logging
import runpy
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / "test/srt/ab/ws3_k42_register.py"


class RegisterFileTest(unittest.TestCase):
    def setUp(self):
        # Stub module exposing the same set_k42_route semantics as gdn_prefill_agg_contract.
        self.state = {"value": "unset", "calls": 0}
        stub = types.ModuleType("sglang.srt.mem_cache.gdn_prefill_agg_contract")

        def set_k42_route(route):
            assert route in ("forward", None)
            self.state["value"] = route
            self.state["calls"] += 1
        stub.set_k42_route = set_k42_route
        self.saved = sys.modules.get(stub.__name__)
        sys.modules[stub.__name__] = stub

    def tearDown(self):
        if self.saved is None:
            sys.modules.pop("sglang.srt.mem_cache.gdn_prefill_agg_contract", None)
        else:
            sys.modules["sglang.srt.mem_cache.gdn_prefill_agg_contract"] = self.saved

    def test_registers_prefill_scope_and_flips_route(self):
        seen = []
        runpy.run_path(str(REGISTER), init_globals=dict(register=lambda *a, **k: seen.append((a, k))))
        self.assertEqual(len(seen), 1)
        (name, apply_a, apply_b), kwargs = seen[0]
        self.assertEqual(name, "k42_publication_route")
        self.assertEqual(kwargs, {"scope": "prefill"})
        for _ in range(3):  # idempotent
            apply_a()
            self.assertEqual(self.state["value"], "forward")
            apply_b()
            self.assertIsNone(self.state["value"])

    def test_real_setter_source_latches_and_logs_once(self):
        # The real setter is plain Python; check its contract from source without importing torch.
        src = (ROOT / "python/sglang/srt/mem_cache/gdn_prefill_agg_contract.py").read_text()
        start = src.index("_AB_ROUTE = {")
        end = src.index("def _control_route")
        scope = {"logger": logging.getLogger("t")}
        exec(src[start:end], scope)
        with self.assertLogs("t", "INFO") as logs:
            scope["set_k42_route"]("forward")
            scope["set_k42_route"]("forward")
            scope["set_k42_route"](None)
            scope["set_k42_route"](None)
        self.assertEqual(len(logs.output), 2)  # one activation line per label
        self.assertIsNone(scope["_AB_ROUTE"]["value"])
        with self.assertRaises(ValueError):
            scope["set_k42_route"]("side")
        self.assertIn('if _AB_ROUTE["value"] == "forward":\n            route = None', src)


class DualPrewarmHelpersTest(unittest.TestCase):
    """set_k31_ab_label / k31_ab_graph / capture_dual, read from source without importing torch."""

    def _scope(self):
        src = (ROOT / "python/sglang/srt/mem_cache/gdn_prefill_agg_contract.py").read_text()
        start = src.index('K31_AB_DUAL = "SGLANG_GDN_K31_AB_DUAL"')
        end = src.index("def _control_route")
        import os as _os
        scope = {"os": _os, "logger": logging.getLogger("t2")}
        exec(src[start:end], scope)
        return scope

    def test_label_switches_graph_and_env(self):
        import os
        sc = self._scope()
        saved = os.environ.pop("SGLANG_GDN_K31_FUSED", None)
        try:
            dual = {"A": "gA", "B": "gB"}
            self.assertEqual(sc["k31_ab_graph"]("dflt", dual), "dflt")   # no label yet
            sc["set_k31_ab_label"]("B")
            self.assertEqual(sc["k31_ab_graph"]("dflt", dual), "gB")
            self.assertEqual(os.environ["SGLANG_GDN_K31_FUSED"], "1")
            sc["set_k31_ab_label"]("A")
            self.assertEqual(sc["k31_ab_graph"]("dflt", dual), "gA")
            self.assertEqual(os.environ["SGLANG_GDN_K31_FUSED"], "0")
            self.assertEqual(sc["k31_ab_graph"]("dflt", None), "dflt")
            with self.assertRaises(ValueError):
                sc["set_k31_ab_label"]("C")
        finally:
            os.environ.pop("SGLANG_GDN_K31_FUSED", None)
            if saved is not None:
                os.environ["SGLANG_GDN_K31_FUSED"] = saved

    def test_capture_dual_toggles_and_restores(self):
        import os
        sc = self._scope()
        os.environ["SGLANG_GDN_K31_FUSED"] = "keep"
        try:
            seen = []
            out = sc["capture_dual"](lambda: seen.append(os.environ["SGLANG_GDN_K31_FUSED"]) or len(seen))
            self.assertEqual(seen, ["0", "1"])
            self.assertEqual(out, {"A": 1, "B": 2})
            self.assertEqual(os.environ["SGLANG_GDN_K31_FUSED"], "keep")
        finally:
            os.environ.pop("SGLANG_GDN_K31_FUSED", None)

    def test_register_file(self):
        stub = types.ModuleType("sglang.srt.mem_cache.gdn_prefill_agg_contract")
        calls = []
        stub.set_k31_ab_label = calls.append
        saved = sys.modules.get(stub.__name__)
        sys.modules[stub.__name__] = stub
        try:
            seen = []
            runpy.run_path(str(ROOT / "test/srt/ab/ws3_ab_register.py"),
                           init_globals=dict(register=lambda *a, **k: seen.append((a, k))))
            (name, apply_a, apply_b), kwargs = seen[0]
            self.assertEqual((name, kwargs), ("k31_ab", {"scope": "prefill"}))
            apply_b(); apply_a()
            self.assertEqual(calls, ["B", "A"])
        finally:
            if saved is None:
                sys.modules.pop(stub.__name__, None)
            else:
                sys.modules[stub.__name__] = saved


if __name__ == "__main__":
    unittest.main()
