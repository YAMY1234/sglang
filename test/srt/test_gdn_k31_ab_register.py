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


class RestoreRegisterTest(unittest.TestCase):
    def test_restore_register_flips_env(self):
        import os
        seen = []
        saved = os.environ.get("SGLANG_GDN_PREFIX_RESTORE_FUSED")
        try:
            runpy.run_path(str(ROOT / "test/srt/ab/ws3_restore_register.py"),
                           init_globals=dict(register=lambda *a, **k: seen.append((a, k))))
            (name, apply_a, apply_b), kwargs = seen[0]
            self.assertEqual((name, kwargs), ("prefix_restore_fused", {"scope": "prefill"}))
            apply_b(); self.assertEqual(os.environ["SGLANG_GDN_PREFIX_RESTORE_FUSED"], "1")
            apply_a(); self.assertEqual(os.environ["SGLANG_GDN_PREFIX_RESTORE_FUSED"], "0")
        finally:
            if saved is None:
                os.environ.pop("SGLANG_GDN_PREFIX_RESTORE_FUSED", None)
            else:
                os.environ["SGLANG_GDN_PREFIX_RESTORE_FUSED"] = saved


if __name__ == "__main__":
    unittest.main()
