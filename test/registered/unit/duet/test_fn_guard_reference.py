"""CPU checks of benchmark/duet_eval/flash_next/guard_reference.py, the reference side of validate_fn_parity_job.sh: its scoring
helpers are the twinstar source they were copied from (nvfp4_duet_guard.py @ e91d2a7a6), the guard's 32 registered windows are
enforced, and reference_pass records the forced NLL of every position in the form parity_verdict.py reads.

DUET_GUARD_ORIGINAL=<twinstar>/twinstar/bench/nvfp4_duet_guard.py (git show e91d2a7a6:twinstar/bench/nvfp4_duet_guard.py) also
runs the original on the same fake model and compares.

Run from this directory:  python -m unittest test_fn_guard_reference
"""
import ast
import contextlib
import hashlib
import importlib.util
import io
import os
import struct
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]   # test/registered/unit/duet -> repo root
VENDORED = ROOT / "benchmark/duet_eval/flash_next/guard_reference.py"
# sha256 (first 16 hex digits) of each copied function's source text in twinstar nvfp4_duet_guard.py @ e91d2a7a6
SOURCE_E91D2A7A6 = {"save": "6300dcf0a2e5de71", "identity": "792c49e6d94b1790", "check_cases": "6a8c64e0963dd1f1",
                    "score_summary": "e4343c9476cb1751", "reference_pass": "33c23e8907b5f7f8"}
VOCAB = 64


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function_sources(path):
    text = Path(path).read_text()
    lines = text.split("\n")
    return {n.name: "\n".join(lines[n.lineno - 1:n.end_lineno]) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}


def quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):   # reference_pass prints one REFERENCE line per window
        return fn(*args)


def manifest(n_cases=32):
    g = torch.Generator().manual_seed(0)
    return {"cases": [{"window": i, "input_ids": torch.randint(VOCAB, (3584,), generator=g).tolist(),
                       "targets": torch.randint(VOCAB, (256,), generator=g).tolist()} for i in range(n_cases)]}


class FakeModel:
    """twinstar's generate_batch contract under forcing: one (1, vocab) logits row per forced step into logits_out."""

    def __init__(self, drop_last=False):
        self.drop_last = drop_last

    @staticmethod
    def logits(prompt, step):
        g = torch.Generator().manual_seed(len(prompt) * 1000 + sum(prompt[:8]) * 7 + step)
        return torch.randn(1, VOCAB, generator=g)

    def generate_batch(self, prompts, max_new_tokens, forced, logits_out):
        for step in range(max_new_tokens - (1 if self.drop_last else 0)):
            logits_out.append(self.logits(prompts[0], step))
        return [list(forced[0])]


class GuardReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = load(VENDORED, "guard_reference_under_test")
        cls.cases = cls.guard.check_cases(manifest())

    def test_helpers_are_the_copied_source(self):
        sources = function_sources(VENDORED)
        got = {name: hashlib.sha256(sources[name].encode()).hexdigest()[:16] for name in SOURCE_E91D2A7A6}
        self.assertEqual(got, SOURCE_E91D2A7A6)

    def test_check_cases_requires_the_32_registered_windows(self):
        self.assertEqual([c["window"] for c in self.guard.check_cases(manifest(33))], list(range(32)))
        bad = [manifest(31)]
        for key, fix in (("window", lambda c: 5), ("input_ids", lambda c: c["input_ids"][:-1]), ("targets", lambda c: c["targets"] + [0])):
            m = manifest()
            m["cases"][3][key] = fix(m["cases"][3])
            bad.append(m)
        for m in bad:
            with self.assertRaises(ValueError):
                self.guard.check_cases(m)

    def test_reference_pass_scores_every_forced_position(self):
        res = quiet(self.guard.reference_pass, FakeModel(), self.cases)
        rows = res["rows"]
        self.assertEqual([(r["window"], r["step"], r["token"]) for r in rows],
                         [(c["window"], s, t) for c in self.cases for s, t in enumerate(c["targets"])])
        for r in rows[::997]:
            case = self.cases[r["window"]]
            expected = -torch.log_softmax(FakeModel.logits(case["input_ids"], r["step"])[0], -1)[r["token"]].item()
            self.assertAlmostEqual(r["nll"], expected, places=6)
        # parity_verdict reads mean_nll and per_window (one value per window, in window order)
        self.assertAlmostEqual(res["mean_nll"], sum(r["nll"] for r in rows) / len(rows), places=9)
        self.assertEqual(len(res["per_window"]), 32)
        self.assertAlmostEqual(res["per_window"][7], sum(r["nll"] for r in rows if r["window"] == 7) / 256, places=9)
        tokens = self.cases[0]["input_ids"] + self.cases[0]["targets"]
        self.assertEqual(res["window_hashes"][0], hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest())
        with self.assertRaises(RuntimeError):
            quiet(self.guard.reference_pass, FakeModel(drop_last=True), self.cases[:1])

    @unittest.skipUnless(os.environ.get("DUET_GUARD_ORIGINAL"), "DUET_GUARD_ORIGINAL not set")
    def test_same_results_as_the_original(self):
        original = load(os.environ["DUET_GUARD_ORIGINAL"], "nvfp4_duet_guard_original")
        sources = function_sources(os.environ["DUET_GUARD_ORIGINAL"])
        self.assertEqual({name: hashlib.sha256(sources[name].encode()).hexdigest()[:16] for name in SOURCE_E91D2A7A6},
                         SOURCE_E91D2A7A6)
        self.assertEqual(original.check_cases(manifest(33)), self.guard.check_cases(manifest(33)))
        mine, theirs = (quiet(m.reference_pass, FakeModel(), self.cases) for m in (self.guard, original))
        for res in (mine, theirs):
            res.pop("window_timings")
        self.assertEqual(mine, theirs)


if __name__ == "__main__":
    unittest.main()
