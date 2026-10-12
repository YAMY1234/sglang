"""benchmark/duet_eval/parity_verdict.py on synthetic score files (stdlib only, no GPU): the gate compares the served DUET - served
base difference with the reference DUET - base difference, so an offset shared by both served arms cancels and an offset of the
DUET arm alone does not; the per-arm served-vs-reference numbers are kept beside the gate as information.

Run from this directory:  python -m unittest test_parity_verdict
"""
import contextlib
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from benchmark.duet_eval import parity_verdict  # noqa: E402

N = 32


def scores(values):
    return {"mean_nll": sum(values) / len(values), "per_window": list(values)}


def served(values):
    return {"mean_nll": sum(values) / len(values), "per_window_nll": list(values)}


def run(served_duet_shift, served_base_shift, window_noise=0.0002, seed=0):
    """Reference base b_i, reference DUET b_i + 0.001 (two passes 0.0002 apart), served arms = reference + shift + window noise."""
    rng = random.Random(seed)
    base = [1.8 + 0.2 * rng.random() for _ in range(N)]
    duet = [b + 0.001 for b in base]
    ref = [scores(duet), scores([x + 0.0002 for x in duet])]
    out = {}
    for mode in parity_verdict.MODES:
        out[("duet", mode)] = served([x + served_duet_shift + rng.gauss(0, window_noise) for x in duet])
        out[("stock", mode)] = served([x + served_base_shift + rng.gauss(0, window_noise) for x in base])
    return ref, scores(base), out


class Gate(unittest.TestCase):
    def test_offset_shared_by_both_served_arms_cancels(self):
        v = parity_verdict.verdict(*run(0.0025, 0.0025))
        self.assertFalse(v["per_arm_rule_passed"])                 # each arm alone is 0.0025 off: the old gate failed
        self.assertEqual(v["duet-seq"]["per_arm_rule"], "FAIL")
        self.assertEqual(v["stock-seq"]["per_arm_rule"], "FAIL")
        self.assertTrue(v["passed"])
        self.assertEqual(v["difference-seq"]["verdict"], "PASS (absolute)")
        self.assertAlmostEqual(v["difference-seq"]["served"], 0.001 + 0.0001, delta=0.0002)

    def test_offset_of_the_duet_arm_alone_fails(self):
        v = parity_verdict.verdict(*run(0.004, 0.0))
        self.assertFalse(v["passed"])
        self.assertEqual(v["difference-conc16"]["verdict"], "FAIL")
        self.assertEqual(v["stock-seq"]["per_arm_rule"], "PASS (absolute)")

    def test_noisy_windows_pass_statistically(self):
        v = parity_verdict.verdict(*run(0.003, 0.0, window_noise=0.02))
        self.assertTrue(v["passed"])
        self.assertIn(v["difference-seq"]["verdict"], ("PASS (statistical)", "PASS (absolute)"))
        self.assertFalse(v["difference-seq"]["absolute"])

    def test_threshold_is_the_reference_noise_when_larger(self):
        ref, base, srv = run(0.0, 0.0)
        ref[1] = scores([x + 0.005 for x in ref[0]["per_window"]])
        v = parity_verdict.verdict(ref, base, srv)
        self.assertAlmostEqual(v["threshold"], 0.005)
        self.assertAlmostEqual(v["reference_noise"], 0.005)

    def test_window_count_mismatch_is_refused(self):
        ref, base, srv = run(0.0, 0.0)
        srv[("stock", "seq")]["per_window_nll"].pop()
        with self.assertRaises(ValueError):
            parity_verdict.verdict(ref, base, srv)

    def test_main_reads_the_job_layout_and_writes_the_verdict(self):
        ref, base, srv = run(0.0025, 0.0025)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "ref").mkdir()
            (out / "ref/reference.json").write_text(json.dumps({"complete": True, "passes": ref}))
            (out / "ref/reference-base.json").write_text(json.dumps({"complete": True, "passes": [base]}))
            for (arm, mode), s in srv.items():
                (out / f"nll-{arm}-{mode}.json").write_text(json.dumps(s))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                parity_verdict.main([str(out)])
            written = json.loads((out / "verdict.json").read_text())
        self.assertTrue(written["passed"])
        self.assertEqual(buf.getvalue().strip().splitlines()[-1], "PARITY PASS")
        self.assertIn("[FAIL]", buf.getvalue())                    # the per-arm lines are still printed


if __name__ == "__main__":
    unittest.main()
