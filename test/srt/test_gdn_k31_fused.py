"""Fused prompt-end k31 (SGLANG_GDN_K31_FUSED, docs/223 §9) against factorize_prefill_k31, CPU only.

Runs the Triton kernels under the CPU interpreter (zero GPU). Stage by stage (k31_fused_xcheck.xcheck):
each fused stage against the reference op on the fused stage input; then end to end with the R1 module
gate thresholds. Interpreter agreement proves algorithm and branch semantics, not fp64 FMA contraction.
"""
import os

os.environ["TRITON_INTERPRET"] = "1"
os.environ.setdefault("SGLANG_GDN_K31_EIGH", "jacobi")

import sys  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k31_fused_xcheck as xc  # noqa: E402


class FusedK31Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref, cls.mixed, cls.fused = xc.load_modules()
        cls.fix = torch.load(xc.FIXTURE, weights_only=False)

    def _check_stages(self, out, rows=None):
        st = out["stages"]
        self.assertTrue(out["a_bitwise"])
        self.assertTrue(out["q_pad_zero"])
        self.assertTrue(out["rows_ge_r_zero"])
        self.assertTrue(st["gn_bitwise_given_g64"])
        self.assertTrue(st["jacobi_bitwise_given_g64"])
        for name in ("y0=x@omega", "z=x^T@o1", "y2=x@o2"):
            self.assertLessEqual(st[name]["max"], 1e-6, name)
        for name in ("orth1", "orth2", "orth3"):
            self.assertLessEqual(st[name]["span"]["max"], 1e-6, name)
        self.assertLessEqual(st["gram64"]["max"], 1e-12)
        self.assertLessEqual(st["newton_vectors"]["max"], 1e-12)
        self.assertLessEqual(st["newton_values"], 1e-12)
        self.assertLessEqual(st["U_given_q_wr"]["max"], 1e-3)
        self.assertLessEqual(st["W_given_q_wr"]["max"], 1e-3)
        self.assertEqual(out["fallback_count"]["fused"], out["fallback_count"]["reference_chained"])

    def test_real_states_fixture(self):
        fx = self.fix
        out = xc.xcheck(self.ref, self.mixed, self.fused, fx["dense"][None], fx["vbar"], fx["omega"][None], 12)
        self._check_stages(out)
        e2e = out["end_to_end"]
        # R1 module gate (lead 21:4x item 3) thresholds
        self.assertTrue(e2e["a_bitwise"])
        self.assertEqual(e2e["nonfinite"], 0)
        self.assertLessEqual(e2e["content"]["median"], 3e-5)
        self.assertLessEqual(e2e["content"]["max"], 2.5e-4)
        self.assertLessEqual(e2e["ratio_max"], 1.01)

    def test_fp32_stage_fallback_counts_per_matrix(self):
        # head 3 overflows the jittered fp32 Gram diagonal: both paths fall back to the fp32 input for the
        # fp64 pass and count it; the other heads are untouched.
        fx = self.fix
        dense = fx["dense"].clone()
        dense[3] = dense[3] / dense[3].abs().max() * 1e19
        heads = [2, 3, 4]
        s = dense[heads][None]
        out = xc.xcheck(self.ref, self.mixed, self.fused, s, fx["vbar"][heads], fx["omega"][heads][None], 3)
        self.assertGreaterEqual(out["fallback_count"]["fused"], 1)
        self.assertEqual(out["fallback_count"]["fused"], out["fallback_count"]["reference_chained"])
        self.assertEqual(out["stages"]["orth1"]["fallbacks"], 1)
        self.assertLessEqual(out["stages"]["orth1"]["span"]["max"], 1e-6)
        ok = self.fused.factorize_k31_fused(dense[[2, 4]][None], fx["vbar"][[2, 4]], fx["omega"][[2, 4]][None],
                                            2, 16, 16, torch.float16)
        mixed_batch = self.fused.factorize_k31_fused(s, fx["vbar"][heads], fx["omega"][heads][None], 3, 16, 16,
                                                     torch.float16)
        for alone, together in zip(ok, mixed_batch):
            self.assertTrue(torch.equal(alone, together[:, [0, 2]]))

    def test_geometry_guard(self):
        s = torch.zeros(1, 2, 128, 128)
        vbar = torch.zeros(2, 128)
        omega = torch.zeros(1, 2, 128, 24)
        self.assertTrue(self.fused.fused_supported(s, vbar, omega, 16, 16, torch.float16))
        self.assertFalse(self.fused.fused_supported(s, vbar, omega, 8, 16, torch.float16))
        self.assertFalse(self.fused.fused_supported(s, vbar, torch.zeros(1, 2, 128, 16), 16, 16, torch.float16))
        self.assertFalse(self.fused.fused_supported(torch.zeros(1, 2, 64, 64), vbar, omega, 16, 16, torch.float16))
        self.assertFalse(self.fused.fused_supported(s, vbar, omega, 16, 16, torch.bfloat16))
        with self.assertRaises(ValueError):
            self.fused.factorize_k31_fused(s, vbar, omega, 2, 8, 16, torch.float16)


if __name__ == "__main__":
    unittest.main()
