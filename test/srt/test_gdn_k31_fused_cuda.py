"""Fused prompt-end k31, CUDA backend (kernels/gdn_k31_cuda.py; docs/223 §9.4) against factorize_prefill_k31.

GPU only. Same chained stage checks as the CPU-interpreter test of the Triton version (k31_fused_xcheck.xcheck),
with GPU tolerances where the fused kernel's fp32 summation order differs from cuBLAS; plus the block-structured
Newton against the forced 32 x 32 Newton (bitwise: the block sums drop exact-zero terms only).
"""
import os

os.environ.setdefault("SGLANG_GDN_K31_EIGH", "jacobi")

import sys  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k31_fused_xcheck as xc  # noqa: E402


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class FusedK31CudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref, cls.mixed, cls.fused = xc.load_modules()
        cls.factorize = staticmethod(xc.cuda_factorize())
        fx = torch.load(xc.FIXTURE, weights_only=False)
        cls.dense, cls.vbar, cls.omega = fx["dense"].cuda(), fx["vbar"].cuda(), fx["omega"].cuda()

    def _check_stages(self, out):
        st = out["stages"]
        self.assertTrue(out["a_bitwise"])
        self.assertTrue(out["q_pad_zero"])
        self.assertTrue(out["rows_ge_r_zero"])
        self.assertTrue(st["gn_bitwise_given_g64"])
        self.assertTrue(st["jacobi_bitwise_given_g64"])
        for name in ("y0=x@omega", "z=x^T@o1", "y2=x@o2"):
            self.assertLessEqual(st[name]["max"], 1e-5, name)
        for name in ("orth1", "orth2", "orth3"):
            self.assertLessEqual(st[name]["span"]["max"], 1e-5, name)
        self.assertLessEqual(st["gram64"]["max"], 1e-5)
        self.assertLessEqual(st["newton_vectors"]["max"], 1e-12)
        self.assertLessEqual(st["newton_values"], 1e-12)
        self.assertLessEqual(st["U_given_q_wr"]["max"], 1e-3)
        self.assertLessEqual(st["W_given_q_wr"]["max"], 1e-3)
        self.assertEqual(out["fallback_count"]["fused"], out["fallback_count"]["reference_chained"])

    def test_fixture_stages_and_module_gate(self):
        out = xc.xcheck(self.ref, self.mixed, self.fused, self.dense[None], self.vbar, self.omega[None], 12,
                        factorize=self.factorize)
        self._check_stages(out)
        e2e = out["end_to_end"]
        self.assertTrue(e2e["a_bitwise"])
        self.assertEqual(e2e["nonfinite"], 0)
        self.assertLessEqual(e2e["content"]["median"], 3e-5)
        self.assertLessEqual(e2e["content"]["max"], 2.5e-4)
        self.assertLessEqual(e2e["ratio_max"], 1.01)

    def test_fp32_stage_fallback_counts_per_matrix(self):
        dense = self.dense.clone()
        dense[3] = dense[3] / dense[3].abs().max() * 1e19
        heads = [2, 3, 4]
        out = xc.xcheck(self.ref, self.mixed, self.fused, dense[heads][None], self.vbar[heads],
                        self.omega[heads][None], 3, factorize=self.factorize)
        self.assertGreaterEqual(out["fallback_count"]["fused"], 1)
        self.assertEqual(out["fallback_count"]["fused"], out["fallback_count"]["reference_chained"])
        self.assertEqual(out["stages"]["orth1"]["fallbacks"], 1)
        alone = self.factorize(dense[[2, 4]][None], self.vbar[[2, 4]], self.omega[[2, 4]][None], 2, 16, 16,
                               torch.float16)
        together = self.factorize(dense[heads][None], self.vbar[heads], self.omega[heads][None], 3, 16, 16,
                                  torch.float16)
        for x, y in zip(alone, together):
            self.assertTrue(torch.equal(x, y[:, [0, 2]]))

    def test_block_newton_equals_full_newton(self):
        args = (self.dense[None], self.vbar, self.omega[None], 12, 16, 16, torch.float16)
        block, full = {}, {}
        out_b = self.factorize(*args, debug=block)
        out_f = self.factorize(*args, debug=full, force_full_newton=True)
        for x, y in zip(out_b, out_f):
            self.assertTrue(torch.equal(x, y))
        self.assertTrue(torch.equal(block["eig"], full["eig"]))

    def test_fp32_storage_and_rmax_zero_rows(self):
        a, u, w = self.factorize(self.dense[None], self.vbar, self.omega[None], 12, 16, 32, torch.float32)
        self.assertEqual(u.dtype, torch.float32)
        self.assertTrue(bool((u[:, :, 16:] == 0).all()) and bool((w[:, :, 16:] == 0).all()))
        a16, u16, w16 = self.factorize(self.dense[None], self.vbar, self.omega[None], 12, 16, 16, torch.float16)
        self.assertTrue(torch.equal(u[:, :, :16].half(), u16))
        self.assertTrue(torch.equal(w[:, :, :16].half(), w16))


if __name__ == "__main__":
    unittest.main()
