"""Numerical contracts of cutoff-group compaction; prospective, not a result.

The standalone Order297 numerical gate additionally carries native/native
controls. Its acceptance criterion must be agreed before this is executed.
"""

import unittest

import torch
from sglang.kernels.ops.sampling.compact_support import compact_top_k_top_p_probs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def native_renorm_probs(probs, top_k, top_p):
    from flashinfer.sampling import top_k_renorm_probs, top_p_renorm_probs

    rows = probs.shape[0]
    probs = top_k_renorm_probs(
        probs, torch.full((rows,), top_k, dtype=torch.int32, device=probs.device)
    )
    return top_p_renorm_probs(
        probs, torch.full((rows,), top_p, dtype=torch.float32, device=probs.device)
    )


class TestCompactSupport(CustomTestCase):
    def check_distribution(self, logits, top_k=20, top_p=0.95):
        temperatures = torch.full(
            (logits.shape[0], 1), 0.6, dtype=torch.float32, device=logits.device
        )
        probs = torch.softmax(logits / temperatures, dim=-1)
        expected = native_renorm_probs(probs, top_k, top_p)
        actual = compact_top_k_top_p_probs(probs, top_k, top_p)
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertTrue((actual >= 0).all().item())
        tv = 0.5 * (actual.double() - expected.double()).abs().sum(dim=-1)
        # Proposed pre-registered FP32 allowance, pending Order297 agreement.
        self.assertLessEqual(tv.max().item(), 1e-6)
        mass_error = (actual.double().sum(dim=-1) - 1).abs()
        native_mass_error = (expected.double().sum(dim=-1) - 1).abs()
        self.assertTrue((mass_error <= native_mass_error + 1e-6).all().item())

    def test_cutoff_ties_are_not_truncated(self):
        # Native >=cutoff semantics can retain thousands of ties, not only k.
        for vocab in (127, 1025, 248320):
            with self.subTest(vocab=vocab):
                logits = torch.zeros((7, vocab), device="cuda", dtype=torch.float32)
                logits[:, :19] = 0.01
                self.check_distribution(logits)

    def test_uniform_and_masked_rows(self):
        for top_k in (1, 20, 32):
            with self.subTest(top_k=top_k):
                logits = torch.zeros((7, 127), device="cuda", dtype=torch.float32)
                self.check_distribution(logits, top_k)
                logits[:, 1:] = float("-inf")
                self.check_distribution(logits, top_k)

    def test_changing_inputs_and_ragged_tiles(self):
        generator = torch.Generator(device="cuda").manual_seed(23)
        for vocab in (20, 21, 127, 4097, 248320):
            logits = torch.randn(
                (7, vocab), generator=generator, device="cuda", dtype=torch.float32
            )
            for top_p in (0.5, 0.95, 1.0):
                with self.subTest(vocab=vocab, top_p=top_p):
                    self.check_distribution(logits, top_p=top_p)
                    self.check_distribution(-logits, top_p=top_p)

    def test_softmax_rounding_keeps_cutoff_support(self):
        # All low logits differ from zero, but subtraction of the row maximum
        # rounds them to the same FP32 softmax value. Selecting raw logits
        # instead would lose almost the entire native cutoff support.
        logits = torch.full((7, 1025), -(2.0**-30), device="cuda", dtype=torch.float32)
        logits[:, :19] = 1.0
        logits[:, 19] = 0.0
        self.check_distribution(logits)

    def test_input_contract(self):
        probs = torch.full((7, 127), 1.0 / 127, device="cuda", dtype=torch.float32)
        with self.assertRaises(RuntimeError):
            compact_top_k_top_p_probs(probs[:, ::2], 20, 0.95)
        with self.assertRaises(RuntimeError):
            compact_top_k_top_p_probs(probs.half(), 20, 0.95)
        with self.assertRaises(RuntimeError):
            compact_top_k_top_p_probs(probs, 33, 0.95)
        with self.assertRaises(RuntimeError):
            compact_top_k_top_p_probs(probs.cpu(), 20, 0.95)


if __name__ == "__main__":
    unittest.main()
