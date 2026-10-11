"""CPU numerical and device-output contract tests; never initialize CUDA."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/sglang/srt/models"))
from kimi_linear_duet.small_eigh import REL_TOL, jacobi_cpu, small_eigh


class JacobiOccupancyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_counts_exclude_convergence_check_and_preserve_batch_shape(self):
        generator = torch.Generator().manual_seed(193)
        a = torch.randn(2, 3, 40, 40, generator=generator, dtype=torch.float64)
        gram = a @ a.transpose(-1, -2)
        gram[0, 0].zero_()
        gram[0, 1] = torch.eye(40, dtype=torch.float64)
        diagnostics = {}
        d, z = small_eigh(gram, cpu_jacobi=True, diagnostics=diagnostics)
        counts = diagnostics["jacobi_sweeps"]
        self.assertEqual(counts.shape, (2, 3))
        self.assertEqual(counts.dtype, torch.int32)
        self.assertEqual(counts.device, gram.device)
        self.assertEqual(counts[0, :2].tolist(), [0, 0])
        self.assertGreater(int(counts[1].min()), 0)
        self.assertLess(int(counts.max()), 12)
        relative = ((z * d[..., None, :]) @ z.transpose(-1, -2) - gram).norm()
        self.assertLess(float(relative / gram.norm()), 1e-9)

    def test_tolerance_comparison_and_fp64_ritz_subspace(self):
        generator = torch.Generator().manual_seed(194)
        a = torch.randn(3, 40, 40, generator=generator, dtype=torch.float64)
        gram = a @ a.transpose(-1, -2)
        strict, relaxed = {}, {}
        _, z = jacobi_cpu(gram, rel_tol=REL_TOL, diagnostics=relaxed)
        jacobi_cpu(gram, rel_tol=1e-14, diagnostics=strict)
        self.assertTrue(
            all(a <= b for a, b in zip(relaxed["sweeps"], strict["sweeps"]))
        )
        expected = torch.linalg.eigh(gram)[1][..., -32:]
        actual = z[..., -32:]
        torch.testing.assert_close(
            actual @ actual.transpose(-1, -2),
            expected @ expected.transpose(-1, -2),
            rtol=1e-8,
            atol=1e-8,
        )

    def test_exact_cpu_solver_clears_stale_device_counts(self):
        diagnostics = {"jacobi_sweeps": torch.ones(5), "solver": "old cut"}
        small_eigh(torch.eye(40, dtype=torch.float64), diagnostics=diagnostics)
        self.assertIsNone(diagnostics["jacobi_sweeps"])
        self.assertNotIn("solver", diagnostics)

    def test_launcher_register_cap_and_device_counter_shape_without_cuda(self):
        try:
            from kimi_linear_duet import jacobi
        except ImportError:
            self.skipTest("Triton not installed; CPU mirror still covered")
        calls = []

        class Capture:
            def __getitem__(self, grid):
                def launch(g, d, z, counts, **kwargs):
                    calls.append((grid, kwargs))
                    counts.copy_(torch.arange(len(counts), dtype=torch.int32) % 8)

                return launch

        gram = torch.eye(40, dtype=torch.float64).expand(20, 32, -1, -1)
        with patch.object(jacobi, "_eigh_kernel", Capture()):
            d, z, counts = jacobi.eigh(gram, return_sweeps=True)
        self.assertEqual(d.shape, (20, 32, 40))
        self.assertEqual(z.shape, gram.shape)
        self.assertEqual(counts.shape, (20, 32))
        self.assertEqual(calls[0][0], (640,))
        self.assertEqual(calls[0][1]["maxnreg"], 128)
        self.assertEqual(calls[0][1]["num_warps"], 8)
        self.assertEqual(calls[0][1]["REL_TOL"], REL_TOL)
        self.assertEqual(counts.flatten()[:8].tolist(), list(range(8)))

    def test_empty_batch_does_not_launch_or_read_counters(self):
        try:
            from kimi_linear_duet import jacobi
        except ImportError:
            self.skipTest("Triton not installed")
        with patch.object(jacobi, "_eigh_kernel") as kernel:
            d, z, counts = jacobi.eigh(
                torch.empty(0, 32, 40, 40, dtype=torch.float64), return_sweeps=True
            )
        kernel.__getitem__.assert_not_called()
        self.assertEqual(counts.shape, (0, 32))
        self.assertEqual(d.numel(), 0)
        self.assertEqual(z.numel(), 0)


if __name__ == "__main__":
    unittest.main()
