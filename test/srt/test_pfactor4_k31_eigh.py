"""Actual Jacobi-kernel parity, residual and orthogonality; CPU interpreter or CUDA.

Run in the service image with TRITON_INTERPRET=1 on CPU, or with CUDA available.
The original fixed twelve-sweep path remains callable as the comparison.
"""
import json
import os
import unittest

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_k31_eigh import eigh


class JacobiIdentitySweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = 'cpu' if os.environ.get('TRITON_INTERPRET') == '1' else 'cuda'
        cls.generator = torch.Generator(device=cls.device).manual_seed(0x504634)

    def check(self, matrix):
        old_d, old_z = eigh(matrix, early_exit=False)
        new_d, new_z = eigh(matrix, early_exit=True)
        # Only sweeps whose every original rotation is the identity may vanish.
        self.assertTrue(torch.equal(old_d, new_d), 'eigenvalue changed')
        self.assertTrue(torch.equal(old_z, new_z), 'eigenvector changed')
        norm = matrix.norm(dim=(-2, -1)).clamp_min(1e-300)
        residual = (matrix @ new_z-new_z*new_d.unsqueeze(-2)).norm(dim=(-2, -1))/norm
        identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
        orth = (new_z.transpose(-1, -2)@new_z-identity).abs().max()
        self.assertLess(float(residual.max()), 1e-11)
        self.assertLess(float(orth), 1e-11)

    def test_diagonal_zero_and_repeated_spectrum(self):
        n = 16
        identity = torch.eye(n, device=self.device, dtype=torch.float64)
        self.check(torch.stack([identity, identity*1e-30, identity*0]))

    def test_full_rank_and_nearly_rank_deficient_grams(self):
        # Real service B1/B8, HV24 shapes are exercised on the CUDA gate. The
        # interpreter executes identical kernels on a smaller number of CTAs.
        batches = (1,) if self.device == 'cpu' else (1, 8)
        heads = 2 if self.device == 'cpu' else 24
        for batch in batches:
            x = torch.randn(batch, heads, 16, 32, device=self.device,
                            dtype=torch.float64, generator=self.generator)
            grams = [x @ x.transpose(-1, -2)]
            x[..., 4:, :] *= 1e-8
            grams.append(x @ x.transpose(-1, -2))
            for gram in grams:
                mean = gram.diagonal(dim1=-2, dim2=-1).mean(-1)
                gram += (mean*1e-7+1e-30)[..., None, None]*torch.eye(
                    16, device=self.device, dtype=torch.float64)
                self.check(gram)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(JacobiIdentitySweepTest))
    print('PFACTOR4_K31_GATE', json.dumps(dict(passed=result.wasSuccessful(),
        tests=result.testsRun, device=JacobiIdentitySweepTest.device)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
