# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU test: no initialized engine/distributed groups required."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "python/sglang/srt/layers/attention/linear/kernels/kda_lowrank_state.py"
spec = importlib.util.spec_from_file_location("kda_lowrank_reference", PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
State = module.LowRankKDAState


class TestLowRankKDA(unittest.TestCase):
    def test_channel_decay_and_boundary(self):
        torch.set_num_threads(2)
        torch.manual_seed(211)
        B, H, D, rank = 1, 2, 32, 8
        state = State.zeros(B, H, D, D, rank, "cpu")
        dense = torch.zeros(B, H, D, D)
        a = torch.zeros(B, H, D)
        mean = torch.randn(H, D)
        learned = mean + torch.randn_like(mean) * .01
        for t in range(137):
            q, k, v = [torch.randn(B, H, D) for _ in range(3)]
            g, beta = -torch.rand(B, H, D) * .04, torch.rand(B, H)
            out = state.step(q, k, v, g, beta, learned, mean)
            kn = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
            qn = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) / D**.5
            dense *= g.exp()[..., None]
            a *= g.exp()
            a += beta[..., None] * kn * (1-(a*kn).sum(-1, keepdim=True))
            delta = beta[..., None] * (v + learned - mean - (dense*kn[..., None]).sum(-2))
            dense += kn[..., None] * delta[..., None, :]
            ref = (dense * qn[..., None]).sum(-2)
            torch.testing.assert_close(out, ref, atol=1e-4, rtol=0)
            if (t+1) % 64 == 0:
                sink = a[..., None] * learned[None, :, None, :]
                u, s, vh = torch.linalg.svd(dense-sink, full_matrices=False)
                dense = (u[..., :rank]*s[..., None, :rank]) @ vh[..., :rank, :] + sink
            torch.testing.assert_close(state.dense(learned), dense, atol=1e-4, rtol=0)
        full = State.zeros(1, 32, 128, 128, 8, "cpu")
        self.assertEqual(full.allocated_bytes_per_head(), 74244)
        self.assertGreater(full.allocated_bytes_per_head(), 128*128*4)


if __name__ == "__main__":
    unittest.main()
