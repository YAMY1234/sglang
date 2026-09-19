# SPDX-License-Identifier: Apache-2.0
"""Standalone eager pool reference: no SGLang server imports required."""
import importlib.util
from pathlib import Path

import torch


def test_pool():
    path = Path(__file__).resolve().parents[3]/"python/sglang/srt/layers/attention/linear/kernels/kda_lowrank_pool.py"
    spec = importlib.util.spec_from_file_location("lr_pool", path)
    pool_impl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pool_impl)
    torch.set_num_threads(4)
    torch.manual_seed(217)
    h, d, r = 2, 32, 8
    for interval in (64, 16, 8):
        slot = torch.zeros(h, d, 2*(r+64)+2)
        content = torch.zeros(h, d, d)
        sink = torch.zeros(h, d)
        center = torch.randn(h, d)*.1
        mean = center+torch.randn(h, d)*.02
        maximum = 0.
        for step in range(137):
            q, k, v = [torch.randn(h, d) for _ in range(3)]
            g, beta = -torch.rand(h, d)*.05, torch.rand(h)
            out = pool_impl.decode_one(slot, q, k, v, g, beta, mean, center, r, interval)
            qt = q/(q.square().sum(-1, keepdim=True)+1e-6).sqrt()/d**.5
            kt = k/(k.square().sum(-1, keepdim=True)+1e-6).sqrt()
            content = g.exp()[..., None]*content
            content += kt[..., None]*(beta[:, None]*(v-center-(content*kt[..., None]).sum(-2)))[:, None, :]
            sink = g.exp()*sink
            sink += beta[:, None]*kt*(1-(sink*kt).sum(-1, keepdim=True))
            reference = (content*qt[..., None]).sum(-2)+(sink*qt).sum(-1, keepdim=True)*mean
            if (step+1) % interval == 0:
                u, s, vh = torch.linalg.svd(content, full_matrices=False)
                content = (u[..., :r]*s[..., None, :r]) @ vh[..., :r, :]
            u, w, a, offset = pool_impl.read_slot(slot, r)
            assert offset == step+1
            maximum = max(maximum, (out-reference).abs().max().item(),
                          (u @ w.transpose(-1, -2)-content).abs().max().item(), (a-sink).abs().max().item())
        assert maximum <= 1e-4, (interval, maximum)
        print("PASS", {"interval": interval, "max_error": maximum,
                       "resident_bytes_per_head": slot.numel()*slot.element_size()//h})
    pool = torch.zeros(5, h, d, 2*(r+64)+2)
    indices = torch.tensor([4, 1, 3])
    for slot, length in zip(indices, (3, 10, 60)):
        for _ in range(length):
            q, k, v = [torch.randn(h, d) for _ in range(3)]
            pool_impl.decode_one(pool[slot], q, k, v, -torch.rand(h, d)*.05, torch.rand(h), mean, center, r)
    reference_pool = pool.clone()
    for _ in range(67):
        q, k, v = [torch.randn(3, h, d) for _ in range(3)]
        g, beta = -torch.rand(3, h, d)*.05, torch.rand(3, h)
        expected = torch.stack([pool_impl.decode_one(reference_pool[j], q[i], k[i], v[i], g[i], beta[i], mean, center, r)
                                for i, j in enumerate(indices)])
        actual = pool_impl.decode_batch(pool, indices, q, k, v, g, beta, mean, center, r)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)
        for j in indices:
            u, w, a, offset = pool_impl.read_slot(pool[j], r)
            ur, wr, ar, offset_r = pool_impl.read_slot(reference_pool[j], r)
            torch.testing.assert_close(u @ w.transpose(-1, -2), ur @ wr.transpose(-1, -2), atol=1e-4, rtol=0)
            assert offset == offset_r
    assert not pool[0].any() and not pool[2].any()
    print("PASS batched recurrence, heterogeneous offsets, noncontiguous slots")


if __name__ == "__main__":
    test_pool()
