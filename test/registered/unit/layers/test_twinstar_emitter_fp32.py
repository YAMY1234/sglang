"""#873 fp32 emitter helpers against the reference emitter's arithmetic (CPU).

Reference copied verbatim from origin/minma/0913: twinstar/models/qwen4_exp.py (sha256 858a5dbea5186bd6...)
GroupRMSNorm.forward, GatedResidual.mix, QSAAttention.kv, QSAIndexer.keys; twinstar/models/blocks.py (sha256
8464e25cd4cff9a5...) GemmaRMSNorm.forward, PartialRotary.forward, _rotate_half, apply_rope_partial and the gate lines
of gdn_mix.  The reference builds the emitter with ``.float()`` over bf16-rounded weights (A_log / dt_bias fp32) and
RoPE tables with ``rotary(positions, h.dtype)`` (bf16 model), and casts K / V / the raw index key to bf16 on write.
"""
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.layers import twinstar_emitter_fp32 as e32


# ------------------------------------------------------------------ reference (verbatim)
class GroupRMSNorm(nn.Module):
    def __init__(self, dim: int, groups: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.groups = groups
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.shape[-1] == self.weight.shape[0]
        xs = x.reshape(*x.shape[:-1], self.groups, -1) if flat else x
        xf = xs.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        out = out * (1.0 + self.weight.float().view(self.groups, -1))
        out = out.type_as(x)
        return out.flatten(-2) if flat else out


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * (1.0 + self.weight.float())).type_as(x)


class PartialRotary(nn.Module):
    def __init__(self, rotary_dim: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64).float() / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor, dtype: torch.dtype):
        freqs = positions[:, :, None].float() * self.inv_freq[None, None, :].to(positions.device)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype)[:, None], emb.sin().to(dtype)[:, None]


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_partial(x, cos, sin):
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    return torch.cat([xr * cos + _rotate_half(xr) * sin, xp], dim=-1)


class GatedResidual(nn.Module):
    def __init__(self, hc, D, lowrank, eps):
        super().__init__()
        self.hc, self.D = hc, D
        self.hc_norm = GroupRMSNorm(hc * D, hc, eps)
        self.input_mix_weight_down = nn.Linear(hc * D, lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(lowrank, hc * D, bias=False)

    def mix(self, streams):
        w = self.hc_norm.weight
        streams = streams.to(w.dtype)
        if streams.shape[-1] == self.hc * self.D:
            streams = streams.reshape(*streams.shape[:-1], self.hc, self.D)
        normed = self.hc_norm(streams)
        flat = normed.flatten(-2)
        gate = torch.sigmoid(self.input_mix_weight_up(F.silu(self.input_mix_weight_down(flat) / self.hc)))
        return (gate.view_as(normed) * normed).mean(dim=-2)


# ------------------------------------------------------------------ helpers
def bf16_exact(*shape, scale=1.0):
    return (torch.randn(*shape) * scale).to(torch.bfloat16).float()


class HCMixTest(unittest.TestCase):
    def test_bitwise_equal(self):
        torch.manual_seed(0)
        hc, D, lr, eps = 4, 64, 8, 1e-6
        ref = GatedResidual(hc, D, lr, eps)
        with torch.no_grad():
            ref.hc_norm.weight.copy_(bf16_exact(hc * D, scale=0.1))
            ref.input_mix_weight_down.weight.copy_(bf16_exact(lr, hc * D, scale=0.05))
            ref.input_mix_weight_up.weight.copy_(bf16_exact(hc * D, lr, scale=0.2))
        served = SimpleNamespace(  # served modules hold the bf16 weights
            hc_count=hc, hidden_size=D,
            hc_norm=SimpleNamespace(weight=ref.hc_norm.weight.detach().to(torch.bfloat16), variance_epsilon=eps),
            input_mix_weight_down=SimpleNamespace(weight=ref.input_mix_weight_down.weight.detach().to(torch.bfloat16)),
            input_mix_weight_up=SimpleNamespace(weight=ref.input_mix_weight_up.weight.detach().to(torch.bfloat16)))
        streams = torch.randn(7, hc * D).to(torch.bfloat16)
        with torch.no_grad():
            want = ref.mix(streams[None])[0]
        got = e32.hc_mix(served, streams)
        self.assertEqual(got.dtype, torch.float32)
        self.assertTrue(torch.equal(got, want), float((got - want).abs().max()))


class QSAKVTest(unittest.TestCase):
    def test_bitwise_equal(self):
        torch.manual_seed(1)
        H, nkv, hd, rot, theta, eps, T = 48, 2, 256, 64, 10_000_000.0, 1e-6, 9
        nh = 4
        q_size, kv_size = nh * hd, nkv * hd
        w_q = bf16_exact(2 * q_size, H, scale=0.05)
        w_k, w_v = bf16_exact(kv_size, H, scale=0.05), bf16_exact(kv_size, H, scale=0.05)
        k_norm = GemmaRMSNorm(hd, eps)
        with torch.no_grad():
            k_norm.weight.copy_(bf16_exact(hd, scale=0.1))
        rotary = PartialRotary(rot, theta)
        x = torch.randn(1, T, H)
        positions = torch.arange(100, 100 + T)[None]
        cos, sin = rotary(positions, torch.bfloat16)
        with torch.no_grad():  # QSAAttention.kv (fp32 emitter)
            k = k_norm(F.linear(x, w_k).view(1, T, nkv, hd)).transpose(1, 2)
            v = F.linear(x, w_v).view(1, T, nkv, hd).transpose(1, 2)
            k = apply_rope_partial(k, cos, sin)
        want_k = k.transpose(1, 2).reshape(T, kv_size).to(torch.bfloat16)
        want_v = v.transpose(1, 2).reshape(T, kv_size).to(torch.bfloat16)
        src = SimpleNamespace(
            q_size=q_size, kv_size=kv_size, attn_output_gate=True, num_kv_heads=nkv, head_dim=hd,
            qkv_proj=SimpleNamespace(weight=torch.cat([w_q, w_k, w_v]).to(torch.bfloat16)),
            k_norm=SimpleNamespace(weight=k_norm.weight.detach().to(torch.bfloat16), variance_epsilon=eps),
            rotary_emb=SimpleNamespace(rotary_dim=rot, base=theta))
        got_k, got_v = e32.qsa_kv(src, x[0], positions[0], torch.bfloat16)
        self.assertTrue(torch.equal(got_v, want_v))
        self.assertTrue(torch.equal(got_k, want_k), int((got_k != want_k).sum()))
        # mrope-style (3, T) positions of a text prompt: all axes equal, the first is the logical position
        got_k3, _ = e32.qsa_kv(src, x[0], positions.expand(3, T), torch.bfloat16)
        self.assertTrue(torch.equal(got_k3, want_k))


class IndexKeyTest(unittest.TestCase):
    def test_bitwise_equal(self):
        torch.manual_seed(2)
        H, nh, hd, T = 48, 3, 128, 5
        w = bf16_exact((nh + 1) * hd, H, scale=0.05)
        x = torch.randn(1, T, H)
        with torch.no_grad():  # QSAIndexer.keys
            want = F.linear(x, w)[..., nh * hd:].unsqueeze(1)[:, 0, :, :].reshape(T, 1, hd).to(torch.bfloat16)
        indexer = SimpleNamespace(index_n_heads=nh, index_head_dim=hd, index_kv_heads=1,
                                  index_qk_proj=SimpleNamespace(weight=w.to(torch.bfloat16)))
        got = e32.index_key(indexer, x[0], torch.bfloat16)
        self.assertTrue(torch.equal(got, want))


class GDNGatingTest(unittest.TestCase):
    def test_bitwise_equal_and_dt_bias_precision(self):
        torch.manual_seed(3)
        T, Hv = 6, 8
        A_log = torch.randn(Hv)
        dt_bias = torch.linspace(-8.87, 0.56, Hv) + 1e-3 * torch.randn(Hv)  # fp32, not bf16-exact (the release)
        a, b = torch.randn(T, Hv), torch.randn(T, Hv)
        want_g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
        want_beta = b.sigmoid()
        g, beta = e32.gdn_gating(A_log, dt_bias, a, b)
        self.assertTrue(torch.equal(g[0], want_g))
        self.assertTrue(torch.equal(beta[0], want_beta))
        g16, _ = e32.gdn_gating(A_log, dt_bias.to(torch.bfloat16), a, b)
        self.assertFalse(torch.equal(g16[0], want_g))  # a bf16 dt_bias parameter changes the decay


class GDNInputsTest(unittest.TestCase):
    def test_rows(self):
        torch.manual_seed(4)
        H, k, v, heads = 32, 16, 24, 3
        wq = bf16_exact(2 * k + v + v, H)  # qkvz rows: q, k, v, z
        wba = bf16_exact(2 * heads, H)     # b rows, then a rows
        gdn = SimpleNamespace(key_dim=k, value_dim=v, num_v_heads=heads, attn_tp_size=1,
                              in_proj_qkvz=SimpleNamespace(weight=wq.to(torch.bfloat16)),
                              in_proj_ba=SimpleNamespace(weight=wba.to(torch.bfloat16)))
        x = torch.randn(5, H)
        mixed, a, b = e32.gdn_inputs(gdn, x)
        self.assertTrue(torch.equal(mixed, F.linear(x, wq[:2 * k + v])))
        self.assertTrue(torch.equal(b, F.linear(x, wba[:heads])))
        self.assertTrue(torch.equal(a, F.linear(x, wba[heads:])))


if __name__ == "__main__":
    unittest.main()
