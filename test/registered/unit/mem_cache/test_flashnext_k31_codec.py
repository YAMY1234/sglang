"""#873: Scheme-C codec under the k31-r4096-u release (Mingyuan's LinearCode, no per-token RMS; e2m1 ties to even).

Reference copied verbatim from origin/minma/0913: twinstar/duet/latentfmt.py (sha256 eefcc9729194b214...) _round_e2m1 and
fq_nvfp4, twinstar/duet/latent.py (sha256 bc53d41b308309f3...) LinearCode.forward with G = 1, values bf16.
"""
import unittest

import torch

from sglang.srt.mem_cache import flashnext_scheme_c as sc

FP8_MAX = 448.0
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_EDGES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _round_e2m1(y):
    a = y.abs()
    edges = torch.tensor(_E2M1_EDGES, device=y.device, dtype=y.dtype)
    levels = torch.tensor(_E2M1, device=y.device, dtype=y.dtype)
    idx = torch.bucketize(a, edges, right=True)
    tie = torch.isin(a, edges) & (idx % 2 == 1)
    idx = torch.where(tie, idx - 1, idx)
    return levels[idx] * torch.sign(y)


def fq_nvfp4(x, block=16):
    xf = x.float()
    shp = xf.shape
    r = shp[-1]
    rows = xf.reshape(-1, r)
    g = rows.abs().amax(-1, keepdim=True).clamp_min(1e-12) / (6.0 * FP8_MAX)
    blk = (rows / g).reshape(rows.shape[0], r // block, block)
    sb = blk.abs().amax(-1, keepdim=True) / 6.0
    sb8 = sb.to(torch.float8_e4m3fn).float().clamp_min(2.0 ** -9)
    q = _round_e2m1((blk / sb8).clamp(-6.0, 6.0)) * sb8
    return (q.reshape(rows.shape) * g).reshape(shp)


def linear_code(x, base, E, D, mu, m):
    """LinearCode.forward, G = 1: z nvfp4, spike values bf16."""
    xf = x.float() - base.float()
    c = xf - mu
    z = fq_nvfp4(c @ E.T)
    rec = z @ D.T
    res = c - rec
    idx = res.abs().topk(m, dim=-1).indices
    vals = res.gather(-1, idx).to(torch.bfloat16).float()
    rec = rec + torch.zeros_like(res).scatter(-1, idx, vals)
    return mu + rec + base.float()


class K31CodecTest(unittest.TestCase):
    def test_e2m1_ties_to_even(self):
        row = torch.zeros(1, 32)
        row[0, :8] = torch.tensor([6.0, 0.75, 1.75, 3.5, 0.25, 1.25, 2.5, 5.0])
        row[0, 16] = 2688.0                  # row scale g = 1 exactly; block 0 scale = 1 exactly
        z, s, g = sc._pack_nvfp4_torch(row)
        out = sc._unpack_nvfp4_torch(z, s, g)[0, :8].tolist()
        self.assertEqual(out, [6.0, 1.0, 2.0, 4.0, 0.0, 1.0, 2.0, 4.0])
        self.assertTrue(torch.equal(sc._unpack_nvfp4_torch(z, s, g), fq_nvfp4(row)))

    def test_nvfp4_matches_reference_random(self):
        torch.manual_seed(1)
        x = torch.randn(64, 256) * torch.logspace(-2, 2, 64)[:, None]
        z, s, g = sc._pack_nvfp4_torch(x)
        self.assertTrue(torch.equal(sc._unpack_nvfp4_torch(z, s, g), fq_nvfp4(x)))

    def test_triton_pack_under_interpreter(self):
        try:
            import os
            if os.environ.get("TRITON_INTERPRET") != "1":
                self.skipTest("Triton kernel check runs under TRITON_INTERPRET=1")
            from sglang.srt.mem_cache import flashnext_scheme_c_kernels as k
        except ImportError:
            self.skipTest("triton unavailable")
        row = torch.zeros(1, 32)
        row[0, :8] = torch.tensor([6.0, 0.75, 1.75, 3.5, 0.25, 1.25, 2.5, 5.0])
        row[0, 16] = 2688.0
        zk, sk, gk = k.pack_nvfp4(row)
        zt, st, gt = sc._pack_nvfp4_torch(row)
        self.assertTrue(torch.equal(zk, zt))

    def test_codec_matches_linear_code(self):
        codec = sc.FlashNextSchemeCCodec(device="cpu", compute_precision="fp32", rms_normalize=False)
        torch.manual_seed(2)
        W, R = codec.WIDTH, codec.RANK
        q, _ = torch.linalg.qr(torch.randn(W, R))
        E = q.T.contiguous()[None]                              # k31 layout [1, R, W]
        D = q.contiguous()[None]                                # [1, W, R]
        mu = 0.1 * torch.randn(W)
        codec.load("E", E)
        codec.load("D", D)
        codec.load("mu", mu)
        codec.finalize()
        base = (0.5 * torch.randn(4, W)).to(torch.bfloat16)
        x = (base.float() + torch.randn(4, W) * torch.linspace(0.1, 3.0, W)).to(torch.bfloat16)
        batch, ours = codec.encode_and_decode(x, torch.tensor([1, 2, 3, 4]), base)
        self.assertTrue(torch.equal(batch.rms, torch.ones_like(batch.rms)))
        # the codec rounds its E / D through bf16 on load (bf16-roundtrip-fp32), as the reference loader does
        ref = linear_code(x, base, E[0].to(torch.bfloat16).float(), D[0].to(torch.bfloat16).float(),
                          mu.to(torch.bfloat16).float(), codec.SPIKES).to(torch.bfloat16)
        agree = (ours.float() - ref.float()).abs() <= 1e-2 * ref.float().abs().clamp_min(1e-2)
        self.assertGreaterEqual(float(agree.float().mean()), 0.999)


if __name__ == "__main__":
    unittest.main()
