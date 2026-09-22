"""Frozen 21a0ea28 P-end truncation algebra for v3 serving.

The service uses a deterministic per-layer probe for batching/radix repeatability;
this is not the reference process's global RNG sequence. The complete K1 gate
validates that difference. Decode's r8/W8 kernels remain separately controlled.
"""
import torch


def _orth_ns(y):
    # Frozen 21a0ea28 twinstar/probes/gdnstep.py: eight Newton-Schulz rounds.
    g = y.transpose(-1, -2) @ y
    eye = torch.eye(g.shape[-1], device=g.device, dtype=g.dtype)
    trace = g.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
    gn = g / trace.clamp_min(1e-30) + 1e-6 * eye
    a, z = gn, eye.expand_as(gn)
    for _ in range(8):
        t = .5 * (3 * eye - z @ a)
        a, z = a @ t, t @ z
    return y @ (z / trace.clamp_min(1e-30).sqrt())


def factorize_prefill_reference(s, vbar, r, rmax, dtype, iters=2, oversample=8,
                    method="iter", omega=None):
    """Paper's P-end algebra, deterministic existing service probes.

    This isolates the truncation algorithm, not the HF RNG sequence or decode
    truncation. Diagnostic only until complete fixed-threshold K1 validation.
    """
    if method != "iter":
        raise ValueError("paper diagnostic only supports prefill iteration")
    s = s.float()
    vb = vbar.float()
    a = torch.einsum("bhvk,hv->bhk", s, vb) / vb.square().sum(-1).clamp_min(1e-30)[None, :, None]
    c = s - vb[None, :, :, None] * a[:, :, None, :]
    x = c.transpose(-1, -2)
    if omega is None:
        gen = torch.Generator(device=s.device).manual_seed(0)
        omega = torch.randn(*s.shape[:2], s.shape[-2], r+oversample,
                            device=s.device, dtype=torch.float32, generator=gen)
    y = x @ omega
    for _ in range(iters):
        y = x @ _orth_ns(x.transpose(-1, -2) @ _orth_ns(y))
    q = _orth_ns(y)
    b = q.transpose(-1, -2) @ x
    if oversample:
        g = (b @ b.transpose(-1, -2)).double()
        jitter = 1e-7*(g.diagonal(dim1=-2, dim2=-1).sum(-1)/g.shape[-1])[..., None, None] + 1e-30
        g = g + jitter*torch.eye(g.shape[-1], device=g.device, dtype=g.dtype)
        z = torch.linalg.eigh(g)[1][..., -r:].flip(-1).float()
        b, q = z.transpose(-1, -2) @ b, q @ z
    u = torch.zeros(*s.shape[:2], rmax, s.shape[-1], device=s.device, dtype=dtype)
    w = torch.zeros(*s.shape[:2], rmax, s.shape[-2], device=s.device, dtype=dtype)
    u[:, :, :r], w[:, :, :r] = q.transpose(-1, -2), b
    return a, u, w
