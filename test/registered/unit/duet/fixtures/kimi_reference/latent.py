"""The residual code shared by both models (docs/56 §56.j step 2): x -> mu + D E (x - mu) + the m largest |residual| entries
exact, with the token's own input embedding as optional side information and the storage quantizer in the forward (QAD).
Formerly the Lightning-only LinearCode; the Flash-Next path used a separate LatentBottleneck with a per-token RMS and an
exempt first token, both dropped in the unification (nvfp4 carries its own per-token scale; every token is coded)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from twinstar.duet import latentfmt


class LinearCode(nn.Module):
    """E (G, r, d), D (G, d, r), mu (G, d) with G heads (G = 1 for the latent).  Spike indices are chosen without gradient; the code
    path is differentiable in E, D with a straight-through quantizer."""

    def __init__(self, groups: int, dim: int, rank: int, spikes: int, fmt: Optional[latentfmt.LatentFormat] = None):
        super().__init__()
        self.G, self.d, self.r, self.m = groups, dim, rank, spikes
        self.fmt = fmt or latentfmt.LatentFormat()
        if self.fmt.z == "nvfp4" and rank % latentfmt.NVFP4_BLOCK:
            raise ValueError(f"latent rank {rank} must be a multiple of {latentfmt.NVFP4_BLOCK} for nvfp4 storage")
        self.E = nn.Parameter(torch.zeros(groups, rank, dim))
        self.D = nn.Parameter(torch.zeros(groups, dim, rank))
        self.register_buffer("mu", torch.zeros(groups, dim))

    @torch.no_grad()
    def init_from_stats(self, mean: torch.Tensor, cov: torch.Tensor):
        """mean (G, d), cov (G, d, d) in the residual's own units: top-r eigenvectors of the covariance (the KL training moves them)."""
        self.mu.copy_(mean.float())
        evals, evecs = torch.linalg.eigh(cov.float())                # ascending
        U = evecs[:, :, -self.r:] if self.r > 0 else evecs[:, :, :0]  # (G, d, r)
        self.E.copy_(U.transpose(1, 2))
        self.D.copy_(U)

    @torch.no_grad()
    def init_from_basis(self, mean: torch.Tensor, E: torch.Tensor, D: torch.Tensor):
        """mean (d,), E (r', d) rows and D (d, r') columns of a principal basis (probes.ckptpca output); the first r are used."""
        self.mu.copy_(mean.float().reshape(self.G, self.d))
        self.E.copy_(E[: self.r].float().reshape(self.G, self.r, self.d))
        self.D.copy_(D[:, : self.r].float().reshape(self.G, self.d, self.r))

    def forward(self, x: torch.Tensor, base: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x (..., G, d) -> coded (..., G, d), same dtype.  base (same shape, optional): the exactly known part of x (the token's own
        input embedding); subtracted before coding and added back after.  Storage format: self.fmt, from the spec."""
        dt = x.dtype
        xf = x.float()
        if base is not None:
            xf = xf - base.float()
        c = xf - self.mu
        z = torch.einsum("...gd,grd->...gr", c, self.E)
        z = latentfmt.quantize(z, self.fmt.z)
        rec = torch.einsum("...gr,gdr->...gd", z, self.D)
        res = c - rec
        if self.m > 0:
            idx = res.abs().detach().topk(self.m, dim=-1).indices
            vals = latentfmt.quantize(res.gather(-1, idx), self.fmt.value)
            rec = rec + torch.zeros_like(res).scatter(-1, idx, vals)
        out = self.mu + rec
        if base is not None:
            out = out + base.float()
        return out.to(dt)

    def bytes_per_vector(self) -> float:
        """Stored size of one coded vector in the spec's storage format (nominal; gap8 indices are data dependent)."""
        return self.fmt.bytes_per_token(self.r, self.m, self.d)

    def describe(self) -> str:
        return f"rank {self.r}, {self.m} exact coordinates, format {self.fmt.describe()} ({self.bytes_per_vector():.0f} B/token)"


class ResidualCode(nn.Module):
    """The code of the residual entering the prefill depth, as the hooked models (Lightning, Kimi-Linear) apply it: one LinearCode over
    the hidden dimension plus the side-channel switch of the spec (the token's own input embedding is subtracted before coding and added
    back after).  Flash-Next wraps the same LinearCode over its four-stream residual (twinstar.models.qwen4_exp.Qwen4Latent)."""

    def __init__(self, hidden_size: int, spec):
        super().__init__()
        self.code = LinearCode(1, hidden_size, spec.latent_rank, spec.latent_spikes, latentfmt.from_spec(spec)) if spec.latent_rank > 0 else None
        self.id_side = bool(spec.latent_id_side)

    def forward(self, h, base=None):
        """h: residual entering block k (B, T, d); base: the prompt tokens' input embeddings (B, T, d), used when the spec sets
        latent_id_side (the caller always passes it; the switch lives here so that the trainer and the served prefill take one path)."""
        if self.code is None:
            return h
        if self.id_side and base is None:
            raise ValueError("latent_id_side is set but no token embedding was passed to the code")
        b = base if self.id_side else None
        if not getattr(self, "_logged", False):
            print(f"[duet] code applied: {self.code.describe()}, id_side={self.id_side}", flush=True); self._logged = True
        return self.code(h[..., None, :], None if b is None else b[..., None, :])[..., 0, :]
