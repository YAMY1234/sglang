# ruff: noqa
# Copied from YAMY1234/twinstar-pd-models cdc30c85972b0b5772a848193c8f752ecf5bce24
# twinstar/models/qwen4_exp.py::Qwen4QSACode. Only a private component-return
# hook is added; default forward and spike selection arithmetic are unchanged.
from __future__ import annotations

import os
from typing import Optional

import torch
from torch import nn

class Qwen4QSACode(nn.Module):
    """Per-QSA-layer code for the prompt cache: values -> mean + D_V E_V (v - mean); keys -> rotated dims exact, non-rotated
    dims -> mean + D_K E_K (k - mean); plus m largest residual coordinates exact (spec.qsa_sparse or QSA_SPARSE).
    Per KV head.  Initialised from probes.qsavbasis (variance bases), trained with the module's KL objective."""

    def __init__(self, cfg: Qwen4ExpConfig, spec: TwinStarSpec, layer: int, init: Optional[dict] = None):
        super().__init__()
        Hkv, hd, rot = cfg.num_key_value_heads, cfg.head_dim, cfg.rotary_dim
        self.rot, self.rv, self.rk = rot, int(spec.qsa_v_rank or 0), int(spec.qsa_k_rank or 0)
        self.sparse = int(getattr(spec, "qsa_sparse", 0) or 0)
        if self.rv:
            self.E_V = nn.Parameter(torch.zeros(Hkv, self.rv, hd))
            self.D_V = nn.Parameter(torch.zeros(Hkv, hd, self.rv))
            self.register_buffer("mean_V", torch.zeros(Hkv, hd))
        if self.rk:
            self.E_K = nn.Parameter(torch.zeros(Hkv, self.rk, hd - rot))
            self.D_K = nn.Parameter(torch.zeros(Hkv, hd - rot, self.rk))
            self.register_buffer("mean_K", torch.zeros(Hkv, hd - rot))
        if init is not None:
            with torch.no_grad():
                if self.rv:
                    U = init["U"][layer].float()[:, :, : self.rv]  # (Hkv, hd, r)
                    self.E_V.copy_(U.transpose(1, 2)); self.D_V.copy_(U); self.mean_V.copy_(init["mean"][layer].float())
                if self.rk:
                    UK = init["UK"][layer].float()[:, :, : self.rk]
                    self.E_K.copy_(UK.transpose(1, 2)); self.D_K.copy_(UK); self.mean_K.copy_(init["meanK"][layer].float())

    def _code(self, x, E, D, mean, *, _return_components=False):
        E, D, mean = E.to(x.device), D.to(x.device), mean.to(x.device)  # layer-sharded evaluation: states live on their layer's GPU
        xf = x.float() - mean[None, :, None, :]
        z = torch.einsum("bhsd,hrd->bhsr", xf, E.float())
        rec = torch.einsum("bhsr,hdr->bhsd", z, D.float())
        m = self.sparse or int(os.environ.get("QSA_SPARSE", "0") or 0)
        if m:
            res = (xf - rec).detach()
            idx = res.abs().topk(m, dim=-1).indices
            rec = rec.scatter(-1, idx, xf.gather(-1, idx))  # selected coordinates exact (no gradient through them)
        if _return_components:
            if not m:
                idx = torch.empty_like(z[..., :0], dtype=torch.int64)
            return z, idx
        return (rec + mean[None, :, None, :]).to(x.dtype)

    def forward(self, k: torch.Tensor, v: torch.Tensor):
        if self.rv:
            v = self._code(v, self.E_V, self.D_V, self.mean_V)
        if self.rk:
            k = torch.cat([k[..., : self.rot], self._code(k[..., self.rot:], self.E_K, self.D_K, self.mean_K)], dim=-1)
        return k, v
