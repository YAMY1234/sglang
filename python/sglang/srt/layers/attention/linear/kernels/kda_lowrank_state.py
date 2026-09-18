# SPDX-License-Identifier: Apache-2.0
"""Reference factored KDA state for the LR-KDA conversion experiment (#211).

Per-channel decay destroys an orthonormal key basis. Keep general factors and
re-orthogonalize BOTH sides before truncating. This module is a numerical
reference; it does not enable a serving backend or replace a Mamba pool.
"""
from dataclasses import dataclass

import torch


@dataclass
class LowRankKDAState:
    a: torch.Tensor  # [B,H,K], exact response to value=1
    u: torch.Tensor  # [B,H,K,R], non-orthogonal between truncations
    w: torch.Tensor  # [B,H,V,R]
    rank: int
    chunk_size: int = 64
    offset: int = 0

    @classmethod
    def zeros(cls, batch, heads, key_dim, value_dim, rank, device, chunk_size=64):
        if not 1 <= rank <= min(key_dim, value_dim):
            raise ValueError(rank)
        if chunk_size < 1:
            raise ValueError(chunk_size)
        return cls(
            torch.zeros(batch, heads, key_dim, device=device),
            torch.empty(batch, heads, key_dim, 0, device=device),
            torch.empty(batch, heads, value_dim, 0, device=device),
            rank,
            chunk_size,
        )

    def dense(self, vbar):
        return self.u @ self.w.transpose(-1, -2) + self.a[..., None] * vbar[None, :, None, :]

    def truncate(self):
        # U is NOT orthogonal after row-wise decay. W^T W alone gives the
        # wrong singular directions; reduce with QR on each factor first.
        qu, ru = torch.linalg.qr(self.u, mode="reduced")
        qw, rw = torch.linalg.qr(self.w, mode="reduced")
        left, singular, right = torch.linalg.svd(ru @ rw.transpose(-1, -2), full_matrices=False)
        keep = min(self.rank, singular.shape[-1])
        self.u = qu @ left[..., :keep]
        self.w = (qw @ right[..., :keep, :].transpose(-1, -2)) * singular[..., None, :keep]

    def step(self, q, k, value, log_decay, beta, vbar, vbar_init):
        """One token; values are post-conv, beta is post-sigmoid, q/k raw.

        Return output BEFORE the boundary projection, matching chunk training.
        All recurrence arithmetic is fp32. vbar and vbar_init are [H,V].
        """
        dtype = value.dtype
        q, k = q.float(), k.float()
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * q.shape[-1] ** -0.5
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        decay = log_decay.float().exp()
        self.a = decay * self.a
        self.a = self.a + beta.float()[..., None] * k * (1 - (k * self.a).sum(-1, keepdim=True))
        self.u = decay[..., None] * self.u
        memory = (self.w @ (self.u.transpose(-1, -2) @ k[..., None]))[..., 0]
        delta = beta.float()[..., None] * (value.float() - vbar_init - memory)
        self.u = torch.cat((self.u, k[..., None]), -1)
        self.w = torch.cat((self.w, delta[..., None]), -1)
        out = (self.w @ (self.u.transpose(-1, -2) @ q[..., None]))[..., 0]
        out = out + (self.a * q).sum(-1, keepdim=True) * vbar
        self.offset += 1
        if self.offset % self.chunk_size == 0:
            self.truncate()
        return out.to(dtype)

    def allocated_bytes_per_head(self, factor_bytes=4):
        """Capacity for r+C pairs, fp32 sink and int32 count (conv excluded)."""
        key_dim, value_dim = self.a.shape[-1], self.w.shape[-2]
        return 4 * key_dim + (self.rank + self.chunk_size) * (key_dim + value_dim) * factor_bytes + 4
