"""The factored recurrent state shared by both models (docs/56 §56.j step 3): every head's state S is stored as
    sink  +  rank-r content,
the sink being the state's exact response to the head's constant direction (gated delta: a v_bar^T with v_bar the head's mean value
direction, a = S v_bar / |v_bar|^2; Mamba-2: x_bar a^T with x_bar the head's mean input direction), and the content the best rank-r
approximation of S - sink by the exact Gram eigendecomposition.  The same module prunes the prompt-final state (once, at the end of
the prefill, in training and inference) and the decode-time state every `state_every` steps.  Formerly
the Lightning-only StateFactor (Mamba-2); Flash-Next used an implicit sink and a randomized truncation."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


EXACT_EIGH_DTYPE = torch.float64   # precision of the reference eigendecomposition (truncate_rank_exact)
TRUNCATION = "warm"                # "warm" (the implementation: warm-started subspace iteration) | "exact" (reference; probes.eigh_precision)
OVERSAMPLE = 8                     # extra directions of the subspace iteration beyond r
POWER = 1                          # subspace (power) iterations


def truncate_rank_exact(S: torch.Tensor, r: int, detach_basis: bool = True) -> torch.Tensor:
    """Reference: best rank-r approximation per head via the P x P Gram eigendecomposition (deterministic, EXACT_EIGH_DTYPE).
    Slow at decode (cuSOLVER's batched Jacobi solver on 128 x 128 Gram matrices: 0.15-0.22 s per layer at batch 50, docs/56 §56.l);
    kept as the ground truth for tests and probes."""
    if r <= 0 or r >= min(S.shape[-2], S.shape[-1]):
        return S
    Sf = S.float()
    G = Sf @ Sf.transpose(-1, -2)                                     # (B, H, P, P)
    G = G + 1e-6 * G.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * torch.eye(G.shape[-1], device=G.device)
    with torch.no_grad() if detach_basis else torch.enable_grad():
        _, U = torch.linalg.eigh(G.to(EXACT_EIGH_DTYPE))
    U = U[..., -r:].to(Sf.dtype)                                        # (B, H, P, r)
    if detach_basis:
        U = U.detach()
    return U @ (U.transpose(-1, -2) @ Sf)


def _orthonormalize(Y: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis of the columns of Y (..., P, m) by Cholesky QR, twice (CholeskyQR2): Gram matrix in fp64 with a jitter of
    1e-7 of its mean diagonal, batched Cholesky, triangular solve -- three cheap batched kernels for m <= r + OVERSAMPLE.  Collapsed
    columns survive through the jitter as near-null directions and are discarded by the final energy ordering.  (Alternatives measured:
    batched torch.linalg.qr loops over the batch on CUDA, 50x slower; a Gram eigendecomposition is 3-5x slower than Cholesky.)"""
    Yd = Y.double()
    for _ in range(2):
        G = Yd.transpose(-1, -2) @ Yd
        G = G + (1e-7 * G.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-30) * torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
        L = torch.linalg.cholesky(G)
        Yd = torch.linalg.solve_triangular(L, Yd.transpose(-1, -2), upper=False).transpose(-1, -2)   # Y L^{-T}
    return Yd.to(Y.dtype)


def _small_eigh(G: torch.Tensor) -> torch.Tensor:
    """Eigenvectors (ascending) of a small symmetric PSD batch (..., m, m), m <= r + OVERSAMPLE: jittered and in fp64 so repeated or
    zero eigenvalues (empty heads, padded rows) do not stop the solver."""
    G = G.double()
    G = G + (1e-7 * G.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-30) * torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
    return torch.linalg.eigh(G)[1].to(torch.float32)


def truncate_rank(S: torch.Tensor, r: int, prev: Optional[torch.Tensor] = None, detach_basis: bool = True):
    """Best rank-r approximation per head of S (B, H, P, N) by a warm-started randomized subspace iteration (the implementation of
    the exact Gram eigendecomposition the paper reasons with; truncate_rank_exact is the reference):
        Y = S [V_prev, Omega]  (V_prev: the right factor kept from the previous truncation of the same states; Omega: OVERSAMPLE
        fixed pseudo-random directions, the same numbers on every call so the map is deterministic)  ->  POWER subspace iterations
        ->  Q = orth(Y)  ->  the top-r eigenvectors of the small Gram (Q^T S)(Q^T S)^T give the kept directions U = Q W.
    Between two prunings the state changes by W rank-one updates, so V_prev spans almost the whole answer and the r + OVERSAMPLE
    dimensional search space tracks the exact top-r closely (probes.eigh_precision measures the difference on real decodes).
    Cost: a few batched matmuls of size P x N x (r + OVERSAMPLE), three Cholesky-QR orthonormalisations and one (r + OVERSAMPLE)^2 eigendecomposition --
    milliseconds per layer at batch 50, against 0.15-0.22 s for the exact eigendecomposition.
    With detach_basis the projector's basis is a constant for autograd (straight-through): gradients flow through the linear map
    U U^T S and not through the eigendecompositions.  Returns (S_r, V) with V (B, H, N, r) the orthonormal right factor for the
    next warm start (None when nothing was truncated)."""
    if r <= 0 or r >= min(S.shape[-2], S.shape[-1]):
        return S, None
    Sf = S.float()
    B, H, P, N = Sf.shape
    m = min(r + OVERSAMPLE, P, N)
    g = torch.Generator(device=Sf.device)
    g.manual_seed(0x5EED)
    use_prev = prev is not None and tuple(prev.shape) == (B, H, N, r) and prev.device == Sf.device
    omega = torch.randn(B, H, N, m - (r if use_prev else 0), generator=g, device=Sf.device, dtype=torch.float32)
    if use_prev:
        omega = torch.cat([prev.float(), omega], -1)
    with torch.no_grad() if detach_basis else torch.enable_grad():
        Y = Sf @ omega                                                # (B, H, P, m)
        for _ in range(POWER):                                        # re-orthonormalised power step (without it the columns collapse
            Y = Sf @ _orthonormalize(Sf.transpose(-1, -2) @ _orthonormalize(Y))   # onto the leading direction and the error grows 10-40%)
        Q = _orthonormalize(Y)                                        # (B, H, P, m)
        Bm = Q.transpose(-1, -2) @ Sf                                 # (B, H, m, N)
        W = _small_eigh(Bm @ Bm.transpose(-1, -2))                    # ascending energy of the m directions
        U = Q @ W[..., -r:]                                           # (B, H, P, r), orthonormal
    if detach_basis:
        U = U.detach()
    UtS = U.transpose(-1, -2) @ Sf                                    # (B, H, r, N)
    with torch.no_grad():
        V = _orthonormalize(UtS.detach().transpose(-1, -2))           # (B, H, N, r)
    return U @ UtS, V


class StateFactor(nn.Module):
    """sink + rank-r content for the states of every recurrent layer of one model.

    side = "left":  S (B, H, P, N), sink direction on the P (input) side  -- Mamba-2, x_bar per head;
    side = "right": S (B, H, Dk, Dv), sink direction on the Dv (value) side -- gated delta, v_bar per head.
    `sink_dir` (num_layers, H, D_side) holds the direction for every layer (zeros for layers without a state).  The buffer was
    called `xbar` in the pre-unification Lightning checkpoints; those keys are renamed on load."""

    def __init__(self, num_layers: int, heads: int, side_dim: int, side: str, rank: int, explicit: bool, every: int):
        super().__init__()
        assert side in ("left", "right")
        self.side, self.r, self.explicit, self.every = side, int(rank), bool(explicit), int(every)
        self.register_buffer("sink_dir", torch.zeros(num_layers, heads, side_dim))
        self.detach_basis = True
        self._warm = {}   # layer -> right factor V of the last truncation of the current batch (the decode-time warm start)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        old = prefix + "xbar"
        if old in state_dict and prefix + "sink_dir" not in state_dict:
            state_dict[prefix + "sink_dir"] = state_dict.pop(old)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def set_sink_dirs(self, dirs):
        """dirs: {layer: (H, D_side)} (probes.gdnconst `vbar`: mean value per gated-delta head) or a full (num_layers, H, D_side)
        tensor (twinstar.duet.init_stats `sink_dir`: mean input per Mamba-2 head, mean value per KDA head)."""
        if isinstance(dirs, torch.Tensor):
            self.sink_dir.copy_(dirs.float().to(self.sink_dir.device))
            return
        for l, v in dirs.items():
            self.sink_dir[int(l)].copy_(v.float().to(self.sink_dir.device))

    def sink(self, l: int, S: torch.Tensor) -> torch.Tensor:
        d = self.sink_dir[l].to(S.device, S.dtype)                        # (H, D_side)
        n2 = (d * d).sum(-1).clamp_min(1e-12)                             # (H,)
        if self.side == "left":                                           # S (B, H, P, N), d (H, P): a = S^T d / |d|^2  (B, H, N)
            a = torch.einsum("bhpn,hp->bhn", S, d) / n2[None, :, None]
            return d[None, :, :, None] * a[:, :, None, :]
        a = torch.einsum("bhkv,hv->bhk", S, d) / n2[None, :, None]        # S (B, H, Dk, Dv), d (H, Dv): a = S d / |d|^2  (B, H, Dk)
        return a[:, :, :, None] * d[None, :, None, :]

    def _truncate(self, l: int, X: torch.Tensor, warm: bool) -> torch.Tensor:
        if TRUNCATION == "exact":
            return truncate_rank_exact(X, self.r, self.detach_basis)
        Xr, V = truncate_rank(X, self.r, self._warm.get(l) if warm else None, self.detach_basis)
        self._warm[l] = V
        return Xr

    def forward(self, l: int, S: torch.Tensor, warm: bool = False) -> torch.Tensor:
        """The stored form of layer l's state S (any dtype; computed in fp32, returned in S's dtype).  warm=False at the prompt-final
        state (training forward and prefill: a fresh batch, the warm start of layer l is reset), warm=True at the decode-time prunings
        (start from the factor kept by the previous pruning of the same batch)."""
        if self.r <= 0:
            return S
        Sf = S.float()
        if not self.explicit:
            return self._truncate(l, Sf, warm).to(S.dtype)
        sink = self.sink(l, Sf)
        return (sink + self._truncate(l, Sf - sink, warm)).to(S.dtype)

    def bytes_per_head(self, dims: tuple) -> int:
        """fp32 storage per head: (P + N) numbers per kept column, plus one column for the explicit sink coefficient."""
        return 4 * sum(dims) * (self.r + (1 if self.explicit else 0))

    def describe(self) -> str:
        return f"{'explicit sink + ' if self.explicit else ''}rank {self.r} every {self.every} ({self.side} side)"
