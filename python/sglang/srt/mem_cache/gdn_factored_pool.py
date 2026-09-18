"""Factored GDN state pool (TwinStar K1, docs/62; kernel maths from docs/60 / twinstar/kernels/gdn_factored.py).

Per layer per slot per head the GDN state S (sglang layout (V, K), fp32 dense = 64 KB) is stored as
    S = vbar a^T + W^T U          a (K) fp32, U (RMAX, K) orthonormal rows (key side), W (RMAX, V), count in [r, r+m]
(bf16 factors: 8.7 KB/head at RMAX = 16 -- 7.5x smaller than dense).  The pool is a `SlotIndexedState` sibling of
`MambaPool` (registered with `register_slot_state`), so it rides along on clear / copy-on-write / host round-trip exactly
like the Qwen4-Exp PLE side states.  `MambaPool.temporal` is allocated EMPTY when this pool is on: the dense state exists
only (a) transiently during an extend (per-layer scratch) and (b) in a small `dense_ring` that keeps the exact dense
state of the few requests that continue a chunked prefill.

Slot flags (all layers share them):
    stale[slot]     1 = the factored form is authoritative (a decode step ran / the slot was copied or tracked into),
                    0 = the dense state in the ring is exact and authoritative (right after an extend).
    dense_of[slot]  ring position holding this slot's exact dense state, -1 = none.  Validated against the host-side
                    `ring_owner` before use (a ring position may have been re-assigned since).

Rollback / commit-point interface (NOT implemented in K1, docs/62 §1.5): `snapshot_commit(slots)` / `rollback(slots)`
would keep (a, U[:count], W[:count], count) at the last commit point and restore it; draft steps append columns without
triggering the count == r+m truncation (RMAX >= r + m + k_draft).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.configs.mamba_utils import BaseLinearStateParams

logger = logging.getLogger(__name__)


# ============================================================================ config
@dataclass
class FactoredGDNConfig:
    r: int = 8
    m: int = 8
    dtype: torch.dtype = torch.bfloat16  # U, W factors (a stays fp32)
    vbar_path: Optional[str] = None  # consts.pt with ["vbar"][layer] (HV, V) fp32; None = zeros (pure low-rank control arm)
    ring: int = 16  # dense-ring positions (exact dense states kept for chunked-prefill continuation)
    init_iters: int = 4  # subspace-iteration rounds of the prefill-end factorisation
    init_oversample: int = 8
    # K2 (docs/63 §4) decode-kernel options: kernel = split (K1: expiry-truncation launch + step launch) | fused (K2: one
    # launch, the expiring program truncates in registers first); None = the kernel module's defaults (env-overridable)
    kernel: Optional[str] = None
    trunc_warps: Optional[int] = None  # split: num_warps of the expiry-truncation launch
    trunc_iters: Optional[int] = None  # subspace-iteration rounds of the truncation (both kernels)
    fused_warps: Optional[int] = None  # fused: num_warps
    async_trunc: int = 1  # split kernel: run the expiry truncation on a side stream after the step (docs/63 §4); 0 = K1 order
    raw: str = ""

    def kernel_kwargs(self) -> dict:
        return dict(kernel=self.kernel, trunc_warps=self.trunc_warps, trunc_iters=self.trunc_iters, fused_warps=self.fused_warps)

    @property
    def use_async_trunc(self) -> bool:
        return bool(self.async_trunc) and (self.kernel or "split") == "split"

    @property
    def rfull(self) -> int:
        return self.r + self.m

    @property
    def rmax(self) -> int:
        n = self.rfull
        p = 16
        while p < n:
            p *= 2
        return p

    @staticmethod
    def parse(s: Optional[str]) -> Optional["FactoredGDNConfig"]:
        """'r=8,m=8,dtype=bf16,vbar=/path/consts.pt,ring=16' -> config; None / '' -> None (feature off)."""
        if not s:
            return None
        cfg = FactoredGDNConfig(raw=s)
        for kv in filter(None, (x.strip() for x in s.split(","))):
            k, _, v = kv.partition("=")
            k = k.strip()
            v = v.strip()
            if k in ("r", "m", "ring", "init_iters", "init_oversample", "trunc_warps", "trunc_iters", "fused_warps"):
                setattr(cfg, k, int(v))
            elif k in ("async", "async_trunc"):
                cfg.async_trunc = int(v)
            elif k == "kernel":
                assert v in ("split", "fused"), f"linear_attn_factored_state: kernel must be split | fused, got {v!r}"
                cfg.kernel = v
            elif k == "dtype":
                cfg.dtype = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp32": torch.float32,
                             "float32": torch.float32}[v]
            elif k == "vbar":
                cfg.vbar_path = v or None
            else:
                raise ValueError(f"linear_attn_factored_state: unknown key {k!r} in {s!r}")
        assert cfg.r >= 1 and cfg.m >= 1 and cfg.rfull <= 32, (
            f"linear_attn_factored_state: K1 supports r + m <= 32 (truncation tile RMAX 16 | 32), got r={cfg.r} m={cfg.m}")
        return cfg

    # ---- byte accounting (per slot, per layer, one TP rank)
    def state_bytes_per_layer(self, shape) -> int:
        hv, v, k = shape.temporal
        return hv * k * 4 + 2 * hv * self.rmax * max(k, v) * (2 if self.dtype == torch.bfloat16 else 4) + hv * 4

    def ring_bytes(self, shape, num_layers: int) -> int:
        hv, v, k = shape.temporal
        return num_layers * self.ring * hv * v * k * 4

    def per_req_bytes(self, cache_params: BaseLinearStateParams) -> int:
        """conv window + factored state, all layers (replaces cache_params.mamba_cache_per_req for pool sizing)."""
        import numpy as np

        conv_numel = int(np.sum([np.prod(cs) for cs in cache_params.shape.conv]))
        per_layer = conv_numel * cache_params.dtype.conv.itemsize + self.state_bytes_per_layer(cache_params.shape)
        return per_layer * len(cache_params.layers)


# ============================================================================ torch helpers (K0 mirrors)
def gram_schmidt(Y: torch.Tensor, passes: int = 2, rel_tol: float = 1e-4) -> torch.Tensor:
    """Orthonormal basis of col(Y) (..., n, r) by modified Gram-Schmidt `passes` times; a column whose first-pass residual
    is below rel_tol x its original norm is dropped (zero column).  Mirror of twinstar.kernels.gdn_factored.gram_schmidt."""
    Q = Y.clone()
    r = Y.shape[-1]
    n0 = Y.norm(dim=-2)
    for p in range(passes):
        for j in range(r):
            y = Q[..., :, j]
            if j > 0:
                Qp = Q[..., :, :j]
                y = y - (Qp * (Qp * y[..., None]).sum(-2, keepdim=True)).sum(-1)
            n = y.norm(dim=-1, keepdim=True)
            ok = n > 1e-12
            if p == 0:
                ok = ok & (n > rel_tol * n0[..., j : j + 1])
            Q[..., :, j] = torch.where(ok, y / n.clamp_min(1e-30), torch.zeros_like(y))
    return Q


def _topk_onehot(d: torch.Tensor, r: int) -> torch.Tensor:
    idx = d.topk(r, dim=-1).indices
    Z = torch.zeros(*d.shape, r, device=d.device, dtype=d.dtype)
    Z.scatter_(-2, idx[..., None, :], 1.0)
    return Z


def orthonormalize(Y: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis of col(Y) (..., n, k) for the prefill-end factorisation: one Triton launch running K0's two-pass
    MGS with the rank tolerance on every matrix of the batch (dependent columns -> zero; the Rayleigh-Ritz step ranks
    them at ~0 energy).  History (docs/62 §3.4): the column-loop torch MGS cost ~200 launches per call (flag-on prefill
    3x slower in-engine, AGA 783233); torch.linalg.qr loops over the batch inside cusolver and was slower still (783378)."""
    if not Y.is_cuda:
        return gram_schmidt(Y)
    from sglang.srt.layers.attention.linear.kernels.gdn_factored import orthonormalize_columns

    return orthonormalize_columns(Y)


def factorize_dense(S: torch.Tensor, vbar: torch.Tensor, r: int, rmax: int, dtype: torch.dtype, iters: int = 4,
                    oversample: int = 8, method: str = "iter") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """S (B, HV, V, K) fp32 sglang layout, vbar (HV, V) fp32 -> a (B, HV, K) fp32, U (B, HV, RMAX, K), W (B, HV, RMAX, V)
    in `dtype` with rows >= r zero.  a = S^T vbar / |vbar|^2 (least squares; C = S - vbar a^T has C^T vbar = 0); U rows =
    the top-r eigenvectors of G = C^T C (K x K) = left singular vectors of the K0-layout content C^T; W rows = (C U_j).
    method "iter" = randomised subspace iteration + Rayleigh-Ritz (matmul only, docs/60 §3.1 item 5: = SVD to 1.000);
    "svd" = torch.linalg.svd reference (cusolver, tests only)."""
    B, HV, V, K = S.shape
    S = S.float()
    vb = vbar.float()
    nv = (vb * vb).sum(-1).clamp_min(1e-30)  # (HV,)
    a = torch.einsum("bhvk,hv->bhk", S, vb) / nv[None, :, None]
    C = S - vb[None, :, :, None] * a[:, :, None, :]  # (B, HV, V, K)
    Ct = C.transpose(-1, -2)  # (B, HV, K, V) = K0-layout content
    if method == "svd":
        P, _, _ = torch.linalg.svd(Ct, full_matrices=False)  # (B, HV, K, K)
        P = P[..., :r]
    else:
        gen = torch.Generator(device=S.device).manual_seed(0)
        Om = torch.randn(B, HV, V, r + oversample, device=S.device, dtype=torch.float32, generator=gen)
        Y = Ct @ Om  # (B, HV, K, r+p)
        for _ in range(iters):
            Y = orthonormalize(Y)
            Y = Ct @ (C @ Y)
        Q = orthonormalize(Y)  # (B, HV, K, r+p)
        T = C @ Q  # (B, HV, V, r+p) = C^T-rows in the subspace
        Gs = T.transpose(-1, -2) @ T  # (B, HV, r+p, r+p)
        Zs = _topk_onehot(Gs.diagonal(dim1=-2, dim2=-1), r)
        for _ in range(iters):
            Zs = orthonormalize(Gs @ (Gs @ Zs))
        P = Q @ Zs  # (B, HV, K, r)
    U = torch.zeros(B, HV, rmax, K, device=S.device, dtype=dtype)
    W = torch.zeros(B, HV, rmax, V, device=S.device, dtype=dtype)
    U[:, :, :r] = P.transpose(-1, -2).to(dtype)
    W[:, :, :r] = (C @ P).transpose(-1, -2).to(dtype)  # row j = C P_j (V)
    return a, U, W


def densify(a: torch.Tensor, U: torch.Tensor, W: torch.Tensor, count: torch.Tensor, vbar: torch.Tensor) -> torch.Tensor:
    """(B, HV, K) / (B, HV, RMAX, K) / (B, HV, RMAX, V) / (B, HV) -> S (B, HV, V, K) fp32 sglang layout."""
    RMAX = U.shape[2]
    rows = torch.arange(RMAX, device=U.device)[None, None, :] < count[:, :, None]  # (B, HV, RMAX)
    Uf = U.float() * rows[..., None]
    Wf = W.float() * rows[..., None]
    S = torch.einsum("bhrv,bhrk->bhvk", Wf, Uf)
    return S + vbar.float()[None, :, :, None] * a.float()[:, :, None, :]


# ============================================================================ extend plan (host-side ring bookkeeping)
@dataclass
class FactoredExtendPlan:
    slots: torch.Tensor  # (B,) int64 device: mamba slot per row
    use_ring: torch.Tensor  # (B,) bool device: initial dense state from the ring (exact) instead of densify
    ring_src: torch.Tensor  # (B,) int64 device: ring position (0 where not used)
    ring_dst: torch.Tensor  # (B,) int64 device: ring position to store the final dense state, -1 = none
    ring_dst_rows: torch.Tensor  # (n,) int64 device: rows with ring_dst >= 0
    n_ring_src: int = 0
    n_ring_miss: int = 0


# ============================================================================ the pool
class FactoredGDNPool:
    """SlotIndexedState sibling of MambaPool holding the factored GDN state of every linear layer."""

    def __init__(self, *, size: int, cache_params: BaseLinearStateParams, mamba_layer_ids: List[int], device,
                 cfg: FactoredGDNConfig, tp_rank: int = 0):
        self.cfg = cfg
        self.size = size
        self.device = device
        self.layer_ids = list(mamba_layer_ids)
        self.layer_map = {lid: i for i, lid in enumerate(self.layer_ids)}
        hv, v, k = cache_params.shape.temporal
        self.hv, self.v, self.k = hv, v, k
        L, S = len(self.layer_ids), size + 1
        R = cfg.rmax
        self.a = torch.zeros(L, S, hv, k, dtype=torch.float32, device=device)
        self.U = torch.zeros(L, S, hv, R, k, dtype=cfg.dtype, device=device)
        self.W = torch.zeros(L, S, hv, R, v, dtype=cfg.dtype, device=device)
        self.count = torch.full((L, S, hv), cfg.r, dtype=torch.int32, device=device)
        self.stale = torch.ones(S, dtype=torch.int32, device=device)
        self.dense_of = torch.full((S,), -1, dtype=torch.int32, device=device)
        self.dense_ring = torch.zeros(L, cfg.ring, hv, v, k, dtype=torch.float32, device=device)
        self.ring_owner: List[int] = [-1] * cfg.ring  # host mirror: slot owning each ring position
        self.ring_lru: List[int] = list(range(cfg.ring))  # least recently used first
        self.vbar = self._load_vbar(cfg.vbar_path, tp_rank)  # (L, hv, v) fp32
        self.stats: Dict[str, int] = {"extends": 0, "rows": 0, "ring_src": 0, "ring_miss": 0, "densified": 0}
        state_mb = self.cfg.state_bytes_per_layer(cache_params.shape) * L * S / (1 << 20)
        ring_mb = self.dense_ring.numel() * 4 / (1 << 20)
        logger.info(
            "Factored GDN pool allocated (docs/62): %s; layers %d, slots %d, HV %d, RMAX %d, factors %s; "
            "state %.1f MB (%.1f KB/layer/slot vs dense %.1f KB), dense ring %d x %.1f MB, vbar %s",
            cfg.raw, L, S, hv, R, str(cfg.dtype).replace("torch.", ""), state_mb,
            self.cfg.state_bytes_per_layer(cache_params.shape) / 1024, hv * v * k * 4 / 1024, cfg.ring,
            ring_mb / max(cfg.ring, 1), "zeros (pure low-rank control)" if cfg.vbar_path is None else cfg.vbar_path,
        )

    # ------------------------------------------------------------------ constants
    def _load_vbar(self, path: Optional[str], tp_rank: int) -> torch.Tensor:
        L = len(self.layer_ids)
        out = torch.zeros(L, self.hv, self.v, dtype=torch.float32, device=self.device)
        if path is None:
            return out
        if not os.path.exists(path):
            raise FileNotFoundError(f"linear_attn_factored_state vbar={path} not found")
        consts = torch.load(path, map_location="cpu")
        vb = consts["vbar"] if isinstance(consts, dict) and "vbar" in consts else consts
        lo = tp_rank * self.hv
        missing = []
        for i, lid in enumerate(self.layer_ids):
            t = vb.get(lid) if isinstance(vb, dict) else vb[lid]
            if t is None:
                missing.append(lid)
                continue
            t = torch.as_tensor(t).float()
            assert t.shape[-1] == self.v and t.shape[0] >= lo + self.hv, (lid, t.shape, lo, self.hv)
            out[i] = t[lo : lo + self.hv].to(self.device)
        if missing:
            logger.warning("Factored GDN pool: vbar missing for layers %s (zeros used)", missing)
        return out

    # ------------------------------------------------------------------ SlotIndexedState protocol
    @property
    def enabled(self) -> bool:
        return True

    def reset_slots(self, indices: torch.Tensor) -> None:
        if indices.numel() == 0:
            return
        self.a[:, indices] = 0
        self.U[:, indices] = 0
        self.W[:, indices] = 0
        self.count[:, indices] = self.cfg.r
        self.stale[indices] = 1
        self.dense_of[indices] = -1

    def copy_slots(self, src_index: torch.Tensor, dst_index: torch.Tensor) -> None:
        if src_index.numel() == 0:
            return
        self.a[:, dst_index] = self.a[:, src_index]
        self.U[:, dst_index] = self.U[:, src_index]
        self.W[:, dst_index] = self.W[:, src_index]
        self.count[:, dst_index] = self.count[:, src_index]
        self.stale[dst_index] = 1  # a copied slot is factored-only (compact prefix cache)
        self.dense_of[dst_index] = -1

    def get_cpu_slots(self, indices: torch.Tensor) -> Any:
        return (self.a[:, indices].to("cpu", non_blocking=True), self.U[:, indices].to("cpu", non_blocking=True),
                self.W[:, indices].to("cpu", non_blocking=True), self.count[:, indices].to("cpu", non_blocking=True))

    def load_cpu_slots(self, data: Any, indices: torch.Tensor) -> None:
        if data is None:
            return
        a, U, W, c = data
        self.a[:, indices] = a.to(self.device, non_blocking=True)
        self.U[:, indices] = U.to(self.device, non_blocking=True)
        self.W[:, indices] = W.to(self.device, non_blocking=True)
        self.count[:, indices] = c.to(self.device, non_blocking=True)
        self.stale[indices] = 1
        self.dense_of[indices] = -1

    def iter_transfer_state_entries(self):
        for lid, li in self.layer_map.items():
            yield ("gdn_factored_a", self.a[li], 0, lid)
            yield ("gdn_factored_u", self.U[li], 0, lid)
            yield ("gdn_factored_w", self.W[li], 0, lid)
            yield ("gdn_factored_count", self.count[li], 0, lid)

    # ------------------------------------------------------------------ accessors
    def layer_index(self, layer_id: int) -> int:
        return self.layer_map[layer_id]

    def is_last_layer(self, layer_id: int) -> bool:
        return self.layer_map[layer_id] == len(self.layer_ids) - 1

    def layer_tensors(self, layer_id: int):
        li = self.layer_map[layer_id]
        return self.a[li], self.U[li], self.W[li], self.count[li], self.vbar[li]

    def mem_usage_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.a, self.U, self.W, self.count, self.stale, self.dense_of,
                                                            self.dense_ring, self.vbar))

    # ------------------------------------------------------------------ extend: plan (host, once per forward)
    def plan_extend(self, slots: torch.Tensor, extend_lens: Sequence[int]) -> FactoredExtendPlan:
        """Decide per row where the exact dense initial state comes from and where the final dense state goes.
        One D2H sync (three small gathers); called from init_forward_metadata for extend batches."""
        B = slots.shape[0]
        extend_lens = list(extend_lens) + [0] * max(0, B - len(extend_lens))  # attn-TP padded rows
        slots64 = slots.to(torch.long)
        safe = slots64.clamp(min=0)
        slots_cpu = slots64.tolist()
        stale_cpu = self.stale[safe].tolist()
        dense_cpu = self.dense_of[safe].tolist()
        use_ring = [False] * B
        ring_src = [0] * B
        for i in range(B):
            s, d = slots_cpu[i], dense_cpu[i]
            if s >= 0 and d >= 0 and self.ring_owner[d] == s and stale_cpu[i] == 0:
                use_ring[i] = True
                ring_src[i] = d
        # destination ring positions: keep an owned position, else allocate (longest extends first)
        ring_dst = [-1] * B
        order = sorted(range(B), key=lambda i: -int(extend_lens[i]))
        taken = set()
        for i in order:
            s = slots_cpu[i]
            if s < 0:
                continue
            d = dense_cpu[i]
            if d >= 0 and self.ring_owner[d] == s and d not in taken:
                ring_dst[i] = d
                taken.add(d)
        owners_stale = None
        for i in order:
            s = slots_cpu[i]
            if s < 0 or ring_dst[i] >= 0:
                continue
            free = [p for p in self.ring_lru if self.ring_owner[p] < 0 and p not in taken]
            if free:
                p = free[0]
            else:
                if owners_stale is None:
                    own = torch.tensor([max(o, 0) for o in self.ring_owner], device=self.device, dtype=torch.long)
                    owners_stale = self.stale[own].tolist()
                cand = [p for p in self.ring_lru if p not in taken and (owners_stale[p] == 1 or self.ring_owner[p] < 0)]
                if not cand:
                    cand = [p for p in self.ring_lru if p not in taken]
                if not cand:
                    ring_dst[i] = -1
                    continue
                p = cand[0]
            ring_dst[i] = p
            taken.add(p)
        for i in range(B):
            p = ring_dst[i]
            if p >= 0:
                self.ring_owner[p] = slots_cpu[i]
                self.ring_lru.remove(p)
                self.ring_lru.append(p)
        dev = self.device
        ring_dst_t = torch.tensor(ring_dst, dtype=torch.long, device=dev)
        plan = FactoredExtendPlan(
            slots=slots64,
            use_ring=torch.tensor(use_ring, dtype=torch.bool, device=dev),
            ring_src=torch.tensor(ring_src, dtype=torch.long, device=dev),
            ring_dst=ring_dst_t,
            ring_dst_rows=torch.tensor([i for i in range(B) if ring_dst[i] >= 0], dtype=torch.long, device=dev),
            n_ring_src=sum(use_ring),
            n_ring_miss=sum(1 for i in range(B) if ring_dst[i] < 0 and slots_cpu[i] >= 0),
        )
        # device-side ownership for validation on the next extend
        self.dense_of[safe] = ring_dst_t.to(torch.int32)
        self.stats["extends"] += 1
        self.stats["rows"] += B
        self.stats["ring_src"] += plan.n_ring_src
        self.stats["ring_miss"] += plan.n_ring_miss
        return plan

    # ------------------------------------------------------------------ extend: per-layer dense in / factored out
    def initial_dense(self, layer_id: int, plan: FactoredExtendPlan) -> torch.Tensor:
        """(B, HV, V, K) fp32 initial states for the chunk kernel: exact ring copies where available, else densified."""
        li = self.layer_map[layer_id]
        safe = plan.slots.clamp(min=0)
        S = densify(self.a[li][safe], self.U[li][safe], self.W[li][safe], self.count[li][safe], self.vbar[li])
        if plan.n_ring_src:
            ring = self.dense_ring[li][plan.ring_src]
            S = torch.where(plan.use_ring[:, None, None, None], ring, S)
        self.stats["densified"] += plan.slots.shape[0] - plan.n_ring_src
        return S.contiguous()

    def commit_extend(self, layer_id: int, plan: FactoredExtendPlan, S_final: torch.Tensor) -> None:
        """Factorise the final dense states of the batch into the slots (count = r, stale = 0) and keep the exact dense
        state in the ring for the rows that got a ring position."""
        li = self.layer_map[layer_id]
        cfg = self.cfg
        a, U, W = factorize_dense(S_final, self.vbar[li], cfg.r, cfg.rmax, cfg.dtype, iters=cfg.init_iters,
                                  oversample=cfg.init_oversample)
        valid = plan.slots >= 0
        safe = plan.slots.clamp(min=0)
        # rows with slot < 0 (padding) must not write: route them to a scratch copy of slot 0's data
        if bool(valid.all()):
            self.a[li][safe] = a
            self.U[li][safe] = U
            self.W[li][safe] = W
            self.count[li][safe] = cfg.r
            self.stale[safe] = 0
        else:
            idx = valid.nonzero(as_tuple=True)[0]
            sl = plan.slots[idx]
            self.a[li][sl] = a[idx]
            self.U[li][sl] = U[idx]
            self.W[li][sl] = W[idx]
            self.count[li][sl] = cfg.r
            self.stale[sl] = 0
        if plan.ring_dst_rows.numel():
            self.dense_ring[li][plan.ring_dst[plan.ring_dst_rows]] = S_final[plan.ring_dst_rows]

    def write_factored_dense(self, layer_id: int, slots: torch.Tensor, S_dense: torch.Tensor) -> None:
        """Factorise dense states (n, HV, V, K) into arbitrary slots (radix track destinations): factored-only, stale."""
        if slots.numel() == 0:
            return
        li = self.layer_map[layer_id]
        cfg = self.cfg
        a, U, W = factorize_dense(S_dense.float(), self.vbar[li], cfg.r, cfg.rmax, cfg.dtype, iters=cfg.init_iters,
                                  oversample=cfg.init_oversample)
        sl = slots.to(torch.long)
        self.a[li][sl] = a
        self.U[li][sl] = U
        self.W[li][sl] = W
        self.count[li][sl] = cfg.r
        self.stale[sl] = 1
        self.dense_of[sl] = -1

    def copy_slots_layer(self, layer_id: int, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Per-layer slot copy (extend-time `track_ssm_final` tracking); dst becomes factored-only."""
        if src.numel() == 0:
            return
        li = self.layer_map[layer_id]
        s, d = src.to(torch.long), dst.to(torch.long)
        self.a[li][d] = self.a[li][s]
        self.U[li][d] = self.U[li][s]
        self.W[li][d] = self.W[li][s]
        self.count[li][d] = self.count[li][s]
        self.stale[d] = 1
        self.dense_of[d] = -1

    def abandon_ring(self, plan: FactoredExtendPlan) -> None:
        """The extend did not produce dense final states for this plan (stepwise debug path): release the ring
        positions it reserved so a later extend does not read stale dense data."""
        for i, p in enumerate(plan.ring_dst.tolist()):
            if p >= 0:
                self.ring_owner[p] = -1
        safe = plan.slots.clamp(min=0)
        self.dense_of[safe] = -1

    def dump_slots(self, layer_id: int, slots: torch.Tensor, meta: dict, out_dir: str, tag: str) -> None:
        """Debug (docs/62 §3.3): save (a, U, W, count) of `slots` for one layer."""
        li = self.layer_map[layer_id]
        s = slots.to(torch.long)
        os.makedirs(out_dir, exist_ok=True)
        n = self._dump_n = getattr(self, "_dump_n", 0) + 1
        torch.save({"kind": "factored", "layer": layer_id, "slots": s.cpu(), "a": self.a[li][s].cpu(), "U": self.U[li][s].cpu(),
                    "W": self.W[li][s].cpu(), "count": self.count[li][s].cpu(), "vbar": self.vbar[li].cpu(), **meta},
                   os.path.join(out_dir, f"{tag}_{n:05d}_L{layer_id:02d}.pt"))

    # ------------------------------------------------------------------ decode tracking (all layers, graph safe)
    def track_copy(self, src_idx: torch.Tensor, mask: torch.Tensor, dst_idx: torch.Tensor) -> None:
        from sglang.srt.layers.attention.linear.kernels.gdn_factored import factored_track_copy

        factored_track_copy(self.a, self.U, self.W, self.count, self.stale, src_idx, mask, dst_idx)

    # ------------------------------------------------------------------ commit-point interface (K1: not implemented)
    def snapshot_commit(self, slots: torch.Tensor) -> None:  # pragma: no cover
        raise NotImplementedError("docs/62 §1.5: commit snapshot (a, U[:count], W[:count], count) is a K2 item")

    def rollback(self, slots: torch.Tensor) -> None:  # pragma: no cover
        raise NotImplementedError("docs/62 §1.5: rollback restores the commit snapshot; K1 serves without speculation")

    # ------------------------------------------------------------------ debug
    def dense_of_slots(self, layer_id: int, slots: torch.Tensor) -> torch.Tensor:
        li = self.layer_map[layer_id]
        s = slots.to(torch.long)
        return densify(self.a[li][s], self.U[li][s], self.W[li][s], self.count[li][s], self.vbar[li])
