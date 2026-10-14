"""GLM dense W32 cut in the GDN precision class and algorithm (twinstar docs/210 §11).

The GLM KDA pool keeps the dense FP32 state S (V, K) per (layer, slot, head); the cut keeps the explicit right sink
vbar a^T (a = S^T vbar / |vbar|^2, the GDN sink) and projects the content C = S - vbar a^T onto a K-side rank-R basis Z:
S <- vbar a^T + C Z Z^T, with Z kept per (layer, slot, head) as the next cut's start.

  cold (prompt end, eager, outside the decode graph): GDN's prefill-end `factorize_dense` (randomised subspace iteration
       + Rayleigh-Ritz, FP32 Cholesky-QR), with GLM's fixed OMEGA_SEED probe of width R + 8 shared by every row.
  warm (every W decode steps, in the decode graph): GDN's expiry truncation -- TRUNC_ITERS rounds of Z <- MGS2(G Z) with
       G = C^T C and GDN's rank tolerance (`_mgs_rolled`, MGS_REL_TOL), started from the previous kept basis; a basis
       column that is missing (zero) restarts from GDN's start rule, the coordinate with the next-largest diagonal of G.
       G is never formed: each round streams S in CH-row tiles, C^T (C Z) by two FP32 `tl.dot` (ieee).

One program per (layer, row, head), several programs per SM; programs of non-due rows return before any matrix work.
"""

import torch
import triton
import triton.language as tl

from .gdn_factored import MGS_REL_TOL, TRUNC_ITERS, _mgs_rolled


@triton.jit(do_not_specialize=["B"])
def _any_due(Indices, Phase, Pending, Active, Flag, B, W: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    slot = tl.load(Indices + row, row < B, -1).to(tl.int64)
    valid = (row < B) & (slot >= 0)
    active = tl.load(Active + slot, valid, 0)
    pending = tl.load(Pending + slot, valid, 0)
    phase = tl.load(Phase + slot, valid, 0)
    due = valid & (active != 0) & (pending == 0) & (phase + 1 == W)
    tl.store(Flag, tl.max(due.to(tl.int32), 0) > 0)


@triton.jit(do_not_specialize=["B", "FIRST", "SLOTS"])
def _warm_cut(
    S,
    VB,
    Warm,
    Indices,
    Phase,
    Pending,
    Active,
    B,
    H: tl.constexpr,
    SLOTS,
    FIRST,
    W: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    CH: tl.constexpr,
    ITERS: tl.constexpr,
    REL_TOL: tl.constexpr,
):
    pid = tl.program_id(0)
    row, head, layer = (pid // H) % B, pid % H, FIRST + pid // (B * H)
    slot = tl.load(Indices + row).to(tl.int64)
    if slot < 0:
        return
    active = tl.load(Active + slot)
    pending = tl.load(Pending + slot)
    phase = tl.load(Phase + slot)
    if not (active & ~pending & (phase + 1 == W)):
        return
    k, r, c = tl.arange(0, D), tl.arange(0, R), tl.arange(0, CH)
    entry = (layer * SLOTS + slot) * H + head
    s_ptr = S + entry * D * D
    v_ptr = VB + (layer * H + head) * D
    w_ptr = Warm + entry * D * R
    vb = tl.load(v_ptr + k)
    nv = tl.maximum(tl.sum(vb * vb, 0), 1e-12)
    sv = tl.zeros([D], tl.float32)
    ssq = tl.zeros([D], tl.float32)
    for v0 in range(0, D, CH):
        s = tl.load(s_ptr + (v0 + c)[:, None] * D + k[None, :])
        vbc = tl.load(v_ptr + v0 + c)
        sv += tl.sum(s * vbc[:, None], 0)
        ssq += tl.sum(s * s, 0)
    a = sv / nv
    Z = tl.load(w_ptr + k[:, None] * R + r[None, :])
    norms = tl.sum(Z * Z, 0)
    if tl.min(norms, 0) < 0.25:
        dg = ssq - a * a * nv  # diag(C^T C)
        better = (dg[None, :] > dg[:, None]) | ((dg[None, :] == dg[:, None]) & (k[None, :] < k[:, None]))
        rank = tl.sum(better.to(tl.int32), 1)
        Z = tl.where(norms[None, :] < 0.25, tl.where(rank[:, None] == r[None, :], 1.0, 0.0), Z)
    for _ in range(ITERS):
        za = tl.sum(a[:, None] * Z, 0)
        acc = tl.zeros([D, R], tl.float32)
        for v0 in range(0, D, CH):
            s = tl.load(s_ptr + (v0 + c)[:, None] * D + k[None, :])
            vbc = tl.load(v_ptr + v0 + c)
            cz = tl.dot(s, Z, input_precision="ieee") - vbc[:, None] * za[None, :]
            czv = tl.sum(vbc[:, None] * cz, 0)
            acc += tl.dot(tl.trans(s), cz, input_precision="ieee") - a[:, None] * czv[None, :]
        Z = _mgs_rolled(acc, r, R, 2, REL_TOL)
    za = tl.sum(a[:, None] * Z, 0)
    for v0 in range(0, D, CH):
        s = tl.load(s_ptr + (v0 + c)[:, None] * D + k[None, :])
        vbc = tl.load(v_ptr + v0 + c)
        cz = tl.dot(s, Z, input_precision="ieee") - vbc[:, None] * za[None, :]
        new = vbc[:, None] * a[None, :] + tl.dot(cz, tl.trans(Z), input_precision="ieee")
        tl.store(s_ptr + (v0 + c)[:, None] * D + k[None, :], new)
    tl.store(w_ptr + k[:, None] * R + r[None, :], Z)


def due_flag(indices, phase, pending, active, flag, window):
    """flag (0-dim bool) <- some row of this batch is at its warm cut this step."""
    _any_due[(1,)](
        indices, phase, pending, active, flag, indices.numel(), window,
        triton.next_power_of_2(max(indices.numel(), 16)),
    )


def warm_cut(states, vbar, warm, indices, phase, pending, active, *, first, layers, window, num_warps=4, chunk=32):
    layers_total, slots, heads, d, d2 = states.shape
    rank = warm.shape[-1]
    assert d == d2 and warm.shape == (layers_total, slots, heads, d, rank)
    assert states.dtype == warm.dtype == torch.float32 and states.is_contiguous() and warm.is_contiguous()
    assert indices.is_contiguous()
    batch = indices.numel()
    _warm_cut[(layers * batch * heads,)](
        states, vbar, warm, indices, phase, pending, active,
        batch, heads, slots, first, window,
        D=d, R=rank, CH=chunk, ITERS=TRUNC_ITERS, REL_TOL=MGS_REL_TOL,
        num_warps=num_warps,
    )


def cold_cut(states, vbar, omega, warm, slots, *, iters=None, chunk=4):
    """Prompt-end cut of every layer of `slots` with GDN's factorize_dense; writes the dense projection and the basis."""
    from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNConfig, factorize_dense

    iters = FactoredGDNConfig.init_iters if iters is None else iters
    layers, _, heads, v, k = states.shape
    rank = warm.shape[-1]
    vb = vbar.reshape(layers * heads, v)
    slots = slots.reshape(-1).long()
    for i in range(0, slots.numel(), chunk):
        sel = slots[i : i + chunk]
        n = sel.numel()
        dense = states[:, sel].transpose(0, 1).reshape(n, layers * heads, v, k)
        probe = omega[None, None].expand(n, layers, heads, v, omega.shape[-1]).reshape(n, layers * heads, v, -1)
        a, u, w = factorize_dense(
            dense, vb, rank, rank, torch.float32, iters=iters, oversample=omega.shape[-1] - rank, omega=probe,
            method="iter",
        )
        new = vb[None, :, :, None] * a[:, :, None, :] + w.transpose(-1, -2) @ u
        states[:, sel] = new.reshape(n, layers, heads, v, k).transpose(0, 1)
        warm[:, sel] = u.transpose(-1, -2).reshape(n, layers, heads, k, rank).transpose(0, 1)
