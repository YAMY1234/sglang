"""Unpadded Kimi small-Gram FP64 Jacobi, specialized from Flash-Next.

One CTA per head matrix; all convergence decisions remain on device. The
logical Gram is 40x40 for r32, while Triton register tiles round to 64x64.
Inactive tile lanes are masked; there is no spectral padding allocation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _eigh_kernel(
    G_ptr,
    D_ptr,
    Z_ptr,
    N: tl.constexpr,
    TILE: tl.constexpr,
    MAX_SWEEPS: tl.constexpr,
    REL_TOL: tl.constexpr,
):
    matrix = tl.program_id(0).to(tl.int64)
    x = tl.arange(0, TILE)
    valid = x < N
    mask = valid[:, None] & valid[None, :]
    eye = x[:, None] == x[None, :]
    offsets = matrix * N * N + x[:, None] * N + x[None, :]
    g = tl.load(G_ptr + offsets, mask=mask, other=0.0)
    z = (eye & mask).to(tl.float64)
    scale = tl.maximum(tl.max(tl.max(tl.abs(g), 1), 0), 1e-300)
    g = g / scale
    sweep = 0
    active = True
    while (sweep < MAX_SWEEPS) & active:
        off2 = tl.sum(tl.sum(tl.where(eye | ~mask, 0.0, g * g), 1), 0)
        norm2 = tl.sum(tl.sum(g * g, 1), 0)
        active = off2 > REL_TOL * REL_TOL * norm2
        if active:
            # N is the logical dimension: 40 uses 39 turns, never 63.
            for turn in range(N - 1):
                partner = tl.where(
                    x == N - 1,
                    turn,
                    tl.where(x == turn, N - 1, (2 * turn - x + 2 * (N - 1)) % (N - 1)),
                )
                partner = tl.where(valid, partner, x)
                diag = tl.sum(tl.where(eye, g, 0.0), 1)
                other = tl.gather(diag, partner, 0)
                cross = tl.sum(tl.where(x[None, :] == partner[:, None], g, 0.0), 1)
                cross = 0.5 * (cross + tl.gather(cross, partner, 0))
                tau = (diag - other) / tl.where(tl.abs(cross) > 1e-300, 2 * cross, 1.0)
                t = tl.where(tau >= 0, 1.0, -1.0) / (
                    tl.abs(tau) + tl.sqrt(1 + tau * tau)
                )
                t = tl.where(diag == other, tl.where(x < partner, 1.0, -1.0), t)
                t = tl.where(
                    valid & (tl.abs(cross) > 1e-30 * tl.sqrt(tl.abs(diag * other))),
                    t,
                    0.0,
                )
                c = 1.0 / tl.sqrt(1 + t * t)
                sine = t * c
                pc = tl.broadcast_to(partner[None, :], (TILE, TILE))
                pr = tl.broadcast_to(partner[:, None], (TILE, TILE))
                g = g * c[None, :] + tl.gather(g, pc, 1) * sine[None, :]
                g = g * c[:, None] + tl.gather(g, pr, 0) * sine[:, None]
                z = z * c[None, :] + tl.gather(z, pc, 1) * sine[None, :]
        sweep += 1
    d = tl.sum(tl.where(eye, g, 0.0), 1)
    rank = tl.sum(
        (
            valid[None, :]
            & (
                (d[None, :] < d[:, None])
                | ((d[None, :] == d[:, None]) & (x[None, :] < x[:, None]))
            )
        ).to(tl.int32),
        1,
    )
    order = tl.sum(
        tl.where(valid[:, None] & (rank[:, None] == x[None, :]), x[:, None], 0), 0
    )
    z = tl.gather(z, tl.broadcast_to(order[None, :], (TILE, TILE)), 1)
    d = tl.gather(d, order, 0) * scale
    tl.store(D_ptr + matrix * N + x, d, mask=valid)
    tl.store(Z_ptr + offsets, z, mask=mask)


def eigh(gram, *, rel_tol=1e-14, max_sweeps=12):
    n = gram.shape[-1]
    if gram.dtype != torch.float64 or n < 2 or n > 64 or n % 2 or gram.shape[-2] != n:
        raise ValueError("Kimi Jacobi requires even FP64 matrices of dimension 2..64")
    flat = gram.reshape(-1, n, n).contiguous()
    d = torch.empty(flat.shape[:2], dtype=torch.float64, device=gram.device)
    z = torch.empty_like(flat)
    if flat.shape[0]:
        _eigh_kernel[(flat.shape[0],)](
            flat,
            d,
            z,
            N=n,
            TILE=triton.next_power_of_2(n),
            MAX_SWEEPS=max_sweeps,
            REL_TOL=rel_tol,
            num_warps=8,
        )
    return d.reshape(gram.shape[:-1]), z.reshape(gram.shape)
