"""#1019: graph-capturable replacement for the k31 prompt-end ``torch.linalg.eigh`` (fp64, 16 x 16 Gram matrices).

``torch.linalg.eigh`` checks cuSOLVER's info on the host, which synchronises and cannot run inside a CUDA graph; the
whole-layer prefill commit graph was therefore off for init_method=k31.  This is the same decomposition done by a
round-robin (parallel-ordering) cyclic Jacobi in fp64 -- the algorithm of the fork's graph-safe fp32 decode solver
``gdn_truncate._jacobi_vectors``, widened to fp64 and run a fixed number of sweeps past machine-precision convergence
(16 x 16 converges quadratically in ~6 sweeps).  One program per matrix, no host reads.  Output = eigenvalues ascending
and the matching eigenvector columns, as ``torch.linalg.eigh``; eigenvectors equal the reference's up to sign (and up to
a rotation inside exactly degenerate eigenspaces), so the rank-r projector and the stored state match to ~1e-15.
"""
import os

import torch
import triton
import triton.language as tl

SWEEPS = 12


@triton.jit
def _k31_eigh_kernel(G_ptr, D_ptr, Z_ptr, N: tl.constexpr, SWEEPS: tl.constexpr):
    m = tl.program_id(0).to(tl.int64)
    x = tl.arange(0, N)
    eye = x[:, None] == x[None, :]
    G = tl.load(G_ptr + m * N * N + x[:, None] * N + x[None, :])
    Z = eye.to(tl.float64)
    scale = tl.maximum(tl.max(tl.max(tl.abs(G), 1), 0), 1.e-300)
    G = G / scale
    for sweep in range(SWEEPS):
        for turn in range(N - 1):
            partner = tl.where(x == N - 1, turn,
                               tl.where(x == turn, N - 1, (2 * turn - x + 2 * (N - 1)) % (N - 1)))
            diag = tl.sum(tl.where(eye, G, 0.), 1)
            other = tl.gather(diag, partner, 0)
            cross = tl.sum(tl.where(x[None, :] == partner[:, None], G, 0.), 1)
            # one shared (symmetrised) off-diagonal per pair, as in the fp32 solver
            cross = 0.5 * (cross + tl.gather(cross, partner, 0))
            tau = (diag - other) / tl.where(tl.abs(cross) > 1.e-300, 2. * cross, 1.)
            t = tl.where(tau >= 0., 1., -1.) / (tl.abs(tau) + tl.sqrt(1. + tau * tau))
            t = tl.where(diag == other, tl.where(x < partner, 1., -1.), t)
            # rotate unless the pair is already diagonal to well below fp64 resolution
            t = tl.where(tl.abs(cross) > 1.e-20 * tl.sqrt(tl.abs(diag * other)), t, 0.)
            c = 1. / tl.sqrt(1. + t * t)
            s = t * c
            pc = tl.broadcast_to(partner[None, :], (N, N))
            pr = tl.broadcast_to(partner[:, None], (N, N))
            G = G * c[None, :] + tl.gather(G, pc, 1) * s[None, :]
            G = G * c[:, None] + tl.gather(G, pr, 0) * s[:, None]
            Z = Z * c[None, :] + tl.gather(Z, pc, 1) * s[None, :]
    d = tl.sum(tl.where(eye, G, 0.), 1)
    # ascending order (ties by index), as torch.linalg.eigh
    rank = tl.sum(((d[None, :] < d[:, None]) | ((d[None, :] == d[:, None]) & (x[None, :] < x[:, None]))).to(tl.int32), 1)
    order = tl.sum(tl.where(rank[:, None] == x[None, :], x[:, None], 0), 0)
    Z = tl.gather(Z, tl.broadcast_to(order[None, :], (N, N)), 1)
    d = tl.gather(d, order, 0) * scale
    tl.store(D_ptr + m * N + x, d)
    tl.store(Z_ptr + m * N * N + x[:, None] * N + x[None, :], Z)


@triton.jit
def _k31_eigh_kernel_early(G_ptr, D_ptr, Z_ptr, N: tl.constexpr, SWEEPS: tl.constexpr,
                     EARLY_EXIT: tl.constexpr = False):
    m = tl.program_id(0).to(tl.int64)
    x = tl.arange(0, N)
    eye = x[:, None] == x[None, :]
    G = tl.load(G_ptr + m * N * N + x[:, None] * N + x[None, :])
    Z = eye.to(tl.float64)
    scale = tl.maximum(tl.max(tl.max(tl.abs(G), 1), 0), 1.e-300)
    G = G / scale
    sweep = 0
    active = True
    while (sweep < SWEEPS) & active:
        if EARLY_EXIT:
            # A complete sweep is the identity when every pair already takes
            # the existing t=0 branch. Check that *same* per-pair threshold,
            # including its symmetrisation, rather than relaxing convergence
            # or lowering the fixed twelve-sweep upper bound. Nothing leaves
            # the device and the branch is uniform within this matrix's CTA.
            diagonal = tl.sum(tl.where(eye, G, 0.), 1)
            cross_all = 0.5 * (G + tl.trans(G))
            threshold = 1.e-20 * tl.sqrt(tl.abs(
                diagonal[:, None] * diagonal[None, :]))
            rotating = tl.where(eye, False, tl.abs(cross_all) > threshold)
            active = tl.max(tl.max(rotating.to(tl.int32), 1), 0) != 0
        if active:
            for turn in range(N - 1):
                partner = tl.where(x == N - 1, turn,
                                   tl.where(x == turn, N - 1, (2 * turn - x + 2 * (N - 1)) % (N - 1)))
                diag = tl.sum(tl.where(eye, G, 0.), 1)
                other = tl.gather(diag, partner, 0)
                cross = tl.sum(tl.where(x[None, :] == partner[:, None], G, 0.), 1)
                # one shared (symmetrised) off-diagonal per pair, as in the fp32 solver
                cross = 0.5 * (cross + tl.gather(cross, partner, 0))
                tau = (diag - other) / tl.where(tl.abs(cross) > 1.e-300, 2. * cross, 1.)
                t = tl.where(tau >= 0., 1., -1.) / (tl.abs(tau) + tl.sqrt(1. + tau * tau))
                t = tl.where(diag == other, tl.where(x < partner, 1., -1.), t)
                # rotate unless the pair is already diagonal to well below fp64 resolution
                t = tl.where(tl.abs(cross) > 1.e-20 * tl.sqrt(tl.abs(diag * other)), t, 0.)
                c = 1. / tl.sqrt(1. + t * t)
                s = t * c
                pc = tl.broadcast_to(partner[None, :], (N, N))
                pr = tl.broadcast_to(partner[:, None], (N, N))
                G = G * c[None, :] + tl.gather(G, pc, 1) * s[None, :]
                G = G * c[:, None] + tl.gather(G, pr, 0) * s[:, None]
                Z = Z * c[None, :] + tl.gather(Z, pc, 1) * s[None, :]
        sweep += 1
    d = tl.sum(tl.where(eye, G, 0.), 1)
    # ascending order (ties by index), as torch.linalg.eigh
    rank = tl.sum(((d[None, :] < d[:, None]) | ((d[None, :] == d[:, None]) & (x[None, :] < x[:, None]))).to(tl.int32), 1)
    order = tl.sum(tl.where(rank[:, None] == x[None, :], x[:, None], 0), 0)
    Z = tl.gather(Z, tl.broadcast_to(order[None, :], (N, N)), 1)
    d = tl.gather(d, order, 0) * scale
    tl.store(D_ptr + m * N + x, d)
    tl.store(Z_ptr + m * N * N + x[:, None] * N + x[None, :], Z)


def eigh(g: torch.Tensor, *, early_exit=None, num_warps=1):
    """Batched symmetric fp64 eigendecomposition, including the r16 24x24 Gram.

    Pad non-power-of-two matrices below their Gershgorin spectral bound, then
    discard those extra eigenpairs. All operations stay on device and capture;
    the existing r8 16x16 path executes exactly the original kernel.
    """
    n = g.shape[-1]
    if g.dtype != torch.float64 or n < 1 or g.shape[-2] != n:
        raise ValueError("k31 Jacobi eigh takes square nonempty fp64 matrices")
    if early_exit is None:
        early_exit = os.environ.get("SGLANG_GDN_K31_EIGH_EARLY_EXIT", "0") == "1"
    if num_warps not in (1, 2, 4):
        raise ValueError("Jacobi experiments support 1, 2 or 4 warps")
    if n & (n - 1):
        from sglang.srt.duet.state_factor import small_eigh
        return small_eigh(g, override="jacobi", _solver=lambda padded:
            eigh(padded, early_exit=early_exit, num_warps=num_warps))
    flat = g.reshape(-1, n, n).contiguous()
    d = torch.empty(flat.shape[:2], dtype=torch.float64, device=g.device)
    z = torch.empty_like(flat)
    if flat.shape[0]:
        if early_exit:
            _k31_eigh_kernel_early[(flat.shape[0],)](flat, d, z, N=n, SWEEPS=SWEEPS,
                                                  EARLY_EXIT=True, num_warps=num_warps)
        else:
            _k31_eigh_kernel[(flat.shape[0],)](flat, d, z, N=n, SWEEPS=SWEEPS, num_warps=num_warps)
    return d.reshape(g.shape[:-1]), z.reshape(g.shape)
