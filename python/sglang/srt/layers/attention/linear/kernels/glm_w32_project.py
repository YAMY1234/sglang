"""Device-predicated GLM dense W32 projection, with the service QR2 precision.

Each program owns one (layer, request, head). Stages use the same stream;
non-due requests return before matrix work. No host readback or eigensolver.
The caller owns stream forks/joins and phase publication, not these stages.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["B", "FIRST"])
def _job(
    Indices,
    Phase,
    Pending,
    Active,
    B,
    H: tl.constexpr,
    FIRST,
    COLD: tl.constexpr,
    W: tl.constexpr,
):
    pid = tl.program_id(0)
    row, head, layer = (pid // H) % B, pid % H, FIRST + pid // (B * H)
    slot = tl.load(Indices + row).to(tl.int64)
    active = tl.load(Active + slot, slot >= 0, 0)
    pending = tl.load(Pending + slot, slot >= 0, 0)
    phase = tl.load(Phase + slot, slot >= 0, 0)
    if COLD:
        due = (slot >= 0) & active & pending
    else:
        due = (slot >= 0) & active & ~pending & (phase + 1 == W)
    return pid, slot, layer, head, due


@triton.jit(do_not_specialize=["B", "FIRST", "SLOTS"])
def _prepare(
    S,
    VB,
    Warm,
    Omega,
    Extra,
    C,
    A,
    Y,
    Indices,
    Phase,
    Pending,
    Active,
    B,
    H: tl.constexpr,
    SLOTS,
    FIRST,
    COLD: tl.constexpr,
    W: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    M: tl.constexpr,
    T: tl.constexpr,
    Audit=None,
):
    pid, slot, layer, head, due = _job(
        Indices, Phase, Pending, Active, B, H, FIRST, COLD, W
    )
    if not due:
        return
    k, v, m = tl.arange(0, D), tl.arange(0, D), tl.arange(0, T)
    state_base = ((layer * SLOTS + slot) * H + head) * D * D
    state = tl.load(S + state_base + k[:, None] + v[None, :] * D)
    vb = tl.load(VB + (layer * H + head) * D + v)
    coeff = tl.sum(state * vb[None, :], 1) / tl.maximum(tl.sum(vb * vb), 1e-12)
    content = state - coeff[:, None] * vb[None, :]
    if COLD:
        probe = tl.load(
            Omega + head * D * M + v[:, None] * M + m[None, :], m[None, :] < M, 0
        )
    else:
        prev = tl.load(
            Warm
            + ((layer * SLOTS + slot) * H + head) * D * R
            + v[:, None] * R
            + m[None, :],
            m[None, :] < R,
            0,
        )
        extra = tl.load(
            Extra + head * D * (M - R) + v[:, None] * (M - R) + m[None, :] - R,
            (m[None, :] >= R) & (m[None, :] < M),
            0,
        )
        probe = prev + extra
    y = tl.dot(content, probe, input_precision="ieee")
    tl.store(C + pid * D * D + k[:, None] * D + v[None, :], content)
    tl.store(A + pid * D + k, coeff)
    tl.store(Y + pid * D * M + k[:, None] * M + m[None, :], y, m[None, :] < M)
    # Optional diagnostic: count completed matrix preparation, not launches.
    # The None specialization removes this code from production kernels.
    if Audit is not None:
        tl.atomic_add(Audit + ((layer * SLOTS + slot) * H + head) * 2 + COLD, 1)


@triton.jit(do_not_specialize=["B", "FIRST"])
def _qr2(
    Y,
    Indices,
    Phase,
    Pending,
    Active,
    B,
    H: tl.constexpr,
    FIRST,
    COLD: tl.constexpr,
    W: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    PITCH: tl.constexpr,
    T: tl.constexpr,
):
    pid, _slot, _layer, _head, due = _job(
        Indices, Phase, Pending, Active, B, H, FIRST, COLD, W
    )
    if not due:
        return
    x, j = tl.arange(0, D), tl.arange(0, T)
    yp = Y + pid * D * PITCH + x[:, None] * PITCH + j[None, :]
    yd = tl.load(yp, j[None, :] < M, 0).to(tl.float64)
    for _ in range(2):
        gram = tl.full((T, T), 0, tl.float64)
        for column in range(M):
            yc = tl.sum(tl.where(j[None, :] == column, yd, 0.0), 1)
            dots = tl.sum(yd * yc[:, None], 0)
            gram = tl.where(j[:, None] == column, dots[None, :], gram)
        diagonal = tl.sum(tl.where(j[:, None] == j[None, :], gram, 0.0), 1)
        jitter = 1e-7 * tl.sum(diagonal) / M + 1e-30
        gram += tl.where((j[:, None] == j[None, :]) & (j[:, None] < M), jitter, 0.0)
        lower = tl.full((T, T), 0, tl.float64)
        for column in range(M):
            gcol = tl.sum(tl.where(j[None, :] == column, gram, 0.0), 1)
            pivot = tl.sqrt(tl.sum(tl.where(j == column, gcol, 0.0)))
            col = tl.where((j >= column) & (j < M), gcol / pivot, 0.0)
            lower = tl.where(j[None, :] == column, col[:, None], lower)
            gram -= tl.where(
                (j[:, None] > column) & (j[None, :] > column),
                col[:, None] * col[None, :],
                0.0,
            )
        for column in range(M):
            lr = tl.sum(tl.where(j[:, None] == column, lower, 0.0), 0)
            pivot = tl.sum(tl.where(j == column, lr, 0.0))
            value = tl.sum(tl.where(j[None, :] == column, yd, 0.0), 1)
            previous = tl.sum(yd * tl.where(j < column, lr, 0.0)[None, :], 1)
            solved = (value - previous) / pivot
            yd = tl.where(j[None, :] == column, solved[:, None], yd)
    tl.store(yp, yd.to(tl.float32), j[None, :] < M)


@triton.jit(do_not_specialize=["B", "FIRST"])
def _range_product(
    C,
    X,
    Y,
    Indices,
    Phase,
    Pending,
    Active,
    B,
    H: tl.constexpr,
    FIRST,
    COLD: tl.constexpr,
    W: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    T: tl.constexpr,
    TRANSPOSE: tl.constexpr,
):
    pid, _slot, _layer, _head, due = _job(
        Indices, Phase, Pending, Active, B, H, FIRST, COLD, W
    )
    if not due:
        return
    x, y, m = tl.arange(0, D), tl.arange(0, D), tl.arange(0, T)
    if TRANSPOSE:
        cp = C + pid * D * D + y[None, :] * D + x[:, None]
    else:
        cp = C + pid * D * D + x[:, None] * D + y[None, :]
    c = tl.load(cp)
    q = tl.load(X + pid * D * M + y[:, None] * M + m[None, :], m[None, :] < M, 0)
    product = tl.dot(c, q, input_precision="ieee")
    tl.store(Y + pid * D * M + x[:, None] * M + m[None, :], product, m[None, :] < M)


@triton.jit(do_not_specialize=["B", "FIRST"])
def _gram(
    C,
    Q,
    G,
    Indices,
    Phase,
    Pending,
    Active,
    B,
    H: tl.constexpr,
    FIRST,
    COLD: tl.constexpr,
    W: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    T: tl.constexpr,
):
    pid, _slot, _layer, _head, due = _job(
        Indices, Phase, Pending, Active, B, H, FIRST, COLD, W
    )
    if not due:
        return
    x, y, m = tl.arange(0, D), tl.arange(0, D), tl.arange(0, T)
    q = tl.load(Q + pid * D * M + x[:, None] * M + m[None, :], m[None, :] < M, 0)
    c = tl.load(C + pid * D * D + x[:, None] * D + y[None, :])
    reduced = tl.dot(tl.trans(q), c, input_precision="ieee")
    gram = tl.dot(reduced, tl.trans(reduced), input_precision="ieee").to(tl.float64)
    diagonal = tl.sum(tl.where(m[:, None] == m[None, :], gram, 0.0), 1)
    jitter = 1e-7 * tl.sum(diagonal) / M + 1e-30
    gram += tl.where(m[:, None] == m[None, :], jitter, 0.0)
    tl.store(
        G + pid * M * M + m[:, None] * M + m[None, :],
        gram,
        (m[:, None] < M) & (m[None, :] < M),
    )


@triton.jit(do_not_specialize=["B", "FIRST", "SLOTS"])
def _reconstruct(
    S,
    VB,
    C,
    A,
    Q,
    Z,
    Temp,
    Indices,
    Phase,
    Pending,
    Active,
    B,
    H: tl.constexpr,
    SLOTS,
    FIRST,
    COLD: tl.constexpr,
    W: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    M: tl.constexpr,
    T: tl.constexpr,
):
    pid, slot, layer, head, due = _job(
        Indices, Phase, Pending, Active, B, H, FIRST, COLD, W
    )
    if not due:
        return
    x, y, m, r = (
        tl.arange(0, D),
        tl.arange(0, D),
        tl.arange(0, T),
        tl.arange(0, triton.next_power_of_2(R)),
    )
    c = tl.load(C + pid * D * D + x[:, None] * D + y[None, :])
    q = tl.load(Q + pid * D * M + x[:, None] * M + m[None, :], m[None, :] < M, 0)
    rotation = tl.load(
        Z + pid * M * M + m[:, None] * M + (M - R) + r[None, :],
        (m[:, None] < M) & (r[None, :] < R),
        0,
    ).to(tl.float32)
    u = tl.dot(q, rotation, input_precision="ieee")
    uts = tl.dot(tl.trans(u), c, input_precision="ieee")
    projected = tl.dot(u, uts, input_precision="ieee")
    a = tl.load(A + pid * D + x)
    vb = tl.load(VB + (layer * H + head) * D + y)
    projected += a[:, None] * vb[None, :]
    tl.store(
        S + ((layer * SLOTS + slot) * H + head) * D * D + x[:, None] + y[None, :] * D,
        projected,
    )
    tl.store(
        Temp + pid * D * M + y[:, None] * M + r[None, :], tl.trans(uts), r[None, :] < R
    )


@triton.jit(do_not_specialize=["B", "FIRST", "SLOTS"])
def _publish_warm(
    Temp,
    Warm,
    Indices,
    Phase,
    Pending,
    Active,
    B,
    H: tl.constexpr,
    SLOTS,
    FIRST,
    COLD: tl.constexpr,
    W: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    M: tl.constexpr,
):
    pid, slot, layer, head, due = _job(
        Indices, Phase, Pending, Active, B, H, FIRST, COLD, W
    )
    if not due:
        return
    x, r = tl.arange(0, D), tl.arange(0, triton.next_power_of_2(R))
    q = tl.load(Temp + pid * D * M + x[:, None] * M + r[None, :], r[None, :] < R, 0)
    tl.store(
        Warm
        + ((layer * SLOTS + slot) * H + head) * D * R
        + x[:, None] * R
        + r[None, :],
        q,
        r[None, :] < R,
    )


@triton.jit(do_not_specialize=["B"])
def _phase(
    Indices,
    Phase,
    Pending,
    Active,
    B,
    W: tl.constexpr,
    CLEAR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(Indices + row, row < B, -1).to(tl.int64)
    valid = (row < B) & (slot >= 0)
    active = tl.load(Active + slot, valid, 0)
    if CLEAR:
        tl.store(Pending + slot, False, valid & active)
    else:
        count = tl.load(Phase + slot, valid, 0)
        tl.store(Phase + slot, (count + 1) % W, valid & active)


def phase_update(indices, phase, pending, active, window, *, clear=False):
    _phase[(triton.cdiv(indices.numel(), 128),)](
        indices, phase, pending, active, indices.numel(), window, clear, 128
    )


class ProjectionWorkspace:
    def __init__(self, batch, heads, layers, device, rank=32, dimension=128):
        self.batch, self.layers = batch, layers
        self.rank, self.dimension = rank, dimension
        self.width = rank + 8
        jobs = batch * heads * layers
        self.content = torch.empty(
            jobs, dimension, dimension, device=device, dtype=torch.float32
        )
        self.coeff = torch.empty(jobs, dimension, device=device, dtype=torch.float32)
        self.q = torch.empty(
            jobs, dimension, self.width, device=device, dtype=torch.float32
        )
        self.temp = torch.empty_like(self.q)
        self.gram = torch.empty(
            jobs, self.width, self.width, device=device, dtype=torch.float64
        )
        self.vectors = torch.empty_like(self.gram)
        self.eigenvalues = torch.empty(
            jobs, self.width, device=device, dtype=torch.float64
        )

    def bytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in vars(self).values()
            if isinstance(t, torch.Tensor)
        )


def project(
    states,
    vbar,
    warm,
    omega,
    extra,
    indices,
    phase,
    pending,
    active,
    workspace,
    *,
    first=0,
    layers=None,
    cold=False,
    window=32,
    audit=None,
):
    layers = states.shape[0] - first if layers is None else layers
    batch, heads = indices.numel(), states.shape[2]
    assert batch <= workspace.batch and layers <= workspace.layers
    assert indices.is_contiguous() and states.is_contiguous()
    d, r, m = workspace.dimension, workspace.rank, workspace.width
    assert states.shape[-2:] == (d, d) and r == 32
    grid = (layers * batch * heads,)
    args = (indices, phase, pending, active)
    kw = {"B": batch, "H": heads, "FIRST": first, "COLD": cold, "W": window}
    c, a, q, t, g, z = (
        workspace.content,
        workspace.coeff,
        workspace.q,
        workspace.temp,
        workspace.gram,
        workspace.vectors,
    )
    tile = triton.next_power_of_2(m)
    _prepare[grid](
        states,
        vbar,
        warm,
        omega,
        extra,
        c,
        a,
        q,
        *args,
        **kw,
        SLOTS=states.shape[1],
        D=d,
        R=r,
        M=m,
        T=tile,
        Audit=audit,
        num_warps=8,
        enable_fp_fusion=False,
    )
    _qr2[grid](
        q, *args, **kw, D=d, M=m, PITCH=m, T=tile, num_warps=16, enable_fp_fusion=False
    )
    _range_product[grid](
        c, q, t, *args, **kw, D=d, M=m, T=tile, TRANSPOSE=True, num_warps=8
    )
    _qr2[grid](
        t, *args, **kw, D=d, M=m, PITCH=m, T=tile, num_warps=16, enable_fp_fusion=False
    )
    _range_product[grid](
        c, t, q, *args, **kw, D=d, M=m, T=tile, TRANSPOSE=False, num_warps=8
    )
    _qr2[grid](
        q, *args, **kw, D=d, M=m, PITCH=m, T=tile, num_warps=16, enable_fp_fusion=False
    )
    _gram[grid](
        c, q, g, *args, **kw, D=d, M=m, T=tile, num_warps=8, enable_fp_fusion=False
    )
    _eigh_kernel[grid](
        g,
        workspace.eigenvalues,
        z,
        *args,
        **kw,
        N=m,
        TILE=tile,
        MAX_SWEEPS=12,
        REL_TOL=1e-14,
        num_warps=8,
        enable_fp_fusion=False,
    )
    _reconstruct[grid](
        states,
        vbar,
        c,
        a,
        q,
        z,
        t,
        *args,
        **kw,
        SLOTS=states.shape[1],
        D=d,
        R=r,
        M=m,
        T=tile,
        num_warps=8,
        enable_fp_fusion=False,
    )
    _qr2[grid](
        t,
        *args,
        **kw,
        D=d,
        M=r,
        PITCH=m,
        T=triton.next_power_of_2(r),
        num_warps=16,
        enable_fp_fusion=False,
    )
    _publish_warm[grid](
        t, warm, *args, **kw, SLOTS=states.shape[1], D=d, R=r, M=m, num_warps=4
    )


# Predicated specialization of Kimi small-Gram FP64 Jacobi.
@triton.jit(do_not_specialize=["B", "FIRST"])
def _eigh_kernel(
    G_ptr,
    D_ptr,
    Z_ptr,
    Indices,
    Phase,
    Pending,
    Active,
    B,
    H: tl.constexpr,
    FIRST,
    COLD: tl.constexpr,
    W: tl.constexpr,
    N: tl.constexpr,
    TILE: tl.constexpr,
    MAX_SWEEPS: tl.constexpr,
    REL_TOL: tl.constexpr,
):
    matrix, _slot, _layer, _head, due = _job(
        Indices, Phase, Pending, Active, B, H, FIRST, COLD, W
    )
    if not due:
        return
    matrix = matrix.to(tl.int64)
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
