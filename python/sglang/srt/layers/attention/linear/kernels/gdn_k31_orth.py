"""Candidate fused fp64 CholeskyQR2; explicit opt-in, original jitter/algebra.

One CTA owns each 128 x 16 matrix. This eliminates the small strided DGEMM,
potrf and trsm launch chain without changing the two QR rounds or their jitter.
It remains separate from the production reference until the numerical gate.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _orth_cholqr2_kernel(Y_ptr, Q_ptr, ROWS: tl.constexpr, COLS: tl.constexpr):
    matrix = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    eye = cols[:, None] == cols[None, :]
    offsets = matrix * ROWS * COLS + rows[:, None] * COLS + cols[None, :]
    Y = tl.load(Y_ptr + offsets).to(tl.float64)
    for repeat in range(2):
        G = tl.zeros((COLS, COLS), tl.float64)
        for j in tl.static_range(COLS):
            column = tl.sum(tl.where(cols[None, :] == j, Y, 0.), axis=1)
            dot = tl.sum(Y * column[:, None], axis=0)
            G = tl.where(cols[None, :] == j, dot[:, None], G)
        trace = tl.sum(tl.sum(tl.where(eye, G, 0.), axis=1), axis=0)
        G = G + tl.where(eye, 1.e-7 * trace / COLS + 1.e-30, 0.)
        L = tl.zeros((COLS, COLS), tl.float64)
        for j in tl.static_range(COLS):
            previous = tl.sum(tl.where(cols[:, None] == j, L, 0.), axis=0)
            column = tl.sum(tl.where(cols[None, :] == j, G, 0.), axis=1)
            value = column - tl.sum(L * previous[None, :], axis=1)
            diagonal = tl.sqrt(tl.sum(tl.where(cols == j, value, 0.), axis=0))
            solved = tl.where(cols >= j, value / diagonal, 0.)
            L = tl.where(cols[None, :] == j, solved[:, None], L)
        Q = tl.zeros((ROWS, COLS), tl.float64)
        for j in tl.static_range(COLS):
            coefficients = tl.sum(tl.where(cols[:, None] == j, L, 0.), axis=0)
            diagonal = tl.sum(tl.where(cols == j, coefficients, 0.), axis=0)
            column = tl.sum(tl.where(cols[None, :] == j, Y, 0.), axis=1)
            value = (column - tl.sum(Q * coefficients[None, :], axis=1)) / diagonal
            Q = tl.where(cols[None, :] == j, value[:, None], Q)
        Y = Q
    tl.store(Q_ptr + offsets, Y.to(Q_ptr.dtype.element_ty))


def orth_cholqr2(y):
    if y.shape[-2:] != (128, 16) or y.dtype != torch.float32:
        raise ValueError('fused k31 orth requires float32 (..., 128, 16)')
    source = y.contiguous()
    output = torch.empty_like(source)
    _orth_cholqr2_kernel[(source.numel() // (128 * 16),)](
        source, output, ROWS=128, COLS=16, num_warps=4, enable_fp_fusion=False)
    return output
