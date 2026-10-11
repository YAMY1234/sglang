"""Kimi batch adapter for Flash-Next's rolled FP32 MGS helper.

Use logical width40, not the64 register tile, for the serial column loop.
Rolling that loop also avoids unrolling256 steps in the four-pass compile.
"""

import triton
import triton.language as tl

from sglang.srt.layers.attention.linear.kernels.gdn_factored import _mgs_rolled


@triton.jit
def _orth_kernel(
    Y_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    ROW_TILE: tl.constexpr,
    COL_TILE: tl.constexpr,
    PASSES: tl.constexpr,
    REL_TOL: tl.constexpr,
):
    matrix = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, ROW_TILE)
    columns = tl.arange(0, COL_TILE)
    mask = (rows[:, None] < ROWS) & (columns[None, :] < COLS)
    offsets = matrix * ROWS * COLS + rows[:, None] * COLS + columns[None, :]
    y = tl.load(Y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    q = _mgs_rolled(y, columns, COLS, PASSES, REL_TOL)
    tl.store(Y_ptr + offsets, q, mask=mask)


def orthonormalize(y, *, passes=4, rel_tol=0.0):
    rows, columns = y.shape[-2:]
    out = y.float().contiguous()
    if out.numel():
        _orth_kernel[(out.numel() // (rows * columns),)](
            out,
            ROWS=rows,
            COLS=columns,
            ROW_TILE=triton.next_power_of_2(rows),
            COL_TILE=max(2, triton.next_power_of_2(columns)),
            PASSES=passes,
            REL_TOL=rel_tol,
            num_warps=8,
        )
    return out
