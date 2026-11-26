"""FP8 KV rows <-> NVFP4 rows for the NVFP4 HiCache host tier.

A row of ``dim`` FP8 E4M3 values becomes ``dim / 2`` bytes of E2M1 values (two
per byte, low nibble first) followed by ``dim / 16`` E4M3 block scales: the
linear-scale NVFP4 KV layout of FlashInfer's ``nvfp4_kv_quantize``. Decoding
rounds each dequantized value to bf16 and then to E4M3. Both roundings run in
integer arithmetic, so the bytes do not depend on how the backend lowers float
casts, subnormals included. CUDA; ``dim`` a multiple of 16.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.numerics import round_bf16_to_fp32

# Flattened 16-value blocks per program. Arbitrary; not tuned.
_BLOCKS_PER_PROGRAM = 128


@triton.jit
def _to_e4m3_byte(v):
    """fp32 -> E4M3FN byte, nearest with ties to even, subnormals included; |v| <= 448."""
    sign = (v.to(tl.uint32, bitcast=True) >> 31).to(tl.int32)
    a = tl.minimum(tl.abs(v), 448.0)
    exp = ((a.to(tl.uint32, bitcast=True) >> 23) & 0xFF).to(tl.int32) - 127
    e = tl.maximum(exp, -6)  # below 2**-6 the grid keeps the 2**-9 step
    q = a * ((127 + 3 - e) << 23).to(tl.float32, bitcast=True)  # a / 2**(e - 3), exact
    f = tl.floor(q)
    r = q - f
    m = f.to(tl.int32)
    m = m + ((r > 0.5) | ((r == 0.5) & ((m & 1) == 1))).to(tl.int32)
    # m == 16 carries into the next exponent, as the encoding is contiguous.
    return ((e + 6) * 8 + m + (sign << 7)).to(tl.uint8)


@triton.jit
def _encode_fp8_rows_to_nvfp4_kernel(
    src_ptr,
    idx_ptr,
    out_ptr,
    gs_ptr,
    n_blocks,
    src_row_stride,
    out_page_stride,
    out_row_stride,
    PAGE_SIZE: tl.constexpr,
    BLOCKS_PER_ROW: tl.constexpr,
    DATA_BYTES: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    b = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    valid = b < n_blocks
    row = (b // BLOCKS_PER_ROW).to(tl.int64)
    blk = (b % BLOCKS_PER_ROW).to(tl.int64)
    src_row = tl.load(idx_ptr + row, mask=valid, other=0)
    j = tl.arange(0, 16)
    x8 = tl.load(
        src_ptr + src_row[:, None] * src_row_stride + blk[:, None] * 16 + j[None, :],
        mask=valid[:, None],
        other=0,
    )
    x = x8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    gs = tl.load(gs_ptr)
    amax = tl.max(tl.abs(x), axis=1)
    sf8 = _to_e4m3_byte(tl.fdiv(amax, 6.0 * gs, ieee_rounding=True))
    sf = sf8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    one = tl.full(sf.shape, 1.0, tl.float32)
    inv = tl.where(amax > 0, tl.fdiv(one, sf * gs, ieee_rounding=True), 0.0)
    y = x * inv[:, None]
    mag = tl.minimum(tl.abs(y), 6.0)
    # Nearest E2M1 magnitude; a tie goes to the even code (up only at 0.75, 1.75, 3.5).
    code = (
        (mag > 0.25).to(tl.int32)
        + (mag >= 0.75).to(tl.int32)
        + (mag > 1.25).to(tl.int32)
        + (mag >= 1.75).to(tl.int32)
        + (mag > 2.5).to(tl.int32)
        + (mag >= 3.5).to(tl.int32)
        + (mag > 5.0).to(tl.int32)
    )
    code = code | ((y.to(tl.int32, bitcast=True) < 0).to(tl.int32) << 3)
    lo, hi = tl.split(tl.reshape(code, (BLOCK_B, 8, 2)))
    packed = (lo | (hi << 4)).to(tl.uint8)
    out_row = (row // PAGE_SIZE) * out_page_stride + (row % PAGE_SIZE) * out_row_stride
    k = tl.arange(0, 8)
    tl.store(
        out_ptr + out_row[:, None] + blk[:, None] * 8 + k[None, :],
        packed,
        mask=valid[:, None],
    )
    tl.store(out_ptr + out_row + DATA_BYTES + blk, sf8, mask=valid)


@triton.jit
def _decode_nvfp4_rows_to_fp8_kernel(
    src_ptr,
    idx_ptr,
    dst_ptr,
    gs_ptr,
    n_blocks,
    src_row_stride,
    dst_row_stride,
    BLOCKS_PER_ROW: tl.constexpr,
    DATA_BYTES: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    b = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    valid = b < n_blocks
    row = (b // BLOCKS_PER_ROW).to(tl.int64)
    blk = (b % BLOCKS_PER_ROW).to(tl.int64)
    k = tl.arange(0, 8)
    packed = tl.load(
        src_ptr + row[:, None] * src_row_stride + blk[:, None] * 8 + k[None, :],
        mask=valid[:, None],
        other=0,
    ).to(tl.int32)
    code = tl.reshape(tl.join(packed & 0xF, packed >> 4), (BLOCK_B, 16))
    sf = (
        tl.load(src_ptr + row * src_row_stride + DATA_BYTES + blk, mask=valid, other=0)
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.float32)
    )
    m = code & 7
    e = m >> 1
    half = (m & 1).to(tl.float32) * 0.5
    # E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6; code 8 is -0.
    mag = tl.where(
        e == 0,
        half,
        (1.0 + half) * tl.where(e == 1, 1.0, tl.where(e == 2, 2.0, 4.0)),
    )
    val = tl.where(code >= 8, -mag, mag)
    gs = tl.load(gs_ptr)
    out = _to_e4m3_byte(round_bf16_to_fp32((val * sf[:, None]) * gs))
    dst_row = tl.load(idx_ptr + row, mask=valid, other=0)
    j = tl.arange(0, 16)
    tl.store(
        dst_ptr + dst_row[:, None] * dst_row_stride + blk[:, None] * 16 + j[None, :],
        out,
        mask=valid[:, None],
    )


def _grid(n_blocks: int):
    return (triton.cdiv(n_blocks, _BLOCKS_PER_PROGRAM),)


def encode_fp8_rows_to_nvfp4(
    src: torch.Tensor,
    rows: torch.Tensor,
    out: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    dim: int,
    page_size: int,
) -> None:
    """NVFP4-encode src[rows] (FP8 bytes, [N, ..., dim]) into out[page, unit, ..., row_bytes],
    unit = position in rows % page_size; out may be strided between pages and units."""
    n_blocks = rows.numel() * (dim // 16)
    if n_blocks == 0:
        return
    _encode_fp8_rows_to_nvfp4_kernel[_grid(n_blocks)](
        src,
        rows,
        out,
        global_scale,
        n_blocks,
        src.stride(0),
        out.stride(0),
        out.stride(1),
        PAGE_SIZE=page_size,
        BLOCKS_PER_ROW=dim // 16,
        DATA_BYTES=dim // 2,
        BLOCK_B=_BLOCKS_PER_PROGRAM,
    )


def decode_nvfp4_rows_to_fp8(
    src: torch.Tensor,
    rows: torch.Tensor,
    dst: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    dim: int,
) -> None:
    """Decode NVFP4 rows src[i] ([n, row_bytes]) into dst[rows[i]] (FP8 bytes, [N, ..., dim])."""
    n_blocks = rows.numel() * (dim // 16)
    if n_blocks == 0:
        return
    _decode_nvfp4_rows_to_fp8_kernel[_grid(n_blocks)](
        src,
        rows,
        dst,
        global_scale,
        n_blocks,
        src.stride(0),
        dst.stride(0),
        BLOCKS_PER_ROW=dim // 16,
        DATA_BYTES=dim // 2,
        BLOCK_B=_BLOCKS_PER_PROGRAM,
    )
