"""Fill the SWA e4m3 copy (SGLANG_OPT_DSV41_SWA_E4M3_COPY) for rows that reached the SWA cache
without the store kernels (PD transfer, host restores): every listed slot of every SWA layer in
one launch. Each row becomes RN_e4m3(payload x 2^(s - 127)) per 32-dim tile, saturating, the same
values the store kernels write."""

import torch
import triton
import triton.language as tl


@triton.jit
def _swa_e4m3_fill_kernel(
    slots,  # (n,) int32 SWA slot ids; negative: skip
    cache_ptrs,  # (layers,) int64: each layer's (pages, page_bytes) uint8 SWA cache
    copy_ptrs,  # (layers,) int64: each layer's (pages, rows_per_page, 512) uint8 copy
    page_size,
    page_bytes,
    rows_per_page,
    BYTES_PER_TOKEN: tl.constexpr,
):
    row = tl.program_id(0)
    layer = tl.program_id(1)
    slot = tl.load(slots + row).to(tl.int64)
    if slot < 0:
        return
    page = slot // page_size
    in_page = slot % page_size
    cache = tl.load(cache_ptrs + layer).to(tl.pointer_type(tl.uint8))
    copy = tl.load(copy_ptrs + layer).to(tl.pointer_type(tl.uint8))
    src = cache + page * page_bytes + in_page * BYTES_PER_TOKEN
    offs = tl.arange(0, 512)
    payload = tl.load(src + offs).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    sb = tl.load(src + 512 + offs // 32).to(tl.int32)
    # 2^(sb - 127) from its bits (sb = 0 gives +0, which rounds to the same e4m3 zero)
    scale = (sb << 23).to(tl.float32, bitcast=True)
    v = tl.minimum(tl.maximum(payload * scale, -448.0), 448.0)
    out = v.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(copy + (page * rows_per_page + in_page) * 512 + offs, out)


def swa_e4m3_fill(
    slots: torch.Tensor,
    cache_ptrs: torch.Tensor,
    copy_ptrs: torch.Tensor,
    page_size: int,
    page_bytes: int,
    bytes_per_token: int,
) -> None:
    n = slots.numel()
    if n == 0:
        return
    _swa_e4m3_fill_kernel[(n, cache_ptrs.numel())](
        slots,
        cache_ptrs,
        copy_ptrs,
        page_size,
        page_bytes,
        page_bytes // bytes_per_token,
        BYTES_PER_TOKEN=bytes_per_token,
    )


@triton.jit
def _fp4_e4m3_fill_kernel(
    slots,  # (n,) int32 slot ids of an interleaved V4.1 fp4 pool; negative: skip
    cache_ptrs,  # (layers,) int64: each layer's (pages, page_bytes) uint8 fp4 cache
    copy_ptrs,  # (layers,) int64: each layer's (pages, rows_per_page, 512) uint8 copy
    page_size,
    page_bytes,
    rows_per_page,
):
    row = tl.program_id(0)
    layer = tl.program_id(1)
    slot = tl.load(slots + row).to(tl.int64)
    if slot < 0:
        return
    page = slot // page_size
    in_page = slot % page_size
    cache = tl.load(cache_ptrs + layer).to(tl.pointer_type(tl.uint8))
    copy = tl.load(copy_ptrs + layer).to(tl.pointer_type(tl.uint8))
    src = cache + page * page_bytes + in_page * 288
    offs = tl.arange(0, 512)
    code = tl.load(src + offs // 2).to(tl.int32)
    # element 2j is the low nibble of byte j
    nib = tl.where(offs % 2 == 0, code & 15, code >> 4)
    mag = nib & 7
    e = mag >> 1
    m = (mag & 1).to(tl.float32)
    # e2m1: {0, .5, 1, 1.5, 2, 3, 4, 6}
    val = tl.where(e == 0, m * 0.5, (1.0 + m * 0.5) * tl.exp2((e - 1).to(tl.float32)))
    val = tl.where(nib >= 8, -val, val)
    scale = (
        tl.load(src + 256 + offs // 16).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    )
    # exact product (at most 2 + 4 significant bits), saturated, one RN to e4m3
    v = tl.minimum(tl.maximum(val * scale, -448.0), 448.0)
    out = v.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(copy + (page * rows_per_page + in_page) * 512 + offs, out)


def fp4_e4m3_fill(
    slots: torch.Tensor,
    cache_ptrs: torch.Tensor,
    copy_ptrs: torch.Tensor,
    page_size: int,
    page_bytes: int,
) -> None:
    """The e4m3 copy of interleaved V4.1 fp4 rows (288 B: 256 B of e2m1, 32 e4m3 scales per 16 values), every
    listed slot of every layer in one launch: RN_e4m3(e2m1 x scale), saturating, what FlashMLA's fp8 decode
    dequantizes them to (SGLANG_OPT_DSV41_C1_E4M3_COPY)."""
    n = slots.numel()
    if n == 0:
        return
    _fp4_e4m3_fill_kernel[(n, cache_ptrs.numel())](
        slots, cache_ptrs, copy_ptrs, page_size, page_bytes, page_bytes // 288
    )
