"""MXFP4 DSA indexer key cache and query (SGLANG_K31_INDEXER_MXFP4).

Format (DeepSeek V4's MXFP4 indexer cache, the one DeepGEMM's ``fp8_fp4_*mqa_logits`` read): every 32 values share one
UE8M0 scale ``2^ceil(log2(max(amax, 6 * 2^-126) / 6))`` (exponent clamped to [-127, 127]); the values are E2M1, rounded to
nearest even and saturated (``cvt.rn.satfinite.e2m1x2.f32``), two per byte with the even element in the low nibble. A
64-token page is ``[64 x D/2 E2M1 | 64 x D/32 UE8M0]``: 68 B per token for D = 128 (FP8: 132). The query is
``[T, H, D/2]`` E2M1 with the four UE8M0 bytes of each head packed into one int32 (``q_scale [T, H]``, block 0 in the
lowest byte); the head weights take only the softmax and head-count scale, the MX scales stay in the logits MMA.

The arithmetic is op for op the one vLLM's Kimi K3.1 MXFP4 indexer writers use, so both engines write the same bytes for
the same bf16 key and query.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

MXFP4_BLOCK = 32
_BLOCK = tl.constexpr(MXFP4_BLOCK)


def index_k_bytes_per_token(head_dim: int) -> int:
    return head_dim // 2 + head_dim // MXFP4_BLOCK


@triton.jit
def _fp32x2_to_e2m1x2(x_lo, x_hi):
    # cvt.rn.satfinite.e2m1x2.f32 d, a, b: a goes to the high nibble, b to the low one.
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@triton.jit
def _mxfp4_quant(x, N: tl.constexpr):
    """MXFP4 of N fp32 values (N a multiple of 32; blocks never straddle). Returns (packed [N // 2] uint8, ue8m0
    [N // 32] uint8)."""
    NUM_BLOCKS: tl.constexpr = N // _BLOCK
    x_lo, x_hi = tl.split(tl.reshape(x, (NUM_BLOCKS, _BLOCK // 2, 2)))
    amax = tl.maximum(tl.max(tl.abs(x_lo), axis=1), tl.max(tl.abs(x_hi), axis=1))
    log2_scale = tl.math.ceil(tl.math.log2(tl.maximum(amax, 6.0 * (2**-126)) / 6.0))
    log2_scale = tl.minimum(tl.maximum(log2_scale, -127.0), 127.0)
    inv_scale = (1.0 / tl.math.exp2(log2_scale))[:, None]
    packed = _fp32x2_to_e2m1x2(x_lo * inv_scale, x_hi * inv_scale)
    return tl.reshape(packed, (N // 2,)), (log2_scale + 127.0).to(tl.uint8)


@triton.jit
def _store_index_k_mxfp4_kernel(
    key_ptr,
    key_stride,
    loc_ptr,
    cache_ptr,
    cache_page_stride,
    dcp_size,
    dcp_rank,
    PAGE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    slot = tl.load(loc_ptr + tok).to(tl.int64)
    # DCP: loc holds widened slots; this rank keeps the slots it owns, at slot // dcp_size (fused FP8 store's rule).
    if dcp_size > 1:
        slot = tl.where((slot >= 0) & (slot % dcp_size == dcp_rank), slot // dcp_size, -1)
    if slot >= 0:
        x = tl.load(key_ptr + tok * key_stride + tl.arange(0, HEAD_DIM)).to(tl.float32)
        packed, ue8m0 = _mxfp4_quant(x, HEAD_DIM)
        page = cache_ptr + (slot // PAGE) * cache_page_stride
        off = slot % PAGE
        tl.store(page + off * (HEAD_DIM // 2) + tl.arange(0, HEAD_DIM // 2), packed)
        scales = page + PAGE * (HEAD_DIM // 2) + off * (HEAD_DIM // _BLOCK)
        tl.store(scales + tl.arange(0, HEAD_DIM // _BLOCK), ue8m0)


def store_index_k_mxfp4(
    key: torch.Tensor,
    cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
    page_size: int = 64,
    dcp_size: int = 1,
    dcp_rank: int = 0,
) -> None:
    """Quantize the normalized bf16 index key ``[N, D]`` to MXFP4 and write it at ``out_cache_loc`` into ``cache``
    ``[pages, page_size * (D/2 + D/32)]`` uint8 (an ``IndexKeyCache`` layer buffer). Negative slots are skipped."""
    key = key.view(-1, key.shape[-1])
    head_dim = key.shape[1]
    assert key.dtype == torch.bfloat16 and key.stride(-1) == 1, (key.dtype, key.stride())
    assert head_dim % MXFP4_BLOCK == 0
    assert cache.dtype == torch.uint8 and cache.dim() == 2
    assert cache.shape[1] == page_size * index_k_bytes_per_token(head_dim), cache.shape
    assert dcp_size >= 1 and 0 <= dcp_rank < dcp_size
    if key.shape[0] == 0:
        return
    _store_index_k_mxfp4_kernel[(key.shape[0],)](
        key,
        key.stride(0),
        out_cache_loc,
        cache,
        cache.stride(0),
        dcp_size,
        dcp_rank,
        PAGE=page_size,
        HEAD_DIM=head_dim,
        num_warps=1,
    )


@triton.jit
def _q_mxfp4_kernel(
    q_ptr,
    q_stride0,
    q_stride1,
    w_ptr,
    w_stride,
    q_out_ptr,
    q_scale_ptr,
    w_out_ptr,
    weight_scale,
    num_heads,
    HEAD_DIM: tl.constexpr,
    HEADS: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    head0 = tl.program_id(1) * HEADS
    offs = tl.arange(0, HEADS * HEAD_DIM)
    head = head0 + offs // HEAD_DIM
    q = tl.load(q_ptr + tok * q_stride0 + head * q_stride1 + offs % HEAD_DIM, mask=head < num_heads, other=0.0)
    packed, ue8m0 = _mxfp4_quant(q.to(tl.float32), HEADS * HEAD_DIM)
    out = head0 * (HEAD_DIM // 2) + tl.arange(0, HEADS * HEAD_DIM // 2)
    sc = head0 * (HEAD_DIM // _BLOCK) + tl.arange(0, HEADS * (HEAD_DIM // _BLOCK))
    heads = head0 + tl.arange(0, HEADS)
    tl.store(q_out_ptr + tok * num_heads * (HEAD_DIM // 2) + out, packed, mask=out < num_heads * (HEAD_DIM // 2))
    tl.store(q_scale_ptr + tok * num_heads * (HEAD_DIM // _BLOCK) + sc, ue8m0, mask=sc < num_heads * (HEAD_DIM // _BLOCK))
    w = tl.load(w_ptr + tok * w_stride + heads, mask=heads < num_heads)
    tl.store(w_out_ptr + tok * num_heads + heads, w.to(tl.float32) * weight_scale, mask=heads < num_heads)


def q_quant_mxfp4(
    q: torch.Tensor, weights: torch.Tensor, weight_scale: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Index query ``[T, H, D]`` and head weights ``[T, H]`` to ``(q [T, H, D/2] uint8, q_scale [T, H] int32,
    weights [T, H] fp32 = weights * weight_scale)``. ``weights`` is taken in its own dtype (vLLM's merged key/weights
    GEMM writes bf16) and scaled in fp32 by one multiply."""
    T, H, D = q.shape
    assert q.stride(-1) == 1 and D % MXFP4_BLOCK == 0 and D == 4 * MXFP4_BLOCK
    assert weights.shape == (T, H)
    q_out = torch.empty((T, H, D // 2), dtype=torch.uint8, device=q.device)
    q_scale = torch.empty((T, H), dtype=torch.int32, device=q.device)
    w_out = torch.empty((T, H), dtype=torch.float32, device=q.device)
    if T == 0:
        return q_out, q_scale, w_out
    heads = min(2 if T < 128 else 8 if T < 1024 else 16, triton.next_power_of_2(H))
    _q_mxfp4_kernel[(T, triton.cdiv(H, heads))](
        q,
        q.stride(0),
        q.stride(1),
        weights,
        weights.stride(0),
        q_out,
        q_scale.view(torch.uint8),
        w_out,
        weight_scale,
        H,
        HEAD_DIM=D,
        HEADS=heads,
        num_warps=1,
    )
    return q_out, q_scale, w_out


# ------------------------------------------------------------------ torch references (tests and offline tools)

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def mxfp4_quant_ref(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact-arithmetic reference of ``_mxfp4_quant`` on fp32 rows ``[..., N]``: ``(packed [..., N/2] uint8,
    ue8m0 [..., N/32] uint8)``. The scale exponent is ceil(log2(.)) computed exactly from the float's bits."""
    x = x.float()
    shp = x.shape
    blocks = x.reshape(*shp[:-1], shp[-1] // MXFP4_BLOCK, MXFP4_BLOCK)
    y = torch.clamp(blocks.abs().amax(-1), min=6.0 * 2.0**-126) / 6.0
    mant, exp = torch.frexp(y)  # y = mant * 2^exp, mant in [0.5, 1)
    s = torch.where(mant == 0.5, exp - 1, exp).float().clamp(-127, 127)
    scaled = blocks * (1.0 / torch.exp2(s))[..., None]
    grid = torch.tensor(_E2M1, device=x.device)
    a = scaled.abs().clamp(max=6.0)
    # round to nearest, ties to the even code (even code = even index into the grid)
    d = (a[..., None] - grid).abs()
    best = d.min(-1, keepdim=True).values
    cand = (d == best)
    idx = torch.where(cand[..., ::2].any(-1),
                      (cand.float() * torch.tensor([1.0, 0, 1, 0, 1, 0, 1, 0], device=x.device)).argmax(-1),
                      cand.float().argmax(-1))
    code = (idx | ((scaled < 0) & (idx != 0)).long() << 3).to(torch.uint8)
    code = code.reshape(*shp[:-1], shp[-1])
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).to(torch.uint8)
    return packed, (s + 127).to(torch.uint8)


def mxfp4_dequant_ref(packed: torch.Tensor, ue8m0: torch.Tensor) -> torch.Tensor:
    grid = torch.tensor(_E2M1, device=packed.device)
    lo, hi = (packed & 0xF).long(), (packed >> 4).long()
    code = torch.stack([lo, hi], -1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    val = grid[code & 7] * torch.where((code & 8) != 0, -1.0, 1.0)
    scale = torch.exp2(ue8m0.float() - 127.0).repeat_interleave(MXFP4_BLOCK, dim=-1)
    return val * scale
