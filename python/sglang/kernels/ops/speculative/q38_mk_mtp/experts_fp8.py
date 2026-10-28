"""Isolated draft FP8-MMA alternative; no model or loader binding.

Raw E4M3 weights and FP32 block128 scales stay in their stock aligned layout.
Activations use group128 amax quantization (epsilon 1e-10, E4M3 RNE). FC1
partials are FP32; after their ordered sum, gate/up and SwiGLU materialize
BF16 before FC2 activation quantization. Those rounding boundaries require
same-input numerical and natural-acceptance gates, including comparison with
the parent's fused activation implementation. No inter-rank communication.
"""

import torch
import triton
import triton.language as tl

from .experts import _q38_mk_mtp_finalize


@triton.jit
def _q38_mk_mtp_fp8_fc1(
    X, Gate, Up, GateSF, UpSF, Ids, GatePartial, UpPartial,
    H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
    TILE_I: tl.constexpr, SPLITS: tl.constexpr,
):
    tile, work, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    expert = tl.load(Ids + work).to(tl.int64)
    row = work // TOPK
    n = tile * TILE_I + tl.arange(0, TILE_I)
    k = tl.arange(0, 128)
    m = tl.arange(0, 16)
    ga = tl.full((16, TILE_I), 0, tl.float32)
    ua = tl.full((16, TILE_I), 0, tl.float32)
    for chunk in range(H // (128 * SPLITS)):
        block = split * (H // (128 * SPLITS)) + chunk
        kk = block * 128 + k
        xv = tl.load(X + row * H + kk).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(xv), 0), 1.0e-10)
        xs = amax * (1.0 / 448.0)
        xq = tl.minimum(tl.maximum(xv * tl.div_rn(448.0, amax), -448.0), 448.0).to(tl.float8e4nv)
        a = tl.where((m == 0)[:, None], xq[None, :], 0.0).to(tl.float8e4nv)
        mask = (n < I)[:, None]
        wg = tl.load(Gate + (expert * I + n[:, None]) * H + kk[None, :], mask, 0.0)
        wu = tl.load(Up + (expert * I + n[:, None]) * H + kk[None, :], mask, 0.0)
        sf = (expert * (I // 128) + n // 128) * (H // 128) + block
        gs = tl.load(GateSF + sf, n < I, 0).to(tl.float32)
        us = tl.load(UpSF + sf, n < I, 0).to(tl.float32)
        ga = ga + tl.dot(a, tl.trans(wg)) * (xs * gs[None, :])
        ua = ua + tl.dot(a, tl.trans(wu)) * (xs * us[None, :])
    off = (work * SPLITS + split) * I + n[None, :] + tl.zeros((16, 1), tl.int32)
    valid = (m == 0)[:, None] & (n < I)[None, :]
    tl.store(GatePartial + off, ga, valid)
    tl.store(UpPartial + off, ua, valid)


@triton.jit
def _q38_mk_mtp_fp8_fc2(
    GatePartial, UpPartial, Down, DownSF, Ids, ExpertRows, Hidden,
    H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
    TILE_N: tl.constexpr, SPLITS: tl.constexpr,
):
    tile, work = tl.program_id(0), tl.program_id(1)
    expert = tl.load(Ids + work).to(tl.int64)
    n = tile * TILE_N + tl.arange(0, TILE_N)
    k = tl.arange(0, 128)
    m = tl.arange(0, 16)
    acc = tl.full((16, TILE_N), 0, tl.float32)
    for block in range(I // 128):
        kk = block * 128 + k
        g = tl.full((128,), 0, tl.float32)
        u = tl.full((128,), 0, tl.float32)
        for split in tl.static_range(SPLITS):
            off = (work * SPLITS + split) * I + kk
            g = g + tl.load(GatePartial + off)
            u = u + tl.load(UpPartial + off)
        g = g.to(tl.bfloat16).to(tl.float32)
        u = u.to(tl.bfloat16).to(tl.float32)
        hidden = (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16)
        if tile == 0:
            tl.store(Hidden + work * I + kk, hidden)
        hf = hidden.to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(hf), 0), 1.0e-10)
        hs = amax * (1.0 / 448.0)
        hq = tl.minimum(tl.maximum(hf * tl.div_rn(448.0, amax), -448.0), 448.0).to(tl.float8e4nv)
        a = tl.where((m == 0)[:, None], hq[None, :], 0.0).to(tl.float8e4nv)
        wd = tl.load(Down + (expert * H + n[:, None]) * I + kk[None, :],
                     (n < H)[:, None], 0.0)
        sf = (expert * (H // 128) + n // 128) * (I // 128) + block
        scale = tl.load(DownSF + sf, n < H, 0).to(tl.float32)
        acc = acc + tl.dot(a, tl.trans(wd)) * (hs * scale[None, :])
    out = work * H + n[None, :] + tl.zeros((16, 1), tl.int32)
    tl.store(ExpertRows + out, acc.to(tl.bfloat16),
             (m == 0)[:, None] & (n < H)[None, :])


def streaming_fp8_experts(x, gate, up, down, gate_sf, up_sf, down_sf,
                          ids, weights, *, tile_i=16, tile_n=64,
                          fc1_splits=1, return_expert_rows=False):
    """Stock block-aligned contiguous fixture only; balanced-I160 is unsupported.

    Selected weights/ids are consumed after ordinary producer completion.
    Every output has one CTA owner; no grid barriers or spin protocols.
    """
    m, h = x.shape
    i, topk = gate.shape[1], ids.shape[1]
    assert h % (128 * fc1_splits) == 0 and i % 128 == 0
    assert fc1_splits in (1, 2, 4) and gate.is_contiguous() and up.is_contiguous()
    assert down.is_contiguous() and x.dtype == torch.bfloat16
    assert gate.dtype == up.dtype == down.dtype == torch.float8_e4m3fn
    partial = torch.empty((2, m, topk, fc1_splits, i), dtype=torch.float32, device=x.device)
    hidden = torch.empty((m, topk, i), dtype=torch.bfloat16, device=x.device)
    rows = torch.empty((m, topk, h), dtype=torch.bfloat16, device=x.device)
    out = torch.empty_like(x)
    _q38_mk_mtp_fp8_fc1[(triton.cdiv(i, tile_i), m * topk, fc1_splits)](
        x, gate, up, gate_sf, up_sf, ids, partial[0], partial[1],
        H=h, I=i, TOPK=topk, TILE_I=tile_i, SPLITS=fc1_splits,
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    _q38_mk_mtp_fp8_fc2[(triton.cdiv(h, tile_n), m * topk)](
        partial[0], partial[1], down, down_sf, ids, rows, hidden,
        H=h, I=i, TOPK=topk, TILE_N=tile_n, SPLITS=fc1_splits,
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    _q38_mk_mtp_finalize[(triton.cdiv(h, 256), m)](
        rows, weights, out, H=h, TOPK=topk, BLOCK=256,
        num_warps=4, enable_fp_fusion=False)
    return (out, hidden, rows) if return_expert_rows else (out, hidden)
