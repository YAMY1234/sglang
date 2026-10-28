"""Streaming FN draft experts, keeping FP8 block128 weights in device memory.

Prototype arithmetic: block-decode weights to BF16, BF16 MMA with FP32
accumulators, BF16 SwiGLU and per-expert FC2 output. Unlike the native FP8
path, activations are not requantized to E4M3. This requires natural-AL
validation before service enablement. No global-spin or resident-grid barrier.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _q38_mk_mtp_fc1_act(
    X, Gate, Up, GateSF, UpSF, Ids, Hidden,
    H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
    TILE_I: tl.constexpr, TILE_K: tl.constexpr, I_SHIFT: tl.constexpr,
):
    tile, row, slot = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    expert = tl.load(Ids + row * TOPK + slot).to(tl.int64)
    n = tile * TILE_I + tl.arange(0, TILE_I)
    k = tl.arange(0, TILE_K)
    m = tl.arange(0, 16)
    gate_acc = tl.full((16, TILE_I), 0, tl.float32)
    up_acc = tl.full((16, TILE_I), 0, tl.float32)
    for start in range(triton.cdiv(H, TILE_K)):
        kk = start * TILE_K + k
        # Selected expert addresses are used only after the route launch has
        # completed; this initial prototype uses ordinary stream dependencies.
        sf_off = (expert * triton.cdiv(I + I_SHIFT, 128) + (n[:, None] + I_SHIFT) // 128) * triton.cdiv(H, 128) + kk[None, :] // 128
        wg = tl.load(Gate + (expert * I + n[:, None]) * H + kk[None, :],
                     (n < I)[:, None] & (kk < H)[None, :], 0.0).to(tl.float32)
        wu = tl.load(Up + (expert * I + n[:, None]) * H + kk[None, :],
                     (n < I)[:, None] & (kk < H)[None, :], 0.0).to(tl.float32)
        gs = tl.load(GateSF + sf_off, (n < I)[:, None] & (kk < H)[None, :], 0).to(tl.float32)
        us = tl.load(UpSF + sf_off, (n < I)[:, None] & (kk < H)[None, :], 0).to(tl.float32)
        wg = (wg * gs).to(tl.bfloat16)
        wu = (wu * us).to(tl.bfloat16)
        x = tl.load(X + row * H + kk[None, :] + tl.zeros((16, 1), tl.int32),
                    (m == 0)[:, None] & (kk < H)[None, :], 0)
        gate_acc = tl.dot(x, tl.trans(wg), gate_acc)
        up_acc = tl.dot(x, tl.trans(wu), up_acc)
    # Preserve BF16 FC1 materialization before the BF16 activation boundary.
    g = gate_acc.to(tl.bfloat16).to(tl.float32)
    u = up_acc.to(tl.bfloat16).to(tl.float32)
    activation = (g / (1 + tl.exp(-g)) * u).to(tl.bfloat16)
    tl.store(Hidden + (row * TOPK + slot) * I + n[None, :] + tl.zeros((16, 1), tl.int32),
             activation, (m == 0)[:, None] & (n < I)[None, :])


@triton.jit
def _q38_mk_mtp_fc2_finalize(
    Hidden, Down, DownSF, Ids, Weights, Out,
    H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
    TILE_N: tl.constexpr, TILE_K: tl.constexpr, I_SHIFT: tl.constexpr,
    SPLIT_EXPERTS: tl.constexpr,
):
    tile, row = tl.program_id(0), tl.program_id(1)
    n = tile * TILE_N + tl.arange(0, TILE_N)
    k = tl.arange(0, TILE_K)
    m = tl.arange(0, 16)
    total = tl.full((16, TILE_N), 0, tl.float32)
    first_slot = tl.program_id(2) if SPLIT_EXPERTS else 0
    slot_count: tl.constexpr = 1 if SPLIT_EXPERTS else TOPK
    for local_slot in range(slot_count):
        slot = first_slot + local_slot
        expert = tl.load(Ids + row * TOPK + slot).to(tl.int64)
        weight = tl.load(Weights + row * TOPK + slot).to(tl.float32)
        acc = tl.full((16, TILE_N), 0, tl.float32)
        for start in range(triton.cdiv(I, TILE_K)):
            kk = start * TILE_K + k
            sf_off = (expert * triton.cdiv(H, 128) + n[:, None] // 128) * triton.cdiv(I + I_SHIFT, 128) + (kk[None, :] + I_SHIFT) // 128
            w = tl.load(Down + (expert * H + n[:, None]) * I + kk[None, :],
                        (n < H)[:, None] & (kk < I)[None, :], 0.0).to(tl.float32)
            scale = tl.load(DownSF + sf_off, (n < H)[:, None] & (kk < I)[None, :], 0).to(tl.float32)
            w = (w * scale).to(tl.bfloat16)
            x = tl.load(Hidden + (row * TOPK + slot) * I + kk[None, :] + tl.zeros((16, 1), tl.int32),
                        (m == 0)[:, None] & (kk < I)[None, :], 0)
            acc = tl.dot(x, tl.trans(w), acc)
        rounded = acc.to(tl.bfloat16).to(tl.float32)
        total = rounded if SPLIT_EXPERTS else total + rounded * weight
    out_offset = (row * TOPK + first_slot) * H if SPLIT_EXPERTS else row * H
    tl.store(Out + out_offset + n[None, :] + tl.zeros((16, 1), tl.int32),
             total.to(tl.bfloat16), (m == 0)[:, None] & (n < H)[None, :])


@triton.jit
def _q38_mk_mtp_finalize(ExpertRows, Weights, Out,
                       H: tl.constexpr, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.program_id(1)
    total = tl.full((BLOCK,), 0, tl.float32)
    # Same top-10 order as the serial prototype, with the same BF16 expert rows.
    for slot in tl.static_range(TOPK):
        value = tl.load(ExpertRows + (row * TOPK + slot) * H + col,
                        col < H, 0).to(tl.float32)
        weight = tl.load(Weights + row * TOPK + slot).to(tl.float32)
        total = total + value * weight
    tl.store(Out + row * H + col, total.to(tl.bfloat16), col < H)


def streaming_experts(x, gate, up, down, gate_sf, up_sf, down_sf, ids, weights,
                      *, tile_i=32, fc1_k=256, tile_n=16, fc2_k=1024,
                      intermediate_offset=0, parallel_fc2=False, return_expert_rows=False):
    """Prototype canonical expert layout; not yet wired into the model loader.

    Wg/Wu [E,I,H], Wd [E,H,I] FP8; corresponding block128 scale matrices
    BF16. intermediate_offset preserves checkpoint scale-block boundaries when
    TP shards start at 160/320/480 (mod128 = 32/64/96). Returns rank-local
    routed BF16[M,H], BEFORE TP all-reduce, shared/HC combination.
    """
    m, h = x.shape
    inter, topk = gate.shape[1], ids.shape[1]
    hidden = torch.empty((m, topk, inter), device=x.device, dtype=torch.bfloat16)
    out = torch.empty((m, h), device=x.device, dtype=torch.bfloat16)
    expert_rows = torch.empty((m, topk, h), device=x.device, dtype=torch.bfloat16) if parallel_fc2 else out
    _q38_mk_mtp_fc1_act[(triton.cdiv(inter, tile_i), m, topk)](
        x, gate, up, gate_sf, up_sf, ids, hidden,
        H=h, I=inter, TOPK=topk, TILE_I=tile_i, TILE_K=fc1_k,
        I_SHIFT=intermediate_offset % 128, num_warps=4,
    )
    _q38_mk_mtp_fc2_finalize[(triton.cdiv(h, tile_n), m, topk if parallel_fc2 else 1)](
        hidden, down, down_sf, ids, weights, expert_rows,
        H=h, I=inter, TOPK=topk, TILE_N=tile_n, TILE_K=fc2_k,
        I_SHIFT=intermediate_offset % 128, SPLIT_EXPERTS=parallel_fc2,
        num_warps=4, enable_fp_fusion=False,
    )
    if parallel_fc2:
        _q38_mk_mtp_finalize[(triton.cdiv(h, 256), m)](
            expert_rows, weights, out, H=h, TOPK=topk, BLOCK=256,
            num_warps=4, enable_fp_fusion=False,
        )
    if return_expert_rows:
        return out, hidden, expert_rows
    return out, hidden
