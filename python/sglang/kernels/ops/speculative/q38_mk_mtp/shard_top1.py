"""Exact local/global draft top1 primitives; no model or collective hooks yet.

A transport packet is int32[rows,4]: FP32 value bits, global ID, zero, zero.
Only the communication layer may reinterpret it as FP32 for opaque16B copies.
Ordinary stream dependencies are intentional: a local PDL trigger cannot make
peer packets ready. Native head GEMM and all vocab computations stay outside.
"""
import os

import torch
import triton
import triton.language as tl

_INVALID_ID = 2**31 - 1
_POLICIES = {"draft_nan_to_minus1e30": 0, "extend_nan_first": 1}


def enabled():
    return os.environ.get("SGLANG_Q38C_MTP_SHARD_TOP1", "0") == "1"


@triton.jit
def _winner(value, global_id, valid, NAN_FIRST: tl.constexpr):
    is_nan = (value != value) & valid
    if NAN_FIRST:
        has_nan = tl.sum(is_nan.to(tl.int32), 0) != 0
        clean = tl.where(valid & ~is_nan, value, -float("inf"))
        maximum = tl.max(clean, 0)
        chosen = tl.where(has_nan, is_nan, valid & (value == maximum))
        best_value = tl.where(has_nan, float("nan"), maximum)
    else:
        # This is the native draft-topk1 policy, including its FP32 sentinel.
        clean = tl.where(is_nan, -1e30, value)
        clean = tl.where(valid, clean, -float("inf"))
        best_value = tl.max(clean, 0)
        chosen = valid & (clean == best_value)
    best_id = tl.min(tl.where(chosen, global_id, 2147483647), 0)
    return best_value, best_id


@triton.jit
def _shard_top1_partial(
    Logits, Partial, ROW_STRIDE, GLOBAL_START,
    VALID_WIDTH: tl.constexpr, SPLITS: tl.constexpr,
    NAN_FIRST: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    col = split * BLOCK + tl.arange(0, BLOCK)
    valid = col < VALID_WIDTH
    value = tl.load(Logits + row * ROW_STRIDE + col, valid, -float("inf")).to(tl.float32)
    best_value, best_id = _winner(value, GLOBAL_START + col, valid, NAN_FIRST)
    out = (row * SPLITS + split) * 2
    tl.store(Partial + out, best_value.to(tl.int32, bitcast=True))
    tl.store(Partial + out + 1, best_id)


@triton.jit
def _shard_top1_local_packet(
    Partial, Packet,
    SPLITS: tl.constexpr, NAN_FIRST: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(Partial + (row * SPLITS + col) * 2, col < SPLITS, 0).to(tl.float32, bitcast=True)
    global_id = tl.load(Partial + (row * SPLITS + col) * 2 + 1, col < SPLITS, 2147483647)
    valid = (col < SPLITS) & (global_id != 2147483647)
    best_value, best_id = _winner(value, global_id, valid, NAN_FIRST)
    tl.store(Packet + row * 4, best_value.to(tl.int32, bitcast=True))
    tl.store(Packet + row * 4 + 1, best_id)
    tl.store(Packet + row * 4 + 2, 0)
    tl.store(Packet + row * 4 + 3, 0)


@triton.jit
def _shard_top1_global_publish(
    Gathered, TopP, TopI, Positions, DraftTokens, Mrope,
    ROWS: tl.constexpr, DRAFT_STRIDE: tl.constexpr, DRAFT_COLUMN: tl.constexpr,
    MROPE_AXIS_STRIDE: tl.constexpr, MROPE_ROW_STRIDE: tl.constexpr,
    NAN_FIRST: tl.constexpr, ADVANCE: tl.constexpr,
    WRITE_DRAFT: tl.constexpr, WRITE_MROPE: tl.constexpr,
):
    row = tl.program_id(0)
    rank = tl.arange(0, 4)
    offset = (rank * ROWS + row) * 4
    value = tl.load(Gathered + offset).to(tl.float32, bitcast=True)
    global_id = tl.load(Gathered + offset + 1)
    best_value, best_id = _winner(value, global_id, global_id != 2147483647, NAN_FIRST)
    tl.store(TopI + row, best_id.to(tl.int64))
    tl.store(TopP + row, 1.0)
    if ADVANCE:
        tl.store(Positions + row, tl.load(Positions + row) + 1)
    if WRITE_DRAFT:
        tl.store(DraftTokens + row * DRAFT_STRIDE + DRAFT_COLUMN, best_id.to(tl.int64))
    if WRITE_MROPE:
        for axis in tl.static_range(3):
            ptr = Mrope + axis * MROPE_AXIS_STRIDE + row * MROPE_ROW_STRIDE
            tl.store(ptr, tl.load(ptr) + 1)


def local_top1_packet(logits, global_start, valid_width, *, policy):
    """Read the complete valid local shard and emit one16B record per row.

    `global_start` is the true original-vocab shard start, not rank×padded width.
    Unsupported added-vocab layouts must be excluded by the future service guard.
    """
    assert logits.ndim == 2 and logits.stride(1) == 1
    assert logits.dtype in (torch.bfloat16, torch.float32)
    assert 0 <= valid_width <= logits.shape[1]
    nan_first = _POLICIES[policy] == 1
    rows, width = logits.shape
    block = 8192
    splits = max(1, triton.cdiv(width, block))
    partial = torch.empty((rows, splits, 2), dtype=torch.int32, device=logits.device)
    packet = torch.empty((rows, 4), dtype=torch.int32, device=logits.device)
    _shard_top1_partial[(rows, splits)](
        logits, partial, logits.stride(0), global_start,
        VALID_WIDTH=valid_width, SPLITS=splits, NAN_FIRST=nan_first,
        BLOCK=block, num_warps=8, launch_pdl=False,
    )
    _shard_top1_local_packet[(rows,)](
        partial, packet, SPLITS=splits, NAN_FIRST=nan_first,
        BLOCK=triton.next_power_of_2(splits), num_warps=1, launch_pdl=False,
    )
    return packet


def global_top1_publish(gathered, *, policy, positions=None, draft_tokens=None,
                        draft_column=0, mrope_positions=None):
    """Consume rank-major [4,rows,4] only after the gather's real completion.

    Callers preserve graph output ownership and exclude positions/mrope aliasing
    before requesting both updates. Extend may omit positions entirely.
    """
    assert gathered.dtype == torch.int32 and gathered.is_contiguous()
    assert gathered.ndim == 3 and gathered.shape[0] == gathered.shape[2] == 4
    rows = gathered.shape[1]
    top_p = torch.empty((rows, 1), dtype=torch.float32, device=gathered.device)
    top_i = torch.empty((rows, 1), dtype=torch.int64, device=gathered.device)
    _shard_top1_global_publish[(rows,)](
        gathered, top_p, top_i, positions if positions is not None else top_i,
        draft_tokens if draft_tokens is not None else top_i,
        mrope_positions if mrope_positions is not None else top_i,
        ROWS=rows, DRAFT_STRIDE=draft_tokens.stride(0) if draft_tokens is not None else 0,
        DRAFT_COLUMN=draft_column,
        MROPE_AXIS_STRIDE=mrope_positions.stride(0) if mrope_positions is not None else 0,
        MROPE_ROW_STRIDE=mrope_positions.stride(1) if mrope_positions is not None else 0,
        NAN_FIRST=_POLICIES[policy] == 1, ADVANCE=positions is not None,
        WRITE_DRAFT=draft_tokens is not None, WRITE_MROPE=mrope_positions is not None,
        num_warps=1, launch_pdl=False,
    )
    return top_p, top_i
