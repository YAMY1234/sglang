# SPDX-License-Identifier: Apache-2.0
"""Default-off B1 chain acceptance plus integer commit outputs.

The old eagle kernels remain the oracle. Only final FINISH calls can consume a
fresh context supplied by the post-verify graph slot; no sampler state lives here.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.eagle import (
    _LINEAR_ACCEPT,
    _chain_is_canonical,
    _chain_linear,
    _chain_walk,
)


@triton.jit
def _chain_accept_commit_kernel(
    predict,
    accept_index,
    num_correct_drafts,
    accept_lens,
    sim_accept_index,
    candidates,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    target_predict,
    bonus_target,
    simulated_acc_len,
    seq_lens,
    new_seq_lens,
    commit_bonus,
    last_steps,
    track_steps,
    correct,
    select,
    ids,
    prefix,
    extend,
    post_lens,
    positions,
    start_locs,
    mrope,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    VERIFY: tl.constexpr,
    SIMULATE: tl.constexpr,
    REAL_DRAFT_TOKENS: tl.constexpr,
    TRACK_INTERVAL: tl.constexpr,
    EXTEND_LAYOUT: tl.constexpr,
    EXTEND_MROPE: tl.constexpr,
    LINEAR: tl.constexpr = False,
):
    FINISH: tl.constexpr = True
    # One program per request; its WIDTH nodes are rows [b * WIDTH, (b + 1) * WIDTH) of
    # predict / candidates / retrieve_*, and accept_index is [bs, WIDTH].
    b = tl.program_id(0).to(tl.int64)
    base = b * WIDTH
    cols = tl.arange(0, BLOCK)
    in_row = cols < WIDTH
    if VERIFY:
        if LINEAR:
            if _chain_is_canonical(
                retrieve_index,
                retrieve_next_token,
                retrieve_next_sibling,
                base,
                cols,
                WIDTH,
            ):
                first, acc, pred, num = _chain_linear(
                    candidates, target_predict, base, cols, WIDTH
                )
            else:
                first, acc, pred, num = _chain_walk(
                    retrieve_index,
                    retrieve_next_token,
                    retrieve_next_sibling,
                    candidates,
                    target_predict,
                    base,
                    cols,
                    WIDTH,
                    BLOCK,
                )
        else:
            first, acc, pred, num = _chain_walk(
                retrieve_index,
                retrieve_next_token,
                retrieve_next_sibling,
                candidates,
                target_predict,
                base,
                cols,
                WIDTH,
                BLOCK,
            )
        tl.store(accept_index + base + cols, acc.to(tl.int32), mask=in_row)
        if not (FINISH and SIMULATE):
            tl.store(predict + base + cols, pred.to(tl.int32), mask=in_row)
            tl.store(num_correct_drafts + b, num.to(tl.int32))
    else:
        pred = tl.load(predict + base + cols, mask=in_row, other=0).to(tl.int64)
        first = tl.load(accept_index + base).to(tl.int64)
        num = tl.load(num_correct_drafts + b).to(tl.int64)
    if FINISH:
        if SIMULATE:
            # spec_utils.generate_simulated_accept_index_device for a length on the device
            length = tl.load(simulated_acc_len).to(tl.int64)
            sim = tl.where(cols < length, first + cols, -1)
            tl.store(sim_accept_index + base + cols, sim.to(tl.int32), mask=in_row)
            num = length - 1
            if REAL_DRAFT_TOKENS:
                nxt = tl.minimum(cols + 1, WIDTH - 1)
                drafted = tl.load(candidates + base + nxt, mask=in_row, other=0)
                bonus = tl.load(bonus_target + base + cols, mask=in_row, other=0)
                pred = tl.where(
                    cols < length - 1,
                    drafted.to(tl.int32).to(tl.int64),
                    tl.where(cols == length - 1, bonus.to(tl.int32).to(tl.int64), pred),
                )
            else:
                pred = tl.full([BLOCK], 100, dtype=tl.int64)
            tl.store(predict + base + cols, pred.to(tl.int32), mask=in_row)
            tl.store(num_correct_drafts + b, num.to(tl.int32))
        tl.store(accept_lens + b, (num + 1).to(tl.int32))

    # Explicitly reproduce the old int32 stores followed by commit's signed reloads.
    # VERIFY=false never rewrites original accept_index; simulation owns another array.
    if SIMULATE:
        final_acc = sim.to(tl.int32).to(tl.int64)
    else:
        final_acc = acc.to(tl.int32).to(tl.int64)
    final_pred = pred.to(tl.int32).to(tl.int64)
    accepted = (num + 1).to(tl.int32).to(tl.int64)
    length = tl.load(seq_lens).to(tl.int64)
    # One live lane per defined B1 index. No global store/reload or inter-CTA barrier.
    idx = tl.sum(tl.where(in_row & (cols == accepted - 1), final_acc, 0), axis=0)
    tl.store(last_steps, idx)
    tl.store(correct, accepted - 1)
    tl.store(select, accepted - 1)
    tl.store(ids + cols, final_pred, mask=in_row)
    if TRACK_INTERVAL > 0:
        post = length + accepted
        crossing = length // TRACK_INTERVAL != post // TRACK_INTERVAL
        tracking_position = tl.maximum(
            post // TRACK_INTERVAL * TRACK_INTERVAL - length - 1, 0
        )
        track = tl.full((), -1, tl.int64)
        if crossing:
            track = tl.sum(
                tl.where(in_row & (cols == tracking_position), final_acc, 0), axis=0
            )
        tl.store(track_steps, track)
    if EXTEND_LAYOUT:
        tl.store(prefix, tl.maximum(length, 0))
        tl.store(extend, WIDTH)
        tl.store(post_lens, length + WIDTH)
        tl.store(start_locs, 0)
        position = tl.maximum(length, 0).to(tl.int32) + cols.to(tl.int32)
        tl.store(positions + cols, position.to(tl.int64), mask=in_row)
        if EXTEND_MROPE:
            for axis in tl.static_range(3):
                tl.store(
                    mrope + axis * WIDTH + cols, position.to(tl.int64), mask=in_row
                )
    # Only bonus lookup normalizes a negative index, once. last/track keep it.
    idx = tl.where(idx < 0, idx + WIDTH, idx)
    token = tl.sum(tl.where(in_row & (cols == idx), final_pred, 0), axis=0)
    tl.store(new_seq_lens, length + accepted)
    tl.store(commit_bonus, token.to(commit_bonus.dtype.element_ty))


def chain_commit_inputs_supported(seq_lens, candidates, *retrieve):
    """Metadata-only guard shared by key selection and the final dispatch."""
    integer = (torch.int32, torch.int64)
    tensors = (seq_lens, candidates, *retrieve)
    return (
        seq_lens.shape == (1,)
        and seq_lens.dtype in integer
        and seq_lens.stride(0) > 0
        and len(retrieve) == 3
        and all(t.is_cuda and t.device == seq_lens.device for t in tensors)
        and all(
            t.dtype in integer and t.numel() == 7 and t.is_contiguous()
            for t in (candidates, *retrieve)
        )
    )


class ChainCommitContext:
    """One _body invocation; outputs is set only by the selected final entry."""

    def __init__(
        self, seq_lens, track_interval, extend_layout, extend_mrope, bonus_dtype
    ):
        self.seq_lens = seq_lens
        self.track_interval = track_interval
        self.extend_layout = extend_layout
        self.extend_mrope = extend_mrope
        self.bonus_dtype = bonus_dtype
        self.outputs = None


def try_chain_accept_commit(
    context,
    *,
    predict,
    accept_index,
    num_correct_drafts,
    candidates,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    target_predict,
    bonus_target,
    simulated_acc_len,
    real_draft_tokens,
):
    """Return the original (lens, index) tuple, or None for the old entry."""
    if context.outputs is not None:
        raise RuntimeError("chain commit context already consumed")
    verify = target_predict is not None
    simulate = simulated_acc_len is not None
    tensors = (predict, accept_index, num_correct_drafts)
    if (
        not chain_commit_inputs_supported(
            context.seq_lens,
            candidates,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
        )
        or predict.shape != (7,)
        or accept_index.shape != (1, 7)
        or num_correct_drafts.shape != (1,)
        or any(
            t.dtype != torch.int32
            or not t.is_contiguous()
            or t.device != predict.device
            for t in tensors
        )
        or predict.device != context.seq_lens.device
        or (not verify and not simulate)
        or context.track_interval < 0
        or context.bonus_dtype not in (torch.int32, torch.int64)
    ):
        return None
    for tensor in (
        target_predict,
        bonus_target if simulate and real_draft_tokens else None,
    ):
        if tensor is not None and (
            tensor.numel() != 7
            or not tensor.is_contiguous()
            or tensor.dtype not in (torch.int32, torch.int64)
            or tensor.device != predict.device
        ):
            return None
    if simulate and (
        simulated_acc_len.shape != (1,)
        or simulated_acc_len.dtype != torch.int64
        or not simulated_acc_len.is_contiguous()
        or simulated_acc_len.device != predict.device
        or (real_draft_tokens and bonus_target is None)
    ):
        return None
    device = predict.device
    accept_lens = torch.empty((1,), device=device, dtype=torch.int32)
    sim_accept_index = torch.empty_like(accept_index) if simulate else None
    new_seq_lens = torch.empty(
        (1,),
        device=device,
        dtype=torch.promote_types(context.seq_lens.dtype, torch.int32),
    )
    bonus = torch.empty((1,), device=device, dtype=context.bonus_dtype)
    last_steps = torch.empty((1,), device=device, dtype=torch.int64)
    track_steps = torch.empty_like(last_steps) if context.track_interval else None
    correct = torch.empty_like(accept_lens, memory_format=torch.contiguous_format)
    select = torch.empty_like(last_steps)
    ids = torch.empty((7,), device=device, dtype=torch.int64)
    layout = None
    if context.extend_layout:
        layout = (
            torch.empty((1,), device=device, dtype=torch.int32),
            torch.empty((1,), device=device, dtype=torch.int32),
            torch.empty((1,), device=device, dtype=context.seq_lens.dtype),
            (
                torch.empty((7,), device=device, dtype=torch.int64),
                torch.empty((1,), device=device, dtype=torch.int32),
                torch.empty((3, 7), device=device, dtype=torch.int64)
                if context.extend_mrope
                else None,
            ),
        )
    _chain_accept_commit_kernel[(1,)](
        predict,
        accept_index,
        num_correct_drafts,
        accept_lens,
        sim_accept_index,
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        target_predict,
        bonus_target,
        simulated_acc_len,
        context.seq_lens,
        new_seq_lens,
        bonus,
        last_steps,
        track_steps,
        correct,
        select,
        ids,
        *(layout[:3] + layout[3] if layout is not None else (None,) * 6),
        WIDTH=7,
        BLOCK=8,
        VERIFY=verify,
        SIMULATE=simulate,
        REAL_DRAFT_TOKENS=real_draft_tokens,
        TRACK_INTERVAL=context.track_interval,
        EXTEND_LAYOUT=layout is not None,
        EXTEND_MROPE=context.extend_layout and context.extend_mrope,
        LINEAR=_LINEAR_ACCEPT,
        num_warps=1,
    )
    context.outputs = (
        new_seq_lens,
        bonus,
        (last_steps, track_steps),
        (correct, select, ids) + (() if layout is None else (layout,)),
    )
    return accept_lens, (sim_accept_index if simulate else accept_index)
