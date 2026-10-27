# SPDX-License-Identifier: Apache-2.0
"""Opt-in B1 final acceptance, commit outputs and the next draft window."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.cache_locs import _prepare_draft_window_kernel
from sglang.kernels.ops.speculative.chain_commit import _chain_accept_commit_kernel


@triton.jit
def _chain_accept_commit_next_kernel(
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
    next_req_pool_indices,
    next_req_to_token,
    next_seq_lens,
    next_positions,
    next_mrope,
    next_draft_locs,
    next_window_locs,
    next_mrope_stride,
    next_fill_seq_len,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    VERIFY: tl.constexpr,
    SIMULATE: tl.constexpr,
    REAL_DRAFT_TOKENS: tl.constexpr,
    TRACK_INTERVAL: tl.constexpr,
    EXTEND_LAYOUT: tl.constexpr,
    EXTEND_MROPE: tl.constexpr,
    NEXT_POOL_LEN: tl.constexpr,
    NEXT_MROPE: tl.constexpr,
    LINEAR: tl.constexpr = False,
):
    # Direct JIT calls in one B1 program. Keep the public typed store in the
    # first callee and its original signed reload in the second callee.
    _chain_accept_commit_kernel(
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
        WIDTH=WIDTH,
        BLOCK=BLOCK,
        VERIFY=VERIFY,
        SIMULATE=SIMULATE,
        REAL_DRAFT_TOKENS=REAL_DRAFT_TOKENS,
        TRACK_INTERVAL=TRACK_INTERVAL,
        EXTEND_LAYOUT=EXTEND_LAYOUT,
        EXTEND_MROPE=EXTEND_MROPE,
        LINEAR=LINEAR,
    )
    _prepare_draft_window_kernel(
        next_req_pool_indices,
        next_req_to_token,
        new_seq_lens,
        next_seq_lens,
        next_positions,
        next_mrope,
        next_draft_locs,
        next_window_locs,
        next_mrope_stride,
        next_fill_seq_len,
        pool_len=NEXT_POOL_LEN,
        STEPS=6,
        WINDOW=7,
        BLOCK=8,
        WRITE_SEQ_LENS=True,
        WRITE_POSITIONS=True,
        WRITE_MROPE=NEXT_MROPE,
        WRITE_DRAFT=True,
        WRITE_WINDOW=True,
    )


def chain_commit_next_sources_supported(
    seq_lens, req_pool_indices, req_to_token, num_steps, window
):
    """Metadata available before a post-verify slot allocates its NextDraft."""
    integer = (torch.int32, torch.int64)
    return (
        num_steps == 6
        and window == 7
        and seq_lens.shape == (1,)
        and seq_lens.dtype in integer
        and seq_lens.is_cuda
        and req_pool_indices.shape == (1,)
        and req_pool_indices.dtype in integer
        and req_pool_indices.is_contiguous()
        and req_to_token.ndim == 2
        and req_to_token.shape[0] > 0
        and req_to_token.shape[1] > 0
        and req_to_token.dtype in integer
        and req_to_token.is_contiguous()
        and req_pool_indices.device == req_to_token.device == seq_lens.device
    )


def chain_commit_next_inputs_supported(seq_lens, stage):
    """Late rejection leaves the fresh context unconsumed for the old path."""
    nxt = stage.next_draft
    if nxt is None or not chain_commit_next_sources_supported(
        seq_lens,
        stage.req_pool_indices,
        stage.req_to_token,
        stage.num_steps,
        stage.window,
    ):
        return False
    if nxt.num_requests != 1:
        return False
    for tensor, shape in (
        (nxt.seq_lens, (1,)),
        (nxt.positions, (1,)),
        (nxt.draft_out_cache_loc, (6,)),
        (nxt.window, (7,)),
    ):
        if (
            tensor.shape != shape
            or tensor.dtype != torch.int64
            or tensor.device != seq_lens.device
            or not tensor.is_contiguous()
        ):
            return False
    return nxt.mrope is None or (
        nxt.mrope.shape == (3, 1)
        and nxt.mrope.dtype == torch.int64
        and nxt.mrope.device == seq_lens.device
        and nxt.mrope.stride(1) == 1
        and nxt.mrope.stride(0) > 0
    )
