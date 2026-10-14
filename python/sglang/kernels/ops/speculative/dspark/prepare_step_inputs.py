"""DSpark decode step inputs in one launch: the verify window's positions and cache
locations, and the draft block's anchor column (port of muLLM's
_prepare_dflash_inputs_all_groups_kernel, 14eee36d)."""

from __future__ import annotations

from typing import Optional

import msgspec
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.dspark.dispatch import inputs_on_cuda


class StepInputs(msgspec.Struct, frozen=True):
    # [bs, verify_len] int64: seq_len + column.
    positions_2d: torch.Tensor
    # [bs * verify_len] int64: req_to_token[req, seq_len + column].
    verify_cache_loc: torch.Tensor


@triton.jit
def _prepare_step_inputs_kernel(
    seq_lens_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    positions_ptr,
    cache_loc_ptr,
    bonus_ptr,
    anchor_ptr,
    anchor_stride,
    pool_len: tl.constexpr,
    VERIFY_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_ANCHOR: tl.constexpr,
):
    r = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + r).to(tl.int64)
    req = tl.load(req_pool_indices_ptr + r).to(tl.int64)
    col = tl.arange(0, BLOCK)
    mask = col < VERIFY_LEN
    pos = seq_len + col
    loc = tl.load(req_to_token_ptr + req * pool_len + pos, mask=mask, other=0)
    tl.store(positions_ptr + r * VERIFY_LEN + col, pos, mask=mask)
    tl.store(cache_loc_ptr + r * VERIFY_LEN + col, loc.to(tl.int64), mask=mask)
    if HAS_ANCHOR:
        tl.store(anchor_ptr + r * anchor_stride, tl.load(bonus_ptr + r))


class PrepareStepInputs:
    @classmethod
    def execute(cls, **kwargs) -> StepInputs:
        if inputs_on_cuda(**kwargs):
            return cls.triton(**kwargs)
        return cls.torch(**kwargs)

    @classmethod
    def torch(
        cls,
        *,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        verify_len: int,
        bonus: Optional[torch.Tensor] = None,
        anchor_ids: Optional[torch.Tensor] = None,
    ) -> StepInputs:
        cols = torch.arange(verify_len, device=seq_lens.device, dtype=torch.int64)
        positions_2d = seq_lens.to(torch.int64).unsqueeze(1) + cols
        rows = req_pool_indices.to(torch.int64).unsqueeze(1)
        verify_cache_loc = req_to_token[rows, positions_2d].to(torch.int64).reshape(-1)
        if anchor_ids is not None:
            anchor_ids[:, 0].copy_(bonus.view(-1))
        return StepInputs(positions_2d=positions_2d, verify_cache_loc=verify_cache_loc)

    @classmethod
    def triton(
        cls,
        *,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        verify_len: int,
        bonus: Optional[torch.Tensor] = None,
        anchor_ids: Optional[torch.Tensor] = None,
    ) -> StepInputs:
        bs = seq_lens.shape[0]
        device = seq_lens.device
        positions_2d = torch.empty((bs, verify_len), dtype=torch.int64, device=device)
        verify_cache_loc = torch.empty(
            (bs * verify_len,), dtype=torch.int64, device=device
        )
        has_anchor = anchor_ids is not None
        if bs > 0:
            _prepare_step_inputs_kernel[(bs,)](
                seq_lens,
                req_pool_indices,
                req_to_token,
                positions_2d,
                verify_cache_loc,
                bonus if has_anchor else seq_lens,
                anchor_ids if has_anchor else positions_2d,
                anchor_ids.stride(0) if has_anchor else 0,
                pool_len=req_to_token.shape[1],
                VERIFY_LEN=verify_len,
                BLOCK=triton.next_power_of_2(verify_len),
                HAS_ANCHOR=has_anchor,
            )
        return StepInputs(positions_2d=positions_2d, verify_cache_loc=verify_cache_loc)
