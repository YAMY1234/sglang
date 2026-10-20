"""DSpark static greedy verify: the accept and its bookkeeping in one launch
(SGLANG_DSPARK_FUSED_ACCEPT_TAIL), one program per request row.

Per row, from the target's argmax ids: the leading draft matches and the bonus
(AcceptGreedy; ``bonus_in`` replaces the bonus, e.g. the in-graph simulated bonus),
the simulated correct length override (build_accept_outs), commit / new seq lens and
a zero cap trim (FinalizeAcceptLens), the output tokens (BuildOutTokens), and the
gated commit lengths the static epilogue's draft-KV commit reads. Every output equals
the separate ops' bit for bit.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.dspark.dispatch import inputs_on_cuda


@triton.jit
def _static_accept_tail_kernel(
    candidates_ptr,
    target_ptr,
    seq_lens_ptr,
    verify_lens_ptr,
    gate_ptr,
    sim_ptr,
    bonus_in_ptr,
    correct_out_ptr,
    bonus_out_ptr,
    cap_trim_out_ptr,
    commit_out_ptr,
    new_seq_out_ptr,
    out_tokens_ptr,
    gated_out_ptr,
    cand_stride,
    out_tokens_stride,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_SIM: tl.constexpr,
    HAS_BONUS_IN: tl.constexpr,
    HAS_GATED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_T)
    in_t = offs < T
    gamma = T - 1
    in_draft = offs < gamma
    drafts = tl.load(
        candidates_ptr + row * cand_stride + 1 + offs, mask=in_draft, other=-1
    )
    targets = tl.load(target_ptr + row * T + offs, mask=in_draft, other=-2)
    eq = (drafts == targets) & in_draft
    num_correct = tl.min(tl.where(eq, BLOCK_T, offs), 0)
    if HAS_BONUS_IN:
        bonus = tl.load(bonus_in_ptr + row).to(tl.int64)
    else:
        bonus = tl.load(target_ptr + row * T + num_correct).to(tl.int64)
    if HAS_SIM:
        correct = tl.load(sim_ptr).to(tl.int32)
    else:
        correct = num_correct.to(tl.int32)
    commit = correct + 1
    new_seq = tl.load(seq_lens_ptr + row) + commit
    tl.store(correct_out_ptr + row, correct)
    tl.store(bonus_out_ptr + row, bonus)
    tl.store(cap_trim_out_ptr + row, 0)
    tl.store(commit_out_ptr + row, commit)
    tl.store(new_seq_out_ptr + row, new_seq)
    token = tl.where(offs == correct, bonus, tl.where(in_draft, drafts.to(tl.int64), 0))
    tl.store(out_tokens_ptr + row * out_tokens_stride + offs, token, mask=in_t)
    if HAS_GATED:
        verify_len = tl.load(verify_lens_ptr + row)
        gate = tl.load(gate_ptr)
        gated = tl.minimum(commit, verify_len.to(tl.int32)) * gate
        tl.store(gated_out_ptr + row, gated)


def static_accept_tail(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
    seq_lens: torch.Tensor,
    verify_lens: torch.Tensor,
    gate: torch.Tensor,
    correct_len_out: torch.Tensor,
    bonus_out: torch.Tensor,
    cap_trim_lens_out: torch.Tensor,
    commit_lens_out: torch.Tensor,
    new_seq_lens_out: torch.Tensor,
    out_tokens_out: torch.Tensor,
    simulated_correct_len: Optional[torch.Tensor] = None,
    bonus_in: Optional[torch.Tensor] = None,
    gated_commit_lens_out: Optional[torch.Tensor] = None,
) -> None:
    """``candidates`` [bs, T] (anchor + drafts), ``target_predict`` [bs * T] argmax ids;
    outputs are written into rows [:bs]."""
    fn = (
        _static_accept_tail_triton
        if inputs_on_cuda(candidates)
        else static_accept_tail_torch
    )
    fn(
        candidates=candidates,
        target_predict=target_predict,
        seq_lens=seq_lens,
        verify_lens=verify_lens,
        gate=gate,
        correct_len_out=correct_len_out,
        bonus_out=bonus_out,
        cap_trim_lens_out=cap_trim_lens_out,
        commit_lens_out=commit_lens_out,
        new_seq_lens_out=new_seq_lens_out,
        out_tokens_out=out_tokens_out,
        simulated_correct_len=simulated_correct_len,
        bonus_in=bonus_in,
        gated_commit_lens_out=gated_commit_lens_out,
    )


def _static_accept_tail_triton(**kw) -> None:
    candidates = kw["candidates"]
    bs, T = candidates.shape
    assert candidates.stride(1) == 1 and kw["out_tokens_out"].stride(1) == 1
    dummy = kw["seq_lens"]
    sim, bonus_in, gated = (
        kw["simulated_correct_len"],
        kw["bonus_in"],
        kw["gated_commit_lens_out"],
    )
    _static_accept_tail_kernel[(bs,)](
        candidates,
        kw["target_predict"],
        kw["seq_lens"],
        kw["verify_lens"],
        kw["gate"],
        sim if sim is not None else dummy,
        bonus_in if bonus_in is not None else dummy,
        kw["correct_len_out"],
        kw["bonus_out"],
        kw["cap_trim_lens_out"],
        kw["commit_lens_out"],
        kw["new_seq_lens_out"],
        kw["out_tokens_out"],
        gated if gated is not None else dummy,
        candidates.stride(0),
        kw["out_tokens_out"].stride(0),
        T=T,
        BLOCK_T=triton.next_power_of_2(T),
        HAS_SIM=sim is not None,
        HAS_BONUS_IN=bonus_in is not None,
        HAS_GATED=gated is not None,
        num_warps=4,
    )


def static_accept_tail_torch(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
    seq_lens: torch.Tensor,
    verify_lens: torch.Tensor,
    gate: torch.Tensor,
    correct_len_out: torch.Tensor,
    bonus_out: torch.Tensor,
    cap_trim_lens_out: torch.Tensor,
    commit_lens_out: torch.Tensor,
    new_seq_lens_out: torch.Tensor,
    out_tokens_out: torch.Tensor,
    simulated_correct_len: Optional[torch.Tensor] = None,
    bonus_in: Optional[torch.Tensor] = None,
    gated_commit_lens_out: Optional[torch.Tensor] = None,
) -> None:
    """Reference: the same outputs from per-row torch ops (CPU or GPU)."""
    bs, T = candidates.shape
    gamma = T - 1
    target = target_predict.view(bs, T)
    matches = (candidates[:, 1:] == target[:, :gamma]).to(torch.int32)
    num_correct = matches.cumprod(dim=1).sum(dim=1).to(torch.int32)
    rows = torch.arange(bs, device=candidates.device)
    if bonus_in is not None:
        bonus = bonus_in[:bs].to(torch.int64)
    else:
        bonus = target[rows, num_correct.long()].to(torch.int64)
    if simulated_correct_len is not None:
        correct = simulated_correct_len.repeat(bs).to(torch.int32)
    else:
        correct = num_correct
    commit = correct + 1
    new_seq = seq_lens[:bs] + commit.to(seq_lens.dtype)
    correct_len_out[:bs].copy_(correct)
    bonus_out[:bs].copy_(bonus)
    cap_trim_lens_out[:bs].zero_()
    commit_lens_out[:bs].copy_(commit)
    new_seq_lens_out[:bs].copy_(new_seq)
    tokens = torch.zeros((bs, T), dtype=torch.int64, device=candidates.device)
    tokens[:, :gamma] = candidates[:, 1:]
    tokens.scatter_(1, correct.long()[:, None], bonus[:, None])
    out_tokens_out[:bs].copy_(tokens)
    if gated_commit_lens_out is not None:
        gated = torch.minimum(commit, verify_lens[:bs].to(torch.int32)) * gate
        gated_commit_lens_out[:bs].copy_(gated)
