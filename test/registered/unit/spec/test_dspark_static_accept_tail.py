"""SGLANG_DSPARK_FUSED_ACCEPT_TAIL: the one-launch static greedy accept tail writes
exactly what the static epilogue's separate ops write (AcceptGreedy, build_accept_outs
and the gated commit lengths), for padded rows, gate 0 / 1, simulated acceptance on /
off and a precomputed (in-graph simulated) bonus."""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    AcceptGreedy,
    _row_argmax,
)
from sglang.kernels.ops.speculative.dspark.static_accept_tail import (
    static_accept_tail,
    static_accept_tail_torch,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    AcceptOuts,
    build_accept_outs,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")

T = 6  # gamma 5
VOCAB = 64
MAX_BS = 64


def _inputs(bs: int, *, gate: int, simulated: bool, bonus_in: bool, seed: int, device):
    g = torch.Generator().manual_seed(seed)
    target = torch.randint(0, VOCAB, (bs, T), generator=g)
    logits = torch.randn(bs * T, VOCAB, generator=g)
    logits[torch.arange(bs * T), target.reshape(-1)] += 100.0
    candidates = torch.empty(bs, T, dtype=torch.int64)
    candidates[:, 0] = torch.randint(0, VOCAB, (bs,), generator=g)
    candidates[:, 1:] = target[:, :-1]
    # First mismatch at a random column (T - 1: every draft accepted).
    stop = torch.randint(0, T, (bs,), generator=g)
    for b in range(bs):
        if stop[b] < T - 1:
            candidates[b, 1 + stop[b]] = (target[b, stop[b]] + 1) % VOCAB
    live = torch.rand(bs, generator=g) < 0.75
    live[0] = True
    tensors = dict(
        candidates=candidates,
        logits=logits,
        seq_lens=torch.randint(10, 1000, (bs,), generator=g),
        verify_lens=torch.where(live, T, 0).to(torch.int64),
        gate=torch.tensor([gate], dtype=torch.int32),
        simulated=(torch.randint(0, T, (1,), generator=g) if simulated else None),
        bonus=(torch.randint(0, VOCAB, (bs,), generator=g) if bonus_in else None),
    )
    return {k: (v.to(device) if v is not None else None) for k, v in tensors.items()}


def _bufs(device) -> AcceptOuts:
    def z(*shape, dtype):
        return torch.full(shape, -7, dtype=dtype, device=device)

    return AcceptOuts(
        correct_len=z(MAX_BS, dtype=torch.int64),
        bonus=z(MAX_BS, dtype=torch.int64),
        cap_trim_lens=z(MAX_BS, dtype=torch.int32),
        commit_lens=z(MAX_BS, dtype=torch.int32),
        new_seq_lens=z(MAX_BS, dtype=torch.int64),
        out_tokens=z(MAX_BS, T, dtype=torch.int64),
    )


def _separate_ops(x, out: AcceptOuts) -> torch.Tensor:
    """The static epilogue's unfused path: returns the gated commit lengths."""
    bs = x["candidates"].shape[0]
    correct_len, bonus, cap_trim_lens = AcceptGreedy.execute(
        candidates=x["candidates"],
        target_logits=x["logits"],
        verify_num_draft_tokens=T,
    )
    if x["bonus"] is not None:
        bonus = x["bonus"].to(bonus.dtype).view_as(bonus)
    tp_sync = SpecTpSync(SimpleNamespace(world_size=1, rank_in_group=0))
    accept = build_accept_outs(
        correct_len=correct_len,
        bonus=bonus,
        cap_trim_lens=cap_trim_lens,
        prefix_lens=x["seq_lens"][:bs],
        draft_tokens=x["candidates"][:, 1:].contiguous(),
        gamma=T - 1,
        verify_num_draft_tokens=T,
        tp_sync=tp_sync,
        sync_site=SpecTpSyncSite.DSPARK_ACCEPT_GRAPH,
        simulated_correct_len=x["simulated"],
        out=AcceptOuts(*(getattr(out, f)[:bs] for f in AcceptOuts.__struct_fields__)),
    )
    return (
        torch.minimum(accept.commit_lens, x["verify_lens"].to(torch.int32)) * x["gate"]
    )


def _fused(fn, x, out: AcceptOuts, gated_out: torch.Tensor) -> None:
    fn(
        candidates=x["candidates"],
        target_predict=_row_argmax(x["logits"]).view(-1),
        seq_lens=x["seq_lens"],
        verify_lens=x["verify_lens"],
        gate=x["gate"],
        correct_len_out=out.correct_len,
        bonus_out=out.bonus,
        cap_trim_lens_out=out.cap_trim_lens,
        commit_lens_out=out.commit_lens,
        new_seq_lens_out=out.new_seq_lens,
        out_tokens_out=out.out_tokens,
        simulated_correct_len=x["simulated"],
        bonus_in=x["bonus"],
        gated_commit_lens_out=gated_out,
    )


def _check(fn, device, bs, gate, simulated, bonus_in, seed):
    x = _inputs(
        bs, gate=gate, simulated=simulated, bonus_in=bonus_in, seed=seed, device=device
    )
    ref, got = _bufs(device), _bufs(device)
    gated_ref = _separate_ops(x, ref)
    gated_got = torch.full((MAX_BS,), -7, dtype=torch.int32, device=device)
    _fused(fn, x, got, gated_got)
    for f in AcceptOuts.__struct_fields__:
        r, o = getattr(ref, f), getattr(got, f)
        assert r.dtype == o.dtype, f
        assert torch.equal(r[:bs].cpu(), o[:bs].cpu()), f
        # Rows past the batch stay untouched.
        assert torch.equal(r[bs:].cpu(), o[bs:].cpu()), f
    assert torch.equal(gated_ref.cpu(), gated_got[:bs].cpu())
    assert bool((gated_got[bs:] == -7).all())


CASES = [
    (bs, gate, simulated, bonus_in)
    for bs in (1, 3, 8, 64)
    for gate in (0, 1)
    for simulated in (False, True)
    for bonus_in in (False, True)
]


@pytest.mark.parametrize("bs,gate,simulated,bonus_in", CASES)
def test_torch_reference_matches_separate_ops(bs, gate, simulated, bonus_in):
    _check(
        static_accept_tail_torch,
        "cpu",
        bs,
        gate,
        simulated,
        bonus_in,
        seed=bs * 7 + gate,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("bs,gate,simulated,bonus_in", CASES)
def test_triton_matches_separate_ops(bs, gate, simulated, bonus_in):
    _check(
        static_accept_tail, "cuda", bs, gate, simulated, bonus_in, seed=bs * 11 + gate
    )
