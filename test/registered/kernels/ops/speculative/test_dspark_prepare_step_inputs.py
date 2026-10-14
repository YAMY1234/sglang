"""SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE: the one-launch verify window and
draft anchor match the separate ops they replace, value for value."""

import pytest
import torch

from sglang.kernels.ops.speculative.dspark.prepare_step_inputs import (
    PrepareStepInputs,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_MASK_ID = 7


def _inputs(bs: int, device: str, req_to_token_dtype=torch.int32):
    gen = torch.Generator().manual_seed(bs)
    pool_len = 512
    req_to_token = torch.randint(
        0, 1 << 30, (64, pool_len), generator=gen, dtype=req_to_token_dtype
    )
    # Strided, non-contiguous rows like a ScheduleBatch slice.
    req_pool_indices = torch.randperm(64, generator=gen)[: 2 * bs : 2]
    seq_lens = torch.randint(0, pool_len - 8, (bs,), generator=gen, dtype=torch.int64)
    bonus = torch.randint(0, 1000, (bs,), generator=gen, dtype=torch.int64)
    return [
        t.to(device) for t in (req_to_token, req_pool_indices, seq_lens, bonus)
    ]


def _separate_ops(req_to_token, req_pool_indices, seq_lens, bonus, verify_len):
    """The verify window as alloc_verify_window built it before the flag."""
    positions_2d = seq_lens.unsqueeze(1) + torch.arange(
        verify_len, device=seq_lens.device, dtype=torch.int64
    )
    locs = [
        req_to_token[int(r), int(s) : int(s) + verify_len]
        for r, s in zip(req_pool_indices.tolist(), seq_lens.tolist())
    ]
    verify_cache_loc = torch.cat(locs).to(torch.int64)
    anchor = torch.full((len(bonus), verify_len - 1), _MASK_ID, dtype=torch.long)
    anchor[:, 0] = bonus.cpu()
    return positions_2d.cpu(), verify_cache_loc.cpu(), anchor


@pytest.mark.parametrize("bs", [1, 3, 8])
def test_reference_matches_separate_ops_cpu(bs):
    req_to_token, rpi, seq_lens, bonus = _inputs(bs, "cpu")
    anchor_ids = torch.full((bs, 5), _MASK_ID, dtype=torch.long)
    out = PrepareStepInputs.execute(
        seq_lens=seq_lens,
        req_pool_indices=rpi,
        req_to_token=req_to_token,
        verify_len=6,
        bonus=bonus,
        anchor_ids=anchor_ids,
    )
    pos, loc, anchor = _separate_ops(req_to_token, rpi, seq_lens, bonus, 6)
    assert torch.equal(out.positions_2d, pos)
    assert torch.equal(out.verify_cache_loc, loc)
    # Only column 0 is written; the mask tokens past it stay.
    assert torch.equal(anchor_ids, anchor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("bs", [1, 2, 5, 16])
@pytest.mark.parametrize("verify_len", [6, 9])
def test_kernel_matches_assign_extend_cache_locs(bs, verify_len):
    from sglang.kernels.ops.speculative.cache_locs import (
        assign_extend_cache_locs_func,
    )

    req_to_token, rpi, seq_lens, bonus = _inputs(bs, "cuda")
    anchor_ids = torch.full(
        (bs, verify_len - 1), _MASK_ID, dtype=torch.long, device="cuda"
    )
    out = PrepareStepInputs.execute(
        seq_lens=seq_lens,
        req_pool_indices=rpi,
        req_to_token=req_to_token,
        verify_len=verify_len,
        bonus=bonus,
        anchor_ids=anchor_ids,
    )
    ref_loc = assign_extend_cache_locs_func(
        req_pool_indices=rpi,
        req_to_token=req_to_token,
        start_offset=seq_lens,
        end_offset=seq_lens + verify_len,
        batch_size=bs,
        draft_token_num=verify_len,
        device="cuda",
    )
    ref_pos = seq_lens.unsqueeze(1) + torch.arange(
        verify_len, device="cuda", dtype=torch.int64
    )
    assert out.verify_cache_loc.dtype == ref_loc.dtype
    assert torch.equal(out.verify_cache_loc, ref_loc)
    assert torch.equal(out.positions_2d, ref_pos)
    _, _, ref_anchor = _separate_ops(req_to_token, rpi, seq_lens, bonus, verify_len)
    assert torch.equal(anchor_ids.cpu(), ref_anchor)
