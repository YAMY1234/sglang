import importlib.util

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_TRITON = importlib.util.find_spec("triton") is not None

if _HAS_TORCH:
    import torch
else:
    torch = None

pytestmark = pytest.mark.skipif(
    not _HAS_TORCH or not _HAS_TRITON or not torch.cuda.is_available(),
    reason="DSV4 metadata Triton tests require CUDA, torch, and triton",
)


def _expand_prefill_casually_reference(
    seq_lens,
    extend_seq_lens,
    req_pool_indices,
    padded_num_tokens=None,
):
    seq_lens_out = []
    req_pool_indices_out = []
    last_non_empty_req = 0
    for i, (seq_len, extend_len) in enumerate(zip(seq_lens, extend_seq_lens)):
        if extend_len <= 0:
            continue
        last_non_empty_req = i
        seq_lens_out.extend(range(seq_len - extend_len + 1, seq_len + 1))
        req_pool_indices_out.extend([req_pool_indices[i]] * extend_len)

    if padded_num_tokens is not None and padded_num_tokens > len(seq_lens_out):
        pad_size = padded_num_tokens - len(seq_lens_out)
        seq_lens_out.extend([1] * pad_size)
        req_pool_indices_out.extend([req_pool_indices[last_non_empty_req]] * pad_size)

    return (
        torch.tensor(seq_lens_out, dtype=torch.int32, device="cuda"),
        torch.tensor(req_pool_indices_out, dtype=torch.int64, device="cuda"),
    )


def test_expand_prefill_casually_ragged_matches_reference():
    from sglang.srt.layers.attention.dsv4.metadata_kernel import (
        expand_prefill_casually_ragged,
    )

    seq_lens_cpu = [10, 5, 17, 9]
    extend_seq_lens_cpu = [3, 1, 5, 2]
    req_pool_indices_cpu = [4, 7, 1, 5]
    num_tokens = sum(extend_seq_lens_cpu)
    padded_num_tokens = num_tokens + 5

    seq_lens = torch.tensor(seq_lens_cpu, dtype=torch.int64, device="cuda")
    extend_seq_lens = torch.tensor(
        extend_seq_lens_cpu, dtype=torch.int32, device="cuda"
    )
    extend_start_loc = torch.zeros_like(extend_seq_lens)
    extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
    req_pool_indices = torch.tensor(req_pool_indices_cpu, dtype=torch.int64, device="cuda")

    actual_seq_lens, actual_req_pool_indices = expand_prefill_casually_ragged(
        seq_lens=seq_lens,
        extend_seq_lens=extend_seq_lens,
        extend_start_loc=extend_start_loc,
        req_pool_indices=req_pool_indices,
        num_tokens=num_tokens,
        padded_num_tokens=padded_num_tokens,
        max_extend_len=max(extend_seq_lens_cpu),
        pad_req_index=len(extend_seq_lens_cpu) - 1,
    )
    expected_seq_lens, expected_req_pool_indices = _expand_prefill_casually_reference(
        seq_lens_cpu,
        extend_seq_lens_cpu,
        req_pool_indices_cpu,
        padded_num_tokens=padded_num_tokens,
    )

    torch.testing.assert_close(actual_seq_lens, expected_seq_lens, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_req_pool_indices, expected_req_pool_indices, rtol=0, atol=0
    )


def test_expand_prefill_casually_fixed_length_matches_reference():
    from sglang.srt.layers.attention.dsv4.metadata_kernel import (
        expand_prefill_casually_fixed_length,
    )

    seq_lens_cpu = [11, 20, 33]
    tokens_per_bs = 4
    extend_seq_lens_cpu = [tokens_per_bs] * len(seq_lens_cpu)
    req_pool_indices_cpu = [3, 9, 2]
    num_tokens = tokens_per_bs * len(seq_lens_cpu)
    padded_num_tokens = num_tokens + 3

    seq_lens = torch.tensor(seq_lens_cpu, dtype=torch.int64, device="cuda")
    req_pool_indices = torch.tensor(req_pool_indices_cpu, dtype=torch.int64, device="cuda")

    actual_seq_lens, actual_req_pool_indices = expand_prefill_casually_fixed_length(
        seq_lens=seq_lens,
        req_pool_indices=req_pool_indices,
        tokens_per_bs=tokens_per_bs,
        num_tokens=num_tokens,
        padded_num_tokens=padded_num_tokens,
    )
    expected_seq_lens, expected_req_pool_indices = _expand_prefill_casually_reference(
        seq_lens_cpu,
        extend_seq_lens_cpu,
        req_pool_indices_cpu,
        padded_num_tokens=padded_num_tokens,
    )

    torch.testing.assert_close(actual_seq_lens, expected_seq_lens, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_req_pool_indices, expected_req_pool_indices, rtol=0, atol=0
    )
