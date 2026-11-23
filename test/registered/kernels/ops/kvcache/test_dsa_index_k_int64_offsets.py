"""Index-K pages past the int32 byte range (page index x page bytes > 2^31).

A gather or store that computes the page's byte offset in int32 wraps once the index-K slot passes ~16.3M tokens
(64 x 132 B FP8 pages) or ~31.6M (64 x 68 B MXFP4 pages), and then reads or writes another page silently.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE = 64
D = 128


def _skip(cap_major: int = 9):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] < cap_major:
        pytest.skip(f"needs SM{cap_major}0 or newer")


def _first_page_past_int32(page_bytes: int) -> int:
    return (2**31) // page_bytes + 1


def test_fp8_gather_reads_pages_past_int32_offsets():
    _skip()
    from sglang.kernels.ops.kvcache.dsa_index_k_cache import _get_k_and_s_triton

    page_bytes = PAGE * (D + 4)
    far = _first_page_past_int32(page_bytes) + 3
    buf = torch.zeros(far + 1, page_bytes, dtype=torch.uint8, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(0)
    for p in (1, far):
        buf[p] = torch.randint(0, 256, (page_bytes,), dtype=torch.uint8, device="cuda", generator=g)
    # one request whose two pages are a near one and a far one
    page_indices = torch.tensor([[1, far]], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([2 * PAGE], dtype=torch.int64, device="cuda")
    k, s = _get_k_and_s_triton(buf, page_indices, seq_lens, seq_len_sum=2 * PAGE, max_seq_len=2 * PAGE,
                               page_size=PAGE, index_head_dim=D)
    torch.cuda.synchronize()
    for i, p in enumerate((1, far)):
        rows = slice(i * PAGE, (i + 1) * PAGE)
        assert torch.equal(k[rows], buf[p, : PAGE * D].view(PAGE, D)), f"page {p}"
        assert torch.equal(s[rows], buf[p, PAGE * D:].view(PAGE, 4)), f"page {p}"


def test_mxfp4_store_writes_pages_past_int32_offsets():
    _skip(cap_major=10)
    from sglang.kernels.ops.attention.dsa.indexer_mxfp4 import (
        index_k_bytes_per_token,
        mxfp4_quant_ref,
        store_index_k_mxfp4,
    )

    page_bytes = PAGE * index_k_bytes_per_token(D)
    far = _first_page_past_int32(page_bytes) + 3
    cache = torch.zeros(far + 1, page_bytes, dtype=torch.uint8, device="cuda")
    key = torch.randn(2, D, device="cuda").to(torch.bfloat16)
    loc = torch.tensor([far * PAGE + 5, PAGE + 5], dtype=torch.int64, device="cuda")
    store_index_k_mxfp4(key, cache, loc, page_size=PAGE)
    torch.cuda.synchronize()
    packed, ue8m0 = mxfp4_quant_ref(key.float())
    for i, p in enumerate((far, 1)):
        assert torch.equal(cache[p, 5 * D // 2: 6 * D // 2], packed[i]), f"page {p}"
        sf = PAGE * D // 2 + 5 * D // 32
        assert torch.equal(cache[p, sf: sf + D // 32], ue8m0[i]), f"page {p}"
    # the wrapped int32 offset would have landed in a low page; every other page must stay zero
    assert int(cache[2:far].count_nonzero()) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
