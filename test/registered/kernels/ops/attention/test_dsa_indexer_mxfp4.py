"""MXFP4 DSA indexer key cache and query: the Triton writers against the exact torch reference, byte for byte."""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE = 64
D = 128


def _skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("cvt.rn.satfinite.e2m1x2.f32 needs SM100 or newer")


def _keys(n: int, kind: str) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(22)
    x = torch.randn(n, D, device="cuda", generator=g)
    if kind == "wide":
        x = x * torch.exp2(torch.randint(-20, 20, (n, D // 32), device="cuda", generator=g).float()).repeat_interleave(32, 1)
    elif kind == "grid":
        # exact e2m1 grid points and every rounding midpoint, scaled by powers of two
        pts = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, -6.0],
                           device="cuda")
        x = pts.repeat(n, D // 16) * torch.exp2(torch.randint(-8, 8, (n, 1), device="cuda", generator=g).float())
    elif kind == "edge":
        x[0] = 0.0
        x[1] = 1e-40
        x[2, :32] = 6.0 * 2.0**-126
        x[3] = 3.0e38
        x[4, ::2] = -x[4, ::2]
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("kind", ["normal", "wide", "grid", "edge"])
def test_quant_matches_reference(kind):
    _skip()
    from sglang.kernels.ops.attention.dsa.indexer_mxfp4 import (
        index_k_bytes_per_token,
        mxfp4_quant_ref,
        store_index_k_mxfp4,
    )

    n = 3 * PAGE + 5
    key = _keys(n, kind)
    loc = torch.randperm(4 * PAGE, device="cuda")[:n].to(torch.int64)
    cache = torch.zeros(4, PAGE * index_k_bytes_per_token(D), dtype=torch.uint8, device="cuda")
    store_index_k_mxfp4(key, cache, loc, page_size=PAGE)
    torch.cuda.synchronize()

    packed, ue8m0 = mxfp4_quant_ref(key.float())
    page, off = loc // PAGE, loc % PAGE
    got_vals = cache[:, : PAGE * D // 2].view(4, PAGE, D // 2)[page, off]
    got_sf = cache[:, PAGE * D // 2:].view(4, PAGE, D // 32)[page, off]
    assert torch.equal(got_vals, packed), (got_vals != packed).nonzero()[:8]
    assert torch.equal(got_sf, ue8m0), (got_sf != ue8m0).nonzero()[:8]


def test_store_skips_other_dcp_ranks_and_negative_slots():
    _skip()
    from sglang.kernels.ops.attention.dsa.indexer_mxfp4 import (
        index_k_bytes_per_token,
        mxfp4_quant_ref,
        store_index_k_mxfp4,
    )

    n, dcp = 2 * PAGE, 4
    key = _keys(n, "normal")
    loc = torch.arange(n, device="cuda", dtype=torch.int64) * 3 + 1
    loc[5] = -1
    for rank in range(dcp):
        cache = torch.zeros(2, PAGE * index_k_bytes_per_token(D), dtype=torch.uint8, device="cuda")
        store_index_k_mxfp4(key, cache, loc, page_size=PAGE, dcp_size=dcp, dcp_rank=rank)
        torch.cuda.synchronize()
        packed, _ = mxfp4_quant_ref(key.float())
        vals = cache[:, : PAGE * D // 2].reshape(-1, D // 2)
        owned = (loc >= 0) & (loc % dcp == rank)
        rows = loc[owned] // dcp
        assert torch.equal(vals[rows], packed[owned])
        untouched = torch.ones(vals.shape[0], dtype=torch.bool, device="cuda")
        untouched[rows] = False
        assert int(vals[untouched].count_nonzero()) == 0


@pytest.mark.parametrize("tokens", [1, 7, 300, 2048])
def test_q_quant_matches_reference(tokens):
    _skip()
    from sglang.kernels.ops.attention.dsa.indexer_mxfp4 import mxfp4_quant_ref, q_quant_mxfp4

    g = torch.Generator(device="cuda").manual_seed(tokens)
    q = torch.randn(tokens, 32, D, device="cuda", generator=g).to(torch.bfloat16)
    w = torch.randn(tokens, 32, device="cuda", generator=g).to(torch.bfloat16)
    scale = 128**-0.5 * 32**-0.5
    q4, q_sf, w_out = q_quant_mxfp4(q, w, scale)
    torch.cuda.synchronize()
    packed, ue8m0 = mxfp4_quant_ref(q.float())
    assert torch.equal(q4, packed)
    assert torch.equal(q_sf.view(torch.uint8).view(tokens, 32, 4), ue8m0)
    assert torch.equal(w_out, w.float() * torch.tensor(scale, dtype=torch.float32))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
