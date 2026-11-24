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
        x = x * torch.exp2(
            torch.randint(-20, 20, (n, D // 32), device="cuda", generator=g).float()
        ).repeat_interleave(32, 1)
    elif kind == "grid":
        # exact e2m1 grid points and every rounding midpoint, scaled by powers of two
        pts = torch.tensor(
            [
                0.0,
                0.25,
                0.5,
                0.75,
                1.0,
                1.25,
                1.5,
                1.75,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
                5.0,
                6.0,
                -6.0,
            ],
            device="cuda",
        )
        x = pts.repeat(n, D // 16) * torch.exp2(
            torch.randint(-8, 8, (n, 1), device="cuda", generator=g).float()
        )
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
    cache = torch.zeros(
        4, PAGE * index_k_bytes_per_token(D), dtype=torch.uint8, device="cuda"
    )
    store_index_k_mxfp4(key, cache, loc, page_size=PAGE)
    torch.cuda.synchronize()

    packed, ue8m0 = mxfp4_quant_ref(key.float())
    page, off = loc // PAGE, loc % PAGE
    got_vals = cache[:, : PAGE * D // 2].view(4, PAGE, D // 2)[page, off]
    got_sf = cache[:, PAGE * D // 2 :].view(4, PAGE, D // 32)[page, off]
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
        cache = torch.zeros(
            2, PAGE * index_k_bytes_per_token(D), dtype=torch.uint8, device="cuda"
        )
        store_index_k_mxfp4(
            key, cache, loc, page_size=PAGE, dcp_size=dcp, dcp_rank=rank
        )
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
    from sglang.kernels.ops.attention.dsa.indexer_mxfp4 import (
        mxfp4_quant_ref,
        q_quant_mxfp4,
    )

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


def _paged_mxfp4_cache(lens, kind="normal", seed=0):
    """Requests of ``lens`` tokens written with the MXFP4 store into scattered pages: (cache, block
    table [B, pages] int32, keys [sum(lens), D] bf16, slot of every key)."""
    from sglang.kernels.ops.attention.dsa.indexer_mxfp4 import (
        index_k_bytes_per_token,
        store_index_k_mxfp4,
    )

    g = torch.Generator(device="cuda").manual_seed(seed)
    pages_per = [-(-n // PAGE) for n in lens]
    num_pages = sum(pages_per) + 3
    perm = torch.randperm(num_pages, device="cuda", generator=g).to(torch.int32)
    table = torch.zeros(len(lens), max(pages_per), dtype=torch.int32, device="cuda")
    slots, start = [], 0
    for i, (n, p) in enumerate(zip(lens, pages_per)):
        table[i, :p] = perm[start : start + p]
        start += p
        pos = torch.arange(n, device="cuda")
        slots.append(table[i, pos // PAGE].long() * PAGE + pos % PAGE)
    slots = torch.cat(slots)
    keys = _keys(slots.numel(), kind)
    cache = torch.zeros(
        num_pages, PAGE * index_k_bytes_per_token(D), dtype=torch.uint8, device="cuda"
    )
    store_index_k_mxfp4(keys, cache, slots, page_size=PAGE)
    return cache, table, keys, slots


def _pow2_fp8_scale(values):
    """The FP8 store's per-token scale max(1e-4, amax) / 448, rounded up to a power of two."""
    s = values.abs().amax(-1).clamp(min=1e-4) / 448.0
    mant, exp = torch.frexp(s)
    return torch.exp2(torch.where(mant == 0.5, exp - 1, exp).float())


@pytest.mark.parametrize("kind", ["normal", "wide", "grid"])
def test_gather_converts_mxfp4_to_fp8_exactly(kind):
    """The prefill gather of an MXFP4 index-K gives E4M3 keys and fp32 scales whose product is the
    MXFP4 value bit for bit, in the FP8 gather's row order (requests concatenated)."""
    _skip()
    from sglang.kernels.ops.attention.dsa.indexer_mxfp4 import mxfp4_dequant_ref
    from sglang.kernels.ops.kvcache.dsa_index_k_cache import _get_k_and_s_mxfp4_triton

    lens = [1, 63, 64, 65, 300, 2 * PAGE]
    cache, table, keys, slots = _paged_mxfp4_cache(
        lens, kind if kind != "wide" else "normal"
    )
    if kind == "wide":
        # block exponents up to 12 binades apart inside a token (the exact range is 14)
        from sglang.kernels.ops.attention.dsa.indexer_mxfp4 import store_index_k_mxfp4

        g = torch.Generator(device="cuda").manual_seed(5)
        spread = torch.exp2(
            torch.randint(
                -6, 7, (keys.shape[0], D // 32), device="cuda", generator=g
            ).float()
        )
        keys = (keys.float() * spread.repeat_interleave(32, 1)).to(torch.bfloat16)
        store_index_k_mxfp4(keys, cache, slots, page_size=PAGE)
    torch.cuda.synchronize()
    k_u8, s_u8 = _get_k_and_s_mxfp4_triton(
        buf=cache,
        page_indices=table,
        seq_lens=torch.tensor(lens, device="cuda", dtype=torch.int64),
        seq_len_sum=sum(lens),
        max_seq_len=max(lens),
        page_size=PAGE,
        index_head_dim=D,
    )
    torch.cuda.synchronize()
    page, off = slots // PAGE, slots % PAGE
    vals = cache[:, : PAGE * D // 2].view(-1, PAGE, D // 2)[page, off]
    sf = cache[:, PAGE * D // 2 :].view(-1, PAGE, D // 32)[page, off]
    want = mxfp4_dequant_ref(vals, sf)
    scale = s_u8.view(torch.float32).view(-1)
    got = k_u8.view(torch.float8_e4m3fn).float() * scale[:, None]
    assert torch.equal(got, want), (got != want).nonzero()[:8]
    assert torch.equal(scale, _pow2_fp8_scale(want))


def test_pool_mxfp4_layout_move_and_gather():
    """With SGLANG_K31_INDEXER_MXFP4 the DSA pool holds 68 B per token, moves keys with their
    scales, and its gather hands the ragged indexer FP8 keys equal to the MXFP4 values."""
    _skip()
    from sglang.kernels.ops.attention.dsa.indexer_mxfp4 import (
        mxfp4_dequant_ref,
        store_index_k_mxfp4,
    )
    from sglang.srt.environ import envs
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.mem_cache.pool_host.dsa import dsa_indexer_bytes_per_token_per_layer
    from sglang.srt.runtime_context import get_parallel

    with (
        envs.SGLANG_K31_INDEXER_MXFP4.override(True),
        get_parallel().override(attn_dcp_size=1),
    ):
        assert DSATokenToKVPool.index_k_token_bytes(D) == (64, 4)
        assert (
            dsa_indexer_bytes_per_token_per_layer(
                index_head_dim=D, quant_block_size=128
            )
            == 68
        )
        pool = DSATokenToKVPool(
            size=256,
            page_size=PAGE,
            kv_lora_rank=512,
            dtype=torch.bfloat16,
            qk_rope_head_dim=64,
            layer_num=2,
            device="cuda",
            index_head_dim=D,
            enable_memory_saver=False,
            kv_cache_dim=576,
            skip_topk_layers=[False, True],
        )
    buf = pool.get_index_k_with_scale_buffer(0)
    assert buf.shape[1] == PAGE * 68 and pool.index_k_with_scale_buffer[1].shape[0] == 0
    slots = torch.arange(320, device="cuda")
    keys = _keys(320, "normal")
    store_index_k_mxfp4(keys, buf, slots, page_size=PAGE)
    source = torch.tensor([64, 127, 128, 255, 256, 319], device="cuda")
    target = source.flip(0)
    pool.move_kv_cache(target, source)
    order = slots.clone()
    order[target] = order[source]
    k_u8, s_u8 = pool.get_index_k_scale_buffer(
        0,
        torch.tensor([320], device="cuda", dtype=torch.int32),
        torch.arange(5, device="cuda", dtype=torch.int32).unsqueeze(0),
        320,
        320,
    )
    torch.cuda.synchronize()
    from sglang.kernels.ops.attention.dsa.indexer_mxfp4 import mxfp4_quant_ref

    packed, ue8m0 = mxfp4_quant_ref(keys.float())
    want = mxfp4_dequant_ref(packed, ue8m0)[order]
    got = k_u8.view(torch.float8_e4m3fn).float() * s_u8.view(torch.float32)
    assert torch.equal(got, want)


def test_pd_layout_names_the_index_k_format():
    """PD peers publish the index-K format, so an FP8 and an MXFP4 peer fail the bootstrap check."""
    from sglang.srt.disaggregation.utils import get_pd_kv_layout
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, HybridLinearKVPool

    layouts = []
    for fp4 in (False, True):
        pool = object.__new__(DSATokenToKVPool)
        pool.index_k_fp4 = fp4
        hybrid = object.__new__(HybridLinearKVPool)
        hybrid.full_kv_pool = pool
        assert get_pd_kv_layout(pool) == get_pd_kv_layout(hybrid)
        layouts.append(get_pd_kv_layout(pool))
    assert layouts == ["dsa_index_k_fp8", "dsa_index_k_mxfp4"]


@pytest.mark.parametrize("next_n", [1, 6])
def test_decode_logits_match_reference(next_n):
    """DeepGEMM's FP4 paged logits over the MXFP4 cache and query equal sum_h w_h relu(q_h . k) on
    the dequantized operands (up to the kernel's head reduction), in the native (next_n rows per
    request) and per-row layouts."""
    _skip()
    import deep_gemm

    from sglang.kernels.ops.attention.dsa.indexer_mxfp4 import (
        mxfp4_dequant_ref,
        mxfp4_paged_mqa_logits,
        q_quant_mxfp4,
    )

    H = 32
    ctx = [700, 64, 1500, 129, 3000, 2049, 5]
    B = len(ctx)
    lens = [c + next_n - 1 for c in ctx]
    cache, table, _, _ = _paged_mxfp4_cache(lens, seed=next_n)
    g = torch.Generator(device="cuda").manual_seed(11)
    rows = B * next_n
    q = torch.randn(rows, H, D, device="cuda", generator=g).to(torch.bfloat16)
    gate = torch.randn(rows, H, device="cuda", generator=g).to(torch.bfloat16)
    q4, q_sf, w = q_quant_mxfp4(q, gate, D**-0.5 * H**-0.5)
    row_len = torch.tensor(
        [c + j for c in ctx for j in range(next_n)], device="cuda", dtype=torch.int32
    )
    if next_n > 1:
        ctx_2d = row_len.view(B, next_n)
        bt = table.repeat_interleave(next_n, 0)
    else:
        ctx_2d = row_len.view(-1, 1)
        bt = table
    sched = deep_gemm.get_paged_mqa_logits_metadata(
        ctx_2d, PAGE, deep_gemm.get_num_sms()
    )
    max_len = table.shape[1] * PAGE
    logits = mxfp4_paged_mqa_logits(
        (q4, q_sf),
        cache.view(-1, PAGE, 1, 68),
        w,
        ctx_2d,
        bt,
        sched,
        max_len,
        q_offset=rows,
        B=B,
        next_n=next_n,
    )
    torch.cuda.synchronize()
    assert logits.dtype == torch.float32 and logits.shape[0] == rows
    qd = mxfp4_dequant_ref(q4, q_sf.view(torch.uint8).view(rows, H, 4))
    vals = cache[:, : PAGE * D // 2].view(-1, PAGE, D // 2)
    sfs = cache[:, PAGE * D // 2 :].view(-1, PAGE, D // 32)
    for r in range(rows):
        n = int(row_len[r])
        pages = table[r // next_n, : -(-n // PAGE)].long()
        k = mxfp4_dequant_ref(
            vals[pages].reshape(-1, D // 2), sfs[pages].reshape(-1, D // 32)
        )[:n]
        want = (torch.relu(qd[r] @ k.T) * w[r][:, None]).sum(0)
        # DeepGEMM's FP4 path may reduce heads in BF16 (median rel. error ~4e-3 on VR);
        # a wrong page, row or scale is off by O(1)
        err = (logits[r, :n] - want).norm() / want.norm().clamp(min=1e-30)
        assert err < 2e-2, (r, n, float(err))


@pytest.mark.parametrize("next_n", [2, 4, 6, 7])
def test_one_block_logits_bitwise_deepgemm(next_n):
    """The one-block MXFP4 kernel (each request's index K read once for its next_n rows) writes
    bitwise DeepGEMM's MXFP4 logits on every column DeepGEMM writes."""
    _skip()
    import deep_gemm

    from sglang.kernels.ops.attention.dsa.indexer_mxfp4 import (
        mxfp4_paged_mqa_logits,
        q_quant_mxfp4,
    )

    H = 32
    ctx = [700, 64, 1500, 129, 3000, 2049, 5, 255, 256, 9000]
    B = len(ctx)
    lens = [c + next_n - 1 for c in ctx]
    cache, table, _, _ = _paged_mxfp4_cache(lens, seed=100 + next_n)
    g = torch.Generator(device="cuda").manual_seed(12)
    rows = B * next_n
    q = torch.randn(rows, H, D, device="cuda", generator=g).to(torch.bfloat16)
    gate = torch.randn(rows, H, device="cuda", generator=g).to(torch.bfloat16)
    q4, q_sf, w = q_quant_mxfp4(q, gate, D**-0.5 * H**-0.5)
    ctx_2d = torch.tensor(
        [[c + j for j in range(next_n)] for c in ctx], device="cuda", dtype=torch.int32
    )
    sched = deep_gemm.get_paged_mqa_logits_metadata(
        ctx_2d, PAGE, deep_gemm.get_num_sms()
    )
    max_len = table.shape[1] * PAGE
    args = ((q4, q_sf), cache.view(-1, PAGE, 1, 68), w, ctx_2d)
    tail = (table.repeat_interleave(next_n, 0), sched, max_len)
    kw = dict(q_offset=rows, B=B, next_n=next_n)
    ref = mxfp4_paged_mqa_logits(*args, *tail, **kw)
    out = mxfp4_paged_mqa_logits(*args, *tail, **kw, one_block=True)
    torch.cuda.synchronize()
    for b in range(B):
        cols = min(-(-int(ctx_2d[b, -1]) // 256) * 256, max_len)
        x = ref[b * next_n : (b + 1) * next_n, :cols].contiguous().view(torch.int32)
        y = out[b * next_n : (b + 1) * next_n, :cols].contiguous().view(torch.int32)
        assert torch.equal(x, y), (b, int((x != y).sum()))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
