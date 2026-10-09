"""DSV4.1 sparse prefill through trtllm-gen on an fp8 workspace (SGLANG_DSV4_PREFILL_TRTLLM_FP8).

1. ``combine_topk_swa_indices(trtllm=True)`` holds the same entries as the FlashMLA
   layout, rearranged: valid window entries left-aligned, top-k compacted, per-row
   window count in ``out_seq``; with implicit and explicit (floored) windows, -1 holes,
   two requests.
2. The V4.1 dequant writes fp8 equal to its bf16 output cast to e4m3.
3. The backend's trtllm-gen call matches FlashMLA on the bf16 workspace within fp8
   precision, against an fp32 reference.
"""

import types

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import dequantize_k_cache_paged
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    combine_topk_swa_indices,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="trtllm-gen sparse MLA requires SM100",
)

DEV = "cuda"
WINDOW, TOPK, D, H, HP = 128, 512, 512, 16, 64
FP8 = torch.float8_e4m3fn


def _i32(x):
    return torch.tensor(x, dtype=torch.int32, device=DEV)


def _chunk(req_lens, prefixes, ratio, gen, floor_rows=0, holes=False):
    """Inputs for combine_topk_swa_indices over requests with the given extend and
    prefix lengths; the workspace holds c_max compressed slots per request, then
    each request's SWA gather. ``floor_rows`` floors the first rows' windows at the
    prefix, as bounded replay does, through explicit SWA indices."""
    num_reqs = len(req_lens)
    seq = [p + n for p, n in zip(prefixes, req_lens)]
    gather = [min(s, n + WINDOW - 1) for s, n in zip(seq, req_lens)]
    c_max = max(max(seq) // ratio, 1)
    qsl = [0]
    for n in req_lens:
        qsl.append(qsl[-1] + n)
    pos = torch.cat([torch.arange(p, p + n) for p, n in zip(prefixes, req_lens)]).int()
    num_tokens = qsl[-1]
    raw = torch.randint(0, c_max, (num_tokens, TOPK), generator=gen, dtype=torch.int32)
    vis = ((pos + 1) // ratio).clamp(max=TOPK)
    raw = torch.where(torch.arange(TOPK)[None, :] < vis[:, None], raw % (vis[:, None].clamp(min=1)), -1)
    if holes:
        raw[torch.rand(raw.shape, generator=gen) < 0.05] = -1
    swa_offsets = [0]
    for g in gather:
        swa_offsets.append(swa_offsets[-1] + g)
    args = dict(
        topk_indices=raw.to(DEV),
        query_start_loc=_i32(qsl),
        query_pos=pos.to(DEV),
        seq_lens=_i32(seq),
        gather_lens=_i32(gather),
        compressed_base=_i32([r * c_max for r in range(num_reqs)]),
        swa_base=_i32([num_reqs * c_max + o for o in swa_offsets[:-1]]),
        window_size=WINDOW,
        compress_ratio=ratio,
        topk=TOPK,
    )
    if floor_rows:
        # Explicit window: offsets into each request's SWA gather, -1 below the floor.
        idx = torch.full((num_tokens, WINDOW), -1, dtype=torch.int32)
        lens = torch.zeros(num_tokens, dtype=torch.int32)
        for r in range(num_reqs):
            start = seq[r] - gather[r]
            for t in range(qsl[r], qsl[r + 1]):
                p = int(pos[t])
                lo = max(p - WINDOW + 1, 0)
                if t - qsl[r] < floor_rows:
                    lo = max(lo, prefixes[r])
                n = p - lo + 1
                idx[t, :n] = torch.arange(lo, p + 1, dtype=torch.int32) - start
                lens[t] = n
        args.update(swa_indices=idx.to(DEV), swa_lengths=lens.to(DEV))
    n_ws = num_reqs * c_max + swa_offsets[-1]
    return args, n_ws


@pytest.mark.parametrize(
    "case",
    [
        dict(req_lens=[300], prefixes=[0], ratio=1),
        dict(req_lens=[1152], prefixes=[7040], ratio=2, floor_rows=128),
        dict(req_lens=[200, 333], prefixes=[700, 4000], ratio=1, floor_rows=128, holes=True),
        dict(req_lens=[257, 64], prefixes=[0, 9000], ratio=4, holes=True),
    ],
)
def test_trtllm_layout_holds_the_same_entries(case):
    gen = torch.Generator().manual_seed(0)
    args, _ = _chunk(gen=gen, **case)
    ref_idx, ref_len = combine_topk_swa_indices(**args)
    seq = torch.empty(ref_idx.shape[0], dtype=torch.int32, device=DEV)
    got_idx, got_len = combine_topk_swa_indices(**args, trtllm=True, out_seq=seq)
    assert got_idx.shape[1] % 4 == 0
    sw_base = int(args["swa_base"].min())
    for t in range(ref_idx.shape[0]):
        row = ref_idx[t, : ref_len[t]]
        row = row[row >= 0]
        sw = row[row >= sw_base].sort().values
        tk = row[row < sw_base].sort().values
        n_sw, n_tk = sw.numel(), tk.numel()
        assert int(seq[t]) == n_sw and int(got_len[t]) == WINDOW + n_tk
        assert torch.equal(got_idx[t, :n_sw].sort().values, sw)
        assert bool((got_idx[t, n_sw:WINDOW] == -1).all())
        assert torch.equal(got_idx[t, WINDOW : WINDOW + n_tk].sort().values, tk)
        assert bool((got_idx[t, WINDOW + n_tk :] == -1).all())


def _v41_cache(layout, num_tokens, page, gen):
    nbytes = layout.page_bytes(page)
    cache = torch.randint(0, 256, (num_tokens // page, nbytes), generator=gen, dtype=torch.uint8)
    if layout is KVLayout.V41:
        # e4m3 codes without NaN; ue8m0 scales near 1.
        data = cache[:, : page * 512].view(-1)
        data[(data & 0x7F) == 0x7F] = 0x3C
        cache[:, page * 512 : page * 528] = torch.randint(124, 130, (num_tokens // page, page * 16), generator=gen, dtype=torch.uint8)
    else:
        cache[:, page * 256 : page * 288] = torch.randint(0x30, 0x40, (num_tokens // page, page * 32), generator=gen, dtype=torch.uint8)
    return cache.to(DEV)


@pytest.mark.parametrize("layout", [KVLayout.V41, KVLayout.V41_FP4])
def test_fp8_dequant_is_the_bf16_dequant_in_e4m3(layout):
    gen = torch.Generator().manual_seed(1)
    page = 64
    cache = _v41_cache(layout, 4096, page, gen)
    ids = torch.randperm(4096, generator=gen)[:1000].int().to(DEV)
    bf = dequantize_k_cache_paged(cache, ids, page, layout=layout)
    f8 = torch.empty(1000, 1, D, dtype=FP8, device=DEV)
    dequantize_k_cache_paged(cache, ids, page, out=f8, layout=layout)
    want = bf.float().clamp(-448, 448).to(FP8)
    assert torch.equal(f8.view(torch.uint8), want.view(torch.uint8))


def _reference(q, ws, idx, lens, sink, scale):
    out = torch.empty(q.shape[0], H, D, device=DEV)
    for t in range(q.shape[0]):
        row = idx[t, : lens[t]]
        kv = ws[row[row >= 0].long()].float()
        lg = (q[t, :H].float() @ kv.T) * scale
        m = torch.maximum(lg.max(1).values, sink)
        p = torch.exp(lg - m[:, None])
        out[t] = (p @ kv) / (p.sum(1) + torch.exp(sink - m))[:, None]
    return out


@pytest.mark.parametrize(
    "case",
    [
        dict(req_lens=[1152], prefixes=[7040], ratio=1, floor_rows=128),
        dict(req_lens=[600, 300], prefixes=[0, 3000], ratio=2, holes=True),
    ],
)
def test_trtllm_fp8_matches_flashmla_within_fp8(case):
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend

    gen = torch.Generator().manual_seed(2)
    args, n_ws = _chunk(gen=gen, **case)
    scale = D**-0.5
    cache = _v41_cache(KVLayout.V41, 1 << 16, 64, gen)
    ids = torch.randperm(1 << 16, generator=gen)[:n_ws].int().to(DEV)
    ws = dequantize_k_cache_paged(cache, ids, 64, layout=KVLayout.V41)
    ws8 = torch.zeros(-(-n_ws // 64) * 64, D, dtype=FP8, device=DEV)
    dequantize_k_cache_paged(cache, ids, 64, out=ws8[:n_ws].unsqueeze(1), layout=KVLayout.V41)
    num_tokens = args["topk_indices"].shape[0]
    q = torch.zeros(num_tokens, HP, D, dtype=torch.bfloat16, device=DEV)
    q[:, :H] = torch.randn(num_tokens, H, D, generator=gen).to(DEV)
    sink = torch.zeros(HP, device=DEV)
    sink[:H] = torch.randn(H, generator=gen).to(DEV)

    idx_a, len_a = combine_topk_swa_indices(**args)
    o_a, _, _ = flash_mla_sparse_fwd(q=q, kv=ws, indices=idx_a.unsqueeze(1), sm_scale=scale, d_v=D, attn_sink=sink, topk_length=len_a)
    seq = torch.empty(num_tokens, dtype=torch.int32, device=DEV)
    idx_t, len_t = combine_topk_swa_indices(**args, trtllm=True, out_seq=seq)
    backend = types.SimpleNamespace(
        device=torch.device(DEV), softmax_scale=scale, _trtllm_prefill_state=None
    )
    for name in ("_trtllm_fp8_sparse_prefill", "_trtllm_prefill_launcher"):
        setattr(backend, name, getattr(DeepseekV4AttnBackend, name).__get__(backend))
    o_c = backend._trtllm_fp8_sparse_prefill(q, ws8, idx_t, len_t, seq, sink, H)

    rows = torch.cat([torch.arange(16), torch.randint(0, num_tokens, (32,), generator=gen)]).to(DEV)
    ref = _reference(q[rows], ws, idx_a[rows], len_a[rows], sink[:H], scale)
    rel = lambda o: ((o[rows, :H].float() - ref).norm() / ref.norm()).item()
    assert not torch.isnan(o_c).any()
    assert rel(o_a) < 5e-3
    # e4m3 Q and P: vLLM's precision, about 2-3e-2 here.
    assert rel(o_c) < 5e-2
