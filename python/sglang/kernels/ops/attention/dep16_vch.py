# SPDX-License-Identifier: Apache-2.0
"""VCH: virtual-chunked q7 verify attention on the native trtllm-gen kernels (DEP16 decode).

For a request of length L the keys [0, L-6) are visible to all 7 query rows. They are cut into page-aligned
chunks of about equal size across the batch (exactly NV chunks, NV = SM count / KV heads, so the q1 call runs
one CTA per SM and needs no KV split reduction). Each chunk is one virtual request of the native q1 decode
kernel, with the 7 x G query rows of a KV head folded into 7 * G heads. The causal tail (keys L-6 .. L-7+t
of row t, at most 6) is computed in FP32 inside the merge, which combines the chunk partials by their log2
LSE and can apply the attention output gate and the static FP8 quantization of the following o_proj.
Requests shorter than min_len are not split (one chunk).
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl

QLEN = 7

# Per-call hand-off from the model to the attention backend (one forward runs at a time per process):
# the output gate and the o_proj static FP8 scale, and whether the attention front already wrote the fold.
_gate = None
_folded = False


def set_gate(gate: torch.Tensor, scale: torch.Tensor):
    global _gate
    _gate = (gate, scale)


def clear_gate():
    global _gate
    _gate = None


def take_gate():
    global _gate
    gate, _gate = _gate, None
    return gate


def mark_folded():
    global _folded
    _folded = True


def take_folded() -> bool:
    global _folded
    folded, _folded = _folded, False
    return folded


@triton.jit
def _vch_plan_kernel(seq_lens_ptr, B, cs_ptr, cn_ptr, NV: tl.constexpr, BLOCK_B: tl.constexpr, PG: tl.constexpr,
                     QL: tl.constexpr, MIN_LEN: tl.constexpr, MAX_PAGES: tl.constexpr):
    """Chunks per request: the smallest chunk size c (pages) whose total chunk count fits in NV, found by
    bisection; requests shorter than MIN_LEN keep one chunk."""
    b = tl.arange(0, BLOCK_B)
    bm = b < B
    L = tl.load(seq_lens_ptr + b, mask=bm, other=0)
    e = tl.maximum(L - (QL - 1), 1)
    p = (e + PG - 1) // PG
    split = bm & (L >= MIN_LEN)
    fixed = tl.sum((bm & ~split).to(tl.int32), axis=0)
    lo = 1
    hi = MAX_PAGES
    while lo < hi:
        mid = (lo + hi) // 2
        cnt = fixed + tl.sum(tl.where(split, (p + mid - 1) // mid, 0), axis=0)
        if cnt <= NV:
            hi = mid
        else:
            lo = mid + 1
    n = tl.where(split, (p + lo - 1) // lo, tl.where(bm, 1, 0))
    tl.store(cs_ptr + b, tl.cumsum(n, axis=0) - n, mask=bm)
    tl.store(cn_ptr + b, n, mask=bm)


@triton.jit
def _vch_table_kernel(seq_lens_ptr, B, cs_ptr, cn_ptr, pt_ptr, pt_stride, vreq_ptr, vstart_ptr, vnp_ptr, vlen_ptr,
                      vt_ptr, vt_stride, BLOCK_B: tl.constexpr, PG: tl.constexpr, QL: tl.constexpr,
                      BLOCK: tl.constexpr):
    """Chunk v: its request, page range (the request's pages split as evenly as possible) and page indices;
    chunks past the plan's total are 1-token dummies on page table entry (0, 0)."""
    v = tl.program_id(0)
    blk = tl.program_id(1)
    b = tl.arange(0, BLOCK_B)
    bm = b < B
    cs = tl.load(cs_ptr + b, mask=bm, other=0)
    n = tl.load(cn_ptr + b, mask=bm, other=0)
    own = bm & (cs <= v) & (v < cs + n)
    valid = tl.sum(own.to(tl.int32), axis=0) > 0
    i = tl.sum(tl.where(own, b, 0), axis=0)
    L = tl.load(seq_lens_ptr + i)
    e = tl.maximum(L - (QL - 1), 1)
    p = (e + PG - 1) // PG
    ni = tl.maximum(tl.sum(tl.where(own, n, 0), axis=0), 1)
    k = v - tl.sum(tl.where(own, cs, 0), axis=0)
    base = p // ni
    extra = p - base * ni
    start = tl.where(valid, k * base + tl.minimum(k, extra), 0)
    npg = tl.where(valid, base + (k < extra).to(tl.int32), 1)
    ln = tl.where(valid, tl.minimum((start + npg) * PG, e) - start * PG, 1)
    if blk == 0:
        tl.store(vreq_ptr + v, i)
        tl.store(vstart_ptr + v, start)
        tl.store(vnp_ptr + v, npg)
        tl.store(vlen_ptr + v, ln)
    off = blk * BLOCK + tl.arange(0, BLOCK)
    pages = tl.load(pt_ptr + i.to(tl.int64) * pt_stride + start + off, mask=off < npg, other=0)
    tl.store(vt_ptr + v.to(tl.int64) * vt_stride + off, pages, mask=off < vt_stride)


@triton.jit
def _vch_fold_kernel(q_ptr, q_stride_t, q_stride_h, qf_ptr, vreq_ptr, QL: tl.constexpr, G: tl.constexpr,
                     NKV: tl.constexpr, D: tl.constexpr):
    """qf[v, kvh * QL * G + t * G + h] = q[vreq[v] * QL + t, kvh * G + h]; one program per (chunk, kvh, t)."""
    v = tl.program_id(0)
    kvh = tl.program_id(1) // QL
    t = tl.program_id(1) % QL
    i = tl.load(vreq_ptr + v).to(tl.int64)
    hh = tl.arange(0, G)
    d = tl.arange(0, D)
    src = q_ptr + (i * QL + t) * q_stride_t + (kvh * G + hh)[:, None] * q_stride_h + d[None, :]
    dst = qf_ptr + ((v.to(tl.int64) * NKV + kvh) * QL * G + t * G + hh)[:, None] * D + d[None, :]
    tl.store(dst, tl.load(src))


@triton.jit
def _vch_merge_kernel(q_ptr, q_stride_t, q_stride_h, k_ptr, v_ptr, kv_s_page, kv_s_head, kv_s_slot, pt_ptr,
                      pt_stride, lens_ptr, ov_ptr, lv_ptr, cs_ptr, cn_ptr, out_ptr, out_stride_t, gate_ptr,
                      gate_stride_t, gate_stride_h, scale_ptr, bmm1_log2, bmm2, QL: tl.constexpr, NH: tl.constexpr,
                      G: tl.constexpr, PG: tl.constexpr, D: tl.constexpr, ROWS_V: tl.constexpr, CB: tl.constexpr,
                      GATE_QUANT: tl.constexpr, USE_PDL: tl.constexpr):
    """One program per output row (request, row t, head): chunk partials merged by log2 LSE, plus the FP32
    causal tail. GATE_QUANT: bf16(o) * sigmoid(gate) -> bf16 -> / scale -> clamp +-448 -> FP8, the same
    arithmetic as fused_sigmoid_mul(static_scale=...)."""
    pid = tl.program_id(0)
    i = pid // (QL * NH)
    rem = pid % (QL * NH)
    t = rem // NH
    head = rem % NH
    kvh = head // G
    h = head % G
    rr = kvh * (QL * G) + t * G + h
    d = tl.arange(0, D)
    if USE_PDL:
        # Every input is upstream of the q1 prefix call, which waits on its own predecessor before it can finish.
        tl.extra.cuda.gdc_wait()
    L = tl.load(lens_ptr + i)
    cs = tl.load(cs_ptr + i)
    cn = tl.load(cn_ptr + i)
    j = tl.arange(0, 8)
    tmask = j < t
    pos = L - (QL - 1) + j
    page = tl.load(pt_ptr + i.to(tl.int64) * pt_stride + pos // PG, mask=tmask, other=0).to(tl.int64)
    base = page * kv_s_page + kvh * kv_s_head + (pos % PG).to(tl.int64) * kv_s_slot
    row = i.to(tl.int64) * QL + t
    q = tl.load(q_ptr + row * q_stride_t + head * q_stride_h + d).to(tl.float32)
    kt = tl.load(k_ptr + base[:, None] + d[None, :], mask=tmask[:, None], other=0.0).to(tl.float32)
    sc = tl.where(tmask, tl.sum(kt * q[None, :], axis=1) * bmm1_log2, -float("inf"))
    m = tl.max(sc, axis=0)
    acc = tl.zeros([D], dtype=tl.float32)
    den = 0.0
    for c0 in range(0, cn, CB):
        cidx = c0 + tl.arange(0, CB)
        cm = cidx < cn
        lse = tl.load(lv_ptr + (cs + cidx) * ROWS_V + rr, mask=cm, other=-float("inf"))
        mn = tl.maximum(m, tl.max(lse, axis=0))
        a = tl.exp2(m - mn)
        w = tl.exp2(lse - mn)
        o = tl.load(ov_ptr + ((cs + cidx).to(tl.int64) * ROWS_V + rr)[:, None] * D + d[None, :], mask=cm[:, None],
                    other=0.0).to(tl.float32)
        acc = acc * a + tl.sum(w[:, None] * o, axis=0)
        den = den * a + tl.sum(w, axis=0)
        m = mn
    wt = tl.exp2(sc - m)
    vt = tl.load(v_ptr + base[:, None] + d[None, :], mask=tmask[:, None], other=0.0).to(tl.float32)
    acc += tl.sum(wt[:, None] * vt, axis=0) * bmm2
    den += tl.sum(wt, axis=0)
    o = acc / den
    if GATE_QUANT:
        y = o.to(tl.bfloat16).to(tl.float32)
        g = tl.load(gate_ptr + row * gate_stride_t + head * gate_stride_h + d).to(tl.float32)
        y = (y * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
        y = tl.clamp(y * (1.0 / tl.load(scale_ptr)), -448.0, 448.0)
        tl.store(out_ptr + row * out_stride_t + head * D + d, y.to(out_ptr.dtype.element_ty))
    else:
        tl.store(out_ptr + row * out_stride_t + head * D + d, o.to(out_ptr.dtype.element_ty))


def vch_supported(num_q_heads, num_kv_heads, head_dim, page_size, kv_dtype, q_len):
    g = num_q_heads // max(num_kv_heads, 1)
    return (num_q_heads % num_kv_heads == 0 and q_len == QLEN and g * QLEN <= 128 and g & (g - 1) == 0
            and head_dim == 256 and page_size == 64 and kv_dtype == torch.float8_e4m3fn)


class VchState:
    """Static, graph-safe buffers of one attention backend (one call site shape)."""

    def __init__(self, device, num_q_heads, num_kv_heads, head_dim, page_size, max_bs, max_pages, nv, min_len,
                 workspace, counter, alloc=None):
        self.nh, self.nkv, self.d, self.pg = num_q_heads, num_kv_heads, head_dim, page_size
        self.g = num_q_heads // num_kv_heads
        self.rows_v = num_kv_heads * QLEN * self.g
        self.nv, self.min_len, self.max_pages = nv, min_len, max_pages
        self.block_b = max(16, triton.next_power_of_2(max_bs))
        # Backends of one process run their forwards one at a time, so they can share these (alloc = get_buffer).
        alloc = alloc or (lambda name, make: make())

        def buf(name, shape, dtype, fill=None):
            key = f"vch_{name}_{'x'.join(map(str, shape))}"
            make = (lambda: torch.full(shape, fill, dtype=dtype, device=device)) if fill is not None else (
                lambda: torch.empty(shape, dtype=dtype, device=device))
            return alloc(key, make)
        self.vreq = buf("vreq", (nv,), torch.int32, 0)
        self.vstart = buf("vstart", (nv,), torch.int32, 0)
        self.vnp = buf("vnp", (nv,), torch.int32, 1)
        self.vlen = buf("vlen", (nv,), torch.int32, 1)
        self.cs = buf("cs", (self.block_b,), torch.int32, 0)
        self.cn = buf("cn", (self.block_b,), torch.int32, 0)
        self.vtab = buf("vtab", (nv, max_pages), torch.int32, 0)
        self.qf = buf("qf", (nv, self.rows_v, head_dim), torch.float8_e4m3fn)
        self.ov = buf("ov", (nv, self.rows_v, head_dim), torch.float16)
        self.lv = buf("lv", (nv, self.rows_v), torch.float32)
        self.workspace, self.counter = workspace, counter
        self.use_pdl = torch.cuda.get_device_capability(device)[0] >= 9

    def plan(self, seq_lens: torch.Tensor, page_table: torch.Tensor):
        """Once per forward: chunk plan from the device seq_lens and the per-chunk page tables."""
        bs = seq_lens.shape[0]
        assert bs <= self.block_b and page_table.shape[1] <= self.max_pages
        _vch_plan_kernel[(1,)](seq_lens, bs, self.cs, self.cn, NV=self.nv, BLOCK_B=self.block_b, PG=self.pg,
                               QL=QLEN, MIN_LEN=self.min_len, MAX_PAGES=self.max_pages, num_warps=4)
        _vch_table_kernel[(self.nv, triton.cdiv(self.max_pages, 1024))](
            seq_lens, bs, self.cs, self.cn, page_table, page_table.stride(0), self.vreq, self.vstart, self.vnp,
            self.vlen, self.vtab, self.vtab.stride(0), BLOCK_B=self.block_b, PG=self.pg, QL=QLEN, BLOCK=1024,
            num_warps=4)

    def fold(self, q: torch.Tensor):
        """Standalone fold (the product folds inside the fused attention front instead)."""
        _vch_fold_kernel[(self.nv, self.nkv * QLEN)](q, q.stride(0), q.stride(1), self.qf, self.vreq, QL=QLEN,
                                                     G=self.g, NKV=self.nkv, D=self.d, num_warps=4)

    def prefix(self, kv_cache, bmm1_scale, bmm2_scale, max_seq_len):
        import flashinfer

        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=self.qf, kv_cache=kv_cache, workspace_buffer=self.workspace, block_tables=self.vtab,
            seq_lens=self.vlen, max_seq_len=max_seq_len, bmm1_scale=bmm1_scale, bmm2_scale=bmm2_scale,
            window_left=-1, out=self.ov, out_dtype=torch.float16, q_len_per_req=1, lse=self.lv,
            multi_ctas_kv_counter_buffer=self.counter)

    def merge(self, q, kv_cache, page_table, seq_lens, bmm1_scale, bmm2_scale, out,
              gate: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None):
        k_cache, v_cache = kv_cache
        assert k_cache.stride() == v_cache.stride()
        bs = seq_lens.shape[0]
        gq = gate is not None
        _vch_merge_kernel[(bs * QLEN * self.nh,)](
            q, q.stride(0), q.stride(1), k_cache, v_cache, k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            page_table, page_table.stride(0), seq_lens, self.ov, self.lv, self.cs, self.cn, out, out.stride(0),
            gate if gq else out, gate.stride(0) if gq else 0, gate.stride(1) if gq else 0,
            scale if gq else self.lv, float(bmm1_scale) * math.log2(math.e), float(bmm2_scale), QL=QLEN, NH=self.nh,
            G=self.g, PG=self.pg, D=self.d, ROWS_V=self.rows_v, CB=16, GATE_QUANT=gq, USE_PDL=self.use_pdl,
            num_warps=4, **({"launch_pdl": True} if self.use_pdl else {}))
        return out
