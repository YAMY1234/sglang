"""Read-split context parallelism ("CP-lite") for TRT-LLM MHA decode-kernel attention.

When the TP size exceeds the KV head count, each KV head is replicated on the ``cp``
ranks of its KV group (``cp = tp // kv_heads``) and every rank reads the whole KV of its
head. CP-lite keeps the KV cache as it is (every rank still writes every token) and
splits the read: page ``p`` of a request's context prefix is read by group rank
``p % cp`` only, for every query head of the group. Per call:

1. ``_cp_gather_q``: all-gather of the group's FP8 queries through peer stores.
2. TRT-LLM GenMHA (FlashInfer) over the owned prefix pages, the ``q_len`` rows folded
   with the group heads into one query row (no causal effect on the prefix), with LSE.
3. ``_cp_a2a_merge``: each rank pushes every owner its heads' partials, then merges its
   own heads' partials with the causal ``q_len x q_len`` block of this step's rows.

The exchange buffer is one symmetric-memory allocation on the attention TP group; each
CTA signals its peers with self-resetting CAS flags, so calls stay graph capturable.
"""

from __future__ import annotations

import logging
import weakref
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

PAGE, HEAD_DIM = 64, 256
_MAX_ROWS = 128  # widest call: rows = requests * q_len (TP8 verify at B16: 112)
_KINDS = 2  # flag regions: q gather, partial exchange


@triton.jit
def _flat_tid():
    return tl.inline_asm_elementwise(
        "mov.u32 $0, %tid.x;", "=r", [], dtype=tl.uint32, is_pure=True, pack=1
    )


@triton.jit
def _sync_threads():
    tl.inline_asm_elementwise(
        "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _send_release(addrs):
    # set the peer's flag 0 -> 1 once its previous signal is consumed
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32 %t;
            .reg .pred %p;
            send:
                atom.global.release.sys.cas.b32 %t, [$1], 0, 1;
                setp.eq.u32 %p, %t, 0;
                @!%p bra send;
        }
        """,
        "=r, l",
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_acquire(addrs):
    # consume this rank's flag 1 -> 0
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32 %t;
            .reg .pred %p;
            wait:
                atom.global.sys.acquire.cas.b32 %t, [$1], 1, 0;
                setp.eq.u32 %p, %t, 1;
                @!%p bra wait;
        }
        """,
        "=r, l",
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _group_barrier(peers, me, flags_off, slot, CP: tl.constexpr):
    """CTA-level release / acquire with the same CTA of every group rank."""
    c = tl.arange(0, CP)
    peer = tl.load(peers + c)
    own = tl.load(peers + me)
    send = (peer + flags_off + (slot * CP + me) * 4).to(tl.pointer_type(tl.uint32))
    wait = (own + flags_off + (slot * CP + c) * 4).to(tl.pointer_type(tl.uint32))
    if _flat_tid() < CP:
        _send_release(send)
        _wait_acquire(wait)


@triton.jit
def _cp_tables(
    page_table,
    pt_stride,
    seq_lens,
    tab,
    tab_stride,
    lens,
    ncols,
    rank,
    QL: tl.constexpr,
    CP: tl.constexpr,
    PG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """This rank's compact page table and token count over the prefix [0, L - QL)."""
    b = tl.program_id(0)
    cb = tl.program_id(1)
    L = tl.load(seq_lens + b)
    prefix = tl.maximum(L - QL, 0)
    n_pages = (prefix + PG - 1) // PG
    n_own = tl.maximum((n_pages - rank + CP - 1) // CP, 0)
    j = cb * BLOCK + tl.arange(0, BLOCK)
    m = j < ncols
    pg = tl.load(page_table + b * pt_stride + j * CP + rank, mask=m & (j < n_own), other=0)
    tl.store(tab + b * tab_stride + j, pg, mask=m)
    if cb == 0:
        last_tok = prefix - (n_pages - 1) * PG
        tail = (n_own > 0) & ((n_own - 1) * CP + rank == n_pages - 1)
        n = n_own * PG - tl.where(tail, PG - last_tok, 0)
        # no owned page: one dummy key, dropped by the merge
        tl.store(lens + b, tl.maximum(n, 1))


@triton.jit
def _cp_gather_q(
    q,
    q_stride,
    peers,
    me,
    q_off,
    flags_off,
    CP: tl.constexpr,
    WORDS: tl.constexpr,
):
    """Row i of this rank's queries into row i, slot me of every group rank's buffer."""
    i = tl.program_id(0)
    w = tl.arange(0, WORDS)
    x = tl.load(q + i * q_stride + w)
    for c in tl.static_range(CP):
        base = tl.load(peers + c) + q_off + (i * CP + me) * WORDS * 4
        tl.store(base.to(tl.pointer_type(tl.int32)) + w, x)
    _sync_threads()
    _group_barrier(peers, me, flags_off, i, CP)


@triton.jit
def _cp_a2a_merge(
    po,
    plse,
    q,
    q_stride,
    k_cache,
    v_cache,
    kv_page_stride,
    kv_tok_stride,
    table,
    t_stride,
    seq_lens,
    out,
    out_stride,
    peers,
    me,
    p_off,
    l_off,
    flags_off,
    slot0,
    bmm1,
    bmm2,
    QL: tl.constexpr,
    HQ: tl.constexpr,
    CP: tl.constexpr,
    D: tl.constexpr,
    PG: tl.constexpr,
):
    i = tl.program_id(0)
    h = tl.arange(0, HQ)
    d = tl.arange(0, D)
    hd = h[:, None] * D + d[None, :]
    # push owner c its heads' partial O (bf16) and LSE (fp32, base 2)
    for c in tl.static_range(CP):
        x = tl.load(po + (i * CP * HQ + c * HQ) * D + hd)
        lse = tl.load(plse + i * CP * HQ + c * HQ + h)
        base = tl.load(peers + c)
        dst = (base + p_off).to(tl.pointer_type(tl.bfloat16)) + (i * CP + me) * HQ * D
        tl.store(dst + hd, x)
        ldst = (base + l_off).to(tl.pointer_type(tl.float32)) + (i * CP + me) * HQ
        tl.store(ldst + h, lse)
    _sync_threads()
    _group_barrier(peers, me, flags_off, slot0 + i, CP)
    _sync_threads()
    own = tl.load(peers + me)
    own_p = (own + p_off).to(tl.pointer_type(tl.bfloat16)) + i * CP * HQ * D
    own_l = (own + l_off).to(tl.pointer_type(tl.float32)) + i * CP * HQ
    b = i // QL
    t = i % QL
    L = tl.load(seq_lens + b)
    # the causal block of this step's rows, for this rank's heads
    j = tl.arange(0, 8)
    pos = L - QL + j
    valid = (j <= t) & (pos >= 0)
    pos = tl.maximum(pos, 0)
    page = tl.load(table + b * t_stride + pos // PG, mask=valid, other=0)
    koff = page.to(tl.int64) * kv_page_stride + (pos % PG) * kv_tok_stride
    kk = tl.load(k_cache + koff[:, None] + d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    vv = tl.load(v_cache + koff[:, None] + d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    qv = tl.load(q + i * q_stride + hd).to(tl.float32)
    s = tl.sum(qv[:, None, :] * kk[None, :, :], axis=2) * bmm1
    s = tl.where(valid[None, :], s, float("-inf"))
    md = tl.max(s, axis=1)
    md0 = tl.where(md > float("-inf"), md, 0.0)
    p = tl.exp(s - md0[:, None])
    ld = tl.sum(p, axis=1)
    od = tl.sum(p[:, :, None] * vv[None, :, :], axis=1) / tl.maximum(ld, 1e-30)[:, None] * bmm2
    lse_d = tl.where(ld > 0, md0 + tl.log(ld), float("-inf"))
    # rank c owns a prefix page iff the prefix has more than c pages
    n_pages = (tl.maximum(L - QL, 0) + PG - 1) // PG
    m = lse_d
    for c in tl.static_range(CP):
        lc = tl.load(own_l + c * HQ + h, cache_modifier=".cg") * 0.6931471805599453
        lc = tl.where(n_pages > c, lc, float("-inf"))
        m = tl.maximum(m, lc)
    m0 = tl.where(m > float("-inf"), m, 0.0)
    w = tl.where(lse_d > float("-inf"), tl.exp(lse_d - m0), 0.0)
    acc = od * w[:, None]
    for c in tl.static_range(CP):
        lc = tl.load(own_l + c * HQ + h, cache_modifier=".cg") * 0.6931471805599453
        wc = tl.where(n_pages > c, tl.exp(lc - m0), 0.0)
        oc = tl.load(own_p + c * HQ * D + hd, cache_modifier=".cg").to(tl.float32)
        acc += oc * wc[:, None]
        w += wc
    o = acc / tl.where(w > 0, w, 1.0)[:, None]
    tl.store(out + i * out_stride + hd, o.to(tl.bfloat16))


class CpLite:
    """One process's exchange buffer and scratch, shared by every TRT-LLM MHA backend
    (target and draft) of the process; built collectively on the attention TP group."""

    _instance: Optional["CpLite"] = None

    def __init__(self, group, tp_rank: int, tp_size: int, cp: int, q_heads: int, device):
        import torch.distributed._symmetric_memory as symm_mem

        self.cp, self.hq, self.device = cp, q_heads, torch.device(device)
        self.me = tp_rank % cp
        g0 = tp_rank - self.me
        self.min_bs = envs.SGLANG_TRTL1_ATTN_CP_MIN_BS.get()
        self.q_bytes = _MAX_ROWS * cp * q_heads * HEAD_DIM  # fp8
        self.p_off = self.q_bytes
        self.p_bytes = _MAX_ROWS * cp * q_heads * HEAD_DIM * 2  # bf16
        self.l_off = self.p_off + self.p_bytes
        self.l_bytes = _MAX_ROWS * cp * q_heads * 4
        self.f_off = self.l_off + self.l_bytes
        total = self.f_off + _KINDS * _MAX_ROWS * cp * 4
        if symm_mem.get_backend(self.device) is None:
            symm_mem.set_backend("NCCL")
        self.buf = symm_mem.empty((total,), dtype=torch.uint8, device=self.device)
        self.buf.zero_()
        torch.cuda.synchronize(self.device)
        handle = symm_mem.rendezvous(self.buf, group)
        assert handle.rank == tp_rank, (handle.rank, tp_rank)
        self.peers = torch.tensor(
            [int(p) for p in handle.buffer_ptrs[g0 : g0 + cp]], dtype=torch.int64, device=self.device
        )
        self._handle = handle
        self.q_all = self.buf[: self.q_bytes].view(torch.float8_e4m3fn).view(_MAX_ROWS, cp * q_heads, HEAD_DIM)
        rows = _MAX_ROWS
        self.po = torch.empty(rows, cp * q_heads, HEAD_DIM, dtype=torch.bfloat16, device=self.device)
        self.plse = torch.empty(rows, cp * q_heads, dtype=torch.float32, device=self.device)
        self.counter = torch.zeros(1 << 16, dtype=torch.int32, device=self.device)
        self._logged: set = set()
        # (page table, lengths, q_len) -> [tables, owned lengths, forward, last layer id that used them]
        self._tabs: dict = {}
        torch.distributed.barrier(group)
        logger.info(
            "TRT-L1 CP-lite: tp rank %d, KV group of %d (group rank %d), min batch %d",
            tp_rank, cp, self.me, self.min_bs,
        )

    @classmethod
    def maybe_create(cls, config, device) -> Optional["CpLite"]:
        from sglang.srt.runtime_context import get_parallel

        if cls._instance is not None:
            return cls._instance
        par = get_parallel()
        tp, rank = par.attn_tp_size, par.attn_tp_rank
        kv = config.get_total_num_kv_heads()
        if tp <= kv or tp % kv or config.head_dim != HEAD_DIM:
            return None
        cp = tp // kv
        q_heads = config.num_attention_heads // tp
        if cp not in (2, 4, 8) or q_heads * cp > 16:
            return None
        group = par.attn_tp_group.device_group
        cls._instance = cls(group, rank, tp, cp, q_heads, device)
        return cls._instance

    def _tables(self, page_table, seq_lens, q_len, forward_batch, layer_id):
        """Once per forward: reused only by later layers of the same forward, so every capture and
        every single-layer draft step computes its own."""
        b = seq_lens.shape[0]
        ncols = triton.cdiv(page_table.shape[1], self.cp)
        key = (page_table.data_ptr(), page_table.shape, seq_lens.data_ptr(), b, q_len)
        ent = self._tabs.get(key)
        if ent is not None and ent[2]() is forward_batch and layer_id > ent[3]:
            ent[3] = layer_id
            return ent[0], ent[1], ncols * PAGE
        if len(self._tabs) > 64:
            self._tabs.clear()
        tab = torch.empty(b, ncols, dtype=torch.int32, device=self.device)
        lens = torch.empty(b, dtype=torch.int32, device=self.device)
        self._tabs[key] = [tab, lens, weakref.ref(forward_batch), layer_id]
        _cp_tables[(b, triton.cdiv(ncols, 256))](
            page_table, page_table.stride(0), seq_lens, tab, tab.stride(0), lens, ncols, self.me,
            QL=q_len, CP=self.cp, PG=PAGE, BLOCK=256,
        )
        return tab, lens, ncols * PAGE

    def attend(
        self,
        q: torch.Tensor,
        kv_cache,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        q_len: int,
        bmm1_scale,
        bmm2_scale,
        workspace: torch.Tensor,
        layer,
        forward_batch,
    ) -> Optional[torch.Tensor]:
        """CP-lite attention of this rank's heads: [rows, heads, head_dim] bf16, or None
        where the call is not served (decided by shapes only, so every rank agrees)."""
        import flashinfer

        rows = q.shape[0]
        b = seq_lens.shape[0]
        if (
            q.dtype != torch.float8_e4m3fn
            or q.dim() != 3
            or q.shape[1] != self.hq
            or q.shape[2] != HEAD_DIM
            or layer.tp_k_head_num != 1
            or layer.sliding_window_size not in (None, -1)
            or rows != b * q_len
            or b < self.min_bs
            or rows > _MAX_ROWS
            or not isinstance(bmm1_scale, float)
            or not isinstance(bmm2_scale, float)
        ):
            return None
        if (b, q_len) not in self._logged:
            self._logged.add((b, q_len))
            logger.info("TRT-L1 CP-lite engaged: %d requests x %d rows (layer %d)", b, q_len, layer.layer_id)
        k_cache, v_cache = kv_cache
        q = q if q.is_contiguous() else q.contiguous()
        words = self.hq * HEAD_DIM // 4
        _cp_gather_q[(rows,)](
            q.view(torch.int32), words, self.peers, self.me, 0, self.f_off,
            CP=self.cp, WORDS=words, num_warps=4,
        )
        tab, lens, max_kv = self._tables(page_table, seq_lens, q_len, forward_batch, layer.layer_id)
        gh = self.cp * self.hq
        po = self.po[:rows].view(b, q_len * gh, HEAD_DIM)
        plse = self.plse[:rows].view(b, q_len * gh)
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=self.q_all[:rows].view(b, q_len * gh, HEAD_DIM),
            kv_cache=(k_cache, v_cache),
            workspace_buffer=workspace,
            block_tables=tab,
            seq_lens=lens,
            max_seq_len=max_kv,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            out=po,
            lse=plse,
            q_len_per_req=1,
            multi_ctas_kv_counter_buffer=self.counter,
        )
        out = torch.empty(rows, self.hq, HEAD_DIM, dtype=torch.bfloat16, device=q.device)
        _cp_a2a_merge[(rows,)](
            self.po, self.plse, q, q.stride(0), k_cache, v_cache, k_cache.stride(0), k_cache.stride(2),
            page_table, page_table.stride(0), seq_lens, out, out.stride(0),
            self.peers, self.me, self.p_off, self.l_off, self.f_off, _MAX_ROWS, bmm1_scale, bmm2_scale,
            QL=q_len, HQ=self.hq, CP=self.cp, D=HEAD_DIM, PG=PAGE, num_warps=8,
        )
        return out
