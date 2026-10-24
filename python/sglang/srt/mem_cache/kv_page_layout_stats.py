"""SGLANG_PLB_LOG_KV_PAGE_STATS = N (PERF-LB): every N target decode forwards, attention TP rank 0 samples the batch's
KV page layout, i.e. the page ids paged attention reads (req_to_token at each page start // page size, as
build_trtllm_mha_page_table builds them), on a side stream and logs how contiguous they are once the copy lands.
No host sync; with N = 0 the model runner never calls in."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_MAX_REQS = 512  # requests sampled per forward (the batch's first ones)
_BLOCK = 1024  # pages per loop step of the kernel


@triton.jit
def _max_combine(a, b):
    return tl.maximum(a, b)


@triton.jit
def _kv_page_runs(req_to_token, stride, req_pool_indices, seq_lens, out, PG: tl.constexpr, BLOCK: tl.constexpr):
    """Program r: request r's KV length, pages, runs of consecutive page ids (page i + 1 == page i + 1) and the sum of
    its run lengths squared, into out[4 r : 4 r + 4] (int64)."""
    r = tl.program_id(0)
    row = req_to_token + tl.load(req_pool_indices + r) * stride
    L = tl.load(seq_lens + r)
    n = (L + PG - 1) // PG
    runs = tl.full([], 0, tl.int64)
    sq = tl.full([], 0, tl.int64)
    start = tl.full([], -1, tl.int64)  # start of the run holding the previous step's last page
    for c in range(0, tl.cdiv(n, BLOCK)):
        i = c * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        m = i < n
        p = tl.load(row + i * PG, mask=m, other=0) // PG
        q = tl.load(row + (i - 1) * PG, mask=m & (i > 0), other=0) // PG
        brk = m & ((i == 0) | (p != q + 1))
        runs += tl.sum(brk.to(tl.int64), axis=0)
        st = tl.maximum(tl.associative_scan(tl.where(brk, i, -1), 0, _max_combine), start)
        # page i sits at offset i - st of its run, and the offsets of a run of length l sum (2 t + 1) to l^2
        sq += tl.sum(tl.where(m, 2 * (i - st) + 1, 0), axis=0)
        start = tl.max(st, axis=0)
    tl.store(out + 4 * r, L)
    tl.store(out + 4 * r + 1, n)
    tl.store(out + 4 * r + 2, runs)
    tl.store(out + 4 * r + 3, sq)


class _KvPageStats:
    def __init__(self, every: int, req_to_token: torch.Tensor, page_size: int, log: bool):
        self.every, self.req_to_token, self.page_size, self.log = every, req_to_token, page_size, log
        device = req_to_token.device
        self.idx = torch.zeros(_MAX_REQS, dtype=torch.int64, device=device)
        self.lens = torch.zeros(_MAX_REQS, dtype=torch.int64, device=device)
        self.out = torch.zeros(4 * _MAX_REQS, dtype=torch.int64, device=device)
        self.host = torch.zeros(4 * _MAX_REQS, dtype=torch.int64, pin_memory=True)
        self.stream = torch.cuda.Stream(device=device)
        self.event = torch.cuda.Event()
        self.ticks = self.sampled_at = self.pending = 0  # pending: requests in the copy in flight
        if log:  # compile now, not at the first sample (one zero-length request, on the side stream)
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                self._launch(1)

    def _launch(self, b: int):
        _kv_page_runs[(b,)](self.req_to_token, self.req_to_token.stride(0), self.idx, self.lens, self.out,
                            PG=self.page_size, BLOCK=_BLOCK)

    def tick(self, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor):
        """One target decode forward, before it launches anything: log a landed sample; every N forwards, sample."""
        if not self.log:
            return
        self.ticks += 1
        if self.pending and self.event.query():
            self._log(self.host[: 4 * self.pending].tolist())
            self.pending = 0
        b = min(req_pool_indices.shape[0], _MAX_REQS)
        if self.pending or self.ticks % self.every or not b:
            return
        # the batch's rows into own buffers in stream order, so later work may rewrite the batch's tensors
        self.idx[:b].copy_(req_pool_indices[:b])
        self.lens[:b].copy_(seq_lens[:b])
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self._launch(b)
            self.host[: 4 * b].copy_(self.out[: 4 * b], non_blocking=True)
            self.event.record()
        self.pending, self.sampled_at = b, self.ticks

    def _log(self, flat: list):
        rows = [flat[i : i + 4] for i in range(0, len(flat), 4)]
        b = len(rows)
        lens, pages, runs = (sorted(r[k] for r in rows) for k in range(3))
        tp, tr, mid = sum(pages), sum(runs), (b - 1) // 2
        logger.info(
            "PLB KV page stats, decode forward %d: B=%d, ctx mean/p50/max %.0f/%d/%d, pages/req mean/p50/max "
            "%.1f/%d/%d, runs/req mean/p50/max %.2f/%d/%d, run length mean %.1f (page-weighted %.1f), "
            "fully contiguous %d/%d",
            self.sampled_at, b, sum(lens) / b, lens[mid], lens[-1], tp / b, pages[mid], pages[-1], tr / b, runs[mid],
            runs[-1], tp / max(tr, 1), sum(r[3] for r in rows) / max(tp, 1), sum(r[1] > 0 and r[2] == 1 for r in rows),
            b,
        )


_sampler: Optional[_KvPageStats] = None


def tick_kv_page_stats(*, forward_mode, req_to_token: torch.Tensor, req_pool_indices: torch.Tensor,
                       seq_lens: torch.Tensor, page_size: int):
    """The target model runner starts a forward (outside any graph); decode / target-verify forwards only."""
    global _sampler
    if not (forward_mode.is_decode() or forward_mode.is_target_verify()):
        return
    if _sampler is None:
        from sglang.srt.runtime_context import get_parallel

        _sampler = _KvPageStats(envs.SGLANG_PLB_LOG_KV_PAGE_STATS.get(), req_to_token, page_size,
                                log=get_parallel().attn_tp_rank == 0)
        logger.info("PERF-LB KV page stats: attention TP rank 0 samples every %d decode forwards", _sampler.every)
    _sampler.tick(req_pool_indices, seq_lens)
