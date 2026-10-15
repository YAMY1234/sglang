"""SGLANG_OPT_DSV41_PREFILL_ASYNC_H2D: the DSV4 sparse-prefill compressed gather runs in
an eager break of the prefill CUDA graph, where the host is ahead of the GPU. Its
per-request lengths upload must not wait for queued GPU work (the pageable copy blocked
the host for the whole queued graph backlog), and the gather must stay unchanged.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    SparsePrefillChunkCache,
)
from sglang.test.test_utils import CustomTestCase

C4 = 4
C_PAGE = 64


def _chunk_cache(device, seq_lens, extend_lens):
    num_qo = sum(extend_lens)
    qsl = [0]
    for n in extend_lens:
        qsl.append(qsl[-1] + n)
    swa = [min(128, s) for s in seq_lens]
    swa_off = [0]
    for n in swa:
        swa_off.append(swa_off[-1] + n)
    i32 = dict(dtype=torch.int32, device=device)
    return SparsePrefillChunkCache(
        num_reqs=len(seq_lens),
        num_qo_tokens=num_qo,
        max_seq_len=max(seq_lens),
        swa_window_size=128,
        swa_page_size=256,
        seq_lens=torch.tensor(seq_lens, **i32),
        query_start_loc=torch.tensor(qsl, **i32),
        query_pos=torch.zeros(num_qo, **i32),
        swa_token_ids=torch.zeros(swa_off[-1], **i32),
        swa_first_pos=torch.zeros(len(seq_lens), **i32),
        swa_gather_lens=torch.tensor(swa, **i32),
        swa_offsets=torch.tensor(swa_off, **i32),
        seq_lens_cpu=list(seq_lens),
    )


def _page_table(device, extend_lens, max_seq_len):
    blocks = max_seq_len // C4 // C_PAGE + 1
    rows = []
    for r, n in enumerate(extend_lens):
        row = torch.arange(blocks, dtype=torch.int32) + 1000 * (r + 1)
        rows.append(row.expand(n, -1))
    return torch.cat(rows).to(device)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestDsv41PrefillAsyncH2D(CustomTestCase):
    seq_lens = [37_000, 9_000, 600]
    extend_lens = [64, 32, 16]

    def _gather(self, flag: bool):
        device = torch.device("cuda")
        with envs.SGLANG_OPT_DSV41_PREFILL_ASYNC_H2D.override(flag):
            cache = _chunk_cache(device, self.seq_lens, self.extend_lens)
            table = _page_table(device, self.extend_lens, max(self.seq_lens))
            return cache.ensure_compressed(C4, table, C_PAGE)

    def test_same_gather_as_pageable_upload(self):
        stock, fast = self._gather(False), self._gather(True)
        for name in ("flat_token_ids", "compressed_base", "swa_base"):
            torch.testing.assert_close(getattr(fast, name), getattr(stock, name))

    def test_host_does_not_wait_for_queued_gpu_work(self):
        device = torch.device("cuda")
        cache = _chunk_cache(device, self.seq_lens, self.extend_lens)
        table = _page_table(device, self.extend_lens, max(self.seq_lens))
        torch.cuda.synchronize()
        done = torch.cuda.Event()
        with envs.SGLANG_OPT_DSV41_PREFILL_ASYNC_H2D.override(True):
            # ~0.5 s of queued GPU work stands in for the replayed graph segments.
            torch.cuda._sleep(1_000_000_000)
            done.record()
            cache.ensure_compressed(C4, table, C_PAGE)
            self.assertFalse(done.query(), "ensure_compressed waited for the GPU")
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
