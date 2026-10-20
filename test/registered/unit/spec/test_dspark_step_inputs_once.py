import os
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_TRITON_OK = torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1"


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@unittest.skipUnless(_TRITON_OK, "needs CUDA or TRITON_INTERPRET=1")
class TestUniformVerifyWindowExtras(unittest.TestCase):
    """The one-launch verify window matches the separate positions add,
    cache-loc gather and anchor copy bit for bit."""

    def _run(self, *, bs, width, page_size, extras):
        from sglang.kernels.ops.speculative.cache_locs import (
            assign_extend_cache_locs_uniform,
        )

        dev = _device()
        g = torch.Generator().manual_seed(bs * 100 + width)
        pool_len, num_pages = 256, 64
        req_to_token = torch.randint(
            0, num_pages * page_size, (bs + 3, pool_len), generator=g
        ).to(dev)
        req_pool_indices = torch.randperm(bs + 3, generator=g)[:bs].to(dev)
        seq_lens = torch.randint(0, pool_len - width, (bs,), generator=g).to(dev)
        full_v2p = torch.randperm(num_pages, generator=g).to(dev)
        swa_v2p = torch.randperm(num_pages, generator=g).to(dev)
        out = torch.empty(bs * width, dtype=torch.int64, device=dev)
        full_out = torch.empty_like(out)
        swa_out = torch.empty_like(out)
        positions = torch.full((bs, width), -7, dtype=torch.int64, device=dev)
        anchor = torch.full((bs, width + 2), -9, dtype=torch.int64, device=dev)
        bonus = torch.randint(0, 1000, (bs,), generator=g).to(dev)
        kw = (
            dict(
                positions_out=positions,
                anchor_src=bonus,
                anchor_out=anchor,
                anchor_stride=anchor.stride(0),
            )
            if extras
            else {}
        )
        assign_extend_cache_locs_uniform[(bs,)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            out,
            pool_len,
            width,
            full_v2p=full_v2p,
            swa_v2p=swa_v2p,
            full_out=full_out,
            swa_out=swa_out,
            PAGE_SIZE=page_size,
            num_v_pages=num_pages,
            num_swa_pages=num_pages,
            **kw,
        )
        return dict(
            out=out,
            full_out=full_out,
            swa_out=swa_out,
            positions=positions,
            anchor=anchor,
            bonus=bonus,
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
        )

    def test_matches_separate_ops(self):
        for bs in (1, 2, 5, 8, 16, 64):
            for width in (6, 7, 9):
                for page_size in (1, 64):
                    with self.subTest(bs=bs, width=width, page_size=page_size):
                        base = self._run(
                            bs=bs, width=width, page_size=page_size, extras=False
                        )
                        got = self._run(
                            bs=bs, width=width, page_size=page_size, extras=True
                        )
                        for k in ("out", "full_out", "swa_out"):
                            self.assertTrue(torch.equal(base[k], got[k]), k)
                        cols = torch.arange(width, device=got["seq_lens"].device)
                        ref_pos = got["seq_lens"].to(torch.int64).unsqueeze(1) + cols
                        self.assertTrue(torch.equal(got["positions"], ref_pos))
                        ref_loc = got["req_to_token"][
                            got["req_pool_indices"].unsqueeze(1), ref_pos
                        ].reshape(-1)
                        self.assertTrue(torch.equal(got["out"], ref_loc))
                        self.assertTrue(torch.equal(got["anchor"][:, 0], got["bonus"]))
                        self.assertTrue(bool((got["anchor"][:, 1:] == -9).all()))


@unittest.skipUnless(_TRITON_OK, "needs CUDA or TRITON_INTERPRET=1")
class TestAllocVerifyWindowOnce(unittest.TestCase):
    """The worker's alloc_verify_window gives the same window with the anchor
    folded in as without it, through the translator's uniform plan."""

    def test_window_with_anchor_matches_plain(self):
        from unittest.mock import patch

        from sglang.kernels.ops.speculative import cache_locs
        from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
        from sglang.srt.speculative.dspark_components.dspark_planner import (
            alloc_verify_window,
        )

        dev = _device()
        bs, width, pool_len = 4, 7, 64
        translator = KVIndexTranslator.__new__(KVIndexTranslator)
        translator.page_size = 1
        translator.req_to_token = torch.randperm(8 * pool_len).view(8, pool_len).to(dev)
        translator._spaces = {}
        runner = type("R", (), {"kv_index_translator": translator})()
        batch = type("B", (), {})()
        batch.seq_lens = torch.tensor([0, 5, 30, 56], device=dev)
        batch.req_pool_indices = torch.tensor([3, 0, 7, 2], device=dev)
        offsets = torch.arange(width, device=dev)
        bonus = torch.tensor([11, 22, 33, 44], device=dev)
        anchor = torch.full((bs, width), -1, dtype=torch.int64, device=dev)
        common = dict(
            batch=batch,
            bs=bs,
            device=dev,
            verify_num_draft_tokens=width,
            block_pos_offsets=offsets,
            model_runner=runner,
            seq_lens_cpu=None,
        )
        with patch.object(cache_locs, "_is_cuda", True):
            plain = alloc_verify_window(**common)
            once = alloc_verify_window(**common, bonus=bonus, anchor_ids=anchor)
        self.assertTrue(torch.equal(plain.positions_2d, once.positions_2d))
        self.assertTrue(torch.equal(plain.verify_cache_loc, once.verify_cache_loc))
        self.assertEqual(once.positions_2d.dtype, plain.positions_2d.dtype)
        self.assertTrue(torch.equal(anchor[:, 0], bonus))
        self.assertTrue(bool((anchor[:, 1:] == -1).all()))


class TestDraftSamplerVerifyIds(unittest.TestCase):
    """The folded draft tail writes [anchor | draft tokens], as the eager concat."""

    def test_tail_writes_verify_ids(self):
        from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
            DsparkDraftSampler,
        )

        bs, gamma, max_bs, hidden = 3, 4, 5, 8
        s = DsparkDraftSampler.__new__(DsparkDraftSampler)
        s.gamma, s.max_bs = gamma, max_bs
        s.sample_from_anchor, s.query_token_num = True, gamma
        s.out = torch.zeros(max_bs * gamma, dtype=torch.int64)
        s.verify_ids = torch.full((max_bs, gamma + 1), -1, dtype=torch.int64)
        s.confidence_out = None
        draft = torch.arange(bs * gamma, dtype=torch.int64).view(bs, gamma) + 100

        def step(model_hidden, anchor):
            s.out[: bs * gamma].copy_(draft.reshape(-1))
            return draft, None

        s._megakernel_steps = lambda b: step
        input_ids = torch.arange(bs * gamma, dtype=torch.int64) * 3
        s(torch.zeros(bs * gamma, hidden), input_ids)
        anchor = input_ids.view(bs, gamma)[:, :1]
        self.assertTrue(
            torch.equal(s.verify_ids[:bs], torch.cat([anchor, draft], dim=1))
        )
        self.assertTrue(bool((s.verify_ids[bs:] == -1).all()))


if __name__ == "__main__":
    unittest.main()
