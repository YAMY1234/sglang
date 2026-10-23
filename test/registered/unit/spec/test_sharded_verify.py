"""Target sampling on vocabulary shards (SGLANG_Q35MF_SHARDED_VERIFY).

Greedy: the per-shard keys reduced across shards pick exactly torch.argmax over the
gathered FP32 logits, ties to the smaller token id included, at TP 2 / 4 / 8.
Temperature rows: the Gumbel-max sample has the distribution softmax(logits / T),
whole or split into shards (chi-square against the exact probabilities). The
fallback gather rebuilds the stock gathered FP32 logits exactly."""

import unittest

import torch

from sglang.srt.speculative.sharded_verify import (
    VocabShard,
    _reduce_keys,
    gather_full_logits,
    key_splits,
    keys_to_tokens,
    shard_keys,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

VOCAB = 248320


def _sharded_tokens(full, tp, temperatures=None, seeds=None, split=False):
    width = -(-full.shape[1] // tp)
    splits = key_splits(width) if split else 1
    keys = []
    for rank in range(tp):
        start = rank * width
        part = full[:, start : start + width].contiguous()
        shard = VocabShard(start=start, valid=part.shape[1], vocab_size=full.shape[1])
        keys.append(shard_keys(part, shard, temperatures, seeds, splits).view(-1))
    return keys_to_tokens(torch.cat(keys), tp, splits)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestShardedGreedy(CustomTestCase):
    def test_matches_argmax(self):
        gen = torch.Generator(device="cuda").manual_seed(0)
        for rows in (1, 7, 112, 896):
            full = (
                torch.randn(rows, VOCAB, device="cuda", generator=gen) * 4
            ).bfloat16()
            # ties across shard boundaries, an all-equal row, a row of zeros of both signs
            peak = full.float().max() + 1
            full[0, 5] = full[0, VOCAB - 3] = peak
            if rows > 2:
                full[1].fill_(1.5)
                full[2].zero_()
                full[2, 1::2] = -0.0
            want = torch.argmax(full.float(), dim=-1)
            for tp in (2, 4, 8, 16):
                for split in (False, True):
                    with self.subTest(rows=rows, tp=tp, split=split):
                        got = _sharded_tokens(full, tp, split=split)
                        self.assertTrue(torch.equal(got, want))

    def test_split_temperature_rows_match_unsplit(self):
        # the Gumbel noise is drawn per (seed, token id), so splitting a shard's columns
        # into parts picks the same token as one program over the shard
        gen = torch.Generator(device="cuda").manual_seed(2)
        rows = 56
        full = (torch.randn(rows, VOCAB, device="cuda", generator=gen) * 3).bfloat16()
        temps = torch.rand(rows, device="cuda", generator=gen) + 0.3
        temps[::5] = 0.0
        seeds = torch.randint(0, 2**31 - 1, (rows,), device="cuda", generator=gen)
        for tp in (1, 8, 16):
            with self.subTest(tp=tp):
                want = _sharded_tokens(full, tp, temps, seeds)
                got = _sharded_tokens(full, tp, temps, seeds, split=True)
                self.assertTrue(torch.equal(got, want))

    def test_reduce_strided_rank_rows(self):
        # the multimem exchange's layout: a buffer row per logits row, rank r's parts at
        # column parts * r, the buffer wider than world * parts
        gen = torch.Generator(device="cuda").manual_seed(3)
        rows, world, parts, pad = 7, 8, 16, 32
        keys = torch.randint(
            -(2**62), 2**62, (world, rows, parts), device="cuda", generator=gen
        )
        buf = torch.zeros(rows, world * parts + pad, dtype=torch.int64, device="cuda")
        buf[:, : world * parts] = keys.permute(1, 0, 2).reshape(rows, -1)
        want = keys_to_tokens(keys.reshape(-1), world, parts)
        got = _reduce_keys(buf, rows, world, parts, parts, buf.stride(0))
        self.assertTrue(torch.equal(got, want))
        best = keys.amax(dim=(0, 2))
        self.assertTrue(torch.equal(got, 0xFFFFFFFF - (best & 0xFFFFFFFF)))

    def test_padding_columns_never_win(self):
        full = torch.zeros(3, 1024, device="cuda", dtype=torch.bfloat16)
        full[:, 1000:] = 100.0
        shard = VocabShard(start=0, valid=1024, vocab_size=1000)
        got = keys_to_tokens(shard_keys(full, shard), 1)
        self.assertTrue(
            torch.equal(got, torch.zeros(3, dtype=torch.int64, device="cuda"))
        )


class _FakeGroup:
    """all_gather_into_tensor over shards held on one device, rank order."""

    def __init__(self, shards):
        self.shards, self.world_size = shards, len(shards)

    def all_gather_into_tensor(self, output, input):
        output.copy_(torch.cat([s.reshape(-1) for s in self.shards]))


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestGatherFullLogits(CustomTestCase):
    def test_matches_concat_slice_cast(self):
        # the fallback must rebuild the stock gathered logits exactly: rank-major
        # vocabulary columns, cut to the vocabulary, FP32
        full = torch.randn(7, 1000, device="cuda").bfloat16()
        for tp in (2, 8):
            with self.subTest(tp=tp):
                width = -(-1000 // tp)
                padded = torch.nn.functional.pad(full, (0, width * tp - 1000))
                shards = list(padded.split(width, dim=1))
                shard = VocabShard(start=0, valid=width, vocab_size=1000)
                got = gather_full_logits(shards[0], shard, _FakeGroup(shards))
                self.assertTrue(torch.equal(got, full.float()))


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestShardedGumbel(CustomTestCase):
    def _check(self, logits, temperature, tp, samples=200_000):
        from scipy.stats import chi2

        # one logits row repeated with stride 0; every row draws its own noise
        full = logits.view(1, -1).expand(samples, -1)
        temps = torch.full((samples,), temperature, device="cuda")
        seeds = torch.arange(samples, device="cuda", dtype=torch.int64) * 7919 + 1
        width = -(-logits.numel() // tp)
        keys = []
        for rank in range(tp):
            start = rank * width
            part = full[:, start : start + width]
            shard = VocabShard(
                start=start, valid=part.shape[1], vocab_size=logits.numel()
            )
            keys.append(shard_keys(part, shard, temps, seeds))
        tokens = keys_to_tokens(torch.cat(keys), tp)
        counts = torch.bincount(tokens, minlength=logits.numel()).double()
        probs = torch.softmax(logits.double() / temperature, dim=-1)
        expected = probs * samples
        # pool the tail so every bin expects at least 5 draws
        big = expected >= 5
        obs = torch.cat([counts[big], counts[~big].sum().view(1)])
        exp = torch.cat([expected[big], expected[~big].sum().view(1)])
        stat = ((obs - exp) ** 2 / exp.clamp_min(1e-12)).sum().item()
        p_value = chi2.sf(stat, obs.numel() - 1)
        self.assertGreater(p_value, 1e-4, f"chi2 {stat:.1f} over {obs.numel()} bins")

    def test_distribution(self):
        gen = torch.Generator(device="cuda").manual_seed(1)
        logits = (torch.randn(4096, device="cuda", generator=gen) * 3).bfloat16()
        for temperature in (1.0, 0.6):
            for tp in (1, 8):
                with self.subTest(temperature=temperature, tp=tp):
                    self._check(logits, temperature, tp)


if __name__ == "__main__":
    unittest.main()
