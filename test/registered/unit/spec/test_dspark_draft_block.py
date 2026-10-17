import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.dspark import (
    GatedMarkovHead,
    Nemotron35VanillaMarkov,
    RNNHead,
    VanillaMarkov,
)
from sglang.srt.speculative.dspark_components import dspark_draft_sampler as ds
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftNoise,
    sample_draft_block,
)
from sglang.srt.speculative.dspark_components.dspark_sharded_markov import (
    narrow_row_limit,
    unsupported_reason,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB, RANK, HIDDEN, BS, GAMMA = 211, 8, 16, 4, 5


def _draft(head, sampling_info, offset, greedy_draft=False):
    torch.manual_seed(0)
    base = torch.randn(BS, GAMMA, VOCAB).to(torch.bfloat16)
    hidden = torch.randn(BS, GAMMA, HIDDEN).to(torch.bfloat16)
    anchor = torch.randint(0, VOCAB, (BS,))
    block = sample_draft_block(
        base_logits=base,
        anchor_tokens=anchor,
        draft_hidden=hidden,
        sampling_info=sampling_info,
        markov_head=head,
        noise=DraftNoise(key=3, offset=offset),
        device="cpu",
        tp_sync=SimpleNamespace(sync=lambda site, tokens: tokens),
        greedy_draft=greedy_draft,
    )
    return block, base, hidden, anchor


def _step_logits_along(head, base, hidden, anchor, tokens):
    state = head.init_state(hidden)
    prev = anchor
    rows = []
    for step in range(GAMMA):
        bias, state = head.step_bias(prev, hidden_states=hidden[:, step], state=state)
        rows.append(base[:, step] + bias)
        prev = tokens[:, step]
    return torch.stack(rows, dim=1)


class TestSampleDraftBlock(CustomTestCase):
    def test_greedy_rows_take_the_argmax_and_keep_their_logits(self):
        for head in (
            VanillaMarkov(vocab_size=VOCAB, markov_rank=RANK),
            RNNHead(vocab_size=VOCAB, markov_rank=RANK, hidden_size=HIDDEN),
        ):
            head = head.to(torch.bfloat16)
            # top_k <= 1 marks a greedy request; rows 0 and 2 are greedy.
            sampling_info = SimpleNamespace(
                is_all_greedy=False,
                temperatures=torch.tensor([[0.7], [1.0], [1.0], [1.3]]),
                top_ks=torch.tensor([1, 50, 1, 50]),
            )
            offset = torch.zeros(1, dtype=torch.int64)
            block, base, hidden, anchor = _draft(head, sampling_info, offset)
            logits = _step_logits_along(head, base, hidden, anchor, block.draft_tokens)
            self.assertEqual(block.corrected_logits.dtype, logits.dtype)
            self.assertTrue(torch.equal(block.corrected_logits, logits))
            greedy = torch.tensor([True, False, True, False])
            self.assertTrue(
                torch.equal(
                    block.draft_tokens[greedy], logits[greedy].float().argmax(-1)
                )
            )
            # The sampled rows drew noise: they leave the argmax somewhere.
            self.assertFalse(
                torch.equal(
                    block.draft_tokens[~greedy], logits[~greedy].float().argmax(-1)
                )
            )
            self.assertEqual(block.greedy_mask.tolist(), greedy.tolist())
            self.assertEqual(offset.item(), 1)

    def test_greedy_draft_takes_every_argmax_and_keeps_no_logits(self):
        head = VanillaMarkov(vocab_size=VOCAB, markov_rank=RANK).to(torch.bfloat16)
        # Sampling requests too: a greedy draft ignores their temperatures.
        sampling_info = SimpleNamespace(
            is_all_greedy=False,
            temperatures=torch.tensor([[0.7], [1.0], [1.0], [1.3]]),
            top_ks=torch.tensor([50, 50, 1, 50]),
        )
        offset = torch.zeros(1, dtype=torch.int64)
        block, base, hidden, anchor = _draft(
            head, sampling_info, offset, greedy_draft=True
        )
        logits = _step_logits_along(head, base, hidden, anchor, block.draft_tokens)
        self.assertTrue(torch.equal(block.draft_tokens, logits.float().argmax(-1)))
        self.assertIsNone(block.corrected_logits)
        self.assertTrue(block.greedy_draft)


class TestFusedGreedyEligibility(CustomTestCase):
    def test_only_the_plain_w2_gemv_head_fuses(self):
        # A head that changes the bias must keep the shared op.
        heads = (
            (VanillaMarkov(vocab_size=VOCAB, markov_rank=RANK), True),
            (
                Nemotron35VanillaMarkov(
                    vocab_size=VOCAB, markov_rank=RANK, quant_config=None, prefix=""
                ),
                False,
            ),
            (
                GatedMarkovHead(vocab_size=VOCAB, markov_rank=RANK, hidden_size=HIDDEN),
                False,
            ),
            (RNNHead(vocab_size=VOCAB, markov_rank=RANK, hidden_size=HIDDEN), False),
        )
        for head, fuses in heads:
            with self.subTest(head=type(head).__name__):
                self.assertIs(head.greedy_w2_weight() is not None, fuses)


class _FakeShardedChain:
    max_rows = BS

    def __init__(self, supported: bool):
        self.supported, self.logits_caches, self.noise_draws = supported, [], 0

    def fork_noise(self, **kwargs):
        self.noise_draws += 1

    def join_noise(self):
        pass

    def supports(self, base_logits):
        return self.supported

    def sample(self, *, draft_tokens, logits_cache, **kwargs):
        draft_tokens.fill_(7)
        self.logits_caches.append(logits_cache)


def _folded_sampler(*, greedy_draft, build_sharded_sampler):
    head = SimpleNamespace(
        vocab_shard=None,
        step_logits_dtype=lambda dtype: torch.float32,
        build_sharded_sampler=build_sharded_sampler,
    )
    model = SimpleNamespace(
        markov_head=head,
        sample_from_anchor=True,
        confidence_head=None,
        compute_base_logits=lambda h: (torch.zeros(h.shape[0], VOCAB), None),
        lm_head=SimpleNamespace(org_vocab_size=VOCAB),
    )
    return ds.DsparkDraftSampler(
        model=model,
        gamma=GAMMA,
        max_bs=BS,
        device="cpu",
        noise=DraftNoise(key=3, offset=torch.zeros(1, dtype=torch.int64)),
        tp_sync=None,
        greedy_draft=greedy_draft,
    )


class TestShardedMarkovDispatch(CustomTestCase):
    def test_supported_batches_take_the_chain_kernel(self):
        for greedy_draft in (False, True):
            with (
                self.subTest(greedy_draft=greedy_draft),
                patch.object(
                    ds,
                    "draft_markov_block",
                    side_effect=AssertionError("per-step path"),
                ),
                patch.object(torch.cuda, "is_current_stream_capturing", lambda: False),
            ):
                chain = _FakeShardedChain(supported=True)
                sampler = _folded_sampler(
                    greedy_draft=greedy_draft,
                    build_sharded_sampler=lambda **kwargs: chain,
                )
                sampler(
                    torch.zeros(GAMMA, HIDDEN), torch.zeros(GAMMA, dtype=torch.int64)
                )
                self.assertTrue(sampler.out[:GAMMA].eq(7).all())
                # A greedy draft keeps no step logits and draws no noise; a sampled
                # block draws once and advances the offset once, as the per-step path.
                self.assertEqual(chain.logits_caches[0] is None, greedy_draft)
                draws = 0 if greedy_draft else 1
                self.assertEqual(chain.noise_draws, draws)
                self.assertEqual(int(sampler._noise.offset), draws)
                if not greedy_draft:
                    # Temperature-0 rows get no logits from the chain; a mixed
                    # accept still reads them.
                    self.assertFalse(sampler.step_logits.any())

    def test_unsupported_batches_keep_the_per_step_path(self):
        per_step = (torch.full((1, GAMMA), 3, dtype=torch.int64), None)
        chain = _FakeShardedChain(supported=False)
        sampler = _folded_sampler(
            greedy_draft=True, build_sharded_sampler=lambda **kwargs: chain
        )
        with patch.object(ds, "draft_markov_block", return_value=per_step) as stock:
            sampler(torch.zeros(GAMMA, HIDDEN), torch.zeros(GAMMA, dtype=torch.int64))
        self.assertEqual(stock.call_count, 1)
        self.assertEqual(chain.logits_caches, [])
        self.assertTrue(sampler.out[:GAMMA].eq(3).all())

    def test_only_a_sampled_draft_asks_for_logit_staging(self):
        """A greedy draft's chain is built without staging buffers; a sampled draft
        built that way would write its logits past the peer buffer."""
        for greedy_draft in (False, True):
            built = []
            _folded_sampler(
                greedy_draft=greedy_draft,
                build_sharded_sampler=lambda **kwargs: built.append(kwargs),
            )
            self.assertEqual(built[0]["keeps_logits"], not greedy_draft)


class TestShardedMarkovEligibility(CustomTestCase):
    def _reason(self, *, world=4, vocab=129_280, rank_dim=256, capability=(10, 3)):
        weight = torch.empty(vocab, rank_dim, dtype=torch.bfloat16)
        return unsupported_reason(
            world=world,
            vocab_size=vocab,
            num_steps=GAMMA,
            embed_weight=weight,
            head_weight=weight,
            cache_dtype=torch.float32,
            capability=capability,
        )

    def test_dsv41_shapes_run_at_every_supported_tp_size_on_sm100_and_sm103(self):
        for world in (1, 2, 4, 8):
            for capability in ((10, 0), (10, 3)):
                self.assertIsNone(self._reason(world=world, capability=capability))

    def test_refusals(self):
        # The candidate key keeps a 17-bit index; shards are whole 16-row tiles.
        for kwargs in (
            dict(world=3, vocab=129_264),
            dict(capability=(9, 0)),
            dict(capability=(12, 0)),
            dict(vocab=1 << 17),
            dict(vocab=129_280 + 16),
            dict(rank_dim=128),
        ):
            with self.subTest(**kwargs):
                self.assertIsNotNone(self._reason(**kwargs))

    def test_narrow_rows_give_each_row_and_rank_one_lane(self):
        self.assertEqual(narrow_row_limit(world=4, max_bs=64), 8)
        self.assertEqual(narrow_row_limit(world=8, max_bs=64), 4)
        self.assertEqual(narrow_row_limit(world=4, max_bs=2), 2)


if __name__ == "__main__":
    unittest.main()
