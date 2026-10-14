"""The folded DSpark draft sampler takes the one-kernel sharded Markov chain
(SGLANG_OPT_DSV41_DSPARK_SHARDED_MARKOV) for the batches it supports and the per-step
path otherwise; the chain's noise offset advances once per sampled block, as the
per-step path does."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.dspark_components import dspark_draft_sampler as ds
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GAMMA, MAX_BS, VOCAB, HIDDEN = 5, 8, 64, 4


class _Sharded:
    max_rows = MAX_BS

    def __init__(self, supported):
        self.supported, self.calls, self.noise = supported, [], 0

    def fork_noise(self, **kwargs):
        self.noise += 1

    def join_noise(self):
        pass

    def supports(self, base_logits):
        return self.supported

    def sample(self, *, draft_tokens, logits_cache, **kwargs):
        draft_tokens.fill_(7)
        self.calls.append(logits_cache)


def _sampler(sharded, greedy_draft):
    sampler = ds.DsparkDraftSampler.__new__(ds.DsparkDraftSampler)
    sampler.model = SimpleNamespace(
        compute_base_logits=lambda h: (torch.zeros(h.shape[0], VOCAB), None),
        lm_head=SimpleNamespace(org_vocab_size=VOCAB),
        markov_head=SimpleNamespace(vocab_shard=None, step_logits_dtype=lambda d: torch.float32),
    )
    sampler.markov_head = sampler.model.markov_head
    sampler.gamma = sampler.query_token_num = GAMMA
    sampler.sample_from_anchor = True
    sampler.max_bs = MAX_BS
    sampler.out = torch.zeros(MAX_BS * GAMMA, dtype=torch.int64)
    sampler.confidence_out = None
    sampler.greedy_draft = greedy_draft
    sampler._noise = SimpleNamespace(key=1, offset=torch.zeros(1, dtype=torch.int64))
    sampler.params = SimpleNamespace(step_temperatures=torch.zeros(MAX_BS))
    sampler.step_logits = None
    sampler._sharded = sharded
    sampler._tp_sync = None
    return sampler


@pytest.mark.parametrize("greedy_draft", [False, True])
def test_supported_batches_take_the_chain_kernel(monkeypatch, greedy_draft):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(ds, "draft_markov_block", lambda **kw: pytest.fail("per-step path"))
    sharded = _Sharded(supported=True)
    sampler = _sampler(sharded, greedy_draft)
    sampler(torch.zeros(GAMMA, HIDDEN), torch.zeros(GAMMA, dtype=torch.int64))
    assert sampler.out[:GAMMA].eq(7).all() and len(sharded.calls) == 1
    # A greedy draft keeps no step logits and draws no noise.
    assert (sharded.calls[0] is None) == greedy_draft
    assert sharded.noise == int(sampler._noise.offset) == (0 if greedy_draft else 1)


def test_unsupported_batches_keep_the_per_step_path(monkeypatch):
    calls = []

    def per_step(**kwargs):
        calls.append(kwargs)
        return torch.full((1, GAMMA), 3, dtype=torch.int64), None

    monkeypatch.setattr(ds, "draft_markov_block", per_step)
    sharded = _Sharded(supported=False)
    sampler = _sampler(sharded, greedy_draft=True)
    sampler(torch.zeros(GAMMA, HIDDEN), torch.zeros(GAMMA, dtype=torch.int64))
    assert len(calls) == 1 and not sharded.calls
    assert sampler.out[:GAMMA].eq(3).all()
