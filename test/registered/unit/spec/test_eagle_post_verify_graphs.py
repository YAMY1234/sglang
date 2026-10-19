"""SGLang's post-verify graphs at batch sizes up to 64: every round's outputs, eager
warm-ups and graph replays alike, must equal bitwise what SGLang's eager sampling
gives on the same inputs, greedy and with forced acceptance, and stay unchanged
through the next round; rounds that run eagerly in between draw from the same
forced-length generator. Requires a GPU."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.speculative.eagle import prepare_verify_commit_outputs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative import eagle_megakernel, spec_utils
from sglang.srt.speculative.eagle_utils import eagle_sample
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

TOKENS = 7
VOCAB = 257
BATCH_SIZES = (1, 3, 16, 64)
# eager warm-ups, the capture of each slot, then replays of both slots
ROUNDS = 2 * eagle_megakernel._NUM_WARMUP + 4


@contextmanager
def _graph_capture(stream):
    # one rank: nothing collective to set up for the capture
    yield SimpleNamespace(stream=stream)


def _round(bs, generator):
    """A chain verify round whose drafts follow the target's argmax for a random
    prefix of each request."""
    logits = torch.randn(bs * TOKENS, VOCAB, device="cuda", generator=generator)
    target = logits.argmax(-1).view(bs, TOKENS)
    drafts = torch.randint(0, VOCAB, (bs, TOKENS), device="cuda", generator=generator)
    keep = torch.randint(0, TOKENS, (bs, 1), device="cuda", generator=generator)
    cols = torch.arange(TOKENS - 1, device="cuda")
    drafts[:, 1:] = torch.where(cols < keep, target[:, :-1], drafts[:, 1:])
    next_token = torch.arange(1, TOKENS + 1, device="cuda")
    next_token[-1] = -1
    verify_input = SimpleNamespace(
        topk=1,
        tree_topk=1,
        draft_probs=None,
        draft_token=drafts.flatten(),
        draft_token_num=TOKENS,
        max_tree_depth=TOKENS,
        retrieve_index=torch.arange(bs * TOKENS, device="cuda").view(bs, TOKENS),
        retrieve_next_token=next_token.expand(bs, TOKENS).contiguous(),
        retrieve_next_sibling=torch.full((bs, TOKENS), -1, device="cuda"),
    )
    batch = SimpleNamespace(
        device="cuda",
        forward_mode=ForwardMode.TARGET_VERIFY,
        seq_lens=torch.randint(16, 512, (bs,), device="cuda", generator=generator),
        req_pool_indices=torch.randperm(bs, device="cuda", generator=generator),
        sampling_info=SimpleNamespace(
            is_all_greedy=True,
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            need_min_p_sampling=False,
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
            sampling_seed=None,
            has_watermark_candidates=False,
            temperatures=None,
            top_ks=None,
            top_ps=None,
        ),
        has_grammar=False,
        hisparse_coordinator=None,
        mamba_track_indices=None,
        tree_cache=SimpleNamespace(page_size=64),
        return_logprob=False,
    )
    return batch, verify_input, logits


def _eager(batch, verify_input, logits, sim_rng):
    predict, accept_lens, accept_index = eagle_sample(
        verify_input,
        batch,
        SimpleNamespace(next_token_logits=logits.clone()),
        simulated_acc_len_rng=sim_rng,
    )
    new_seq_lens, bonus_tokens, _, _ = prepare_verify_commit_outputs(
        predict, accept_index, accept_lens, batch.seq_lens, num_draft_tokens=TOKENS
    )
    return predict, accept_lens, accept_index, new_seq_lens, bonus_tokens


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestPostVerifyGraphsMatchEager(CustomTestCase):
    def setUp(self):
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)

    def _check(self, *, sim_acc_len):
        # the worker's forced lengths, drawn by its graphs and its eager rounds alike;
        # the all-eager reference draws the same stream from its own generator
        worker_rng = eagle_megakernel.forced_length_generator(0)
        reference_rng = eagle_megakernel.forced_length_generator(0)
        # SGLang's sampling in the graphs: neither model's megakernel offers an entry
        megakernel = eagle_megakernel.EagleMegakernel(
            target_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    model=torch.nn.Module(), watermark_state=None
                ),
                random_seed=0,
                device="cuda",
            ),
            draft_worker=SimpleNamespace(
                draft_runner=SimpleNamespace(model=torch.nn.Module(), model_config=None)
            ),
            simulated_acc_len_rng=worker_rng,
        )
        generator = torch.Generator(device="cuda").manual_seed(0)
        stream = torch.cuda.Stream()
        with (
            torch.cuda.stream(stream),
            patch.object(spec_utils, "SIMULATE_ACC_LEN", sim_acc_len),
            patch.object(spec_utils, "SIMULATE_ACC_METHOD", "match-expected"),
            patch.object(spec_utils, "SIMULATE_ACC_TOKEN_MODE", "real-draft-token"),
            patch.object(spec_utils, "commit_mamba_states_after_verify"),
            patch(
                "sglang.srt.distributed.parallel_state.graph_capture", _graph_capture
            ),
        ):
            for bs in BATCH_SIZES:
                # the verify graph's logits buffer: one per batch size, refilled
                logits_buf = torch.empty(bs * TOKENS, VOCAB, device="cuda")
                previous = None
                # every third round runs eagerly, as when its verify ran without a graph
                for i in range(ROUNDS * 3 // 2):
                    batch, verify_input, logits = _round(bs, generator)
                    logits_buf.copy_(logits)
                    want = _eager(batch, verify_input, logits, reference_rng)
                    graphed = i % 3 != 2
                    got = megakernel.verify_post(
                        batch=batch,
                        verify_input=verify_input,
                        logits_output=SimpleNamespace(next_token_logits=logits_buf),
                        can_run_cuda_graph=graphed,
                    )
                    megakernel.join()
                    if graphed:
                        self.assertIsNotNone(got)
                    else:
                        self.assertIsNone(got)
                        got = _eager(batch, verify_input, logits, worker_rng)
                    rows = want[2].clamp(min=0).long()
                    with self.subTest(bs=bs, round=i):
                        self.assertTrue(torch.equal(got[0][rows], want[0][rows]))
                        for name, g, w in zip(
                            ("accept_lens", "accept_index", "new_seq_lens", "bonus"),
                            got[1:5],
                            want[1:5],
                        ):
                            self.assertTrue(torch.equal(g, w), name)
                        # the host may still read a round's outputs in the next round
                        if previous is not None:
                            for kept, snapshot in zip(*previous):
                                self.assertTrue(torch.equal(kept, snapshot))
                    previous = (got[:5], [t.clone() for t in got[:5]])

    def test_greedy(self):
        self._check(sim_acc_len=-1.0)

    def test_forced_acceptance(self):
        self._check(sim_acc_len=4.8)


if __name__ == "__main__":
    unittest.main()
