"""SGLANG_DSV41_FUSED_TAIL: decode tails fed through the DSpark verify steps."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.disaggregation.dsv41_late_tail import LateTailFeed
from sglang.srt.disaggregation.utils import (
    dsv41_encoder_prefix_len,
    dsv41_late_tail_prefix_len,
    dsv41_late_tail_rows,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.late_tail_inputs import (
    LateTailInputs,
    LateTailStepInputs,
)
from sglang.srt.layers.attention.graph_variants import (
    DSV41_EARLY_VERIFY_VARIANTS,
    Dsv41LateTailGraphVariants,
    is_late_tail_variant,
)
from sglang.srt.models.deepseek_common.decoder_swa_bounded_replay import (
    enter_verify_late_rows,
    exit_verify_late_rows,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.logprob_result_processor import (
    SchedulerLogprobResultProcessor,
)
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
    move,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.kernels.ops.speculative.dspark.blockwise_rejection import (
    blockwise_rejection_torch,
    make_rejection_noise,
)
from sglang.kernels.ops.speculative.dspark.dspark_accept import AcceptGreedy
from sglang.srt.speculative.dspark_components.dspark_verify import (
    build_accept_outs,
    force_verify_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import release_only_retirer
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

Mixin = SchedulerDisaggregationDecodeMixin
WIDTH = 6


def _fed_req(prompt_len, seed=0):
    g = torch.Generator().manual_seed(seed)
    prompt = torch.randint(1000, 20000, (prompt_len,), generator=g).tolist()
    start = dsv41_encoder_prefix_len(prompt_len, 128)
    return SimpleNamespace(
        origin_input_ids=prompt,
        dsv41_feed_floor=start,
        dsv41_feed_next=start,
        dsv41_feed_prompt_left=prompt_len - 1 - start,
    )


def _decode_batch(reqs):
    return SimpleNamespace(
        reqs=reqs,
        forward_mode=ForwardMode.DECODE,
        device="cpu",
        dsv41_forced_tokens=None,
        dsv41_swa_floor=None,
    )


def _sched():
    return SimpleNamespace(
        model_worker=SimpleNamespace(verify_num_draft_tokens=WIDTH),
        model_config=SimpleNamespace(sliding_window_size=128),
    )


class TestFeedSchedule(CustomTestCase):
    def test_steps_force_exactly_the_prompt_after_the_start(self):
        req = _fed_req(1000)
        start = req.dsv41_feed_floor
        self.assertEqual(start, 872)
        normal = SimpleNamespace(dsv41_feed_next=None)
        sched = _sched()
        row0 = start  # position of verify row 0 (holding the prompt token there)
        steps = 0
        while req.dsv41_feed_next is not None:
            batch = _decode_batch([normal, req])
            Mixin._set_dsv41_feed_inputs(sched, batch)
            forced, floor = batch.dsv41_forced_tokens, batch.dsv41_swa_floor
            steps += 1
            self.assertEqual(forced.shape, (2, WIDTH))
            self.assertEqual(forced[0].tolist(), [-1] * WIDTH)
            self.assertEqual(floor.tolist(), [0, start])
            # Row j sits at row0 + j and is forced to the prompt token after it.
            for j in range(WIDTH):
                want = req.origin_input_ids[row0 + j + 1] if row0 + j + 1 < 1000 else -2
                self.assertEqual(forced[1, j].item(), want, (steps, j))
            row0 += WIDTH
        # 128 tail tokens, six per step; the last row of the last step sits at the
        # final prompt position and samples the first output token.
        self.assertEqual(steps, 22)
        self.assertEqual(row0 - WIDTH, 998)

        batch = _decode_batch([normal, req])
        Mixin._set_dsv41_feed_inputs(sched, batch)
        self.assertIsNone(batch.dsv41_forced_tokens)
        self.assertIsNone(batch.dsv41_swa_floor)

    def test_first_output_step_stays_in_the_feed(self):
        # Block 3 (width 4) feeds a 129-token tail's 128 prompt tokens in 32 full
        # steps; the next step samples the first output and keeps the served
        # accept (-2 rows), so simulated acceptance cannot replace that token.
        req = _fed_req(1001)
        sched = _sched()
        sched.model_worker.verify_num_draft_tokens = 4
        rows = []
        while req.dsv41_feed_next is not None:
            batch = _decode_batch([req])
            Mixin._set_dsv41_feed_inputs(sched, batch)
            rows.append(batch.dsv41_forced_tokens[0].tolist())
        self.assertEqual(len(rows), 33)
        self.assertEqual(rows[31], req.origin_input_ids[997:1001])
        self.assertEqual(rows[32], [-2] * 4)

    def test_late_layer_floor_is_the_tail_extend_replay_start(self):
        # The tail extend replays the late layers from seq_len - window: the tail
        # start for a 128-token tail, one row later for a 129-token one.
        for prompt_len, floor in ((1000, 872), (1001, 873)):
            req = _fed_req(prompt_len)
            self.assertEqual(req.dsv41_feed_floor, 872)
            batch = _decode_batch([req])
            Mixin._set_dsv41_feed_inputs(_sched(), batch)
            self.assertEqual(batch.dsv41_swa_floor.tolist(), [floor])

    def test_only_decode_steps_get_feed_inputs(self):
        req = _fed_req(1000)
        batch = _decode_batch([req])
        batch.forward_mode = ForwardMode.EXTEND
        Mixin._set_dsv41_feed_inputs(_sched(), batch)
        self.assertIsNone(batch.dsv41_forced_tokens)
        self.assertEqual(req.dsv41_feed_next, 872)


class TestForceVerifyRows(CustomTestCase):
    def test_forced_rows_pick_their_token_and_others_stay_bitwise(self):
        torch.manual_seed(0)
        bs, vocab = 3, 50
        logits = torch.randn(bs * WIDTH + 2, vocab)  # graph padding rows at the end
        original = logits.clone()
        forced = torch.full((bs, WIDTH), -1, dtype=torch.int64)
        forced[1] = torch.tensor([7, 8, 9, 10, 11, 12])
        forced[2, :2] = torch.tensor([3, 49])
        force_verify_rows(logits, forced)

        flat = forced.reshape(-1)
        for row in range(bs * WIDTH + 2):
            if row < flat.numel() and flat[row] >= 0:
                token = flat[row].item()
                self.assertEqual(logits[row].argmax().item(), token)
                self.assertEqual(logits[row, token].item(), 0.0)
                self.assertTrue(torch.isinf(logits[row]).sum() == vocab - 1)
                # Any temperature keeps it one-hot.
                probs = torch.softmax(logits[row] / 0.7, dim=-1)
                self.assertEqual(probs[token].item(), 1.0)
            else:
                self.assertTrue(torch.equal(logits[row], original[row]), row)


class TestFeedKeepsItsWindow(CustomTestCase):
    """Decode retires window rows behind seqlen - 1. A fed tail's seqlen is the
    whole prompt while its KV still stands at the tail, so retiring then frees
    the transferred encoder window its next rows read."""

    def test_window_retires_only_after_the_first_output(self):
        def req(floor, output_ids):
            return SimpleNamespace(
                decode_batch_idx=3,
                kv=SimpleNamespace(holds_kv=True),
                dsv41_feed_floor=floor,
                output_ids=output_ids,
                seqlen=1000 + len(output_ids),
            )

        feeding, fed, normal = req(872, []), req(872, [5]), req(None, [5, 6])
        evict = MagicMock()
        batch = SimpleNamespace(
            tree_cache=SimpleNamespace(
                supports_swa=lambda: True,
                swa_retain_window=128,
                window_sub_pools=lambda: ["swa"],
            ),
            token_to_kv_pool_allocator=MagicMock(),
            forward_mode=ForwardMode.DECODE,
            reqs=[feeding, fed, normal],
            _window_eviction_trigger=lambda interval: ((lambda req: True), 1),
            _evict_swa=evict,
        )
        ScheduleBatch.maybe_evict_swa(batch)
        self.assertEqual(
            [(c.args[0], c.args[1]) for c in evict.call_args_list],
            [(fed, 1000), (normal, 1001)],
        )


class TestForcedDraftRows(CustomTestCase):
    """A sampled accept takes a draft at p / q and turns away one its draft
    distribution gives no mass, as an fp32 softmax does far below the max. The
    feed assumes every forced row is taken, so the draft rows are one-hot too."""

    VOCAB = 64

    def _accept(self, candidates, target, draft_logits):
        bs = candidates.shape[0]
        temps = torch.ones(bs)
        correct_len, bonus, _ = blockwise_rejection_torch(
            candidates=candidates,
            target_logits=target,
            draft_logits=draft_logits,
            target_temperatures=temps,
            draft_temperatures=temps,
            noise=make_rejection_noise(0, "cpu"),
        )
        return correct_len.tolist(), bonus.tolist()

    def test_sampled_accept_takes_every_forced_row(self):
        torch.manual_seed(0)
        gamma = WIDTH - 1
        prompt = [7, 8, 9, 10, 11, 12]
        forced = torch.full((2, WIDTH), -1, dtype=torch.int64)
        forced[1] = torch.tensor(prompt)
        drafts = torch.tensor([[1, 2, 3, 4, 5], prompt[:gamma]])
        candidates = torch.cat([torch.zeros(2, 1, dtype=torch.long), drafts], dim=1)
        target = torch.randn(2 * WIDTH, self.VOCAB)
        force_verify_rows(target, forced)
        draft_logits = torch.randn(2, gamma, self.VOCAB)
        # The Markov draft drafted its own chain: the prompt token at step 2 sits
        # 120 below its max, so the fp32 draft probability is exactly zero.
        draft_logits[1, 2, prompt[2]] = draft_logits[1, 2].max() - 120
        self.assertEqual(torch.softmax(draft_logits[1, 2], -1)[prompt[2]].item(), 0.0)
        before = draft_logits.clone()
        before[1, 2, prompt[2]] = float("-inf")  # the accept's no-mass guard
        self.assertEqual(self._accept(candidates, target, before)[0][1], 2)

        original = draft_logits.clone()
        force_verify_rows(draft_logits.view(2 * gamma, self.VOCAB), forced[:, :gamma])
        self.assertTrue(torch.equal(draft_logits[0], original[0]))
        for j, token in enumerate(prompt[:gamma]):
            self.assertEqual(torch.softmax(draft_logits[1, j], -1)[token].item(), 1.0)
        correct_len, bonus = self._accept(candidates, target, draft_logits)
        self.assertEqual((correct_len[1], bonus[1]), (gamma, prompt[gamma]))

    def test_vocab_shard_keeps_the_token_on_its_shard(self):
        logits = torch.zeros(2, 8)
        force_verify_rows(logits, torch.tensor([[20, 3]]), vocab_start=16)
        self.assertEqual(
            logits[0].tolist(), [float("-inf")] * 4 + [0.0] + [float("-inf")] * 3
        )
        self.assertTrue(torch.isneginf(logits[1]).all())


class TestSimulatedAcceptance(CustomTestCase):
    """SGLANG_SIMULATE_ACC_LEN replaces every request's accept length. A fed tail
    must keep its served accept: _set_dsv41_feed_inputs advances the feed by the
    whole verify width, assuming every forced row is accepted."""

    VOCAB = 64

    def _accept(self, forced, drafts, simulated_len, sampling_info=None):
        bs = forced.shape[0]
        torch.manual_seed(1)
        logits = torch.randn(bs * WIDTH, self.VOCAB)
        original = logits.clone()
        candidates = torch.cat([torch.zeros(bs, 1, dtype=torch.long), drafts], dim=1)
        force_verify_rows(logits, forced)
        correct_len, bonus, cap_trim_lens = AcceptGreedy.execute(
            candidates=candidates, target_logits=logits, verify_num_draft_tokens=WIDTH
        )
        accept = build_accept_outs(
            correct_len=correct_len,
            bonus=bonus,
            cap_trim_lens=cap_trim_lens,
            prefix_lens=torch.full((bs,), 1000),
            draft_tokens=drafts,
            gamma=WIDTH - 1,
            verify_num_draft_tokens=WIDTH,
            tp_sync=SimpleNamespace(sync=lambda site, values: values),
            sync_site=None,
            simulated_correct_len=torch.tensor([simulated_len - 1]),
            target_logits=logits,
            simulate_bonus_sampling_info=sampling_info,
            simulated_bonus_seed=0,
            forced_rows=(forced != -1).any(dim=1),
        )
        return accept, original

    def test_sampled_simulated_bonus_spares_fed_rows(self):
        # SGLANG_SIMULATE_ACC_GREEDY draws the bonus at the simulated row (a Triton
        # sampler; stubbed here to a sentinel). The fed row keeps its served bonus.
        prompt = [7, 8, 9, 10, 11, 12]
        forced = torch.full((2, WIDTH), -1, dtype=torch.int64)
        forced[1] = torch.tensor(prompt)
        drafts = torch.tensor([[1, 2, 3, 4, 5], prompt[:5]])
        sentinel = self.VOCAB - 1
        with patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "sample_simulated_bonus",
            lambda *, greedy_bonus, **_: torch.full_like(greedy_bonus, sentinel),
        ):
            accept, _ = self._accept(forced, drafts, 2, sampling_info=object())
        self.assertEqual(accept.out_tokens[0, :2].tolist(), [1, sentinel])
        self.assertEqual(accept.out_tokens[1].tolist(), prompt)

    def test_first_output_step_keeps_the_served_accept(self):
        # The step after a full last feed step has no prompt rows (-2): its served
        # accept stands, so the first output token is a real sample.
        forced = torch.full((2, WIDTH), -1, dtype=torch.int64)
        forced[1] = -2
        drafts = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        accept, original = self._accept(forced, drafts, simulated_len=4)
        served = original[WIDTH].argmax().item()
        self.assertNotEqual(served, drafts[1, 0].item())
        self.assertEqual(accept.commit_lens.tolist(), [4, 1])
        self.assertEqual(accept.out_tokens[1, 0].item(), served)

    def test_mid_feed_commits_the_whole_forced_width(self):
        prompt = [7, 8, 9, 10, 11, 12]
        forced = torch.full((2, WIDTH), -1, dtype=torch.int64)
        forced[1] = torch.tensor(prompt)
        drafts = torch.tensor([[1, 2, 3, 4, 5], prompt[:5]])
        accept, _ = self._accept(forced, drafts, simulated_len=2)
        # The ordinary request takes the simulated length; the fed one all six.
        self.assertEqual(accept.commit_lens.tolist(), [2, 6])
        self.assertEqual(accept.out_tokens[1].tolist(), prompt)
        self.assertEqual(accept.new_seq_lens.tolist(), [1002, 1006])

    def test_last_feed_step_ends_at_the_served_bonus(self):
        # Two prompt tokens left, then real rows: the draft after the prompt is
        # chosen wrong, so the served accept stops right after the prompt and the
        # first output token is the target's argmax at the last prompt position.
        forced = torch.full((2, WIDTH), -1, dtype=torch.int64)
        forced[1, :2] = torch.tensor([7, 8])
        torch.manual_seed(1)
        first_out = torch.randn(2 * WIDTH, self.VOCAB)[WIDTH + 2].argmax().item()
        drafts = torch.tensor(
            [[1, 2, 3, 4, 5], [7, 8, (first_out + 1) % self.VOCAB, 30, 31]]
        )
        accept, original = self._accept(forced, drafts, simulated_len=5)
        self.assertEqual(original[WIDTH + 2].argmax().item(), first_out)
        self.assertEqual(accept.commit_lens.tolist(), [5, 3])
        self.assertEqual(accept.out_tokens[1, :3].tolist(), [7, 8, first_out])


def _processor():
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        model_config=SimpleNamespace(
            think_end_ids=None, self_opened_think_start_ids=None
        ),
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        kv_weight_version_tracker=None,
        decode_offload_manager=None,
        metrics_collector=None,
        # Main snapshots forward-pass metrics before resolving the tokens.
        metrics_reporter=SimpleNamespace(
            snapshot_spec_decode_metrics=lambda *a, **k: None
        ),
        draft_worker=None,
        model_worker=SimpleNamespace(on_verify_complete_cpu=lambda *a, **k: None),
        logprob_result_processor=SchedulerLogprobResultProcessor,
        output_streamer=SimpleNamespace(),
        beam_coordinator=SimpleNamespace(),
        request_retirer=release_only_retirer(),
        abort_request=lambda *a, **k: None,
        still_in_batch=lambda req: True,
        prefill_launched_reqs=KeyedReqList(ListKind.PREFILL_LAUNCHED),
    )


def _live_req(rid, prompt_left=0):
    sp = SamplingParams(max_new_tokens=256, temperature=0)
    sp.normalize(None)
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=[1, 2, 3],
        sampling_params=sp,
        return_logprob=True,
    )
    req.kv.kv_committed_len = 0
    req.dsv41_feed_prompt_left = prompt_left
    return req


class _Batch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.has_grammar = False
        self.forward_mode = ForwardMode.DECODE
        self.spec_algorithm = SimpleNamespace(
            is_none=lambda: False, is_dflash=lambda: False
        )


class TestFedResults(CustomTestCase):
    def test_prompt_tokens_commit_to_kv_but_never_reach_output(self):
        # Three requests: an ordinary one (3 accepted), one mid-feed (6 prompt
        # tokens), one finishing its feed (2 prompt tokens, then 2 outputs).
        normal, feeding, finishing = (
            _live_req("n"),
            _live_req("f", prompt_left=10),
            _live_req("e", prompt_left=2),
        )
        batch = _Batch([normal, feeding, finishing])
        stride = WIDTH
        tokens = [0] * (3 * stride)
        tokens[0:3] = [101, 102, 103]
        tokens[stride : stride + 6] = [201, 202, 203, 204, 205, 206]
        tokens[2 * stride : 2 * stride + 4] = [301, 302, 303, 304]
        result = GenerationBatchResult(
            next_token_ids=torch.tensor(tokens, dtype=torch.long),
            accept_lens=torch.tensor([3, 6, 4], dtype=torch.long),
            speculative_num_draft_tokens=WIDTH,
        )
        with envs.SGLANG_DSV41_FUSED_TAIL.override(True):
            out = _processor()._resolve_spec_v2_tokens(result, batch, [True] * 3)

        self.assertEqual(out, [[101, 102, 103], [], [303, 304]])
        self.assertEqual([r.kv.kv_committed_len for r in batch.reqs], [3, 6, 4])
        self.assertEqual(feeding.dsv41_feed_prompt_left, 4)
        self.assertEqual(finishing.dsv41_feed_prompt_left, 0)
        self.assertEqual(result.num_fed_only_reqs, 1)
        # Generated tokens = correct drafts + one per request = real outputs.
        self.assertEqual(result.get_num_generated_tokens(3), 3 + 0 + 2)
        # Spec stats skip fed verifies; the normal request counts as usual.
        self.assertEqual([r.spec_verify_ct for r in batch.reqs], [1, 0, 0])

        # Output logprobs skip the fed prompt rows like the tokens do.
        row_logprobs = [[-0.1 * (10 * i + j) for j in range(stride)] for i in range(3)]
        _processor()._apply_decode_logprobs(
            req=finishing,
            i=2,
            batch=batch,
            next_token_id=out[2],
            next_token_logprobs=row_logprobs,
            logits_output=SimpleNamespace(
                next_token_top_logprobs_val=None,
                next_token_top_logprobs_idx=None,
                next_token_token_ids_logprobs_val=None,
                next_token_token_ids_logprobs_idx=None,
            ),
            fed_per_req=result.num_fed_per_req,
        )
        self.assertEqual(finishing.logprob.output_token_logprobs_idx, [303, 304])
        self.assertEqual(
            finishing.logprob.output_token_logprobs_val, row_logprobs[2][2:4]
        )


class _Req(SimpleNamespace):
    # By identity, as Req: the request lists key requests by it.
    __eq__ = object.__eq__
    __hash__ = object.__hash__


def _waiting_req(prompt_len, **fields):
    start = dsv41_encoder_prefix_len(prompt_len, 128)
    values = dict(
        rid=f"r{prompt_len}",
        owner_list=None,
        origin_input_ids=list(range(prompt_len)),
        dsv41_tail_start=start,
        kv=SimpleNamespace(kv_allocated_len=start),
        output_ids=[],
        return_logprob=False,
        grammar=None,
        time_stats=MagicMock(),
        dsv41_feed_floor=None,
        dsv41_feed_next=None,
        dsv41_feed_prompt_left=0,
    )
    values.update(fields)
    return _Req(**values)


def _waiting(reqs):
    queue = IndexedReqList(ListKind.WAITING)
    move(reqs, queue)
    return queue


def _admit_sched(
    waiting, running_size, graph_bs=(1, 2, 4, 8, 16, 32, 64), max_running=64
):
    running = MagicMock()
    running.is_empty.return_value = running_size == 0
    running.batch_size.return_value = running_size
    return SimpleNamespace(
        waiting_queue=_waiting(waiting),
        running_reqs=KeyedReqList(ListKind.DECODING),
        req_to_token_pool=MagicMock(size=64),
        max_running_requests=max_running,
        tp_worker=SimpleNamespace(get_decode_cuda_graph_bs=lambda: list(graph_bs)),
        token_to_kv_pool_allocator=MagicMock(
            page_size=256, available_size=MagicMock(return_value=1 << 20)
        ),
        tree_cache=None,
        model_config=None,
        enable_overlap=False,
        spec_algorithm=None,
        ngram_embedding_manager=MagicMock(),
        future_map=MagicMock(),
    ), running


@patch("sglang.srt.disaggregation.decode.set_time_batch")
@patch("sglang.srt.disaggregation.decode.ScheduleBatch")
class TestAdmission(CustomTestCase):
    def test_eligible_tails_join_the_running_batch(self, schedule_batch, _):
        fed = _waiting_req(1000)
        constrained = _waiting_req(1000, grammar=object())
        short = _waiting_req(100)  # nothing transferred
        sched, running = _admit_sched([constrained, fed, short], running_size=4)
        with envs.SGLANG_DSV41_FUSED_TAIL.override(True):
            out = Mixin._admit_dsv41_fed_tails(sched, running)
        self.assertIs(out, running)
        self.assertEqual(schedule_batch.init_new.call_args.args[0], [fed])
        new_batch = schedule_batch.init_new.return_value
        new_batch.prepare_for_prebuilt.assert_called_once()
        new_batch.process_prebuilt.assert_called_once_with(sched.future_map)
        running.merge_batch.assert_called_once_with(new_batch)
        self.assertEqual(list(sched.waiting_queue), [constrained, short])
        # They run from the admission on, as a prebuilt batch's do.
        self.assertEqual(list(sched.running_reqs), [fed])
        self.assertEqual((fed.prefix_len, fed.extend_end), (0, 872))
        self.assertEqual((fed.dsv41_feed_floor, fed.dsv41_feed_next), (872, 872))
        self.assertEqual(fed.dsv41_feed_prompt_left, 127)
        self.assertIsNone(fed.dsv41_tail_start)

    def test_idle_rank_or_flag_off_keeps_the_extend(self, schedule_batch, _):
        for running_size, flag in ((0, True), (4, False)):
            sched, running = _admit_sched([_waiting_req(1000)], running_size)
            with envs.SGLANG_DSV41_FUSED_TAIL.override(flag):
                self.assertIs(Mixin._admit_dsv41_fed_tails(sched, running), running)
            self.assertEqual(len(sched.waiting_queue), 1)
        schedule_batch.init_new.assert_not_called()

    def _admit(self, running_size, **limits):
        waiting = [_waiting_req(1000) for _ in range(5)]
        sched, running = _admit_sched(waiting, running_size, **limits)
        with envs.SGLANG_DSV41_FUSED_TAIL.override(True):
            Mixin._admit_dsv41_fed_tails(sched, running)
        return waiting, sched

    def test_feeding_stops_at_the_largest_decode_graph(self, schedule_batch, _):
        # Graphs up to bs 8, 5 running: 3 slots stay graphed. The other two
        # tails stay queued for the extend, in arrival order.
        waiting, sched = self._admit(5, graph_bs=(1, 2, 4, 8))
        self.assertEqual(schedule_batch.init_new.call_args.args[0], waiting[:3])
        self.assertEqual(list(sched.waiting_queue), waiting[3:])

    def test_full_graph_batch_feeds_nothing(self, schedule_batch, _):
        for running_size in (8, 9):
            waiting, sched = self._admit(running_size, graph_bs=(1, 2, 4, 8))
            self.assertEqual(list(sched.waiting_queue), waiting)
        schedule_batch.init_new.assert_not_called()

    def test_running_cap_below_the_graph_still_binds(self, schedule_batch, _):
        # max_running_requests 6 < graph bs 8: 5 running leaves 1 slot.
        waiting, _ = self._admit(5, graph_bs=(1, 2, 4, 8), max_running=6)
        self.assertEqual(schedule_batch.init_new.call_args.args[0], waiting[:1])

    def test_no_decode_graphs_leaves_only_the_running_cap(self, schedule_batch, _):
        waiting, sched = self._admit(5, graph_bs=())
        self.assertEqual(schedule_batch.init_new.call_args.args[0], waiting)
        self.assertEqual(list(sched.waiting_queue), [])


class TestLateTailSplit(CustomTestCase):
    def test_prefix_is_unchanged_with_the_switch_off(self):
        for prompt_len, start in ((1000, 872), (1001, 872), (129, 0), (5, 0)):
            self.assertEqual(dsv41_encoder_prefix_len(prompt_len, 128), start)

    def test_late_fed_prefix_stops_one_or_two_tokens_short_and_even(self):
        with envs.SGLANG_DSV41_LATE_TAIL_ROWS.override(64):
            for prompt_len, end in ((1000, 998), (1001, 1000), (130, 128), (129, 128)):
                self.assertEqual(dsv41_encoder_prefix_len(prompt_len, 128), end)
            # Too short to hold a window of late rows: the encoder split as before.
            self.assertEqual(dsv41_encoder_prefix_len(128, 128), 0)
            self.assertEqual(dsv41_encoder_prefix_len(5, 128), 0)
        self.assertIsNone(dsv41_late_tail_prefix_len(128, 128))

    def test_late_rows_are_one_window_floored_at_the_prompt_window(self):
        for prompt_len in (1000, 1001, 129, 130, 65665):
            start, end, floor = dsv41_late_tail_rows(prompt_len, 128)
            self.assertEqual(end - start, 128)
            self.assertEqual(end, dsv41_late_tail_prefix_len(prompt_len, 128))
            self.assertEqual(floor, prompt_len - 128)
            # One or two rows sit below the floor; the verify step covers the rest.
            self.assertIn(floor - start, (1, 2))
            self.assertIn(prompt_len - end, (1, 2))


class TestLateTailFeed(CustomTestCase):
    def _feed(self, rows=64):
        return LateTailFeed(rows_per_step=rows, width=4, window=128)

    def test_budget_takes_contiguous_slots_oldest_tail_first(self):
        feed = self._feed(rows=48)
        a, b = object(), object()
        feed.push(req=a, prompt_len=1001, staging_slot=3)
        feed.push(req=b, prompt_len=1000, staging_slot=7)
        self.assertEqual(feed.pending_rows(), 256)
        seen = {a: [], b: []}
        finished = []
        steps = 0
        while (step := feed.take()) is not None:
            steps += 1
            self.assertLessEqual(len(step.slots), 12)
            for slot in step.slots:
                seen[slot.req].append(slot)
            finished.append(step.finished)
        self.assertEqual(steps, 6)
        self.assertEqual(feed.pending_rows(), 0)
        # a ends in the third step (32 slots = 12 + 12 + 8), and b starts there.
        self.assertEqual(finished, [(), (), (a,), (), (), (b,)])
        for req, start, floor, staging in ((a, 872, 873, 3), (b, 870, 872, 7)):
            slots = seen[req]
            self.assertEqual(
                [s.position for s in slots], list(range(start, start + 128, 4))
            )
            self.assertEqual([s.staging_row for s in slots], list(range(0, 128, 4)))
            self.assertEqual({s.floor for s in slots}, {floor})
            self.assertEqual({s.staging_slot for s in slots}, {staging})

    def test_full_budget_until_the_last_partial_step(self):
        feed = self._feed(rows=64)
        feed.push(req=object(), prompt_len=4000, staging_slot=0)
        feed.push(req=object(), prompt_len=4000, staging_slot=1)
        feed.push(req=object(), prompt_len=4000, staging_slot=2)
        sizes = []
        while (step := feed.take()) is not None:
            sizes.append(len(step.slots))
        self.assertEqual(sizes, [16] * 6)

    def test_budget_must_be_whole_verify_slots(self):
        with self.assertRaises(AssertionError):
            self._feed(rows=50)


def _late_inputs(slots=3, width=4, window=8):
    return LateTailInputs(
        slots=slots,
        width=width,
        window=window,
        hc_mult=2,
        hidden_size=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _late_step(slots, active=True):
    if not slots:
        return LateTailStepInputs(active=active)
    req, pos, floor, row = zip(*slots)
    t = lambda xs: torch.tensor(xs, dtype=torch.int64)
    return LateTailStepInputs(
        active=active,
        req_pool_indices=t(req),
        positions=t(pos),
        floors=t(floor),
        staging_rows=t(row),
    )


class TestLateTailInputs(CustomTestCase):
    def setUp(self):
        self.late = _late_inputs()
        # Pool row r, position p holds KV slot 1000 * r + p.
        self.req_to_token = torch.arange(4)[:, None] * 1000 + torch.arange(64)
        self.late.staging.copy_(
            torch.arange(self.late.staging.shape[0], dtype=torch.float32)[
                :, None, None
            ].expand_as(self.late.staging)
        )

    def _view(self):
        return self.late.build_view(
            req_to_token=self.req_to_token,
            req_pool_indices=torch.tensor([3, 1]),
            seq_lens=torch.tensor([40, 50]),
            out_cache_loc=torch.arange(8, dtype=torch.int64) + 7000,
            positions=torch.tensor([40, 41, 42, 43, 50, 51, 52, 53]),
        )

    def test_pseudo_slots_follow_the_verify_rows(self):
        # Two valid slots of request 2 at positions 20 and 24, staged at rows 8 and 12.
        self.late.set_step(_late_step([(2, 20, 21, 8), (2, 24, 21, 12)]))
        view = self._view()
        self.assertEqual(view.req_pool_indices.tolist(), [3, 1, 2, 2, 0])
        self.assertEqual(view.seq_lens.tolist(), [40, 50, 20, 24, 0])
        self.assertEqual(
            view.positions.tolist()[8:], list(range(20, 28)) + [0, 1, 2, 3]
        )
        # Valid rows write their own KV slots; the empty slot writes padding slot 0.
        self.assertEqual(
            view.out_cache_loc.tolist(),
            list(range(7000, 7008)) + [2000 + p for p in range(20, 28)] + [0] * 4,
        )
        self.assertEqual(
            view.residual_rows[:, 0, 0].tolist()[:8], [float(r) for r in range(8, 16)]
        )
        self.assertEqual(view.residual_rows.shape, (12, 2, 3))

    def test_an_inactive_or_empty_step_clears_every_slot(self):
        self.late.set_step(_late_step([(2, 20, 21, 8)]))
        self.assertTrue(self.late.active)
        self.late.set_step(None)
        self.assertFalse(self.late.active)
        self.assertEqual(self._view().out_cache_loc.tolist()[8:], [0] * 12)
        # Active with no slots of its own: a peer rank has late rows.
        self.late.set_step(_late_step([]))
        self.assertTrue(self.late.active)
        self.assertEqual(self._view().out_cache_loc.tolist()[8:], [0] * 12)

    def test_more_slots_than_the_budget_is_refused(self):
        with self.assertRaises(AssertionError):
            self.late.set_step(_late_step([(1, 0, 0, 0)] * 4))


class TestLateTailGraphVariants(CustomTestCase):
    def _variants(self, inner):
        backend = SimpleNamespace(late_tail=SimpleNamespace(active=False))
        return backend, Dsv41LateTailGraphVariants(inner=inner, backend=backend)

    def test_one_late_graph_on_top_of_the_inner_variants(self):
        inner = SimpleNamespace(
            capture_labels=("candidate_unfiltered", "candidate_filtered"),
            select=lambda fb: "candidate_unfiltered",
        )
        backend, variants = self._variants(inner)
        self.assertEqual(
            variants.capture_labels,
            (
                "candidate_unfiltered",
                "candidate_filtered",
                "candidate_filtered+late_tail",
            ),
        )
        self.assertEqual(variants.select(None), "candidate_unfiltered")
        backend.late_tail.active = True
        # The late graph runs the full candidate path whatever the running lengths.
        self.assertEqual(variants.select(None), "candidate_filtered+late_tail")
        self.assertTrue(is_late_tail_variant(variants.select(None)))
        self.assertIn(variants.select(None), DSV41_EARLY_VERIFY_VARIANTS)

    def test_without_inner_variants(self):
        backend, variants = self._variants(None)
        self.assertEqual(variants.capture_labels, ("dense", "dense+late_tail"))
        self.assertFalse(is_late_tail_variant(variants.select(None)))
        backend.late_tail.active = True
        self.assertEqual(variants.select(None), "dense+late_tail")


class TestVerifyLateRowsSwap(CustomTestCase):
    @patch("sglang.srt.layers.moe.utils.get_moe_a2a_backend")
    @patch("sglang.srt.model_executor.forward_context.get_attn_backend")
    @patch("sglang.srt.models.deepseek_common.decoder_swa_bounded_replay.get_parallel")
    def test_late_layers_see_every_row_then_the_verify_batch_returns(
        self, get_parallel, get_attn_backend, get_moe_a2a_backend
    ):
        get_parallel.return_value = SimpleNamespace(attn_dp_size=4)
        get_moe_a2a_backend.return_value = SimpleNamespace(is_none=lambda: False)
        backend = SimpleNamespace(forward_metadata="verify")
        get_attn_backend.return_value = backend
        fb = SimpleNamespace(
            out_cache_loc=torch.arange(8),
            positions=torch.arange(8),
            req_pool_indices=torch.tensor([1, 2]),
            seq_lens=torch.tensor([10, 20]),
            batch_size=2,
            num_token_non_padded=torch.tensor(6),
            global_num_token_non_padded=torch.tensor(6),
            global_num_token_non_padded_cpu=6,
        )
        before = dict(vars(fb))
        late = SimpleNamespace(
            out_cache_loc=torch.arange(16),
            positions=torch.arange(16),
            req_pool_indices=torch.tensor([1, 2, 3, 3]),
            seq_lens=torch.tensor([10, 20, 30, 34]),
        )
        saved = enter_verify_late_rows(fb, "late", late)
        self.assertEqual(backend.forward_metadata, "late")
        self.assertIs(fb.out_cache_loc, late.out_cache_loc)
        self.assertEqual(fb.batch_size, 4)
        # The pseudo-slots sit after the graph's padded rows: no prefix bound fits.
        self.assertEqual(int(fb.num_token_non_padded), 16)
        self.assertEqual(fb.global_num_token_non_padded_cpu, 16)
        exit_verify_late_rows(fb, saved)
        self.assertEqual(backend.forward_metadata, "verify")
        for key, value in before.items():
            self.assertIs(getattr(fb, key), value, key)

    @patch("sglang.srt.layers.moe.utils.get_moe_a2a_backend")
    @patch("sglang.srt.models.deepseek_common.decoder_swa_bounded_replay.get_parallel")
    def test_dp_gathered_moe_is_refused(self, get_parallel, get_moe_a2a_backend):
        get_parallel.return_value = SimpleNamespace(attn_dp_size=4)
        get_moe_a2a_backend.return_value = SimpleNamespace(is_none=lambda: True)
        with self.assertRaises(AssertionError):
            enter_verify_late_rows(SimpleNamespace(), "late", None)


if __name__ == "__main__":
    unittest.main()
