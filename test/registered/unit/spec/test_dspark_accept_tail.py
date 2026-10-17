import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers import logits_processor
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.dspark_components.dspark_local_argmax import (
    unsupported_reason,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    DsparkVerifyEpilogue,
    TargetVerifyExecutor,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BS, STRIDE, VOCAB = 3, 5, 50


class _RecordingTpSync:
    def __init__(self):
        self.calls = []

    def sync(self, site, values):
        self.calls.append((site, values.clone()))
        return values


def _greedy_inputs():
    torch.manual_seed(0)
    candidates = torch.randint(0, VOCAB, (BS, STRIDE))
    target_predict = torch.randint(0, VOCAB, (BS, STRIDE))
    # Rows match the drafts for 0, 2 and all positions.
    target_predict[1, :2] = candidates[1, 1:3]
    target_predict[2, :-1] = candidates[2, 1:]
    logits = torch.randn(BS * STRIDE, VOCAB)
    logits.scatter_(1, target_predict.view(-1, 1), 100.0)
    prefix_lens = torch.tensor([7, 11, 13], dtype=torch.int64)
    return candidates, logits, prefix_lens


class TestAcceptTailParity(CustomTestCase):
    def _run(self, simulated_correct_len, verify_lens):
        candidates, logits, prefix_lens = _greedy_inputs()
        draft_tokens = candidates[:, 1:].contiguous()

        graph_sync = _RecordingTpSync()
        epilogue = DsparkVerifyEpilogue(
            max_bs=BS + 1,
            verify_num_draft_tokens=STRIDE,
            device="cpu",
            tp_sync=graph_sync,
            simulated_correct_len=simulated_correct_len,
        )
        epilogue._accept(
            candidates=candidates,
            logits=logits,
            draft_tokens=draft_tokens,
            seq_lens=prefix_lens,
            cutoff_verify_lens=verify_lens,
        )
        graph = epilogue.read_accept(BS)

        eager_sync = _RecordingTpSync()
        target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_text_config=SimpleNamespace(model_type="test")
                )
            )
        )
        executor = TargetVerifyExecutor(
            target_worker=target_worker,
            gamma=STRIDE - 1,
            verify_num_draft_tokens=STRIDE,
            model_runner=None,
            kv_injector=None,
            tp_sync=eager_sync,
            simulated_correct_len=simulated_correct_len,
        )
        eager = executor.accept_and_finalize(
            folded_accept=False,
            bs=BS,
            verify_ids_2d=candidates,
            target_logits=logits,
            draft_block=DraftBlockResult(
                draft_tokens=draft_tokens,
                corrected_logits=None,
                greedy_mask=torch.ones(BS, dtype=torch.bool),
                temperatures=torch.ones(BS),
                greedy_draft=True,
            ),
            sampling_info=None,
            draft_input=None,
            layout=None
            if verify_lens is None
            else SimpleNamespace(verify_lens=verify_lens),
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
        )

        for field in eager.__struct_fields__:
            self.assertTrue(
                torch.equal(
                    getattr(graph, field).cpu().long(),
                    getattr(eager, field).cpu().long(),
                ),
                field,
            )
        self.assertEqual(
            [site for site, _ in graph_sync.calls],
            [SpecTpSyncSite.DSPARK_ACCEPT_GRAPH] * 3,
        )
        self.assertEqual(
            [site for site, _ in eager_sync.calls],
            [SpecTpSyncSite.DSPARK_ACCEPT_GREEDY] * 3,
        )
        for (_, graph_values), (_, eager_values) in zip(
            graph_sync.calls, eager_sync.calls
        ):
            self.assertTrue(torch.equal(graph_values, eager_values))

    def test_graph_accept_matches_eager_greedy(self):
        # Row 1 has a 2-token budget, so its 2 correct drafts are capped to 1.
        ragged = torch.tensor([STRIDE, 2, 3], dtype=torch.int64)
        for simulated in (None, 1):
            for verify_lens in (None, ragged):
                with self.subTest(simulated=simulated, ragged=verify_lens is not None):
                    simulated_correct_len = (
                        None if simulated is None else torch.tensor([simulated])
                    )
                    self._run(simulated_correct_len, verify_lens)


class _FakeSink:
    """A LocalArgmaxSink that forked with ``ids``; counts joins."""

    def __init__(self, ids):
        self.ids = ids
        self.joins = 0

    @property
    def pending(self) -> bool:
        return self.ids is not None

    def supports(self, local_logits, *, vocab_size) -> bool:
        return True

    def join(self) -> None:
        self.joins += 1
        self.ids = None


class TestStaticGreedyArgmaxIds(CustomTestCase):
    def setUp(self):
        self.addCleanup(logits_processor.set_local_argmax_sink, None)

    def _capture(self, *, ids, logits, draft_worker=False):
        candidates, _, prefix_lens = _greedy_inputs()
        sink = _FakeSink(ids)
        epilogue = DsparkVerifyEpilogue(
            max_bs=BS,
            verify_num_draft_tokens=STRIDE,
            device="cpu",
            tp_sync=_RecordingTpSync(),
            local_argmax=sink,
        )
        runner = SimpleNamespace(
            model_runner=SimpleNamespace(is_draft_worker=draft_worker),
            ragged_verify_mode=None,
        )
        out = LogitsProcessorOutput(
            next_token_logits=logits, hidden_states=torch.zeros(BS * STRIDE, 4)
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=BS,
            input_ids=candidates.reshape(-1),
            seq_lens=prefix_lens,
        )
        epilogue.capture_hook(runner, out, forward_batch, BS * STRIDE)
        return epilogue.read_accept(BS), sink

    def test_accept_reads_the_exchange_ids(self):
        """With the gather skipped the full logits are stale; the accept must follow
        the exchange ids when they cover every verify row, the logits otherwise, and
        the capture tail must join the sink's stream either way."""
        _, logits, _ = _greedy_inputs()
        want, _ = self._capture(ids=None, logits=logits)
        ids = logits.argmax(dim=1)
        stale = torch.randn_like(logits)
        got, sink = self._capture(ids=ids, logits=stale)
        self.assertEqual(sink.joins, 1)
        for field in want.__struct_fields__:
            self.assertTrue(torch.equal(getattr(got, field), getattr(want, field)))
        # Ids of another row count (an unsupported capture) leave the logits in charge.
        got, sink = self._capture(ids=ids[:-1], logits=logits)
        self.assertEqual(sink.joins, 1)
        self.assertTrue(torch.equal(got.out_tokens, want.out_tokens))
        # A capture tail that returns early still joins.
        _, sink = self._capture(ids=ids, logits=stale, draft_worker=True)
        self.assertEqual(sink.joins, 1)


class TestLocalArgmaxEligibility(CustomTestCase):
    def test_unsupported_configurations(self):
        for capability in ((10, 0), (10, 3)):
            for world in (2, 4, 8):
                self.assertIsNone(
                    unsupported_reason(
                        world=world, vocab_size=129_280, capability=capability
                    )
                )
        # The key's 17-bit index field holds 2**17 columns.
        self.assertIsNone(
            unsupported_reason(world=4, vocab_size=1 << 17, capability=(10, 3))
        )
        for world, vocab_size, capability in (
            (1, 129_280, (10, 3)),
            (16, 129_280, (10, 3)),
            (4, 129_280, (9, 0)),
            (4, 129_280, (12, 0)),
            (4, (1 << 17) + 1, (10, 3)),
        ):
            self.assertIsNotNone(
                unsupported_reason(
                    world=world, vocab_size=vocab_size, capability=capability
                ),
                (world, vocab_size, capability),
            )

    def test_sink_serves_only_target_verify_capture(self):
        """The sink forks only inside a target-verify graph capture over a plain
        vocab-parallel TP gather; eager warm-up and other LM-head layouts keep the
        stock gather."""
        processor = LogitsProcessor.__new__(LogitsProcessor)
        torch.nn.Module.__init__(processor)
        processor.vocab_size = 64
        processor.do_tensor_parallel_all_gather = True
        processor.do_tensor_parallel_all_gather_dp_attn = False
        processor.use_attn_tp_group = False
        processor.use_tp_lm_head_all_to_all = False
        processor.final_logit_softcapping = None
        sink = _FakeSink(None)
        logits_processor.set_local_argmax_sink(sink)
        self.addCleanup(logits_processor.set_local_argmax_sink, None)
        logits = torch.zeros(4, 16)

        def selected(mode=ForwardMode.TARGET_VERIFY, capturing=True, **fields):
            for name, value in fields.items():
                setattr(processor, name, value)
            try:
                with mock.patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=capturing
                ):
                    return processor._local_argmax_sink_for(
                        logits, SimpleNamespace(forward_mode=mode)
                    )
            finally:
                processor.do_tensor_parallel_all_gather = True
                processor.do_tensor_parallel_all_gather_dp_attn = False
                processor.use_attn_tp_group = False
                processor.use_tp_lm_head_all_to_all = False
                processor.final_logit_softcapping = None

        self.assertIs(selected(), sink)
        self.assertIsNone(selected(capturing=False))
        self.assertIsNone(selected(mode=ForwardMode.DECODE))
        self.assertIsNone(selected(do_tensor_parallel_all_gather=False))
        self.assertIsNone(selected(do_tensor_parallel_all_gather_dp_attn=True))
        self.assertIsNone(selected(use_attn_tp_group=True))
        self.assertIsNone(selected(use_tp_lm_head_all_to_all=True))
        self.assertIsNone(selected(final_logit_softcapping=30.0))


if __name__ == "__main__":
    unittest.main()
