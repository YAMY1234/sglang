import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.speculative.dspark.simulated_bonus import (
    simulated_bonus_select_partials,
)
from sglang.srt.speculative.dspark_components import dspark_verify
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
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


class _NoSiteTpSync(_RecordingTpSync):
    def enabled(self, site):
        return False


class TestFusedStaticAcceptTail(CustomTestCase):
    def _static_accept(self, *, fused, simulated, tp_sync):
        candidates, logits, prefix_lens = _greedy_inputs()
        epilogue = DsparkVerifyEpilogue(
            max_bs=BS + 1,
            verify_num_draft_tokens=STRIDE,
            device="cpu",
            tp_sync=tp_sync,
            simulated_correct_len=simulated,
            fused_accept_tail=fused,
        )
        epilogue._static_epilogue(
            SimpleNamespace(next_token_logits=logits),
            SimpleNamespace(
                batch_size=BS, input_ids=candidates.view(-1), seq_lens=prefix_lens
            ),
        )
        return epilogue

    def test_fused_matches_separate_ops(self):
        for simulated in (None, torch.tensor([1])):
            with self.subTest(simulated=simulated is not None):
                fused = self._static_accept(
                    fused=True, simulated=simulated, tp_sync=_NoSiteTpSync()
                )
                self.assertTrue(fused._serves_accept_tail(BS))
                separate = self._static_accept(
                    fused=False, simulated=simulated, tp_sync=_NoSiteTpSync()
                )
                got, ref = fused.read_accept(BS), separate.read_accept(BS)
                for field in ref.__struct_fields__:
                    self.assertTrue(
                        torch.equal(getattr(got, field), getattr(ref, field)), field
                    )

    def test_accept_site_broadcast_keeps_separate_ops(self):
        # Rank 0's results are broadcast between the ops; one kernel would skip it.
        class _AllSites(_RecordingTpSync):
            def enabled(self, site):
                return True

        sync = _AllSites()
        epilogue = self._static_accept(fused=True, simulated=None, tp_sync=sync)
        self.assertFalse(epilogue._serves_accept_tail(BS))
        self.assertEqual(
            [site for site, _ in sync.calls], [SpecTpSyncSite.DSPARK_ACCEPT_GRAPH] * 3
        )


def _sim_partials(*, target_logits, correct_len, bs, rows_per_request, **_):
    """Torch stand-in for the in-graph simulated bonus at temperature 0: per-split
    (max, first index) of each request's row at ``correct_len``."""
    rows = torch.arange(bs) * rows_per_request + correct_len.long()
    x = target_logits[rows]
    split = 16
    pad = (-x.shape[1]) % split
    x = torch.nn.functional.pad(x, (0, pad), value=float("-inf")).view(bs, -1, split)
    val, idx = x.max(dim=2)
    idx = idx + torch.arange(x.shape[1]) * split
    return val, idx.to(torch.int32)


def _sim_sample(**kw):
    val, idx = _sim_partials(**kw)
    return simulated_bonus_select_partials(
        part_val=val, part_idx=idx, vocab=kw["target_logits"].shape[1]
    )


class TestFusedTailSimulateFold(CustomTestCase):
    """simulate_fold: the tail takes the in-graph bonus and skips the argmax; with the
    accept-site broadcast selected it broadcasts the length and the draw first."""

    def _run(self, *, fused, tp_sync, forbid_argmax=False):
        candidates, logits, prefix_lens = _greedy_inputs()
        epilogue = DsparkVerifyEpilogue(
            max_bs=BS + 1,
            verify_num_draft_tokens=STRIDE,
            device="cpu",
            tp_sync=tp_sync,
            simulated_correct_len=torch.tensor([2]),
            simulate_fold=True,
            fused_accept_tail=fused,
        )

        def _no_argmax(*a, **k):
            raise AssertionError("argmax ran under simulate_fold")

        patches = [
            mock.patch.object(
                dspark_verify, "simulated_bonus_partials_graph", _sim_partials
            ),
            mock.patch.object(
                dspark_verify, "simulated_bonus_sample_graph", _sim_sample
            ),
        ]
        if forbid_argmax:
            patches.append(mock.patch.object(dspark_verify, "_row_argmax", _no_argmax))
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            epilogue._static_epilogue(
                SimpleNamespace(next_token_logits=logits),
                SimpleNamespace(
                    batch_size=BS, input_ids=candidates.view(-1), seq_lens=prefix_lens
                ),
            )
        return epilogue

    def _assert_same(self, got, ref):
        for field in ref.__struct_fields__:
            self.assertTrue(
                torch.equal(getattr(got, field), getattr(ref, field)), field
            )

    def test_fused_matches_separate_without_argmax(self):
        fused = self._run(fused=True, tp_sync=_NoSiteTpSync(), forbid_argmax=True)
        self.assertTrue(fused._serves_accept_tail(BS))
        separate = self._run(fused=False, tp_sync=_NoSiteTpSync())
        self._assert_same(fused.read_accept(BS), separate.read_accept(BS))

    def test_accept_site_broadcast_moves_to_the_inputs(self):
        class _AllSites(_RecordingTpSync):
            def enabled(self, site):
                return True

        sync = _AllSites()
        fused = self._run(fused=True, tp_sync=sync, forbid_argmax=True)
        self.assertTrue(fused._serves_accept_tail(BS))
        # The length and the draw, before the kernel; not the three derived tensors.
        self.assertEqual(
            [(site, tuple(v.shape)) for site, v in sync.calls],
            [
                (SpecTpSyncSite.DSPARK_ACCEPT_GRAPH, (1,)),
                (SpecTpSyncSite.DSPARK_ACCEPT_GRAPH, (BS,)),
            ],
        )
        ref_sync = _AllSites()
        separate = self._run(fused=False, tp_sync=ref_sync)
        self.assertEqual(len(ref_sync.calls), 3)
        self._assert_same(fused.read_accept(BS), separate.read_accept(BS))
        # The values broadcast are the separate path's length and bonus.
        self.assertTrue(torch.equal(sync.calls[1][1], ref_sync.calls[1][1]))


if __name__ == "__main__":
    unittest.main()
