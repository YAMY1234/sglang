"""Decoder replay graphs of DeepSeek-V4's trimmed late layers: bucketing, the state's
static-buffer round trip and the replay refill. CPU only; capture runs on GPU."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.models.deepseek_v4_mhc import HcPending, HcState
from sglang.srt.models.deepseek_v4_replay_graphs import (
    DecoderReplayGraphs,
    _flatten_state,
    _ReplayGraph,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _graphs(max_rows=1024):
    return DecoderReplayGraphs(model=None, run_layers=None, max_rows=max_rows)


class TestBucketRows(CustomTestCase):
    def test_multiples_of_128_up_to_max(self):
        graphs = _graphs(max_rows=1024)
        self.assertIsNone(graphs.bucket_rows(0))
        self.assertEqual(graphs.bucket_rows(1), 128)
        self.assertEqual(graphs.bucket_rows(128), 128)
        self.assertEqual(graphs.bucket_rows(129), 256)
        self.assertEqual(graphs.bucket_rows(1024), 1024)
        self.assertIsNone(graphs.bucket_rows(1025))


class TestFlattenState(CustomTestCase):
    def test_residual_with_pre(self):
        state = HcState(torch.randn(3, 4, 8), torch.randn(3, 4))
        leaves, structure, rebuild = _flatten_state(state)
        self.assertEqual(len(leaves), 2)
        back = rebuild(leaves)
        self.assertIs(back.streams, state.streams)
        self.assertIs(back.pre, state.pre)

    def test_pending_without_pre(self):
        pending = HcPending(*(torch.randn(3, 4) for _ in range(4)))
        leaves, structure, rebuild = _flatten_state(HcState(pending))
        self.assertEqual(len(leaves), 4)
        back = rebuild(leaves)
        self.assertIsInstance(back.streams, HcPending)
        self.assertIsNone(back.pre)
        other = _flatten_state(HcState(torch.randn(3, 4)))[1]
        self.assertNotEqual(structure, other)


class TestReplayRefill(CustomTestCase):
    def test_second_step_refills_static_rows_without_recapture(self):
        graphs = _graphs()
        graphs._break_context = lambda batch: nullcontext()
        captured = {}

        def capture(key, live, rebuild, forward_batch):
            rows = key[0]
            inputs = [t.new_zeros((rows, *t.shape[1:])) for t in live]
            for buf, t in zip(inputs, live):
                buf[: t.shape[0]].copy_(t)
            graph = _ReplayGraph(
                SimpleNamespace(replay=MagicMock()),
                inputs,
                torch.zeros((), dtype=torch.int32),
                inputs[0] * 2,
                None,
                forward_batch,
            )
            captured[key] = graph
            return graph

        graphs._capture = MagicMock(side_effect=capture)
        batch = SimpleNamespace(global_num_token_non_padded_cpu=None)

        def step(n, seed):
            torch.manual_seed(seed)
            state = HcState(torch.randn(n, 4))
            residual, pre = graphs.run(
                state=state,
                positions=torch.arange(n),
                input_ids=torch.arange(n) + 7,
                input_ids_global=torch.arange(n) + 7,
                forward_batch=batch,
            )
            return state, residual, pre

        step(100, 0)
        state, residual, pre = step(90, 1)
        self.assertEqual(graphs._capture.call_count, 1)
        (graph,) = captured.values()
        self.assertEqual(graph.inputs[0].shape[0], 128)
        torch.testing.assert_close(graph.inputs[0][:90], state.streams)
        torch.testing.assert_close(graph.inputs[1][:90], torch.arange(90))
        self.assertEqual(graph.graph.replay.call_count, 2)
        # MoE top-k must not route the pad rows of this step.
        self.assertEqual(int(graph.num_token_non_padded), 90)
        self.assertEqual(residual.shape[0], 90)
        self.assertIsNone(pre)
        # The caller's batch is never edited; the breaks read a copy.
        self.assertIsNone(batch.global_num_token_non_padded_cpu)


class TestTrimFollowsTheBank(CustomTestCase):
    """Merged tree: the backend trims an eager step only where the bank will replay
    the trimmed layers (or the step is large enough to be GPU-bound)."""

    def setUp(self):
        from sglang.srt.models import deepseek_v4_replay_graphs as rg

        self.rg = rg
        saved = rg._ACTIVE
        rg._ACTIVE = None
        self.addCleanup(setattr, rg, "_ACTIVE", saved)
        schedule = patch.object(
            rg, "get_schedule", return_value=SimpleNamespace(chunked_prefill_size=8192)
        )
        schedule.start()
        self.addCleanup(schedule.stop)

    def _bank(self, dspark=None, max_rows=2048):
        return DecoderReplayGraphs(
            model=SimpleNamespace(dspark_layers_to_capture=dspark),
            run_layers=None,
            max_rows=max_rows,
        )

    def test_bank_runs_from_the_first_full_chunk_on(self):
        bank = self._bank()
        self.assertFalse(bank.would_run(num_tokens=1152, tail_rows=128))
        self.assertTrue(bank.would_run(num_tokens=8192, tail_rows=128))
        self.assertFalse(bank._warm, "the prediction must not warm the bank")
        bank._warm = True
        self.assertTrue(bank.would_run(num_tokens=1152, tail_rows=128))
        self.assertFalse(bank.would_run(num_tokens=9216, tail_rows=4096))

    def test_no_bank_under_dspark_or_without_one(self):
        self.assertFalse(self.rg.decoder_replay_would_run(num_tokens=8192, tail_rows=128))
        bank = self._bank(dspark=[1, 2])
        self.assertFalse(bank.would_run(num_tokens=8192, tail_rows=128))

    def test_backend_trims_where_the_bank_replays(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        backend = object.__new__(DeepseekV4AttnBackend)
        cold_8k = SimpleNamespace(forward_mode=None, extend_seq_lens_cpu=[8192])
        big = SimpleNamespace(forward_mode=None, extend_seq_lens_cpu=[8192, 8192])
        # No bank: an 8K eager step is host-paced, so no trim (L6-graph Round 2).
        self.assertFalse(backend._decoder_trim_pays(cold_8k))
        self.assertTrue(backend._decoder_trim_pays(big))
        self._bank()
        self.assertTrue(backend._decoder_trim_pays(cold_8k))


if __name__ == "__main__":
    unittest.main()
