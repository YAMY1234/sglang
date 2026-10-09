"""Eager replay graphs of DeepSeek-V4 layer ranges: bucketing, the state's static-buffer
round trip, refill without recapture, no capture outside the startup scope, and the
replay pointer guard. CPU only; capture runs on GPU."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.models import deepseek_v4_replay_graphs as rg
from sglang.srt.models.deepseek_v4_mhc import HcPending, HcState
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def setUpModule():
    # The default probes read the live KV pool and flashinfer; CPU tests use none.
    patcher = patch.dict(rg._pointer_probes, clear=True)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


def _graphs(buckets=(128, 256, 384, 512), tail_rows=True):
    graphs = rg.EagerReplayGraphs(
        name="test", model=None, run_layers=None, buckets=list(buckets),
        tail_rows=tail_rows,
    )
    graphs._break_context = lambda batch: nullcontext()
    return graphs


def _fake_capture(graphs, captured):
    def capture(*, rows, num_rows, shape_key, static, num_leaves, rebuild, names,
                forward_batch):
        out = [t[:rows] * 2 for t in static[:num_leaves]]
        graph = rg._ReplayGraph(
            SimpleNamespace(replay=MagicMock()),
            rows,
            lambda n: (rebuild([t[:n] for t in out]), []),
            {},
            {"num_token_non_padded": torch.zeros((), dtype=torch.int32),
             "q_pad_buffer": None},
            None,
        )
        captured.append(graph)
        return graph

    graphs._capture = MagicMock(side_effect=capture)


def _step(graphs, n, seed=0):
    torch.manual_seed(seed)
    state = HcState(torch.randn(n, 4))
    out = graphs.run(
        state=state,
        forward_batch=SimpleNamespace(global_num_token_non_padded_cpu=None),
        positions=torch.arange(n),
        hash_ids=None,
    )
    return state, out


class TestBuckets(CustomTestCase):
    def test_smallest_bucket_at_or_above_rows(self):
        graphs = _graphs(buckets=(256, 512, 768))
        self.assertIsNone(graphs.bucket_rows(0))
        self.assertEqual(graphs.bucket_rows(1), 256)
        self.assertEqual(graphs.bucket_rows(256), 256)
        self.assertEqual(graphs.bucket_rows(257), 512)
        self.assertIsNone(graphs.bucket_rows(769))


class TestFlattenState(CustomTestCase):
    def test_pending_round_trip_and_structure_key(self):
        pending = HcPending(*(torch.randn(3, 4) for _ in range(4)))
        leaves, structure, rebuild = rg._flatten_state(HcState(pending, torch.randn(3, 4)))
        self.assertEqual(len(leaves), 5)
        back = rebuild(leaves)
        self.assertIsInstance(back.streams, HcPending)
        self.assertIs(back.pre, leaves[-1])
        self.assertNotEqual(structure, rg._flatten_state(HcState(torch.randn(3, 4)))[1])

    def test_side_outputs_follow_the_state_leaves(self):
        """DSpark aux rows ride after the state's leaves in the static outputs; a
        graph with them must not share output buffers with one without them."""
        state = HcState(torch.randn(3, 4), torch.randn(3, 4))
        aux = [torch.randn(3, 2), torch.randn(3, 2)]
        leaves, structure, rebuild = rg._flatten_output((state, aux))
        self.assertEqual(len(leaves), 4)
        back, back_aux = rebuild([t[:2] for t in leaves])
        torch.testing.assert_close(back.pre, state.pre[:2])
        torch.testing.assert_close(back_aux[1], aux[1][:2])
        self.assertNotEqual(structure, rg._flatten_output((state, []))[1])


class TestRunAndCapture(CustomTestCase):
    def test_no_capture_outside_startup_scope(self):
        graphs, captured = _graphs(), []
        _fake_capture(graphs, captured)
        _, out = _step(graphs, 100)
        self.assertIsNone(out)
        self.assertEqual(graphs._capture.call_count, 0)

    def test_refill_without_recapture_and_row_count(self):
        graphs, captured = _graphs(), []
        _fake_capture(graphs, captured)
        with graphs.capture_scope():
            _step(graphs, 100)
        state, out = _step(graphs, 90, seed=1)
        self.assertEqual(graphs._capture.call_count, 1)
        static = next(iter(graphs._static.values()))
        self.assertEqual(static[0].shape[0], 512)
        torch.testing.assert_close(static[0][:90], state.streams)
        torch.testing.assert_close(static[1][:90], torch.arange(90))
        self.assertEqual(captured[0].graph.replay.call_count, 2)
        self.assertEqual(int(captured[0].owned["num_token_non_padded"]), 90)
        self.assertEqual(out[0].streams.shape[0], 90)


class TestPointerGuard(CustomTestCase):
    def test_assert_fires_when_a_probed_buffer_is_reallocated(self):
        holder = {"buf": torch.zeros(16)}
        rg.register_pointer_probe("fake", lambda: {"buf": holder["buf"].data_ptr()})
        owned = {"q_pad_buffer": torch.zeros(4)}
        recorded = rg._pointer_snapshot(owned)
        rg.check_pointers(recorded, owned, "test graph")
        holder["buf"] = torch.zeros(32)
        with self.assertRaisesRegex(AssertionError, "fake.buf"):
            rg.check_pointers(recorded, owned, "test graph")
        holder["buf"] = torch.zeros(16)
        recorded = rg._pointer_snapshot(owned)
        owned["q_pad_buffer"] = torch.zeros(4)
        with self.assertRaisesRegex(AssertionError, "owned.q_pad_buffer"):
            rg.check_pointers(recorded, owned, "test graph")


class TestStartupPlan(CustomTestCase):
    def test_full_chunk_first_then_every_bucket_largest_first(self):
        runs = []
        runner = SimpleNamespace(
            _alloc_dummy_decode_buffers=lambda bs, num_tokens_per_req: SimpleNamespace(
                positions=torch.zeros(bs * num_tokens_per_req, dtype=torch.int64)
            ),
            _dummy_run=lambda bs, forward_mode_override, buffers,
            extend_num_tokens_per_req: runs.append((bs, extend_num_tokens_per_req)),
        )
        full = _graphs(buckets=(256, 512), tail_rows=False)
        tail = _graphs(buckets=(128, 256), tail_rows=True)
        with patch.object(rg, "get_parallel", lambda: SimpleNamespace(tp_rank=1)):
            rg.capture_at_startup(eager_runner=runner, request_window=None,
                                  graphs=[full, tail])
        self.assertEqual(runs[0], (1, 512))
        self.assertEqual(sorted(runs[1:], key=lambda r: -r[0] * r[1]), runs[1:])
        self.assertEqual(set(runs[1:]), {(1, 512), (1, 256), (1, 128), (2, 128)})
        self.assertFalse(full._capture_open or tail._capture_open)


class TestStartupCaptureUnderSpeculation(CustomTestCase):
    def test_target_captures_under_dspark_draft_does_not(self):
        """A DSpark target once skipped startup capture, so its eager steps never
        replayed; the draft model has no eager replay graphs to capture."""
        from sglang.srt.model_executor.model_runner_components import (
            cuda_graph_setup,
        )

        def runner(is_draft_worker):
            return SimpleNamespace(
                model=SimpleNamespace(model=SimpleNamespace(eager_replay_graphs=[1])),
                is_draft_worker=is_draft_worker,
                token_to_kv_pool=SimpleNamespace(request_window=None),
            )

        with patch.object(rg, "capture_at_startup") as capture:
            cuda_graph_setup._capture_eager_replay_graphs(runner(False), None)
            cuda_graph_setup._capture_eager_replay_graphs(runner(True), None)
        self.assertEqual(capture.call_count, 1)


class TestTrimFollowsTheBank(CustomTestCase):
    """Merged tree: the backend trims an eager step only where the bank will replay
    the trimmed layers (or the step is large enough to be GPU-bound)."""

    def setUp(self):
        saved = rg._ACTIVE
        rg._ACTIVE = None
        self.addCleanup(setattr, rg, "_ACTIVE", saved)

    def _bank(self, dspark=None):
        return rg.EagerReplayGraphs(
            name="decoder-replay",
            model=SimpleNamespace(dspark_layers_to_capture=dspark),
            run_layers=None,
            buckets=list(range(128, 2049, 128)),
            tail_rows=True,
        )

    def test_bank_runs_only_for_captured_buckets(self):
        bank = self._bank()
        self.assertFalse(bank.would_run(num_tokens=1152, tail_rows=128))
        with bank.capture_scope():
            self.assertTrue(bank.would_run(num_tokens=8192, tail_rows=128))
        bank._graphs[(128, None)] = object()
        self.assertTrue(bank.would_run(num_tokens=1152, tail_rows=100))
        self.assertFalse(bank.would_run(num_tokens=1152, tail_rows=256))
        self.assertFalse(bank.would_run(num_tokens=9216, tail_rows=4096))

    def test_bank_runs_under_dspark_not_without_one(self):
        """Under DSpark the bank must replay (it emits the draft's aux rows); it
        once never did, so DSpark eager steps lost the trim and the graphs."""
        self.assertFalse(rg.decoder_replay_would_run(num_tokens=8192, tail_rows=128))
        bank = self._bank(dspark=[1, 2])
        bank._graphs[(128, None)] = object()
        self.assertTrue(bank.would_run(num_tokens=8192, tail_rows=128))

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
        bank = self._bank()
        bank._graphs[(128, None)] = object()
        self.assertTrue(backend._decoder_trim_pays(cold_8k))


class TestIndexerStepRowsCache(CustomTestCase):
    """Lean breaks share the DeepGEMM prefill indexer's row plumbing across the index
    layers of one step; a new step (new page table) or other rows must never reuse it."""

    def setUp(self):
        from sglang.srt.layers.attention.dsv4.v41_indexer import scoring

        self.scoring = scoring
        scoring._step_rows_cache.clear()
        self.addCleanup(scoring._step_rows_cache.clear)
        self.req_to_token = torch.arange(4 * 64).reshape(4, 64)
        self.positions = torch.arange(40)

    def _inputs(self, page_table, positions):
        return SimpleNamespace(
            compress_ratio=2,
            positions=positions,
            seq_lens_cpu=[24, 16],
            req_pool_indices=self.req_pool,
            kv_page_table=page_table,
            rows_per_request_device=torch.tensor([24, 16]),
        )

    def test_reused_within_a_step_only(self):
        self.req_pool = torch.tensor([1, 3])
        step1 = torch.zeros(40, 8, dtype=torch.int32)
        first = self.scoring._step_rows(self._inputs(step1, self.positions), self.req_to_token)
        # Another index layer of the same step passes a fresh view of the same rows.
        again = self.scoring._step_rows(
            self._inputs(step1, self.positions[:40]), self.req_to_token
        )
        self.assertIs(again, first)
        torch.testing.assert_close(first["k_slots"][:12], self.req_to_token[1, 0:24:2] // 2)
        step2 = torch.zeros(40, 8, dtype=torch.int32)
        self.assertIsNot(
            self.scoring._step_rows(self._inputs(step2, self.positions), self.req_to_token),
            first,
        )

    def test_not_cached_when_lean_breaks_off(self):
        from sglang.srt.environ import envs

        self.req_pool = torch.tensor([1, 3])
        page_table = torch.zeros(40, 8, dtype=torch.int32)
        with envs.SGLANG_DSV4_EAGER_GRAPH_LEAN_BREAKS.override(False):
            first = self.scoring._step_rows(
                self._inputs(page_table, self.positions), self.req_to_token
            )
            again = self.scoring._step_rows(
                self._inputs(page_table, self.positions), self.req_to_token
            )
        self.assertIsNot(again, first)


class TestSizeRouting(CustomTestCase):
    """Steps of many rows and few requests leave the prefill graph for trimmed
    eager with the bank; everything else keeps the graph (L6-merged, order #34)."""

    def setUp(self):
        from sglang.srt.environ import envs

        saved = rg._ACTIVE
        rg._ACTIVE = None
        self.addCleanup(setattr, rg, "_ACTIVE", saved)
        self.envs = envs
        # The bank-only crossover (6K rows, at most 2 requests); defaults below.
        envs.SGLANG_DSV4_PREFILL_GRAPH_ROUTE_ROWS.set(6144)
        self.addCleanup(envs.SGLANG_DSV4_PREFILL_GRAPH_ROUTE_ROWS.clear)
        envs.SGLANG_DSV4_PREFILL_GRAPH_ROUTE_MAX_REQS.set(2)
        self.addCleanup(envs.SGLANG_DSV4_PREFILL_GRAPH_ROUTE_MAX_REQS.clear)

    def _route(self, lens):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        backend = object.__new__(DeepseekV4AttnBackend)
        backend.enable_decoder_swa_bounded_replay = True
        return backend._routes_prefill_eager(
            SimpleNamespace(extend_seq_lens_cpu=list(lens))
        )

    def test_rows_and_requests_decide_the_path(self):
        self.assertFalse(self._route([8192]), "no bank: eager would be slower")
        bank = rg.EagerReplayGraphs(
            name="decoder-replay",
            model=SimpleNamespace(dspark_layers_to_capture=None),
            run_layers=None,
            buckets=list(range(128, 2049, 128)),
            tail_rows=True,
        )
        for rows in (128, 256, 1024):  # captured at startup
            bank._graphs[(rows, None)] = object()
        self.assertTrue(self._route([8192]))  # cold 8K chunk
        self.assertTrue(self._route([4096, 4096]))
        self.assertFalse(self._route([1024] * 8))  # 8 short requests
        self.assertFalse(self._route([1152] * 8))  # a 7K+1K hit wave, folded
        self.assertFalse(self._route([4096]))  # below the crossover
        # Defaults (full-layer graphs): 1.5K rows, any number of requests.
        self.envs.SGLANG_DSV4_PREFILL_GRAPH_ROUTE_ROWS.clear()
        self.envs.SGLANG_DSV4_PREFILL_GRAPH_ROUTE_MAX_REQS.clear()
        self.assertTrue(self._route([1024] * 8))
        self.assertTrue(self._route([2048]))
        self.assertFalse(self._route([1152]))  # a 7K+1K hit alone keeps the graph


if __name__ == "__main__":
    unittest.main()
