"""Full-N composition routing, real track preparation and first-token stand-ins."""
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from test_flash_next_agg_fulln import worker, schedule, init_graphs, torch, agg
from test_flash_next_p48_contract import worker as pd_worker

FLAG = 'SGLANG_GDN_AGG_FULLN_FINAL_ONLY'
TRACK = ('mamba_last_track_idx', 'mamba_next_track_idx', 'mamba_last_track_seqlen')


def execute(w, batch):
    fb = w.Forward.init_new(batch, w.runner)
    w.owner.prepare_forward_batch(fb)
    w.backend.init_forward_metadata(fb)
    out = w.owner.forward(fb.input_ids, fb.input_ids, fb)
    return fb, out


def track_state(batch):
    return [[getattr(req.kv, name) for name in TRACK] for req in batch.reqs]


class FullNFinalOnlyTest(unittest.TestCase):
    def test_unset_and_zero_keep_fullN_nonfinal_batches_bitwise(self):
        results = []
        for value in (None, '0'):
            with worker() as w:
                os.environ.pop(FLAG, None)
                if value is not None:
                    w.stack.enter_context(patch.dict(os.environ, {FLAG: value}))
                self.assertEqual(os.environ.get(FLAG), value)
                init_graphs(w.runner, w.capture)
                batch = schedule(w, rows=2, finals=[True, False])
                fb, out = execute(w, batch)
                self.assertTrue(fb._pfactor_agg_contract)
                w.pool._agg_prefill_graph.run.assert_called_once()
                results.append((out.next_token_logits.clone(), w.pool.count.clone(), track_state(batch)))
        self.assertTrue(torch.equal(results[0][0], results[1][0]))
        self.assertTrue(torch.equal(results[0][1], results[1][1]))
        self.assertEqual(results[0][2], results[1][2])

    def test_all_final_multiline_replays_without_new_row_threshold(self):
        for rows in (1, 2, 3, 4, 8, 16):
            results = []
            for value in ('0', '1'):
                with self.subTest(rows=rows, flag=value), worker() as w, patch.dict(os.environ, {FLAG: value}):
                    self.assertEqual(os.environ.get(FLAG), value)
                    init_graphs(w.runner, w.capture)
                    batch = schedule(w, rows=rows, finals=[True] * rows)
                    fb, out = execute(w, batch)
                    self.assertTrue(fb._pfactor_agg_contract)
                    self.assertEqual(w.backend.plan_calls, 1)
                    self.assertEqual(w.owner._agg_fulln_trunk_replays, 1)
                    w.pool._agg_prefill_graph.run.assert_called_once()
                    self.assertEqual(fb.factored_prefill_boundary_steps, 0)
                    results.append((out.next_token_logits.clone(), w.pool.count.clone(), track_state(batch)))
            self.assertTrue(torch.equal(results[0][0], results[1][0]), rows)
            self.assertTrue(torch.equal(results[0][1], results[1][1]), rows)
            self.assertEqual(results[0][2], results[1][2])

    def test_any_nonfinal_falls_back_whole_batch_before_ownership(self):
        for rows in (1, 2, 4, 8):
            for nonfinal in {0, rows - 1}:
                finals = [True] * rows
                finals[nonfinal] = False
                results = []
                for contract in (False, True):
                    with self.subTest(rows=rows, nonfinal=nonfinal, contract=contract), \
                         worker(contract) as w, patch.dict(os.environ, {FLAG: '1'}):
                        self.assertEqual(os.environ.get(FLAG), '1')
                        init_graphs(w.runner, w.capture)
                        batch = schedule(w, rows=rows, finals=finals)
                        before = track_state(batch)
                        fb, out = execute(w, batch)
                        self.assertFalse(getattr(fb, '_pfactor_agg_contract', False))
                        self.assertEqual(before, track_state(batch))
                        if contract:
                            self.assertTrue(all(not r._pfactor_agg_contract for r in batch.reqs))
                            self.assertEqual(w.owner._agg_fulln_summary.fallbacks, 1)
                            self.assertEqual(w.owner._agg_fulln_summary.batch_publications, 0)
                            w.pool._agg_prefill_graph.run.assert_not_called()
                        results.append((out.next_token_logits.clone(), w.pool.count.clone(), before,
                                        fb.mamba_track_mask.clone(), fb.mamba_track_seqlens.clone()))
                for index in (0, 1, 3, 4):
                    self.assertTrue(torch.equal(results[0][index], results[1][index]), (rows, nonfinal, index))
                self.assertEqual(results[0][2], results[1][2])

    def test_late_composition_change_restores_entire_batch_before_plan(self):
        with worker() as w, patch.dict(os.environ, {FLAG: '1'}):
            init_graphs(w.runner, w.capture)
            batch = schedule(w, rows=2)
            self.assertEqual([r.kv.mamba_last_track_seqlen for r in batch.reqs], [64, 64])
            # A later batch view includes a continuation row. Re-select all rows
            # before metadata planning; final row must recover the old N-1 rule.
            batch.reqs[1].origin_input_ids.extend([0] * 32)
            fb = w.Forward.init_new(batch, w.runner)
            self.assertFalse(fb._pfactor_agg_contract)
            self.assertEqual([r.kv.mamba_last_track_seqlen for r in batch.reqs], [None, 64])
            self.assertEqual(fb.mamba_track_mask.tolist(), [False, True])
            self.assertEqual(w.backend.plan_calls, 0)
            w.pool._agg_prefill_graph.run.assert_not_called()

    def test_missing_or_changed_forward_metadata_cannot_publish(self):
        for value in (None, [], [True], [True, False], [1, True]):
            self.assertFalse(agg.all_prompt_final(NS(batch_size=2, twinstar_prompt_final=value)))
        self.assertTrue(agg.all_prompt_final(NS(batch_size=2, twinstar_prompt_final=[True, True])))
        with worker() as w, patch.dict(os.environ, {FLAG: '1'}):
            init_graphs(w.runner, w.capture)
            batch = schedule(w, rows=2)
            fb = w.Forward.init_new(batch, w.runner)
            w.owner.prepare_forward_batch(fb)
            w.backend.init_forward_metadata(fb)
            fb.twinstar_prompt_final[1] = False
            with self.assertRaisesRegex(RuntimeError, 'changed after checkpoint planning'):
                w.owner.forward(fb.input_ids, fb.input_ids, fb)
            w.pool._agg_prefill_graph.run.assert_not_called()

    def test_PD_contract_is_unchanged_by_AGG_composition_flag(self):
        for value in ('0', '1'):
            with pd_worker() as w, patch.dict(os.environ, {FLAG: value}):
                init_graphs(w.runner, w.capture)
                self.assertTrue(w.owner._pfactor_agg_installed)
                self.assertTrue(w.owner._exact_tail_installed)
                self.assertTrue(w.pool._agg_prefill_graph.warmed)


if __name__ == '__main__':
    unittest.main()
