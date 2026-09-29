"""P48 full-N dispatch, checkpoint phase selection and unchanged native factors."""
import copy
from contextlib import ExitStack
from functools import wraps
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.state_handoff import FactorStateHandoff
from sglang.srt.mem_cache import gdn_factored_pool as native
from sglang.srt.mem_cache import gdn_prefill_agg_contract as agg
from sglang.srt.mem_cache import gdn_prefill_batch_graph as batch_graph
from test_gdn_prefill_batch_graph import fake_pool, make_plan


def mode(mixed=False):
    return NS(is_extend=lambda: True, is_mixed=lambda: mixed)


def classes():
    class Schedule:
        def _mamba_radix_cache_v2_req_prepare_for_extend(self, req):
            req.kv.mamba_last_track_idx = req.kv.mamba_next_track_idx
            req.kv.mamba_next_track_idx ^= 1
            req.kv.mamba_last_track_seqlen = req.extend_range.length
            return NS(track_mask=True, track_index=5, track_seqlen=req.extend_range.length)
    original_track = Schedule._mamba_radix_cache_v2_req_prepare_for_extend
    @wraps(original_track)
    def track(self, req):
        proxy = copy.copy(req)
        proxy.extend_range = NS(length=req.extend_range.length-1)
        return original_track(self, proxy)
    Schedule._mamba_radix_cache_v2_req_prepare_for_extend = track

    class Forward:
        @classmethod
        def init_new(cls, batch, model_runner, **kwargs):
            return NS(forward_mode=batch.forward_mode, batch_size=len(batch.reqs),
                input_ids=batch.input_ids,
                extend_seq_lens_cpu=[r.extend_range.length for r in batch.reqs],
                spec_info=None, can_run_tbo=batch.tbo_split_seq_index is not None,
                tbo_split_seq_index=batch.tbo_split_seq_index,
                pd_factor_only_full_batch=True,
                mamba_track_mask=batch.mamba_track_mask,
                mamba_track_indices=batch.mamba_track_indices,
                mamba_track_seqlens=batch.mamba_track_seqlens)
    class Backend:
        def init_forward_metadata(self, batch):
            self.plan_calls += 1
            return tuple(batch.extend_seq_lens_cpu)
    original_metadata = Backend.init_forward_metadata
    @wraps(original_metadata)
    def metadata(self, batch):
        return None if batch.pd_factor_only_full_batch else original_metadata(self, batch)
    Backend.init_forward_metadata = metadata
    class Handoff(FactorStateHandoff):
        pass
    @wraps(FactorStateHandoff.before_send)
    def send(self, req):
        req.factored_prefill_boundary_steps = 1
        return FactorStateHandoff.before_send(self, req)
    Handoff.before_send = send
    return Forward, Schedule, Backend, Handoff


def request(tokens=8192):
    return NS(extend_range=NS(start=0, end=tokens, length=tokens),
        origin_input_ids=list(range(tokens)), kv=NS(mamba_last_track_idx=None,
        mamba_next_track_idx=0, mamba_last_track_seqlen=None, mamba_pool_idx=torch.tensor(1)))


def schedule(cls, req, *, mixed=False):
    batch = cls()
    batch.reqs = [req]
    batch.forward_mode = mode(mixed)
    batch.spec_info = None
    batch.tbo_split_seq_index = None
    batch.input_ids = torch.arange(req.extend_range.length)
    return batch


def prepare_track(batch, req):
    entry = batch._mamba_radix_cache_v2_req_prepare_for_extend(req)
    batch.mamba_track_mask = torch.tensor([entry.track_mask])
    batch.mamba_track_indices = torch.tensor([entry.track_index])
    batch.mamba_track_seqlens = torch.tensor([entry.track_seqlen])
    return entry


class AggContractTest(unittest.TestCase):
    def test_observer_wrappers_do_not_restore_the_legacy_boundary_contract(self):
        Forward, Schedule, Backend, Handoff = classes()
        def observe(operation):
            @wraps(operation)
            def call(*args, **kwargs):
                return operation(*args, **kwargs)
            return call
        Schedule._mamba_radix_cache_v2_req_prepare_for_extend = observe(
            Schedule._mamba_radix_cache_v2_req_prepare_for_extend)
        Handoff.before_send = observe(Handoff.before_send)
        agg.install_contracts(Forward, Schedule, Backend, Handoff)
        with patch.dict(os.environ, {agg.FLAG: '1'}):
            req = request(); batch = schedule(Schedule, req)
            self.assertEqual(prepare_track(batch, req).track_seqlen, 8192)
            Forward.init_new(batch, NS(device='cpu'))
            pool = fake_pool(36); pool.cfg.strict_chunk = 1
            pool.count.fill_(pool.cfg.r)
            Handoff(pool).before_send(req=req)
            self.assertEqual(req.factored_prefill_boundary_steps, 0)

    def test_full_n_track_single_plan_phase_zero_and_decode_payload_unchanged(self):
        Forward, Schedule, Backend, Handoff = classes()
        agg.install_contracts(Forward, Schedule, Backend, Handoff)
        with patch.dict(os.environ, {agg.FLAG: '1'}):
            req = request()
            batch = schedule(Schedule, req)
            entry = prepare_track(batch, req)
            self.assertEqual(entry.track_seqlen, 8192)
            forward = Forward.init_new(batch, NS(device='cpu'))
            self.assertTrue(forward._pfactor_agg_contract)
            self.assertEqual(req.kv.mamba_last_track_seqlen, 8192)
            self.assertEqual(req.kv.mamba_next_track_idx, 1)
            backend = Backend(); backend.plan_calls = 0
            self.assertEqual(backend.init_forward_metadata(forward), (8192,))
            self.assertEqual(backend.plan_calls, 1)
            pool = fake_pool(36)
            pool.count.fill_(pool.cfg.r)
            Handoff(pool).before_send(req)
            self.assertEqual(req.factored_prefill_boundary_steps, 0)
            before = pool.count.clone()
            pool.mark_transferred_slots = Mock()
            FactorStateHandoff(pool).commit_receive(req)
            torch.testing.assert_close(pool.count, before, atol=0, rtol=0)

    def test_late_tbo_or_mixed_restores_n_minus_one_and_one_ping_pong_swap(self):
        for reason in ('tbo', 'mixed', 'flag-off'):
            with self.subTest(reason=reason):
                Forward, Schedule, Backend, Handoff = classes()
                agg.install_contracts(Forward, Schedule, Backend, Handoff)
                with patch.dict(os.environ, {agg.FLAG: '1'}):
                    req = request(); batch = schedule(Schedule, req)
                    prepare_track(batch, req)
                    if reason == 'tbo': batch.tbo_split_seq_index = 0
                    if reason == 'mixed': batch.forward_mode = mode(True)
                    with patch.dict(os.environ, {agg.FLAG: '0' if reason == 'flag-off' else '1'}):
                        forward = Forward.init_new(batch, NS(device='cpu'))
                    self.assertFalse(forward._pfactor_agg_contract)
                    self.assertEqual(forward.mamba_track_seqlens.tolist(), [8191])
                    self.assertEqual(req.kv.mamba_last_track_seqlen, 8191)
                    self.assertEqual(req.kv.mamba_next_track_idx, 1)
                    self.assertTrue(forward.pd_factor_only_full_batch)
                    self.assertEqual(forward._pfactor_legacy_mixed, reason == 'mixed')
                    if reason == 'mixed':
                        self.assertEqual(forward.twinstar_prompt_final, [True])
                    backend = Backend(); backend.plan_calls = 0
                    self.assertIsNone(backend.init_forward_metadata(forward))
                    self.assertEqual(backend.plan_calls, 0)
                    pool = fake_pool(36); pool.cfg.strict_chunk = 1
                    pool.count.fill_(pool.cfg.r+1)
                    Handoff(pool).before_send(req)
                    self.assertEqual(req.factored_prefill_boundary_steps, 1)

    def test_empty_prefix_and_batch32_select_legacy_before_any_metadata(self):
        for tokens, rows in ((1, 1), (8192, 32)):
            Forward, Schedule, Backend, Handoff = classes()
            agg.install_contracts(Forward, Schedule, Backend, Handoff)
            with patch.dict(os.environ, {agg.FLAG: '1'}):
                req = request(tokens); batch = schedule(Schedule, req)
                batch.reqs = [req] * rows
                self.assertFalse(agg.eligible(batch))

    def test_whole_commit_matches_native_agg_pools_bitwise(self):
        torch.set_num_threads(2)
        torch.manual_seed(1325)
        for rows in (1, 8, 16):
            for tracked in (False, True):
                with self.subTest(rows=rows, tracked=tracked):
                    pool = fake_pool(36, capacity=100)
                    pool.prefill_factor_graph = None
                    pool.batch_prefill_final_copy = True
                    pool.save_prefix_dense = lambda lid, slots, dense: pool.prefix_valid.index_fill_(0, slots, 1)
                    pool.copy_slots = lambda src, dst: [native.FactoredGDNPool._copy_slots_layer_eager(pool, lid, src, dst)
                                                        for lid in pool.layer_ids]
                    plan = make_plan(pool, rows)
                    states = [(torch.randn(rows, 2, 16, 16),
                               torch.randn(rows, 2, 16, 16).bfloat16() if tracked else None)
                              for _ in pool.layer_ids]
                    slots = torch.arange(40, 40+rows) if tracked else None
                    src, dst = plan.slots, torch.arange(70, 70+rows)
                    fields = ('a', 'U', 'W', 'count', 'stale', 'dense_of', 'dense_required', 'prefix_valid')
                    original = {name: getattr(pool, name).clone() for name in fields}
                    def store(a, u, w, pa, pu, pw, count, stale, dense_of, ids, r, *, stale_value, **kwargs):
                        pa[ids] = a; pu[ids] = u; pw[ids] = w; count[ids] = r
                        stale[ids] = stale_value
                        if stale_value: dense_of[ids] = -1
                    def scatter(source, target, ids):
                        for i, slot in enumerate(ids.tolist()):
                            if slot >= 0: target[:, slot] = source[:, i]
                    class Publish:
                        def __getitem__(self, grid):
                            return lambda valid, ids, *a: valid.index_fill_(0, ids, 1)
                    with patch('sglang.srt.layers.attention.linear.kernels.gdn_factored_io.store_factored', side_effect=store), \
                         patch.object(batch_graph, 'scatter_rows', side_effect=scatter), \
                         patch.object(batch_graph, '_publish_valid', Publish()), \
                         patch.dict(os.environ, {'SGLANG_GDN_PREFILL_COMMIT_GRAPH': '0'}):
                        plan.pending = states.copy()
                        native.FactoredGDNPool._commit_extend_group(pool, 35, plan,
                            *states[-1], slots, src, dst)
                        expected = {name: getattr(pool, name).clone() for name in fields}
                        for name, values in original.items(): getattr(pool, name).copy_(values)
                        buffers = batch_graph.BatchBuffers(pool, rows, rows if tracked else None, include_tail=False)
                        self.assertFalse(buffers.tail)
                        buffers.bind(plan, states, slots, src, dst)
                        buffers.evaluate(native.factorize_layers)
                        for name in fields:
                            torch.testing.assert_close(getattr(pool, name), expected[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
