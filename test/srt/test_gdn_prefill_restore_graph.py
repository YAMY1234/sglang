"""Real imports and real restore arithmetic. CPU CUDA driver shim, not GPU proof.

The shim replaces only stream/capture/graph driver operations. Pool constructor,
startup route, prewarm, bind, evaluate, initial_dense and replay dispatch execute
from the imported production modules. Every replay re-evaluates the real body.
"""
import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch

ROOT = Path(os.environ.get('PFACTOR077_SOURCE', Path(__file__).resolve().parents[2]))
BASE = Path(os.environ['PFACTOR075_BASE'])
sys.path.insert(0, str(ROOT / 'python'))
fp = importlib.import_module('sglang.srt.mem_cache.gdn_factored_pool')
gm = importlib.import_module('sglang.srt.mem_cache.gdn_prefill_initial_graph')
from sglang.srt.environ import envs
# Full module imports: no AST extraction or replacement implementation.
runner_module = importlib.import_module('sglang.srt.model_executor.model_runner')
spec = importlib.util.spec_from_file_location('_pfactor077_base', BASE / 'python/sglang/srt/mem_cache/gdn_factored_pool.py')
frozen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = frozen
spec.loader.exec_module(frozen)
TRACE = []


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def make_pool(module=fp, h=3, k=8, layers=36, rank=2):
    cfg = module.FactoredGDNConfig.parse(f'r={rank},m={rank},dtype=fp16,ring=2,strict_chunk=1,factored_prefix=1')
    params = types.SimpleNamespace(shape=types.SimpleNamespace(temporal=(h, k, k)))
    pool = module.FactoredGDNPool(size=3, cache_params=params, mamba_layer_ids=list(range(layers)),
                                  device='cpu', cfg=cfg)
    g = torch.Generator().manual_seed(77)
    for t in (pool.a, pool.U, pool.W, pool.vbar, pool.dense_ring):
        t.copy_(torch.randn(t.shape, generator=g).to(t.dtype))
    pool.count.fill_(3)
    pool.stats['densified'] = 0
    return pool


def plan(pool, *, fresh=False, rows=1, ring=False, slot=1):
    return fp.FactoredExtendPlan(
        slots=torch.arange(slot, slot + rows), use_ring=torch.full((rows,), ring),
        ring_src=torch.zeros(rows, dtype=torch.long), ring_dst=torch.zeros(rows, dtype=torch.long),
        ring_dst_rows=torch.arange(rows), n_ring_src=rows if ring else 0,
        all_fresh=fresh, last_layer=len(pool.layer_ids) - 1)


class Driver:
    def __init__(self):
        self.stream_id = 7
        self.capturing = False
        self.active = None
        self.events = []
        self.graphs = []

    def stream(self, device=None):
        owner = self
        class Stream:
            @property
            def cuda_stream(self): return owner.stream_id
            def wait_stream(self, other): owner.events.append('stream_wait')
        return Stream()

    @contextlib.contextmanager
    def graph(self, graph, **kwargs):
        self.active = graph
        try: yield
        finally: self.active = None

    def new_graph(self):
        owner = self
        class Graph:
            def replay(self):
                owner.events.append('replay')
                self.output.copy_(self.buffers.evaluate())
        g = Graph(); self.graphs.append(g); return g

    @contextlib.contextmanager
    def use(self):
        driver = self
        original = gm.RestoreBuffers.evaluate
        def evaluate(buffers):
            output = original(buffers)
            if driver.active is not None:
                driver.active.buffers, driver.active.output = buffers, output
            return output
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(torch.Tensor, 'is_cuda', property(lambda _: True)))
            stack.enter_context(mock.patch.object(torch.cuda, 'is_current_stream_capturing', lambda: self.capturing))
            stack.enter_context(mock.patch.object(torch.cuda, 'current_stream', self.stream))
            stack.enter_context(mock.patch.object(torch.cuda, 'Stream', self.stream))
            stack.enter_context(mock.patch.object(torch.cuda, 'stream', lambda _: contextlib.nullcontext()))
            stack.enter_context(mock.patch.object(torch.cuda, 'CUDAGraph', self.new_graph))
            stack.enter_context(mock.patch.object(torch.cuda, 'graph', self.graph))
            stack.enter_context(mock.patch.object(gm.RestoreBuffers, 'evaluate', evaluate))
            yield self


def startup(pool, enabled, role='null', fulln='0'):
    # Invoke the imported default runner init route; unrelated model/commit
    # graph initialization is stubbed, but the newly changed restore route is not.
    fake = types.SimpleNamespace(
        req_to_token_pool=types.SimpleNamespace(factored_gdn_pool=pool),
        server_args=types.SimpleNamespace(disaggregation_mode=role),
        model=types.SimpleNamespace(fullstack={'gdn_rank':16}))
    capture = types.SimpleNamespace(eager_runner=None, prefill=types.SimpleNamespace(runner=None),
                                   decode=types.SimpleNamespace(runner=None), memory_usage=0, time_usage=0)
    contract = importlib.import_module('sglang.srt.mem_cache.gdn_prefill_agg_contract')
    guard = importlib.import_module('sglang.srt.mem_cache.gdn_prefill_recipe_guard')
    with mock.patch.dict(os.environ, {'SGLANG_GDN_PREFILL_RESTORE_GRAPH':str(enabled),
            'SGLANG_GDN_AGG_FULLN_PREFILL':fulln, 'SGLANG_GDN_PREFILL_EXACT_TAIL_BATCH':'0',
            'SGLANG_GDN_PREFILL_INITIAL_GRAPH':'0', 'SGLANG_GDN_PSIDE_GRAPH':'0'}), \
            mock.patch.object(pool, 'prewarm_k31_batch_graph'), \
            mock.patch.object(pool, 'prewarm_commit_graph'), \
            mock.patch.object(contract, 'install'), mock.patch.object(contract, 'prewarm'), \
            mock.patch.object(guard, 'report_startup'), \
            mock.patch.object(runner_module, 'capture_cuda_graphs', return_value=capture):
        runner_module.ModelRunner.init_cuda_graphs(fake)
    assert fake.graph_time_usage == 0
    return fake


def restore(pool, p):
    return torch.stack([pool.initial_dense(lid, p) for lid in pool.layer_ids])


class RestoreTest(unittest.TestCase):
    def test_real_startup_off_on_hit_miss_four_states(self):
        for fulln in ('0','1'):
            for enabled in (0,1):
                for fresh in (False,True):
                    with self.subTest(fulln=fulln,enabled=enabled,fresh=fresh):
                        pool = make_pool(); old = make_pool(frozen)
                        driver = Driver()
                        with driver.use():
                            startup(pool,enabled,fulln=fulln)
                            p=plan(pool,fresh=fresh)
                            expected=restore(old,plan(old,fresh=fresh))
                            result=restore(pool,p)
                            self.assertEqual(sha(result),sha(expected))
                            calls = 0 if pool.prefill_restore_graph is None else pool.prefill_restore_graph.stats['replayed']
                            self.assertEqual(calls, int(enabled and not fresh))
                            self.assertEqual(pool.stats['densified'],old.stats['densified'])
                        TRACE.append(dict(enabled=enabled,fulln=fulln,fresh=fresh,sha=sha(result),replays=calls))

    def test_h48_real_shape_eager_bytes(self):
        pool=make_pool(h=48,k=128,rank=16)
        pool.count[:] = (torch.arange(pool.count.numel()).reshape_as(pool.count) % 33).to(torch.int32)
        with Driver().use():
            startup(pool,1)
            p=plan(pool)
            expected=torch.stack([pool._initial_dense_eager(l,p) for l in pool.layer_ids])
            result=restore(pool,p)
            self.assertEqual(sha(result),sha(expected))
            self.assertEqual(result.numel()*4,108*1024*1024)
            self.assertEqual(pool.prefill_restore_graph.stats['replayed'],1)

    def test_live_slots_and_private_stage_survive_replay(self):
        pool=make_pool()
        with Driver().use():
            startup(pool,1)
            p=plan(pool,slot=1); one=restore(pool,p); saved=one.clone()
            # A live writer updates factors (slot reuse) after the first plan.
            pool.U[:,2].mul_(2)
            q=plan(pool,slot=2); two=restore(pool,q)
            self.assertEqual(sha(one),sha(saved))
            self.assertNotEqual(p.stage.data_ptr(),q.stage.data_ptr())
            self.assertEqual(sha(two),sha(torch.stack([pool._initial_dense_eager(l,q) for l in pool.layer_ids])))
            # Recurrence mutation cannot change the captured output slab.
            p.stage.add_(7)
            three=restore(pool,plan(pool,slot=1))
            self.assertEqual(sha(three),sha(saved))

    def test_slot_join_precedes_replay_and_reads_live_factors(self):
        pool=make_pool(); driver=Driver()
        with driver.use():
            startup(pool,1)
            def join(slots,forward_local=False):
                driver.events.append('slot_join')
                self.assertTrue(forward_local)
                self.assertEqual(slots.tolist(),[1])
            with mock.patch.object(pool,'pside_join',side_effect=join):
                restore(pool,plan(pool))
            self.assertLess(driver.events.index('slot_join'),driver.events.index('replay'))

    def test_capture_other_stream_backing_precision_fallback(self):
        for failure in ('capture','stream','backing','precision'):
            pool=make_pool(); d=Driver()
            with d.use():
                startup(pool,1); restore(pool,plan(pool))
                if failure=='capture': d.capturing=True
                if failure=='stream': d.stream_id=11
                if failure=='backing': pool.U=pool.U.clone()
                previous=torch.backends.cuda.matmul.allow_tf32
                try:
                    if failure=='precision': torch.backends.cuda.matmul.allow_tf32=not previous
                    p=plan(pool); result=restore(pool,p)
                    expected=torch.stack([pool._initial_dense_eager(l,p) for l in pool.layer_ids])
                    self.assertEqual(sha(result),sha(expected))
                    self.assertEqual(pool.prefill_restore_graph.stats['replayed'],1)
                    self.assertIsNone(p.stage)
                finally: torch.backends.cuda.matmul.allow_tf32=previous

    def test_ring_multirow_partial_layers_keep_eager(self):
        pool=make_pool()
        with Driver().use():
            startup(pool,1)
            for p in (plan(pool,ring=True),plan(pool,rows=2),plan(pool)):
                if p.slots.numel()==1 and not p.n_ring_src:p.last_layer=20
                result=restore(pool,p)
                self.assertEqual(sha(result),sha(torch.stack([pool._initial_dense_eager(l,p) for l in pool.layer_ids])))
                self.assertIsNone(p.stage)
            self.assertEqual(pool.prefill_restore_graph.stats['replayed'],0)

    def test_restore_adds_no_host_value_reads(self):
        pool=make_pool()
        with Driver().use():
            startup(pool,1)
            with mock.patch.object(torch.Tensor,'cpu',side_effect=AssertionError('cpu read')), \
                 mock.patch.object(torch.Tensor,'tolist',side_effect=AssertionError('tolist read')), \
                 mock.patch.object(torch.Tensor,'item',side_effect=AssertionError('item read')):
                restore(pool,plan(pool))
            self.assertEqual(pool.prefill_restore_graph.stats['replayed'],1)

    def test_p_and_d_not_installed(self):
        for role in ('prefill','decode'):
            for enabled in (0,1):
                pool=make_pool()
                with Driver().use():startup(pool,enabled,role)
                self.assertIsNone(pool.prefill_restore_graph)

    def test_real_cpu_no_cuda_keeps_eager(self):
        pool=make_pool(); startup(pool,1)
        self.assertIsNone(pool.prefill_restore_graph)
        p=plan(pool)
        self.assertEqual(sha(restore(pool,p)),sha(torch.stack([pool._initial_dense_eager(l,p) for l in pool.layer_ids])))

    def test_prewarm_negative_pool_qualification(self):
        for change in ('dense','warm','partial','budget'):
            pool=make_pool()
            if change=='dense':pool.prefix_dense=torch.zeros(1)
            if change=='warm':pool.warm_v=torch.zeros(1)
            if change=='partial':pool.cfg.factored_prefix=False
            if change=='budget':pool._STAGE_MAX_BYTES=1
            with Driver().use():startup(pool,1)
            self.assertIsNone(pool.prefill_restore_graph)


if __name__=='__main__':
    torch.set_num_threads(2)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(RestoreTest)
    if os.environ.get('PFACTOR077_REVERSE')=='1':
        # With only the pool restore dispatch withdrawn, startup still captures
        # but real initial_dense never replays: this must fail the positive case.
        suite=unittest.TestSuite([RestoreTest('test_h48_real_shape_eager_bytes'),
                                 RestoreTest('test_real_startup_off_on_hit_miss_four_states')])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    record=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),
                errors=len(result.errors),skipped=len(result.skipped),cases=TRACE,
                driver='CPU CUDA driver shim; real production Python and torch arithmetic',
                cuda_bitwise='NOT TESTED',d2h_added=0)
    if os.environ.get('PFACTOR077_RECEIPT'):Path(os.environ['PFACTOR077_RECEIPT']).write_text(json.dumps(record,indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
