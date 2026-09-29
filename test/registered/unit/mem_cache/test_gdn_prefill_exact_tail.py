"""Exact model tails own their states; deferred wire publication is independent."""
import ast
from contextlib import ExitStack, nullcontext
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache import gdn_prefill_exact_tail as exact
from sglang.srt.mem_cache import gdn_prefill_batch_graph as batch_graph
from sglang.srt.mem_cache import gdn_factored_pool as native
from test_gdn_prefill_batch_graph import fake_pool, make_plan


def layer(lid, heads=2, width=16):
    return NS(layer_id=lid, num_q_heads=heads, num_v_heads=heads,
              head_k_dim=width, head_v_dim=width,
              q_dim=heads*width, k_dim=heads*width, v_dim=heads*width,
              A_log=torch.zeros(heads), dt_bias=torch.zeros(heads),
              conv_weights=torch.zeros(3*heads*width, 4))


def fixture(rows=1, layers=36, tails=36):
    pool = fake_pool(layers=layers, capacity=100)
    pool.cfg.strict_chunk = 1
    pool._exact_tail_layers = [layer(i) for i in range(tails)]
    pool._prefill_batch_graph = NS(warmed=True, run=Mock())
    rp = NS(req_generation=torch.zeros(32, dtype=torch.long), factored_gdn_pool=pool)
    batch = NS(req_pool_indices_cpu=torch.arange(rows), batch_size=rows,
               forward_mode=NS(is_extend=lambda: True, is_mixed=lambda: False),
               mamba_track_mask=torch.zeros(rows, dtype=torch.bool))
    return NS(pool=pool, rp=rp, batch=batch, plan=make_plan(pool, rows))


def mod(name, **attrs):
    result = ModuleType(name); result.__dict__.update(attrs); return result


class ExactTailTest(unittest.TestCase):
    def test_same_prefix_factorization_matches_old_layers_with_fixed_omega(self):
        torch.manual_seed(1257)
        torch.set_num_threads(2)
        for rows in (1, 3, 5, 8, 15, 16):
            c = fixture(rows, layers=36)
            bucket = next(b for b in (1, 2, 4, 8, 16) if rows <= b)
            states = [torch.randn(rows, 2, 16, 16) for _ in range(36)]
            tracked = [torch.randn(rows, 2, 16, 16).bfloat16() for _ in range(36)]
            track_slots = torch.arange(50, 50 + rows)
            with exact.ExactTailTransaction(c.pool, c.rp, c.batch) as transaction:
                for lid, state in enumerate(states):
                    transaction.add(lid, c.plan, state, tracked[lid], track_slots, None, None)
                for role in (0, 1):
                    owned = [s[role].float() for s in transaction.states]
                    padded = [torch.cat((s, s.new_zeros(bucket-rows, *s.shape[1:]))) for s in owned]
                    fused = native.factorize_layers(padded, c.pool.vbar, c.pool.cfg,
                                                    omega=c.pool.init_omega(bucket))
                    for lid, state in enumerate(padded):
                        old = native.factorize_layers([state], c.pool.vbar[lid:lid+1], c.pool.cfg,
                                                      omega=c.pool.init_omega(bucket))[0]
                        def dense(values):
                            a, u, w = values
                            return w.transpose(-1, -2) @ u + c.pool.vbar[lid][None, :, :, None]*a[:, :, None, :]
                        torch.testing.assert_close(dense(fused[lid])[:rows], dense(old)[:rows], atol=2e-4, rtol=2e-4)
            self.assertEqual(c.pool._prefill_batch_graph.run.call_count, 1)

    def test_dense_tail_cannot_overwrite_normal_or_tracked_and_retains_activations(self):
        c = fixture(layers=1, tails=1); tracked = torch.full((1, 2, 16, 16), 3.)
        source = torch.ones_like(tracked); original = source.clone()
        mixed = torch.randn(1, 96); gates = torch.randn(1, 2)
        backend = NS(_track_mamba_state_decode=Mock())
        def packed(*args, **kw):
            torch.testing.assert_close(kw['ssm_states'], original)
            kw['ssm_states'].fill_(99)
            return torch.ones(1, 1, 2, 16)
        kernel = mod('sglang.srt.layers.attention.linear.kernels.gdn_triton',
                     TritonGDNKernel=lambda: NS(packed_decode=packed))
        with patch.dict(sys.modules, {kernel.__name__: kernel}):
            with exact.ExactTailTransaction(c.pool, c.rp, c.batch) as transaction:
                transaction.add(0, c.plan, source, tracked, torch.tensor([50]), torch.tensor([1]), torch.tensor([80]))
                transaction.prefix_rows = torch.tensor([0])
                value = transaction.decode(backend, layer(0), c.batch, mixed, gates, gates,
                                           torch.empty(0), torch.empty(0), c.plan.slots)
                source.fill_(42); tracked.fill_(43); mixed.fill_(44)
                torch.testing.assert_close(transaction.states[0][0], original)
                self.assertTrue(torch.all(transaction.states[0][1] == 3))
                self.assertFalse(torch.all(transaction.tails[0][0] == 44))
                self.assertEqual(value.shape, (1, 1, 2, 16))

    def test_abort_generation_alias_and_missing_layers_do_not_publish(self):
        for failure in ('abort', 'generation', 'missing', 'controls', 'alias'):
            c = fixture(layers=2, tails=0)
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                with exact.ExactTailTransaction(c.pool, c.rp, c.batch) as transaction:
                    for lid in range(1 if failure == 'missing' else 2):
                        slots = c.plan.slots if failure == 'alias' else torch.tensor([50])
                        # Reuse the same control tensor as the real metadata does.
                        if lid == 0: tracked_slots = slots
                        transaction.add(lid, c.plan, torch.ones(1, 2, 16, 16),
                                        torch.zeros(1, 2, 16, 16), tracked_slots, None, None)
                    if failure == 'abort': raise RuntimeError('abort')
                    if failure == 'generation': c.rp.req_generation[0] += 1
                    if failure == 'controls': c.plan.slots.add_(1)
            self.assertFalse(c.pool._prefill_batch_graph.run.called)
            self.assertIsNone(c.pool._exact_tail_transaction)

    def test_slot_reset_rejected_and_later_generation_can_reuse_slot(self):
        pool = native.FactoredGDNPool.__new__(native.FactoredGDNPool)
        pool._exact_tail_transaction = object()
        with self.assertRaisesRegex(RuntimeError, 'recycle'):
            pool.reset_slots(torch.tensor([1]))
        c = fixture(layers=1, tails=0)
        for generation in range(2):
            c.rp.req_generation.fill_(generation); c.plan = make_plan(c.pool)
            with torch.inference_mode(), exact.ExactTailTransaction(c.pool, c.rp, c.batch) as transaction:
                transaction.add(0, c.plan, torch.full((1, 2, 16, 16), float(generation)), None, None, None, None)
            self.assertIsNone(c.pool._exact_tail_transaction)
        self.assertEqual(c.pool._prefill_batch_graph.run.call_count, 2)

    def test_partial_tail_rebind_clears_old_rows_and_keeps_tracked_separate(self):
        c = fixture(rows=3, layers=1, tails=1)
        buffers = batch_graph.BatchBuffers(c.pool, 8, 8)
        states = [(torch.ones(3, 2, 16, 16), torch.full((1, 2, 16, 16), 8.))]
        values = (torch.ones(3, 96), torch.ones(3, 2), torch.ones(3, 2), torch.tensor([1, 2, 3]))
        c.plan.exact_tail_inputs = {0: values}
        buffers.bind(c.plan, states, torch.tensor([50]), None, None)
        self.assertEqual(buffers.tail[0][3].tolist(), [1, 2, 3, -1, -1, -1, -1, -1])
        self.assertTrue(torch.all(buffers.normal[0][:3] == 1))
        self.assertTrue(torch.all(buffers.tracked[0][:1] == 8))
        c.plan.exact_tail_inputs = None
        c.pool.dense_ring = torch.zeros(1, 32, 2, 16, 16); c.pool.ring_generation += 1
        buffers.bind(c.plan, states, torch.tensor([51]), None, None)
        self.assertTrue(torch.all(buffers.tail[0][3] == -1))
        self.assertTrue(torch.all(buffers.tail[0][0] == 0))
        self.assertEqual(buffers.ring_pointers.tolist(), [x.data_ptr() for x in c.pool.dense_ring])

    def test_publish_order_snapshots_rank_r_before_wire_append(self):
        c = fixture(layers=1, tails=1); pool = c.pool
        plan = c.plan; plan.exact_tail_inputs = {0: (torch.ones(1, 96), torch.ones(1, 2),
                                                    torch.ones(1, 2), plan.slots)}
        buffers = batch_graph.BatchBuffers(pool, 1, 1)
        buffers.bind(plan, [(torch.ones(1, 2, 16, 16), torch.zeros(1, 2, 16, 16))],
                     torch.tensor([50]), torch.tensor([1]), torch.tensor([80]))
        def store(a, u, w, pa, pu, pw, count, stale, dense_of, slots, r, **kw):
            for idx, slot in enumerate(slots.tolist()):
                if slot >= 0:
                    pa[slot] = a[idx]; pu[slot] = u[idx]; pw[slot] = w[idx]; count[slot] = r
        def scatter(source, target, slots):
            for idx, slot in enumerate(slots.tolist()):
                if slot >= 0: target[:, slot] = source[:, idx]
        def append(*args, **kw):
            self.assertTrue(torch.all(pool.count[0, 80] == 8))
            self.assertTrue(torch.all(pool.count[0, 50] == 8))
            kw['fcount'][kw['ssm_state_indices'].long()] += 1
        class Publish:
            def __getitem__(self, grid): return lambda valid, slots, *a: valid.index_fill_(0, slots.clamp_min(0), 1)
        with patch('sglang.srt.layers.attention.linear.kernels.gdn_factored_io.store_factored', side_effect=store), \
             patch('sglang.srt.layers.attention.linear.kernels.gdn_factored.factored_packed_decode', side_effect=append), \
             patch.object(batch_graph, 'scatter_rows', side_effect=scatter), \
             patch.object(batch_graph, '_publish_valid', Publish()):
            buffers.evaluate(native.factorize_layers)
        self.assertTrue(torch.all(pool.count[0, 1] == 9))
        self.assertTrue(torch.all(pool.count[0, 80] == 8))
        self.assertTrue(torch.all(pool.count[0, 50] == 8))

    def test_native_two_arm_install_and_full_n_split_publish_once_before_handoff(self):
        for limit in (31, 48):
            c = fixture(); ids = [i for i in range(48) if i % 4 != 3]
            c.pool.layer_ids = ids; c.pool.layer_map = {lid: i for i, lid in enumerate(ids)}
            c.pool.batch_prefill = True
            layers = [layer(lid) for lid in ids if lid < limit]
            class NativeLayer:
                pass
            native_layers = []
            for value in layers:
                obj = NativeLayer(); obj.__dict__.update(vars(value)); native_layers.append(obj)
            body = NS(modules=lambda: iter(native_layers))
            plan = c.plan; plan.last_layer = 35
            prefix = NS(req_pool_indices_cpu=torch.tensor([0]), batch_size=1)
            boundary = NS(req_pool_indices_cpu=torch.tensor([0]), batch_size=1,
                          mamba_track_mask=torch.zeros(1, dtype=torch.bool))
            metadata = NS(factored_extend=plan, mamba_cache_indices=plan.slots)
            order = []
            backend = NS(factored=c.pool, forward_metadata=metadata)
            def prefix_forward(layer, batch, mixed, a, b, **kw):
                c.pool._exact_tail_transaction.add(layer.layer_id, plan, torch.ones(1, 2, 16, 16),
                                                   None, None, None, None)
                order.append(('prefix', layer.layer_id))
                return torch.ones(1, mixed.shape[0], 2, 16)
            backend.forward_extend = prefix_forward
            def decode(layer, batch, mixed, a, b, **kw):
                tx = c.pool._exact_tail_transaction
                tx.tails[layer.layer_id] = (mixed, a, b, plan.slots)
                order.append(('exact_tail', layer.layer_id))
                self.assertFalse(c.pool._prefill_batch_graph.run.called)
                return torch.ones(1, 1, 2, 16)
            backend.forward_decode = decode
            path = Path(exact.__file__).parents[1] / 'layers/attention/hybrid_linear_attn_backend.py'
            tree = ast.parse(path.read_text())
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                       and any(isinstance(m, ast.FunctionDef) and m.name == '_is_full_attn' for m in n.body))
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward_extend')
            future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
            namespace = {}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])),
                         str(path), 'exec'), namespace)
            hybrid = NS(_is_full_attn=lambda *a: False, linear_attn_backend=backend)
            def original(input_ids, positions, batch):
                # Actual installed external split entry, with all full-N layers retained.
                with external.split_boundary(backend, prefix, torch.tensor([0, 1]), boundary,
                        torch.tensor([2]), metadata, object(), split_layer_limit=limit):
                    for lid in ids:
                        namespace['forward_extend'](hybrid, layer=layer(lid), forward_batch=batch,
                            mixed_qkv=torch.zeros(3 if lid < limit else 2, 96),
                            a=torch.zeros(3 if lid < limit else 2, 2),
                            b=torch.zeros(3 if lid < limit else 2, 2))
                self.assertEqual(c.pool._prefill_batch_graph.run.call_count, 1)
                order.append('handoff')
                return 'result'
            owner = NS(forward=original, model=NS(model=body), pd_shallow_role='prefill' if limit == 31 else None)
            runner = NS(model=owner, req_to_token_pool=c.rp, server_args=NS(disaggregation_mode='prefill'))
            external = mod('twinstar_sgl.pd_shallow_gdn', split_boundary=Mock())
            modules = {'twinstar_sgl': mod('twinstar_sgl', pd_shallow_gdn=external), external.__name__: external,
                       'sglang.srt.layers.radix_linear_attention': mod('radix', RadixLinearAttention=NativeLayer)}
            with patch.dict(sys.modules, modules), patch.dict('os.environ', {exact.FLAG:'1',
                    'SGLANG_GDN_PREFILL_COMMIT_GRAPH':'1', 'TWINSTAR_PD_FACTOR_ONLY_TAIL':str(int(limit == 48))}, clear=True), \
                 patch('sglang.srt.runtime_context.get_schedule', return_value=NS(disable_overlap_schedule=True)):
                exact.install(runner)
                self.assertIs(external.split_boundary, exact.split_boundary)
                self.assertEqual(owner.forward(input_ids=torch.arange(3), positions=torch.arange(3),
                                               forward_batch=c.batch), 'result')
            self.assertEqual(len([x for x in order if isinstance(x, tuple) and x[0] == 'exact_tail']), 24 if limit == 31 else 36)
            self.assertEqual(order[-1], 'handoff')
            self.assertIs(backend.forward_extend, prefix_forward)

    def test_actual_backend_and_pool_entries_dispatch_to_transaction(self):
        path = Path(exact.__file__).parents[1] / 'layers/attention/linear/gdn_backend.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GDNAttnBackend')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_forward_decode_factored')
        namespace = {}
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])), str(path), 'exec'), namespace)
        transaction = NS(decode=Mock(return_value='exact-result'), add=Mock())
        c = fixture(layers=1, tails=1); c.pool._exact_tail_transaction = transaction
        backend = NS(factored=c.pool)
        args = [layer(0), c.batch, torch.zeros(1, 96), torch.zeros(1, 2), torch.zeros(1, 2),
                torch.empty(0), torch.empty(0), c.plan.slots]
        self.assertEqual(namespace['_forward_decode_factored'](backend, *args), 'exact-result')
        transaction.decode.assert_called_once_with(backend, *args)
        native.FactoredGDNPool.commit_extend_batched(c.pool, 0, c.plan, torch.ones(1, 2, 16, 16))
        transaction.add.assert_called_once()
        c.pool._prefill_batch_graph.run.assert_not_called()

    def test_model_runner_installs_before_prewarm_and_slot_zero_is_not_a_warm_target(self):
        path = Path(exact.__file__).parents[1] / 'model_executor/model_runner.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ModelRunner')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'init_cuda_graphs')
        imports = [n for n in tree.body if isinstance(n, ast.Import) and any(a.name == 'os' for a in n.names)]
        order=[]; capture=NS(eager_runner=None,prefill=NS(runner=None),decode=NS(runner=None),memory_usage=0,time_usage=0)
        ns={'capture_cuda_graphs':lambda **kw:capture}
        exec(compile(ast.Module(body=imports+[method],type_ignores=[]),str(path),'exec'), ns)
        runner=NS(model=object(),req_to_token_pool=NS(factored_gdn_pool=NS(prewarm_commit_graph=lambda:order.append('prewarm'))),
                  server_args=NS(disaggregation_mode='prefill'))
        with patch.dict('os.environ',{exact.FLAG:'1'},clear=True), patch.object(exact,'install',side_effect=lambda r:order.append('install')):
            ns['init_cuda_graphs'](runner)
        self.assertEqual(order,['install','prewarm'])
        c=fixture(layers=1,tails=1); graph=batch_graph.PrefillBatchGraph(); stream=Mock()
        seen=[]
        def evaluate(buffers, eager): seen.append(buffers.tail[0][3].tolist())
        with ExitStack() as stack:
            stack.enter_context(patch.object(batch_graph.BatchBuffers,'evaluate',evaluate))
            for name,value in (('current_stream',stream),('Stream',stream),('CUDAGraph',Mock()),
                               ('memory_allocated',0),('memory_reserved',0),('graph_pool_handle',object()),('synchronize',None)):
                stack.enter_context(patch.object(torch.cuda,name,return_value=value))
            for name in ('stream','graph'):
                stack.enter_context(patch.object(torch.cuda,name,side_effect=lambda *a,**kw:nullcontext()))
            graph.prewarm(c.pool,eager=native.factorize_layers,policy=())
        self.assertEqual(len(graph.entries),18)
        self.assertTrue(all(v == -1 for rows in seen for v in rows))


if __name__ == '__main__': unittest.main()
