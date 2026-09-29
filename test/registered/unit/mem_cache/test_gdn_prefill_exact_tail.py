"""Exact model tails own their states; deferred wire publication is independent."""
import ast
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
import sys
from types import MethodType, ModuleType, SimpleNamespace as NS
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
                        torch.tensor([2]), metadata, NS(mamba_cache_indices=plan.slots), split_layer_limit=limit):
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
                self.assertIsNot(external.split_boundary, exact.split_boundary)
                self.assertEqual(owner.forward(input_ids=torch.arange(3), positions=torch.arange(3),
                                               forward_batch=c.batch), 'result')
            self.assertEqual(len([x for x in order if isinstance(x, tuple) and x[0] == 'exact_tail']), 24 if limit == 31 else 36)
            self.assertEqual(order[-1], 'handoff')
            self.assertIs(backend.forward_extend, prefix_forward)

    def check_guard_fallback(self, rows, mixed=False, tbo=False, normalized_mixed=False, split=None):
        for limit in (31, 48):
            with self.subTest(limit=limit, rows=rows, mixed=mixed, tbo=tbo):
                c = fixture(rows=rows)
                lids = [i for i in range(48) if i % 4 != 3]
                c.pool.layer_ids = lids
                c.pool.layer_map = {lid: i for i, lid in enumerate(lids)}
                c.pool.batch_prefill = True
                c.batch.forward_mode.is_mixed = lambda: mixed
                c.batch.can_run_tbo = tbo
                c.batch._pfactor_legacy_mixed = normalized_mixed
                c.batch.tbo_split_seq_index = split
                class NativeLayer:
                    pass
                layers = []
                for lid in lids:
                    if lid < limit:
                        obj = NativeLayer()
                        obj.__dict__.update(vars(layer(lid)))
                        layers.append(obj)
                seen = []
                backend = NS(factored=c.pool)
                @contextmanager
                def layerwise(backend_arg, *args, **kwargs):
                    self.assertIs(backend_arg, backend)
                    self.assertIsNone(getattr(c.pool, '_exact_tail_transaction', None))
                    self.assertEqual(kwargs, dict(split_layer_limit=limit))
                    seen.append('layerwise')
                    yield
                external = mod('twinstar_sgl.pd_shallow_gdn', split_boundary=layerwise)
                def original(input_ids, positions, batch, *, offset):
                    self.assertIs(batch, c.batch)
                    with external.split_boundary(backend, split_layer_limit=limit):
                        return input_ids + positions + offset
                owner = NS(forward=original, model=NS(model=NS(modules=lambda: iter(layers))),
                           pd_shallow_role='prefill' if limit == 31 else None)
                runner = NS(model=owner, req_to_token_pool=c.rp,
                            server_args=NS(disaggregation_mode='prefill'))
                modules = {'twinstar_sgl': mod('twinstar_sgl', pd_shallow_gdn=external),
                           external.__name__: external,
                           'sglang.srt.layers.radix_linear_attention': mod('radix', RadixLinearAttention=NativeLayer)}
                ids, positions = torch.arange(rows), torch.arange(rows) * 2
                expected = original(ids, positions, c.batch, offset=7)
                seen.clear()
                with patch.dict(sys.modules, modules), patch.dict('os.environ', {exact.FLAG: '1',
                        'SGLANG_GDN_PREFILL_COMMIT_GRAPH': '1',
                        'TWINSTAR_PD_FACTOR_ONLY_TAIL': str(int(limit == 48))}, clear=True), \
                     patch('sglang.srt.runtime_context.get_schedule', return_value=NS(disable_overlap_schedule=True)), \
                     patch.object(exact, 'ExactTailTransaction', side_effect=AssertionError('fallback opened transaction')), \
                     patch.object(exact.logger, 'warning') as warning:
                    exact.install(runner)
                    result = owner.forward(input_ids=ids, positions=positions, forward_batch=c.batch, offset=7)
                    torch.testing.assert_close(result, expected)
                    warning.assert_called_once()
                self.assertEqual(seen, ['layerwise'])
                self.assertEqual(owner._exact_tail_fallbacks, 1)
                c.pool._prefill_batch_graph.run.assert_not_called()
                self.assertIsNone(getattr(c.pool, '_exact_tail_transaction', None))

    def test_batch32_extend_falls_back_to_layerwise(self):
        self.check_guard_fallback(32)

    def test_mixed_extend_decode_falls_back_to_layerwise(self):
        self.check_guard_fallback(2, mixed=True)

    def test_tbo_falls_back_to_layerwise(self):
        self.check_guard_fallback(8, tbo=True)

    def test_normalized_mixed_stays_layerwise_without_a_transaction(self):
        self.check_guard_fallback(2, normalized_mixed=True)

    def test_tbo_split_marker_stays_layerwise_before_can_run_is_set(self):
        self.check_guard_fallback(8, split=0)

    def empty_fixture(self, limit=48, rows=1, history="fresh"):
        c = fixture(rows=rows)
        ids = [i for i in range(48) if i % 4 != 3]
        p = c.pool
        p.layer_ids = ids
        p.layer_map = {lid: i for i, lid in enumerate(ids)}
        p._exact_tail_layers = [layer(lid) for lid in ids if lid < limit]
        p.device = p.a.device
        p.ring_owner, p.ring_lru = [-1] * 16, list(range(16))
        p.stats = dict(densified=0, extends=0, rows=0, ring_src=0, ring_miss=0)
        p._initial_warmed = False
        for name in ("initial_dense", "_initial_dense_eager", "plan_extend", "invalidate_prefix_dense"):
            setattr(p, name, MethodType(getattr(native.FactoredGDNPool, name), p))
        p.count.fill_(p.cfg.r)
        p.prefix_valid.fill_(1)
        c.batch.extend_seq_lens_cpu = [1] * rows
        c.batch.extend_prefix_lens_cpu = [0 if history == "fresh" else 64] * rows
        c.batch.twinstar_prompt_final = [True] * rows
        if history == "fresh":
            p.a.zero_(); p.U.zero_(); p.W.zero_(); p.prefix_valid.zero_()
        elif history == "ring":
            for i, slot in enumerate(c.plan.slots.tolist()):
                p.ring_owner[i] = slot; p.dense_of[slot] = i
                p.dense_required[slot] = 1; p.dense_ring[:, i].fill_(3 + i)
        return c

    def test_empty_health_and_cache_hit_use_native_initial_state_without_retruncating(self):
        for limit in (31, 48):
            for history in ("fresh", "cached", "ring"):
                with self.subTest(limit=limit, history=history), patch.dict('os.environ', {}, clear=True):
                    c = self.empty_fixture(limit, history=history)
                    p = c.pool
                    prefix = NS(batch_size=0, req_pool_indices_cpu=torch.empty(0, dtype=torch.long))
                    tail = c.batch
                    metadata = NS(mamba_cache_indices=c.plan.slots)
                    # Compare against the real native plan+initial-state functions on an independent pool.
                    reference = self.empty_fixture(limit, history=history)
                    rp = reference.pool
                    rp.vbar.copy_(p.vbar)
                    plan = rp.plan_extend(reference.plan.slots, [0],
                        prefix_lens=reference.batch.extend_prefix_lens_cpu, prompt_final=[True])
                    expected = [rp.initial_dense(lid, plan) for lid in rp.layer_ids]
                    saved = tuple(t.clone() for t in (p.a, p.U, p.W, p.count))
                    owners = list(p.ring_owner)
                    wire = []
                    backend = NS(factored=p, forward_metadata='original', forward_extend=Mock(),
                                 _track_mamba_state_decode=Mock())
                    def dense(mixed, a, b, **kw):
                        idx = len(wire_dense)
                        torch.testing.assert_close(kw['ssm_states'], expected[p.layer_map[p._exact_tail_layers[idx].layer_id]])
                        wire_dense.append(idx)
                        return mixed.new_full((1, mixed.shape[0], 2, 16), 17.)
                    def append(mixed, a, b, **kw):
                        self.assertFalse(kw['truncate'])
                        self.assertEqual(kw['ssm_state_indices'].tolist(), [1])
                        kw['fcount'][kw['ssm_state_indices']] += 1
                        kw['stale'][kw['ssm_state_indices']] = 1
                        wire.append(kw['fa'].data_ptr())
                        return mixed.new_full((1, 1, 2, 16), -99.)
                    def decode(obj, batch, mixed, a, b, **kw):
                        return p._exact_tail_transaction.decode(backend, obj, batch, mixed, a, b,
                            torch.empty(0), torch.empty(0), metadata.mamba_cache_indices)
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
                    kernel = mod('sglang.srt.layers.attention.linear.kernels.gdn_triton',
                                 TritonGDNKernel=lambda: NS(packed_decode=dense))
                    wire_dense = []
                    with patch.dict(sys.modules, {kernel.__name__: kernel}), \
                         patch('sglang.srt.layers.attention.linear.kernels.gdn_factored.factored_packed_decode', side_effect=append):
                        with exact.ExactTailTransaction(p, c.rp, c.batch):
                            with exact.split_boundary(backend, prefix, torch.empty(0, dtype=torch.long), tail,
                                    torch.tensor([0]), None, metadata, split_layer_limit=limit):
                                for obj in p._exact_tail_layers:
                                    result = namespace['forward_extend'](hybrid, layer=obj, forward_batch=c.batch,
                                        mixed_qkv=torch.ones(1, 96), a=torch.ones(1, 2), b=torch.ones(1, 2))
                                    self.assertTrue(torch.all(result == 17))
                                    self.assertFalse(wire)
                    self.assertEqual(len(wire), len(p._exact_tail_layers))
                    self.assertEqual(p.ring_owner, owners)
                    for now, before in zip((p.a, p.U, p.W), saved):
                        torch.testing.assert_close(now, before, rtol=0, atol=0)
                    recurrent = len(p._exact_tail_layers)
                    torch.testing.assert_close(p.count[recurrent:], saved[3][recurrent:], rtol=0, atol=0)
                    self.assertTrue(torch.all(p.count[:recurrent, 1] == p.cfg.r + 1))
                    self.assertEqual(p.dense_required[1].item(), 0)
                    p._prefill_batch_graph.run.assert_not_called()
                    backend.forward_extend.assert_not_called()
                    self.assertEqual(backend.forward_metadata, 'original')

    def test_mixed_empty_rows_preserve_tracked_and_filter_only_new_prefix_graph_tails(self):
        c = self.empty_fixture(rows=2, history='cached')
        p = c.pool
        c.batch.extend_seq_lens_cpu = [3, 1]
        prefix = NS(batch_size=1, req_pool_indices_cpu=torch.tensor([0]))
        boundary = NS(req_pool_indices_cpu=torch.tensor([0, 1]), mamba_track_mask=None)
        metadata = NS(mamba_cache_indices=torch.tensor([1, 2]))
        plan = make_plan(p, 1)
        track_slots = torch.tensor([50])
        normal = torch.full((1, 2, 16, 16), 7.)
        tracked = torch.full_like(normal, 19.)
        backend = NS(_track_mamba_state_decode=Mock())
        seen = []
        def packed(mixed, a, b, **kw):
            torch.testing.assert_close(kw['ssm_states'][0], normal[0])
            seen.append(kw['ssm_states'][1].clone())
            return mixed.new_ones((1, 2, 2, 16))
        def graph(pool, got_plan, states, slots, *args, **kwargs):
            self.assertIs(got_plan, plan)
            self.assertIs(slots, track_slots)
            for state, checkpoint in states:
                torch.testing.assert_close(state, normal)
                torch.testing.assert_close(checkpoint, tracked)
            self.assertTrue(all(values[-1].tolist() == [1] for values in got_plan.exact_tail_inputs.values()))
        p._prefill_batch_graph.run.side_effect = graph
        kernel = mod('sglang.srt.layers.attention.linear.kernels.gdn_triton',
                     TritonGDNKernel=lambda: NS(packed_decode=packed))
        with patch.dict('os.environ', {}, clear=True), patch.dict(sys.modules, {kernel.__name__: kernel}), \
             patch('sglang.srt.layers.attention.linear.kernels.gdn_factored.factored_packed_decode') as append:
            with exact.ExactTailTransaction(p, c.rp, c.batch) as tx:
                tx.prepare_split([0], boundary, metadata)
                for lid in p.layer_ids:
                    native.FactoredGDNPool.commit_extend_batched(p, lid, plan, normal, tracked, track_slots)
                    tx.decode(backend, layer(lid), c.batch, torch.ones(2, 96), torch.ones(2, 2),
                              torch.ones(2, 2), torch.empty(0), torch.empty(0), metadata.mamba_cache_indices)
            self.assertEqual(append.call_count, 36)
            self.assertTrue(all(call.kwargs['ssm_state_indices'].tolist() == [2] for call in append.call_args_list))
        self.assertEqual(len(seen), 36)
        p._prefill_batch_graph.run.assert_called_once()

    def test_empty_generation_control_and_missing_checkpoint_fail_before_wire(self):
        for failure in ('generation', 'control', 'missing', 'nonfinal'):
            with self.subTest(failure=failure), patch.dict('os.environ', {}, clear=True):
                c = self.empty_fixture(history='cached')
                boundary = NS(req_pool_indices_cpu=torch.tensor([0]), mamba_track_mask=None)
                metadata = NS(mamba_cache_indices=c.plan.slots)
                if failure == 'missing': c.pool.prefix_valid.zero_()
                if failure == 'nonfinal': c.batch.twinstar_prompt_final = [False]
                with patch('sglang.srt.layers.attention.linear.kernels.gdn_factored.factored_packed_decode') as append:
                    with self.assertRaises(RuntimeError):
                        with exact.ExactTailTransaction(c.pool, c.rp, c.batch) as tx:
                            tx.prepare_split([], boundary, metadata)
                            for obj in c.pool._exact_tail_layers:
                                tx.tails[obj.layer_id] = (torch.ones(1, 96), torch.ones(1, 2), torch.ones(1, 2), c.plan.slots.clone())
                            if failure == 'generation': c.rp.req_generation[0] += 1
                            if failure == 'control': metadata.mamba_cache_indices.add_(1)
                    append.assert_not_called()
                self.assertIsNone(c.pool._exact_tail_transaction)

    def test_nonfinal_first_chunk_without_tracked_keeps_all_layer_publication(self):
        c = self.empty_fixture()
        c.batch.extend_seq_lens_cpu = [4]
        c.batch.twinstar_prompt_final = [False]
        with exact.ExactTailTransaction(c.pool, c.rp, c.batch) as tx:
            for lid in c.pool.layer_ids:
                native.FactoredGDNPool.commit_extend_batched(c.pool, lid, c.plan,
                                                            torch.ones(1, 2, 16, 16))
        c.pool._prefill_batch_graph.run.assert_called_once()
        self.assertIsNone(c.pool._prefill_batch_graph.run.call_args.args[3])
        self.assertIsNone(c.pool._exact_tail_transaction)

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
