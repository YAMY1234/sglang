"""Candidate A: real dispatch/reader code with stub streams, CPU graph bodies.

No GPU result is asserted by these tests. The real Triton packed recurrence
runs in the CPU interpreter in the same-image gate; stream/capture order is
checked separately. Runtime NLL and needle remain mandatory admission gates.
"""
import ast
import copy
import logging
import os
from pathlib import Path
import types
import unittest

os.environ.setdefault("TRITON_INTERPRET", "1")
from test_gdn_tracked_factor_side import FakeCuda, Vector, extract, function, controller

BASE = Path(__file__).resolve().parents[2] / "python/sglang/srt"
SOURCE = BASE / "mem_cache/gdn_final_factor_deferred.py"
OWNER = BASE / "models/flash_next_duet/final_factor_deferred.py"
BACKEND = BASE / "layers/attention/linear/gdn_backend.py"
POOL = BASE / "mem_cache/gdn_factored_pool.py"


def make_controller():
    side, cuda = controller(deferred=True)
    pool = side.pool
    pool.ring_generation = 0
    pool.layer_ids, pool.layer_map = [3, 7], {3: 0, 7: 1}
    pool.invalidate_prefix_dense = lambda x: cuda.trace.append(("invalidate", x))
    tree = ast.parse(SOURCE.read_text())
    cls = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.ClassDef)))
    for node in ast.walk(cls):
        if isinstance(node, ast.FunctionDef):
            node.body = [s for s in node.body if not isinstance(s, ast.ImportFrom)]
    scope = dict(torch=cuda, logger=logging.getLogger("test"))
    from test_gdn_tracked_factor_side import SIDE
    extract([function(SIDE, "disjoint_destinations"), cls], scope)
    candidate = scope["FinalFactorDeferred"](pool, side)
    graph = lambda name: types.SimpleNamespace(replay=lambda: cuda.trace.append((name, cuda.current)))
    buffers = types.SimpleNamespace(bind=lambda *args: cuda.trace.append(("bind", cuda.current, args)))
    key = side.whole_graph.key(1, 1, "eager", "policy", False)
    for k in (0, 1):
        candidate.entries[(k, key)] = (buffers, graph("ring"), graph("T"), graph("F_SN"))
    return candidate, cuda


def prepare(candidate, **changes):
    plan = types.SimpleNamespace(final_ring_cpu=[0], final_slots_cpu=[1],
        next_layer=0, last_layer=1, pending=[], checkpoint_group=None, slots=Vector([1]))
    plan.__dict__.update(changes)
    meta = types.SimpleNamespace(has_mamba_track_mask=True, track_ssm_h_dst=Vector([2]),
        track_ssm_final_src=Vector([]), track_ssm_final_dst=Vector([]))
    batch = types.SimpleNamespace(req_pool_indices_cpu=Vector([9]))
    accepted = candidate.prepare(batch, plan, meta, eager="eager", policy="policy")
    return accepted, plan, meta


def stage(candidate):
    accepted, plan, meta = prepare(candidate)
    assert accepted
    controls = (meta.track_ssm_h_dst, meta.track_ssm_final_src, meta.track_ssm_final_dst)
    candidate.add(3, plan, object(), object(), *controls)
    candidate.add(7, plan, object(), object(), *controls)
    return plan


class StreamAndReaderTest(unittest.TestCase):
    def test_no_final_factor_until_dense_boundary_then_same_stream_after_T(self):
        candidate, cuda = make_controller()
        stage(candidate)
        names = [e[0] for e in cuda.trace]
        self.assertEqual(names, ["invalidate", "invalidate", "bind", "ring", "record",
                                 "wait", "T", "record"])
        self.assertIs(cuda.trace[3][1], cuda.main)
        cuda.trace.append(("dense_boundary", cuda.main))
        candidate.after_boundary()
        self.assertEqual([e[0] for e in cuda.trace][-5:],
                         ["dense_boundary", "record", "wait", "F_SN", "record"])
        self.assertIs(cuda.trace[-3][2], candidate.boundary_ready)
        self.assertIs(cuda.trace[-2][1], candidate.side.stream)
        self.assertTrue(candidate._prefill_side_pending)

    def test_first_factor_decode_and_all_reader_streams_wait_once(self):
        c, cuda = make_controller();stage(c);c.after_boundary();cuda.trace.clear()
        c.join();c.join()
        other = cuda.Stream()
        with cuda.stream(other):
            c.join();c.join()
        self.assertEqual(cuda.trace, [("wait", cuda.main, c.done), ("wait", other, c.done)])
        self.assertFalse(c._prefill_side_pending)

    def test_other_request_can_run_until_any_slot_reader_requires_join(self):
        c, cuda = make_controller();stage(c);c.after_boundary();cuda.trace.clear()
        ids = Vector([10]);ids.device = types.SimpleNamespace(type="cpu")
        c.join_for_batch(types.SimpleNamespace(req_pool_indices_cpu=ids))
        self.assertFalse(cuda.trace)
        ids.values = [9]
        c.join_for_batch(types.SimpleNamespace(req_pool_indices_cpu=ids))
        self.assertEqual(cuda.trace, [("wait", cuda.main, c.done)])

    def test_missing_host_mirror_fences_without_device_readback(self):
        c, cuda = make_controller();stage(c);c.after_boundary();cuda.trace.clear()
        c.join_for_batch(types.SimpleNamespace())
        self.assertEqual(cuda.trace, [("wait", cuda.main, c.done)])

    def test_early_factor_reader_is_error_not_silent_old_factors(self):
        c, cuda = make_controller();stage(c)
        with self.assertRaisesRegex(RuntimeError, "before the dense boundary"):
            c.join()
        c.abort()
        self.assertIsNone(c.staged)
        self.assertIsNone(c.active_plan)
        self.assertNotIn("F_SN", [r[0] for r in cuda.trace])

    def test_ring_full_and_capture_fallback_before_mutation(self):
        for reason in ("ring", "capture"):
            c, cuda = make_controller()
            cuda.capturing = reason == "capture"
            accepted, plan, _ = prepare(c, final_ring_cpu=[-1] if reason == "ring" else [0])
            self.assertFalse(accepted)
            self.assertFalse(cuda.trace)
            self.assertIsNone(c.active_plan)
            self.assertFalse(plan.pending)
            self.assertEqual(c.stats["fallback_"+reason], 1)

    def test_grid_aligned_prefix_keeps_S_N_minus_1(self):
        c, cuda = make_controller()
        _, plan, meta = prepare(c)
        c.abort();cuda.trace.clear()
        meta.track_ssm_final_src = Vector([1])
        meta.track_ssm_final_dst = Vector([3])
        accepted = c.prepare(types.SimpleNamespace(req_pool_indices_cpu=Vector([9])),
                             plan, meta, eager="eager", policy="policy")
        self.assertFalse(accepted)
        self.assertFalse(cuda.trace)
        self.assertEqual(c.stats["fallback_prefix_final_copy"], 1)

    def test_j5_next_bind_waits_F_and_two_commits_old_T(self):
        c, cuda = make_controller()
        for _ in range(2):
            stage(c);c.after_boundary()
        cuda.trace.clear();stage(c)
        names = [e[0] for e in cuda.trace]
        self.assertLess(names.index("wait"), names.index("bind"))
        waits = [e[2] for e in cuda.trace if e[0] == "wait" and e[1] is cuda.main]
        self.assertIn(c.done, waits)
        self.assertIn(c.side.set_done[0], waits)

    def test_capture_requires_pending_final_complete(self):
        c, cuda = make_controller();stage(c);c.after_boundary();cuda.capturing=True
        with self.assertRaisesRegex(RuntimeError, "must finish"):
            c.join()
        c.done.complete=True;c.join()
        self.assertFalse(c.recorded)

    def test_pool_common_fence_covers_final_even_live_only(self):
        c, cuda = make_controller();stage(c);c.after_boundary();cuda.trace.clear()
        scope={};extract([function(POOL, "pside_join")], scope)
        pool=types.SimpleNamespace(_final_factor_deferred=c, _pside_deferred_commit=None,
                                  _tracked_factor_side=c.side)
        scope["pside_join"](pool, tracked=False)
        self.assertEqual(cuda.trace, [("wait", cuda.main, c.done)])

    def test_decode_graph_metadata_fences_before_native_replay_metadata(self):
        events=[]
        class Base:
            def init_forward_metadata_out_graph(self, *args, **kw):events.append("metadata")
            def init_forward_metadata(self, *args):
                events.append("metadata");raise StopIteration()
        scope=dict(Base=Base)
        nodes=[function(BACKEND, n) for n in ("_join_deferred_final", "init_forward_metadata_out_graph", "init_forward_metadata")]
        cls=ast.ClassDef(name="GDN",bases=[ast.Name(id="Base",ctx=ast.Load())],
                         keywords=[],body=nodes,decorator_list=[])
        extract([cls],scope)
        backend=scope["GDN"]()
        backend._final_boundary_dense=False
        backend.factored=types.SimpleNamespace(_final_factor_deferred=types.SimpleNamespace(
            join_for_batch=lambda batch:events.append("join")))
        backend.init_forward_metadata_out_graph(object())
        self.assertEqual(events,["join","metadata"])
        events.clear()
        with self.assertRaises(StopIteration):backend.init_forward_metadata(object())
        self.assertEqual(events,["join","metadata"])
        events.clear();backend._final_boundary_dense=True
        backend.init_forward_metadata_out_graph(object())
        self.assertEqual(events,["metadata"])  # J1': no wait on its own future F.


class AdmissionTest(unittest.TestCase):
    def test_role_and_dependencies(self):
        scope={};extract([function(OWNER,"startup_rejection")],scope)
        owner=types.SimpleNamespace(fullstack=True,fullstack_v3_latent=False)
        args=types.SimpleNamespace(disaggregation_mode="null",is_embedding=False,pp_size=1,
            disable_overlap_schedule=True,enable_two_batch_overlap=False,
            enable_torch_compile=False,enable_linear_replayssm=False,disable_cuda_graph=False)
        runner=types.SimpleNamespace(server_args=args,is_draft_worker=False,
            spec_algorithm=types.SimpleNamespace(is_none=lambda:True),lora_manager=None)
        pool=types.SimpleNamespace(_k31_batch_graph=object(),
            _tracked_factor_side=types.SimpleNamespace(deferred=True),cfg=types.SimpleNamespace(strict_chunk=True),prefix_dense=None)
        reject=scope['startup_rejection']
        self.assertIsNone(reject(owner,runner,pool))
        for mode in ('prefill','decode'):
            args.disaggregation_mode=mode;self.assertEqual(reject(owner,runner,pool),'pd')
        args.disaggregation_mode='null'
        for field,value in [('pp_size',2),('is_embedding',True),('disable_overlap_schedule',False)]:
            old=getattr(args,field);setattr(args,field,value)
            self.assertIsNotNone(reject(owner,runner,pool));setattr(args,field,old)
        pool._tracked_factor_side.deferred=False
        self.assertEqual(reject(owner,runner,pool),'requires_k31_tracked_2b')

    def test_request_logprob_spec_capture_fallback(self):
        cuda=FakeCuda();scope={'torch':cuda};extract([function(SOURCE,'request_rejection')],scope)
        batch=types.SimpleNamespace(spec_info=None,return_logprob=False,batch_size=1,
            extend_seq_lens_cpu=[8192],twinstar_prompt_final=[True],
            req_pool_indices_cpu=types.SimpleNamespace(device=types.SimpleNamespace(type='cpu')),
            capture_hidden_mode=types.SimpleNamespace(is_full=lambda:False))
        owner=types.SimpleNamespace(_boundary_lens=lambda b:[1],state_audit_dir=None,fullstack_v3_latent=False)
        reject=scope['request_rejection'];self.assertIsNone(reject(owner,batch))
        batch.return_logprob=True;self.assertEqual(reject(owner,batch),'logprob')
        batch.return_logprob=False;batch.spec_info=object();self.assertEqual(reject(owner,batch),'spec')
        batch.spec_info=None;cuda.capturing=True;self.assertEqual(reject(owner,batch),'capture')

    def test_default_off_and_dense_kernel_arguments_match_stock(self):
        self.assertIn('SGLANG_GDN_FINAL_FACTOR_DEFERRED = EnvBool(False)', (BASE/'environ.py').read_text())
        stock=function(BACKEND,'forward_decode')
        new=function(SOURCE,'dense_boundary_decode')
        def packed(node):
            return next(n for n in ast.walk(node) if isinstance(n,ast.Call)
                        and isinstance(n.func,ast.Attribute) and n.func.attr=='packed_decode')
        a,b=packed(stock),packed(new)
        self.assertEqual({k.arg for k in a.keywords},{k.arg for k in b.keywords})
        for key in ('A_log','dt_bias','scale','num_v_heads','head_v_dim'):
            self.assertEqual(ast.dump(next(k.value for k in a.keywords if k.arg==key)),
                             ast.dump(next(k.value for k in b.keywords if k.arg==key)))


try:
    import torch
    import triton
    HAVE_TORCH=True
except ImportError:
    HAVE_TORCH=False


@unittest.skipUnless(HAVE_TORCH, 'same-image gate requires torch + triton; skips rejected')
class TensorParityTest(unittest.TestCase):
    def test_ring_factor_body_matches_direct_S_N_factor_bytes(self):
        from test_gdn_prefill_k31_batch_graph import _modules, _pool, _plan, LAYERS, HV, V, K
        import importlib
        torch.set_num_threads(1)
        fp,bg=_modules()
        module=importlib.import_module('sglang.srt.mem_cache.gdn_final_factor_deferred')
        gen=torch.Generator().manual_seed(101)
        states=[(torch.randn(1,HV,V,K,generator=gen),torch.randn(1,HV,V,K,generator=gen)) for _ in range(LAYERS)]
        pools=[]
        for deferred in (False,True):
            p=_pool(fp,11);plan=_plan(fp,p,1);plan.dense_required_after_commit.zero_()
            b=bg.BatchBuffers(p,1,1,include_tail=False)
            b.bind(plan,states,torch.tensor([2]),None,None)
            if deferred:
                before=[t.clone() for t in (p.a,p.U,p.W,p.count)]
                module.copy_exact_to_ring(b)
                for original,current in zip(before,(p.a,p.U,p.W,p.count)):
                    self.assertTrue(torch.equal(original,current))
                for old,ring in zip(states,p.dense_ring):self.assertTrue(torch.equal(old[0][0],ring[0]))
                # A deterministic dense update is already in S_N. Factorization
                # is compared to the same exact updated state on the main path.
                for ring in p.dense_ring:ring[0].add_(0.125)
                b.evaluate(fp.factorize_layers,branch='tracked')
                module.factorize_ring(b,fp.factorize_layers)
            else:
                final_states = [(normal + 0.125, tracked) for normal, tracked in states]
                b.bind(plan,final_states,torch.tensor([2]),None,None)
                b.evaluate(fp.factorize_layers)
            pools.append(p)
        # Both arms used the same exact baseline and the same dense update.
        for name in ('a','U','W','count','stale','dense_of','dense_required','prefix_valid'):
            self.assertTrue(torch.equal(getattr(pools[0],name),getattr(pools[1],name)),name)
        self.assertTrue(torch.isfinite(pools[1].U).all())
        self.assertTrue(torch.isfinite(pools[1].W).all())
        self.assertEqual(int(pools[1].count[0,0,0]),pools[1].cfg.r)


    def test_real_dense_recurrence_matches_P_kernel_state_and_output(self):
        from test_gdn_prefill_k31_batch_graph import _modules, _pool, HV, V, K
        import importlib
        import functools
        torch.set_num_threads(1)
        fp,_=_modules()
        module=importlib.import_module('sglang.srt.mem_cache.gdn_final_factor_deferred')
        # Import the native packed kernel, not a mathematical replacement.
        from sglang.kernels.ops.attention.fla.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
        scope=dict(torch=torch, fused_recurrent_gated_delta_rule_packed_decode=fused_recurrent_gated_delta_rule_packed_decode)
        extract([function(BASE/'layers/attention/linear/kernels/gdn_triton.py','packed_decode')],scope)
        native=functools.partial(scope['packed_decode'],None)
        p=_pool(fp,17);p.dense_of[4]=2
        gen=torch.Generator().manual_seed(137)
        p.dense_ring[0].normal_(generator=gen)
        reference=p.dense_ring[0].clone()
        mixed=torch.randn(1,(2+HV)*K,generator=gen).to(torch.bfloat16)
        a=torch.randn(1,HV,generator=gen).to(torch.bfloat16)
        b=torch.randn(1,HV,generator=gen).to(torch.bfloat16)
        layer=types.SimpleNamespace(layer_id=0,A_log=torch.randn(HV,generator=gen),
            dt_bias=torch.randn(HV,generator=gen),head_k_dim=K,num_v_heads=HV,head_v_dim=V)
        calls=[]
        def packed(**kw):calls.append(kw);return native(**kw)
        backend=types.SimpleNamespace(factored=p,kernel_dispatcher=types.SimpleNamespace(packed_decode=packed),
            _track_mamba_state_decode=lambda *args:None)
        result=module.dense_boundary_decode(backend,layer,None,mixed,a,b,None,None,torch.tensor([4],dtype=torch.int32))
        expected=native(mixed,a,b,A_log=layer.A_log,dt_bias=layer.dt_bias,scale=K**-0.5,
            ssm_states=reference,cache_indices=torch.tensor([2],dtype=torch.int32),num_v_heads=HV,head_v_dim=V)
        self.assertTrue(torch.equal(result,expected))
        self.assertTrue(torch.equal(p.dense_ring[0],reference))
        self.assertIs(calls[0]['mixed_qkv'],mixed)
        self.assertEqual(calls[0]['ssm_states'].dtype,torch.float32)
        self.assertEqual(calls[0]['cache_indices'].tolist(),[2])


if __name__ == '__main__':unittest.main()
