"""CPU install/dispatch contracts; actual CUDA arithmetic is a separate gate.

The installers, model routing/slicing bodies, exact-tail transaction and send
validator are production code. Allocation and CUDA kernels are CPU stand-ins.
"""
import ast
import copy
import itertools
import json
import logging
import os
from pathlib import Path
import sys
from contextlib import ExitStack, contextmanager
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from sglang.test.test_utils import maybe_stub_sgl_kernel
maybe_stub_sgl_kernel()
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.disaggregation.state_handoff import FactorStateHandoff
from sglang.srt.models.flash_next_duet import pd_shallow_install as native_install
from sglang.srt.mem_cache import gdn_prefill_exact_tail as exact
from twinstar_sgl import pd_factor_only, pd_shallow_install as legacy_install, pd_shallow_gdn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'test/registered/unit/mem_cache'))
from test_gdn_prefill_exact_tail import fixture, layer
IDS = [i for i in range(48) if i % 4 != 3]


def model_class(native=True):
    path = ROOT / 'python/sglang/srt/models/flash_next_duet/model.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'Qwen4ExpForConditionalGeneration')
    names = {'forward', '_is_twinstar_prefill', '_emit_ids', '_sub_batch', '_decode_batch',
             '_pd_trunk_prefill_graph_runner', '_run_pd_trunk_prefill_graph',
             '_pd_trunk_prefill_graph_forward'}
    methods = [copy.deepcopy(n) for n in cls.body if getattr(n, 'name', '') in names]
    assert len(methods) == len(names)
    for method in methods:
        method.decorator_list = []
    scope = dict(torch=torch, os=os, copy=copy, itertools=itertools, ForwardMode=ForwardMode,
                 _dev=lambda values, dtype, device: torch.as_tensor(values, dtype=dtype, device=device),
                 get_is_capture_mode=lambda: False, logger=logging.getLogger(__name__),
                 LogitsProcessorOutput=NS)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *methods], type_ignores=[])), str(path), 'exec'), scope)

    class Model:
        def __init__(self, config, quant_config=None, prefix=''):
            self.config = config
            self.fullstack = config.twinstar['fullstack'] if native else None
            self.twinstar = config.twinstar if native else None
            self.fullstack_v3_latent = False
            self.fullstack_final = True
            self.n_layers = 48
            self.p_layer_ids = list(range(48))
            self.emitter_ids = list(range(31, 48)) if native else []
            self.emitters = {str(i): object() for i in self.emitter_ids}
            self.tp_rank = 0
            self.alloff_prefill_graph = False
            self.pd_trunk_prefill_graph = False
            self.n_pd_trunk_graph = 0
            self.dump_dir = self.state_audit_dir = None
            self.model = NS(model=NS(layers=[object() for _ in range(48)]), forward=lambda *a, **k: 'stock')
        def load_weights(self, weights): pass
        def prepare_before_cuda_graph_capture(self, runner): pass
        def _twinstar_prefill(self, *args):
            raise AssertionError('native model must not perform a second P/boundary split')
        def _boundary_graph(self, *args):
            raise AssertionError('factor-only has no emitter boundary forward')
    for name in names:
        setattr(Model, name, scope[name])
    return Model


def config(trim=False):
    fs = dict(duet_spec={}, prefill_layer_trim=trim, gdn_rank=8, gdn_every=8,
              prefill_saving_policy='kv-and-ssm', qsa_code='off')
    return NS(twinstar=dict(fullstack=fs, p_layers=list(range(31))),
              layers_block_type=['attention' if i % 4 == 3 else 'linear_attention' for i in range(48)])


@contextmanager
def installed(*, native=True, enabled=True, trim=False, role='prefill'):
    # Install the actual three-level chain. Restore the real global hooks after
    # each worker simulation, exactly as independent worker processes would.
    with TemporaryDirectory() as directory, ExitStack() as stack:
        env = dict(TWINSTAR_PD_FACTOR_ONLY_TAIL=str(int(enabled)),
                   PD_FACTOR_ONLY_CONTRACT=directory, TWINSTAR_FULLSTACK='1',
                   SGLANG_GDN_PREFILL_COMMIT_GRAPH='1')
        stack.enter_context(patch.dict(os.environ, env, clear=True))
        stack.enter_context(patch('sglang.srt.runtime_context.get_disagg',
            return_value=NS(disaggregation_mode=role, flashnext_pd_shallow_prefill=False)))
        for cls, name in ((ForwardBatch, 'init_new'),
                          (ScheduleBatch, '_mamba_radix_cache_v2_req_prepare_for_extend'),
                          (GDNAttnBackend, 'init_forward_metadata'),
                          (FactorStateHandoff, 'before_send'),
                          (pd_shallow_gdn, 'split_boundary')):
            stack.enter_context(patch.object(cls, name, cls.__dict__[name]))
        cls = model_class(native)
        stock = NS(Qwen4ExpForConditionalGeneration=object)
        if native:
            native_install.install(cls, stock, legacy_install)
        else:
            legacy_install.install(cls, stock)
        model = cls(config(trim))
        receipt = Path(directory) / 'rank0.json'
        yield model, (json.loads(receipt.read_text()) if receipt.exists() else None)


class NativeFactorTailTest(unittest.TestCase):
    def test_actual_install_chain_legacy_and_native_full_depth(self):
        for native in (False, True):
            with self.subTest(native=native), installed(native=native) as (model, receipt):
                self.assertEqual(receipt['checkpoint_extent'], 'N-1')
                self.assertEqual(receipt['tail'], 'native-recurrent-all-GDN')
                self.assertEqual(receipt['layers'], 48)
                self.assertFalse(model._is_twinstar_prefill(NS()))
                self.assertEqual(receipt['emitters'], 17 if native else 0)
                self.assertEqual(receipt['fullstack'], native)
                if native:
                    self.assertEqual(receipt['active_emitters'], 0)
                    self.assertEqual(model.p_layer_ids, list(range(48)))
                self.assertIsNotNone(ForwardBatch.init_new.__wrapped__)
                self.assertIsNotNone(GDNAttnBackend.init_forward_metadata.__wrapped__)
                self.assertIsNotNone(FactorStateHandoff.before_send.__wrapped__)

    def test_trim_and_non_P_still_rejected(self):
        with self.assertRaisesRegex(ValueError, 'native factor-only requires trim=0'):
            with installed(trim=True): pass
        for role in ('decode', 'null'):
            with self.assertRaisesRegex(ValueError, 'P-only'):
                with installed(role=role): pass

    def test_guard_checks_route_not_just_emitter_presence(self):
        for mutate in (lambda m: setattr(m, 'p_layer_ids', list(range(31))),
                       lambda m: setattr(m, 'pd_shallow_role', 'prefill'),
                       lambda m: m.fullstack.update(prefill_saving_policy='latent-and-ssm'),
                       lambda m: m.fullstack.update(gdn_rank=0),
                       lambda m: m.model.model.layers.pop(),
                       lambda m: m.config.layers_block_type.__setitem__(0, 'attention')):
            with installed() as (model, _):
                mutate(model)
                with self.assertRaisesRegex(ValueError, 'native factor-only requires'):
                    native_install.factor_only_contract(model)

    def test_flag_off_does_not_replace_native_predicate_or_contract(self):
        with installed(enabled=False) as (model, receipt):
            self.assertIsNone(receipt)
            self.assertEqual(model._is_twinstar_prefill.__name__, '_is_twinstar_prefill')
            model.fullstack['gdn_rank'] = 0
            self.assertFalse(model._is_twinstar_prefill(NS()))

    def test_native_scheduler_keeps_N_minus_one_with_either_cache_policy(self):
        # Native fullstack already uses prompt_p_extent. The adapter's proxy
        # must neither subtract twice nor alter the underlying request extent.
        from sglang.srt.managers import schedule_batch
        for p_only in (False, True):
            with installed(), ExitStack() as stack:
                stack.enter_context(patch.object(schedule_batch, 'mamba_cache_chunk_size', return_value=64))
                stack.enter_context(patch.object(schedule_batch, 'mamba_checkpoint_grid', return_value=64))
                stack.enter_context(patch.object(schedule_batch, 'get_exec', return_value=NS(
                    mamba=NS(enable_mamba_extra_buffer_lazy=False))))
                scheduler = ScheduleBatch.__new__(ScheduleBatch)
                scheduler.model_config = NS(hf_text_config=NS(mamba_chunk_size=64))
                scheduler.tree_cache = NS(page_size=64)
                scheduler.req_to_token_pool = NS(_prefill_prompt_only_state_cache=p_only,
                    get_mamba_ping_pong_other_idx=lambda i: 1-i)
                for end, total, prefix, expected in (
                    (8192, 8192, 0, 8128), (8193, 8193, 0, 8192),
                    (8192, 9000, 0, 8192), (8193, 8193, 8192, None),
                ):
                    extent = NS(start=prefix, end=end, length=end-prefix)
                    req = NS(extend_range=extent, origin_input_ids=[0]*total,
                        prefix_indices=[0]*prefix, mamba_branching_seqlen=None,
                        kv=NS(mamba_ping_pong_track_buffer=torch.tensor([2, 3]), mamba_next_track_idx=0))
                    track = scheduler._mamba_radix_cache_v2_req_prepare_for_extend(req)
                    self.assertEqual(req.extend_range.end, end)
                    self.assertIs(req.extend_range, extent)
                    self.assertEqual(track.track_mask, expected is not None)
                    if expected is not None:
                        self.assertEqual(req.kv.mamba_last_track_seqlen, expected)

    def run_cpu_publication(self, native, exact_on, trunk_on=False):
        """CPU routing equivalence, not a replacement for CUDA/NLL admission."""
        with installed(native=native) as (model, _), ExitStack() as stack:
            c = fixture()
            p = c.pool
            p.layer_ids = IDS
            p.layer_map = {lid: i for i, lid in enumerate(IDS)}
            p.layer_index = lambda lid: p.layer_map[lid]
            p.batch_prefill = True
            plan = c.plan
            states = torch.zeros(36, 2, 16, 16)
            tracked = torch.zeros_like(states)
            tail_calls = []
            class NativeLayer: pass
            layers = []
            for lid in IDS:
                item = NativeLayer(); item.__dict__.update(vars(layer(lid))); layers.append(item)
            model.model.model.modules = lambda: iter(layers)
            stack.enter_context(patch('sglang.srt.layers.radix_linear_attention.RadixLinearAttention', NativeLayer))
            stack.enter_context(patch('sglang.srt.runtime_context.get_schedule', return_value=NS(disable_overlap_schedule=True)))
            stack.enter_context(patch.dict(os.environ, {exact.FLAG: str(int(exact_on))}))
            def publish(li, dense, track):
                states[li].copy_(dense[0]); tracked[li].copy_(track[0])
                # Deterministic CPU factor-store stand-in, writing the actual
                # wire fields at both destinations rather than comparing only
                # pre-factor dense snapshots. CUDA factor arithmetic is gated
                # separately; this test catches row/layer/publication changes.
                for slot, value in ((1, dense[0]), (50, track[0])):
                    p.a[li, slot].copy_(value[:, 0])
                    p.U[li, slot].zero_()
                    p.W[li, slot].zero_()
                    p.U[li, slot, :, :8].copy_(value[:, :8])
                    p.W[li, slot, :, :8].copy_(torch.eye(16)[:8].expand(2, -1, -1))
                    p.count[li, slot] = 8
            def graph_run(pool, active_plan, values, *args, **kw):
                self.assertEqual(len(values), 36)
                for li, (dense, track) in enumerate(values):
                    publish(li, dense, track)
                    p.count[li, 1] += 1
            p._prefill_batch_graph.run.side_effect = graph_run
            track_slots = torch.tensor([50])
            class Backend:
                factored = p
                forward_metadata = None
                def init_forward_metadata(self, fb):
                    if fb.forward_mode.is_extend():
                        self.forward_metadata = NS(factored_extend=plan, mamba_cache_indices=plan.slots,
                            track_ssm_final_src=None, track_ssm_final_dst=None)
                    else:
                        self.forward_metadata = NS(mamba_cache_indices=plan.slots)
                def forward_extend(self, layer, batch, mixed_qkv, a, b, **kw):
                    active = self.forward_metadata.factored_extend
                    dense = torch.full((1, 2, 16, 16), float(layer.layer_id + 1))
                    tx = getattr(p, '_exact_tail_transaction', None)
                    if tx is None:
                        publish(p.layer_index(layer.layer_id), dense, dense + 2)
                        active.next_layer += 1
                    else:
                        tx.add(layer.layer_id, active, dense, dense + 2, track_slots, None, None)
                    return mixed_qkv[:, :32].reshape(1, -1, 2, 16) + 100
                def forward_decode(self, layer, batch, mixed_qkv, a, b, **kw):
                    tail_calls.append(layer.layer_id)
                    self_outer.assertFalse(batch.mamba_track_mask.any())
                    tx = getattr(p, '_exact_tail_transaction', None)
                    if tx is None:
                        p.count[p.layer_index(layer.layer_id), 1] += 1
                    else:
                        tx.tails[layer.layer_id] = (mixed_qkv.clone(), a.clone(), b.clone(), plan.slots.clone())
                    return mixed_qkv[:, :32].reshape(1, -1, 2, 16) + 200
            self_outer = self
            backend = Backend()
            from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
            from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import get_tc_piecewise_forward_context
            from sglang.srt.layers.radix_linear_attention import _unified_linear_attention_with_output_impl
            hybrid = NS(linear_attn_backend=backend, forward=lambda **kw: backend.forward_extend(**kw))
            stack.enter_context(forward_context(ForwardContext(attn_backend=hybrid)))
            break_calls = []
            def core(ids, positions, fb, **kw):
                values = []
                for obj in layers:
                    mixed = ids[:, None].float().expand(-1, 96).clone()
                    a = b = torch.zeros(len(ids), 2)
                    if get_tc_piecewise_forward_context() is not None:
                        # This is the production GDN eager-break body, under
                        # the production runner's two real forward contexts.
                        # It must dynamically reach the active split wrapper.
                        value = torch.empty(1, len(ids), 2, 16)
                        _unified_linear_attention_with_output_impl(mixed, a, b, value, obj.layer_id)
                        break_calls.append(obj.layer_id)
                        self.assertEqual(fb.twinstar_prompt_final, [True])
                        self.assertTrue(fb.pd_factor_only_full_batch)
                        self.assertEqual(fb.req_pool_indices_cpu.tolist(), [0])
                    else:
                        value = backend.forward_extend(layer=obj, forward_batch=fb,
                            mixed_qkv=mixed, a=a, b=b)
                    values.append(value.squeeze(0))
                return torch.stack(values, dim=1)
            model.model.forward = core
            model.model.model.hyper_connection_mixer = NS(mix=lambda value: (value, None))
            model.model.logits_processor = lambda ids, hidden, *args: NS(next_token_logits=hidden)
            model.model.lm_head = None
            from test_flash_next_pd_trunk_prefill_graph import cpu_runner
            model.pd_trunk_prefill_graph = trunk_on
            model._p_trunk = lambda fb: (core(fb.input_ids, fb.positions, fb), None)
            by_id = {obj.layer_id: obj for obj in layers}
            trunk = cpu_runner(model, hybrid, attention_layers=[by_id.get(i) for i in range(48)], padded_tokens=4)
            model._prefill_runners = dict(trunk=trunk)
            exact.install(NS(model=model, req_to_token_pool=c.rp, server_args=NS(disaggregation_mode='prefill')))
            ids = torch.arange(3)
            fb = NS(batch_size=1, forward_mode=ForwardMode.EXTEND, spec_info=None,
                twinstar_prompt_final=[True], pd_factor_only_full_batch=True, input_ids=ids, positions=ids,
                req_pool_indices=torch.tensor([0]), req_pool_indices_cpu=torch.tensor([0]),
                seq_lens=torch.tensor([3]), seq_lens_cpu=torch.tensor([3]), orig_seq_lens=None,
                out_cache_loc=ids+100, extend_num_tokens=3, extend_seq_lens=torch.tensor([3]),
                extend_seq_lens_cpu=[3], extend_prefix_lens=torch.tensor([0]), extend_prefix_lens_cpu=[0],
                extend_start_loc=torch.tensor([0]), extend_logprob_start_lens_cpu=None,
                mamba_track_mask=torch.zeros(1, dtype=torch.bool), mamba_track_indices=torch.tensor([50]),
                mamba_track_seqlens=torch.tensor([-1]))
            output = model.forward(ids, ids, fb)
            if trunk_on:
                output = output.next_token_logits
            self.assertEqual(trunk.run_count, int(trunk_on))
            self.assertEqual(model.n_pd_trunk_graph, int(trunk_on))
            self.assertEqual(break_calls, IDS if trunk_on else [])
            self.assertEqual(tail_calls, IDS)
            self.assertTrue(torch.all(p.count[:, 1] == 9))
            self.assertEqual(p._prefill_batch_graph.run.call_count, int(exact_on))
            req = NS(kv=NS(mamba_pool_idx=torch.tensor([1])))
            FactorStateHandoff(p).before_send(req)
            self.assertEqual(req.factored_prefill_boundary_steps, 1)
            self.assertTrue(torch.all(p.count[:, 1] == 9))
            return (output, states, tracked,
                    *(t[:, [1, 50]].clone() for t in (p.a, p.U, p.W, p.count)))

    def test_cpu_eager_and_exact_installed_output_publication_bitwise(self):
        expected = self.run_cpu_publication(False, False)
        for native, exact_on, trunk_on in ((True, False, False), (False, True, False),
                                            (True, True, False), (True, False, True),
                                            (True, True, True)):
            with self.subTest(native=native, exact=exact_on, trunk=trunk_on):
                actual = self.run_cpu_publication(native, exact_on, trunk_on)
                for left, right in zip(expected, actual):
                    self.assertTrue(torch.equal(left, right))


if __name__ == '__main__':
    unittest.main()
