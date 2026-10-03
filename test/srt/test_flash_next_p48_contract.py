"""Real PD install/prewarm dispatch and native checkpoint policy on CPU.

CUDA graph allocation/kernels are replaced, not the installers or collector.
"""

import ast
import copy
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from test_flash_next_native_factor_tail import (
    installed,
    fixture,
    layer,
    IDS,
    ForwardBatch,
    ScheduleBatch,
    ForwardMode,
    torch,
)
from sglang.srt.mem_cache import gdn_prefill_agg_contract as agg
from sglang.srt.mem_cache import gdn_prefill_batch_graph as batch_graph
from sglang.srt.mem_cache import gdn_factored_pool as native
from sglang.srt.disaggregation.state_handoff import FactorStateHandoff

ROOT = Path(__file__).resolve().parents[2]


def init_graphs(runner, capture):
    path = ROOT / "python/sglang/srt/model_executor/model_runner.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner"
    )
    method = copy.deepcopy(
        next(n for n in cls.body if getattr(n, "name", "") == "init_cuda_graphs")
    )
    from sglang.srt.environ import envs

    scope = dict(os=os, envs=envs, capture_cuda_graphs=capture)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            str(path),
            "exec",
        ),
        scope,
    )
    scope["init_cuda_graphs"](runner)


@contextmanager
def worker(*, native_model=True, flag=True):
    with installed(native=native_model) as (owner, receipt), ExitStack() as stack:
        stack.enter_context(
            patch.object(
                ForwardBatch, "_pfactor_agg_contract_installed", False, create=True
            )
        )
        stack.enter_context(
            patch.dict(
                os.environ,
                {"SGLANG_GDN_PREFILL_EXACT_TAIL_BATCH": "1", agg.FLAG: str(int(flag))},
            )
        )
        stack.enter_context(
            patch(
                "sglang.srt.runtime_context.get_schedule",
                return_value=NS(disable_overlap_schedule=True),
            )
        )
        c = fixture()
        p = c.pool
        p.layer_ids = IDS
        p.layer_map = {lid: i for i, lid in enumerate(IDS)}
        p.layer_index = lambda lid: p.layer_map[lid]
        p.batch_prefill = True
        p._prefill_batch_graph.shared = {}

        class NativeLayer:
            pass

        layers = []
        for lid in IDS:
            obj = NativeLayer()
            obj.__dict__.update(vars(layer(lid)))
            layers.append(obj)
        owner.model.model.modules = lambda: iter(layers)
        stack.enter_context(
            patch(
                "sglang.srt.layers.radix_linear_attention.RadixLinearAttention",
                NativeLayer,
            )
        )
        events = []

        def commit_prewarm():
            if flag:
                assert owner._exact_tail_installed and owner._pfactor_agg_installed
            events.append("commit-prewarm")

        p.prewarm_commit_graph = commit_prewarm

        def prewarm(graph, pool, **kwargs):
            assert graph.include_tail is False
            assert graph.shared is p._prefill_batch_graph.shared
            graph.warmed = True
            events.append("full-N-prewarm")

        stack.enter_context(
            patch.object(batch_graph.PrefillBatchGraph, "prewarm", prewarm)
        )
        capture = NS(
            eager_runner=object(),
            prefill=NS(runner=None),
            decode=NS(runner=None),
            memory_usage=0,
            time_usage=0,
        )

        def capture_graphs(**kwargs):
            events.append("framework-capture")
            return capture

        runner = NS(
            model=owner,
            req_to_token_pool=c.rp,
            server_args=NS(disaggregation_mode="prefill"),
        )
        yield NS(
            owner=owner,
            pool=p,
            plan=c.plan,
            runner=runner,
            events=events,
            capture=capture_graphs,
            stack=stack,
            receipt=receipt,
        )


class NativeP48ContractTest(unittest.TestCase):
    def test_native_and_legacy_real_init_chain_and_prewarm(self):
        for native_model in (False, True):
            with (
                self.subTest(native=native_model),
                worker(native_model=native_model) as w,
            ):
                init_graphs(w.runner, w.capture)
                self.assertTrue(w.owner._exact_tail_installed)
                self.assertTrue(w.owner._pfactor_agg_installed)
                self.assertEqual([x.layer_id for x in w.pool._exact_tail_layers], IDS)
                self.assertEqual(
                    w.events, ["commit-prewarm", "full-N-prewarm", "framework-capture"]
                )
                self.assertFalse(w.pool._agg_prefill_graph.include_tail)

    def test_trim_or_active_emitter_rejected_at_contract_gate(self):
        for mutate in (
            lambda m: m.fullstack.update(prefill_layer_trim=True),
            lambda m: setattr(m, "_emit_ids", lambda: [31]),
            lambda m: m.fullstack.update(gdn_rank=0),
        ):
            with worker() as w:
                mutate(w.owner)
                with self.assertRaisesRegex(ValueError, "native factor-only requires"):
                    init_graphs(w.runner, w.capture)
                self.assertEqual(w.events, [])
                self.assertFalse(getattr(w.owner, "_pfactor_agg_installed", False))

    def test_contract_off_and_AGG_gate_remain_unchanged(self):
        with worker(flag=False) as w:
            init_graphs(w.runner, w.capture)
            self.assertTrue(w.owner._exact_tail_installed)
            self.assertFalse(getattr(w.owner, "_pfactor_agg_installed", False))
            self.assertEqual(w.events, ["commit-prewarm", "framework-capture"])
        with worker() as w:
            w.runner.server_args.disaggregation_mode = "null"
            init_graphs(w.runner, w.capture)
            self.assertFalse(getattr(w.owner, "_pfactor_agg_installed", False))
            self.assertFalse(getattr(w.owner, "_exact_tail_installed", False))
            self.assertEqual(w.events, ["framework-capture"])

    def test_PD_still_requires_exact_tail_fallback(self):
        with worker() as w:
            with self.assertRaisesRegex(ValueError, "strict k31 P48"):
                agg.install(w.runner)

    def test_native_scheduler_full_N_and_fallback_checkpoint_depth(self):
        from sglang.srt.managers import schedule_batch

        for flag, mixed, rows, tokens, expected in (
            (True, False, 1, 8192, 8192),
            (False, False, 1, 8192, 8128),
            (True, True, 1, 8192, 8128),
            (True, False, 17, 8192, 8128),
            (True, False, 1, 1, None),
        ):
            with (
                self.subTest(flag=flag, mixed=mixed, rows=rows, tokens=tokens),
                worker(flag=flag) as w,
            ):
                init_graphs(w.runner, w.capture)
                w.stack.enter_context(
                    patch.object(
                        schedule_batch, "mamba_cache_chunk_size", return_value=64
                    )
                )
                w.stack.enter_context(
                    patch.object(
                        schedule_batch, "mamba_checkpoint_grid", return_value=64
                    )
                )
                w.stack.enter_context(
                    patch.object(
                        schedule_batch,
                        "get_exec",
                        return_value=NS(mamba=NS(enable_mamba_extra_buffer_lazy=False)),
                    )
                )
                scheduler = ScheduleBatch.__new__(ScheduleBatch)
                scheduler.model_config = NS(hf_text_config=NS(mamba_chunk_size=64))
                scheduler.tree_cache = NS(page_size=64)
                scheduler.req_to_token_pool = NS(
                    _prefill_prompt_only_state_cache=True,
                    get_mamba_ping_pong_other_idx=lambda i: 1 - i,
                )
                req = NS(
                    extend_range=NS(start=0, end=tokens, length=tokens),
                    origin_input_ids=[0] * tokens,
                    prefix_indices=[],
                    mamba_branching_seqlen=None,
                    kv=NS(
                        mamba_ping_pong_track_buffer=torch.tensor([2, 3]),
                        mamba_next_track_idx=0,
                        mamba_last_track_idx=None,
                        mamba_last_track_seqlen=None,
                    ),
                )
                scheduler.reqs = [req] * rows
                scheduler.forward_mode = (
                    ForwardMode.MIXED if mixed else ForwardMode.EXTEND
                )
                scheduler.spec_info = None
                entry = scheduler._mamba_radix_cache_v2_req_prepare_for_extend(req)
                self.assertEqual(entry.track_mask, expected is not None)
                if expected is not None:
                    self.assertEqual(req.kv.mamba_last_track_seqlen, expected)
                    self.assertEqual(req.kv.mamba_next_track_idx, 1)
                self.assertEqual(req.extend_range.length, tokens)

    def test_full_N_collector_logits_and_phase_zero_publication(self):
        for native_model in (False, True):
            with (
                self.subTest(native=native_model),
                worker(native_model=native_model) as w,
            ):
                ids = torch.arange(8)
                logits = torch.stack((ids.float(), -ids.float()), dim=-1)
                backend = NS(forward_metadata=NS(factored_extend=w.plan))

                def core(input_ids, positions, fb, **kw):
                    self.assertEqual(input_ids.tolist(), ids.tolist())
                    for lid in IDS:
                        state = torch.full((1, 2, 16, 16), float(lid))
                        native.FactoredGDNPool.commit_extend_batched(
                            w.pool, lid, w.plan, state
                        )
                    self.assertEqual(w.pool._agg_prefill_graph.run.call_count, 0)
                    return logits.clone()

                w.owner.model.forward = core
                w.stack.enter_context(
                    patch(
                        "sglang.srt.model_executor.forward_context.get_attn_backend",
                        return_value=NS(linear_attn_backend=backend),
                    )
                )
                init_graphs(w.runner, w.capture)

                def publish(pool, plan, values, *args, **kwargs):
                    self.assertEqual(len(values), 36)
                    self.assertEqual([v[0][0, 0, 0, 0].item() for v in values], IDS)
                    pool.count[:, plan.slots] = pool.cfg.r

                w.pool._agg_prefill_graph.run = Mock(side_effect=publish)
                fb = NS(
                    _pfactor_agg_contract=True,
                    batch_size=1,
                    forward_mode=ForwardMode.EXTEND,
                    input_ids=ids,
                    extend_seq_lens_cpu=[8],
                    spec_info=None,
                )
                output = w.owner.forward(ids, ids, fb)
                self.assertTrue(torch.equal(output, logits))
                self.assertEqual(w.pool._agg_prefill_graph.run.call_count, 1)
                self.assertEqual(w.plan.pending, [])
                req = NS(
                    _pfactor_agg_contract=True, kv=NS(mamba_pool_idx=torch.tensor(1))
                )
                before = w.pool.count.clone()
                FactorStateHandoff(w.pool).before_send(req)
                self.assertEqual(req.factored_prefill_boundary_steps, 0)
                self.assertTrue(torch.equal(before, w.pool.count))
                self.assertTrue(torch.all(w.pool.count[:, 1] == w.pool.cfg.r))

    def test_PD_full_N_real_trunk_and_rejection_keep_checkpoint_and_count(self):
        from test_flash_next_pd_trunk_prefill_graph import cpu_runner
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        for enabled, reject, extra in (
            (False, False, {}),
            (True, False, {}),
            (True, True, {}),
            (True, False, {"get_embedding": True}),
            (True, False, {"pp_proxy_tensors": object()}),
        ):
            with (
                self.subTest(enabled=enabled, reject=reject, extra=extra),
                worker() as w,
            ):
                owner = w.owner
                owner.pd_trunk_prefill_graph = enabled
                ids = torch.arange(8)
                expected = torch.stack((ids.float(), -ids.float()), -1)
                backend = NS(forward_metadata=NS(factored_extend=w.plan))
                hybrid = NS(linear_attn_backend=backend)
                w.stack.enter_context(
                    forward_context(ForwardContext(attn_backend=hybrid))
                )

                def core(batch):
                    self.assertIs(backend.forward_metadata.factored_extend, w.plan)
                    for lid in IDS:
                        value = torch.full((1, 2, 16, 16), float(lid))
                        native.FactoredGDNPool.commit_extend_batched(
                            w.pool, lid, w.plan, value
                        )
                    return torch.stack(
                        (batch.input_ids.float(), -batch.input_ids.float()), -1
                    )

                owner.model.forward = lambda ids, positions, fb, **kwargs: NS(
                    next_token_logits=core(fb)
                )
                owner._p_trunk = lambda batch: (core(batch), None)
                owner.model.model.hyper_connection_mixer = NS(
                    mix=lambda value: (value, None)
                )
                owner.model.logits_processor = lambda ids, hidden, *args: NS(
                    next_token_logits=hidden
                )
                owner.model.lm_head = None
                trunk = cpu_runner(owner, hybrid)
                trunk.can_run_graph = lambda batch: not reject
                owner._prefill_runners = dict(trunk=trunk)
                init_graphs(w.runner, w.capture)

                def publish(pool, plan, values, *args, **kwargs):
                    self.assertEqual(len(values), 36)
                    pool.count[:, plan.slots] = pool.cfg.r

                w.pool._agg_prefill_graph.run = Mock(side_effect=publish)
                fb = NS(
                    _pfactor_agg_contract=True,
                    pd_factor_only_full_batch=True,
                    batch_size=1,
                    forward_mode=ForwardMode.EXTEND,
                    input_ids=ids,
                    positions=ids,
                    extend_seq_lens_cpu=[8],
                    spec_info=None,
                    out_cache_loc=ids + 100,
                    twinstar_prompt_final=[True],
                    req_pool_indices_cpu=torch.tensor([0]),
                )
                output = owner.forward(ids, ids, fb, **extra)
                self.assertTrue(torch.equal(output.next_token_logits, expected))
                count = int(enabled and not reject and not extra)
                self.assertEqual(trunk.run_count, count)
                self.assertEqual(owner.n_pd_trunk_graph, count)
                self.assertEqual(owner._agg_fulln_trunk_replays, count)
                self.assertEqual(w.pool._agg_prefill_graph.run.call_count, 1)
                self.assertEqual(w.pool._prefill_batch_graph.run.call_count, 0)
                self.assertEqual(fb.factored_prefill_boundary_steps, 0)
                self.assertIs(backend.forward_metadata.factored_extend, w.plan)
                req = NS(
                    _pfactor_agg_contract=True, kv=NS(mamba_pool_idx=torch.tensor(1))
                )
                before = w.pool.count.clone()
                FactorStateHandoff(w.pool).before_send(req)
                self.assertEqual(req.factored_prefill_boundary_steps, 0)
                self.assertTrue(torch.equal(before, w.pool.count))
                self.assertTrue(torch.all(w.pool.count[:, 1] == w.pool.cfg.r))
                print(
                    f"CPU PD trunk arm=C full_N=1 runs={count} full_N_commit_runs=1 exact_commit_runs=0 phase=0"
                )


if __name__ == "__main__":
    unittest.main()
