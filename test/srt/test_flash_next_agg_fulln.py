"""AGG full-N selection, real scheduler/collector, and preserved fallback.

CPU allocation and graph replay stand-ins; CUDA/NLL remain separate gates.
"""

import ast
import copy
import os
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from test_flash_next_native_factor_tail import (
    installed,
    IDS,
    torch,
    ScheduleBatch,
    ForwardMode,
)
from test_flash_next_p48_contract import init_graphs
from sglang.srt.mem_cache import gdn_prefill_agg_contract as agg
from sglang.srt.mem_cache import gdn_prefill_batch_graph as bg
from sglang.srt.mem_cache import gdn_factored_pool as native
from sglang.srt.disaggregation.state_handoff import FactorStateHandoff
from test_gdn_prefill_batch_graph import fake_pool, make_plan

ROOT = Path(__file__).resolve().parents[2]


def real_prepare(owner):
    path = ROOT / "python/sglang/srt/models/flash_next_duet/model.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpForConditionalGeneration"
    )
    method = copy.deepcopy(
        next(n for n in cls.body if getattr(n, "name", "") == "prepare_forward_batch")
    )
    method.decorator_list = []
    scope = {}
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, method], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        scope,
    )
    return MethodType(scope["prepare_forward_batch"], owner)


@contextmanager
def worker(flag=True):
    # Native constructor chain, explicitly without any PD-only wrappers.
    with installed(enabled=False, role="null") as (owner, _), ExitStack() as stack:
        stack.enter_context(
            patch.dict(
                os.environ,
                {
                    agg.AGG_FLAG: str(int(flag)),
                    "SGLANG_GDN_PREFILL_EXACT_TAIL_BATCH": "0",
                    "TWINSTAR_PD_FACTOR_ONLY_TAIL": "0",
                },
            )
        )
        stack.enter_context(
            patch(
                "sglang.srt.runtime_context.get_schedule",
                return_value=NS(disable_overlap_schedule=True),
            )
        )
        owner.defer_shallow_factor_plan = True
        owner.strict = True
        owner.ratio = 64
        owner.n_fallback = 0
        owner.prepare_forward_batch = real_prepare(owner)
        owner.model.logits_processor = lambda ids, hidden, *a: NS(
            next_token_logits=hidden
        )
        owner.model.model.hyper_connection_mixer = NS(mix=lambda x: (x, None))
        owner.model.lm_head = None
        fallback = torch.tensor([[-2.0, 3.0]])
        owner._twinstar_prefill = Mock(return_value=fallback.clone())
        pool = fake_pool(36, capacity=100)
        pool.layer_ids = IDS
        pool.layer_map = {lid: i for i, lid in enumerate(IDS)}
        pool.cfg.strict_chunk = 1
        pool.batch_prefill = True
        events = []
        pool.prewarm_commit_graph = lambda: events.append("commit-prewarm")
        # The cumulative #23 runner also visits the independent, opt-in k31
        # prewarm hook. This fixture replaces captures, not that runner call.
        pool.prewarm_k31_batch_graph = Mock()

        class Schedule:
            _mamba_radix_cache_v2_req_prepare_for_extend = (
                ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend
            )

        class Forward:
            @classmethod
            def init_new(cls, batch, runner, **kw):
                lengths = [r.extend_range.length for r in batch.reqs]
                return NS(
                    forward_mode=batch.forward_mode,
                    batch_size=len(lengths),
                    input_ids=batch.input_ids,
                    extend_seq_lens_cpu=lengths,
                    extend_prefix_lens_cpu=[0] * len(lengths),
                    twinstar_prompt_final=[True] * len(lengths),
                    input_embeds=None,
                    spec_info=batch.spec_info,
                    can_run_tbo=batch.can_run_tbo,
                    tbo_split_seq_index=batch.tbo_split_seq_index,
                    mamba_track_mask=batch.mamba_track_mask,
                    mamba_track_indices=batch.mamba_track_indices,
                    mamba_track_seqlens=batch.mamba_track_seqlens,
                )

        class Backend:
            def __init__(self):
                self.plan_calls = 0
                self.forward_metadata = NS(factored_extend=None)

            def init_forward_metadata(self, fb):
                if not getattr(fb, "_twinstar_defer_factor_plan", False):
                    self.plan_calls += 1
                    self.forward_metadata.factored_extend = make_plan(
                        pool, fb.batch_size
                    )

        class Handoff(FactorStateHandoff):
            pass

        stack.enter_context(
            patch("sglang.srt.model_executor.forward_batch_info.ForwardBatch", Forward)
        )
        stack.enter_context(
            patch("sglang.srt.managers.schedule_batch.ScheduleBatch", Schedule)
        )
        stack.enter_context(
            patch(
                "sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend", Backend
            )
        )
        stack.enter_context(
            patch("sglang.srt.disaggregation.state_handoff.FactorStateHandoff", Handoff)
        )
        from sglang.srt.managers import schedule_batch

        stack.enter_context(
            patch.object(schedule_batch, "mamba_cache_chunk_size", return_value=64)
        )
        stack.enter_context(
            patch.object(schedule_batch, "mamba_checkpoint_grid", return_value=64)
        )
        stack.enter_context(
            patch.object(
                schedule_batch,
                "get_exec",
                return_value=NS(mamba=NS(enable_mamba_extra_buffer_lazy=False)),
            )
        )
        backend = Backend()
        stack.enter_context(
            patch(
                "sglang.srt.model_executor.forward_context.get_attn_backend",
                return_value=NS(linear_attn_backend=backend),
            )
        )

        def core(ids, positions, fb, **kw):
            if not getattr(fb, "_pfactor_agg_contract", False):
                return fallback.clone()
            plan = backend.forward_metadata.factored_extend
            for lid in IDS:
                dense = torch.full((fb.batch_size, 2, 16, 16), float(lid))
                native.FactoredGDNPool.commit_extend_batched(pool, lid, plan, dense)
            events.append("whole-logits")
            return torch.stack((ids.float(), -ids.float()), -1)

        owner.model.forward = lambda ids, positions, fb, **kw: NS(
            next_token_logits=core(ids, positions, fb, **kw)
        )
        # Real owner runner.run, with only CUDA graph replay replaced by an
        # eager break callback. The callback uses the real BatchCollector.
        from sglang.srt.models.flash_next_duet.prefill_graph import _runner_class

        runner_cls = _runner_class()
        trunk = runner_cls.__new__(runner_cls)
        trunk.body = NS(owner=owner)
        trunk.run_count = 0
        trunk.raw_num_tokens = 0
        trunk.can_run = Mock(return_value=True)

        def load_batch(fb):
            trunk.raw_num_tokens = len(fb.input_ids)
            return fb

        trunk.load_batch = load_batch
        trunk._prefill_forward_context = lambda *a, **kw: nullcontext()
        trunk.backend = NS(
            replay_session=nullcontext,
            replay=lambda key, fb: core(fb.input_ids, fb.input_ids, fb),
        )
        owner._prefill_runners = {"trunk": trunk}

        def graph_prewarm(graph, active_pool, **kw):
            assert not graph.include_tail and not hasattr(pool, "_prefill_batch_graph")
            graph.warmed = True
            events.append("full-N-prewarm")

            def publish(p, plan, values, *args, **kwargs):
                assert len(values) == 36 and events[-1] == "whole-logits"
                p.count[:, plan.slots] = p.cfg.r
                events.append("publish")

            graph.run = Mock(side_effect=publish)

        stack.enter_context(
            patch.object(bg.PrefillBatchGraph, "prewarm", graph_prewarm)
        )
        capture = NS(
            eager_runner=None,
            prefill=NS(runner=None),
            decode=NS(runner=None),
            memory_usage=0,
            time_usage=0,
        )
        runner = NS(
            model=owner,
            req_to_token_pool=NS(factored_gdn_pool=pool),
            device="cpu",
            server_args=NS(
                disaggregation_mode="null",
                is_embedding=False,
                pp_size=1,
                dp_size=1,
                speculative_algorithm=None,
            ),
        )

        def capture_graphs(**kw):
            events.append("framework-capture")
            return capture

        yield NS(
            owner=owner,
            pool=pool,
            runner=runner,
            events=events,
            capture=capture_graphs,
            stack=stack,
            Forward=Forward,
            Schedule=Schedule,
            backend=backend,
            Handoff=Handoff,
            fallback=fallback,
        )


def schedule(w, *, rows=1, tokens=64, mixed=False, tbo=False, prefix=0):
    batch = w.Schedule()
    batch.model_config = NS(hf_text_config=NS(mamba_chunk_size=64))
    batch.tree_cache = NS(page_size=64)
    batch.req_to_token_pool = NS(
        _prefill_prompt_only_state_cache=True,
        get_mamba_ping_pong_other_idx=lambda i: 1 - i,
    )
    batch.reqs = [
        NS(
            extend_range=NS(start=prefix, end=prefix + tokens, length=tokens),
            origin_input_ids=[0] * (prefix + tokens),
            prefix_indices=list(range(prefix)),
            mamba_branching_seqlen=None,
            kv=NS(
                req_pool_idx=i + 1,
                mamba_ping_pong_track_buffer=torch.tensor([2, 3]),
                mamba_next_track_idx=0,
                mamba_last_track_idx=None,
                mamba_last_track_seqlen=None,
            ),
        )
        for i in range(rows)
    ]
    batch.forward_mode = ForwardMode.MIXED if mixed else ForwardMode.EXTEND
    batch.input_ids = torch.arange(rows * tokens)
    batch.spec_info = None
    batch.can_run_tbo = tbo
    batch.tbo_split_seq_index = 0 if tbo else None
    entries = [
        batch._mamba_radix_cache_v2_req_prepare_for_extend(req) for req in batch.reqs
    ]
    for field, key in (
        ("mamba_track_mask", "track_mask"),
        ("mamba_track_indices", "track_index"),
        ("mamba_track_seqlens", "track_seqlen"),
    ):
        setattr(batch, field, torch.tensor([getattr(entry, key) for entry in entries]))
    return batch


class AggFullNTest(unittest.TestCase):
    def test_actual_AGG_install_and_prewarm_without_exact_tail_or_PD_wrapper(self):
        with worker() as w:
            before_send = w.Handoff.before_send
            init_graphs(w.runner, w.capture)
            self.assertTrue(w.owner._pfactor_agg_installed)
            self.assertFalse(getattr(w.owner, "_exact_tail_installed", False))
            self.assertIs(w.Handoff.before_send, before_send)
            self.assertEqual(
                w.events, ["commit-prewarm", "full-N-prewarm", "framework-capture"]
            )
            self.assertFalse(w.pool._agg_prefill_graph.include_tail)

    def test_flag_off_preserves_original_forward_and_planning_bitwise(self):
        with worker(False) as w:
            before_forward = w.owner.forward
            before_prepare = w.owner.prepare_forward_batch
            init_graphs(w.runner, w.capture)
            self.assertEqual(w.events, ["framework-capture"])
            self.assertIs(w.owner.prepare_forward_batch, before_prepare)
            self.assertEqual(w.owner.forward, before_forward)
            batch = schedule(w)
            fb = w.Forward.init_new(batch, w.runner)
            w.owner.prepare_forward_batch(fb)
            output = w.owner.forward(fb.input_ids, fb.input_ids, fb)
            self.assertTrue(torch.equal(output.next_token_logits, w.fallback))
            self.assertEqual(batch.reqs[0].kv.mamba_last_track_seqlen, None)

    def test_full_N_single_plan_first_token_and_phase_zero(self):
        for rows in (1, 8, 16):
            with self.subTest(rows=rows), worker() as w:
                init_graphs(w.runner, w.capture)
                batch = schedule(w, rows=rows)
                fb = w.Forward.init_new(batch, w.runner)
                self.assertTrue(fb._pfactor_agg_contract)
                self.assertTrue(
                    all(r.kv.mamba_last_track_seqlen == 64 for r in batch.reqs)
                )
                self.assertTrue(
                    all(r.factored_prefill_boundary_steps == 0 for r in batch.reqs)
                )
                w.owner.prepare_forward_batch(fb)
                self.assertFalse(fb._twinstar_defer_factor_plan)
                w.backend.init_forward_metadata(fb)
                self.assertEqual(w.backend.plan_calls, 1)
                output = w.owner.forward(fb.input_ids, fb.input_ids, fb)
                expected = torch.stack(
                    (fb.input_ids.float(), -fb.input_ids.float()), -1
                )
                self.assertTrue(torch.equal(output.next_token_logits, expected))
                self.assertTrue(
                    torch.equal(
                        output.next_token_logits.argmax(-1), expected.argmax(-1)
                    )
                )
                self.assertEqual(w.owner._agg_fulln_trunk_replays, 1)
                self.assertTrue(torch.equal(output.hidden_states, expected))
                self.assertEqual(w.pool._agg_prefill_graph.run.call_count, 1)
                self.assertTrue(
                    torch.all(w.pool.count[:, 1 : rows + 1] == w.pool.cfg.r)
                )
                self.assertEqual(fb.factored_prefill_boundary_steps, 0)
                self.assertEqual(w.owner._agg_fulln_prefills, 1)
                w.owner._twinstar_prefill.assert_not_called()

    def test_mixed_TBO_single_token_empty_and_over16_keep_fallback(self):
        for kwargs in (
            dict(mixed=True),
            dict(tbo=True),
            dict(tokens=1),
            dict(tokens=0),
            dict(rows=17),
        ):
            with self.subTest(kwargs=kwargs), worker() as w:
                init_graphs(w.runner, w.capture)
                batch = schedule(w, **kwargs)
                fb = w.Forward.init_new(batch, w.runner)
                self.assertFalse(fb._pfactor_agg_contract)
                w.owner.prepare_forward_batch(fb)
                output = w.owner.forward(fb.input_ids, fb.input_ids, fb)
                self.assertTrue(torch.equal(output.next_token_logits, w.fallback))
                w.pool._agg_prefill_graph.run.assert_not_called()
                self.assertEqual(w.owner._agg_fulln_prefills, 0)

    def test_graph_rejection_restores_original_checkpoint_before_planning(self):
        for missing in (False, True):
            with self.subTest(missing=missing), worker() as w:
                init_graphs(w.runner, w.capture)
                batch = schedule(w)
                self.assertEqual(batch.reqs[0].kv.mamba_last_track_seqlen, 64)
                if missing:
                    w.owner._prefill_runners["trunk"] = None
                else:
                    w.owner._prefill_runners["trunk"].can_run.return_value = False
                fb = w.Forward.init_new(batch, w.runner)
                self.assertFalse(fb._pfactor_agg_contract)
                self.assertIsNone(batch.reqs[0].kv.mamba_last_track_seqlen)
                self.assertEqual(w.backend.plan_calls, 0)
                w.owner.prepare_forward_batch(fb)
                output = w.owner.forward(fb.input_ids, fb.input_ids, fb)
                self.assertTrue(torch.equal(output.next_token_logits, w.fallback))
                self.assertEqual(w.owner._agg_fulln_trunk_replays, 0)

    def test_late_TBO_restores_native_N_minus_one_once(self):
        with worker() as w:
            init_graphs(w.runner, w.capture)
            batch = schedule(w)
            self.assertEqual(batch.reqs[0].kv.mamba_last_track_seqlen, 64)
            batch.can_run_tbo = True
            batch.tbo_split_seq_index = 0
            fb = w.Forward.init_new(batch, w.runner)
            self.assertFalse(fb._pfactor_agg_contract)
            self.assertEqual(batch.reqs[0].kv.mamba_last_track_seqlen, None)
            self.assertEqual(batch.reqs[0].kv.mamba_next_track_idx, 0)
            self.assertEqual(fb.mamba_track_mask.tolist(), [False])

    def test_unqualified_arms_modes_and_capture_reject_or_fallback(self):
        mutations = (
            lambda w: w.owner.fullstack.update(prefill_layer_trim=True),
            lambda w: w.owner.fullstack.update(gdn_rank=0),
            lambda w: setattr(w.owner, "_emit_ids", lambda: [31]),
            lambda w: setattr(w.runner.server_args, "pp_size", 2),
            lambda w: setattr(w.runner.server_args, "is_embedding", True),
            lambda w: setattr(w.runner.server_args, "speculative_algorithm", "EAGLE"),
            lambda w: setattr(w.pool.cfg, "strict_chunk", 0),
        )
        for mutate in mutations:
            with worker() as w:
                mutate(w)
                with self.assertRaises(ValueError):
                    init_graphs(w.runner, w.capture)
        with worker() as w:
            init_graphs(w.runner, w.capture)
            with patch(
                "sglang.srt.model_executor.runner.get_is_capture_mode",
                return_value=True,
            ):
                batch = schedule(w)
                fb = w.Forward.init_new(batch, w.runner)
                self.assertFalse(fb._pfactor_agg_contract)
            for mode in (
                ForwardMode.DECODE,
                ForwardMode.TARGET_VERIFY,
                ForwardMode.DRAFT_EXTEND_V2,
            ):
                fb.forward_mode = mode
                self.assertFalse(agg.agg_eligible(fb))


if __name__ == "__main__":
    unittest.main()
