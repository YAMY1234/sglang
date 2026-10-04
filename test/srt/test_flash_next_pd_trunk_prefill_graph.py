"""PD trunk execution contracts; CPU replay is not CUDA/NLL qualification.

Use production AST methods and real forward contexts. Only capture allocation,
CUDA graph segments and model weights are CPU stand-ins. Exact-tail publication
has a distinct counter and cannot satisfy a trunk execution assertion.
"""

import ast
import copy
import logging
import itertools
import os
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "python/sglang/srt/models/flash_next_duet/model.py"
GRAPH = MODEL.with_name("prefill_graph.py")
PREFILL_RUNNER = (
    ROOT / "python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py"
)


def methods(path, class_name, names, scope, *, keep_decorators=False):
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    nodes = [copy.deepcopy(n) for n in cls.body if getattr(n, "name", None) in names]
    assert len(nodes) == len(names), names
    if not keep_decorators:
        for node in nodes:
            node.decorator_list = []
    compile_nodes(path, nodes, scope)
    return {name: scope[name] for name in names}


def compile_nodes(path, nodes, scope):
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.Module(body=[future, *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)


def model_methods(scope=None):
    namespace = dict(
        os=os,
        get_is_capture_mode=lambda: False,
        logger=logging.getLogger(__name__),
        LogitsProcessorOutput=NS,
    )
    namespace.update(scope or {})
    names = {
        "forward",
        "_pd_trunk_prefill_graph_runner",
        "_run_pd_trunk_prefill_graph",
        "_pd_trunk_prefill_graph_forward",
        "_alloff_prefill_graph_runner",
        "_p_trunk",
        "_emit_ids",
        "_sub_batch",
        "_decode_batch",
        "_boundary_lens",
    }
    result = methods(MODEL, "Qwen4ExpForConditionalGeneration", names, namespace)
    return type("CPUOwner", (), result), namespace


def cpu_runner(owner, attention_backend, *, attention_layers=(), padded_tokens=None):
    """Production run/can_run/context/slicing methods with CPU static buffers."""
    import torch
    from sglang.srt.models.flash_next_duet.prefill_graph import _Body, _runner_class

    runner_cls = _runner_class()
    runner = runner_cls.__new__(runner_cls)
    runner.body = _Body(owner, True, [])
    runner.run_count = 0
    runner.max_num_tokens = 128
    runner.can_run_graph = lambda fb: True
    runner.model_runner = NS(attn_backend=attention_backend)
    runner.attention_layers = list(attention_layers)
    runner.quant_config = None
    runner.moe_layers = runner.moe_fusions = []
    runner.dsa_indexers = runner.mha_companion_layers = []
    runner._is_full_backend = False
    runner.seen = []

    def load_batch(fb):
        static = copy.copy(fb)
        raw = len(fb.input_ids)
        size = padded_tokens or raw
        runner.raw_num_tokens = raw
        for key in ("input_ids", "positions", "out_cache_loc"):
            value = getattr(fb, key)
            setattr(static, key, torch.cat((value, value.new_zeros(size - raw))))
        static.global_num_token_non_padded_cpu = raw
        # Match constructor defaults in PrefillCudaGraphRunner.load_batch.
        static.twinstar_prompt_final = None
        static.pd_factor_only_full_batch = False
        static.req_pool_indices_cpu = None
        for key in ("mamba_track_mask", "mamba_track_indices", "mamba_track_seqlens"):
            value = getattr(fb, key, None)
            setattr(static, key, value.clone() if value is not None else None)
        runner._prepare_forward_metadata_for_replay(fb, static, size)
        return static

    def replay(key, static):
        runner.seen.append(static)
        return runner.body.forward(static.input_ids, static.positions, static)

    runner.load_batch = load_batch
    runner.backend = NS(replay_session=nullcontext, replay=replay)
    return runner


class Mode:
    def __init__(self, kind="extend"):
        self.kind = kind

    def is_extend(self):
        return self.kind != "decode"

    def is_mixed(self):
        return self.kind == "mixed"

    def is_target_verify(self):
        return self.kind == "verify"

    def is_draft_extend_v2(self):
        return self.kind == "draft"


class PDTrunkGateTest(unittest.TestCase):
    """These routing tests run without torch or SGLang imports."""

    def setUp(self):
        self.enabled = False
        cls, self.scope = model_methods()
        self.owner = cls()
        self.owner.pd_trunk_prefill_graph = True
        self.owner.fullstack = dict(
            duet_spec={},
            prefill_layer_trim=False,
            gdn_rank=0,
            prefill_saving_policy="kv-and-ssm",
            qsa_code="off",
        )
        self.runner = NS(body=NS(trunk=True, emit_ids=[]), can_run=lambda fb: True)
        self.owner._prefill_runners = dict(trunk=self.runner)
        self.fb = NS(
            forward_mode=Mode(),
            spec_info=None,
            input_ids=NS(shape=(3,)),
            extend_seq_lens_cpu=[3],
        )
        self.scope["envs"] = NS(
            SGLANG_FLASHNEXT_PD_TRUNK_PREFILL_GRAPH=NS(get=lambda: self.enabled)
        )
        tree = ast.parse(MODEL.read_text())
        compile_nodes(
            MODEL,
            [
                next(
                    n
                    for n in tree.body
                    if getattr(n, "name", "") == "_pd_trunk_prefill_graph_enabled"
                )
            ],
            self.scope,
        )
        self.gate = self.scope["_pd_trunk_prefill_graph_enabled"]

    def test_default_and_all_four_arm_role_gates(self):
        for role in ("prefill", "decode", "null"):
            args = NS(disaggregation_mode=role, is_embedding=False, pp_size=1)
            for trim, rank in ((False, 0), (False, 8), (True, 0), (True, 8)):
                fs = {
                    **self.owner.fullstack,
                    "prefill_layer_trim": trim,
                    "gdn_rank": rank,
                }
                for flag in (False, True):
                    self.enabled = flag
                    self.assertEqual(self.gate(fs, args), flag and role == "prefill")
        self.enabled = True
        for fs in (
            None,
            {},
            {**self.owner.fullstack, "qsa_code": "on"},
            {**self.owner.fullstack, "prefill_saving_policy": "latent-and-ssm"},
        ):
            self.assertFalse(
                self.gate(
                    fs, NS(disaggregation_mode="prefill", is_embedding=False, pp_size=1)
                )
            )
        for embedding, pp in ((True, 1), (False, 2)):
            self.assertFalse(
                self.gate(
                    self.owner.fullstack,
                    NS(
                        disaggregation_mode="prefill",
                        is_embedding=embedding,
                        pp_size=pp,
                    ),
                )
            )

    def test_env_declaration_is_independent_default_off(self):
        text = (ROOT / "python/sglang/srt/environ.py").read_text()
        self.assertIn("SGLANG_FLASHNEXT_PD_TRUNK_PREFILL_GRAPH = EnvBool(False)", text)
        self.assertIn("SGLANG_FLASHNEXT_ALLOFF_PREFILL_GRAPH_PD = EnvBool(False)", text)
        node = next(
            n
            for n in ast.parse(MODEL.read_text()).body
            if getattr(n, "name", "") == "_alloff_prefill_graph_enabled"
        )
        self.assertNotIn("PD_TRUNK", ast.unparse(node))

    def test_C_requires_installed_contract_and_full_batch_marker(self):
        self.owner.fullstack["gdn_rank"] = 8
        self.assertIsNone(self.owner._pd_trunk_prefill_graph_runner(self.fb))
        self.owner._pd_factor_only_contract = True
        self.assertIsNone(self.owner._pd_trunk_prefill_graph_runner(self.fb))
        self.fb.pd_factor_only_full_batch = True
        self.assertIs(self.owner._pd_trunk_prefill_graph_runner(self.fb), self.runner)
        with patch.dict(os.environ, {"SGLANG_GDN_PREFILL_STOCK_DENSE_COMMIT": "1"}):
            self.assertIsNone(self.owner._pd_trunk_prefill_graph_runner(self.fb))

    def test_shallow_requires_P_role_and_trunk_only_body(self):
        self.owner.fullstack["prefill_layer_trim"] = True
        self.assertIsNone(self.owner._pd_trunk_prefill_graph_runner(self.fb))
        self.owner.pd_shallow_role = "prefill"
        self.assertIs(self.owner._pd_trunk_prefill_graph_runner(self.fb), self.runner)
        self.runner.body.emit_ids = [31]
        self.assertIsNone(self.owner._pd_trunk_prefill_graph_runner(self.fb))

    def test_rejected_modes_capture_inputs_and_buckets(self):
        for field, value in (
            ("forward_mode", Mode("decode")),
            ("forward_mode", Mode("mixed")),
            ("forward_mode", Mode("verify")),
            ("forward_mode", Mode("draft")),
            ("spec_info", object()),
            ("input_embeds", object()),
            ("can_run_tbo", True),
            ("tbo_parent_token_range", (0, 1)),
            ("tbo_split_seq_index", 0),
            ("extend_seq_lens_cpu", [2]),
        ):
            fb = copy.copy(self.fb)
            setattr(fb, field, value)
            self.assertIsNone(self.owner._pd_trunk_prefill_graph_runner(fb), field)
        for args in ((True, None), (False, object())):
            self.assertIsNone(self.owner._pd_trunk_prefill_graph_runner(self.fb, *args))
        self.scope["get_is_capture_mode"] = lambda: True
        self.assertIsNone(self.owner._pd_trunk_prefill_graph_runner(self.fb))
        self.scope["get_is_capture_mode"] = lambda: False
        self.runner.can_run = lambda fb: False
        self.assertIsNone(self.owner._pd_trunk_prefill_graph_runner(self.fb))
        self.owner._prefill_runners = {}
        self.assertIsNone(self.owner._pd_trunk_prefill_graph_runner(self.fb))


class PDTrunkReplayTest(unittest.TestCase):
    def test_capture_PD_owns_only_trunk_and_AGG_joint_is_unchanged(self):
        import torch
        from sglang.srt.models.flash_next_duet import prefill_graph as graph

        for pd in (False, True):
            captured = []
            owner = NS(
                config=NS(hc_count=1, hidden_size=32),
                fullstack_code=False,
                fullstack_v3_latent=False,
                pd_trunk_prefill_graph=pd,
                _emit_ids=lambda: list(range(31, 48)),
            )

            class Capture:
                def __init__(self, mr, body, buckets, name, attributes):
                    self.body, self.run_count = body, 0
                    captured.append((name, body.emit_ids))

            with patch.object(graph, "enabled", return_value=True), patch.object(
                graph, "_runner_class", return_value=Capture
            ), patch(
                "sglang.srt.arg_groups.cuda_graph_hook.generate_prefill_cuda_graph_batch_sizes",
                return_value=[8, 16],
            ), patch(
                "sglang.srt.runtime_context.get_schedule",
                return_value=NS(chunked_prefill_size=16),
            ), patch.object(
                torch.cuda, "mem_get_info", return_value=(2**30, 2**30)
            ), patch.dict(
                os.environ, {"SGLANG_QWEN4_PREFILL_GRAPH": "1"}, clear=True
            ):
                runners = graph.capture(owner, NS(attention_layers=[]))
            self.assertEqual(captured, [("trunk", [] if pd else list(range(31, 48)))])
            self.assertEqual(runners["trunk"].run_count, 0)
            self.assertIs(runners["emitters"], None if pd else runners["trunk"])

    def test_runner_refreshes_host_fields_and_keeps_padding_out_of_outputs(self):
        import torch

        owner = NS(
            config=NS(hc_count=1, hidden_size=1),
            pd_trunk_prefill_graph=True,
            _p_trunk=lambda batch: (batch.input_ids[:, None].float(), None),
        )
        runner = cpu_runner(owner, NS(), padded_tokens=4)
        for final, slot in ((False, 1), (True, 7)):
            ids = torch.arange(3)
            fb = NS(
                input_ids=ids,
                positions=ids,
                out_cache_loc=ids + 100,
                twinstar_prompt_final=[final],
                pd_factor_only_full_batch=final,
                req_pool_indices_cpu=torch.tensor([slot]),
                mamba_track_indices=torch.tensor([50]),
                mamba_track_mask=torch.tensor([True]),
                mamba_track_seqlens=torch.tensor([2]),
            )
            result = runner.run(fb)
            self.assertTrue(torch.equal(result, ids[:, None].float()))
            static = runner.seen[-1]
            self.assertEqual(static.twinstar_prompt_final, [final])
            self.assertEqual(static.pd_factor_only_full_batch, final)
            self.assertEqual(static.req_pool_indices_cpu.tolist(), [slot])
            self.assertEqual(static.input_ids.shape, (4,))
            for key in (
                "mamba_track_indices",
                "mamba_track_mask",
                "mamba_track_seqlens",
            ):
                self.assertTrue(torch.equal(getattr(static, key), getattr(fb, key)))
            self.assertEqual(fb.out_cache_loc.tolist(), [100, 101, 102])
        self.assertEqual(runner.run_count, 2)
        with patch.dict(os.environ, {"SGLANG_PREFILL_GRAPH_CAPTURE_ONLY": "1"}):
            self.assertFalse(runner.can_run(fb))

    def test_failed_replay_does_not_increment_execution_counters(self):
        import torch

        cls, _ = model_methods()
        owner = cls()
        owner.config = NS(hc_count=1, hidden_size=1)
        owner.pd_trunk_prefill_graph = True
        owner.n_pd_trunk_graph = 0
        runner = cpu_runner(owner, NS())

        def fail(*args):
            raise RuntimeError("replay failed")

        runner.backend.replay = fail
        ids = torch.arange(3)
        fb = NS(input_ids=ids, positions=ids, out_cache_loc=ids + 100)
        with self.assertRaisesRegex(RuntimeError, "replay failed"):
            owner._run_pd_trunk_prefill_graph(runner, fb)
        self.assertEqual(runner.run_count, 0)
        self.assertEqual(owner.n_pd_trunk_graph, 0)

    def run_arm(self, arm, graph_on, *, reject=False, final=True, exact_on=False):
        """Actual native shallow envelope and per-layer split with CPU kernels."""
        import torch
        from sglang.srt.models import qwen4_exp as stock
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.models.flash_next_duet import pd_shallow as shallow
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (
            get_tc_piecewise_forward_context,
        )
        from sglang.srt.layers.radix_linear_attention import (
            _unified_linear_attention_with_output_impl,
        )
        from twinstar_sgl.pd_shallow import SplitBoundaryPhase
        from twinstar_sgl.pd_dense import DenseBoundaryHandoff
        from test_flash_next_native_factor_tail import IDS, fixture, layer
        from sglang.srt.mem_cache import gdn_prefill_exact_tail as exact
        from twinstar_sgl import pd_shallow_gdn

        trim, factor = arm in ("P", "PC"), arm == "PC"
        events, break_ids = [], []
        recorder = NS(with_current_layer=lambda lid: nullcontext())
        cls, scope = model_methods(
            dict(
                torch=torch,
                LogitsProcessorOutput=LogitsProcessorOutput,
                copy=copy,
                itertools=itertools,
                ForwardMode=ForwardMode,
                _stock=stock,
                _optional_prefill_graph=lambda: None,
                get_global_expert_distribution_recorder=lambda: recorder,
                _dev=lambda values, dtype, device: torch.as_tensor(
                    values, dtype=dtype, device=device
                ),
            )
        )
        owner = cls()
        owner.__dict__.update(
            config=NS(hc_count=1, hidden_size=32, vocab_size=7),
            fullstack=dict(
                duet_spec={},
                prefill_layer_trim=trim,
                gdn_rank=8 if factor else 0,
                gdn_every=8 if factor else 0,
                latent="off",
                qsa_code="off",
            ),
            fullstack_code=False,
            fullstack_v3_latent=False,
            fullstack_final=True,
            n_layers=48,
            p_layer_ids=list(range(31 if trim else 48)),
            emitter_ids=list(range(31, 48)) if trim else [],
            bridges=[],
            pd_shallow_role="prefill" if trim else None,
            pd_trunk_prefill_graph=graph_on,
            alloff_prefill_graph=False,
            n_pd_trunk_graph=0,
            n_alloff_graph=0,
            n_twinstar=0,
            n_prefix=0,
            dump_dir=None,
            state_audit_dir=None,
            _is_twinstar_prefill=lambda fb: trim,
        )
        ids = torch.arange(3)
        fb = NS(
            batch_size=1,
            forward_mode=ForwardMode.EXTEND,
            spec_info=None,
            twinstar_prompt_final=[final],
            pd_factor_only_full_batch=False,
            input_ids=ids,
            positions=ids,
            req_pool_indices=torch.tensor([0]),
            req_pool_indices_cpu=torch.tensor([0]),
            seq_lens=torch.tensor([3]),
            seq_lens_cpu=torch.tensor([3]),
            orig_seq_lens=None,
            out_cache_loc=ids + 100,
            extend_num_tokens=3,
            extend_seq_lens=torch.tensor([3]),
            extend_seq_lens_cpu=[3],
            extend_prefix_lens=torch.tensor([0]),
            extend_prefix_lens_cpu=[0],
            extend_start_loc=torch.tensor([0]),
            extend_logprob_start_lens_cpu=None,
            mamba_track_mask=torch.zeros(1, dtype=torch.bool),
            mamba_track_indices=torch.tensor([50]),
            mamba_track_seqlens=torch.tensor([-1]),
            return_logprob=False,
            global_num_token_non_padded_cpu=3,
        )
        c = fixture()
        pool, plan = c.pool, c.plan
        pool.layer_ids = IDS
        pool.layer_map = {lid: i for i, lid in enumerate(IDS)}
        pool.layer_index = lambda lid: pool.layer_map[lid]
        pool.batch_prefill = True
        dense = torch.zeros(36, 2, 16, 16)
        saved = torch.zeros_like(dense)
        track_slots = torch.tensor([50])
        state = NS(
            hidden=torch.zeros(2, 32),
            position=torch.full((2, 1), -1),
            valid=torch.zeros(2, 1, dtype=torch.int32),
        )
        rp = c.rp
        rp.pd_boundary_state = state
        rp.get_mamba_indices = lambda indices: indices + 1
        rp.translate_mamba_indices = lambda indices: indices

        def publish(li, value, track):
            dense[li].copy_(value[0])
            saved[li].copy_(track[0])
            for slot, v in ((1, value[0]), (50, track[0])):
                pool.a[li, slot].copy_(v[:, 0])
                pool.U[li, slot].zero_()
                pool.W[li, slot].zero_()
                pool.U[li, slot, :, :8].copy_(v[:, :8])
                pool.W[li, slot, :, :8].copy_(torch.eye(16)[:8].expand(2, -1, -1))
                pool.count[li, slot] = 8

        def commit(p, active_plan, values, *args, **kwargs):
            self.assertEqual(len(values), 36)
            for li, (value, track) in enumerate(values):
                publish(li, value, track)
                if IDS[li] < 31 and final:
                    pool.count[li, 1] += 1
            events.append("exact-publication")

        pool._prefill_batch_graph.run.side_effect = commit

        class Backend:
            factored = pool if factor else None
            forward_metadata = None

            def init_forward_metadata(self, batch):
                events.append(("plan", batch.forward_mode, len(batch.input_ids)))
                self.forward_metadata = NS(
                    factored_extend=plan,
                    mamba_cache_indices=plan.slots,
                    track_ssm_final_src=None,
                    track_ssm_final_dst=None,
                )

            def forward_extend(self, layer, forward_batch, mixed_qkv, a, b, **kw):
                lid = layer.layer_id
                li = pool.layer_index(lid)
                events.append(("prefix", lid, mixed_qkv.shape[0]))
                value = torch.full((1, 2, 16, 16), float(lid + 1))
                tx = getattr(pool, "_exact_tail_transaction", None)
                if tx is None:
                    publish(li, value, value + 2)
                    self.forward_metadata.factored_extend.next_layer += 1
                else:
                    tx.add(
                        lid,
                        self.forward_metadata.factored_extend,
                        value,
                        value + 2,
                        track_slots,
                        None,
                        None,
                    )
                return mixed_qkv[:, :32].reshape(1, -1, 2, 16) + 100

            def forward_decode(self, layer, batch, mixed_qkv, a, b, **kw):
                lid = layer.layer_id
                events.append(("tail", lid, mixed_qkv.shape[0]))
                self_outer.assertFalse(batch.mamba_track_mask.any())
                tx = getattr(pool, "_exact_tail_transaction", None)
                if tx is None:
                    pool.count[pool.layer_index(lid), 1] += 1
                else:
                    tx.tails[lid] = (
                        mixed_qkv.clone(),
                        a.clone(),
                        b.clone(),
                        plan.slots.clone(),
                    )
                return mixed_qkv[:, :32].reshape(1, -1, 2, 16) + 200

        self_outer = self
        linear = Backend()
        full = NS(
            init_forward_metadata=lambda batch: events.append(
                ("qsa-plan", len(batch.input_ids))
            )
        )
        hybrid = NS(
            linear_attn_backend=linear,
            full_attn_backend=full,
            req_to_token_pool=rp,
            forward=lambda **kw: linear.forward_extend(**kw),
        )
        layers = [layer(lid) if lid in IDS else NS(layer_id=lid) for lid in range(48)]
        qsa = {}

        def compute(lid, hidden, batch):
            if lid not in IDS:
                raw = getattr(batch, "global_num_token_non_padded_cpu", len(hidden))
                qsa[lid] = hidden[:raw].clone()
                return hidden + 1
            mixed = hidden.repeat(1, 3)
            a = b = torch.zeros(len(hidden), 2)
            if get_tc_piecewise_forward_context() is not None:
                result = torch.empty(1, len(hidden), 2, 16)
                _unified_linear_attention_with_output_impl(mixed, a, b, result, lid)
                break_ids.append(lid)
            else:
                result = linear.forward_extend(
                    layer=layers[lid], forward_batch=batch, mixed_qkv=mixed, a=a, b=b
                )
            return result.reshape(-1, 32)

        class Layer:
            def __init__(self, lid):
                self.lid, self.ple = lid, None

            def __call__(self, *, hidden_states, forward_batch, **kw):
                self_outer.assertLess(self.lid, 31 if trim else 48)
                return compute(self.lid, hidden_states, forward_batch), None

        body = NS(
            embed_tokens=lambda values: values[:, None].float().expand(-1, 32).clone(),
            has_ple=True,
            ple_ngram_size=3,
            ple_ngram_eos_token_id=0,
            layers=[Layer(i) for i in range(48)],
            modules=lambda: iter(layers),
            hyper_connection_mixer=NS(mix=lambda streams: (streams, None)),
        )
        owner.model = NS(
            model=body,
            lm_head=None,
            logits_processor=lambda ids, hidden, *args: NS(next_token_logits=hidden),
        )
        owner.emitters = {
            str(lid): NS(
                is_attn=lid not in IDS,
                emit=lambda streams, batch, lid=lid: (
                    events.append(("emitter", lid, len(batch.input_ids))),
                    compute(lid, streams, batch),
                ),
            )
            for lid in owner.emitter_ids
        }
        owner._publish_qsa_prefix = lambda batch, ms: events.append(
            ("qsa-publish", list(ms))
        )
        owner._twinstar_prefill = lambda ids, pos, batch: shallow.prefill_extend(
            owner, ids, pos, batch
        )

        def stock_forward(ids, pos, batch, **kwargs):
            streams, _ = owner._p_trunk(batch)
            return NS(next_token_logits=streams, hidden_states=streams)

        owner.model.forward = stock_forward
        runner = cpu_runner(owner, hybrid, attention_layers=layers, padded_tokens=4)
        runner.can_run_graph = lambda batch: not reject
        owner._prefill_runners = dict(trunk=runner)
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {}, clear=True))
            stack.enter_context(forward_context(ForwardContext(attn_backend=hybrid)))
            stack.enter_context(
                patch(
                    "sglang.srt.layers.communicator.get_attn_tp_context",
                    return_value=NS(maybe_input_scattered=lambda batch: nullcontext()),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.eplb.expert_distribution.get_global_expert_distribution_recorder",
                    return_value=recorder,
                )
            )
            stack.enter_context(
                patch.object(
                    stock,
                    "_prepare_ple_batch",
                    side_effect=lambda *a, **k: events.append("ple-prepare"),
                )
            )
            stack.enter_context(
                patch.object(
                    stock,
                    "_commit_ple_batch",
                    side_effect=lambda *a: events.append("ple-commit"),
                )
            )
            stack.enter_context(patch("twinstar_sgl.pd_shallow_audit.snapshot"))
            stack.enter_context(
                patch("twinstar_sgl.pd_emitter_graph.try_emit", return_value=False)
            )
            if exact_on:
                self.assertTrue(factor)
                pool._exact_tail_layers = [layers[lid] for lid in IDS if lid < 31]
                stack.enter_context(
                    patch.object(pd_shallow_gdn, "split_boundary", exact.split_boundary)
                )
                stack.enter_context(exact.ExactTailTransaction(pool, rp, fb))
            if not trim:
                linear.init_forward_metadata(fb)
            output = owner.forward(ids, ids, fb)
        expected_runs = int(graph_on and not reject)
        self.assertEqual(runner.run_count, expected_runs)
        self.assertEqual(owner.n_pd_trunk_graph, expected_runs)
        self.assertEqual(
            break_ids,
            [lid for lid in IDS if lid < (31 if trim else 48)] if expected_runs else [],
        )
        self.assertEqual(pool._prefill_batch_graph.run.call_count, int(exact_on))
        if trim:
            self.assertEqual(
                [e for e in events if isinstance(e, tuple) and e[0] == "emitter"],
                [("emitter", lid, 2 if final else 3) for lid in range(31, 48)],
            )
            self.assertEqual(
                [e[1] for e in events if isinstance(e, tuple) and e[0] == "tail"],
                [lid for lid in IDS if lid < 31] if final else [],
            )
            if final:
                req = NS(kv=NS(mamba_pool_idx=torch.tensor([1])))
                if factor:
                    SplitBoundaryPhase(NS(pool=pool), state).before_send(req)
                else:
                    DenseBoundaryHandoff(state).before_send(req)
                self.assertEqual(state.position[1].tolist(), [2])
                self.assertEqual(state.valid[1].tolist(), [1])
        self.assertEqual(events.count("ple-prepare"), 1)
        self.assertEqual(events.count("ple-commit"), 1)
        print(
            f"CPU PD trunk arm={arm} final={final} exact={exact_on} runs={runner.run_count} commit_runs={pool._prefill_batch_graph.run.call_count}"
        )
        return (
            output.next_token_logits,
            state.hidden,
            state.position,
            state.valid,
            dense,
            saved,
            *(t[:, [1, 50]].clone() for t in (pool.a, pool.U, pool.W, pool.count)),
            *[qsa[lid] for lid in sorted(qsa)],
        )

    def test_S_P_PC_real_trunk_runs_and_flagoff_outputs_match(self):
        import torch

        for arm in ("S", "P", "PC"):
            for final in (False, True):
                expected = self.run_arm(arm, False, final=final)
                for graph_on, reject in ((True, False), (True, True)):
                    with self.subTest(
                        arm=arm, final=final, graph=graph_on, reject=reject
                    ):
                        actual = self.run_arm(arm, graph_on, final=final, reject=reject)
                        for a, b in zip(expected, actual, strict=True):
                            self.assertTrue(torch.equal(a, b))

    def test_PC_exact_tail_commit_is_independent_of_trunk_execution(self):
        import torch

        expected = self.run_arm("PC", False, exact_on=True)
        actual = self.run_arm("PC", True, exact_on=True)
        for a, b in zip(expected, actual, strict=True):
            self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
