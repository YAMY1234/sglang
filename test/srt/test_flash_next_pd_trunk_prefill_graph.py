"""PD trunk execution contracts; CPU replay is not CUDA/NLL qualification.

Use production AST methods and real forward contexts. Only capture allocation,
CUDA graph segments and model weights are CPU stand-ins. Exact-tail publication
has a distinct counter and cannot satisfy a trunk execution assertion.
"""

import ast
import copy
import logging
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "python/sglang/srt/models/flash_next_duet/model.py"
GRAPH = MODEL.with_name("prefill_graph.py")
PREFILL_RUNNER = ROOT / "python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py"


def methods(path, class_name, names, scope, *, keep_decorators=False):
    tree = ast.parse(path.read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name)
    nodes = [copy.deepcopy(n) for n in cls.body if getattr(n, "name", None) in names]
    assert len(nodes) == len(names), names
    if not keep_decorators:
        for node in nodes:
            node.decorator_list = []
    compile_nodes(path, nodes, scope)
    return {name: scope[name] for name in names}


def compile_nodes(path, nodes, scope):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future, *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)


def model_methods(scope=None):
    namespace = dict(os=os, get_is_capture_mode=lambda: False,
                     logger=logging.getLogger(__name__), LogitsProcessorOutput=NS)
    namespace.update(scope or {})
    names = {"forward", "_pd_trunk_prefill_graph_runner", "_run_pd_trunk_prefill_graph",
             "_pd_trunk_prefill_graph_forward", "_alloff_prefill_graph_runner",
             "_p_trunk", "_emit_ids", "_sub_batch", "_decode_batch", "_boundary_lens"}
    result = methods(MODEL, "Qwen4ExpForConditionalGeneration", names, namespace)
    return type("CPUOwner", (), result), namespace


def cpu_runner(owner, attention_backend, *, attention_layers=(), padded_tokens=None):
    """Production run/can_run/context/slicing methods with CPU static buffers."""
    import torch
    from sglang.srt.models.flash_next_duet.prefill_graph import _Body
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (
        set_tc_piecewise_forward_context,
    )
    namespace = dict(os=os, torch=torch, ShapeKey=NS, PPProxyTensors=type("UnusedProxy", (), {}),
                     contextmanager=contextmanager,
                     ForwardContext=ForwardContext, forward_context=forward_context,
                     set_tc_piecewise_forward_context=set_tc_piecewise_forward_context)
    tree = ast.parse(PREFILL_RUNNER.read_text())
    compile_nodes(PREFILL_RUNNER, [next(n for n in tree.body if getattr(n, "name", "") == "_slice_output_rows")], namespace)
    values = methods(GRAPH, "TwinStarPrefillRunner",
                     {"run", "can_run", "_prepare_forward_metadata_for_replay"}, namespace)
    values.update(methods(PREFILL_RUNNER, "PrefillCudaGraphRunner", {"_prefill_forward_context"},
                          namespace, keep_decorators=True))
    runner = type("CPURunner", (), values)()
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
            setattr(static, key, torch.cat((value, value.new_zeros(size-raw))))
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
    def is_extend(self): return self.kind != "decode"
    def is_mixed(self): return self.kind == "mixed"
    def is_target_verify(self): return self.kind == "verify"
    def is_draft_extend_v2(self): return self.kind == "draft"


class PDTrunkGateTest(unittest.TestCase):
    """These routing tests run without torch or SGLang imports."""
    def setUp(self):
        self.enabled = False
        cls, self.scope = model_methods()
        self.owner = cls()
        self.owner.pd_trunk_prefill_graph = True
        self.owner.fullstack = dict(duet_spec={}, prefill_layer_trim=False, gdn_rank=0,
                                    prefill_saving_policy="kv-and-ssm", qsa_code="off")
        self.runner = NS(body=NS(trunk=True, emit_ids=[]), can_run=lambda fb: True)
        self.owner._prefill_runners = dict(trunk=self.runner)
        self.fb = NS(forward_mode=Mode(), spec_info=None,
                     input_ids=NS(shape=(3,)), extend_seq_lens_cpu=[3])
        self.scope["envs"] = NS(SGLANG_FLASHNEXT_PD_TRUNK_PREFILL_GRAPH=NS(get=lambda: self.enabled))
        tree = ast.parse(MODEL.read_text())
        compile_nodes(MODEL, [next(n for n in tree.body if getattr(n, "name", "") == "_pd_trunk_prefill_graph_enabled")], self.scope)
        self.gate = self.scope["_pd_trunk_prefill_graph_enabled"]

    def test_default_and_all_four_arm_role_gates(self):
        for role in ("prefill", "decode", "null"):
            args = NS(disaggregation_mode=role, is_embedding=False, pp_size=1)
            for trim, rank in ((False, 0), (False, 8), (True, 0), (True, 8)):
                fs = {**self.owner.fullstack, "prefill_layer_trim": trim, "gdn_rank": rank}
                for flag in (False, True):
                    self.enabled = flag
                    self.assertEqual(self.gate(fs, args), flag and role == "prefill")
        self.enabled = True
        for fs in (None, {}, {**self.owner.fullstack, "qsa_code": "on"},
                   {**self.owner.fullstack, "prefill_saving_policy": "latent-and-ssm"}):
            self.assertFalse(self.gate(fs, NS(disaggregation_mode="prefill", is_embedding=False, pp_size=1)))
        for embedding, pp in ((True, 1), (False, 2)):
            self.assertFalse(self.gate(self.owner.fullstack, NS(disaggregation_mode="prefill", is_embedding=embedding, pp_size=pp)))

    def test_env_declaration_is_independent_default_off(self):
        text = (ROOT / "python/sglang/srt/environ.py").read_text()
        self.assertIn("SGLANG_FLASHNEXT_PD_TRUNK_PREFILL_GRAPH = EnvBool(False)", text)
        self.assertIn("SGLANG_FLASHNEXT_ALLOFF_PREFILL_GRAPH_PD = EnvBool(False)", text)
        node = next(n for n in ast.parse(MODEL.read_text()).body if getattr(n, "name", "") == "_alloff_prefill_graph_enabled")
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
        for field, value in (("forward_mode", Mode("decode")), ("forward_mode", Mode("mixed")),
                             ("forward_mode", Mode("verify")), ("forward_mode", Mode("draft")),
                             ("spec_info", object()), ("input_embeds", object()),
                             ("can_run_tbo", True), ("tbo_parent_token_range", (0, 1)),
                             ("tbo_split_seq_index", 0), ("extend_seq_lens_cpu", [2])):
            fb = copy.copy(self.fb); setattr(fb, field, value)
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


if __name__ == "__main__":
    unittest.main()
