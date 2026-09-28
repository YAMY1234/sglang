"""Whole-prefix publication must precede either model's recurrent tail."""

import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor import gdn_prefill_model_split as split


class Pool:
    def __init__(self, order):
        self.cfg = NS(strict_chunk=1, init_method="k31", factored_prefix=1)
        self.layer_ids = [i for i in range(48) if i % 4 != 3]
        self.num_layers = 36
        self.order = order
        self.count = torch.zeros(36, dtype=torch.int64)
        self.collected = []
        self.plan = None

    def prefix_layer_count(self):
        return 36

    @contextmanager
    def collect_prefill_batch(self, plan):
        self.plan = plan
        self.order.append("collect")
        yield
        if self.collected != list(range(36)):
            raise RuntimeError("missing prefix layers")
        self.count.fill_(8)
        self.order.append("flush")


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


class ModelSplitTest(unittest.TestCase):
    def test_model_runner_entry_installs_and_prewarms_before_capture(self):
        path = Path(split.__file__).with_name("model_runner.py")
        tree = ast.parse(path.read_text())
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                     and node.name == "ModelRunner")
        method = next(node for node in owner.body if isinstance(node, ast.FunctionDef)
                      and node.name == "init_cuda_graphs")
        imports = [node for node in tree.body if isinstance(node, ast.Import)
                   and any(alias.name == "os" for alias in node.names)]
        order = []
        captured = NS(eager_runner=object(), prefill=NS(runner=object()),
                      decode=NS(runner=object()), memory_usage=12, time_usage=3)
        namespace = {"capture_cuda_graphs": lambda **kwargs:
                     (order.append("capture"), captured)[1]}
        exec(compile(ast.Module(body=imports + [method], type_ignores=[]),
                     str(path), "exec"), namespace)
        pool = NS(prewarm_commit_graph=lambda: order.append("prewarm"))
        runner = NS(req_to_token_pool=NS(factored_gdn_pool=pool),
                    server_args=NS(disaggregation_mode="prefill"), model=object())
        with patch.dict("os.environ", {split.FLAG: "1"}), \
             patch.object(split, "install_prefill_model_split",
                          side_effect=lambda model: order.append("install")):
            namespace["init_cuda_graphs"](runner)
            self.assertEqual(order, ["install", "prewarm", "capture"])
            self.assertIs(runner.eager_runner, captured.eager_runner)
            order.clear()
            runner.server_args.disaggregation_mode = "decode"
            namespace["init_cuda_graphs"](runner)
            self.assertEqual(order, ["capture"])

    def setup_case(self):
        order = []
        pool = Pool(order)
        saved = object()
        linear = NS(factored=pool, forward_metadata=saved)
        plans = []

        def initialize(prefix):
            plan = NS(next_layer=0, last_layer=0, pending=[])
            plans.append(plan)
            linear.forward_metadata = NS(factored_extend=plan)
            order.append("prefix_metadata")

        linear.init_forward_metadata = Mock(side_effect=initialize)
        full = NS(init_forward_metadata=Mock(side_effect=lambda b: order.append("qsa_" + b.kind)))
        backend = NS(linear_attn_backend=linear, full_attn_backend=full)

        def tail_metadata(tail):
            self.assertEqual(order[-1], "flush")
            self.assertEqual(tail.kind, "tail")
            linear.forward_metadata = object()
            full.init_forward_metadata(tail)

        backend.init_forward_metadata = Mock(side_effect=tail_metadata)
        ids = torch.arange(1, 7)
        positions = torch.tensor([0, 1, 2, 0, 1, 2])
        prefix_indices = torch.tensor([0, 1, 3, 4])
        tail_indices = torch.tensor([2, 5])
        prefix = NS(kind="prefix", input_ids=ids[prefix_indices], positions=positions[prefix_indices],
                    batch_size=2, extend_seq_lens_cpu=[2, 2], extend_prefix_lens_cpu=[0, 0])
        tail = NS(kind="tail", input_ids=ids[tail_indices], positions=positions[tail_indices],
                  batch_size=2, mamba_track_mask=torch.zeros(2, dtype=torch.bool))
        batch = NS(kind="original", input_ids=ids, positions=positions, batch_size=2,
                   extend_prefix_lens_cpu=[0, 0])
        owner = NS(_boundary_lens=lambda b: [1, 1],
                   _sub_batch=lambda *a: (prefix, prefix_indices),
                   _gdn_prefill_model_split_stats={"1+2": 0, "factor": 0},
                   fullstack_v3_latent=False, n_twinstar=0, n_prefix=0,
                   config=NS(vocab_size=4))
        owner._publish_qsa_prefix = Mock(side_effect=lambda b, lengths: order.append("publish_qsa"))
        return NS(order=order, pool=pool, linear=linear, backend=backend, saved=saved,
                  plans=plans, ids=ids, positions=positions, prefix_indices=prefix_indices,
                  tail_indices=tail_indices, prefix=prefix, tail=tail, batch=batch, owner=owner)

    def test_shallow_prefix_emitters_flush_before_tail_and_boundary(self):
        c = self.setup_case()

        def hidden(owner, batch):
            if batch.kind == "prefix":
                c.order.append("shallow_prefix")
                c.pool.collected.extend(range(24))
            else:
                self.assertEqual(c.pool.count.tolist(), [8] * 36)
                self.assertFalse(batch.mamba_track_mask.any())
                c.pool.count[:24] += 1
                c.order.append("shallow_tail")
            return torch.zeros(batch.input_ids.numel(), 3), torch.ones(batch.input_ids.numel(), 3)

        def emit(owner, prefix, hidden, embeddings, backend, metadata):
            self.assertIs(metadata.factored_extend, c.pool.plan)
            self.assertIs(backend.linear_attn_backend.forward_metadata, metadata)
            c.order.append("emitters")
            c.pool.collected.extend(range(24, 36))

        def capture(tail, rows, hidden):
            self.assertEqual(rows.tolist(), [0, 1])
            self.assertEqual(c.pool.count.tolist(), [9] * 24 + [8] * 12)
            c.order.append("capture")

        pd = module("twinstar_sgl.pd_shallow", boundary_inputs=lambda *a: (c.tail, c.tail_indices),
                    capture_extend_boundary=Mock(side_effect=capture))
        pkg = module("twinstar_sgl", pd_shallow=pd)
        modules = {"twinstar_sgl": pkg, pd.__name__: pd,
                   "twinstar_sgl.pd_final_metadata": module("metadata", initialize_shallow=
                       lambda backend, prefix: backend.init_forward_metadata(prefix)),
                   "twinstar_sgl.pd_shallow_audit": module("audit", snapshot=Mock()),
                   "sglang.srt.layers.logits_processor": module("logits", LogitsProcessorOutput=NS)}
        with patch.dict(sys.modules, modules), patch.object(split, "_backend", return_value=c.backend), \
             patch.object(split, "_input_scope", return_value=nullcontext()), \
             patch.object(split, "_shallow_hidden", side_effect=hidden), \
             patch.object(split, "_emit_prefix", side_effect=emit):
            output = split._shallow_prefill(c.owner, c.ids, c.positions, c.batch)
        self.assertEqual(c.order, ["prefix_metadata", "qsa_prefix", "collect", "shallow_prefix",
                                   "emitters", "flush", "qsa_tail", "shallow_tail", "capture", "publish_qsa"])
        self.assertEqual(c.linear.init_forward_metadata.call_count, 1)
        self.assertEqual(c.plans[0].last_layer, 35)
        self.assertIs(c.linear.forward_metadata, c.saved)
        self.assertEqual(output.next_token_logits.shape, (2, 4))
        self.assertEqual(c.owner._gdn_prefill_model_split_stats["1+2"], 1)

    def test_full_depth_split_reassembles_outputs_and_hc_without_second_outer_forward(self):
        c = self.setup_case()
        body = NS(last_hc_hidden_states=None)

        def body_forward(ids, positions, batch):
            if batch.kind == "prefix":
                c.pool.collected.extend(range(36))
                c.order.append("full_prefix")
            else:
                self.assertEqual(c.pool.count.tolist(), [8] * 36)
                self.assertFalse(batch.mamba_track_mask.any())
                c.pool.count += 1
                c.order.append("full_tail")
            body.last_hc_hidden_states = ids[:, None] * 2
            return ids[:, None] * 10

        body.forward = body_forward
        c.owner.model = NS(model=body)
        outer = Mock(side_effect=lambda: body.forward(c.ids, c.positions, c.batch))
        with patch.object(split, "_input_scope", return_value=nullcontext()):
            with split._split_body(c.owner, c.batch, c.prefix, c.prefix_indices,
                                   c.tail, c.tail_indices, c.backend):
                output = outer()
        torch.testing.assert_close(output, c.ids[:, None] * 10)
        torch.testing.assert_close(body.last_hc_hidden_states, c.ids[:, None] * 2)
        self.assertEqual(c.pool.count.tolist(), [9] * 36)
        self.assertEqual(c.order, ["prefix_metadata", "qsa_prefix", "collect", "full_prefix",
                                   "flush", "qsa_tail", "full_tail"])
        self.assertIs(body.forward, body_forward)
        self.assertIs(c.linear.forward_metadata, c.saved)
        self.assertEqual(outer.call_count, 1)
        self.assertEqual(c.linear.init_forward_metadata.call_count, 1)

    def test_failed_prefix_does_not_publish_or_run_tail_and_restores_body(self):
        c = self.setup_case()

        def fail(*args):
            raise RuntimeError("prefix failure")

        body = NS(forward=fail, last_hc_hidden_states=None)
        c.owner.model = NS(model=body)
        with patch.object(split, "_input_scope", return_value=nullcontext()):
            with self.assertRaisesRegex(RuntimeError, "prefix failure"):
                with split._split_body(c.owner, c.batch, c.prefix, c.prefix_indices,
                                       c.tail, c.tail_indices, c.backend):
                    body.forward(c.ids, c.positions, c.batch)
        self.assertNotIn("flush", c.order)
        c.backend.init_forward_metadata.assert_not_called()
        self.assertIs(body.forward, fail)
        self.assertIs(c.linear.forward_metadata, c.saved)

    def test_partial_factor_layer_configuration_is_rejected(self):
        c = self.setup_case()
        c.pool.prefix_layer_count = lambda: 24
        with self.assertRaisesRegex(RuntimeError, "every prefix layer"):
            split._prefix_metadata(c.backend, c.prefix)
        c.linear.init_forward_metadata.assert_not_called()

    def test_unsupported_batches_keep_legacy_path_with_reason(self):
        owner = NS(bridges=[], model=NS(language_model_only=True))
        mode = NS(is_extend=lambda: True, is_mixed=lambda: False)
        batch = NS(forward_mode=mode, spec_info=None, extend_seq_lens_cpu=[2, 2],
                   twinstar_prompt_final=[True, True], batch_size=2, return_logprob=False)
        with patch.dict("os.environ", {"SGLANG_FLASHNEXT_ARRIVAL_OVERLAP": "0"}):
            self.assertIsNone(split._unsupported(owner, batch, torch.zeros(4)))
            batch.twinstar_prompt_final = [True, False]
            self.assertIsNone(split._unsupported(owner, batch, torch.zeros(4)))
            batch.twinstar_prompt_final = [True, True]
            batch.extend_seq_lens_cpu = [1, 3]
            self.assertEqual(split._unsupported(owner, batch, torch.zeros(4)), "empty-prefix")
            batch.batch_size = 17
            self.assertEqual(split._unsupported(owner, batch, torch.zeros(4)), "batch-bucket")

    def test_mixed_final_keeps_unfinished_state_and_reassembles_every_token(self):
        c = self.setup_case()
        c.pool.count = torch.zeros(36, 2, dtype=torch.int64)
        c.batch.twinstar_prompt_final = [True, False]
        c.batch.rids = ["finishing", "continuing"]
        prefix_indices, tail_indices = torch.tensor([0, 1, 3, 4, 5]), torch.tensor([2])
        c.prefix.input_ids = c.ids[prefix_indices]
        c.prefix.positions = c.positions[prefix_indices]
        c.prefix.extend_seq_lens_cpu = [2, 3]
        c.prefix.twinstar_prompt_final = c.batch.twinstar_prompt_final[:]
        c.tail.input_ids = c.ids[tail_indices]
        c.tail.positions = c.positions[tail_indices]
        c.tail.batch_size = 1
        c.tail.mamba_track_mask = torch.zeros(1, dtype=torch.bool)
        split._tail_rids(c.batch, c.tail)
        self.assertEqual(c.tail.rids, ["finishing"])
        body = NS(last_hc_hidden_states=None)

        def forward(ids, positions, batch):
            if batch is c.prefix:
                self.assertEqual(batch.twinstar_prompt_final, [True, False])
                c.pool.collected.extend(range(36))
                c.order.append("mixed_prefix")
            else:
                self.assertEqual(batch.rids, ["finishing"])
                self.assertTrue(torch.all(c.pool.count == 8))
                c.pool.count[:, 0] += 1
                c.order.append("mixed_tail")
            body.last_hc_hidden_states = ids[:, None] * 3
            return ids[:, None] * 7

        body.forward = forward
        c.owner.model = NS(model=body)
        with patch.object(split, "_input_scope", return_value=nullcontext()):
            with split._split_body(c.owner, c.batch, c.prefix, prefix_indices,
                                   c.tail, tail_indices, c.backend):
                output = body.forward(c.ids, c.positions, c.batch)
        torch.testing.assert_close(output, c.ids[:, None] * 7)
        torch.testing.assert_close(body.last_hc_hidden_states, c.ids[:, None] * 3)
        self.assertTrue(torch.all(c.pool.count[:, 0] == 9))
        self.assertTrue(torch.all(c.pool.count[:, 1] == 8))
        self.assertEqual(c.linear.init_forward_metadata.call_count, 1)

    def test_text_requests_do_not_require_language_model_only_config(self):
        owner = NS(model=NS(language_model_only=False))
        mode = NS(is_extend=lambda: True, is_mixed=lambda: False)
        batch = NS(forward_mode=mode, spec_info=None, extend_seq_lens_cpu=[4],
                   twinstar_prompt_final=[True], batch_size=1, mm_inputs=[None])
        with patch.dict("os.environ", {"SGLANG_FLASHNEXT_ARRIVAL_OVERLAP": "0"}):
            self.assertIsNone(split._unsupported(owner, batch, torch.zeros(4), factor=True))
            batch.mm_inputs = [object()]
            self.assertEqual(split._unsupported(owner, batch, torch.zeros(4), factor=True), "multimodal")

    def test_emitter_collection_uses_native_layers_and_preserves_latent_filter(self):
        c = self.setup_case()
        called = []
        emitters = {str(i): NS(is_attn=i % 4 == 3,
                    emit=lambda streams, batch, i=i: called.append((i, streams, batch)))
                    for i in range(31, 48)}
        owner = NS(fullstack_v3_latent=False, fullstack={"latent": "off"},
                   fullstack_final=True, emitters=emitters, emitter_ids=list(range(31, 48)))
        hidden, embeddings, decoded = object(), object(), object()
        metadata = object()
        with patch.object(split, "_input_scope", return_value=nullcontext()):
            split._emit_prefix(owner, c.prefix, hidden, embeddings, c.backend, metadata)
        self.assertEqual([i for i, _, _ in called], list(range(31, 48)))
        self.assertTrue(all(s is hidden and b is c.prefix for _, s, b in called))
        self.assertIs(c.linear.forward_metadata, metadata)
        called.clear()
        owner.fullstack_v3_latent = True
        encode = Mock(return_value=decoded)
        with patch.dict(sys.modules, {"twinstar_sgl.fullstack_v3_serving": module("codec", encode_prefix=encode)}), \
             patch.object(split, "_input_scope", return_value=nullcontext()):
            split._emit_prefix(owner, c.prefix, hidden, embeddings, c.backend, metadata)
        encode.assert_called_once_with(owner, hidden, embeddings, c.prefix)
        self.assertEqual([i for i, _, _ in called], [i for i in range(31, 48) if i % 4 != 3])
        self.assertTrue(all(s is decoded and b is c.prefix for _, s, b in called))

    def test_phase_observer_delegates_without_synchronization(self):
        events = []
        def observe(name, function, args, kwargs, *, batch):
            events.append((name, batch))
            return function(*args, **kwargs)
        batch = object()
        with patch.dict("os.environ", {"TWINSTAR_CUDA_TIMELINE": "enabled"}), \
             patch.dict(sys.modules, {"twinstar_sgl.pd_boundary_observe": module("observe", call=observe)}):
            self.assertEqual(split._observe("P_model_tail", lambda x: x + 1, batch, 4), 5)
        self.assertEqual(events, [("P_model_tail", batch)])

    def test_installer_is_prefill_only_idempotent_and_does_not_intercept_other_models(self):
        legacy = Mock(return_value="legacy")
        external = module("pd_shallow", prefill_extend=legacy)
        owner = NS(n_layers=48, p_layer_ids=list(range(31)), pd_shallow_role="prefill")
        context = module("runtime", get_disagg=lambda: NS(disaggregation_mode="decode"))
        with patch.dict("os.environ", {split.FLAG: "1", "TWINSTAR_PD_FACTOR_ONLY_TAIL": "0"}), \
             patch.dict(sys.modules, {"sglang.srt.runtime_context": context}), \
             patch.object(split.importlib, "import_module", return_value=external):
            self.assertFalse(split.install_prefill_model_split(owner))
            context.get_disagg = lambda: NS(disaggregation_mode="prefill")
            self.assertTrue(split.install_prefill_model_split(owner))
            installed = external.prefill_extend
            self.assertTrue(split.install_prefill_model_split(owner))
            self.assertIs(external.prefill_extend, installed)
            self.assertEqual(installed(object(), None, None, None), "legacy")
            with patch.object(split, "_unsupported", return_value="empty-prefix"), \
                 patch.object(split, "_shallow_prefill") as new:
                self.assertEqual(installed(owner, None, None, None), "legacy")
                new.assert_not_called()
            self.assertEqual(owner._gdn_prefill_model_split_stats["fallback_empty-prefix"], 1)


if __name__ == "__main__":
    unittest.main()
