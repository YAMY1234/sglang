"""Full-N checkpoint deferral retains live tails and publishes only complete states."""
import ast
import copy
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache import gdn_factored_pool as native
from sglang.srt.mem_cache import gdn_prefill_tracked_graph as tracked


def fixture(batch=1, rows=1):
    graph = NS(warmed=True, run=Mock())
    layers = [i for i in range(48) if i % 4 != 3]
    pool = NS(layer_ids=layers, layer_map={lid: i for i, lid in enumerate(layers)},
              pside_join=Mock(), invalidate_prefix_dense=Mock(), _prefill_tracked_graph=graph)
    request_pool = NS(req_generation=torch.arange(32))
    forward = NS(req_pool_indices_cpu=torch.arange(batch),
                 forward_mode=NS(is_extend=lambda: True, is_mixed=lambda: False))
    plan = NS(slots=torch.arange(1, batch + 1), next_layer=0, last_layer=35, pending=[])
    metadata = NS(factored_extend=plan, has_mamba_track_mask=True,
                  track_ssm_h_dst=torch.arange(20, 20 + rows),
                  track_ssm_final_src=torch.arange(1, batch + 1),
                  track_ssm_final_dst=torch.arange(40, 40 + batch))
    return NS(pool=pool, graph=graph, request_pool=request_pool, forward=forward,
              plan=plan, metadata=metadata)


class TrackedGraphTest(unittest.TestCase):
    def test_both_model_depths_keep_full_n_calls_and_flush_after_final_model_layer(self):
        for shallow in (True, False):
            with self.subTest(shallow=shallow):
                c = fixture(); order = []; c.pool.cfg = NS(strict_chunk=1, factored_prefix=1, init_method="k31")
                c.pool.prefix_dense = None; c.pool.prefix_layer_count = lambda: 36; c.pool.batch_prefill = True
                def forward(ids, positions, batch, **kwargs):
                    self.assertIs(batch, c.forward)
                    self.assertEqual(ids.numel(), 8)
                    tracked.prepare(c.pool, c.metadata)
                    self.assertIs(c.plan.tracked_transaction, c.pool._tracked_transaction)
                    for layer in range(31 if shallow else 48):
                        order.append(("model", layer, ids.numel()))
                        if layer % 4 != 3:
                            # The real PD adapter shallow-copies the prefix plan per GDN layer.
                            plan = copy.copy(c.plan)
                            state = torch.full((1, 2, 16, 16), float(layer))
                            plan.tracked_transaction.add(layer, state, c.metadata.track_ssm_h_dst)
                        self.assertFalse(c.graph.run.called)
                    if shallow:
                        for layer in range(31, 48):
                            order.append(("emitter", layer, 7))
                            if layer % 4 != 3:
                                c.plan.tracked_transaction.add(layer, torch.full((1, 2, 16, 16), float(layer)),
                                                               c.metadata.track_ssm_h_dst)
                    order.append("model_done")
                    return "native-output"
                def flush(pool, states, slots, **kw):
                    self.assertEqual(order[-1], "model_done")
                    self.assertEqual([int(s[0, 0, 0, 0]) for s in states], c.pool.layer_ids)
                    self.assertEqual(slots.tolist(), [20])
                    order.append("tracked_flush")
                c.graph.run.side_effect = flush
                owner = NS(forward=forward, pd_shallow_role="prefill" if shallow else None)
                c.request_pool.factored_gdn_pool = c.pool
                runner = NS(model=owner, req_to_token_pool=c.request_pool,
                            server_args=NS(disaggregation_mode="prefill"))
                env = {tracked.FLAG: "1", "TWINSTAR_PD_FACTOR_ONLY_TAIL": str(int(not shallow)),
                       "SGLANG_GDN_PREFILL_COMMIT_GRAPH": "1"}
                with patch.dict("os.environ", env, clear=True), \
                     patch("sglang.srt.runtime_context.get_schedule", return_value=NS(disable_overlap_schedule=True)):
                    tracked.install(runner)
                    self.assertEqual(owner.forward(torch.arange(8), torch.arange(8), c.forward), "native-output")
                self.assertEqual(order[-1], "tracked_flush")
                self.assertEqual(c.graph.run.call_count, 1)
                self.assertIsNone(c.pool._tracked_transaction)

    def test_normal_each_layer_is_immediate_and_does_not_reuse_tracked(self):
        c = fixture(); pool = native.FactoredGDNPool.__new__(native.FactoredGDNPool)
        pool.layer_map = c.pool.layer_map; pool.cfg = NS(factored_prefix=1)
        pool.batch_prefill_max_bytes = 1 << 30; pool.invalidate_prefix_dense = Mock()
        normal, checkpoint = torch.ones(1, 2, 16, 16), torch.full((1, 2, 16, 16), 9.)
        calls = []
        def commit(layer, plan, dense, hs, slots, src, dst):
            self.assertIs(dense, normal); self.assertIsNone(hs); self.assertIsNone(slots)
            self.assertEqual(len(plan.pending), 1)
            calls.append(layer); plan.pending.clear()
        pool._commit_extend_group = commit
        with tracked.TrackedTransaction(c.pool, c.request_pool, c.forward) as transaction:
            transaction.prepare(c.plan, c.metadata)
            for i, layer in enumerate(c.pool.layer_ids):
                pool.commit_extend_batched(layer, c.plan, normal, checkpoint, c.metadata.track_ssm_h_dst)
                self.assertEqual(calls, c.pool.layer_ids[:i + 1])
                self.assertIs(transaction.states[-1], checkpoint)
        self.assertEqual(c.graph.run.call_count, 1)

    def test_deferral_keeps_normal_padding_chosen_by_original_tracked_count(self):
        from sglang.srt.mem_cache import gdn_prefill_commit_graph as commit
        c = fixture(batch=3, rows=9)
        with tracked.TrackedTransaction(c.pool, c.request_pool, c.forward) as transaction:
            transaction.prepare(c.plan, c.metadata)
            self.assertEqual(c.plan.prefill_normal_bucket, 16)
            self.assertEqual(transaction.bucket, 16)
            c.pool.cfg = native.FactoredGDNConfig(init_method="k31", factored_prefix=1)
            c.pool.prefix_dense = None
            graph = commit.PrefillCommitGraph()
            c.plan.pending = [None]
            c.plan.ring_dst = torch.zeros(3, dtype=torch.long)
            stream = Mock()
            with ExitStack() as stack:
                stack.enter_context(patch.object(torch.Tensor, "is_cuda", property(lambda _: True)))
                stack.enter_context(patch.object(commit, "k31_graph_safe", return_value=True))
                stack.enter_context(patch.object(commit, "CommitBuffers", return_value=Mock()))
                for name, value in (("is_current_stream_capturing", False), ("current_stream", stream),
                                    ("Stream", stream), ("CUDAGraph", Mock())):
                    stack.enter_context(patch.object(torch.cuda, name, return_value=value))
                for name in ("stream", "graph"):
                    stack.enter_context(patch.object(torch.cuda, name, side_effect=lambda *a, **k: nullcontext()))
                self.assertTrue(graph.run(c.pool, 0, c.plan, torch.zeros(3, 2, 16, 16), None, None,
                                          eager=Mock(), policy=()))
            self.assertEqual(next(iter(graph.entries))[1][0][0][0], 16)
            transaction.fallback = "test did not execute model"

    def test_aliases_use_original_path_and_never_publish_deferred_states(self):
        for slots in ([1], [40], [20, 20]):
            c = fixture(); c.metadata.track_ssm_h_dst = torch.tensor(slots)
            with tracked.TrackedTransaction(c.pool, c.request_pool, c.forward) as transaction:
                transaction.prepare(c.plan, c.metadata)
                self.assertIsNotNone(transaction.fallback)
                self.assertFalse(hasattr(c.plan, "tracked_transaction"))
            self.assertFalse(c.graph.run.called)

    def test_recycled_generation_changed_controls_missing_layers_and_exception_fail_closed(self):
        for error in ("generation", "controls", "missing", "exception"):
            c = fixture()
            with self.subTest(error=error), self.assertRaises((RuntimeError, ValueError)):
                with tracked.TrackedTransaction(c.pool, c.request_pool, c.forward) as transaction:
                    transaction.prepare(c.plan, c.metadata)
                    for layer in c.pool.layer_ids[:-1] if error == "missing" else c.pool.layer_ids:
                        transaction.add(layer, torch.zeros(1, 2, 16, 16), c.metadata.track_ssm_h_dst)
                    if error == "generation":c.request_pool.req_generation[0] += 1
                    if error == "controls":c.metadata.track_ssm_h_dst.add_(1)
                    if error == "exception":raise ValueError("native model failed")
            self.assertFalse(c.graph.run.called)
            self.assertIsNone(c.pool._tracked_transaction)

    def test_inference_tensors_and_repeated_request_slots_keep_independent_destinations(self):
        c = fixture()
        with torch.inference_mode():
            c.metadata.track_ssm_h_dst = torch.tensor([20])
            for destination in (20, 21):
                c.plan = NS(slots=torch.tensor([1]), next_layer=0, last_layer=35, pending=[])
                c.metadata.factored_extend = c.plan
                c.metadata.track_ssm_h_dst = torch.tensor([destination])
                with tracked.TrackedTransaction(c.pool, c.request_pool, c.forward) as transaction:
                    transaction.prepare(c.plan, c.metadata)
                    for layer in c.pool.layer_ids:
                        transaction.add(layer, torch.zeros(1, 2, 16, 16), c.metadata.track_ssm_h_dst)
                self.assertEqual(c.graph.run.call_args.args[2].tolist(), [destination])
                c.request_pool.req_generation[0] += 1

    def test_slot_reset_is_rejected_during_transaction(self):
        pool = native.FactoredGDNPool.__new__(native.FactoredGDNPool)
        pool._tracked_transaction = object()
        with self.assertRaisesRegex(RuntimeError, "recycle"):
            pool.reset_slots(torch.tensor([1]))

    def test_all_five_buckets_prewarms_once_and_fp32_bf16_rebind_without_capture(self):
        pool = NS(layer_ids=list(range(36)), hv=2, v=16, k=16, a=torch.zeros(36, 2, 2, 16),
                  init_omega=lambda b: None)
        graph = tracked.TrackedGraph(); eager = Mock(); cuda_graph = Mock(); stream = Mock()
        with ExitStack() as stack:
            for name, result in (("memory_allocated", 0), ("current_stream", stream), ("Stream", stream),
                                 ("graph_pool_handle", object()), ("CUDAGraph", cuda_graph), ("synchronize", None)):
                stack.enter_context(patch.object(torch.cuda, name, return_value=result))
            for name in ("stream", "graph"):
                stack.enter_context(patch.object(torch.cuda, name, side_effect=lambda *a, **k: nullcontext()))
            stack.enter_context(patch.object(tracked.TrackedBuffers, "evaluate"))
            graph.prewarm(pool, eager=eager, policy=())
            self.assertEqual({k[0] for k in graph.entries}, {1, 2, 4, 8, 16})
            self.assertEqual(graph.stats["captured"], 5)
            for batch in (1, 8, 16):
                for dtype in (torch.bfloat16, torch.float32):
                    states = [torch.randn(batch, 2, 16, 16).to(dtype) for _ in range(36)]
                    graph.run(pool, states, torch.arange(batch), batch=batch, eager=eager, policy=())
                    buffers = graph.entries[graph.key(batch, eager, ())][0]
                    self.assertTrue(torch.equal(buffers.states[0], states[0].float()))
                    self.assertEqual(buffers.states[0].dtype, torch.float32)
            self.assertEqual(graph.stats["captured"], 5)
            with self.assertRaisesRegex(RuntimeError, "missing"):
                graph.run(pool, states, torch.arange(16), batch=16, eager=eager, policy=("changed",))

    def test_padded_rebind_cannot_publish_previous_slots(self):
        pool = NS(a=torch.zeros(1), init_omega=lambda b: None)
        buffers = tracked.TrackedBuffers(pool, 8, [torch.ones(16, 2, 16, 16) for _ in range(36)])
        buffers.bind([torch.ones(8, 2, 16, 16) for _ in range(36)], torch.arange(8))
        buffers.bind([torch.full((3, 2, 16, 16), 2.) for _ in range(36)], torch.tensor([40, 41, 42]))
        self.assertEqual(buffers.slots.tolist(), [40, 41, 42, -1, -1, -1, -1, -1])
        self.assertTrue(all(torch.count_nonzero(s[3:]) == 0 for s in buffers.states))

    def test_model_runner_installs_full_n_adapter_before_startup_prewarm(self):
        from sglang.srt.model_executor import gdn_prefill_model_split as split
        path = Path(split.__file__).with_name("model_runner.py")
        tree = ast.parse(path.read_text())
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner")
        method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "init_cuda_graphs")
        imports = [n for n in tree.body if isinstance(n, ast.Import) and any(a.name == "os" for a in n.names)]
        order = []
        capture = NS(eager_runner=None, prefill=NS(runner=None), decode=NS(runner=None),
                     memory_usage=0, time_usage=0)
        namespace = {"capture_cuda_graphs": lambda **k: (order.append("capture"), capture)[1]}
        exec(compile(ast.Module(body=imports + [method], type_ignores=[]), str(path), "exec"), namespace)
        runner = NS(req_to_token_pool=NS(factored_gdn_pool=NS(prewarm_commit_graph=lambda: order.append("prewarm"))),
                    server_args=NS(disaggregation_mode="prefill"), model=object())
        with patch.dict("os.environ", {tracked.FLAG: "1"}, clear=True), \
             patch.object(tracked, "install", side_effect=lambda r: order.append("tracked")), \
             patch.object(split, "install_prefill_model_split") as split_install:
            namespace["init_cuda_graphs"](runner)
        self.assertEqual(order, ["tracked", "prewarm", "capture"])
        split_install.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_all_layer_replay_matches_independent_checkpoints_and_publication(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_factored_io import store_factored
        cfg = native.FactoredGDNConfig(init_method="k31", dtype=torch.float16, factored_prefix=1)
        layers, heads, width, capacity = 36, 2, 16, 48
        vbar = torch.nn.functional.normalize(torch.randn(layers, heads, width, device="cuda"), dim=-1)
        omega = torch.randn(1, heads, width, cfg.r + 8, device="cuda")
        def make_pool():
            return NS(cfg=cfg, layer_ids=list(range(layers)), hv=heads, v=width, k=width, vbar=vbar,
                      init_omega=lambda b: omega.expand(b, -1, -1, -1),
                      a=torch.full((layers, capacity, heads, width), .125, device="cuda"),
                      U=torch.zeros(layers, capacity, heads, cfg.rmax, width, dtype=cfg.dtype, device="cuda"),
                      W=torch.zeros(layers, capacity, heads, cfg.rmax, width, dtype=cfg.dtype, device="cuda"),
                      count=torch.zeros(layers, capacity, heads, dtype=torch.int32, device="cuda"),
                      stale=torch.zeros(capacity, dtype=torch.int32, device="cuda"),
                      dense_of=torch.full((capacity,), 7, dtype=torch.int32, device="cuda"),
                      prefix_valid=torch.zeros(capacity, dtype=torch.int32, device="cuda"))
        pool, reference = make_pool(), make_pool(); graph = tracked.TrackedGraph()
        policy = (native.ORTH_METHOD, native.ORTH_WARPS_OVERRIDE, native.factorize_dense)
        graph.prewarm(pool, eager=native.factorize_layers, policy=policy)
        for bucket in (1, 8, 16):
            for rows in (bucket, max(1, bucket // 2)):
                slots = torch.arange(20, 20 + rows, device="cuda")
                states = [torch.randn(rows, heads, width, width, device="cuda", dtype=torch.bfloat16)
                          for _ in range(layers)]
                for i, state in enumerate(states):
                    padded = torch.zeros(bucket, heads, width, width, device="cuda")
                    padded[:rows].copy_(state)
                    values = native.factorize_layers([padded], vbar[i:i+1], cfg,
                                                      omega=reference.init_omega(bucket))[0]
                    indices = torch.full((bucket,), -1, dtype=torch.long, device="cuda")
                    indices[:rows].copy_(slots)
                    store_factored(*values, reference.a[i], reference.U[i], reference.W[i], reference.count[i],
                                   reference.stale, reference.dense_of, indices, cfg.r, stale_value=1)
                reference.prefix_valid[slots] = 1
                graph.run(pool, states, slots, batch=bucket, eager=native.factorize_layers, policy=policy)
                torch.cuda.synchronize()
                for name in ("a", "U", "W", "count", "stale", "dense_of", "prefix_valid"):
                    tolerance = 2e-3 if name in ("U", "W") else 5e-6 if name == "a" else 0
                    torch.testing.assert_close(getattr(pool, name), getattr(reference, name),
                                               rtol=tolerance, atol=tolerance)
                self.assertEqual(graph.stats["captured"], 5)
                self.assertTrue(torch.all(pool.a[:, :20] == .125).item())

    def test_thirty_six_layer_algebra_matches_per_layer(self):
        torch.set_num_threads(1); torch.manual_seed(1235)
        cfg = native.FactoredGDNConfig(init_method="k31", dtype=torch.float32)
        states = [torch.randn(1, 2, 16, 16) for _ in range(36)]
        vbar, omega = torch.randn(36, 2, 16), torch.randn(1, 2, 16, 16)
        together = native.factorize_layers(states, vbar, cfg, omega=omega)
        for i, state in enumerate(states):
            expected = native.factorize_layers([state], vbar[i:i + 1], cfg, omega=omega)[0]
            for a, b in zip(together[i], expected):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
