"""2c event ordering uses stub streams; graph-body parity uses real CPU tensors."""
import ast
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch

os.environ.setdefault("TRITON_INTERPRET", "1")
from test_gdn_tracked_factor_side import (
    BASE, POOL, SIDE, FakeCuda, controller, extract, function,
)


import test_gdn_tracked_factor_side as tracked_tests


class BoundaryOrderTest(unittest.TestCase):
    launch = tracked_tests.SideDispatchTest.launch

    def test_T_waits_for_boundary_and_hook_is_idempotent(self):
        side, cuda = controller(deferred=True, after_boundary=True)
        self.launch(side)
        self.assertEqual([r[0] for r in cuda.trace], ["bind", "record", "final"])
        self.assertTrue(side._prefill_side_pending)
        cuda.trace.append(("boundary", cuda.main))
        self.assertTrue(side.launch_pending())
        self.assertFalse(side.launch_pending())
        self.assertEqual([r[0] for r in cuda.trace],
                         ["bind", "record", "final", "boundary", "record", "wait", "tracked", "record"])
        self.assertEqual(cuda.trace[4], ("record", cuda.main, side.boundary_done))
        self.assertEqual(cuda.trace[5], ("wait", side.stream, side.boundary_done))
        self.assertEqual(side.stats["after_boundary"], 1)
        self.assertEqual(side.stats["early_reader"], 0)

    def test_early_reader_launches_before_join_on_each_stream(self):
        side, cuda = controller(deferred=True, after_boundary=True)
        self.launch(side)
        reader = cuda.Stream()
        with cuda.stream(reader):
            side.join(); side.join()
        self.assertEqual(cuda.trace[3], ("record", cuda.main, side.boundary_done))
        self.assertEqual(cuda.trace[-1], ("wait", reader, side.done))
        side.join()
        self.assertEqual(cuda.trace[-1], ("wait", cuda.main, side.done))
        self.assertEqual(side.stats["early_reader"], 1)

    def test_next_bind_drains_pending_and_double_buffer_fence_survives(self):
        side, cuda = controller(deferred=True, after_boundary=True)
        self.launch(side)
        self.launch(side)
        self.launch(side)
        names = [r[0] for r in cuda.trace]
        self.assertEqual(names.count("tracked"), 1)
        self.assertEqual(names.count("tracked_alt"), 1)
        self.assertEqual(side.stats["next_bind"], 2)
        last_bind = max(i for i, name in enumerate(names) if name == "bind")
        self.assertEqual(cuda.trace[last_bind - 1], ("wait", cuda.main, side.set_done[0]))
        side.launch_pending()
        side.join()
        self.assertEqual(cuda.trace[-1], ("wait", cuda.main, side.set_done[0]))

    def test_pending_capture_fails_before_any_launch_or_bind(self):
        side, cuda = controller(deferred=True, after_boundary=True)
        self.launch(side); before = list(cuda.trace)
        cuda.capturing = True
        with self.assertRaisesRegex(RuntimeError, "before graph capture"):
            side.join()
        self.assertFalse(self.launch(side))
        self.assertEqual(cuda.trace, before)
        cuda.capturing = False
        side.join()
        self.assertIsNone(side.pending_launch)

    def test_wrong_boundary_stream_rejected_and_writer_not_a_reader(self):
        side, cuda = controller(deferred=True, after_boundary=True)
        self.launch(side)
        with cuda.stream(cuda.Stream()):
            with self.assertRaisesRegex(RuntimeError, "changed the producer"):
                side.launch_pending()
        with cuda.stream(side.stream):
            side.join()
        self.assertIsNotNone(side.pending_launch)
        side.launch_pending()

    def test_low_priority_is_independent_and_does_not_change_main(self):
        real = FakeCuda.Stream
        calls = []
        def spy(cuda, device=None, **kw):
            calls.append((device, kw))
            return real(cuda, device, **kw)
        with patch.object(FakeCuda, "Stream", spy):
            side, cuda = controller(low_priority=True)
        self.assertIn(("cuda:0", {"priority": 0}), calls)
        self.assertEqual(side.stream.priority, 0)
        self.assertEqual(cuda.main.priority, 0)
        self.assertFalse(side.after_boundary)
        self.launch(side)  # independent of 2b/2c
        self.assertIn("tracked", [r[0] for r in cuda.trace])

    def test_flag_defaults_dependency_and_real_pool_wiring(self):
        text = (BASE / "environ.py").read_text()
        for suffix in ("AFTER_BOUNDARY", "LOW_PRIORITY"):
            self.assertIn("SGLANG_GDN_TRACKED_FACTOR_SIDE_STREAM_" + suffix + " = EnvBool(False)", text)
        with self.assertRaisesRegex(ValueError, "DEFERRED=1"):
            controller(after_boundary=True)
        # Execute the actual prewarm dependency guard before any CUDA allocation.
        import sys
        flag = lambda value: types.SimpleNamespace(get=lambda: value)
        env = types.ModuleType("sglang.srt.environ")
        env.envs = types.SimpleNamespace(**{
            "SGLANG_GDN_TRACKED_FACTOR_SIDE_STREAM": flag(True),
            "SGLANG_GDN_TRACKED_FACTOR_SIDE_STREAM_PD_P": flag(True),
            "SGLANG_GDN_TRACKED_FACTOR_SIDE_STREAM_AFTER_BOUNDARY": flag(True),
            "SGLANG_GDN_TRACKED_FACTOR_SIDE_STREAM_DEFERRED": flag(False),
        })
        scope = {}
        extract([function(POOL, "prewarm_k31_batch_graph")], scope)
        for role in ("null", "prefill"):
            with patch.dict(sys.modules, {env.__name__: env}):
                with self.assertRaisesRegex(ValueError, "DEFERRED=1"):
                    scope["prewarm_k31_batch_graph"](types.SimpleNamespace(), disaggregation_mode=role)

    def test_pool_hook_after_all_AGG_boundary_forms_and_before_replan(self):
        # Exercise the exact common hook and verify placement after both graph
        # branches / intermediate-no-boundary / eager branch, before replan.
        node = function(BASE / "models/flash_next_duet/model.py", "_twinstar_prefill")
        hooks = [n for n in ast.walk(node) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "launch_pending_tracked"]
        boundaries = [n for n in ast.walk(node) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Attribute) and n.func.attr == "_boundary_graph"]
        self.assertEqual(len(hooks), 1)
        self.assertGreater(hooks[0].lineno, max(n.lineno for n in boundaries))
        scope = {}; extract([function(POOL, "launch_pending_tracked")], scope)
        for boundary in (False, True):
            side, cuda = controller(deferred=True, after_boundary=True)
            self.launch(side)
            if boundary:
                cuda.trace.append(("boundary", cuda.main))
            scope["launch_pending_tracked"](types.SimpleNamespace(_tracked_factor_side=side))
            self.assertIsNone(side.pending_launch)
            self.assertEqual(side.stats["after_boundary"], 1)
        scope["launch_pending_tracked"](types.SimpleNamespace(_tracked_factor_side=None))

    def test_PD_tail_completion_then_T_optional_F_then_return_join(self):
        source = BASE / "mem_cache/gdn_pd_factor_deferred.py"
        for final in (False, True):
            side, cuda = controller(deferred=True, after_boundary=True)
            scope = dict(torch=cuda, graph_shape=lambda n, t: (n, t))
            extract([function(source, "launch_side"), function(source, "finish_return")], scope)
            slots = types.SimpleNamespace(numel=lambda: 1)
            scope["launch_side"](side, types.SimpleNamespace(slots=slots), [], (slots, None, None),
                                 final=final, eager="eager", policy="policy")
            self.assertEqual([r[0] for r in cuda.trace], ["bind"])
            cuda.trace.append(("P_tail_and_emitters", cuda.main))
            state = types.SimpleNamespace(valid=types.SimpleNamespace(
                index_fill_=lambda *a: cuda.trace.append(("valid2", cuda.main))))
            slots.long = lambda: slots
            scope["WIRE_FINAL_DEFERRED"] = 2
            tx = types.SimpleNamespace(published=True, final=final,
                plan=types.SimpleNamespace(slots=slots), controller=types.SimpleNamespace(state=state),
                pool=types.SimpleNamespace(launch_pending_tracked=side.launch_pending, pside_join=side.join))
            scope["finish_return"](tx)
            self.assertEqual([r[0] for r in cuda.trace],
                ["bind", "P_tail_and_emitters", "record", "wait", "tracked"]
                + (["final"] if final else []) + ["record", "wait"] + (["valid2"] if final else []))
            self.assertIs(cuda.trace[2][2], side.boundary_done)

    def test_AGG_A_dense_boundary_precedes_T_and_final(self):
        from test_gdn_final_factor_deferred import make_controller, stage
        c, cuda = make_controller()
        c.side.after_boundary = True
        stage(c)
        self.assertNotIn("T", [r[0] for r in cuda.trace])
        cuda.trace.append(("dense_boundary", cuda.main))
        c.after_boundary()
        names = [r[0] for r in cuda.trace]
        self.assertLess(names.index("dense_boundary"), names.index("T"))
        self.assertLess(names.index("T"), names.index("F_SN"))
        c.join()
        self.assertEqual(cuda.trace[-1], ("wait", cuda.main, c.done))


try:
    import torch
    import triton
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "same-image CPU gate requires torch and Triton")
class ReplayBytesTest(unittest.TestCase):
    def test_actual_T_graph_body_before_after_boundary_has_identical_pool_bytes(self):
        from test_gdn_prefill_k31_batch_graph import _modules, _pool, _plan, LAYERS, HV, V, K
        torch.set_num_threads(1)
        fp, bg = _modules()
        for batch in (1, 3):
            gen = torch.Generator().manual_seed(4815)
            states = [(torch.randn(batch, HV, V, K, generator=gen),
                       torch.randn(batch, HV, V, K, generator=gen)) for _ in range(LAYERS)]
            outputs = []
            for delayed in (False, True):
                pool = _pool(fp, 1); plan = _plan(fp, pool, batch)
                bucket = next(b for b in bg.BATCH_BUCKETS if batch <= b)
                buffers = bg.BatchBuffers(pool, bucket, bucket, include_tail=False)
                buffers.bind(plan, states, torch.arange(batch) + batch, None, None)
                side, cuda = controller(deferred=True, after_boundary=delayed)
                graph = types.SimpleNamespace(replay=lambda: buffers.evaluate(fp.factorize_layers, branch="tracked"))
                buffers.evaluate(fp.factorize_layers, branch="normal")
                if delayed:
                    side.defer_until_boundary((graph,), 0, cuda.main)
                else:
                    graph.replay()
                # A live-only boundary consumer must not see different F; its
                # writes cannot touch the independently owned tracked slots.
                before = tuple(getattr(pool, n)[:, :batch].clone() for n in ("a", "U", "W", "count"))
                pool.count[:, :batch].add_(1)
                if delayed:
                    side.launch_pending(); side.join()
                outputs.append((pool, before))
            for left, right in zip(outputs[0][1], outputs[1][1]):
                self.assertTrue(torch.equal(left.view(torch.uint8), right.view(torch.uint8)))
            for name in ("a", "U", "W", "count", "stale", "dense_of", "dense_required", "prefix_valid", "dense_ring"):
                self.assertTrue(torch.equal(getattr(outputs[0][0], name).contiguous().view(torch.uint8),
                                            getattr(outputs[1][0], name).contiguous().view(torch.uint8)), (batch, name))


if __name__ == "__main__":
    unittest.main()
