"""#042 reader/stream dispatch and byte equality of the captured graph bodies.

CUDA streams are stubbed for control flow. Tensor tests use real k31 arithmetic
and Triton stores on CPU; hardware graph replay remains a separate GPU gate.
"""
import ast
import copy
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import types
import unittest

os.environ.setdefault("TRITON_INTERPRET", "1")
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "python/sglang/srt"
SIDE = BASE / "mem_cache/gdn_tracked_factor_side.py"
POOL = BASE / "mem_cache/gdn_factored_pool.py"


def function(path, name):
    return copy.deepcopy(next(n for n in ast.walk(ast.parse(path.read_text()))
                              if isinstance(n, ast.FunctionDef) and n.name == name))


def extract(nodes, scope):
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), "<real-source>", "exec"), scope)


class Vector:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)
    def numel(self):
        return len(self.values)
    def tolist(self):
        return self.values[:]
    def reshape(self, *shape):
        return self


class FakeCuda:
    def __init__(self):
        self.trace = []
        self.capturing = False
        self.current = self.Stream()
        self.main = self.current
        self.cuda = self
    def current_stream(self, device=None):
        return self.current
    def is_current_stream_capturing(self):
        return self.capturing
    def graph_pool_handle(self):
        return object()
    def cat(self, vectors):
        return Vector([v for vector in vectors for v in vector.values])
    def Stream(self, device=None):
        owner = self
        class Stream:
            def __init__(self):
                self.cuda_stream = id(self)
            def wait_event(self, event):
                owner.trace.append(("wait", self, event))
        return Stream()
    def Event(self):
        owner = self
        class Event:
            complete = False
            def record(self, stream):
                owner.trace.append(("record", stream, self))
            def query(self):
                return self.complete
        return Event()
    @contextmanager
    def stream(self, stream):
        old, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = old


def controller(deferred=False):
    cuda = FakeCuda()
    tree = ast.parse(SIDE.read_text())
    cls = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.ClassDef)))
    # Only replace this import with its literal bucket configuration.
    for n in ast.walk(cls):
        if isinstance(n, ast.FunctionDef):
            n.body = [s for s in n.body if not isinstance(s, ast.ImportFrom)]
    scope = dict(torch=cuda, logger=logging.getLogger("test"), BATCH_BUCKETS=(1, 2, 4, 8, 16))
    extract([function(SIDE, "disjoint_destinations"), cls], scope)
    def key(*args):
        return args
    whole = types.SimpleNamespace(key=key)
    pool = types.SimpleNamespace(a=types.SimpleNamespace(device="cuda:0"))
    side = scope["TrackedFactorSide"](pool, whole, deferred=deferred)
    def binder(name):
        return lambda *args: cuda.trace.append((name, cuda.current, args))
    def graph(name):
        return types.SimpleNamespace(replay=lambda: cuda.trace.append((name, cuda.current)))
    side.entries[key(1, 1, "eager", "policy", False)] = (
        types.SimpleNamespace(bind=binder("bind")), graph("final"), graph("tracked"))
    side.alt_entries[key(1, 1, "eager", "policy", False)] = (
        types.SimpleNamespace(bind=binder("bind_alt")), graph("final_alt"), graph("tracked_alt"))
    return side, cuda


class SideDispatchTest(unittest.TestCase):
    def launch(self, side, src=(1,), dst=(3,), tracked=(2,), live=(1,)):
        return side.run(types.SimpleNamespace(slots=Vector(live)),
                        [(Vector([0]), Vector([0]))], Vector(tracked),
                        Vector(src), Vector(dst), eager="eager", policy="policy")

    def test_one_graph_on_each_stream_and_bind_fence(self):
        side, cuda = controller()
        self.assertTrue(self.launch(side))
        self.assertEqual([r[0] for r in cuda.trace],
                         ["bind", "record", "final", "wait", "tracked", "record"])
        self.assertIs(cuda.trace[2][1], cuda.main)
        self.assertIs(cuda.trace[4][1], side.stream)
        self.assertIs(cuda.trace[3][2], side.bound)
        self.assertIsNot(side.final_arena, side.tracked_arena)
        self.assertTrue(side._prefill_side_pending)

    def test_readers_wait_once_per_stream_and_writer_does_not_consume_fence(self):
        side, cuda = controller()
        self.launch(side)
        with cuda.stream(side.stream):
            side.join()
        self.assertTrue(side._prefill_side_pending)
        side.join()
        side.join()
        other = cuda.Stream()
        with cuda.stream(other):
            side.join()
            side.join()
        waits = [r for r in cuda.trace if r[0] == "wait" and r[2] is side.done]
        self.assertEqual([r[1] for r in waits], [cuda.main, other])

    def test_next_bind_waits_before_overwriting_tracked_inputs(self):
        side, cuda = controller()
        self.launch(side)
        cuda.trace.clear()
        self.launch(side)
        self.assertEqual(cuda.trace[0], ("wait", cuda.main, side.done))
        self.assertEqual(cuda.trace[1][0], "bind")
        self.assertEqual(side.stats["split"], 2)

    def test_tracked_source_and_destination_aliases_fall_back(self):
        for kwargs, reason in [
            (dict(src=(2,)), "final_from_tracked"),
            (dict(live=(2,)), "aliased_slots"),
            (dict(dst=(2,)), "aliased_slots"),
            (dict(tracked=(2, 2)), "aliased_slots"),
        ]:
            with self.subTest(kwargs=kwargs):
                side, cuda = controller()
                self.assertFalse(self.launch(side, **kwargs))
                self.assertEqual(side.stats["fallback_" + reason], 1)
                self.assertFalse(cuda.trace)

    def test_capture_does_not_launch_or_touch_inputs(self):
        side, cuda = controller()
        cuda.capturing = True
        self.assertFalse(self.launch(side))
        self.assertEqual(side.stats["capture_fallback"], 1)
        self.assertFalse(cuda.trace)

    def test_capture_with_outstanding_work_fails_until_event_complete(self):
        side, cuda = controller()
        self.launch(side)
        cuda.capturing = True
        with self.assertRaisesRegex(RuntimeError, "must finish"):
            side.join()
        side.done.complete = True
        side.join()
        self.assertFalse(side.recorded)

    def test_no_tracked_rows_uses_whole_graph(self):
        side, cuda = controller()
        self.assertFalse(self.launch(side, tracked=()))
        self.assertEqual(side.stats["fallback_no_tracked"], 1)
        self.assertFalse(cuda.trace)


class DeferredSideTest(unittest.TestCase):
    launch = SideDispatchTest.launch

    def test_tracked_starts_only_after_final(self):
        side, cuda = controller(deferred=True)
        self.assertTrue(self.launch(side))
        self.assertEqual([r[0] for r in cuda.trace],
                         ["bind", "record", "final", "record", "wait", "tracked", "record"])
        self.assertIs(cuda.trace[3][2], side.final_done)
        # The side stream waits for F, never only for the bind.
        self.assertEqual(cuda.trace[4][1:], (side.stream, side.final_done))
        self.assertIs(cuda.trace[5][1], side.stream)
        self.assertIs(cuda.trace[6][2], side.set_done[0])

    def test_inputs_alternate_and_bind_waits_for_t_two_commits_back(self):
        side, cuda = controller(deferred=True)
        binds = []
        for _ in range(3):
            cuda.trace.clear()
            self.launch(side)
            binds.append([r for r in cuda.trace if r[0] in ("bind", "bind_alt", "wait")
                          and r[1] is cuda.main])
        # k=0 and k=1 bind fresh sets; k=2 reuses set 0 after waiting for its T.
        self.assertEqual([r[0] for r in binds[0]], ["bind"])
        self.assertEqual([r[0] for r in binds[1]], ["bind_alt"])
        self.assertEqual([(r[0], r[2] if r[0] == "wait" else None) for r in binds[2]],
                         [("wait", side.set_done[0]), ("bind", None)])

    def test_readers_wait_for_latest_tracked(self):
        side, cuda = controller(deferred=True)
        self.launch(side)
        self.launch(side)
        cuda.trace.clear()
        side.join()
        self.assertEqual(cuda.trace, [("wait", cuda.main, side.set_done[1])])


class ReaderCoverageTest(unittest.TestCase):
    def test_real_reader_entries_join_before_access(self):
        class ReachedJoin(Exception):
            pass
        reads = {
            "reset_slots": (Vector([1]),), "copy_slots": (None, None),
            "get_cpu_slots": (None,), "load_cpu_slots": (None, None),
            "iter_transfer_state_entries": (), "mark_transferred_slots": (None,),
            "layer_tensors": (0,), "plan_extend": (None, None),
            "initial_dense": (0, None), "copy_slots_layer": (0, None, None),
            "track_copy": (None, None, None), "dense_of_slots": (0, None),
            "abandon_ring": (None,), "dump_slots": (0, None, None, None, None),
            "save_prefix_dense": (0, None, None), "invalidate_prefix_dense": (None,),
            "commit_extend": (0, None, None), "write_factored_dense": (0, None, None),
        }
        for name, args in reads.items():
            with self.subTest(reader=name):
                scope = {}
                node = function(POOL, name)
                node.decorator_list = []
                extract([node], scope)
                def join(**kwargs):
                    self.assertTrue(kwargs.get("tracked", True))
                    raise ReachedJoin()
                pool = types.SimpleNamespace(pside_join=join)
                with self.assertRaises(ReachedJoin):
                    value = scope[name](pool, *args)
                    if name == "iter_transfer_state_entries":
                        next(value)

    def test_live_accessor_does_not_consume_tracked_join(self):
        scope = {}
        extract([function(POOL, "layer_tensors")], scope)
        calls = []
        pool = types.SimpleNamespace(pside_join=lambda **kw: calls.append(kw),
                                     layer_map={7: 0})
        for name in ("a", "U", "W", "count", "vbar"):
            setattr(pool, name, [name])
        self.assertEqual(scope["layer_tensors"](pool, 7, live_only=True),
                         ("a", "U", "W", "count", "vbar"))
        self.assertEqual(calls, [{"tracked": False}])

    def test_radix_donation_and_pd_transport_join(self):
        # Execute the real insert until its first pool fence. No cache/transport stub
        # is allowed to hide a missed publication hook.
        class ReachedJoin(Exception):
            pass
        def join():
            raise ReachedJoin()
        scope = {}
        extract([function(BASE / "mem_cache/mamba_radix_cache.py", "insert")], scope)
        cache = types.SimpleNamespace(disable=False, req_to_token_pool=types.SimpleNamespace(
            factored_gdn_pool=types.SimpleNamespace(pside_join=join)))
        with self.assertRaises(ReachedJoin):
            scope["insert"](cache, None)
        tree = ast.parse((BASE / "disaggregation/state_handoff.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "FactorStateHandoff")
        extract([copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                                    and n.name == "before_send"))], scope)
        handoff = types.SimpleNamespace(pool=types.SimpleNamespace(pside_join=join))
        with self.assertRaises(ReachedJoin):
            scope["before_send"](handoff, None)

    def test_hicache_backup_and_restore_join_before_direct_tensor_access(self):
        class ReachedJoin(Exception):
            pass
        def join():
            raise ReachedJoin()
        host = types.SimpleNamespace(factor=types.SimpleNamespace(pside_join=join))
        source = BASE / "mem_cache/pool_host/flashnext_factored.py"
        for name, args in (("backup_from_device_all_layer", (None, None, None)),
                           ("load_to_device_per_layer", (None, None, None, 0))):
            with self.subTest(reader=name):
                scope = {}
                extract([function(source, name)], scope)
                with self.assertRaises(ReachedJoin):
                    scope[name](host, *args)

    def test_default_off_and_role_guard(self):
        env = (BASE / "environ.py").read_text()
        self.assertIn("SGLANG_GDN_TRACKED_FACTOR_SIDE_STREAM = EnvBool(False)", env)
        fn = function(POOL, "prewarm_k31_batch_graph")
        text = ast.unparse(fn)
        self.assertIn("disaggregation_mode in (None, 'null')", text)
        self.assertIn("self._generic_prompt_only_state_cache", text)
        backend = ast.parse((BASE / "layers/attention/linear/gdn_backend.py").read_text())
        calls = [n for n in ast.walk(backend) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "layer_tensors"]
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(any(k.arg == "live_only" and isinstance(k.value, ast.Constant)
                                and k.value.value is True for k in n.keywords) for n in calls))

    def test_actual_prewarm_requires_agg_or_explicit_PD_and_prompt_policy(self):
        import sys
        from unittest.mock import patch
        for role, requested, graph_on, prompt, expected in (
            ("null", True, True, True, 1), (None, True, True, True, 1),
            ("prefill", True, True, True, 0), ("decode", True, True, True, 0),
            ("null", False, True, True, 0), ("null", True, False, True, 0),
            ("null", True, True, False, 0),
            ("explicit_prefill", False, True, True, 1),
        ):
            with self.subTest(role=role, requested=requested, graph=graph_on, prompt=prompt):
                calls = []
                flag = lambda value: types.SimpleNamespace(get=lambda: value)
                env = types.ModuleType("sglang.srt.environ")
                env.envs = types.SimpleNamespace(
                    SGLANG_GDN_TRACKED_FACTOR_SIDE_STREAM=flag(requested),
                    SGLANG_GDN_TRACKED_FACTOR_SIDE_STREAM_PD_P=flag(role == "explicit_prefill"),
                    SGLANG_GDN_TRACKED_FACTOR_SIDE_STREAM_DEFERRED=flag(False),
                    SGLANG_GDN_PREFILL_FACTOR_GRAPH_K31=flag(graph_on),
                    SGLANG_GDN_PREFILL_FACTOR_GRAPH_K31_MAX_BATCH=flag(4))
                graph = types.ModuleType("sglang.srt.mem_cache.gdn_prefill_batch_graph")
                graph.PrefillBatchGraph = lambda **kw: types.SimpleNamespace(
                    prewarm=lambda *a, **kw: calls.append("whole"))
                side = types.ModuleType("sglang.srt.mem_cache.gdn_tracked_factor_side")
                side.TrackedFactorSide = lambda *a, **kw: types.SimpleNamespace(
                    prewarm=lambda **kw: calls.append("side"))
                scope = dict(__package__="sglang.srt.mem_cache", logger=logging.getLogger("test"),
                             k31_graph_safe=lambda device: True, factorize_layers="eager",
                             factorize_dense="dense", ORTH_METHOD=None, ORTH_WARPS_OVERRIDE=None)
                extract([function(POOL, "prewarm_k31_batch_graph")], scope)
                pool = types.SimpleNamespace(
                    cfg=types.SimpleNamespace(init_method="k31", factored_prefix=True),
                    batch_prefill=True, batch_prefill_final_copy=True,
                    prefix_dense=None, warm_v=None, layer_ids=list(range(36)),
                    prefix_layer_count=lambda: 36, a=types.SimpleNamespace(is_cuda=True),
                    device="cuda", _generic_prompt_only_state_cache=prompt)
                with patch.dict(sys.modules, {m.__name__: m for m in (env, graph, side)}):
                    scope["prewarm_k31_batch_graph"](pool, disaggregation_mode="prefill" if role == "explicit_prefill" else role)
                self.assertEqual(calls.count("side"), expected)


try:
    import torch
    import triton
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "needs torch + triton; same-image gate rejects skips")
class GraphBodyBitwiseTest(unittest.TestCase):
    def test_disjoint_split_matches_whole_graph_bytes_both_orders(self):
        from test_gdn_prefill_k31_batch_graph import _modules, _pool, _plan, LAYERS, HV, V, K
        torch.set_num_threads(1)
        fp, bg = _modules()
        for batch in (1, 3):
            generator = torch.Generator().manual_seed(42)
            states = [(torch.randn(batch, HV, V, K, generator=generator),
                       torch.randn(batch, HV, V, K, generator=generator))
                      for _ in range(LAYERS)]
            track = torch.arange(batch) + batch
            outputs = []
            for branches in (("both",), ("normal", "tracked"), ("tracked", "normal")):
                pool = _pool(fp, 1)
                plan = _plan(fp, pool, batch)
                bucket = next(b for b in bg.BATCH_BUCKETS if batch <= b)
                buffers = bg.BatchBuffers(pool, bucket, bucket, include_tail=False)
                buffers.bind(plan, states, track, torch.tensor([0]), torch.tensor([7]))
                for branch in branches:
                    buffers.evaluate(fp.factorize_layers, branch=branch)
                outputs.append(pool)
            for pool in outputs[1:]:
                for name in ("a", "U", "W", "count", "stale", "dense_of",
                             "dense_required", "prefix_valid", "dense_ring"):
                    expected = getattr(outputs[0], name).contiguous().view(torch.uint8)
                    actual = getattr(pool, name).contiguous().view(torch.uint8)
                    self.assertTrue(torch.equal(expected, actual), (batch, name))

    def test_alternate_input_set_matches_whole_graph_bytes(self):
        from test_gdn_prefill_k31_batch_graph import _modules, _pool, _plan, LAYERS, HV, V, K
        torch.set_num_threads(1)
        fp, bg = _modules()
        generator = torch.Generator().manual_seed(7)
        states = [(torch.randn(1, HV, V, K, generator=generator),
                   torch.randn(1, HV, V, K, generator=generator)) for _ in range(LAYERS)]
        outputs = []
        for alternate in (False, True):
            pool = _pool(fp, 1)
            plan = _plan(fp, pool, 1)
            shared = {}
            first = bg.BatchBuffers(pool, 1, 1, shared, include_tail=False)
            buffers = first
            if alternate:
                normal_only = {k: v for k, v in shared.items() if k[0] == "normal"}
                buffers = bg.BatchBuffers(pool, 1, 1, normal_only, include_tail=False)
                self.assertIs(buffers.normal, first.normal)
                self.assertIsNot(buffers.tracked, first.tracked)
            buffers.bind(plan, states, torch.tensor([1]), torch.tensor([0]), torch.tensor([7]))
            for branch in (("both",) if not alternate else ("normal", "tracked")):
                buffers.evaluate(fp.factorize_layers, branch=branch)
            outputs.append(pool)
        for name in ("a", "U", "W", "count", "stale", "dense_of",
                     "dense_required", "prefix_valid", "dense_ring"):
            expected = getattr(outputs[0], name).contiguous().view(torch.uint8)
            actual = getattr(outputs[1], name).contiguous().view(torch.uint8)
            self.assertTrue(torch.equal(expected, actual), name)

    def test_false_decode_track_mask_cannot_rewrite_published_valid_bit(self):
        from test_gdn_prefill_k31_batch_graph import _modules
        import importlib
        _modules()
        module = importlib.import_module("sglang.srt.mem_cache.gdn_tracked_factor_side")
        valid = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
        # Duplicate destination: the false-mask row must not overwrite the true
        # row's invalidation. This also verifies that padding does not touch 0.
        module.invalidate_tracked_masked(valid, torch.tensor([2, 2, -1]),
                                        torch.tensor([True, False, False]))
        self.assertEqual(valid.tolist(), [1, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
