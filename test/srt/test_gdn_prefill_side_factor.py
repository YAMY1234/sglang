"""CPU control-flow replay of the side-stream prompt-end factor commit.

The real pool/model methods are extracted from source and run against stub
streams; this checks grouping, stream placement and join coverage, NOT CUDA
ordering on hardware or numerical equality.
"""

import ast
import copy
import os
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
POOL = REPO / "python/sglang/srt/mem_cache/gdn_factored_pool.py"
MODEL = REPO / "python/sglang/srt/models/flash_next_duet/model.py"
READERS = (
    "reset_slots",
    "copy_slots",
    "get_cpu_slots",
    "load_cpu_slots",
    "iter_transfer_state_entries",
    "layer_tensors",
    "plan_extend",
    "copy_slots_layer",
    "track_copy",
    "dense_of_slots",
)


def _function(tree, name):
    found = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    assert len(found) == 1, (name, len(found))
    return found[0]


def _calls(node, attr):
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == attr
    ]


class _Stream:
    def __init__(self, name):
        self.name = name
        self.waits = []

    def wait_stream(self, other):
        self.waits.append(other.name)


class _Torch:
    """Just enough of torch.cuda for the extracted methods."""

    def __init__(self):
        self.main = _Stream("main")
        self.current = self.main
        self.capturing = False
        self.created = []
        cuda = types.SimpleNamespace()
        cuda.current_stream = lambda device=None: self.current
        cuda.is_current_stream_capturing = lambda: self.capturing
        cuda.Stream = self._new_stream
        cuda.stream = self._use
        self.cuda = cuda

    def _new_stream(self, device=None):
        stream = _Stream("side")
        self.created.append(stream)
        return stream

    @contextmanager
    def _use(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous


class _State:
    is_cuda = True
    device = "cuda:0"

    def __init__(self, nbytes=4):
        self.recorded = []
        self._nbytes = nbytes

    def numel(self):
        return self._nbytes

    def element_size(self):
        return 1

    def record_stream(self, stream):
        self.recorded.append(stream.name)


def _pool_class(fake_torch):
    tree = ast.parse(POOL.read_text())
    names = (
        "pside_join",
        "commit_extend_batched",
        "_prefill_side_enabled",
        "_commit_extend_side",
    )
    nodes = []
    for name in names:
        node = copy.deepcopy(_function(tree, name))
        node.decorator_list = []
        nodes.append(node)
    module = ast.Module(
        body=[ast.ClassDef(name="Pool", bases=[], keywords=[], body=nodes,
                           decorator_list=[])],
        type_ignores=[],
    )
    scope = {"torch": fake_torch, "os": os, "getattr": getattr}
    exec(compile(ast.fix_missing_locations(module), str(POOL), "exec"), scope)
    return scope["Pool"]


def _make_pool(fake_torch, layers=36, max_bytes=512 << 20):
    cls = _pool_class(fake_torch)
    pool = cls()
    pool.layer_map = {lid: i for i, lid in enumerate(range(layers))}
    pool.cfg = types.SimpleNamespace(factored_prefix=False)
    pool.batch_prefill_max_bytes = max_bytes
    pool.device = "cuda:0"
    pool._prefill_side_stream = None
    pool._prefill_side_pending = False
    pool.groups = []

    def commit_group(layer_id, plan, *args):
        pool.groups.append(
            (fake_torch.current.name, [s for s, _ in plan.pending], layer_id)
        )
        # The real group may read a slot copy, which joins from inside the group.
        pool.pside_join()
        plan.pending.clear()

    pool._commit_extend_group = commit_group
    return pool


def _plan(layers, group):
    return types.SimpleNamespace(
        next_layer=0,
        last_layer=layers - 1,
        pending=[],
        slots=_State(),
        ring_dst=_State(),
        dense_required_after_commit=None,
        side_group=group,
    )


def _run(pool, plan, layers, nbytes=4):
    states = []
    for lid in range(layers):
        state = _State(nbytes)
        states.append(state)
        pool.commit_extend_batched(lid, plan, state)
    return states


class SideFactorGroupingTest(unittest.TestCase):
    def setUp(self):
        os.environ.pop("SGLANG_GDN_PSIDE_GRAPH", None)

    def test_off_commits_once_on_main_stream(self):
        torch = _Torch()
        pool = _make_pool(torch)
        _run(pool, _plan(36, 0), 36)
        self.assertEqual([(g[0], len(g[1])) for g in pool.groups], [("main", 36)])
        self.assertFalse(pool._prefill_side_pending)
        self.assertEqual(torch.created, [])

    def test_group_one_commits_every_layer_on_side_stream(self):
        torch = _Torch()
        pool = _make_pool(torch)
        states = _run(pool, _plan(36, 1), 36)
        self.assertEqual([(g[0], len(g[1])) for g in pool.groups], [("side", 1)] * 36)
        self.assertEqual([g[2] for g in pool.groups], list(range(36)))
        side = torch.created[0]
        self.assertEqual(len(torch.created), 1)
        self.assertEqual(side.waits, ["main"] * 36)
        self.assertTrue(all(s.recorded == ["side"] for s in states))
        # The in-group reader must not consume the main-stream join.
        self.assertTrue(pool._prefill_side_pending)
        pool.pside_join()
        self.assertFalse(pool._prefill_side_pending)
        self.assertEqual(torch.main.waits, ["side"])
        pool.pside_join()
        self.assertEqual(torch.main.waits, ["side"])

    def test_group_six_keeps_commit_order(self):
        torch = _Torch()
        pool = _make_pool(torch)
        _run(pool, _plan(36, 6), 36)
        self.assertEqual([len(g[1]) for g in pool.groups], [6] * 6)
        self.assertEqual([g[2] for g in pool.groups], [5, 11, 17, 23, 29, 35])
        self.assertTrue(all(g[0] == "side" for g in pool.groups))

    def test_partial_last_group_and_byte_cap(self):
        torch = _Torch()
        pool = _make_pool(torch, layers=10, max_bytes=12)
        # Byte cap 12 // 4 = 3 layers wins over G=6; the last group is partial.
        _run(pool, _plan(10, 6), 10, nbytes=4)
        self.assertEqual([len(g[1]) for g in pool.groups], [3, 3, 3, 1])

    def test_capture_keeps_original_path(self):
        torch = _Torch()
        torch.capturing = True
        pool = _make_pool(torch)
        _run(pool, _plan(36, 1), 36)
        self.assertEqual([(g[0], len(g[1])) for g in pool.groups], [("main", 36)])
        self.assertFalse(pool._prefill_side_pending)

    def test_pside_graph_experiment_untouched(self):
        torch = _Torch()
        pool = _make_pool(torch)
        os.environ["SGLANG_GDN_PSIDE_GRAPH"] = "1"
        try:
            _run(pool, _plan(36, 1), 36)
        finally:
            os.environ.pop("SGLANG_GDN_PSIDE_GRAPH")
        self.assertEqual(pool.groups, [])
        self.assertIsNotNone(pool._pside_deferred_commit)


class SideFactorJoinCoverageTest(unittest.TestCase):
    def test_every_factor_reader_joins_the_side_stream(self):
        tree = ast.parse(POOL.read_text())
        for name in READERS:
            joins = _calls(_function(tree, name), "pside_join")
            self.assertTrue(joins, name)
            self.assertTrue(
                all(not j.keywords for j in joins), f"{name} skips the side join"
            )

    def test_initial_dense_skips_only_the_side_join(self):
        tree = ast.parse(POOL.read_text())
        joins = _calls(_function(tree, "initial_dense"), "pside_join")
        self.assertEqual(len(joins), 1)
        self.assertEqual(
            [(k.arg, k.value.value) for k in joins[0].keywords],
            [("side_stream", False)],
        )

    def test_model_joins_before_every_prefill_exit(self):
        tree = ast.parse(MODEL.read_text())
        body = _function(tree, "_twinstar_prefill")
        joins = _calls(body, "pside_join")
        boundary = _calls(body, "_boundary_graph")
        intermediate = _calls(body, "LogitsProcessorOutput")
        nested = {
            id(r)
            for f in ast.walk(body)
            if isinstance(f, ast.FunctionDef) and f is not body
            for r in ast.walk(f)
        }
        returns = [
            n for n in ast.walk(body) if isinstance(n, ast.Return) and id(n) not in nested
        ]
        self.assertEqual(len(joins), 1)
        # One join precedes the boundary decode graph, the intermediate-chunk
        # output and the only return.
        for node in boundary + intermediate + returns:
            self.assertLess(joins[0].lineno, node.lineno)


if __name__ == "__main__":
    unittest.main()
