"""docs/223 §16 J1 (SGLANG_GDN_J1_SPLIT_PUBLICATION): CPU checks of the publication state machine and the forward
join routing.  The two-graph bytes (normal + tracked halves == one graph) are the GPU module gate's job
(test/srt/j1_module_bitwise.py)."""
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

MEM = Path(__file__).resolve().parents[2] / "python/sglang/srt/mem_cache"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _batch_graph_module():
    """gdn_prefill_batch_graph with its GPU-only imports stubbed (triton, graph capture lock, commit graph)."""
    triton = types.ModuleType("triton")
    triton.jit = lambda f=None, **k: f if f is not None else (lambda g: g)
    triton.cdiv = lambda a, b: -(-a // b)
    triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
    language = types.ModuleType("triton.language")
    language.constexpr = object
    triton.language = language
    pkg = types.ModuleType("j1pkg")
    pkg.__path__ = [str(MEM)]
    commit = types.ModuleType("j1pkg.gdn_prefill_commit_graph")
    commit.BATCH_BUCKETS, commit._publish_valid, commit.prewarm_shapes = (1, 2, 4), None, lambda: []
    capture = types.ModuleType("sglang.srt.utils.graph_capture")
    capture.graph_capture_lock = None
    with patch.dict(sys.modules, {"triton": triton, "triton.language": language, "j1pkg": pkg,
                                  "j1pkg.gdn_prefill_commit_graph": commit,
                                  "sglang.srt.utils.graph_capture": capture}):
        _load("j1pkg.gdn_prefill_joint", MEM / "gdn_prefill_joint.py")
        return _load("j1pkg.gdn_prefill_batch_graph", MEM / "gdn_prefill_batch_graph.py")


pub = _load("j1_gdn_pd_publication", MEM / "gdn_pd_publication.py")


class Event:
    def __init__(self, *a, **k):
        self.recorded = 0

    def record(self, stream=None):
        self.recorded += 1


class Runtime:
    """Records the order of side-stream work and joins."""
    def __init__(self):
        self.log = []

    def launch_after_forward(self, operation, inputs):
        self.log.append("launch")
        operation()
        end = NS(name="end")
        self.log.append("end_recorded")
        return end

    def join(self, ticket):
        self.log.append(("join", getattr(ticket, "name", "normal")))


class Graph:
    include_tail = False
    warmed = True

    def __init__(self, log):
        self.log = log
        self.calls = []

    def run(self, pool, plan, states, track_slots, final_src, final_dst, *, eager, policy, split=False,
            between=None, launch_replay=None):
        self.calls.append(dict(split=split))
        self.log.append("normal_half" if split else "one_graph")
        if split:
            between()
            self.log.append("normal_event")
            self.log.append("tracked_half")


def publication(mode):
    pool = NS(a=torch.zeros(1))
    p = pub.PDBatchPublication(pool, agg=True)
    p.runtime = Runtime()
    if mode is not None:
        p.j1 = pub.J1Split(mode)
    graph = Graph(p.runtime.log)
    plan = NS(slots=torch.tensor([7]), ring_dst=torch.tensor([3]), dense_required_after_commit=None)
    return p, graph, plan


def submit_and_launch(p, graph, plan):
    p.submit(graph, plan, [(torch.zeros(1), None)], (None, None, None), eager=None, policy=None)
    with patch.object(torch.cuda, "Event", Event), patch.object(torch.cuda, "current_stream", lambda *a: None):
        p.start_after_forward()


class TestJ1Labels(unittest.TestCase):
    def test_mode1_always_b(self):
        s = pub.J1Split("1")
        self.assertEqual([s.next_label(1) for _ in range(6)], ["B"] * 6)

    def test_ab_pattern_and_log(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"TWINSTAR_CUDA_TIMELINE": d}):
            s = pub.J1Split("ab")
            labels = [s.next_label(2) for _ in range(8)]
            s.sink.close()
            self.assertEqual("".join(labels), "ABBAABBA")
            rows = [json.loads(x) for x in open(os.path.join(d, f"j1-{os.getpid()}.jsonl"))]
            self.assertEqual([r["l"] for r in rows], labels)
            self.assertEqual([r["j"] for r in rows], list(range(8)))
            self.assertTrue(all(r["rows"] == 2 for r in rows))


class TestJ1Publication(unittest.TestCase):
    def test_default_path_one_graph(self):
        p, graph, plan = publication(None)
        submit_and_launch(p, graph, plan)
        self.assertEqual(graph.calls, [dict(split=False)])
        self.assertIsNone(p.ticket_normal)

    def test_split_launch_records_event_between_halves(self):
        p, graph, plan = publication("1")
        submit_and_launch(p, graph, plan)
        self.assertEqual(p.runtime.log, ["launch", "normal_half", "normal_event", "tracked_half", "end_recorded"])
        self.assertEqual(p.label, "B")
        self.assertEqual(p.ticket_normal.recorded, 1)

    def test_b_decode_joins_normal_only_then_whole(self):
        p, graph, plan = publication("1")
        submit_and_launch(p, graph, plan)
        p.join_decode()
        self.assertEqual(p.runtime.log[-1], ("join", "normal"))
        self.assertIsNotNone(p.pending)  # tracked half still owned by the publication
        n = len(p.runtime.log)
        p.join_decode()  # step 2: nothing more
        self.assertEqual(len(p.runtime.log), n)
        p.join()  # any other reader: the whole publication
        self.assertEqual(p.runtime.log[-1], ("join", "end"))
        self.assertIsNone(p.pending)
        self.assertIsNone(p.ticket_normal)
        self.assertFalse(p.normal_joined)

    def test_a_decode_joins_whole(self):
        p, graph, plan = publication("ab")  # first label A
        submit_and_launch(p, graph, plan)
        self.assertEqual(p.label, "A")
        p.join_decode()
        self.assertEqual(p.runtime.log[-1], ("join", "end"))
        self.assertIsNone(p.pending)

    def test_early_decode_reader_launches_first(self):
        p, graph, plan = publication("1")
        p.submit(graph, plan, [(torch.zeros(1), None)], (None, None, None), eager=None, policy=None)
        with patch.object(torch.cuda, "Event", Event), patch.object(torch.cuda, "current_stream", lambda *a: None):
            p.join_decode()
        self.assertEqual(p.stats["early_reader"], 1)
        self.assertEqual(p.runtime.log[0], "launch")
        self.assertEqual(p.runtime.log[-1], ("join", "normal"))

    def test_no_pending_is_noop(self):
        p, graph, plan = publication("1")
        p.join_decode()
        self.assertEqual(p.runtime.log, [])

    def test_failed_publication_fails_closed(self):
        p, graph, plan = publication("1")
        p.failed = RuntimeError("x")
        p.pending = object()
        with self.assertRaises(RuntimeError):
            p.join_decode()


class Mode:
    def __init__(self, decode):
        self._decode = decode

    def is_decode(self):
        return self._decode


def batch(decode=True, **kw):
    b = NS(forward_mode=Mode(decode), spec_info=None, can_run_tbo=False, mamba_cow_src_indices=None,
           mamba_cow_dst_indices=None, mamba_clear_indices=None)
    for k, v in kw.items():
        setattr(b, k, v)
    return b


class TestForwardJoinRouting(unittest.TestCase):
    def setUp(self):
        self.capture = False
        stub = types.ModuleType("sglang.srt.model_executor.runner")
        stub.get_is_capture_mode = lambda: self.capture
        self.mods = patch.dict(sys.modules, {"sglang.srt.model_executor.runner": stub})
        self.mods.start()

    def tearDown(self):
        self.mods.stop()

    def route(self, b, *, mode="1", prompt_only=True, pending=True, deferred=None):
        calls = []
        pool = NS(a=torch.zeros(1), _pside_deferred_commit=deferred, pside_join=lambda *a, **k: calls.append("whole"))
        p = pub.PDBatchPublication(pool, agg=True)
        if mode is not None:
            p.j1 = pub.J1Split(mode)
        p.pending = object() if pending else None
        p.join_decode = lambda: calls.append("decode")
        pool._pd_batch_publication = p
        runner = NS(forward=lambda *a, **k: calls.append("forward"),
                    req_to_token_pool=NS(_prompt_only_state_cache=prompt_only))
        pub.install_forward_join(runner, pool)
        runner.forward(b)
        return calls

    def test_plain_decode_takes_decode_join(self):
        self.assertEqual(self.route(batch()), ["decode", "forward"])

    def test_extend_takes_whole_join(self):
        self.assertEqual(self.route(batch(decode=False)), ["whole", "forward"])

    def test_pool_ops_take_whole_join(self):
        for name in ("mamba_cow_src_indices", "mamba_cow_dst_indices", "mamba_clear_indices"):
            self.assertEqual(self.route(batch(**{name: torch.tensor([3])})), ["whole", "forward"], name)
        self.assertEqual(self.route(batch(mamba_clear_indices=torch.tensor([], dtype=torch.long))),
                         ["decode", "forward"])

    def test_spec_tbo_capture_take_whole_join(self):
        self.assertEqual(self.route(batch(spec_info=object())), ["whole", "forward"])
        self.assertEqual(self.route(batch(can_run_tbo=True)), ["whole", "forward"])
        self.capture = True
        self.assertEqual(self.route(batch()), ["whole", "forward"])

    def test_decode_tracking_takes_whole_join(self):
        self.assertEqual(self.route(batch(), prompt_only=False), ["whole", "forward"])
        self.assertEqual(self.route(batch(), prompt_only=None), ["whole", "forward"])

    def test_eager_deferred_commit_takes_whole_join(self):
        self.assertEqual(self.route(batch(), deferred=("x",)), ["whole", "forward"])

    def test_switch_off_or_nothing_pending(self):
        self.assertEqual(self.route(batch(), mode=None), ["whole", "forward"])
        self.assertEqual(self.route(batch(), pending=False), ["whole", "forward"])


class TestSplitMode(unittest.TestCase):
    def test_values(self):
        g = _batch_graph_module()
        for v in ("0", "1", "ab"):
            with patch.dict(os.environ, {g.SPLIT_FLAG: v, "SGLANG_GDN_PREFILL_JOIN_BRANCHES": "0"}):
                self.assertEqual(g.split_mode(), v)
        with patch.dict(os.environ, {g.SPLIT_FLAG: "2"}):
            with self.assertRaises(ValueError):
                g.split_mode()
        with patch.dict(os.environ, {g.SPLIT_FLAG: "1", "SGLANG_GDN_PREFILL_JOIN_BRANCHES": "1"}):
            with self.assertRaises(ValueError):
                g.split_mode()


if __name__ == "__main__":
    unittest.main()
