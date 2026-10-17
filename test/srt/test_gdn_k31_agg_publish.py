"""K4.2 (docs/223): AGG prompt-end k31 on the publication side stream.

CPU only; fake CUDA runtimes record the stream order. The default (no done
event) path keeps the forward-stream completion assertion; the deferred path
accepts a pending side-stream event because every GPU reader joins it.
"""
import copy
import os
import importlib.util
import sys
import types
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / rel)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


ov = _load("sglang.srt.mem_cache.gdn_fulln_overlap", "python/sglang/srt/mem_cache/gdn_fulln_overlap.py")
pub = _load("sglang.srt.mem_cache.gdn_pd_publication", "python/sglang/srt/mem_cache/gdn_pd_publication.py")


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_collector_modules():
    """P1: the real batch-graph and contract modules. Without the full sglang
    import (CPU venv) their packages and heavy siblings are stubs."""
    try:
        import sglang.srt.mem_cache  # noqa: F401
    except Exception:
        for name in ("sglang", "sglang.srt", "sglang.srt.mem_cache", "sglang.srt.utils"):
            sys.modules.setdefault(name, _package(name))
        _load("sglang.srt.utils.graph_capture", "python/sglang/srt/utils/graph_capture.py")
        commit = types.ModuleType("sglang.srt.mem_cache.gdn_prefill_commit_graph")
        commit.BATCH_BUCKETS = (1, 2, 4, 8, 16)
        commit._publish_valid = None
        commit.prewarm_shapes = lambda: iter(())
        sys.modules.setdefault(commit.__name__, commit)
    _load("sglang.srt.mem_cache.gdn_prefill_joint", "python/sglang/srt/mem_cache/gdn_prefill_joint.py")
    _load("sglang.srt.mem_cache.gdn_prefill_recipe_guard", "python/sglang/srt/mem_cache/gdn_prefill_recipe_guard.py")
    graph = _load("sglang.srt.mem_cache.gdn_prefill_batch_graph",
                  "python/sglang/srt/mem_cache/gdn_prefill_batch_graph.py")
    contract = _load("sglang.srt.mem_cache.gdn_prefill_agg_contract",
                     "python/sglang/srt/mem_cache/gdn_prefill_agg_contract.py")
    return graph, contract


class Runtime:
    def __init__(self):
        self.events, self.done = [], []

    def attach(self, scheduler):
        pass

    def published(self):
        event = NS(ready=False, stream="forward")
        self.done.append(event)
        return event

    def wait_for_result(self, event):
        self.events.append(("schedule-wait", event.stream))
        return event

    def elapsed_us(self, timing):
        return None

    def complete(self, event):
        return event.ready


def _request(slot=1):
    return NS(kv=NS(req_pool_idx=slot, mamba_last_track_idx=0,
                    mamba_next_track_idx=1, mamba_last_track_seqlen=64))


def _batch():
    return NS(reqs=[_request()], fulln_overlap_record=None,
              forward_mode=NS(is_mixed=lambda: False, is_decode=lambda: False))


def _prepared(controller, source):
    record = controller.begin(source, True)
    req = source.reqs[0]
    fields = tuple(getattr(req.kv, n) for n in ov.TRACK_FIELDS)
    record.tracks[id(req)] = ov.TrackSnapshot(req, fields, fields, True)
    record.seal(True)
    record.plan = NS(pending=[], slots=(req.kv.req_pool_idx,))
    return record


class DeferredRecordTest(unittest.TestCase):
    def test_default_publication_still_requires_completion(self):
        rt = Runtime(); c = ov.FullNOverlap("cpu", rt); r = _prepared(c, _batch()); c.publish(r)
        self.assertFalse(r.deferred)
        with self.assertRaisesRegex(RuntimeError, "before publication"):
            r.before_result()

    def test_side_stream_event_is_accepted_while_pending(self):
        rt = Runtime(); c = ov.FullNOverlap("cpu", rt); sb = _batch(); r = _prepared(c, sb)
        side = NS(ready=False, stream="side")
        c.publish(r, done=side)
        self.assertTrue(r.deferred)
        self.assertIs(r.publication_done, side)
        self.assertEqual(rt.done, [])  # no forward-stream event recorded
        r.before_result()
        self.assertTrue(r.result_validated)
        saved = copy.copy(sb)
        c.result_consumed(saved)
        self.assertTrue(r.consumed)
        self.assertIsNone(c.pending)

    def test_cpu_owner_reader_waits_on_the_side_stream_event(self):
        rt = Runtime(); c = ov.FullNOverlap("cpu", rt); sb = _batch(); r = _prepared(c, sb)
        c.publish(r, done=NS(ready=False, stream="side"))
        queue = deque([(copy.copy(sb), NS())])
        scheduler = NS(result_queue=queue)

        def consume():
            saved, _ = queue.popleft()
            saved.fulln_overlap_record.before_result()
            c.result_consumed(saved)

        self.assertTrue(c.drain_before_planning(scheduler, consume, touched_slots={1}))
        self.assertEqual(rt.events, [("schedule-wait", "side")])

    def test_slot_and_track_checks_still_apply(self):
        rt = Runtime(); c = ov.FullNOverlap("cpu", rt); sb = _batch(); r = _prepared(c, sb)
        c.publish(r, done=NS(ready=False, stream="side"))
        sb.reqs[0].kv.req_pool_idx = 9
        with self.assertRaisesRegex(RuntimeError, "slot reused"):
            r.before_result()


class FakeCudaRuntime:
    def __init__(self, log):
        self.log, self.serial = log, 0
        self.inputs = []

    def launch_after_forward(self, operation, inputs):
        self.inputs.append(list(inputs))
        self.log.append("side-wait-forward-done")
        operation()
        self.serial += 1
        self.log.append(f"side-done-{self.serial}")
        return f"ticket-{self.serial}"

    def join(self, done):
        self.log.append(f"current-wait-{done}")


class FakeGraph:
    include_tail, warmed = False, True

    def __init__(self, log):
        self.log = log

    def run(self, pool, plan, states, *controls, eager, policy, launch_replay=None):
        self.log.append("k31-replay")


class AggPublicationTest(unittest.TestCase):
    def _publication(self, log):
        pool = NS(a=NS(device="cpu"))
        publication = pub.PDBatchPublication(pool, FakeCudaRuntime(log), agg=True)
        return publication

    def _submit(self, publication, log):
        plan = NS(slots=NS(numel=lambda: 1), ring_dst=None, dense_required_after_commit=None)
        publication.submit(FakeGraph(log), plan, [(None, None)], (None, None, None),
                           eager=None, policy=None)

    def test_replay_runs_on_side_stream_and_reader_joins_once(self):
        log = []
        p = self._publication(log)
        self.assertFalse(p.offload_join)
        self._submit(p, log)
        with self.assertLogs(pub.logger, "INFO") as logs:
            p.start_after_forward()
        self.assertTrue(any("K42_AGG_PUBLISH_DEFERRED_ACTIVE" in line for line in logs.output))
        self.assertEqual(log, ["side-wait-forward-done", "k31-replay", "side-done-1"])
        self.assertEqual(p.transfer_event(), "ticket-1")
        p.join_reader(None)
        p.join_reader(None)  # second reader: nothing pending
        self.assertEqual(log[-1], "current-wait-ticket-1")
        self.assertEqual(sum(e.startswith("current-wait") for e in log), 1)

    def test_activation_logged_only_on_first_launch(self):
        log = []
        p = self._publication(log)
        for _ in range(2):
            self._submit(p, log)
            p.start_after_forward()
            p.join()
        self.assertEqual(p.stats["launched"], 2)

    def test_after_replay_hook_runs_before_done_event(self):
        log = []
        p = self._publication(log)
        p.after_replay = lambda: log.append("checksum")
        self._submit(p, log)
        p.start_after_forward()
        self.assertEqual(log, ["side-wait-forward-done", "k31-replay", "checksum", "side-done-1"])

    def test_early_reader_launches_and_joins(self):
        log = []
        p = self._publication(log)
        self._submit(p, log)
        p.join()
        self.assertEqual(p.stats["early_reader"], 1)
        self.assertEqual(log[-1], "current-wait-ticket-1")

    def test_install_agg_wraps_runner_forward_with_join(self):
        log = []
        pool = NS(a=NS(device="cpu"), host_sync_free=False)
        pool.pside_join = lambda *a, **k: log.append("pside-join")
        runner = NS(forward=lambda *a, **k: log.append("forward") or "out")
        envs = types.ModuleType("sglang.srt.environ")
        envs.envs = NS(SGLANG_GDN_PD_PUBLISH_JOIN_OFFLOAD=NS(get=lambda: True))
        ctx = types.ModuleType("sglang.srt.runtime_context")
        ctx.get_schedule = lambda: NS(disable_overlap_schedule=False)
        saved = {k: sys.modules.get(k) for k in (envs.__name__, ctx.__name__)}
        sys.modules[envs.__name__], sys.modules[ctx.__name__] = envs, ctx
        try:
            pub.install(pool, runner, agg=True)  # PD offload env must not leak into AGG
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        self.assertTrue(pool._pd_batch_publication.agg)
        self.assertFalse(pool._pd_batch_publication.offload_join)
        self.assertEqual(runner.forward(NS()), "out")
        self.assertEqual(log, ["pside-join", "forward"])


class PoolRouteTest(unittest.TestCase):
    """Non-full-N AGG route: commit_extend_batched's k31 branch (BF16 C+ recipe)."""

    def _pool(self, log):
        pool = NS(a=NS(device="cpu"))
        pool._pd_batch_publication = pub.PDBatchPublication(pool, FakeCudaRuntime(log), agg=True, pool_route=True)
        return pool

    def _plan(self):
        return NS(slots=NS(numel=lambda: 1), ring_dst=None, dense_required_after_commit=None,
                  pending=[(None, None)])

    def test_default_submits_then_hook_launches(self):
        log = []
        pool = self._pool(log)
        saved = os.environ.pop("SGLANG_GDN_K42_CONTROL", None)
        try:
            pub.commit_k31_pool_route(pool, FakeGraph(log), self._plan(), (None, None, None),
                                      eager=None, policy=None)
        finally:
            if saved is not None:
                os.environ["SGLANG_GDN_K42_CONTROL"] = saved
        self.assertEqual(log, [])  # nothing on any stream until after logits
        pool._pd_batch_publication.start_after_forward()
        self.assertEqual(log, ["side-wait-forward-done", "k31-replay", "side-done-1"])

    def test_control_zero_replays_on_forward_stream(self):
        import tempfile

        log = []
        pool = self._pool(log)
        with tempfile.NamedTemporaryFile("w", delete=False) as control:
            control.write("0")
        os.environ["SGLANG_GDN_K42_CONTROL"] = control.name
        try:
            pub.commit_k31_pool_route(pool, FakeGraph(log), self._plan(), (None, None, None),
                                      eager=None, policy=None)
        finally:
            del os.environ["SGLANG_GDN_K42_CONTROL"]
            os.unlink(control.name)
        self.assertEqual(log, ["k31-replay"])
        self.assertIsNone(pool._pd_batch_publication.pending)


# ---------------------------------------------------------------- P1 (docs/229 §3): launch from the last layer's add
class CollectorPool:
    """Pool surface BatchCollector touches; pside_join joins like FactoredGDNPool.pside_join."""

    def __init__(self, log, layers=3):
        import torch

        self.log = log
        self.layer_ids = [i * 4 for i in range(layers)]
        self.layer_map = {lid: i for i, lid in enumerate(self.layer_ids)}
        self.a = torch.zeros(1)
        self._pd_batch_publication = None

    def prefix_layer_count(self):
        return len(self.layer_ids)

    def invalidate_prefix_dense(self, slots):
        self.log.append("invalidate")

    def pside_join(self, slots=None, *, forward_local=False):
        self.log.append("pside-join")
        if self._pd_batch_publication is not None:
            self._pd_batch_publication.join_reader(slots, forward_local=forward_local)


class SlabWorkspace:
    def __init__(self, log):
        self.log = log

    def snapshot(self, index, dense, tracked, token_count):
        self.log.append(f"snapshot-{index}")
        return ("slab", index), None if tracked is None else ("tracked-slab", index)


class CollectorGraph:
    include_tail, warmed = False, True

    def __init__(self, log, workspace=True):
        self.log = log
        self.workspace = SlabWorkspace(log) if workspace else None
        self.runs = []

    def bind_controls(self, pool, plan, first, *controls, eager, policy):
        self.log.append("bind-controls")
        return "entry-key"

    def run(self, pool, plan, states, *controls, eager, policy, launch_replay=None, **kwargs):
        self.runs.append(dict(states=list(states), controls=controls, **kwargs))
        self.log.append("k31-replay" + "".join(f" {k}={v}" for k, v in sorted(kwargs.items())))


class LaunchAtLastLayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        cls.bg, cls.contract = _load_collector_modules()
        cls.saved_policy = cls.bg.BatchCollector.__dict__["_policy"]
        cls.bg.BatchCollector._policy = staticmethod(lambda: ("eager", "policy"))

    @classmethod
    def tearDownClass(cls):
        cls.bg.BatchCollector._policy = cls.saved_policy

    def setUp(self):
        self.contract._AB_ROUTE["value"] = None

    def _forward(self, early, *, workspace=True, reader_inside=False):
        """One AGG full-N forward: trunk layers fill the collector, then logits, then the
        contract's after-forward start; returns (log, publication, collector, graph, runtime)."""
        torch = self.torch
        log = []
        pool = CollectorPool(log)
        runtime = FakeCudaRuntime(log)
        publication = pub.PDBatchPublication(pool, runtime, agg=True)
        pool._pd_batch_publication = publication
        graph = CollectorGraph(log, workspace)
        plan = NS(next_layer=0, last_layer=len(pool.layer_ids) - 1, pending=[], batch_collector=None,
                  checkpoint_group=None, slots=torch.tensor([5]), ring_dst=torch.tensor([-1]),
                  dense_required_after_commit=torch.tensor([1], dtype=torch.int32))
        controls = (None, torch.tensor([5]), torch.tensor([9]))  # track_slots, final_src, final_dst
        collector = self.bg.BatchCollector(pool, plan, graph=graph, token_count=8,
                                           publication=publication, launch_at_last_layer=early)
        with collector:
            for lid in pool.layer_ids:
                log.append(f"layer-{lid}")
                collector.add(lid, torch.zeros(1, 2, 2, 2), None, *controls)
            if reader_inside:
                pool.pside_join()
            log.append("logits")
        publication.start_after_forward()  # gdn_prefill_agg_contract forward() after logits
        log.append("forward-returned")
        self.plan, self.controls = plan, controls
        return log, publication, collector, graph, runtime

    def test_early_launches_from_last_add_and_exit_is_noop(self):
        with self.assertLogs(pub.logger, "INFO") as logs:
            log, p, collector, graph, _ = self._forward(early=True)
        self.assertTrue(any("K42_LAUNCH_AT_LAST_LAYER_ACTIVE" in line for line in logs.output))
        self.assertEqual(log, [
            "pside-join", "layer-0", "invalidate", "snapshot-0", "bind-controls",
            "layer-4", "snapshot-1", "layer-8", "snapshot-2",
            "side-wait-forward-done", "k31-replay controls_bound=entry-key", "side-done-1",
            "logits", "forward-returned"])
        self.assertTrue(collector.published)
        self.assertEqual(self.plan.pending, [])
        self.assertEqual(len(graph.runs), 1)
        self.assertEqual(graph.runs[0]["states"], [(("slab", i), None) for i in range(3)])
        self.assertEqual(p.stats["launched_at_last_layer"], 1)
        self.assertEqual(p.transfer_event(), "ticket-1")  # overlap fence gets the side-stream event
        p.join_reader(None)  # next forward's runner hook
        self.assertEqual(log[-1], "current-wait-ticket-1")
        self.assertIsNone(p.pending)
        self.assertEqual(p.stats["joined_in_forward"], 0)

    def test_switch_off_keeps_the_after_logits_launch(self):
        log, p, collector, graph, _ = self._forward(early=False)
        self.assertEqual(log, [
            "pside-join", "layer-0", "invalidate", "snapshot-0", "layer-4", "snapshot-1",
            "layer-8", "snapshot-2", "logits",
            "side-wait-forward-done", "k31-replay", "side-done-1", "forward-returned"])
        self.assertNotIn("controls_bound", graph.runs[0])
        self.assertEqual(p.stats["launched_at_last_layer"], 0)

    def test_early_records_only_controls_on_the_side_stream(self):
        _, _, _, _, runtime = self._forward(early=True)
        early_inputs, early_controls = runtime.inputs[0], self.controls
        _, _, _, graph, runtime = self._forward(early=False)
        late_inputs = runtime.inputs[0]
        self.assertEqual([id(t) for t in early_inputs], [id(t) for t in early_controls])
        # the after-logits route still records the slab views and plan tensors (FAST_BIND off)
        self.assertEqual(len(late_inputs), 2 * 3 + 3 + 3)
        self.assertIs(late_inputs[-3], self.plan.slots)

    def test_reader_inside_forward_after_launch_is_counted(self):
        log, p, _, _, _ = self._forward(early=True, reader_inside=True)
        self.assertEqual(p.stats["joined_in_forward"], 1)
        self.assertLess(log.index("current-wait-ticket-1"), log.index("logits"))
        self.assertEqual(log.count("k31-replay controls_bound=entry-key"), 1)

    def test_route_policy(self):
        c = self.contract
        ws = NS(workspace=object())
        self.assertTrue(c.launch_early("route", True, graph=ws))
        self.assertFalse(c.launch_early("route", False, graph=ws))
        self.assertFalse(c.launch_early(None, True, graph=ws))  # forward-stream route
        for debug in (dict(digest=object()), dict(dump=object()), dict(flags="")):
            self.assertFalse(c.launch_early("route", True, graph=ws, **debug))
        with self.assertLogs(c.logger, "WARNING"):
            self.assertFalse(c.launch_early("route", True, graph=NS(workspace=None)))
        c.set_k42_route("early")
        self.assertTrue(c.launch_early("route", False, graph=ws))
        c.set_k42_route("late")
        self.assertFalse(c.launch_early("route", True, graph=ws))
        c.set_k42_route(None)
        with self.assertRaises(ValueError):
            c.set_k42_route("side")

    def test_prebound_controls_equal_a_full_bind(self):
        """Real BatchBuffers / PrefillBatchGraph on CPU: first-add control bind + controls_bound
        replay leaves the same static inputs as today's bind at launch."""
        torch, bg = self.torch, self.bg
        pool = NS(hv=2, v=4, k=4, layer_ids=[0, 4, 8], a=torch.zeros(1), dense_ring=[torch.zeros(1)] * 3,
                  init_omega=lambda batch: None, ring_generation=0)
        graph = bg.PrefillBatchGraph(include_tail=False)
        plan = NS(slots=torch.tensor([7, 3, 11]), ring_dst=torch.tensor([2, -1, 5]),
                  dense_required_after_commit=torch.tensor([1, 0, 1], dtype=torch.int32))
        gen = torch.Generator().manual_seed(1)
        states = [(torch.randn(3, 2, 4, 4, generator=gen), torch.randn(3, 2, 4, 4, generator=gen))
                  for _ in pool.layer_ids]
        controls = (torch.tensor([20, 21, 22]), torch.tensor([7, 3, 11]), torch.tensor([30, -1, 31]))
        key, normal_batch, tracked_batch, joined = graph.select(pool, plan, states[0], controls[0], "e", "p")
        self.assertEqual((normal_batch, tracked_batch, joined), (4, 4, False))
        buffers = bg.BatchBuffers(pool, normal_batch, tracked_batch, {}, include_tail=False)
        replays = []
        graph.entries[key] = (buffers, NS(replay=lambda: replays.append("replay")))
        graph.warmed = True

        def static():
            names = ("slots", "ring_dst", "track_slots", "final_src", "final_dst", "required")
            out = {n: getattr(buffers, n).clone() for n in names}
            out.update(normal=[t.clone() for t in buffers.normal], tracked=[t.clone() for t in buffers.tracked])
            return out

        graph.run(pool, plan, states, *controls, eager="e", policy="p")
        today = static()
        for t in [buffers.slots, buffers.ring_dst, buffers.track_slots, buffers.final_src, buffers.final_dst,
                  buffers.required, *buffers.normal, *buffers.tracked]:
            t.fill_(99)
        bound = graph.bind_controls(pool, plan, states[0], *controls, eager="e", policy="p")
        self.assertEqual(bound, key)
        graph.run(pool, plan, states, *controls, eager="e", policy="p", controls_bound=bound)
        p1 = static()
        for name, value in today.items():
            for x, y in (zip(value, p1[name]) if isinstance(value, list) else ((value, p1[name]),)):
                self.assertTrue(torch.equal(x, y), name)
        self.assertEqual(replays, ["replay", "replay"])
        with self.assertRaisesRegex(RuntimeError, "another graph entry"):
            graph.run(pool, plan, states, *controls, eager="e", policy="p", controls_bound="other")

    def test_launch_inside_graph_pool_replay_scope(self):
        """Fake trunk replay: segments under graph_pool_replay_scope (borrow on), breaks run the
        collector adds; the side-stream launch happens inside the scope and must not claim it."""
        poolmod = _load_runner_pool()
        state = poolmod._get_graph_pool_borrow_state()
        state.static_runs = [(0, 1)]
        self.assertTrue(poolmod.graph_pool_borrow_enabled())
        torch = self.torch
        log = []
        pool = CollectorPool(log)
        runtime = FakeCudaRuntime(log)
        seen = []
        launch = runtime.launch_after_forward

        def launch_in_scope(operation, inputs):
            seen.append(state.active_user)
            return launch(operation, inputs)

        runtime.launch_after_forward = launch_in_scope
        publication = pub.PDBatchPublication(pool, runtime, agg=True)
        pool._pd_batch_publication = publication
        graph = CollectorGraph(log)
        plan = NS(next_layer=0, last_layer=2, pending=[], batch_collector=None, checkpoint_group=None,
                  slots=torch.tensor([5]), ring_dst=torch.tensor([-1]), dense_required_after_commit=None)
        controls = (None, torch.tensor([5]), torch.tensor([9]))
        collector = self.bg.BatchCollector(pool, plan, graph=graph, token_count=8, publication=publication,
                                           launch_at_last_layer=True)
        breaks = [lambda lid=lid: collector.add(lid, torch.zeros(1), None, *controls) for lid in pool.layer_ids]
        with collector:
            with poolmod.graph_pool_replay_scope():  # BreakableCUDAGraph.replay: segment, break, ...
                for i, fn in enumerate(breaks):
                    log.append(f"segment-{i}")
                    fn()
                log.append("segment-tail")
            self.assertIsNone(state.active_user)
        self.assertEqual(seen, ["CUDA graph"])
        self.assertLess(log.index("side-done-1"), log.index("segment-tail"))
        self.assertTrue(collector.published)


def _load_runner_pool():
    """runner_utils/pool.py under a private name with its runtime imports stubbed (CPU)."""
    name = "p1_test_runner_pool"
    if name in sys.modules:
        return sys.modules[name]
    resources = NS(graph_pool_borrow=None)
    stubs = {
        "sglang.srt.environ": NS(envs=NS(SGLANG_ENABLE_GRAPH_POOL_BORROW=NS(get=lambda: True))),
        "sglang.srt.runtime_context": NS(get_resources=lambda: resources, get_stream=lambda *a: None),
        "sglang.srt.utils": NS(is_cuda=lambda: True),
        "sglang.srt.utils.cuda_vmm_utils": NS(BumpArenaStub=object),
    }
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        return _load(name, "python/sglang/srt/model_executor/runner_utils/pool.py")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


if __name__ == "__main__":
    unittest.main()
