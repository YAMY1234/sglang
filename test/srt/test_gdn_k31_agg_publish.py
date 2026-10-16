"""K4.2 (docs/223): AGG prompt-end k31 on the publication side stream.

CPU only; fake CUDA runtimes record the stream order. The default (no done
event) path keeps the forward-stream completion assertion; the deferred path
accepts a pending side-stream event because every GPU reader joins it.
"""
import copy
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

    def launch_after_forward(self, operation, inputs):
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


if __name__ == "__main__":
    unittest.main()
