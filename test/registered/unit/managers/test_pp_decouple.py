"""PP stage decoupling (SGLANG_PP_DECOUPLE_HOST): bounded send windows and the
per-epoch state digest that checks every stage holds the same scheduler state.

Decoupled stages reach a micro-batch epoch at different wall times, but each one
must apply the same allocator and radix tree operations in the same epoch order;
the digest, compared per epoch, is how a run proves that.
"""

import random
import threading
import time
import unittest
import weakref
from array import array
from collections import defaultdict, deque
from contextlib import nullcontext
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch  # noqa: E402

from sglang.srt.distributed import parallel_state  # noqa: E402
from sglang.srt.managers import pp_decouple  # noqa: E402
from sglang.srt.managers.pp_decouple import PPSendWindow, PPSendWindows  # noqa: E402
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Req,
    ReqKvInfo,
    ScheduleBatch,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin  # noqa: E402
from sglang.srt.runtime_context import get_parallel  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator  # noqa: E402
from sglang.srt.mem_cache.base_prefix_cache import (  # noqa: E402
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey  # noqa: E402

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _Payload:
    is_cpu = False


class _FakeWork:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def wait(self):
        self.log.append(self.name)


class _FakeP2PWork:
    """P2PWork shape: the work handle and the tensor it keeps alive."""

    def __init__(self, name, log):
        self.work = _FakeWork(name, log)
        self.payload = _Payload()


def _round(name, log, n=2):
    return [_FakeP2PWork(f"{name}.{i}", log) for i in range(n)]


class TestPPSendWindow(CustomTestCase):
    def test_waits_only_rounds_beyond_bound_oldest_first_and_releases_them(self):
        log = []
        window = PPSendWindow(bound=2)
        rounds = [_round(f"r{i}", log) for i in range(4)]
        payloads = [weakref.ref(work.payload) for r in rounds for work in r]
        for r in rounds[:2]:
            self.assertEqual(window.push(works=r), 0)
        self.assertEqual(log, [])
        self.assertEqual(window.push(works=rounds[2]), 1)
        self.assertEqual(window.push(works=rounds[3]), 1)
        self.assertEqual(log, ["r0.0", "r0.1", "r1.0", "r1.1"])
        # Callers clear their work list after handing it over (_pp_commit_comm_work).
        for r in rounds:
            r.clear()
        self.assertEqual([ref() is None for ref in payloads], [True] * 4 + [False] * 4)
        # Empty rounds (ranks that sent nothing) do not occupy the window.
        self.assertEqual(window.push(works=[]), 0)
        self.assertEqual(window.drain(), 2)
        self.assertEqual(log[4:], ["r2.0", "r2.1", "r3.0", "r3.1"])
        self.assertTrue(all(ref() is None for ref in payloads))

    def test_peak_counts_outstanding_rounds_and_restarts_at_the_current_depth(self):
        log = []
        windows = PPSendWindows.create(bound=3, tensor_bound=1)
        for i in range(3):
            windows.req.push(works=_round(f"q{i}", log))
            windows.proxy.push(works=_round(f"p{i}", log))
        self.assertEqual(
            windows.take_peaks(),
            {"req": 3, "proxy": 2, "output": 0, "transfer": 0, "release": 0},
        )
        self.assertEqual(windows.take_peaks()["req"], 3)
        windows.req.drain()
        self.assertEqual(windows.take_peaks()["req"], 3)
        self.assertEqual(windows.take_peaks()["req"], 0)

    def test_inbox_counts_pops_that_found_their_message_received(self):
        items = iter(range(3))
        gate = threading.Event()

        def recv_one():
            gate.wait()
            return next(items)

        inbox = pp_decouple.PPRecvInbox(recv_one=recv_one, depth=2, name="t-inbox")
        gate.set()
        deadline = time.monotonic() + 10
        while inbox._queue.qsize() < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertEqual((inbox.pop(), inbox.pop(), inbox.pop()), (0, 1, 2))
        pops, ready = inbox.take_counts()
        self.assertEqual(pops, 3)
        self.assertGreaterEqual(ready, 2)
        self.assertEqual(inbox.take_counts(), (0, 0))

    def test_commit_windows_sends_but_waits_a_cuda_graph_fenced_proxy_now(self):
        log = []
        scheduler = SchedulerPPMixin()
        scheduler.pp_comm_stream_ctx = nullcontext()
        scheduler.scheduler_stage_metrics = None
        scheduler.pp_send_windows = PPSendWindows.create(bound=1)
        recorded = []
        scheduler._pp_record_comm_event = lambda: recorded.append("event") or "event"

        work = _round("req0", log)
        scheduler._pp_commit_comm_work(work, window=scheduler.pp_send_windows.req)
        self.assertEqual((log, work, len(scheduler.pp_send_windows.req)), ([], [], 1))

        scheduler.send_proxy_work = _round("proxy0", log)
        scheduler.send_proxy_requires_forward_fence = False
        scheduler._pp_commit_proxy_send_work()
        self.assertEqual(log, [])
        # A CUDA-graph proxy aliases replay buffers: the next forward is fenced on it.
        scheduler.send_proxy_work = _round("proxy1", log)
        scheduler.send_proxy_requires_forward_fence = True
        scheduler._pp_commit_proxy_send_work()
        self.assertEqual(log, ["proxy1.0", "proxy1.1"])
        self.assertEqual(scheduler.pp_send_done_event, "event")
        self.assertEqual(len(scheduler.pp_send_windows.proxy), 1)


class _CountedWork:
    """A gloo send Work: is_completed() turns true only once it is waited."""

    def __init__(self):
        self.waits = 0

    def wait(self):
        self.waits += 1

    def is_completed(self):
        return self.waits > 0


class TestHostSendsBeforeProfilerStop(CustomTestCase):
    def test_waits_each_unwaited_main_thread_send_once_and_never_a_waited_one(self):
        """torch.profiler.stop() frees the events of RecordFunctions that gloo sends opened
        under it; a send waited after the stop writes into that memory. The stop waits the
        tracked sends first. A send waited twice blocks, so one the loop already waited is
        skipped and the loop's own later wait of a drained send must not reach gloo."""
        parallel_state.enable_host_send_tracking()
        parallel_state.track_host_sends()
        works = [_CountedWork() for _ in range(3)]
        sends = [parallel_state.host_send(w, torch.zeros(1)) for w in works]
        sends[0].work.wait()  # the loop's own window wait, before the stop
        other = []
        thread = threading.Thread(
            target=lambda: other.append(parallel_state.host_send(_CountedWork(), None))
        )
        thread.start()
        thread.join()

        self.assertEqual(parallel_state.wait_host_sends(), 2)
        for send in sends[1:]:
            send.work.wait()
        self.assertEqual([w.waits for w in works], [1, 1, 1])
        self.assertEqual(other[0].work.waits, 0)
        parallel_state.host_send(_CountedWork(), None)  # after the stop: not tracked
        self.assertEqual(parallel_state.wait_host_sends(), 0)


class TestPPLateRecv(CustomTestCase):
    def _scheduler(self, late, log):
        scheduler = SchedulerPPMixin()
        scheduler.pp_comm_stream_ctx = nullcontext()
        scheduler._pp_tensor_dict_inbox = defaultdict(deque)
        scheduler._pp_late_recv = late
        scheduler.launch_event = "fwd-end"
        scheduler.pp_comm_stream = SimpleNamespace(
            wait_event=lambda event: log.append(("wait", event))
        )
        scheduler.pp_group = SimpleNamespace(
            recv_tensor_dict=lambda **kw: log.append(("recv",)) or {"__msg_type__": "proxy"}
        )
        scheduler._pp_record_comm_event = lambda: None
        return scheduler

    def test_late_receive_is_posted_behind_the_previous_forward(self):
        """A receive kernel posted during a forward spins there until its sender sends and
        blocks the forward's cooperative kernels; late receive orders it after that forward."""
        log = []
        self._scheduler(True, log)._pp_recv_typed_dict(expected_kind="proxy")
        self.assertEqual(log, [("wait", "fwd-end"), ("recv",)])
        log.clear()
        self._scheduler(False, log)._pp_recv_typed_dict(expected_kind="proxy")
        self.assertEqual(log, [("recv",)])


class TestPPCEPush(CustomTestCase):
    def test_layout_is_aligned_and_the_slot_reuse_waits_for_the_previous_user(self):
        from sglang.srt.managers.pp_ce_push import CESlice, free_target, pack_layout, slot_of

        layout, used = pack_layout(
            [("hidden_states", torch.bfloat16, (3, 5), 30), ("residual", torch.bfloat16, (3, 5), 30)]
        )
        self.assertEqual(
            [(k, v.offset) for k, v in layout], [("hidden_states", 0), ("residual", 256)]
        )
        self.assertEqual(used, 512)
        self.assertIsInstance(layout[0][1], CESlice)
        # Two slots: seq 2 reuses slot 0 and waits until seq 0 was released (released value = seq + 1).
        self.assertEqual((slot_of(2, 2), free_target(2, 2)), (0, 1))
        self.assertLessEqual(free_target(1, 2), 0)

    def _scheduler(self, last, log):
        from sglang.srt.managers.pp_ce_push import CEWait

        scheduler = SchedulerPPMixin()
        scheduler.pp_comm_stream_ctx = nullcontext()
        scheduler.forward_stream = "forward"
        scheduler.pp_send_done_event = None
        scheduler.schedule_stream = "schedule"
        scheduler.forward_stream = SimpleNamespace(
            name="forward",
            wait_stream=lambda s: None,
            wait_event=lambda e: log.append(("fwd_wait_event", e)),
        )
        scheduler.pp_group = SimpleNamespace(is_last_rank=last)
        scheduler._pp_record_comm_event = lambda: "comm-tail"
        ce = SimpleNamespace(
            stream=SimpleNamespace(name="push", wait_event=lambda e: log.append(("push_wait", e))),
            wait_full=lambda stream, ss: log.append(("wait_full", stream.name, ss)),
            release=lambda stream, ss: log.append(("release", stream.name, ss)),
        )
        scheduler._pp_ce = ce
        scheduler.pp_proxy_recv_event = CEWait((1, 7))
        return scheduler

    def test_forward_waits_on_the_slot_and_releases_it_after_its_own_push(self):
        """The slot may be read in place by the stage's outgoing send (an in-place residual): a
        non-last stage releases it on the push stream after that push, an NCCL fallback after the
        fallback send's work; the last stage releases it on the forward stream."""
        log = []
        s = self._scheduler(False, log)
        s._pp_wait_forward_dependencies()
        self.assertEqual(log, [("wait_full", "forward", (1, 7))])
        s._pp_ce_release_input()
        self.assertEqual(log[1:], [("release", "push", (1, 7))])
        s._pp_ce_release_input()
        self.assertEqual(len(log), 2)

        log.clear()
        s = self._scheduler(False, log)
        s._pp_wait_forward_dependencies()
        sent = _round("nccl", log)
        s._pp_ce_release_input(nccl_send_work=sent)
        self.assertEqual(
            log[1:], ["nccl.0", "nccl.1", ("push_wait", "comm-tail"), ("release", "push", (1, 7))]
        )

        log.clear()
        s = self._scheduler(True, log)
        s._pp_wait_forward_dependencies()
        s._pp_ce_release_input()
        self.assertEqual(log[1:], [("release", "forward", (1, 7))])

    def test_a_fallback_send_after_a_pushed_input_waits_each_gloo_send_once(self):
        """A stage whose input came by push and whose output does not fit a slot sends it over
        NCCL, with the metadata over gloo. The release fences only the NCCL send; the gloo sends
        stay in the send window, which waits them once (a second wait never returns)."""
        log = []
        s = self._scheduler(False, log)
        s._pp_wait_forward_dependencies()
        host = [_CountedWork() for _ in range(2)]
        sent = [parallel_state.P2PWork(w, torch.zeros(1)) for w in host] + _round("nccl", log, n=1)
        s._pp_ce_release_input(nccl_send_work=sent)
        self.assertEqual(
            log[1:], ["nccl.0", ("push_wait", "comm-tail"), ("release", "push", (1, 7))]
        )
        window = PPSendWindow(0)
        window.push(sent)
        self.assertEqual([w.waits for w in host], [1, 1])


class _StageModel:
    """One PP stage's allocator and radix tree, driven by per-epoch operations."""

    def __init__(self, page_count=512):
        self.allocator = TokenToKVPoolAllocator(
            size=page_count,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self.req_pool = ReqToTokenPool(
            size=8, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        self.cache = RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=1,
            )
        )
        self.locked = {}

    def apply(self, op):
        kind, rid, arg = op
        if kind == "admit":
            key = RadixKey(array("q", arg))
            kv = self.allocator.alloc(len(arg))
            matched = self.cache.insert(InsertParams(key=key, value=kv)).prefix_len
            self.allocator.free(kv[:matched])
            node = self.cache.match_prefix(MatchPrefixParams(key=key)).last_device_node
            self.cache.inc_lock_ref(node)
            self.locked[rid] = node
        elif kind == "release":
            self.cache.dec_lock_ref(self.locked.pop(rid))
        else:
            self.cache.evict(EvictParams(num_tokens=arg))

    def digest(self):
        sizes, tensors = pp_decouple.collect_pp_state(
            token_to_kv_pool_allocator=self.allocator,
            req_to_token_pool=self.req_pool,
            tree_cache=self.cache,
        )
        return pp_decouple.pp_state_digest(sizes=sizes, tensors=tensors)


def _epoch_script(num_epochs, seed=7):
    """Admissions share a few system prompts (node splits), releases retire the
    oldest locked request, evictions free unlocked leaves."""
    rng = random.Random(seed)
    prompts = [[1000 * p + i for i in range(6)] for p in range(3)]
    active, script = deque(), []
    for epoch in range(num_epochs):
        ops = []
        for _ in range(rng.randint(0, 2)):
            rid = f"e{epoch}r{len(ops)}"
            suffix = [10 * len(script) + i for i in range(rng.randint(2, 6))]
            ops.append(("admit", rid, rng.choice(prompts) + suffix))
            active.append(rid)
        if active and rng.random() < 0.6:
            ops.append(("release", active.popleft(), None))
        if rng.random() < 0.3:
            ops.append(("evict", None, rng.randint(1, 12)))
        script.append(ops)
    return script


def _run_stages(script, num_stages, every, late_release_stage=None, seed=11):
    """Advance stages through the epochs in a random interleaving; each logs a
    digest line every `every` epochs, so log order is not epoch order."""
    rng = random.Random(seed)
    stages = [_StageModel() for _ in range(num_stages)]
    next_epoch = [0] * num_stages
    carried = [[] for _ in range(num_stages)]
    log = []
    while min(next_epoch) < len(script):
        rank = rng.choice([r for r in range(num_stages) if next_epoch[r] < len(script)])
        epoch = next_epoch[rank]
        ops = carried[rank] + [op for op in script[epoch] if op[0] != "release"]
        releases = [op for op in script[epoch] if op[0] == "release"]
        if rank == late_release_stage:
            # Models applying a relayed event when it arrives, not at its epoch.
            carried[rank] = releases
        else:
            ops = ops + releases
            carried[rank] = []
        for op in ops:
            stages[rank].apply(op)
        next_epoch[rank] = epoch + 1
        if (epoch + 1) % every == 0:
            record = pp_decouple.PPStateDigest(
                pp_rank=rank, dp_rank=0, epoch=epoch + 1, digest=stages[rank].digest()
            )
            log.append("INFO scheduler " + pp_decouple.format_digest_line(record))
    return log


class TestPPStateDigest(CustomTestCase):
    NUM_STAGES = 4
    NUM_EPOCHS = 48
    EVERY = 4

    def test_stages_agree_at_equal_epochs_whatever_their_interleaving(self):
        script = _epoch_script(self.NUM_EPOCHS)
        log = _run_stages(script, self.NUM_STAGES, self.EVERY)
        records = pp_decouple.parse_digest_lines(log)
        self.assertEqual(len(records), self.NUM_STAGES * self.NUM_EPOCHS // self.EVERY)
        # The interleaving must actually skew stages, or grouping is untested.
        self.assertNotEqual(
            [r.epoch for r in records], sorted(r.epoch for r in records)
        )
        report = pp_decouple.find_digest_mismatches(records, pp_size=self.NUM_STAGES)
        self.assertEqual((report.mismatched, report.incomplete), ({}, []))
        self.assertGreater(len({r.digest for r in records}), 1)

    def test_an_event_applied_off_epoch_is_flagged_at_exactly_those_epochs(self):
        script = _epoch_script(self.NUM_EPOCHS)
        log = _run_stages(script, self.NUM_STAGES, self.EVERY, late_release_stage=2)
        report = pp_decouple.find_digest_mismatches(
            pp_decouple.parse_digest_lines(log), pp_size=self.NUM_STAGES
        )
        expected = {
            (epoch, 0)
            for epoch in range(self.EVERY, self.NUM_EPOCHS + 1, self.EVERY)
            if any(op[0] == "release" for op in script[epoch - 1])
        }
        self.assertTrue(expected)
        self.assertEqual(set(report.mismatched), expected)
        for stages in report.mismatched.values():
            others = {digest for rank, digest in stages.items() if rank != 2}
            self.assertEqual(len(others), 1)
            self.assertNotIn(stages[2], others)

    def test_free_list_order_is_part_of_the_state(self):
        first, second = _StageModel(), _StageModel()
        for stage, order in ((first, (0, 1)), (second, (1, 0))):
            kv = [stage.allocator.alloc(4), stage.allocator.alloc(4)]
            for i in order:
                stage.allocator.free(kv[i])
        # Same free set, different order: the next allocation would diverge.
        self.assertEqual(
            first.allocator.available_size(), second.allocator.available_size()
        )
        self.assertNotEqual(first.digest(), second.digest())

    def test_a_cut_short_log_is_incomplete_not_mismatched(self):
        records = [
            pp_decouple.PPStateDigest(pp_rank=r, dp_rank=0, epoch=e, digest="d")
            for e in (4, 8)
            for r in range(self.NUM_STAGES)
            if not (e == 8 and r == 3)
        ]
        report = pp_decouple.find_digest_mismatches(records, pp_size=self.NUM_STAGES)
        self.assertEqual((report.mismatched, report.incomplete), ({}, [(8, 0)]))


def _batch(*rows):
    """rows: (rid, prefix_len, extend_len)."""
    return SimpleNamespace(
        reqs=[SimpleNamespace(rid=rid) for rid, _, _ in rows],
        prefix_lens=[prefix for _, prefix, _ in rows],
        extend_lens=[extend for _, _, extend in rows],
    )


def _record(rank, epoch, **parts):
    return pp_decouple.PPStateDigest(
        pp_rank=rank,
        dp_rank=0,
        epoch=epoch,
        digest=pp_decouple.combine_parts(parts, ("batches", "slots")),
        parts=parts,
    )


class TestPPRankInvariantDigest(CustomTestCase):
    def test_batch_log_matches_only_for_the_same_batches_in_build_order(self):
        logs = [pp_decouple.PPBatchLog() for _ in range(3)]
        first, second = _batch(("r0", 0, 16), ("r1", 4096, 512)), _batch(("r2", 0, 8))
        for log in logs[:2]:
            for batch in (first, None, second):
                log.add(batch)
        # Same request, split at another prefix length.
        for batch in (first, None, _batch(("r2", 4, 4))):
            logs[2].add(batch)
        self.assertEqual(logs[0].hexdigest(), logs[1].hexdigest())
        self.assertNotEqual(logs[0].hexdigest(), logs[2].hexdigest())

    def test_slot_map_ignores_collection_order_duplicates_and_slotless_requests(self):
        reqs = [SimpleNamespace(rid=f"r{i}", kv=ReqKvInfo(req_pool_idx=i)) for i in range(4)]
        waiting = SimpleNamespace(rid="w", kv=ReqKvInfo())
        self.assertEqual(
            pp_decouple.slot_map_digest(reqs),
            pp_decouple.slot_map_digest(reqs[::-1] + reqs[:2] + [waiting]),
        )
        swapped = [
            SimpleNamespace(rid=r.rid, kv=ReqKvInfo(req_pool_idx=3 - i)) for i, r in enumerate(reqs)
        ]
        self.assertNotEqual(
            pp_decouple.slot_map_digest(reqs), pp_decouple.slot_map_digest(swapped)
        )

    def test_digest_reads_fields_that_real_req_and_schedule_batch_have(self):
        """The digest ran only against stand-ins and crashed at its first epoch on the
        server (Req keeps its pool row under kv); exercise it on the real classes."""
        reqs = [
            Req(f"r{i}", "", array("q", [1, 2, 3]), SamplingParams()) for i in range(2)
        ]
        reqs[0].kv.req_pool_idx = 5
        batch = ScheduleBatch(reqs=reqs, batch_is_full=False)
        batch.prefix_lens, batch.extend_lens = [0, 64], [3, 3]
        log = pp_decouple.PPBatchLog()
        log.add(batch)
        self.assertEqual(
            pp_decouple.slot_map_digest(reqs),
            pp_decouple.slot_map_digest([reqs[0]]),
        )
        self.assertEqual(len(log.hexdigest()), 16)

    def test_a_proxy_from_another_micro_batch_stops_the_stage(self):
        """A stage that ran its forward on another batch's activations would corrupt KV silently when
        the token counts happen to match; the batch identity in the proxy makes it fail loudly."""
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

        def batch(rids, tokens):
            b = ScheduleBatch(
                reqs=[Req(r, "", array("q", [1, 2]), SamplingParams()) for r in rids],
                batch_is_full=False,
            )
            b.extend_num_tokens = tokens
            return b

        scheduler = SchedulerPPMixin()
        scheduler.forward_ct = 3
        a, b = batch(["r1", "r2"], 16384), batch(["r3", "r4"], 16384)
        proxy = PPProxyTensors({"hidden_states": None, pp_decouple.PROXY_BATCH_KEY: pp_decouple.proxy_batch_tag(a)})
        scheduler._pp_assert_proxy_batch(proxy, a)
        self.assertNotIn(pp_decouple.PROXY_BATCH_KEY, proxy.tensors)
        proxy.tensors[pp_decouple.PROXY_BATCH_KEY] = pp_decouple.proxy_batch_tag(a)
        with get_parallel().override(pp_rank=4), self.assertRaisesRegex(
            RuntimeError, "arrived for batch"
        ):
            scheduler._pp_assert_proxy_batch(proxy, b)

    def test_an_informational_part_does_not_make_stages_mismatch(self):
        records = [
            _record(
                rank, epoch, batches="b", slots="s", device=f"d{rank * (epoch > 4)}"
            )
            for epoch in (4, 8)
            for rank in range(4)
        ] + [
            _record(rank, 12, batches=f"b{rank == 3}", slots="s", device=f"d{rank}")
            for rank in range(4)
        ]
        report = pp_decouple.find_digest_mismatches(records, pp_size=4)
        self.assertEqual(set(report.mismatched), {(12, 0)})
        self.assertEqual(
            pp_decouple.first_part_mismatches(records),
            {("device", 0): 8, ("batches", 0): 12},
        )

    def test_scheduler_logs_the_invariant_parts_of_its_live_requests(self):
        stage = _StageModel()
        stage.apply(("admit", "r0", [1, 2, 3, 4]))
        running = SimpleNamespace(
            rid="r1", kv=ReqKvInfo(req_pool_idx=stage.req_pool.alloc_rows(1)[0])
        )
        inflight = SimpleNamespace(
            rid="r2", kv=ReqKvInfo(req_pool_idx=stage.req_pool.alloc_rows(1)[0])
        )
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=stage.allocator,
            req_to_token_pool=stage.req_pool,
            tree_cache=stage.cache,
            mbs=[None, SimpleNamespace(reqs=[running])],
            running_mbs=[SimpleNamespace(reqs=[running]), SimpleNamespace(reqs=[])],
            disagg_prefill_inflight_queue=[inflight],
            chunked_req=None,
        )
        batch_log = pp_decouple.PPBatchLog()
        batch_log.add(_batch(("r1", 0, 4)))
        with (
            get_parallel().override(pp_rank=0, attn_dp_rank=0),
            self.assertLogs(
                "sglang.srt.managers.scheduler_pp_mixin", level="INFO"
            ) as logs,
        ):
            SchedulerPPMixin._pp_log_state_digest(scheduler, 256, batch_log)
        (record,) = pp_decouple.parse_digest_lines(logs.output)
        self.assertEqual(
            (record.epoch, sorted(record.parts)), (256, ["batches", "device", "slots"])
        )
        self.assertEqual(
            record.parts["slots"], pp_decouple.slot_map_digest([running, inflight])
        )
        self.assertEqual(
            record.digest, pp_decouple.combine_parts(record.parts, ("batches", "slots"))
        )

    def test_records_without_parts_still_parse(self):
        line = pp_decouple.DIGEST_LOG_PREFIX + (
            '{"pp_rank":1,"dp_rank":0,"epoch":256,"digest":"abc"}'
        )
        (record,) = pp_decouple.parse_digest_lines([line])
        self.assertEqual((record.digest, record.parts), ("abc", {}))


if __name__ == "__main__":
    unittest.main()
