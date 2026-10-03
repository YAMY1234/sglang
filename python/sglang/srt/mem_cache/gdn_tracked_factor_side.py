"""Tracked-only replay of the whole-layer k31 graph, with explicit readers.

The ordinary PrefillBatchGraph owns the input buffers. F and T use its exact
non-joint graph body and bucket shapes, but distinct capture arenas. There is
one pending T per pool; every rebind and publication waits for its event.
"""

import logging

import torch
import triton
import triton.language as tl

from sglang.srt.utils.graph_capture import graph_capture_lock

logger = logging.getLogger(__name__)


def disjoint_destinations(live, tracked, final_src, final_dst):
    """Return a fallback reason; padding slots do not participate."""
    live = [int(i) for i in live if i >= 0]
    tracked = [int(i) for i in tracked if i >= 0]
    src = [int(i) for i in final_src if i >= 0]
    dst = [int(i) for i in final_dst if i >= 0]
    if set(tracked).intersection(src):
        return "final_from_tracked"
    if not tracked or len(set(tracked)) != len(tracked):
        return "aliased_slots"
    if set(tracked).intersection(live + dst):
        return "aliased_slots"
    return None


@triton.jit
def _invalidate_tracked_masked(VALID, SLOTS, MASK, N: tl.constexpr,
                               BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    active = tl.load(MASK + row, row < N, other=0) != 0
    slot = tl.load(SLOTS + row, (row < N) & active, other=-1).to(tl.int64)
    # A false mask must not read or write a concurrently published checkpoint.
    tl.store(VALID + slot, 0, (row < N) & active & (slot >= 0))


def invalidate_tracked_masked(valid, slots, mask):
    if slots.numel():
        _invalidate_tracked_masked[(1,)](
            valid, slots, mask, slots.numel(), triton.next_power_of_2(slots.numel())
        )


class TrackedFactorSide:
    def __init__(self, pool, whole_graph, *, deferred=False, after_boundary=False,
                 low_priority=False):
        if after_boundary and not deferred:
            raise ValueError("tracked after-boundary replay requires DEFERRED=1")
        self.pool = pool
        self.whole_graph = whole_graph
        self.entries = {}
        # Deferred: T starts after F and alternates between two tracked input
        # sets, so a bind only waits for the T two commits back.
        self.deferred = deferred
        self.after_boundary = after_boundary
        self.pending_launch = None
        self.boundary_done = torch.cuda.Event()
        self.alt_entries = {}
        self.parity = 0
        self.final_done = torch.cuda.Event()
        self.set_done = (torch.cuda.Event(), torch.cuda.Event())
        self.set_recorded = [False, False]
        self.stream = (torch.cuda.Stream(device=pool.a.device, priority=0)
                       if low_priority else torch.cuda.Stream(device=pool.a.device))
        logger.info(
            "k31 tracked replay policy: after_boundary=%d low_priority=%d "
            "side_priority=%d main_priority=%d (smaller is higher; no preemption)",
            after_boundary, low_priority, self.stream.priority,
            torch.cuda.current_stream(pool.a.device).priority,
        )
        self.capture_stream = torch.cuda.Stream(device=pool.a.device)
        # Captured temporaries MUST NOT alias between concurrently replayed F/T.
        self.final_arena = torch.cuda.graph_pool_handle()
        self.tracked_arena = torch.cuda.graph_pool_handle()
        self.bound = torch.cuda.Event()
        self.done = torch.cuda.Event()
        self.recorded = False
        self._prefill_side_pending = False
        self.waited_streams = set()
        self.stats = dict(split=0, fallback_final_from_tracked=0,
                          fallback_aliased_slots=0, fallback_no_tracked=0,
                          capture_fallback=0, joins=0, after_boundary=0,
                          early_reader=0, next_bind=0)

    def defer_until_boundary(self, graphs, k, producer):
        """Retain graph/arena ownership without enqueueing an unrecorded wait."""
        if not self.after_boundary or self.pending_launch is not None:
            raise RuntimeError("invalid or overlapping tracked pending launch")
        self.pending_launch = (graphs, k, producer)
        self.done = self.set_done[k]
        self.parity = 1 - k
        self._prefill_side_pending = True
        self.waited_streams.clear()

    def launch_pending(self, *, reason="after_boundary"):
        if self.pending_launch is None:
            return False
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("pending tracked replay must launch before graph capture")
        graphs, k, producer = self.pending_launch
        current = torch.cuda.current_stream(self.pool.a.device)
        if current == self.stream:
            raise RuntimeError("tracked writer cannot launch its own reader fence")
        if reason == "after_boundary" and current != producer:
            raise RuntimeError("tracked boundary hook changed the producer stream")
        # Early readers record on the original producer too. A CUDA wait on an
        # event never recorded is a no-op, so do not enqueue T until this point.
        self.boundary_done.record(producer)
        self.stream.wait_event(self.boundary_done)
        with torch.cuda.stream(self.stream):
            for graph in graphs:
                graph.replay()
            self.done.record(self.stream)
        self.set_recorded[k] = True
        self.recorded = self._prefill_side_pending = True
        self.pending_launch = None
        self.waited_streams.clear()
        self.stats[reason] += 1
        if self.stats[reason] == 1:
            logger.info("k31 tracked pending launch: reason=%s graphs=%d buffer=%d counts=%s",
                        reason, len(graphs), k, self.stats)
        self.log_stats()
        return True

    def prewarm(self, *, eager):
        before = torch.cuda.memory_allocated(self.pool.a.device)
        current = torch.cuda.current_stream(self.pool.a.device)
        for key, (buffers, _) in self.whole_graph.entries.items():
            if key[1] is None or key[-1]:
                continue
            graphs = []
            for branch, arena in (("normal", self.final_arena),
                                  ("tracked", self.tracked_arena)):
                self.capture_stream.wait_stream(current)
                with torch.cuda.stream(self.capture_stream):
                    buffers.evaluate(eager, branch=branch)
                current.wait_stream(self.capture_stream)
                graph = torch.cuda.CUDAGraph()
                with graph_capture_lock, torch.cuda.graph(
                    graph, stream=self.capture_stream, pool=arena,
                    capture_error_mode="thread_local"
                ):
                    buffers.evaluate(eager, branch=branch)
                graphs.append(graph)
            self.entries[key] = (buffers, *graphs)
            if self.deferred:
                self.alt_entries[key] = self._capture_alternate(buffers, eager)
        torch.cuda.synchronize(self.pool.a.device)
        logger.info(
            "k31 tracked side stream: enabled=1 join_branches=0 graphs=F,T "
            "signatures=%d extra_retained_bytes=%d",
            len(self.entries), torch.cuda.memory_allocated(self.pool.a.device) - before,
        )

    def _capture_alternate(self, buffers, eager):
        """Second input set: own tracked states and controls, shared normal states."""
        from .gdn_prefill_batch_graph import BatchBuffers

        shared = getattr(self, "_alt_shared", None)
        if shared is None:
            # F stays ordered on the producer stream, so only tracked inputs need a copy.
            shared = self._alt_shared = {
                k: v for k, v in self.whole_graph.shared.items() if k[0] == "normal"}
        alt = BatchBuffers(self.pool, buffers.batch, buffers.tracked_batch, shared,
                           include_tail=self.whole_graph.include_tail, join_branches=False)
        current = torch.cuda.current_stream(self.pool.a.device)
        graphs = []
        for branch, arena in (("normal", self.final_arena), ("tracked", self.tracked_arena)):
            self.capture_stream.wait_stream(current)
            with torch.cuda.stream(self.capture_stream):
                alt.evaluate(eager, branch=branch)
            current.wait_stream(self.capture_stream)
            graph = torch.cuda.CUDAGraph()
            with graph_capture_lock, torch.cuda.graph(
                graph, stream=self.capture_stream, pool=arena,
                capture_error_mode="thread_local"
            ):
                alt.evaluate(eager, branch=branch)
            graphs.append(graph)
        return (alt, *graphs)

    def join(self):
        if self.pending_launch is not None:
            if torch.cuda.current_stream(self.pool.a.device) == self.stream:
                return
            self.launch_pending(reason="early_reader")
        if not self.recorded:
            return
        current = torch.cuda.current_stream(self.pool.a.device)
        if current == self.stream:
            return  # A writer cannot consume another stream's reader fence.
        if torch.cuda.is_current_stream_capturing():
            if self.done.query():
                self.recorded = self._prefill_side_pending = False
                return
            raise RuntimeError("tracked factor work must finish before graph capture")
        # Publication and forward execution can use different CUDA streams.
        # Keep the event after the first join so a second reader also waits.
        identity = current.cuda_stream
        if identity not in self.waited_streams:
            current.wait_event(self.done)
            self.waited_streams.add(identity)
            self.stats["joins"] += 1
        self._prefill_side_pending = False

    def fallback(self, reason):
        key = "capture_fallback" if reason == "capture" else "fallback_" + reason
        self.stats[key] += 1
        self.log_stats()
        return False

    def log_stats(self):
        total = sum(v for k, v in self.stats.items() if k != "joins")
        if total == 1 or total % 500 == 0:
            logger.info("k31 tracked side stream counts: %s", self.stats)

    def run(self, plan, states, track_slots, final_src, final_dst, *, eager, policy):
        if torch.cuda.is_current_stream_capturing():
            return self.fallback("capture")
        self.launch_pending(reason="next_bind")
        if not self.deferred:
            self.join()  # J5: before any write to the shared static input buffers.
        if track_slots is None or not track_slots.numel() or states[0][1] is None:
            return self.fallback("no_tracked")
        from .gdn_prefill_commit_graph import BATCH_BUCKETS

        size = max(plan.slots.numel(), track_slots.numel())
        bucket = next(b for b in BATCH_BUCKETS if size <= b)
        normal = 1 if plan.slots.numel() == 1 else bucket
        tracked = 1 if states[0][1].shape[0] == 1 else bucket
        key = self.whole_graph.key(normal, tracked, eager, policy, False)
        # One small D2H for the alias guard, before either graph is enqueued.
        # Preserve #23's padded shapes and fixed directions, including B=3.
        controls = (plan.slots, track_slots, final_src, final_dst)
        rows = torch.cat([c.reshape(-1) for c in controls if c is not None]).tolist()
        values, offset = [], 0
        for control in controls:
            size = 0 if control is None else control.numel()
            values.append(rows[offset:offset + size])
            offset += size
        reason = disjoint_destinations(
            *values,
        )
        if reason is not None:
            return self.fallback(reason)
        current = torch.cuda.current_stream(self.pool.a.device)
        k = self.parity if self.deferred else 0
        buffers, final_graph, tracked_graph = (self.alt_entries if k else self.entries)[key]
        if self.deferred and self.set_recorded[k]:
            current.wait_event(self.set_done[k])  # J5: T two commits back used these inputs.
        buffers.bind(plan, states, track_slots, final_src, final_dst)
        self.bound.record(current)
        final_graph.replay()
        if self.after_boundary:
            self.defer_until_boundary((tracked_graph,), k, current)
            self.stats["split"] += 1
            self.log_stats()
            return True
        if self.deferred:
            # T after F: it overlaps boundary decode and the next host prep, not F.
            self.final_done.record(current)
            self.stream.wait_event(self.final_done)
            self.done = self.set_done[k]
            self.set_recorded[k] = True
            self.parity = 1 - k
        else:
            self.stream.wait_event(self.bound)
        with torch.cuda.stream(self.stream):
            tracked_graph.replay()
            self.done.record(self.stream)
        self.recorded = self._prefill_side_pending = True
        self.waited_streams.clear()
        self.stats["split"] += 1
        self.log_stats()
        return True
