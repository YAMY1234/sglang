"""Per-stage PP bubble diagnostics (SGLANG_PP_BUBBLE_DIAG, default off).

On every PP rank, partitions the scheduler thread's wall time into host work per
loop phase and blocking waits per primitive (exclusive: nested waits pause their
parent), times each forward and the stream gap before it with CUDA events (no
synchronization), and counts admission passes: queue length, prefetch-readiness
skips by reason, and passes that left budget unused while requests waited.
One ``PP bubble diag: {json}`` line is logged per rank per interval; with
SGLANG_PP_BUBBLE_DIAG_FWD_DIR set, each forward is also appended to a per-rank CSV.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from typing import Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Lower edges of the waiting-queue bins that per-pass relay waits are split by.
QUEUE_BINS = (0, 1, 16, 32, 60)
# Relay callers whose per-pass cost is binned by queue length.
VOTE_CALLER = "_can_terminate_prefetch"
TICKET_CALLER = "_check_pp_prefetch_progress"
_MAX_PENDING_FORWARDS = 256
# A forward counts as full when it carries this share of the chunk budget.
_FULL_SHARE = 0.95
FWD_CSV_HEADER = "wall_s,fwd_ms,gap_cpu_ms,gap_dep_ms,new_tokens,cached_tokens,has_load,queue_len\n"
_NULL = nullcontext()


def _bin(queue_len: int) -> int:
    i = 0
    while i + 1 < len(QUEUE_BINS) and queue_len >= QUEUE_BINS[i + 1]:
        i += 1
    return i


class _Span:
    __slots__ = ("diag", "label", "prev", "t0")

    def __init__(self, diag: "PPBubbleDiag", label: str):
        self.diag = diag
        self.label = label

    def __enter__(self):
        self.prev = self.diag._switch(self.label)
        self.diag.calls[self.label] += 1
        self.t0 = self.diag._t

    def __exit__(self, *exc):
        self.diag._switch(self.prev)
        if self.diag.slow_ns and self.diag._t - self.t0 >= self.diag.slow_ns:
            logger.info(
                "PP slow span rank %d: %s ms=%.1f wall=%.3f",
                self.diag.pp_rank,
                self.label,
                (self.diag._t - self.t0) / 1e6,
                time.time(),
            )
        return False


def _fold_stack(frame, max_depth: int = 96) -> str:
    """Root-to-leaf frames below the scheduler's event loop. The loop frame and the
    leaf keep their line numbers (which call, which line); frames between do not."""
    names = []
    while frame is not None and len(names) < max_depth:
        code = frame.f_code
        name = f"{code.co_name}@{os.path.basename(code.co_filename)}"
        if not names or code.co_name.startswith("event_loop"):
            name += f":{frame.f_lineno}"
        names.append(name)
        if code.co_name.startswith("event_loop"):
            break
        frame = frame.f_back
    return ";".join(reversed(names))


class _StackSampler:
    """Samples one thread's Python stack from a daemon thread and counts it under the
    diag's current label. Each sample holds the GIL for one stack walk, and is taken
    when the sampled thread next releases the GIL (long GIL-holding C calls show up
    on the line after them)."""

    def __init__(self, diag: "PPBubbleDiag", thread_id: int, hz: float, path: str):
        self._diag = diag
        self._thread_id = thread_id
        self._period = 1.0 / hz
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._file = open(path, "a", buffering=1 << 20)
        threading.Thread(target=self._run, name="pp-stack-sampler", daemon=True).start()

    def _run(self) -> None:
        while True:
            time.sleep(self._period)
            frame = sys._current_frames().get(self._thread_id)
            if frame is None:
                continue
            key = self._diag._label + ";" + _fold_stack(frame)
            with self._lock:
                self._counts[key] += 1

    def flush(self) -> None:
        with self._lock:
            counts, self._counts = self._counts, defaultdict(int)
        self._file.write(f"# {time.time():.3f}\n")
        for key, n in counts.items():
            self._file.write(f"{key} {n}\n")
        self._file.flush()


class _GCTimer:
    """gc.callbacks timer: per-generation collections, total and max ms per interval, and
    one log line (with wall time) for every collection over long_ms."""

    def __init__(self, pp_rank: int, long_ms: float = 50.0):
        self._pp_rank = pp_rank
        self._long_ms = long_ms
        self._t0 = 0.0
        self.it_ms = 0.0
        self._reset()
        logger.info(
            "PP GC rank %d: enabled=%s threshold=%s freeze_count=%d counts=%s",
            pp_rank,
            gc.isenabled(),
            gc.get_threshold(),
            gc.get_freeze_count(),
            gc.get_count(),
        )
        gc.callbacks.append(self._callback)

    def take_iteration_ms(self) -> float:
        ms, self.it_ms = self.it_ms, 0.0
        return ms

    def _reset(self) -> None:
        self.n = [0, 0, 0]
        self.ms = [0.0, 0.0, 0.0]
        self.max_ms = [0.0, 0.0, 0.0]
        self.collected = [0, 0, 0]

    def _callback(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._t0 = time.perf_counter()
            return
        ms = (time.perf_counter() - self._t0) * 1e3
        self.it_ms += ms
        gen = info.get("generation", 0)
        self.n[gen] += 1
        self.ms[gen] += ms
        self.max_ms[gen] = max(self.max_ms[gen], ms)
        self.collected[gen] += info.get("collected", 0)
        if ms > self._long_ms:
            logger.info(
                "PP GC long rank %d: gen=%d ms=%.1f wall=%.3f collected=%d uncollectable=%d",
                self._pp_rank,
                gen,
                ms,
                time.time(),
                info.get("collected", 0),
                info.get("uncollectable", 0),
            )

    def snapshot(self) -> dict:
        stats = {
            "n": self.n,
            "ms": [round(x, 1) for x in self.ms],
            "max_ms": [round(x, 1) for x in self.max_ms],
            "collected": self.collected,
            "freeze_count": gc.get_freeze_count(),
        }
        self._reset()
        return stats


class PPBubbleDiag:
    """Scheduler-thread accumulator for one PP rank."""

    def __init__(
        self,
        pp_rank: int,
        event_factory=None,
        interval_s: float = 60.0,
        chunk_budget: int = 0,
        fwd_file=None,
    ):
        self.pp_rank = pp_rank
        self._fwd_file = fwd_file
        self._event_factory = event_factory
        self.interval_s = interval_s
        self.chunk_budget = chunk_budget
        self._phase = "other"
        self._label = "host.other"
        self._t = time.perf_counter_ns()
        self._logged_at = time.monotonic()
        self._pending: deque = deque()
        self._prev_end = None
        self._pass_queue = 0
        self._pass_skips = 0
        self._pass_relay0 = (0, 0, 0)
        self.stack_sampler: Optional[_StackSampler] = None
        self.gc_timer: Optional[_GCTimer] = None
        # Slow-event log (SGLANG_PP_DIAG_SLOW_MS): per-iteration label time and walls.
        self.slow_ns = 0
        self._it_ns: dict[str, int] = defaultdict(int)
        self._it_t0 = self._t
        self._it_walls: deque = deque(maxlen=64)
        self._reset()

    def _reset(self) -> None:
        self.ns: dict[str, int] = defaultdict(int)
        self.calls: dict[str, int] = defaultdict(int)
        self.passes = 0
        self.queued_passes = 0
        self.queue_sum = 0
        self.queue_max = 0
        self.skips: dict[str, int] = defaultdict(int)
        self.skip_passes = 0
        self.under_budget_queued = 0
        self.lost_tokens_skip_passes = 0
        self.admitted_tokens = 0
        self.fwd_n = 0
        self.fwd_ms = 0.0
        self.gap_n = 0
        self.gap_cpu_ms = 0.0
        self.gap_dep_ms = 0.0
        self.qgap_n = 0
        self.qgap_cpu_ms = 0.0
        self.qgap_dep_ms = 0.0
        # Full forwards split by whether a host load-back overlapped them.
        self.full_load_n = 0
        self.full_load_ms = 0.0
        self.full_load_cached = 0
        self.full_noload_n = 0
        self.full_noload_ms = 0.0
        self.full_noload_cached = 0
        n = len(QUEUE_BINS)
        self.bin_passes = [0] * n
        self.bin_vote_ms = [0.0] * n
        self.bin_vote_calls = [0] * n
        self.bin_ticket_ms = [0.0] * n

    # ---- wall-time partition ----

    def _switch(self, label: str) -> str:
        now = time.perf_counter_ns()
        self.ns[self._label] += now - self._t
        if self.slow_ns:
            self._it_ns[self._label] += now - self._t
        self._t = now
        prev = self._label
        self._label = label
        return prev

    def phase(self, name: str) -> None:
        """Mark the loop phase; host time from here on is booked to it."""
        self._phase = name
        if self._label.startswith("host."):
            self._switch("host." + name)
        if self.slow_ns and name == "ingest":
            self._end_iteration()

    def _end_iteration(self) -> None:
        """At each loop start: log the previous iteration if it ran slow_ns over the
        rolling median of recent iterations, with its time per label and its GC time."""
        self._switch(self._label)
        wall = self._t - self._it_t0
        gc_ms = self.gc_timer.take_iteration_ms() if self.gc_timer is not None else -1.0
        if len(self._it_walls) >= 16:
            median = sorted(self._it_walls)[len(self._it_walls) // 2]
            if wall - median >= self.slow_ns:
                labels = {
                    k: round(v / 1e6, 1)
                    for k, v in self._it_ns.items()
                    if v >= 2_000_000
                }
                logger.info(
                    "PP slow iter rank %d: wall_ms=%.1f median_ms=%.1f gc_ms=%.1f end=%.3f labels=%s",
                    self.pp_rank,
                    wall / 1e6,
                    median / 1e6,
                    gc_ms,
                    time.time(),
                    json.dumps(labels, separators=(",", ":")),
                )
        self._it_walls.append(wall)
        self._it_ns.clear()
        self._it_t0 = self._t

    def span(self, label: str) -> _Span:
        return _Span(self, "wait." + label)

    def span_in_phase(self, prefix: str) -> _Span:
        return _Span(self, f"wait.{prefix}.{self._phase}")

    # ---- admission passes ----

    def _relay_totals(self) -> tuple[int, int, int]:
        vote = "wait.relay." + VOTE_CALLER
        return (
            self.ns[vote],
            self.calls[vote],
            self.ns["wait.relay." + TICKET_CALLER],
        )

    def begin_pass(self, queue_len: int) -> None:
        self._pass_queue = queue_len
        self._pass_skips = 0
        self._pass_relay0 = self._relay_totals()

    def note_skip(self, reason: str) -> None:
        self.skips[reason] += 1
        self._pass_skips += 1

    def end_pass(self, admitted_tokens: int) -> None:
        q = self._pass_queue
        self.passes += 1
        b = _bin(q)
        vote_ns, vote_calls, ticket_ns = self._relay_totals()
        v0, c0, t0 = self._pass_relay0
        self.bin_passes[b] += 1
        self.bin_vote_ms[b] += (vote_ns - v0) / 1e6
        self.bin_vote_calls[b] += vote_calls - c0
        self.bin_ticket_ms[b] += (ticket_ns - t0) / 1e6
        if q == 0:
            return
        self.queued_passes += 1
        self.queue_sum += q
        self.queue_max = max(self.queue_max, q)
        self.admitted_tokens += admitted_tokens
        if self.chunk_budget and admitted_tokens < self.chunk_budget:
            self.under_budget_queued += 1
            if self._pass_skips:
                self.lost_tokens_skip_passes += self.chunk_budget - admitted_tokens
        if self._pass_skips:
            self.skip_passes += 1

    # ---- GPU forward timing ----

    def forward_event(self):
        """A timing event recorded on the current (forward) stream."""
        event = self._event_factory(enable_timing=True)
        event.record()
        return event

    def on_forward(
        self, pre, start, end, new_tokens: int, cached_tokens: int, has_load: bool
    ) -> None:
        """pre: before the forward's stream waits; start: after them; end: after
        the forward. Gaps are read only once the later event has completed."""
        if len(self._pending) >= _MAX_PENDING_FORWARDS:
            self._pending.popleft()
        self._pending.append(
            (
                self._prev_end,
                pre,
                start,
                end,
                self._pass_queue,
                new_tokens,
                cached_tokens,
                has_load,
                time.time(),
            )
        )
        self._prev_end = end
        self._drain_events()

    def _drain_events(self) -> None:
        while self._pending and self._pending[0][3].query():
            prev_end, pre, start, end, q, new, cached, has_load, wall = (
                self._pending.popleft()
            )
            fwd = start.elapsed_time(end)
            self.fwd_n += 1
            self.fwd_ms += fwd
            if self.chunk_budget and new >= _FULL_SHARE * self.chunk_budget:
                if has_load:
                    self.full_load_n += 1
                    self.full_load_ms += fwd
                    self.full_load_cached += cached
                else:
                    self.full_noload_n += 1
                    self.full_noload_ms += fwd
                    self.full_noload_cached += cached
            cpu = dep = -1.0
            if prev_end is not None:
                cpu = max(0.0, prev_end.elapsed_time(pre))
                dep = max(0.0, pre.elapsed_time(start))
                self.gap_n += 1
                self.gap_cpu_ms += cpu
                self.gap_dep_ms += dep
                if q > 0:
                    self.qgap_n += 1
                    self.qgap_cpu_ms += cpu
                    self.qgap_dep_ms += dep
            if self._fwd_file is not None:
                self._fwd_file.write(
                    f"{wall:.3f},{fwd:.3f},{cpu:.3f},{dep:.3f},{new},{cached},"
                    f"{int(has_load)},{q}\n"
                )

    # ---- interval log ----

    def snapshot(self, interval_s: float) -> dict:
        self._switch(self._label)
        r = lambda x: round(x, 2)
        stats = {
            "pp_rank": self.pp_rank,
            "interval_s": r(interval_s),
            "wall_ms": r(sum(self.ns.values()) / 1e6),
            "ms": {k: r(v / 1e6) for k, v in sorted(self.ns.items())},
            "calls": {k: v for k, v in sorted(self.calls.items())},
            "fwd_n": self.fwd_n,
            "fwd_ms": r(self.fwd_ms),
            "gap_n": self.gap_n,
            "gap_cpu_ms": r(self.gap_cpu_ms),
            "gap_dep_ms": r(self.gap_dep_ms),
            "qgap_n": self.qgap_n,
            "qgap_cpu_ms": r(self.qgap_cpu_ms),
            "qgap_dep_ms": r(self.qgap_dep_ms),
            "full_load_n": self.full_load_n,
            "full_load_ms": r(self.full_load_ms),
            "full_load_cached": self.full_load_cached,
            "full_noload_n": self.full_noload_n,
            "full_noload_ms": r(self.full_noload_ms),
            "full_noload_cached": self.full_noload_cached,
            "passes": self.passes,
            "queued_passes": self.queued_passes,
            "queue_sum": self.queue_sum,
            "queue_max": self.queue_max,
            "skips": dict(self.skips),
            "skip_passes": self.skip_passes,
            "under_budget_queued": self.under_budget_queued,
            "lost_tokens_skip_passes": self.lost_tokens_skip_passes,
            "admitted_tokens": self.admitted_tokens,
            "queue_bins": list(QUEUE_BINS),
            "bin_passes": self.bin_passes,
            "bin_vote_ms": [r(x) for x in self.bin_vote_ms],
            "bin_vote_calls": self.bin_vote_calls,
            "bin_ticket_ms": [r(x) for x in self.bin_ticket_ms],
        }
        if self.gc_timer is not None:
            stats["gc"] = self.gc_timer.snapshot()
        return stats

    def maybe_log(self, now: Optional[float] = None) -> Optional[dict]:
        now = time.monotonic() if now is None else now
        if now - self._logged_at < self.interval_s:
            return None
        self._drain_events()
        stats = self.snapshot(now - self._logged_at)
        logger.info("PP bubble diag: %s", json.dumps(stats, separators=(",", ":")))
        if self._fwd_file is not None:
            self._fwd_file.flush()
        if self.stack_sampler is not None:
            self.stack_sampler.flush()
        self._logged_at = now
        self._reset()
        return stats


_diag: Optional[PPBubbleDiag] = None


def init(pp_rank: int, event_factory, chunk_budget: int) -> Optional[PPBubbleDiag]:
    """Create this process's diag when SGLANG_PP_BUBBLE_DIAG is on."""
    global _diag
    if envs.SGLANG_PP_BUBBLE_DIAG.get():
        fwd_file = None
        fwd_dir = envs.SGLANG_PP_BUBBLE_DIAG_FWD_DIR.get()
        if fwd_dir:
            os.makedirs(fwd_dir, exist_ok=True)
            path = os.path.join(fwd_dir, f"fwd-pp{pp_rank}-{os.getpid()}.csv")
            fwd_file = open(path, "a", buffering=1 << 20)
            fwd_file.write(FWD_CSV_HEADER)
        _diag = PPBubbleDiag(
            pp_rank,
            event_factory,
            envs.SGLANG_PP_BUBBLE_DIAG_INTERVAL_S.get(),
            chunk_budget,
            fwd_file,
        )
        logger.info("PP bubble diag on (SGLANG_PP_BUBBLE_DIAG), rank %d", pp_rank)
        slow_ms = envs.SGLANG_PP_DIAG_SLOW_MS.get()
        if slow_ms > 0:
            _diag.slow_ns = int(slow_ms * 1e6)
            _diag.gc_timer = _GCTimer(pp_rank)
            logger.info("PP slow-event log on, rank %d, %.0f ms", pp_rank, slow_ms)
        hz = envs.SGLANG_PP_STACK_SAMPLE_HZ.get()
        ranks = envs.SGLANG_PP_STACK_SAMPLE_RANKS.get().split(",")
        if fwd_dir and hz > 0 and str(pp_rank) in ranks:
            path = os.path.join(fwd_dir, f"stacks-pp{pp_rank}-{os.getpid()}.txt")
            _diag.stack_sampler = _StackSampler(_diag, threading.get_ident(), hz, path)
            if _diag.gc_timer is None:
                _diag.gc_timer = _GCTimer(pp_rank)
            logger.info(
                "PP stack sampler on, rank %d, %.1f Hz -> %s", pp_rank, hz, path
            )
    return _diag


def get() -> Optional[PPBubbleDiag]:
    return _diag


def wait(label: str):
    """Context for a blocking wait; a no-op when the diag is off."""
    return _NULL if _diag is None else _diag.span(label)


def sub(name: str):
    """Context for a named piece of host work, booked as host.sub.<name>."""
    return _NULL if _diag is None else _Span(_diag, "host.sub." + name)


def wait_in_phase(prefix: str):
    return _NULL if _diag is None else _diag.span_in_phase(prefix)


def relay_wait():
    """Context for a HiCache PP relay receive, labelled by the function that
    called _all_reduce (frames: relay_wait <- _pp_sync <- _all_reduce <- caller)."""
    if _diag is None:
        return _NULL
    frame = sys._getframe(3)
    if frame.f_code.co_name == "agree_on_all_ranks":
        frame = frame.f_back
    return _diag.span("relay." + frame.f_code.co_name)
