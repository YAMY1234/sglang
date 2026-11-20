"""p-micro rig counters (SGLANG_PMICRO_HOSTSTATS, default off).

Host-side ints and time.perf_counter() samples only: no tensor reads beyond CPU
host indices, no device sync. The scheduler flushes them as one PMICRO_HOSTSTAT
line per launch and one PMICRO_HOSTOPS line per OPS_INTERVAL_S.

After the per-op fields comes the counted L3 write-ack drain:

  ... | ackdrain n= blocked= block_ms= max_ms= | ...

ackdrain: q.get() calls of the synchronized ack_backup_queue drain, which takes
  exactly the count relayed from PP0 and blocks until this rank has that many
  acks (all ranks must release the same acks). n = gets, blocked = gets that
  found the queue empty, block_ms = their summed host wall time (all gets),
  max_ms = the longest one. Near zero on PP0; on later stages it is the time
  the pipeline waits on this stage's own L3 puts.

The PMICRO_HOSTOPS line ends with the eviction D2H copies and the L3 puts:

  ... | d2h n= ms_sum= p50_ms= p90_ms= gbps= | put n= ms_sum= p50_ms= p90_ms=
  gbps= win_gbps= busy_frac= | overlap_frac= ovl_n=

d2h: write-back D2H copies (HiCacheController.start_writing) whose finish event
  had completed when polled (Event.query(), once per launch and per line).
  ms = GPU time between the copy's start and finish events on the D2H stream;
  gbps = sum bytes / sum GPU seconds (effective copy bandwidth).
put: L3 put calls of the storage backup thread (page_set_func per
  STORAGE_BATCH_SIZE pages; hybrid sidecar batch_set_v2 per group), host wall
  time. Bytes = slots x host-pool size_per_token, from index counts only.
  gbps = sum bytes / sum put seconds; win_gbps = sum bytes / window_s;
  busy_frac = share of the window with a put in flight (puts clipped to it).
overlap_frac: share of D2H copy time that overlaps an in-flight put (this or the
  previous window's puts, plus a put still running at flush). A copy's host-clock
  span is anchored with a timing event recorded on an idle stream once per
  window (GPU time - anchor GPU time + anchor host time), so it is accurate to
  the anchor's launch latency (microseconds). ovl_n = copies with a span; copies
  polled before the anchor completed have none.

Then host free space against pending L3 writes:

  ... | l3pend mean= max= at_evict_p50= mamba_max= | hostfree_evict n=
  need_host_evict_frac= free_p10= free_p50= | hostwait n= ms_sum= l3_attrib_ms=
  other_ms=

l3pend: host KV tokens held by an L3 write not yet drained (write-behind ops of
  SGLANG_HICACHE_L3_WRITE_ON_HOST_EVICT plus StorageWriteBack writes). Each such
  write holds a host lock until the scheduler drains its ack, and host eviction
  skips locked nodes. mean/max over one sample per scheduler step (after the
  drain and the write-behind walk); at_evict_p50 over the samples taken at each
  write_back device eviction. mamba_max: Mamba host-leaf write-behind ops.
hostfree_evict: per host allocation of a write_back device-eviction backup (one
  per chain node with a transfer), the anchor host pool's available_size()
  before it, and whether a host eviction (UnifiedRadixCache.evict_host, any
  component) ran inside it. The Mamba/SWA slot pre-eviction of an internal-state
  backup runs before this sample and is not counted.
hostwait: time blocked because host space was not available: the batched
  eviction's retry flush after a failed host allocation (nested in evict_wait),
  Mamba tombstone L3 write waits, StorageWriteBack ack waits, and shared host
  layout compaction waiting for I/O leases and transfer events. l3_attrib_ms:
  waits that wait on an L3 write by construction, or (retry flush, compaction)
  that began while l3pend > 0 / an I/O lease was held; other_ms: the rest.
"""

import threading
import time
from collections import deque

import numpy as np

from sglang.srt.environ import envs

ON = envs.SGLANG_PMICRO_HOSTSTATS.get()
OPS_INTERVAL_S = 60.0
# Percentiles use the first SAMPLE_CAP samples of an interval; n and sum are exact.
SAMPLE_CAP = 8192
# Past this many node ids the set restarts, so rebacked is a lower bound.
BACKED_CAP = 1 << 22

COUNTS = dict(d2h_subs=0, d2h_tokens=0, d2h_runs=0, rebacked=0, stat_ns=0)
# Node ids committed to host at least once; a second backup of one means its
# host copy was evicted in between (the Rust tree core evicts host values itself).
_backed: set = set()


def host_runs(host_indices, page_size: int) -> int:
    """Contiguous host-page runs in one submission's (CPU) host token indices."""
    if host_indices.is_cuda:
        return -1
    idx = host_indices.numpy()
    if idx.size == 0:
        return 0
    if idx.size % page_size == 0:
        pages = idx[::page_size] // page_size
    else:
        p = idx // page_size
        pages = p[np.r_[True, p[1:] != p[:-1]]]
    return 1 + int(np.count_nonzero(np.diff(pages) != 1))


def count_d2h(host_indices, page_size: int) -> None:
    t0 = time.perf_counter_ns()
    COUNTS["d2h_subs"] += 1
    COUNTS["d2h_tokens"] += len(host_indices)
    COUNTS["d2h_runs"] += host_runs(host_indices, page_size)
    COUNTS["stat_ns"] += time.perf_counter_ns() - t0


def note_backup(node_id, num_tokens: int) -> None:
    if node_id in _backed:
        COUNTS["rebacked"] += num_tokens
        return
    if len(_backed) >= BACKED_CAP:
        _backed.clear()
    _backed.add(node_id)


def take_launch_line(forward_iter: int, host_avail: int) -> str:
    """The PMICRO_HOSTSTAT line for the counters since the previous launch."""
    c = COUNTS
    line = (
        "PMICRO_HOSTSTAT iter=%d host_avail=%d d2h_subs=%d d2h_tokens=%d "
        "d2h_runs=%d rebacked=%d stat_us=%d"
        % (
            forward_iter,
            host_avail,
            c["d2h_subs"],
            c["d2h_tokens"],
            c["d2h_runs"],
            c["rebacked"],
            c["stat_ns"] // 1000,
        )
    )
    for k in c:
        c[k] = 0
    return line


class OpStat:
    """Wall time per operation on the calling (scheduler) thread."""

    __slots__ = ("sized", "n", "sum_s", "tokens", "nbytes", "samples")

    def __init__(self, sized: bool = False):
        self.sized = sized
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.sum_s = 0.0
        self.tokens = 0
        self.nbytes = 0
        self.samples = []

    def add(self, seconds: float, tokens: int = 0, nbytes: int = 0) -> None:
        self.n += 1
        self.sum_s += seconds
        self.tokens += tokens
        self.nbytes += nbytes
        if len(self.samples) < SAMPLE_CAP:
            self.samples.append(seconds)

    def fmt(self, name: str) -> str:
        s = sorted(self.samples)

        def q(f):
            return s[min(len(s) - 1, int(f * len(s)))] * 1e3 if s else 0.0

        text = "%s n=%d sum_ms=%.1f p50_ms=%.3f p90_ms=%.3f" % (
            name,
            self.n,
            self.sum_s * 1e3,
            q(0.5),
            q(0.9),
        )
        if self.sized:
            n = max(self.n, 1)
            text += " tok_per_op=%.0f bytes_per_op=%.0f" % (
                self.tokens / n,
                self.nbytes / n,
            )
        return text


# evict_wait: blocked on write_back D2H acks (writing_check(write_back=True)).
# backup_launch: one D2H submission (HiCacheController.start_writing).
# host_evict: UnifiedRadixCache.evict_host; host_evict_l3: its L3 part
# (StorageWriteBack.evict_host, Mamba state-with-storage victims).
# write_behind: the per-step L3 write-behind of the host tail.
OPS = {
    "evict_wait": OpStat(),
    "backup_launch": OpStat(sized=True),
    "host_evict": OpStat(),
    "host_evict_l3": OpStat(),
    "write_behind": OpStat(),
}
_ops_window_start = time.perf_counter()


def take_ops_line(now: float):
    """The PMICRO_HOSTOPS line once per OPS_INTERVAL_S, else None."""
    global _ops_window_start
    window = now - _ops_window_start
    if window < OPS_INTERVAL_S:
        return None
    _ops_window_start = now
    line = "PMICRO_HOSTOPS window_s=%.1f | %s | %s | %s" % (
        window,
        " | ".join(op.fmt(name) for name, op in OPS.items()),
        take_ackdrain_field(),
        take_l3_fields(now - window, now),
    )
    line += " | " + take_hostfree_fields()
    for op in OPS.values():
        op.reset()
    return line


# ---- Synchronized L3 write-ack drain (after the per-op fields) ----

_ackdrain = dict(n=0, blocked=0, sum_s=0.0, max_s=0.0)


def note_ack_get(seconds: float, blocked: bool) -> None:
    a = _ackdrain
    a["n"] += 1
    a["blocked"] += int(blocked)
    a["sum_s"] += seconds
    a["max_s"] = max(a["max_s"], seconds)


def take_ackdrain_field() -> str:
    a = _ackdrain
    line = "ackdrain n=%d blocked=%d block_ms=%.1f max_ms=%.3f" % (
        a["n"],
        a["blocked"],
        a["sum_s"] * 1e3,
        a["max_s"] * 1e3,
    )
    a.update(n=0, blocked=0, sum_s=0.0, max_s=0.0)
    return line


# ---- D2H copy GPU time and L3 put timing (tail of the PMICRO_HOSTOPS line) ----


def _quantiles(samples):
    s = sorted(samples)

    def q(f):
        return s[min(len(s) - 1, int(f * len(s)))] * 1e3 if s else 0.0

    return q(0.5), q(0.9)


def _union(spans):
    merged = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            if b > merged[-1][1]:
                merged[-1][1] = b
        else:
            merged.append([a, b])
    return merged


def _covered(span, merged) -> float:
    a, b = span
    return sum(max(0.0, min(b, y) - max(a, x)) for x, y in merged)


# (start_event, finish_event, nbytes) of submitted copies, oldest first.
_d2h_pending: deque = deque()
D2H = OpStat()
_d2h_spans: list = []  # host-clock (t0, t1) of polled copies
_anchor = None  # (timing event on an idle stream, host perf_counter at record)
_anchor_stream = None


def _record_anchor():
    """A timing event on an idle stream: it completes right after its record,
    so its GPU time maps to the host time taken just before the record. None
    where the device has no timing events."""
    global _anchor_stream
    from sglang.srt.mem_cache.l2_transfer import _timing_events_supported
    from sglang.srt.utils import get_device_module

    if not _timing_events_supported():
        return None
    dm = get_device_module()
    if _anchor_stream is None:
        _anchor_stream = dm.Stream()
    event = dm.Event(enable_timing=True)
    t = time.perf_counter()
    event.record(_anchor_stream)
    return event, t


def note_d2h(start_event, finish_event, nbytes: int) -> None:
    """One submitted copy; read back by poll_d2h once its finish event completed."""
    global _anchor
    if _anchor is None:
        _anchor = _record_anchor()
    if len(_d2h_pending) >= SAMPLE_CAP:
        _d2h_pending.popleft()
    _d2h_pending.append((start_event, finish_event, nbytes))


def poll_d2h() -> None:
    """Drain completed copies without synchronizing. Copies share one stream and
    complete in order, so the first incomplete one ends the drain."""
    anchor_ready = _anchor is not None and _anchor[0].query()
    while _d2h_pending and _d2h_pending[0][1].query():
        start, finish, nbytes = _d2h_pending.popleft()
        ms = start.elapsed_time(finish)
        D2H.add(ms / 1e3, nbytes=nbytes)
        if anchor_ready and len(_d2h_spans) < SAMPLE_CAP:
            t0 = _anchor[1] + _anchor[0].elapsed_time(start) / 1e3
            _d2h_spans.append((t0, t0 + ms / 1e3))


# The backup thread writes these; the scheduler thread swaps them under the lock.
_put_lock = threading.Lock()
PUT = OpStat()
_put_spans: list = []
_prev_put_spans: list = []
_put_inflight_t0 = None


def note_put_begin(t0: float) -> None:
    global _put_inflight_t0
    _put_inflight_t0 = t0


def note_put(t0: float, t1: float, nbytes: int) -> None:
    global _put_inflight_t0
    with _put_lock:
        _put_inflight_t0 = None
        PUT.add(t1 - t0, nbytes=nbytes)
        if len(_put_spans) < SAMPLE_CAP:
            _put_spans.append((t0, t1))


def take_l3_fields(window_start: float, now: float) -> str:
    """The d2h / put / overlap tail of the PMICRO_HOSTOPS line; resets the window."""
    global _put_spans, _prev_put_spans, _d2h_spans, _anchor
    poll_d2h()
    with _put_lock:
        put_n, put_s, put_bytes = PUT.n, PUT.sum_s, PUT.nbytes
        put_q = _quantiles(PUT.samples)
        spans = _put_spans
        prev = _prev_put_spans
        if _put_inflight_t0 is not None:
            spans = spans + [(_put_inflight_t0, now)]
        _prev_put_spans, _put_spans = _put_spans, []
        PUT.reset()
    window = max(now - window_start, 1e-9)
    busy = _union((max(a, window_start), min(b, now)) for a, b in spans if b > a)
    busy_s = sum(b - a for a, b in busy if b > a)
    puts = _union(prev + spans)
    d2h_total = sum(b - a for a, b in _d2h_spans)
    d2h_ovl = sum(_covered(s, puts) for s in _d2h_spans)
    d2h_q = _quantiles(D2H.samples)
    line = (
        "d2h n=%d ms_sum=%.1f p50_ms=%.3f p90_ms=%.3f gbps=%.2f | "
        "put n=%d ms_sum=%.1f p50_ms=%.3f p90_ms=%.3f gbps=%.2f win_gbps=%.2f "
        "busy_frac=%.3f | overlap_frac=%.3f ovl_n=%d"
        % (
            D2H.n,
            D2H.sum_s * 1e3,
            *d2h_q,
            D2H.nbytes / D2H.sum_s / 1e9 if D2H.sum_s > 0 else 0.0,
            put_n,
            put_s * 1e3,
            *put_q,
            put_bytes / put_s / 1e9 if put_s > 0 else 0.0,
            put_bytes / window / 1e9,
            busy_s / window,
            d2h_ovl / d2h_total if d2h_total > 0 else 0.0,
            len(_d2h_spans),
        )
    )
    D2H.reset()
    _d2h_spans = []
    # Re-anchor every window: GPU and host clocks drift apart over hours.
    if _anchor is not None:
        _anchor = _record_anchor()
    return line


# ---- Host free space vs pending L3 writes (end of the PMICRO_HOSTOPS line) ----

# Bumped by every UnifiedRadixCache.evict_host call; an eviction backup compares it.
HOST_EVICT_CALLS = [0]
_l3pend = dict(n=0, sum=0, max=0, mamba_max=0)
_l3pend_at_evict: list = []
_evict_free: list = []
_evict_n = [0, 0]  # backups, backups that evicted host first
_hostwait = dict(n=0, l3_s=0.0, other_s=0.0)


def note_host_evict_call() -> None:
    HOST_EVICT_CALLS[0] += 1


def sample_l3_pending(kv_tokens: int, mamba_ops: int = 0) -> None:
    """One per scheduler step."""
    p = _l3pend
    p["n"] += 1
    p["sum"] += kv_tokens
    p["max"] = max(p["max"], kv_tokens)
    p["mamba_max"] = max(p["mamba_max"], mamba_ops)


def note_evict_backup(host_free: int, l3pend_tokens: int, evicted_host: bool) -> None:
    _evict_n[0] += 1
    _evict_n[1] += int(evicted_host)
    if len(_evict_free) < SAMPLE_CAP:
        _evict_free.append(host_free)
        _l3pend_at_evict.append(l3pend_tokens)


def note_host_wait(seconds: float, l3_blocked: bool) -> None:
    _hostwait["n"] += 1
    _hostwait["l3_s" if l3_blocked else "other_s"] += seconds


def _pct(samples, f) -> int:
    s = sorted(samples)
    return s[min(len(s) - 1, int(f * len(s)))] if s else 0


def take_hostfree_fields() -> str:
    """The l3pend / hostfree_evict / hostwait tail; resets the window."""
    p, w = _l3pend, _hostwait
    line = (
        "l3pend mean=%d max=%d at_evict_p50=%d mamba_max=%d | "
        "hostfree_evict n=%d need_host_evict_frac=%.3f free_p10=%d free_p50=%d | "
        "hostwait n=%d ms_sum=%.1f l3_attrib_ms=%.1f other_ms=%.1f"
        % (
            p["sum"] // p["n"] if p["n"] else 0,
            p["max"],
            _pct(_l3pend_at_evict, 0.5),
            p["mamba_max"],
            _evict_n[0],
            _evict_n[1] / _evict_n[0] if _evict_n[0] else 0.0,
            _pct(_evict_free, 0.1),
            _pct(_evict_free, 0.5),
            w["n"],
            (w["l3_s"] + w["other_s"]) * 1e3,
            w["l3_s"] * 1e3,
            w["other_s"] * 1e3,
        )
    )
    p.update(n=0, sum=0, max=0, mamba_max=0)
    _l3pend_at_evict.clear()
    _evict_free.clear()
    _evict_n[:] = [0, 0]
    w.update(n=0, l3_s=0.0, other_s=0.0)
    return line
