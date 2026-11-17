"""p-micro rig counters (SGLANG_PMICRO_HOSTSTATS, default off).

Host-side ints and time.perf_counter() samples only: no tensor reads beyond CPU
host indices, no device sync. The scheduler flushes them as one PMICRO_HOSTSTAT
line per launch and one PMICRO_HOSTOPS line per OPS_INTERVAL_S.
"""

import time

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
    line = "PMICRO_HOSTOPS window_s=%.1f | %s" % (
        window,
        " | ".join(op.fmt(name) for name, op in OPS.items()),
    )
    for op in OPS.values():
        op.reset()
    return line
