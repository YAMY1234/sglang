"""Host-side timeline of every scheduler pass (observation only, no output change).

SGLANG_Q35MF_HOST_TRACE=<dir> turns it on. Each rank appends (mark, perf_counter_ns,
thread_time_ns) at fixed points of the overlap loop and the spec-v2 worker, records CUDA
events around draft / verify / draft extend, and logs GC pauses and the main thread's
context switches. The buffer goes to <dir>/ht-<host>-<pid>.jsonl only while the scheduler
is idle, so recording never adds file I/O to a busy pass.
"""

from __future__ import annotations

import gc
import json
import os
import resource
import socket
import sys
import threading
import time
from array import array

ENABLED = bool(os.environ.get("SGLANG_Q35MF_HOST_TRACE", ""))

(
    LOOP,
    RECV0,
    RECV1,
    HC0,
    HC1,
    NEXT1,
    RUN0,
    RUN1,
    PROC0,
    SYNC0,
    SYNC1,
    PROC1,
    DRAFT0,
    DRAFTG,
    DRAFT1,
    VERIFY0,
    VERIFY1,
    EXT0,
    EXT1,
    PASS_END,
) = range(1, 21)
GC0, GC1, RUSAGE = 30, 31, 32
GPU_MARKS = (DRAFT0, DRAFTG, DRAFT1, VERIFY0, VERIFY1, EXT1)

_buf = array("q")
_pass_start = 0
_events: list = []
_free_events: list = []
_last_flush = 0.0
_path = None
_main_ident = None


def apply_switch_interval() -> None:
    # SGLANG_Q35MF_SWITCH_INTERVAL_US: how long another thread may keep the GIL
    # while the scheduler thread waits for it (CPython default 5000 us).
    us = os.environ.get("SGLANG_Q35MF_SWITCH_INTERVAL_US", "")
    if us:
        sys.setswitchinterval(float(us) * 1e-6)


def mark(code: int) -> None:
    _buf.extend((code, time.perf_counter_ns(), time.thread_time_ns()))


def gpu_mark(code: int) -> None:
    import torch

    e = _free_events.pop() if _free_events else torch.cuda.Event(enable_timing=True)
    e.record()
    _events.append((code, len(_buf) // 3, e))
    mark(code)


def pass_begin() -> None:
    global _pass_start, _main_ident
    if _main_ident is None:
        _main_ident = threading.get_ident()
        gc.callbacks.append(_gc_cb)
    _pass_start = len(_buf)
    ru = resource.getrusage(resource.RUSAGE_THREAD)
    _buf.extend((RUSAGE, ru.ru_nvcsw, ru.ru_nivcsw))
    mark(LOOP)


def pass_end(keep: bool) -> None:
    if keep:
        mark(PASS_END)
    else:
        del _buf[_pass_start:]


def _gc_cb(phase, info) -> None:
    side = 0 if threading.get_ident() == _main_ident else 16
    _buf.extend(
        (GC0 if phase == "start" else GC1, time.perf_counter_ns(), info["generation"] + side)
    )


def _thread_stats() -> list:
    names = {t.native_id: t.name for t in threading.enumerate()}
    out = []
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/stat") as f:
                stat = f.read()
            comm = stat[stat.index("(") + 1 : stat.rindex(")")]
            rest = stat[stat.rindex(")") + 2 :].split()
            rec = dict(
                tid=int(tid),
                comm=comm,
                name=names.get(int(tid), ""),
                utime=int(rest[11]),
                stime=int(rest[12]),
                cpu=int(rest[36]),
            )
            try:
                with open(f"/proc/self/task/{tid}/sched") as f:
                    for line in f:
                        k, _, v = line.partition(":")
                        k = k.strip()
                        if k in (
                            "se.sum_exec_runtime",
                            "nr_voluntary_switches",
                            "nr_involuntary_switches",
                        ):
                            rec[k] = float(v)
            except OSError:
                pass
            out.append(rec)
        except (OSError, ValueError, IndexError):
            continue
    return out


def maybe_flush() -> None:
    """Call only while the scheduler is idle."""
    global _last_flush, _path
    if not ENABLED or (not _buf and not _events):
        return
    now = time.monotonic()
    if now - _last_flush < 1.0:
        return
    _last_flush = now
    if _path is None:
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
        d = os.environ["SGLANG_Q35MF_HOST_TRACE"]
        os.makedirs(d, exist_ok=True)
        _path = os.path.join(d, f"ht-r{rank:02d}-{socket.gethostname()}-{os.getpid()}.jsonl")
        with open(_path, "a") as f:
            f.write(
                json.dumps(
                    dict(
                        kind="meta",
                        rank=rank,
                        host=socket.gethostname(),
                        pid=os.getpid(),
                        switchinterval=sys.getswitchinterval(),
                        gc_threshold=gc.get_threshold(),
                        affinity=sorted(os.sched_getaffinity(0)),
                    )
                )
                + "\n"
            )
    gpu = []
    if _events:
        ref = _events[0][2]
        for code, idx, e in _events:
            e.synchronize()
            gpu.append((code, idx, int(ref.elapsed_time(e) * 1e6)))
        _free_events.extend(e for _, _, e in _events)
        _events.clear()
    rec = dict(
        kind="chunk",
        wall=time.time(),
        marks=_buf.tolist(),
        gpu=gpu,
        threads=_thread_stats(),
    )
    del _buf[:]
    with open(_path, "a") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")
