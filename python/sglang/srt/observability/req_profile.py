"""Request-level profiling export, enabled by SGLANG_REQ_PROFILE_DIR.

Each scheduler process appends to <dir>/reqprof-<host>-<pid>.jsonl:
  {"k": "header"}  once: host, pid, boot id, ranks, disaggregation mode
  {"k": "anchor"}  every few seconds: wall ns, perf_counter, monotonic, written/dropped counts
  {"k": "req"}     once per request finished on this scheduler (attn TP rank 0 only)
  {"k": "batch"}   once per processed batch (attn TP rank 0 only)
Times are raw time.perf_counter() / time.monotonic() seconds; anchors map them to wall clock
offline, so the scheduler thread only builds a small dict. Encoding and I/O run on a daemon
thread; when the backlog exceeds the cap, records are dropped and counted, never blocked on.
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import threading
import time
from typing import Any, Dict, List, Optional

from sglang.srt.environ import envs

_DIR = envs.SGLANG_REQ_PROFILE_DIR.get() or ""
ENABLED = bool(_DIR)
_BATCH = ENABLED and envs.SGLANG_REQ_PROFILE_BATCH.get()
_MAX_PENDING = 200_000
_FLUSH_S = 0.5
_ANCHOR_S = 5.0
_BATCH_RIDS_MAX = 512

# Timestamps (perf_counter seconds) and durations exported from SchedulerReqTimeStats.
_TS_FIELDS = (
    "scheduler_recv_time",
    "prefill_bootstrap_queue_entry_time",
    "bootstrap_done_time",
    "wait_queue_entry_time",
    "forward_entry_time",
    "last_forward_entry_time",
    "last_chunked_prefill_finish_time",
    "prefill_finished_time",
    "last_prefill_finished_time",
    "prefill_transfer_queue_entry_time",
    "prefill_kv_transfer_finish_time",
    "decode_prealloc_queue_entry_time",
    "decode_transfer_queue_entry_time",
    "decode_prebuilt_finish_time",
    "first_decode_finish_time",
    "last_decode_finish_time",
    "completion_time",
)
_VAL_FIELDS = (
    "queue_duration_s",
    "decode_ct",
    "transfer_total_mb",
    "kv_transfer_latency_ms",
    "kv_bootstrap_ms",
    "kv_alloc_ms",
    "storage_prefetch_wait_s",
    "storage_staging_wait_s",
)
_RANK_FIELDS = (
    "tp_rank", "tp_size", "pp_rank", "pp_size", "dp_rank", "dp_size", "attn_tp_rank",
    "attn_tp_size", "attn_dp_rank", "attn_dp_size", "attn_cp_rank", "moe_ep_rank", "node_rank",
)


class _Writer:
    def __init__(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, f"reqprof-{socket.gethostname()}-{os.getpid()}.jsonl")
        self.pending: List[Dict[str, Any]] = []
        self.written = 0
        self.dropped = 0
        self.lock = threading.Lock()
        self.fh = open(self.path, "a", buffering=1 << 20)
        self.header_done = False
        self.last_anchor = 0.0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="reqprof-writer", daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def put(self, rec: Dict[str, Any]) -> None:
        # list.append is atomic under the GIL; the lock only guards the swap in _drain.
        if len(self.pending) >= _MAX_PENDING:
            self.dropped += 1
            return
        self.pending.append(rec)

    def _drain(self) -> None:
        with self.lock:
            batch, self.pending = self.pending, []
        now = time.perf_counter()
        lines = [json.dumps(r, separators=(",", ":")) for r in batch]
        if not self.header_done:
            lines.insert(0, json.dumps(_header(), separators=(",", ":")))
            self.header_done = True
        self.written += len(batch)
        if now - self.last_anchor >= _ANCHOR_S or self.stop.is_set():
            self.last_anchor = now
            lines.append(json.dumps(_anchor(self.written, self.dropped), separators=(",", ":")))
        if lines:
            self.fh.write("\n".join(lines) + "\n")
            self.fh.flush()

    def _run(self) -> None:
        while not self.stop.wait(_FLUSH_S):
            try:
                self._drain()
            except Exception as e:  # never take the scheduler down for profiling
                self.dropped += 1
                print(f"[reqprof] writer error: {e!r}", flush=True)

    def close(self) -> None:
        if self.stop.is_set():
            return
        self.stop.set()
        self.thread.join(timeout=5)
        try:
            self._drain()
            self.fh.close()
        except Exception:
            pass


def _anchor(written: int, dropped: int) -> Dict[str, Any]:
    return {"k": "anchor", "wall_ns": time.time_ns(), "pc": time.perf_counter(),
            "mono": time.monotonic(), "written": written, "dropped": dropped}


def _header() -> Dict[str, Any]:
    from sglang.srt.runtime_context import get_parallel

    par = get_parallel()
    ranks = {}
    for name in _RANK_FIELDS:
        try:
            ranks[name] = getattr(par, name)
        except Exception:  # not every width is published in every launch mode
            ranks[name] = None
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            boot_id = f.read().strip()
    except OSError:
        boot_id = None
    return {"k": "header", "v": 1, "host": socket.gethostname(), "pid": os.getpid(),
            "boot_id": boot_id, "ranks": ranks, "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "wall_ns": time.time_ns(), "pc": time.perf_counter(), "mono": time.monotonic()}


_writer: Optional[_Writer] = None
_writer_lock = threading.Lock()
_io_rank: Optional[bool] = None


def _get() -> Optional[_Writer]:
    global _writer, _io_rank
    if _io_rank is None:
        from sglang.srt.runtime_context import get_parallel

        _io_rank = get_parallel().attn_tp_rank == 0
    if not _io_rank:
        return None
    if _writer is None:
        with _writer_lock:
            if _writer is None:
                _writer = _Writer(_DIR)
    return _writer


def on_request_finished(req, mode: str, cached_details: Optional[dict] = None) -> None:
    """Called once per request when this scheduler streams its finished output."""
    w = _get()
    if w is None:
        return
    ts = req.time_stats
    d = vars(ts)
    t = {k: d[k] for k in _TS_FIELDS if d.get(k)}
    v = {k: d[k] for k in _VAL_FIELDS if d.get(k) is not None and d.get(k) != 0}
    fr = req.finished_reason
    rec = {
        "k": "req",
        "pc": time.perf_counter(),
        "rid": req.rid,
        "room": req.bootstrap_room,
        "mode": mode,
        "isl": len(req.origin_input_ids),
        "cached": req.cached_tokens,
        "out": len(req.output_ids),
        "attempts": req.prefill_attempt_count,
        "retract": req.retraction_count,
        "spec_verify_ct": req.spec_verify_ct,
        "finish": type(fr).__name__ if fr is not None else None,
        "routed_dp": req.disagg_prefill_dp_rank,
        # DP rank pinned by the caller (e.g. Dynamo's KV router); None means the DP controller's balancer chose
        "ext_dp": req.routed_dp_rank,
        # first output token ids: tells an empty-text first token (special / split multi-byte) from a visible one
        "first_ids": req.output_ids[:2],
        "t": t,
        "v": v,
    }
    kv = req.reqprof_kv
    if kv is not None:
        t["kv_first_send_time"], t["kv_last_send_time"] = kv[0], kv[1]
        v["kv_send_tokens"], v["kv_send_calls"] = kv[2], kv[3]
    if cached_details:
        rec["cached_details"] = cached_details
    w.put(rec)


def on_kv_send(req, tokens: int) -> None:
    """Called per prefill KV send call: first/last call time, tokens and calls per request."""
    now = time.perf_counter()
    kv = req.reqprof_kv
    if kv is None:
        req.reqprof_kv = [now, now, tokens, 1]
    else:
        kv[1] = now
        kv[2] += tokens
        kv[3] += 1


def on_batch_result(batch) -> None:
    """Called once per processed batch; decode batches omit rids (join by rank + time)."""
    if not _BATCH:
        return
    w = _get()
    if w is None:
        return
    mode = batch.forward_mode
    reqs = batch.reqs
    rec = {
        "k": "batch",
        "pc": time.perf_counter(),
        "launch_mono": getattr(batch, "launch_ts", None),
        "iter": getattr(batch, "forward_iter", None),
        "mode": mode.name if hasattr(mode, "name") else str(mode),
        "bs": len(reqs),
        "ext_tokens": batch.extend_num_tokens or 0,
    }
    if mode.is_extend() and len(reqs) <= _BATCH_RIDS_MAX:
        rec["rids"] = [r.rid for r in reqs]
        rec["ext_lens"] = list(batch.extend_lens) if batch.extend_lens is not None else None
    w.put(rec)
