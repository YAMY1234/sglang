"""Host-sync audit for the PP scheduler (SGLANG_PP_SYNC_AUDIT with SGLANG_PP_BUBBLE_DIAG, default off).

Counts every synchronizing CUDA call by call site, thread and bubble-diag label:
- implicit syncs that PyTorch's sync debug mode reports (``.item()``, ``.cpu()``, ``nonzero``, boolean masks, ...),
  counted only;
- ``Event.synchronize``, ``Stream.synchronize``, ``torch.cuda.synchronize`` and the scalar / host reads of CUDA
  tensors (``item``, ``tolist``, ``cpu``, ``numpy``, ``__bool__``, ``__int__``, ``__float__``, ``__index__``), counted
  and timed on the host.
One ``PP sync audit: {json}`` line per bubble-diag interval, with that interval's forward count. Diagnostic only: the
wrappers and the per-sync stack walk cost host time.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import warnings
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_SYNC_WARNING = "synchronizing CUDA operation"
_TENSOR_METHODS = ("item", "tolist", "cpu", "numpy", "__bool__", "__int__", "__float__", "__index__")
_SKIP_FILES = ("pp_sync_audit.py", "/warnings.py", "/torch/", "/contextlib.py")


def call_site(frame, depth: int = 3) -> str:
    """The innermost `depth` frames outside this module, warnings and torch, as `file:line fn` joined by ' <- '."""
    out: List[str] = []
    while frame is not None and len(out) < depth:
        path = frame.f_code.co_filename
        if not any(s in path for s in _SKIP_FILES):
            name = path.split("/sglang/", 1)[-1] if "/sglang/" in path else os.path.basename(path)
            out.append(f"{name}:{frame.f_lineno} {frame.f_code.co_name}")
        frame = frame.f_back
    return " <- ".join(out)


class SyncAudit:
    def __init__(self, label_fn: Callable[[], str], top: int = 40):
        self._label_fn = label_fn
        self._top = top
        self._local = threading.local()
        self._lock = threading.Lock()
        self._main = threading.get_ident()
        self.counts: Dict[str, List[float]] = defaultdict(lambda: [0, 0.0])

    def record(self, kind: str, frame, ms: Optional[float]) -> None:
        if getattr(self._local, "inside", False):
            return
        if threading.get_ident() == self._main:
            where = self._label_fn()
        else:
            where = "thread:" + threading.current_thread().name.split("-")[0]
        key = f"{kind}|{where}|{call_site(frame)}"
        with self._lock:
            c = self.counts[key]
            c[0] += 1
            if ms is not None:
                c[1] += ms

    def timed(self, kind: str, fn, only_cuda: bool):
        audit = self

        def wrapper(*args, **kwargs):
            if only_cuda and not (args and isinstance(args[0], torch.Tensor) and args[0].is_cuda):
                return fn(*args, **kwargs)
            if getattr(audit._local, "inside", False):
                return fn(*args, **kwargs)
            audit._local.inside = True
            t0 = time.perf_counter_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                ms = (time.perf_counter_ns() - t0) / 1e6
                audit._local.inside = False
                audit.record(kind, sys._getframe(1), ms)

        return wrapper

    def on_warning(self, message, category, filename, lineno, file=None, line=None):
        if _SYNC_WARNING in str(message):
            self.record("implicit", sys._getframe(1), None)
            return
        self._showwarning(message, category, filename, lineno, file, line)

    def install(self) -> None:
        for name in _TENSOR_METHODS:
            setattr(torch.Tensor, name, self.timed("tensor." + name, getattr(torch.Tensor, name), True))
        torch.cuda.Event.synchronize = self.timed("event.synchronize", torch.cuda.Event.synchronize, False)
        torch.cuda.Stream.synchronize = self.timed("stream.synchronize", torch.cuda.Stream.synchronize, False)
        torch.cuda.synchronize = self.timed("cuda.synchronize", torch.cuda.synchronize, False)
        self._showwarning = warnings.showwarning
        warnings.showwarning = self.on_warning
        warnings.filterwarnings("always", message=".*" + _SYNC_WARNING)
        torch.cuda.set_sync_debug_mode("warn")

    def snapshot(self, fwd_n: int) -> dict:
        with self._lock:
            counts, self.counts = self.counts, defaultdict(lambda: [0, 0.0])
        top = sorted(counts.items(), key=lambda kv: (-kv[1][1], -kv[1][0]))[: self._top]
        return {
            "fwd_n": fwd_n,
            "sites": {k: [int(n), round(ms, 2)] for k, (n, ms) in top},
            "n_sites": len(counts),
        }


_audit: Optional[SyncAudit] = None


def init(diag) -> Optional[SyncAudit]:
    """Install the audit in this scheduler process when SGLANG_PP_SYNC_AUDIT and the bubble diag are on."""
    global _audit
    if not envs.SGLANG_PP_SYNC_AUDIT.get() or diag is None or not torch.cuda.is_available():
        return None
    _audit = SyncAudit(lambda: diag._label)
    _audit.install()
    logger.info("PP sync audit on (SGLANG_PP_SYNC_AUDIT), rank %d", diag.pp_rank)
    return _audit


def log(stats: dict) -> None:
    """After a bubble-diag interval line: this interval's syncs, with its forward count."""
    if _audit is None:
        return
    logger.info(
        "PP sync audit: %s",
        json.dumps(_audit.snapshot(stats.get("fwd_n", 0)), separators=(",", ":")),
    )
