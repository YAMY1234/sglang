"""Host fast paths for the per-round decode prep (default off: SGLANG_Q35MF_LL16_PREP=1).

- Scheduler.record_batch_in_overlap snapshots the batch fields with one cached attrgetter.
- CudaGraphBufferRegistry.fill_from reuses a per-layout plan (slot views, source names and one
  copy descriptor) when the batch needs no padding.
- build_chain_tree and draft_extend_layout write into rotating preallocated outputs.

SGLANG_Q35MF_LL16_PREP_AB_BLOCK=N (measurement only) alternates the stock paths and the fast
paths every N scheduler passes, starting with the stock paths. When Q35MF_RT_DIR is set, the
switch times (perf_counter_ns) are appended to $Q35MF_RT_DIR/ll16-prepab-r<rank>.jsonl.
"""

from __future__ import annotations

import dataclasses
import json
import operator
import os
import time
from typing import Callable, Dict, Hashable, List, Tuple

import torch
import torch.distributed as dist

from sglang.srt.environ import envs

# A rotating output is rewritten this many rounds after it was handed out; the overlap
# scheduler holds a round's tensors for at most two rounds.
RING_DEPTH = 8


class _State:
    def __init__(self):
        self.enabled = envs.SGLANG_Q35MF_LL16_PREP.get()
        self.block = envs.SGLANG_Q35MF_LL16_PREP_AB_BLOCK.get()
        self.fast = self.enabled and self.block <= 0
        self.passes = 0
        self.switches: List[List[int]] = []


_S = _State()


def on() -> bool:
    return _S.fast


def tick() -> None:
    """Called once per scheduler pass; only the A/B schedule needs it."""
    if not _S.enabled or _S.block <= 0:
        return
    p = _S.passes
    _S.passes += 1
    _S.fast = (p // _S.block) % 2 == 1
    if p % _S.block == 0:
        _S.switches.append([time.perf_counter_ns(), int(_S.fast)])
        if len(_S.switches) >= 8:
            _dump_switches()


def _dump_switches() -> None:
    new, _S.switches = _S.switches, []
    out = os.environ.get("Q35MF_RT_DIR")
    if not out:
        return
    try:
        with open(os.path.join(out, f"ll16-prepab-r{dist.get_rank()}.jsonl"), "a") as f:
            f.writelines(json.dumps(dict(t_ns=t, fast=m, block=_S.block)) + "\n" for t, m in new)
    except OSError:
        pass


_FIELD_GETTERS: Dict[type, Tuple[Tuple[str, ...], Callable]] = {}


def snapshot_fields(obj) -> list:
    """[getattr(obj, f.name, None) for f in dataclasses.fields(obj)] with the names cached per class."""
    cls = type(obj)
    entry = _FIELD_GETTERS.get(cls)
    if entry is None:
        names = tuple(f.name for f in dataclasses.fields(cls))
        entry = _FIELD_GETTERS[cls] = (names, operator.attrgetter(*names))
    names, getter = entry
    try:
        values = getter(obj)
    except AttributeError:
        return [getattr(obj, name, None) for name in names]
    return list(values) if len(names) > 1 else [values]


class OutputRing:
    """RING_DEPTH sets of preallocated tensors per key, handed out in turn."""

    def __init__(self):
        self._sets: Dict[Hashable, list] = {}
        self._turn: Dict[Hashable, int] = {}

    def next(self, key: Hashable, make: Callable[[], tuple]) -> tuple:
        sets = self._sets.get(key)
        if sets is None:
            sets = self._sets[key] = [make() for _ in range(RING_DEPTH)]
            self._turn[key] = 0
        turn = self._turn[key]
        self._turn[key] = (turn + 1) % RING_DEPTH
        return sets[turn]


def empty_like_spec(shapes_dtypes, device) -> tuple:
    return tuple(
        None if spec is None else torch.empty(spec[0], dtype=spec[1], device=device)
        for spec in shapes_dtypes
    )
