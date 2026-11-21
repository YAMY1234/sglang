"""SGLANG_DSV41_TAIL_STAGE on decode: the tail server's entries and the draft-aux staging.

The tail server sends decode, as PP rank 1, the late layers' SWA of each late-fed
request's window and the window rows' DSpark target captures. Decode registers both
after the encoder-only prefill's entries; the captures land in a staging slot reserved
at pre-allocation and fill the draft KV of the window when the request is admitted.
"""

from __future__ import annotations

import heapq
import time
from collections import deque
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.dsv41_tail_stage.layout import tail_swa_buf_infos
from sglang.srt.disaggregation.utils import append_state_component

# Arbitrary: forward steps (about 10 s at 16 ms) before a slot an aborted transfer may
# still write is reused.
TAIL_AUX_ABORT_REST_STEPS = 600
# A slot read by the step just scheduled is free once it and the overlapped one ran.
TAIL_AUX_READ_REST_STEPS = 2


class TailAuxStaging:
    """[slots * window, captures * hidden] rows: one item per tail, its window's rows in
    position order with the captures concatenated, as the draft KV writer reads them."""

    def __init__(
        self,
        *,
        slots: int,
        window: int,
        captures: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        mem_pool=None,
    ) -> None:
        self.window = window
        # The tail server writes it over Mooncake: under a custom memory pool (MNNVL)
        # it must come from that pool, as the KV pools' buffers do.
        with torch.cuda.use_mem_pool(mem_pool) if mem_pool else nullcontext():
            self.buf = torch.zeros(
                slots * window, captures * hidden_size, dtype=dtype, device=device
            )
        self._free = deque(range(slots))
        # Slot -> the request it is reserved for, until its rows were read.
        self._holders: Dict[int, object] = {}
        # Heap of (forward step from which the slot is free, slot).
        self._retiring: List[Tuple[int, int]] = []

    def transfer_buf_infos(self):
        return (
            [self.buf.data_ptr()],
            [self.buf.nbytes],
            [self.buf[: self.window].nbytes],
        )

    def reserve(self, req, *, step: int) -> Optional[int]:
        """A slot for a late-fed transfer, or None while all are taken."""
        self._retire(step)
        if not self._free:
            return None
        slot = self._free.popleft()
        self._holders[slot] = req
        return slot

    def rows(self, slot: int) -> torch.Tensor:
        return self.buf[slot * self.window : (slot + 1) * self.window]

    def release(self, slot: int, *, step: int, rest: int) -> None:
        self._holders.pop(slot, None)
        heapq.heappush(self._retiring, (step + rest, slot))

    def finished_holders(self) -> List[object]:
        """Requests that finished (aborted) while holding a slot."""
        return [req for req in self._holders.values() if req.finished()]

    def _retire(self, step: int) -> None:
        while self._retiring and self._retiring[0][0] <= step:
            self._free.append(heapq.heappop(self._retiring)[1])


# SGLANG_DSV41_LEAN_ADMIT=alt: forward steps per arm.
LEAN_ADMIT_ALT_STEPS = 2000
# SGLANG_DSV41_LEAN_ADMIT: forward steps between scans for aborted aux holders.
LEAN_RECLAIM_EVERY = 32


def lean_admit_on(mode: str, forward_ct: int) -> bool:
    """SGLANG_DSV41_LEAN_ADMIT for this step: '1' on, 'alt' on for every other block
    of LEAN_ADMIT_ALT_STEPS forward steps, anything else off."""
    if mode == "alt":
        return (forward_ct // LEAN_ADMIT_ALT_STEPS) % 2 == 1
    return mode == "1"


class DecodeCpuStats:
    """SGLANG_DSV41_DECODE_CPU_STATS: the decode scheduler loop's wall time per phase
    and step, split by (lean admission arm, step admitted late-fed tails). The result
    phases include waiting for the previous forward; the rest is host work."""

    PHASES = (
        "ingest",
        "decode_queue",
        "result_early",
        "next_batch",
        "launch",
        "result",
        "sample",
    )
    WAITS = ("result_early", "result")

    def __init__(
        self,
        *,
        interval_s: float = 10.0,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.clock = clock
        self.interval_s = interval_s
        self._t = 0.0
        self._cur: Dict[str, float] = {}
        self._rows: Dict[Tuple[bool, bool], List[Tuple[float, ...]]] = {}
        self._last_log = clock()
        self.admitted = 0
        self.lean = False

    def start(self) -> None:
        self._cur = dict.fromkeys(self.PHASES, 0.0)
        self.admitted = 0
        self._t = self.clock()

    def mark(self, phase: str) -> None:
        now = self.clock()
        self._cur[phase] += now - self._t
        self._t = now

    def end(self) -> Optional[str]:
        """Close the step; every interval_s, the summary line (then a fresh window)."""
        key = (self.lean, self.admitted > 0)
        self._rows.setdefault(key, []).append(tuple(self._cur[p] for p in self.PHASES))
        now = self.clock()
        if now - self._last_log < self.interval_s:
            return None
        self._last_log = now
        line = self.summary()
        self._rows = {}
        return line

    def summary(self) -> str:
        def q(v: List[float], f: float) -> float:
            v = sorted(v)
            return v[min(len(v) - 1, int(len(v) * f))] * 1e3

        parts = []
        for (lean, admit), rows in sorted(self._rows.items()):
            host = [
                sum(r[i] for i, p in enumerate(self.PHASES) if p not in self.WAITS)
                for r in rows
            ]
            total = [sum(r) for r in rows]
            phases = ", ".join(
                f"{p} {q([r[i] for r in rows], 0.5):.2f}/{q([r[i] for r in rows], 0.9):.2f}"
                for i, p in enumerate(self.PHASES)
            )
            parts.append(
                f"[lean={int(lean)} admit={int(admit)}] {len(rows)} steps, host ms p50/p90 "
                f"{q(host, 0.5):.2f}/{q(host, 0.9):.2f}, total {q(total, 0.5):.2f}/{q(total, 0.9):.2f}; {phases}"
            )
        return "SGLANG_DSV41_DECODE_CPU_STATS: " + " | ".join(parts)


def append_tail_stage_entries(
    kv_args, *, pool, layer_ids, aux: Optional[TailAuxStaging]
) -> None:
    """The tail server's entries, after the encoder-only prefill's (W0) on decode as on
    the tail server: DSV41_TAIL_SWA, then DSV41_TAIL_DRAFT_AUX when decode drafts."""
    swa = tail_swa_buf_infos(pool, layer_ids)
    append_state_component(
        kv_args, StateType.DSV41_TAIL_SWA, swa.ptrs, swa.lens, swa.item_lens
    )
    kv_args.state_window_tails.append(None)
    if aux is not None:
        ptrs, lens, item_lens = aux.transfer_buf_infos()
        append_state_component(
            kv_args, StateType.DSV41_TAIL_DRAFT_AUX, ptrs, lens, item_lens
        )
        kv_args.state_window_tails.append(None)
