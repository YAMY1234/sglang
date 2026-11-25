"""SGLANG_DSV41_TAIL_STAGE on decode: the tail server's entries and the draft-aux staging.

The tail server sends decode, as PP rank 1, the late layers' SWA of each late-fed
request's window and the window rows' DSpark target captures. Decode registers both
after the encoder-only prefill's entries; the captures land in a staging slot reserved
at pre-allocation and fill the draft KV of the window when the request is admitted.
"""

from __future__ import annotations

import heapq
from collections import deque
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

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
