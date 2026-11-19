"""SGLANG_DSV41_TAIL_STAGE on an encoder-only prefill: each late-fed request's window rows
of the late layers' input, written at the last KV source layer into a slot the PD transfer
sends from (one item per slot) to the tail stage."""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from typing import Dict, Optional

import torch


class LateTailCapture:
    def __init__(
        self,
        *,
        slots: int,
        window: int,
        num_req_rows: int,
        hc_mult: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        mem_pool=None,
    ) -> None:
        self.window = window
        # [slots * window + 1, hc_mult, hidden]: the last row takes every row outside
        # a capture window. Mooncake reads it: under a custom memory pool (MNNVL) it
        # must come from that pool, as the KV pools' buffers do.
        with torch.cuda.use_mem_pool(mem_pool) if mem_pool else nullcontext():
            self.pool = torch.zeros(
                slots * window + 1, hc_mult, hidden_size, dtype=dtype, device=device
            )
        # Per request-pool row: its capture slot (-1: none) and window start.
        self.slot_of_row = torch.full(
            (num_req_rows,), -1, dtype=torch.int64, device=device
        )
        self.start_of_row = torch.zeros(num_req_rows, dtype=torch.int64, device=device)
        self.free_slots = deque(range(slots))
        # Host mirror of slot_of_row's bound entries: a row the pool reused for
        # another request keeps that request's binding.
        self._bound: Dict[int, int] = {}

    def transfer_buf_infos(self):
        rows = self.pool[: self.pool.shape[0] - 1]
        return [rows.data_ptr()], [rows.nbytes], [self.pool[: self.window].nbytes]

    def take(self) -> Optional[int]:
        """A free slot, or None while every slot holds a transfer."""
        return self.free_slots.popleft() if self.free_slots else None

    def bind(self, *, req_pool_idx: int, slot: int, window_start: int) -> None:
        if self._bound.get(req_pool_idx) == slot:
            return
        self._bound[req_pool_idx] = slot
        self.slot_of_row[req_pool_idx] = slot
        self.start_of_row[req_pool_idx] = window_start

    def release(self, *, req_pool_idx: Optional[int], slot: int) -> None:
        """Return the slot once its transfer is over; clear the row if still its."""
        if req_pool_idx is not None and self._bound.get(req_pool_idx) == slot:
            del self._bound[req_pool_idx]
            self.slot_of_row[req_pool_idx] = -1
        self.free_slots.append(slot)

    def capture(
        self, *, residual: torch.Tensor, row_req: torch.Tensor, positions: torch.Tensor
    ) -> None:
        """Device ops only: copy each row inside its request's window to its slot."""
        slot = self.slot_of_row[row_req]
        offset = positions.to(torch.int64) - self.start_of_row[row_req]
        inside = (slot >= 0) & (offset >= 0) & (offset < self.window)
        trash = self.pool.shape[0] - 1
        dst = torch.where(inside, slot * self.window + offset, trash)
        self.pool.index_copy_(0, dst, residual.to(self.pool.dtype))
