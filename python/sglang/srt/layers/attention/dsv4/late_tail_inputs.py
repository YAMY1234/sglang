"""SGLANG_DSV41_LATE_TAIL_ROWS on decode: the late-only rows a target verify appends.

Each verify step may carry `slots` pseudo-slots of `width` rows. A pseudo-slot is a
late-fed tail's rows [position, position + width): they enter the forward at the
late-layer start with the residual prefill shipped, and run the late layers only.
The buffers are persistent, so eager steps, graph capture and replay read the same
tensors; a step without late rows leaves every slot invalid.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch

# Arbitrary: tails staged per decode rank (5.24 MB each at window 128); the scheduler
# admits no more late-fed transfers than this.
LATE_TAIL_STAGING_TAILS = 64


class LateTailStepInputs(msgspec.Struct, frozen=True):
    """One rank's pseudo-slots for a verify step, built on the schedule stream.

    `active` is the DP-agreed decision: every attention-DP rank runs the late variant
    when any rank has late rows, so a rank may be active with no slots."""

    active: bool
    req_pool_indices: Optional[torch.Tensor] = None  # [n] int64
    positions: Optional[torch.Tensor] = None  # [n] int64, each slot's first row
    floors: Optional[torch.Tensor] = None  # [n] int64, the late-layer SWA floor
    staging_rows: Optional[torch.Tensor] = None  # [n] int64, first row in the pool


class VerifyLateInputs(msgspec.Struct, frozen=True):
    """The late layers' view of the verify batch: the verify rows, then the
    pseudo-slots' rows."""

    req_pool_indices: torch.Tensor  # [bs + slots]
    seq_lens: torch.Tensor  # [bs + slots]
    out_cache_loc: torch.Tensor  # [width * (bs + slots)]
    positions: torch.Tensor  # [width * (bs + slots)]
    residual_rows: torch.Tensor  # [width * slots, hc_mult, hidden]


class LateTailInputs:
    def __init__(
        self,
        *,
        slots: int,
        width: int,
        window: int,
        hc_mult: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.slots = slots
        self.width = width
        self.req_pool_indices = torch.zeros(slots, dtype=torch.int64, device=device)
        self.positions = torch.zeros(slots, dtype=torch.int64, device=device)
        self.floors = torch.zeros(slots, dtype=torch.int64, device=device)
        self.staging_rows = torch.zeros(slots, dtype=torch.int64, device=device)
        self.valid = torch.zeros(slots, dtype=torch.bool, device=device)
        # [tails * window, hc_mult, hidden]: the late layers' input rows per staged tail.
        self.staging = torch.zeros(
            LATE_TAIL_STAGING_TAILS * window,
            hc_mult,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        self.active = False
        self._row_offsets = torch.arange(width, dtype=torch.int64, device=device)

    def set_step(self, step: Optional[LateTailStepInputs]) -> None:
        """Before each target verify; None or an inactive step clears every slot."""
        self.active = step is not None and step.active
        self.valid.zero_()
        if not self.active or step.req_pool_indices is None:
            return
        n = step.req_pool_indices.shape[0]
        assert n <= self.slots, f"{n} late slots exceed the budget {self.slots}"
        if self.valid.is_cuda:
            stream = torch.cuda.current_stream(self.valid.device)
            for scheduled in (
                step.req_pool_indices,
                step.positions,
                step.floors,
                step.staging_rows,
            ):
                # Built on the schedule stream, read by this step's copies.
                scheduled.record_stream(stream)
        self.req_pool_indices[:n].copy_(step.req_pool_indices)
        self.positions[:n].copy_(step.positions)
        self.floors[:n].copy_(step.floors)
        self.staging_rows[:n].copy_(step.staging_rows)
        self.valid[:n] = True

    def per_row(self, per_slot: torch.Tensor) -> torch.Tensor:
        return per_slot.repeat_interleave(self.width)

    def row_positions(self) -> torch.Tensor:
        return (self.positions[:, None] + self._row_offsets).reshape(-1)

    def build_view(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> VerifyLateInputs:
        """Device ops only (recorded in the verify graph). Invalid slots write KV to
        the padding slot 0 and read pool row 0; their rows are dropped after the
        late layers."""
        row_positions = self.row_positions()
        rows_req = self.per_row(self.req_pool_indices)
        loc = req_to_token[rows_req, row_positions].to(out_cache_loc.dtype)
        loc = torch.where(self.per_row(self.valid), loc, torch.zeros_like(loc))
        staging_index = (self.staging_rows[:, None] + self._row_offsets).reshape(-1)
        return VerifyLateInputs(
            req_pool_indices=torch.cat(
                [req_pool_indices, self.req_pool_indices.to(req_pool_indices.dtype)]
            ),
            seq_lens=torch.cat([seq_lens, self.positions.to(seq_lens.dtype)]),
            out_cache_loc=torch.cat([out_cache_loc, loc]),
            positions=torch.cat([positions, row_positions.to(positions.dtype)]),
            residual_rows=self.staging[staging_index],
        )
