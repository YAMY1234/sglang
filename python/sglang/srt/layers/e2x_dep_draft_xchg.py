"""K75P draft token select in one launch (SGLANG_Q35MF_DEP_DX_XCHG).

Each rank computes every gathered row's logits on its vocab shard (K75P); instead of the pair
all-gather and select, each rank sends its candidate to the row's owner, which picks the token
with K75P's keys and ties and, in draft decode steps, also writes the topk1 finish outputs.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.kernels.ops.speculative.e2x_dep_xchg import (
    MAX_BLOCKS,
    SLOT_MARKER,
    dep_slots_shape,
    e2x_dep_xchg,
)

logger = logging.getLogger(__name__)

# (topk_p [rows, 1], topk_index [rows, 1], whether mrope_positions got its + 1)
StepFinish = Tuple[torch.Tensor, torch.Tensor, bool]


class DepDraftXchg:
    def __init__(
        self,
        *,
        device_group: dist.ProcessGroup,
        rank: int,
        world: int,
        max_rows: int,
        proof: bool,
    ):
        import torch.distributed._symmetric_memory as symm_mem

        self.rank = rank
        self.world = world
        max_gathered = world * max_rows
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        with torch.inference_mode(False), torch.no_grad():
            slots = symm_mem.empty(
                dep_slots_shape(max_gathered, world), dtype=torch.int32, device=device
            )
        # Every word holds the marker before any peer can see the buffer (rendezvous is collective).
        slots.fill_(SLOT_MARKER)
        torch.cuda.synchronize()
        hdl = symm_mem.rendezvous(slots, group=device_group)
        if hdl.rank != rank or int(hdl.buffer_ptrs[rank]) != slots.data_ptr():
            raise RuntimeError("DEP-DX exchange: unexpected symmetric memory layout")
        self.slots = slots
        self.peer_slots = torch.tensor(
            [int(p) for p in hdl.buffer_ptrs], dtype=torch.int64, device=device
        )
        self.epoch = torch.zeros(max_gathered, dtype=torch.int32, device=device)
        self.counter = torch.zeros(max_gathered, dtype=torch.int32, device=device)
        self.part_key = torch.zeros(
            (max_gathered, MAX_BLOCKS), dtype=torch.int32, device=device
        )
        self.part_idx = torch.zeros_like(self.part_key)
        self.no_ints = torch.empty(0, dtype=torch.int64, device=device)
        self.no_floats = torch.empty(0, dtype=torch.float32, device=device)
        # Finish targets of the draft decode step being run, set by the EAGLE worker.
        self._step: Optional[tuple] = None
        self._finish: Optional[StepFinish] = None
        self.proof = DepDraftXchgProof() if proof else None
        logger.info(
            "DEP-DX draft token exchange active (tp=%d, rows/rank <= %d).",
            world,
            max_rows,
        )

    def begin_step(
        self,
        *,
        positions: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_column: int,
        mrope_positions: Optional[torch.Tensor],
    ) -> None:
        self._step = (positions, draft_tokens, draft_column, mrope_positions)
        self._finish = None

    def end_step(self) -> Optional[StepFinish]:
        """The finish outputs the step's select wrote, or None when the stock finish must run."""
        finish = self._finish
        self._step = None
        self._finish = None
        return finish

    def select(
        self,
        *,
        logits: torch.Tensor,
        num_local_rows: int,
        torch_nan: bool,
        stock_select: Callable[..., torch.Tensor],
    ) -> torch.Tensor:
        rows = num_local_rows
        tokens = torch.empty((rows, 1), dtype=torch.int64, device=logits.device)
        # Draft extend picks with torch.argmax's NaN rule and has no finish to fuse.
        step = None if torch_nan else self._step
        if step is None:
            topk_p = self.no_floats
            positions = draft_tokens = mrope = self.no_ints
            column = 0
        else:
            positions, draft_tokens, column, mrope = step
            positions = positions[:rows]
            draft_tokens = draft_tokens[:rows]
            # The stock finish adds 1 to the whole mrope tensor; fuse only when that is these rows.
            if mrope is None or mrope.shape != (3, rows) or mrope.stride(1) != 1:
                mrope = self.no_ints
            topk_p = torch.empty((rows, 1), dtype=torch.float32, device=logits.device)
        before = None
        if self.proof is not None and step is not None:
            before = (positions.clone(), mrope.clone())
        e2x_dep_xchg(
            logits=logits,
            part_key=self.part_key,
            part_idx=self.part_idx,
            counter=self.counter,
            slots=self.slots,
            peer_slots=self.peer_slots,
            epoch=self.epoch,
            tokens=tokens,
            topk_p=topk_p,
            draft_tokens=draft_tokens,
            positions=positions,
            mrope_positions=mrope,
            vocab_start=self.rank * logits.shape[1],
            torch_nan=torch_nan,
            rank=self.rank,
            rows=rows,
            draft_column=column,
        )
        if step is not None:
            self._finish = (topk_p, tokens, mrope.numel() > 0)
        if self.proof is not None:
            stock = stock_select(
                logits=logits, num_local_rows=rows, torch_nan=torch_nan
            )
            self.proof.check(
                extend=torch_nan,
                tokens=tokens,
                stock_tokens=stock,
                finish=None if step is None else (topk_p, positions, mrope),
                draft_column=None if step is None else draft_tokens[:, column],
                before=before,
            )
        return tokens


class DepDraftXchgProof:
    """Counts rows whose fused outputs differ from the stock K75P select + topk1 finish on the same
    logits, eager or in CUDA graphs; counters are mirrored to pinned memory as K75PProof does."""

    PERIOD_S = 30.0
    KINDS = ("decode", "extend")

    def __init__(self):
        # [kind][calls, rows, mismatched rows]
        self.counters = torch.zeros(
            (2, 3), dtype=torch.int64, device=torch.cuda.current_device()
        )
        self.host_counters = torch.zeros(
            (2, 3), dtype=torch.int64, device="cpu", pin_memory=True
        )
        logger.info("DEP-DX proof active")
        threading.Thread(target=self._report, daemon=True).start()

    def check(
        self,
        *,
        extend: bool,
        tokens: torch.Tensor,
        stock_tokens: torch.Tensor,
        finish: Optional[tuple],
        draft_column: Optional[torch.Tensor],
        before: Optional[tuple],
    ) -> None:
        bad = (tokens != stock_tokens).view(-1)
        if finish is not None:
            topk_p, positions, mrope = finish
            positions_before, mrope_before = before
            stock = stock_tokens.view(-1)
            bad = bad | (topk_p.view(-1) != 1.0)
            bad = bad | (positions != positions_before + 1)
            bad = bad | (draft_column != stock)
            if mrope.numel() > 0:
                bad = bad | (mrope != mrope_before + 1).any(dim=0)
        counters = self.counters[int(extend)]
        counters[0].add_(1)
        counters[1].add_(bad.numel())
        counters[2].add_(torch.count_nonzero(bad))
        self.host_counters.copy_(self.counters, non_blocking=True)

    def _report(self) -> None:
        while True:
            time.sleep(self.PERIOD_S)
            for kind, (calls, rows, mismatched) in zip(
                self.KINDS, self.host_counters.tolist()
            ):
                logger.info(
                    "DEP-DX proof: kind=%s calls=%d rows=%d mismatched=%d",
                    kind,
                    calls,
                    rows,
                    mismatched,
                )
