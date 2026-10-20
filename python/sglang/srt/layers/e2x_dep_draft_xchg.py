"""K75P draft token select in one launch (SGLANG_Q35MF_DEP_DX_XCHG).

Each rank computes every gathered row's logits on its vocab shard (K75P); instead of the pair
all-gather and select, each rank sends its candidate to the row's owner, which picks the token
with K75P's keys and ties and, in draft decode steps, also writes the topk1 finish outputs.
With the memo (SGLANG_Q35MF_DEP_DX_MEMO_FULL) the owner also writes the winner's row of the served
pre-fc embedding norm into the next step's fc input; the hidden half is normed on a side stream
beside the lm_head, so the next step skips the embedding, both norms and the cat.
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

# Memo rows per norm call at build time; the served norm is row-invariant across widths (checked).
_MEMO_BLOCK = 1024
_MEMO_CHECK_ROWS = 64


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
        self.device_group = device_group
        self.max_rows = max_rows
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
        self.no_rows = torch.empty(0, dtype=torch.bfloat16, device=device)
        # Memo state (enable_memo): fc_in holds the next draft step's fc input.
        self.memo: Optional[torch.Tensor] = None
        self.fc_in: Optional[torch.Tensor] = None
        self.side_stream: Optional[torch.cuda.Stream] = None
        self._produce = False
        self._embed_rows: Optional[int] = None
        self._hidden_rows: Optional[int] = None
        self._side_pending = False
        self._fc_ready: Optional[int] = None
        logger.info(
            "DEP-DX draft token exchange active (tp=%d, rows/rank <= %d).",
            world,
            max_rows,
        )

    def enable_memo(
        self,
        *,
        embed_weight: torch.Tensor,
        norm: Callable[[torch.Tensor], torch.Tensor],
        vocab: int,
    ) -> None:
        """Collective (the check all-reduces). The whole vocab's rows of the served pre-fc embedding norm."""
        hidden = embed_weight.shape[1]
        if embed_weight.shape[0] < vocab or vocab % self.world != 0 or hidden % 8 != 0:
            raise RuntimeError(
                f"DEP-DX memo needs the full embedding table, got {tuple(embed_weight.shape)}"
            )
        rows = embed_weight[:vocab]
        memo = torch.empty((vocab, hidden), dtype=torch.bfloat16, device=rows.device)
        for i in range(0, vocab, _MEMO_BLOCK):
            memo[i : i + _MEMO_BLOCK] = norm(rows[i : i + _MEMO_BLOCK].contiguous())
        self._check_memo(memo=memo, rows=rows, norm=norm)
        self.memo = memo
        self.fc_in = torch.zeros(
            (self.max_rows, 2 * hidden), dtype=torch.bfloat16, device=rows.device
        )
        self.side_stream = torch.cuda.Stream(device=rows.device)
        logger.info(
            "DEP-DX memo active (%d rows, %.1f MB on this rank).",
            vocab,
            memo.numel() * 2 / 2**20,
        )

    def _check_memo(
        self,
        *,
        memo: torch.Tensor,
        rows: torch.Tensor,
        norm: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        # Served draft steps norm 1..max_rows rows at once; check rows at every served width.
        gen = torch.Generator(device="cpu").manual_seed(self.rank)
        ids = torch.randint(0, rows.shape[0], (_MEMO_CHECK_ROWS,), generator=gen).to(
            rows.device
        )
        bad = torch.zeros(1, dtype=torch.int64, device=rows.device)
        for width in range(1, self.max_rows + 1):
            for i in range(0, _MEMO_CHECK_ROWS, width):
                sel = ids[i : i + width]
                ref = norm(rows[sel].contiguous())
                bad += (ref.view(torch.int16) != memo[sel].view(torch.int16)).any(dim=1).sum()
        dist.all_reduce(bad, group=self.device_group)
        if int(bad[0]) != 0:
            raise RuntimeError(
                f"DEP-DX memo differs from the served norm on {int(bad[0])} checked rows"
            )

    def begin_draft(self) -> None:
        self._fc_ready = None

    def end_draft(self) -> None:
        # A capture must rejoin the side stream; so must eager work that reuses fc_in.
        if self._side_pending:
            torch.cuda.current_stream().wait_stream(self.side_stream)
        self._side_pending = False
        self._fc_ready = None

    def hidden_half(
        self, *, hidden_states: torch.Tensor, norm: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        """The next step's normed hidden half on the side stream, beside the lm_head."""
        if not self._produce:
            return
        rows = hidden_states.shape[0]
        hidden = hidden_states.shape[1]
        self.side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.side_stream):
            self.fc_in[:rows, hidden:].copy_(norm(hidden_states))
        self._hidden_rows = rows
        self._side_pending = True

    def take_fc_input(self, *, rows: int) -> Optional[torch.Tensor]:
        """This step's fc input [rows, 2H] when the previous step of the same draft wrote it."""
        if self._fc_ready != rows:
            return None
        self._fc_ready = None
        torch.cuda.current_stream().wait_stream(self.side_stream)
        self._side_pending = False
        return self.fc_in[:rows]

    def begin_step(
        self,
        *,
        positions: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_column: int,
        mrope_positions: Optional[torch.Tensor],
        produce_fc: bool,
    ) -> None:
        self._step = (positions, draft_tokens, draft_column, mrope_positions)
        self._finish = None
        self._produce = produce_fc and self.memo is not None
        self._embed_rows = None
        self._hidden_rows = None

    def end_step(self) -> Optional[StepFinish]:
        """The finish outputs the step's select wrote, or None when the stock finish must run."""
        finish = self._finish
        if (
            self._produce
            and self._embed_rows is not None
            and self._embed_rows == self._hidden_rows
        ):
            self._fc_ready = self._embed_rows
        self._step = None
        self._finish = None
        self._produce = False
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
        memo = fc_out = self.no_rows
        if step is not None and self._produce:
            memo = self.memo
            fc_out = self.fc_in[:rows, : self.memo.shape[1]]
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
            memo=memo,
            fc_out=fc_out,
            vocab_start=self.rank * logits.shape[1],
            torch_nan=torch_nan,
            rank=self.rank,
            rows=rows,
            draft_column=column,
        )
        if step is not None:
            self._finish = (topk_p, tokens, mrope.numel() > 0)
            if fc_out.numel() > 0:
                self._embed_rows = rows
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
    KINDS = ("decode", "extend", "fc")

    def __init__(self):
        # [kind][calls, rows, mismatched rows]
        self.counters = torch.zeros(
            (len(self.KINDS), 3), dtype=torch.int64, device=torch.cuda.current_device()
        )
        self.host_counters = torch.zeros(
            (len(self.KINDS), 3), dtype=torch.int64, device="cpu", pin_memory=True
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
        self._count(kind=int(extend), bad=bad)

    def check_fc(self, *, fc_input: torch.Tensor, stock: torch.Tensor) -> None:
        """The memo-built fc input against the stock embedding + norms + cat of the same step."""
        self._count(
            kind=2, bad=(fc_input.view(torch.int16) != stock.view(torch.int16)).any(dim=1)
        )

    def _count(self, *, kind: int, bad: torch.Tensor) -> None:
        counters = self.counters[kind]
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
