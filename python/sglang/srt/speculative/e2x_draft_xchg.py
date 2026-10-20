"""TP2 greedy MTP draft-step boundary in one launch for up to 8 rows (E2xDraftFuse.XCHG).

Today (TP2, fc split): shard argmax + peer exchange of the winner's embedding row, then per rank
the pre-fc norm and a cuBLAS fp32 partial of its fc input slice (rank 0: embedding half, rank 1:
hidden half), then a peer exchange summing the two partials into bf16. Here every rank carries a
load-time memo of the embedding-half partial of each of its vocabulary rows (the same norm and
cuBLAS call, 8-row blocks), rank 1 computes its hidden-half partial beside the lm_head, and one
launch selects the token and writes bf16(memo[winner] + hidden partial). The cuBLAS partial is
batch-invariant up to 8 rows on GB300, so the fc output equals the split fc bit for bit there.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist

from sglang.kernels.ops.speculative.draft_vocab_argmax import (
    NAN_KEY_SPLIT_ARGMAX,
    NAN_KEY_TORCH_ARGMAX,
    PACK_BLOCKS,
)
from sglang.kernels.ops.speculative.e2x_draft_xchg import (
    FP32_MARKER,
    e2x_draft_xchg,
    e2x_slots_shape,
)
from sglang.srt.layers.kdense_draft_xchg import PeerSlots
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

# The fc split gives rank 1 the hidden half of the fc input columns.
_HIDDEN_RANK = 1
# Memo rows per cuBLAS call: the fp32 slice partial is batch-invariant only up to 8 rows (GB300).
_MEMO_BLOCK = 8
# Rows checked against the served split-fc arithmetic after building the memo.
_MEMO_CHECK_ROWS = 64
# Draft calls between checks that every rank takes the same extend-fc fill decision (arbitrary).
_FILL_AGREE_EVERY = 512


def _clear_negative_zero(rows: torch.Tensor) -> torch.Tensor:
    # The exchanged embedding row has -0.0 cleared; the memo is built from the same row.
    bits = rows.view(torch.int16)
    return torch.where(bits == -32768, torch.zeros_like(bits), bits).view(
        torch.bfloat16
    )


def _unsupported_reason(model, exchange) -> Optional[str]:
    if model.fc_split_slice is None:
        return "needs SGLANG_Q35MF_DRAFT_FC_SPLIT"
    if get_parallel().tp_size != 2:
        return f"tp_size {get_parallel().tp_size}"
    if exchange.token_map.numel() != 0:
        return "hot vocabulary"
    if model.config.hidden_size % 8 != 0:
        return f"hidden size {model.config.hidden_size}"
    return None


class E2xDraftXchg:
    """Memo table, exchange slots and the per-step launches; same interface as MkdDraftBoundary."""

    max_rows = 8
    extend_fc = True
    extend_partial_in_graph = True

    def __init__(self, model, exchange) -> None:
        self.model = model
        parallel = get_parallel()
        self.rank = parallel.tp_rank
        self.hidden = model.config.hidden_size
        dev = model.fc.weight.device
        self.is_hidden_rank = self.rank == _HIDDEN_RANK
        h = self.hidden
        # Same values as each rank's served _fc_shard (contiguous column slices of the replicated fc).
        self.w_embed = model.fc.weight[:, :h].contiguous()
        self.w_hidden = (
            model.fc.weight[:, h:].contiguous() if self.is_hidden_rank else None
        )
        self.vocab_start = exchange.vocab_start
        self.memo = self._build_memo(exchange.embed_weight)
        self.peer = PeerSlots(
            name="e2x",
            shape=e2x_slots_shape(self.max_rows, h),
            dtype=torch.float32,
            marker=FP32_MARKER,
            epoch_shape=(self.max_rows,),
        )
        if not self.peer.ready():
            raise RuntimeError("E2X draft exchange: peer slots unavailable")
        # Wider batches keep the stock exchanges; build their slots now, as their first eager
        # call (which builds them in the stock flow) may now come after graph capture.
        if exchange.peer is not None:
            exchange.peer.ready()
        if model._fc_peer_sum is not None:
            model._fc_peer_sum.peer.ready()
        self.part_key = torch.zeros(
            self.max_rows, PACK_BLOCKS, dtype=torch.int32, device=dev
        )
        self.part_idx = torch.zeros_like(self.part_key)
        self.counter = torch.zeros(self.max_rows, dtype=torch.int32, device=dev)
        # Zeros: graph-capture warmups read it before any exchange has written it.
        self.fc_buf = torch.zeros(self.max_rows, h, dtype=torch.bfloat16, device=dev)
        # Rank 1: hidden-half partial of the draft extend's selected rows, written inside the extend graph.
        self.p_extend = (
            torch.zeros(self.max_rows, h, dtype=torch.float32, device=dev)
            if self.is_hidden_rank
            else None
        )
        self.no_partial = torch.empty(0, dtype=torch.float32, device=dev)
        self.no_ints = torch.empty(0, dtype=torch.int64, device=dev)
        self.no_out = torch.empty(0, dtype=torch.bfloat16, device=dev)
        self.side_stream = torch.cuda.Stream(device=dev)
        self.p_hidden: Optional[torch.Tensor] = None
        self.partial_pending = False
        self.armed = False
        self.pending = False
        self.last_step = False
        self.step = 0
        self.rows = 1
        # Draft input sequence numbers that decide whether fc_buf holds the coming draft's first fc;
        # every rank follows the same SPMD control flow.
        self.input_seq = 0
        self.fc_buf_key: Optional[int] = None
        # Requests of fc_buf's rows, in order: a draft input whose batch was filtered or reordered since
        # its extend must not take them (a finished request shifts every later row).
        self.extend_rows: Optional[tuple] = None
        self.fc_buf_rows: Optional[tuple] = None
        self._fill_checks = 0
        logger.info(
            "E2X draft exchange active (tp=2, <= %d rows, memo %.1f MB on this rank).",
            self.max_rows,
            self.memo.numel() * 4 / 2**20,
        )

    @classmethod
    def maybe_create(cls, model, exchange) -> Optional[E2xDraftXchg]:
        reason = _unsupported_reason(model, exchange)
        if reason is not None:
            logger.warning(
                "E2X draft exchange disabled (%s); using the stock path.", reason
            )
            return None
        return cls(model, exchange)

    # ---------------------------------------------------------------- memo table

    def _embed_partial(self, rows: torch.Tensor) -> torch.Tensor:
        norm = self.model.pre_fc_norm_embedding
        return torch.mm(norm(rows), self.w_embed.T, out_dtype=torch.float32)

    def _build_memo(self, embed: torch.Tensor) -> torch.Tensor:
        rows = embed.shape[0]
        memo = torch.empty(rows, self.hidden, dtype=torch.float32, device=embed.device)
        for i in range(0, rows, _MEMO_BLOCK):
            block = _clear_negative_zero(embed[i : i + _MEMO_BLOCK].clone())
            memo[i : i + _MEMO_BLOCK] = self._embed_partial(block)
        # Served draft steps compute 1-8 rows at once; check rows at every served width.
        gen = torch.Generator(device="cpu").manual_seed(self.rank)
        ids = torch.randint(0, rows, (_MEMO_CHECK_ROWS,), generator=gen).to(
            embed.device
        )
        bad = 0
        for width in range(1, self.max_rows + 1):
            for i in range(0, _MEMO_CHECK_ROWS, width):
                sel = ids[i : i + width]
                ref = self._embed_partial(_clear_negative_zero(embed[sel].clone()))
                bad += int(
                    (ref.view(torch.int32) != memo[sel].view(torch.int32))
                    .any(dim=1)
                    .sum()
                )
        bad_t = torch.tensor([bad], dtype=torch.int64, device=embed.device)
        dist.all_reduce(bad_t, group=get_parallel().tp_group.device_group)
        if int(bad_t[0]) != 0:
            raise RuntimeError(
                f"E2X draft memo differs from the split fc on {int(bad_t[0])} checked rows"
            )
        return memo

    # ---------------------------------------------------------------- per step

    def hidden_partial(self, hidden_states: torch.Tensor) -> None:
        """Rank 1's hidden-half fc partial of the next draft step, on a side stream beside the lm_head."""
        if not self.is_hidden_rank or self.last_step:
            return
        norm = self.model.pre_fc_norm_hidden
        self.side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.side_stream):
            self.p_hidden = torch.mm(
                norm(hidden_states), self.w_hidden.T, out_dtype=torch.float32
            )
        self.partial_pending = True

    def extend_hidden_partial(
        self, hidden_states: torch.Tensor, select_index: Optional[torch.Tensor]
    ) -> bool:
        """Rank 1: the draft extend's selected rows' hidden-half partial on the side stream (joined after
        the lm_head by join_extend_partial); same norm and cuBLAS call as the stock first draft step.
        """
        if (
            not self.is_hidden_rank
            or select_index is None
            or not 0 < select_index.numel() <= self.max_rows
        ):
            return False
        norm = self.model.pre_fc_norm_hidden
        self.side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.side_stream):
            partial = torch.mm(
                norm(hidden_states[select_index]),
                self.w_hidden.T,
                out_dtype=torch.float32,
            )
            self.p_extend[: partial.shape[0]].copy_(partial)
        return True

    def join_extend_partial(self) -> None:
        torch.cuda.current_stream().wait_stream(self.side_stream)

    def _launch(
        self,
        local_logits: torch.Tensor,
        topk_index: torch.Tensor,
        topk_p: torch.Tensor,
        draft_tokens: Optional[torch.Tensor],
        positions: Optional[torch.Tensor],
        mrope_positions: Optional[torch.Tensor],
        draft_column: int,
        nan_key: int,
        payload: bool,
    ) -> None:
        m = local_logits.shape[0]
        p_hidden = (
            self.p_hidden[:m]
            if payload and self.is_hidden_rank and self.p_hidden is not None
            else self.no_partial
        )
        e2x_draft_xchg(
            local_logits,
            self.memo,
            p_hidden,
            self.part_key[:m],
            self.part_idx[:m],
            self.counter[:m],
            self.peer.slots,
            self.peer.peer_slots,
            self.peer.epoch,
            topk_index,
            topk_p,
            draft_tokens if draft_tokens is not None else self.no_ints,
            positions if positions is not None else self.no_ints,
            mrope_positions if mrope_positions is not None else self.no_ints,
            self.fc_buf[:m] if payload else self.no_out,
            self.vocab_start,
            nan_key,
            self.rank,
            _HIDDEN_RANK,
            draft_column,
        )
        self.rows = m

    def exchange(
        self,
        local_logits: torch.Tensor,
        topk_index: torch.Tensor,
        topk_p: torch.Tensor,
        draft_tokens: Optional[torch.Tensor],
        positions: torch.Tensor,
        draft_column: int,
        mrope_positions: Optional[torch.Tensor],
    ) -> None:
        if self.partial_pending:
            torch.cuda.current_stream().wait_stream(self.side_stream)
            self.partial_pending = False
        self._launch(
            local_logits,
            topk_index,
            topk_p,
            draft_tokens,
            positions,
            mrope_positions,
            draft_column,
            NAN_KEY_SPLIT_ARGMAX,
            payload=not self.last_step,
        )
        self.pending = not self.last_step

    def exchange_extend(
        self, local_logits: torch.Tensor, hidden_states: Optional[torch.Tensor]
    ):
        """Draft-extend token (torch.argmax NaN rule); also writes the next draft's first fc into fc_buf."""
        m = local_logits.shape[0]
        topk_index = torch.empty((m, 1), dtype=torch.int64, device=local_logits.device)
        topk_p = torch.empty((m, 1), dtype=torch.float32, device=local_logits.device)
        payload = hidden_states is not None and hidden_states.shape[0] == m
        if payload and self.is_hidden_rank:
            # Written by extend_hidden_partial in the extend forward for these rows.
            self.p_hidden = self.p_extend
        self._launch(
            local_logits,
            topk_index,
            topk_p,
            None,
            None,
            None,
            0,
            NAN_KEY_TORCH_ARGMAX,
            payload=payload,
        )
        self.fc_buf_key = self.input_seq if payload else None
        self.fc_buf_rows = self.extend_rows if payload else None
        return topk_p, topk_index

    def note_draft_input(self) -> None:
        self.input_seq += 1

    def take_fc_buf(self, rows: Optional[tuple] = None) -> bool:
        """True when fc_buf holds the first fc of the coming draft: its extend produced the input and
        the draft batch has the same requests in the same order."""
        holds = (
            self.fc_buf_key is not None
            and self.fc_buf_key == self.input_seq
            and rows is not None
            and rows == self.fc_buf_rows
        )
        self.fc_buf_key = None
        self.fc_buf_rows = None
        self._fill_checks += 1
        if self._fill_checks % _FILL_AGREE_EVERY == 1:
            self._check_ranks_agree(holds)
        return holds

    def _check_ranks_agree(self, holds: bool) -> None:
        # The decision feeds collective launches (fill_fc); confirm the SPMD premise now and then.
        t = torch.tensor([self.input_seq, int(holds)], dtype=torch.int64)
        lo, hi = t.clone(), t.clone()
        group = get_parallel().tp_group.cpu_group
        dist.all_reduce(lo, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX, group=group)
        if not torch.equal(lo, hi):
            raise RuntimeError(
                f"E2X draft extend fc: ranks disagree on the fill decision ({lo.tolist()} vs {hi.tolist()})"
            )

    def fill_fc(
        self, token_ids: torch.Tensor, hidden_states: torch.Tensor, forward_batch
    ) -> None:
        """Stock split fc of a draft's first step into fc_buf, when no extend exchange produced it."""
        model = self.model
        self.pending = False
        embeds = model.model.embed_tokens(token_ids.flatten())
        m = embeds.shape[0]
        model._fc_split_ready(m)
        self.fc_buf[:m].copy_(model._split_fc(embeds, hidden_states, forward_batch))
        self.rows = m

    def fc(self) -> torch.Tensor:
        """The draft fc output [rows, H] of the step after an exchange."""
        self.pending = False
        return self.fc_buf[: self.rows]
