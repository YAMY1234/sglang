"""Single-token greedy MTP draft step boundary in two launches instead of seven.

Today: shard argmax pack, all-gather, select, next step's fc slice GEMV, all-gather of the
fp32 partials, rank-order sum. Here: one launch does the shard argmax and a multicast push of
each rank's candidate row together with its hidden-half fc partial (computed beside the
lm_head), selects the winner on every rank, and a second launch finishes the fc from the
exchanged data. Every value equals the split-fc path bit for bit: same argmax keys and ties,
same per-slice dot arithmetic, same rank-order fp32 sum and bf16 rounding.

The memo variant ships, per candidate, the embedding-half partials already summed in rank
order (built at load time from the checkpoint weights with the same kernel arithmetic and
checked against the split-fc kernels on every hot row), so the embedding-half GEMV leaves
the critical path.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist

from sglang.kernels.ops.speculative.draft_fc_partial import (
    draft_fc_partial,
    draft_fc_partial_rows,
)
from sglang.kernels.ops.speculative.draft_vocab_argmax import NAN_KEY_SPLIT_ARGMAX
from sglang.kernels.ops.speculative.mkd_draft_fc import mkd_fc_memo, mkd_fc_sum
from sglang.kernels.ops.speculative.mkd_draft_xchg import (
    mkd_xchg,
    mkd_xchg_words,
    mkd_xchg_zero,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

# Memo rows built per launch at load time.
_MEMO_BLOCK = 4096


def _clear_negative_zero(rows: torch.Tensor) -> torch.Tensor:
    # The exchanged embedding row has -0.0 cleared (vocab_argmax_pack); memo from the same row.
    bits = rows.view(torch.int16)
    return torch.where(bits == -32768, torch.zeros_like(bits), bits).view(torch.bfloat16)


def _unsupported_reason(model, exchange) -> Optional[str]:
    if model.fc_split_slice is None or model._fc_gemv_rows is None:
        return "needs SGLANG_Q35MF_DRAFT_FC_SPLIT and SGLANG_Q35MF_DRAFT_FC_GEMV"
    if exchange.token_map.numel() == 0:
        return "needs the hot vocabulary (SGLANG_Q35MF_SPEC_HOT_VOCAB)"
    world = get_parallel().tp_size
    if world not in (4, 8):
        return f"tp_size {world}"
    if model.config.hidden_size != 4096:
        return f"hidden size {model.config.hidden_size}"
    return None


class MkdDraftBoundary:
    """Exchange buffers, the optional memo table and the per-step launches."""

    def __init__(self, model, exchange) -> None:
        from flashinfer.comm.mnnvl import McastGPUBuffer

        from sglang.srt.distributed.parallel_state import get_tp_group
        from sglang.srt.layers import flashinfer_comm_fusion

        self.model = model
        parallel = get_parallel()
        self.world, self.rank = parallel.tp_size, parallel.tp_rank
        hidden = model.config.hidden_size
        self.hidden = hidden
        use_hidden, start, width = model.fc_split_slice
        self.is_hidden_rank = use_hidden
        self.slice_start, self.width = start, width
        self.n_embed = hidden // width
        dev = model.fc.weight.device
        self.memo = envs.SGLANG_Q35MF_MKD_TTABLE.get()
        self.token_map = exchange.token_map
        self.cand = self._build_memo(exchange) if self.memo else exchange.embed_weight

        tp = get_tp_group()
        backend = flashinfer_comm_fusion._TorchDistBackend(
            device_group=tp.device_group, cpu_group=tp.cpu_group
        )
        words = mkd_xchg_words(hidden, self.world, self.memo)
        self.buf = McastGPUBuffer(words * 8, self.world, self.rank, dev, backend)
        self.mc = self.buf.get_multicast_ptr()
        self.uc = self.buf.get_unicast_ptr(self.rank)
        mkd_xchg_zero(hidden, self.world, self.memo, self.uc)
        torch.cuda.synchronize(dev)
        dist.barrier(group=tp.cpu_group)

        self.epoch_done = torch.tensor([1, 0], dtype=torch.int32, device=dev)
        self.own_partial = torch.zeros(1, hidden, dtype=torch.float32, device=dev)
        self.out_partials = torch.zeros(
            self.world - self.n_embed, hidden, dtype=torch.float32, device=dev
        )
        self.out_cand = torch.empty(
            1, hidden, dtype=torch.float32 if self.memo else torch.bfloat16, device=dev
        )
        self.fc_out = torch.empty(1, hidden, dtype=torch.bfloat16, device=dev)
        self.side_stream = torch.cuda.Stream(device=dev)
        # armed: this draft step ends in the exchange; pending: the next step's fc reads it.
        self.armed = False
        self.pending = False
        logger.info(
            "MKD draft boundary active (tp=%d, %s candidates, %.1f MB exchange buffer).",
            self.world,
            "memo" if self.memo else "embedding",
            words * 8 / 2**20,
        )

    @classmethod
    def maybe_create(cls, model, exchange) -> Optional["MkdDraftBoundary"]:
        reason = _unsupported_reason(model, exchange)
        if reason is not None:
            logger.warning("MKD draft boundary disabled (%s); using the stock path.", reason)
            return None
        return cls(model, exchange)

    # ---------------------------------------------------------------- memo table

    def _build_memo(self, exchange) -> torch.Tensor:
        model = self.model
        emb = exchange.embed_weight
        rows = emb.shape[0]
        memo = torch.empty(rows, self.hidden, dtype=torch.float32, device=emb.device)
        norm = model.pre_fc_norm_embedding
        for i in range(0, rows, _MEMO_BLOCK):
            block = _clear_negative_zero(emb[i : i + _MEMO_BLOCK].clone())
            mkd_fc_memo(
                block,
                model.fc.weight,
                norm.weight.data,
                norm.variance_epsilon,
                self.world,
                memo[i : i + _MEMO_BLOCK],
            )
        bad = self._check_memo(emb, memo)
        bad_t = torch.tensor([bad, rows], dtype=torch.int64, device=emb.device)
        dist.all_reduce(bad_t, group=get_parallel().tp_group.device_group)
        if int(bad_t[0]) != 0:
            raise RuntimeError(
                f"MKD memo table differs from the split fc on {int(bad_t[0])} of {int(bad_t[1])} hot rows"
            )
        logger.info(
            "MKD memo table: %d rows on this rank (%.1f MB), equal to the split fc on all %d hot rows.",
            rows,
            memo.numel() * 4 / 2**20,
            int(bad_t[1]),
        )
        return memo

    def _check_memo(self, emb: torch.Tensor, memo: torch.Tensor) -> int:
        """Rows whose memo differs from the split-fc kernels (every row, all slices)."""
        model = self.model
        norm = model.pre_fc_norm_embedding
        sms = torch.cuda.get_device_properties(emb.device).multi_processor_count
        width = self.width
        gemv_rows = draft_fc_partial_rows(self.hidden, width, sms)
        shards = [
            model.fc.weight[:, s * width : (s + 1) * width].contiguous()
            for s in range(self.n_embed)
        ]
        part = torch.empty(1, self.hidden, dtype=torch.float32, device=emb.device)
        bad = torch.zeros((), dtype=torch.int64, device=emb.device)
        for v in range(emb.shape[0]):
            x = _clear_negative_zero(emb[v : v + 1].clone())
            acc = None
            for s in range(self.n_embed):
                draft_fc_partial(x, norm.weight.data, norm.variance_epsilon, shards[s], s * width, part, gemv_rows)
                acc = part.clone() if acc is None else acc + part
            bad += (acc.view(torch.int32) != memo[v : v + 1].view(torch.int32)).any()
        return int(bad)

    # ---------------------------------------------------------------- per step

    def hidden_partial(self, hidden_states: torch.Tensor) -> None:
        """This rank's hidden-slice fc partial for the next draft step, on a side stream."""
        if not self.is_hidden_rank:
            return
        model = self.model
        norm = model.pre_fc_norm_hidden
        cur = torch.cuda.current_stream()
        self.side_stream.wait_stream(cur)
        with torch.cuda.stream(self.side_stream):
            draft_fc_partial(
                hidden_states,
                norm.weight.data,
                norm.variance_epsilon,
                model._fc_shard,
                self.slice_start,
                self.own_partial,
                model._fc_gemv_rows,
            )

    def join(self) -> None:
        if self.is_hidden_rank:
            torch.cuda.current_stream().wait_stream(self.side_stream)

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
        self.join()
        mkd_xchg(
            local_logits,
            self.cand,
            self.token_map,
            self.own_partial,
            self.epoch_done,
            self.out_cand,
            self.out_partials,
            topk_index,
            topk_p,
            draft_tokens,
            positions,
            mrope_positions,
            self.mc,
            self.uc,
            self.rank,
            self.world,
            draft_column,
            NAN_KEY_SPLIT_ARGMAX,
        )
        self.pending = True

    def fc(self) -> torch.Tensor:
        """The draft fc output [1, H] of the step after an exchange."""
        model = self.model
        norm = model.pre_fc_norm_embedding
        out = torch.empty_like(self.fc_out)
        mkd_fc_sum(
            self.out_cand,
            self.out_cand if self.memo else None,
            self.out_partials,
            model.fc.weight,
            norm.weight.data,
            norm.variance_epsilon,
            out,
            self.world,
        )
        self.pending = False
        return out
