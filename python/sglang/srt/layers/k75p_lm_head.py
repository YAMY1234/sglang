"""Greedy draft tokens on the vocab-parallel TP LM head without the logits all-to-all.

S2 gathers the draft hidden rows MAX_LEN-style with a multimem one-shot store instead
of zero-fill + all-reduce; S1 reduces each row on this rank's vocab shard and gathers
the (value, id) pairs. Tokens equal the argmax over the full vocab, ties to the lowest
id, with the NaN rule of the argmax each draft call replaces.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.kernels.ops.speculative.k75p_argmax import k75p_select, k75p_shard_argmax
from sglang.srt.distributed.device_communicators import triton_symm_mem_ag
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

# One (fp32 bits, int32 id) pair as bf16 elements for the multimem gather.
_PAIR_BF16 = 4


class K75PDraftSelect:
    def __init__(self, *, max_rows: int, hidden_size: int):
        tp_group = get_parallel().tp_group
        self.world = tp_group.world_size
        self.rank = tp_group.rank_in_group
        self.max_rows = max_rows
        self.hidden_size = hidden_size
        # Flat one-token buffers: rank r's block lands at offset r * block, so the
        # gathered rows come out rank-major ([world * rows, ...]).
        self.hidden_state = triton_symm_mem_ag.create_state(
            group=tp_group.device_group,
            rank_in_group=self.rank,
            max_tokens=1,
            hidden_size=self.world * max_rows * hidden_size,
        )
        self.pairs_state = triton_symm_mem_ag.create_state(
            group=tp_group.device_group,
            rank_in_group=self.rank,
            max_tokens=1,
            hidden_size=self.world * self.world * max_rows * _PAIR_BF16,
        )
        self.pairs = torch.empty(
            (self.world * max_rows, 2),
            dtype=torch.int32,
            device=self.hidden_state.device,
        )

    @classmethod
    def create(
        cls, *, max_rows: int, hidden_size: int
    ) -> Optional["K75PDraftSelect"]:
        select = cls(max_rows=max_rows, hidden_size=hidden_size)
        if (
            select.hidden_state.symm_mem_hdl.multicast_ptr == 0
            or select.pairs_state.symm_mem_hdl.multicast_ptr == 0
        ):
            logger.warning("K75P draft LM head disabled: no multicast on the TP group.")
            return None
        logger.info(
            "K75P draft LM head active (tp=%d, rows/rank <= %d).", select.world, max_rows
        )
        return select

    def fits(self, hidden_states: torch.Tensor) -> bool:
        return (
            0 < hidden_states.shape[0] <= self.max_rows
            and hidden_states.shape[1] == self.hidden_size
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
        )

    def gather_rows(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # No entry barrier: every rank passes this call's pair gather, which follows
        # its GEMM, before any rank stores the next call's rows (and vice versa).
        rows = hidden_states.shape[0]
        return triton_symm_mem_ag.all_gather_inner(
            self.hidden_state,
            hidden_states.view(1, rows * self.hidden_size),
            tp_hidden_dim=self.world * rows * self.hidden_size,
            skip_entry_sync=True,
            safe=False,
        ).view(self.world * rows, self.hidden_size)

    def select(
        self, *, logits: torch.Tensor, num_local_rows: int, torch_nan: bool
    ) -> torch.Tensor:
        total_rows = logits.shape[0]
        pairs = self.pairs[:total_rows]
        k75p_shard_argmax(
            logits=logits,
            pairs=pairs,
            vocab_start=self.rank * logits.shape[1],
            torch_nan=torch_nan,
        )
        gathered = triton_symm_mem_ag.all_gather_inner(
            self.pairs_state,
            pairs.view(torch.bfloat16).view(1, total_rows * _PAIR_BF16),
            tp_hidden_dim=self.world * total_rows * _PAIR_BF16,
            skip_entry_sync=True,
            safe=False,
        )
        return k75p_select(
            gathered=gathered.view(torch.int32).view(self.world, total_rows, 2),
            row_start=self.rank * num_local_rows,
            num_rows=num_local_rows,
            torch_nan=torch_nan,
        )
