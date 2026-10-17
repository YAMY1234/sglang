# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DSpark static greedy verify: the target argmax from per-rank keys.

Developed at NVIDIA. Inside the static verify graph, the LogitsProcessor hands this
rank's LM-head shard to the sink before the all-gather. The exchange kernel
(sglang.kernels.ops.speculative.dspark.local_argmax) turns it into the global argmax ids
the greedy accept reads, and the full-logits gather and copy move to a side stream
that the verify epilogue joins at its end. When no reader of the full logits applies
to a replay, a device flag skips the gather and the copy on every rank.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist

from sglang.kernels.ops.speculative.dspark import local_argmax as kernel

logger = logging.getLogger(__name__)

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))
SUPPORTED_WORLDS = (2, 4, 8)
# Arbitrary; the grid is fixed so that each CTA counts its own launches.
CTAS_PER_SM = 2


def unsupported_reason(
    *, world: int, vocab_size: int, capability: tuple[int, int]
) -> Optional[str]:
    """Why the exchange cannot serve this configuration, or None; the same on every rank."""
    if world not in SUPPORTED_WORLDS:
        return f"a tensor-parallel size of {world}"
    if tuple(capability) not in SUPPORTED_CAPABILITIES:
        return f"compute capability {capability} (needs SM100 or SM103)"
    if vocab_size > kernel.MAX_VOCAB:
        return f"a vocabulary of {vocab_size} (the key holds {kernel.MAX_VOCAB})"
    return None


def verify_reads_full_logits(
    *, fold_eligible: bool, return_logprob: bool, observers_read_logits: bool
) -> bool:
    """Whether anything after a static verify replay reads its full logits.

    A folded greedy accept reads the argmax ids instead. The inputs are batch
    properties and env, so every TP rank agrees, which the gather's barriers need.
    """
    return not fold_eligible or return_logprob or observers_read_logits


def _all_ranks_true(group, value: bool) -> bool:
    """``value`` on every rank of the CPU ``group`` (None: this rank only)."""
    if group is None:
        return value
    flag = torch.tensor([int(value)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


class LocalArgmaxExchange:
    """Global row argmax ids from this rank's [rows, vocab_local] logits shard.

    Construction is collective over the CPU ``group`` (None: one process).
    """

    def __init__(
        self,
        *,
        max_rows: int,
        group,
        rank: int,
        world: int,
        device,
        timeout_s: float,
    ):
        if timeout_s <= 0:
            raise ValueError(f"the argmax exchange needs a deadline, got {timeout_s}")
        self.rank, self.world, self.max_rows = rank, world, int(max_rows)
        self.device = torch.device(device)
        self.timeout_ns = int(timeout_s * 1e9)
        self._handle = None
        # Compile before the collective set-up so a build failure cannot leave peers waiting.
        per_sm, failure = 0, None
        try:
            kernel.precompile()
            per_sm = min(
                kernel.max_ctas_per_sm(device=self.device, logit_dtype=dtype)
                for dtype in kernel.LOGIT_DTYPES
            )
            if per_sm == 0:
                raise RuntimeError("the argmax exchange kernel does not fit on an SM")
        except (RuntimeError, ImportError) as error:
            failure = error
        if not _all_ranks_true(group, failure is None):
            raise RuntimeError(
                f"argmax exchange set-up failed on a rank: {failure or 'another rank'}"
            )
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.ctas = min(per_sm, CTAS_PER_SM) * sms
        self.max_items = max(self.ctas, self.max_rows)
        self.counters = torch.zeros(self.ctas, dtype=torch.int32, device=self.device)
        self.ids = torch.zeros(self.max_rows, dtype=torch.int64, device=self.device)
        size = kernel.mailbox_bytes(max_items=self.max_items, world=world)
        self._mailbox, pointers = self._allocate_mailbox(size, group=group)
        self.peers = torch.tensor(pointers, dtype=torch.int64, device=self.device)

    def _allocate_mailbox(self, size: int, *, group) -> tuple[torch.Tensor, list[int]]:
        if group is None:
            buffer = torch.zeros(size, dtype=torch.uint8, device=self.device)
            return buffer, [buffer.data_ptr()]
        import torch.distributed._symmetric_memory as symm_mem

        buffer = symm_mem.empty(size, dtype=torch.uint8, device=self.device)
        handle = symm_mem.rendezvous(buffer, group.group_name)
        pointers = [
            handle.get_buffer(peer, (size,), torch.uint8, storage_offset=0).data_ptr()
            for peer in range(self.world)
        ]
        buffer.zero_()
        self._handle = handle
        torch.cuda.synchronize(self.device)
        dist.barrier(group=group)
        return buffer, pointers

    def supports(self, local_logits: torch.Tensor, *, vocab_size: int) -> bool:
        """Shape and layout only, which tensor parallelism keeps identical across ranks."""
        if local_logits.dim() != 2:
            return False
        rows, vocab_local = local_logits.shape
        return (
            local_logits.dtype in kernel.LOGIT_DTYPES
            and 0 < rows <= self.max_rows
            and local_logits.stride(1) == 1
            and vocab_local % kernel.CHUNK_ALIGN == 0
            and local_logits.stride(0) % kernel.CHUNK_ALIGN == 0
            and local_logits.data_ptr() % 16 == 0
            and vocab_size <= min(vocab_local * self.world, kernel.MAX_VOCAB)
        )

    def __call__(self, local_logits: torch.Tensor, *, vocab_size: int) -> torch.Tensor:
        """[rows] int64 global argmax ids over columns < ``vocab_size`` (a view of a
        persistent buffer)."""
        assert self.supports(local_logits, vocab_size=vocab_size), (
            tuple(local_logits.shape),
            local_logits.dtype,
            local_logits.stride(),
        )
        rows, vocab_local = local_logits.shape
        # Columns past the vocabulary are LM-head padding, which the gathered path slices off.
        valid_local = min(max(vocab_size - self.rank * vocab_local, 0), vocab_local)
        kernel.local_argmax_exchange(
            logits=local_logits,
            valid_local=valid_local,
            peers=self.peers,
            ids=self.ids,
            counters=self.counters,
            rank=self.rank,
            world=self.world,
            max_items=self.max_items,
            timeout_ns=self.timeout_ns,
            ctas=self.ctas,
        )
        return self.ids[:rows]


class LocalArgmaxSink:
    """Static verify capture only: the argmax ids come from the LM-head shard on the
    main stream; the full-logits gather and copy run on ``stream``, which ``join``
    merges back. Construction is collective over the TP CPU group."""

    def __init__(self, *, exchange: LocalArgmaxExchange, device):
        self._exchange = exchange
        self.stream = torch.cuda.Stream(device=device)
        self.ids: Optional[torch.Tensor] = None
        self._main: Optional[torch.cuda.Stream] = None
        # Read in the graph: 0 skips the full-logits gather and copy on `stream`.
        # Rank-uniform by contract (set from batch properties all TP ranks share).
        self.full_logits_flag = torch.ones((1,), dtype=torch.int32, device=device)
        self._full_logits_needed = True
        # The flag-gated gather's barriers trap at the exchange's deadline.
        self.timeout_ns = exchange.timeout_ns

    @property
    def pending(self) -> bool:
        return self.ids is not None

    def supports(self, local_logits: torch.Tensor, *, vocab_size: int) -> bool:
        return self._exchange.supports(local_logits, vocab_size=vocab_size)

    def set_full_logits_needed(self, needed: bool) -> None:
        """Before a verify replay; rewrites the device flag only on change."""
        if needed != self._full_logits_needed:
            self.full_logits_flag.fill_(int(needed))
            self._full_logits_needed = needed

    def fork(self, local_logits: torch.Tensor, *, vocab_size: int) -> None:
        """Argmax ids on the current stream, then make ``stream`` current."""
        self.ids = self._exchange(local_logits, vocab_size=vocab_size)
        self._main = torch.cuda.current_stream()
        self.stream.wait_stream(self._main)
        torch.cuda.set_stream(self.stream)

    def unfork(self) -> None:
        """Back to the main stream; the logits tail ran on ``stream``."""
        torch.cuda.set_stream(self._main)

    def join(self) -> None:
        torch.cuda.current_stream().wait_stream(self.stream)
        self.ids = None


def make_local_argmax_sink(
    *, max_rows: int, vocab_size: int, device, timeout_s: float
) -> Optional[LocalArgmaxSink]:
    """Build the sink on every TP rank or on none (reason logged). Every TP rank must
    call it: it runs CPU collectives."""
    from sglang.srt.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    world, rank = tp_group.world_size, tp_group.rank_in_group
    if world == 1:
        return None
    group = tp_group.cpu_group
    reason = unsupported_reason(
        world=world,
        vocab_size=vocab_size,
        capability=torch.cuda.get_device_capability(device),
    )
    if not _all_ranks_true(group, reason is None):
        if rank == 0:
            logger.info(
                "DSpark verify argmax exchange not used: %s.",
                reason or "another rank cannot use it",
            )
        return None
    exchange = None
    try:
        exchange = LocalArgmaxExchange(
            max_rows=max_rows,
            group=group,
            rank=rank,
            world=world,
            device=device,
            timeout_s=timeout_s,
        )
    except RuntimeError as error:
        logger.warning("DSpark verify argmax exchange set-up failed: %s", error)
    if not _all_ranks_true(group, exchange is not None):
        logger.warning(
            "DSpark verify argmax exchange not used: set-up failed on a rank."
        )
        return None
    if rank == 0:
        logger.info(
            "DSpark verify argmax exchange installed (%d ranks, %d CTAs per rank, up "
            "to %d rows, waits bounded at %.1f s).",
            world,
            exchange.ctas,
            max_rows,
            timeout_s,
        )
    return LocalArgmaxSink(exchange=exchange, device=device)
