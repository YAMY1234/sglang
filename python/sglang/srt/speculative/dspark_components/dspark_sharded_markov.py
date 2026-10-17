# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The folded DSpark draft's Markov chain as one vocab-sharded kernel per rank.

Developed at NVIDIA. This module owns what the kernel
(sglang.kernels.ops.speculative.dspark.sharded_markov) needs from the runtime: the
eligibility rule, the symmetric-memory peer buffers, the control words, the noise
buffer and its side stream. Request rows map to themselves (temperature = the row's
step temperature, 0 = argmax); a greedy draft keeps no step logits, so its rows are
passed as inert (argmax, no cache write).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist

from sglang.kernels.ops.speculative.dspark import sharded_markov as kernel

logger = logging.getLogger(__name__)

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))
SUPPORTED_WORLDS = (1, 2, 4, kernel.MAX_RANKS)
# Measured on GB300 TP4: the direct exchange wins at 1 row, the rank-local fold at 2-8.
DIRECT_MAX_ROWS = 1


def unsupported_reason(
    *,
    world: int,
    vocab_size: int,
    num_steps: int,
    embed_weight: torch.Tensor,
    head_weight: torch.Tensor,
    cache_dtype: torch.dtype,
    capability: tuple[int, int],
) -> Optional[str]:
    """Why the chain kernel cannot serve this configuration, or None; the same on every rank."""
    if world not in SUPPORTED_WORLDS:
        return f"a tensor-parallel size of {world}"
    if tuple(capability) not in SUPPORTED_CAPABILITIES:
        return f"compute capability {capability} (needs SM100 or SM103)"
    if not 1 <= num_steps <= kernel.MAX_STEPS:
        return f"{num_steps} draft positions"
    if vocab_size > kernel.MAX_VOCAB or vocab_size % (16 * world) != 0:
        return f"a vocabulary of {vocab_size} for {world} ranks"
    if cache_dtype not in kernel.LOGIT_DTYPES:
        return f"a {cache_dtype} step-logits buffer"
    for name, weight in (("embedding", embed_weight), ("head", head_weight)):
        if (
            weight.dim() != 2
            or weight.shape[1] != kernel.RANK_DIM
            or weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
        ):
            return f"a Markov {name} that is not contiguous BF16 [*, {kernel.RANK_DIM}]"
    if head_weight.shape[0] < vocab_size or embed_weight.shape[0] < vocab_size:
        return "a Markov table smaller than the vocabulary"
    return None


def narrow_row_limit(*, world: int, max_bs: int) -> int:
    """Request rows of the narrow kernel: its exchange gives each (row, rank) pair one lane."""
    return max(1, min(kernel.MAX_ROWS, 32 // world, max_bs))


def _all_ranks_true(group, value: bool) -> bool:
    """``value`` on every rank of the CPU ``group`` (None: this rank only)."""
    if group is None:
        return value
    flag = torch.tensor([int(value)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


class DSparkShardedMarkovSampler:
    """Persistent buffers, peer mappings and launchers of the sharded chain."""

    def __init__(
        self,
        *,
        embed_weight: torch.Tensor,
        head_weight: torch.Tensor,
        vocab_size: int,
        num_steps: int,
        cache_dtype: torch.dtype,
        group,
        rank: int,
        world: int,
        max_bs: int,
        timeout_s: float,
        keeps_logits: bool,
    ):
        device = embed_weight.device
        reason = unsupported_reason(
            world=world,
            vocab_size=vocab_size,
            num_steps=num_steps,
            embed_weight=embed_weight,
            head_weight=head_weight,
            cache_dtype=cache_dtype,
            capability=torch.cuda.get_device_capability(device),
        )
        if reason is not None:
            raise ValueError(f"sharded Markov sampling does not support {reason}")
        if timeout_s <= 0:
            raise ValueError(
                f"sharded Markov sampling needs a deadline, got {timeout_s}"
            )
        self.group, self.rank, self.world = group, rank, world
        self.device = device
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.cache_dtype = cache_dtype
        self.timeout_ns = int(timeout_s * 1e9)
        self.vocab_local = vocab_size // world
        self.vocab_offset = rank * self.vocab_local
        self.embed_weight = embed_weight
        self.head_shard = head_weight[
            self.vocab_offset : self.vocab_offset + self.vocab_local
        ]
        tiles = self.vocab_local // 16
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.ctas = min(sms, -(-tiles // kernel.WARPS))
        if self.ctas > kernel.MAX_CTAS:
            raise ValueError(
                f"the Markov chain counts launches for up to {kernel.MAX_CTAS} CTAs, not {self.ctas}"
            )
        self.narrow_rows = narrow_row_limit(world=world, max_bs=max_bs)
        # The wide kernel takes more rows when every warp owns one W2 tile.
        wide = max_bs > self.narrow_rows and self.ctas * kernel.WARPS >= tiles
        self.wide_rows = min(kernel.WIDE_ROWS, max_bs) if wide else 0
        self.max_rows = max(self.narrow_rows, self.wide_rows)
        self.noise = torch.empty(
            self.max_rows * num_steps,
            self.vocab_local,
            dtype=torch.float32,
            device=device,
        )
        self.control = torch.zeros(
            kernel.CONTROL_WORDS, dtype=torch.int32, device=device
        )
        # Request row r reads temperature[r] and keeps its step logits at cache[r].
        rows = torch.arange(self.max_rows, dtype=torch.int32, device=device)
        self.sampled_rows = rows.repeat_interleave(num_steps)
        self.inert_rows = torch.full_like(self.sampled_rows, -1)
        self.anchor = torch.zeros(self.max_rows, dtype=torch.int32, device=device)
        size = self._layout_peer_buffer(keeps_logits=keeps_logits)
        self.wide_control = (
            torch.zeros(kernel.WIDE_CONTROL_WORDS, dtype=torch.int32, device=device)
            if self.wide_rows
            else None
        )
        # Compile before the collective set-up so a build failure cannot leave peers waiting.
        failure = None
        try:
            self._check_coresidency()
        except (RuntimeError, ImportError) as error:
            failure = error
        if not _all_ranks_true(group, failure is None):
            raise RuntimeError(
                f"kernel set-up failed on a rank: {failure or 'another rank'}"
            )
        self._peer_buffer, peer_pointers = self._allocate_peer_buffer(size)
        self.peers = torch.tensor(peer_pointers, dtype=torch.int64, device=device)
        self._noise_stream = torch.cuda.Stream(device)

    def _layout_peer_buffer(self, *, keeps_logits: bool) -> int:
        """Byte size of the peer buffer; sets the direct mailbox and wide-kernel offsets.

        Fold records, then staging logits [exchange 2][parity 2][narrow rows] (none for a
        greedy draft, which never writes them), then the direct mailbox, then the wide
        kernel's records and staging logits [parity 2][wide rows].
        """
        row_bytes = self.num_steps * self.vocab_size * self.cache_dtype.itemsize
        if not keeps_logits:
            row_bytes = 0
        size = kernel.STAGING_OFFSET + 4 * self.narrow_rows * row_bytes
        self.mailbox_offset = size
        size += kernel.direct_mailbox_bytes(
            steps=self.num_steps,
            world=self.world,
            ctas=self.ctas,
            max_rows=self.narrow_rows,
        )
        self.wide_offset = -(-size // 16) * 16
        if self.wide_rows:
            size = self.wide_offset + kernel.WIDE_RECORD_BYTES
            size += 2 * self.wide_rows * row_bytes
        return size

    def _kernel_builds(self) -> list[tuple[torch.dtype, bool, bool]]:
        """(base dtype, direct, wide) of every kernel a launch may pick."""
        exchanges = {rows <= DIRECT_MAX_ROWS for rows in range(1, self.narrow_rows + 1)}
        builds = [(direct, False) for direct in sorted(exchanges)]
        if self.wide_rows:
            builds.append((False, True))
        return [
            (base_dtype, direct, wide)
            for base_dtype in kernel.LOGIT_DTYPES
            for direct, wide in builds
        ]

    def _check_coresidency(self) -> None:
        builds = self._kernel_builds()
        kernel.precompile(
            steps=self.num_steps, cache_dtype=self.cache_dtype, builds=builds
        )
        for base_dtype, direct, wide in builds:
            max_ctas = kernel.max_coresident_ctas(
                device=self.device,
                steps=self.num_steps,
                base_dtype=base_dtype,
                cache_dtype=self.cache_dtype,
                direct=direct,
                wide=wide,
            )
            if self.ctas > max_ctas:
                raise RuntimeError(
                    f"sharded Markov sampling needs {self.ctas} co-resident CTAs; "
                    f"the device holds {max_ctas}"
                )

    def _allocate_peer_buffer(self, size: int) -> tuple[torch.Tensor, list[int]]:
        if self.group is None:
            buffer = torch.zeros(size, dtype=torch.uint8, device=self.device)
            return buffer, [buffer.data_ptr()]
        import torch.distributed._symmetric_memory as symm_mem

        buffer = symm_mem.empty(size, dtype=torch.uint8, device=self.device)
        handle = symm_mem.rendezvous(buffer, self.group.group_name)
        pointers = [
            handle.get_buffer(peer, (size,), torch.uint8, storage_offset=0).data_ptr()
            for peer in range(self.world)
        ]
        buffer.zero_()
        self._handle = handle
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        return buffer, pointers

    def supports(self, base_logits: torch.Tensor) -> bool:
        """``[rows, num_steps, vocab or vocab // world]`` base logits the chain takes; depends
        only on shapes, which tensor parallelism keeps identical across ranks."""
        return (
            base_logits.dim() == 3
            and 1 <= base_logits.shape[0] <= self.max_rows
            and base_logits.shape[1] == self.num_steps
            and base_logits.shape[2] in (self.vocab_size, self.vocab_local)
            and base_logits.dtype in kernel.LOGIT_DTYPES
            and base_logits.stride(2) == 1
        )

    def launch_noise(self, *, rows: int, key: int, offset: torch.Tensor) -> None:
        kernel.launch_shard_noise(
            noise=self.noise,
            offset=offset,
            key=key,
            vocab_start=self.vocab_offset,
            vocab_local=self.vocab_local,
            rows=rows,
            steps=self.num_steps,
        )

    def fork_noise(self, *, rows: int, key: int, offset: torch.Tensor) -> None:
        """Draw the noise on a side stream; join_noise must precede sample()."""
        self._noise_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._noise_stream):
            self.launch_noise(rows=rows, key=key, offset=offset)

    def join_noise(self) -> None:
        torch.cuda.current_stream(self.device).wait_stream(self._noise_stream)

    def sample(
        self,
        *,
        base_logits: torch.Tensor,
        anchor_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        logits_cache: Optional[torch.Tensor],
        draft_tokens: torch.Tensor,
    ) -> None:
        """Draft tokens of every row into ``draft_tokens`` [rows, num_steps] and, when
        ``logits_cache`` [>= rows, num_steps, vocab] is given, the full-vocab step logits
        of every row at a nonzero temperature (temperature-0 rows are left untouched);
        without it every row is argmaxed. The noise must be drawn already."""
        rows = base_logits.shape[0]
        assert self.supports(base_logits)
        assert temperatures.dtype == torch.float32
        assert draft_tokens.dtype == torch.int64 and draft_tokens.stride(1) == 1
        wide = rows > self.narrow_rows
        self.anchor[:rows].copy_(anchor_tokens)
        if logits_cache is None:
            idx_mapping, cache, stride0, stride1 = self.inert_rows, self.noise, 0, 0
        else:
            assert logits_cache.dtype == self.cache_dtype
            assert logits_cache.size(2) >= self.vocab_size
            assert logits_cache.stride(2) == 1 and logits_cache.data_ptr() % 16 == 0
            idx_mapping, cache = self.sampled_rows, logits_cache
            stride0, stride1 = logits_cache.stride(0), logits_cache.stride(1)
        full_vocab = base_logits.shape[2] == self.vocab_size
        kernel.dspark_sharded_markov_sample(
            base_logits=base_logits,
            embed_table=self.embed_weight,
            head_shard=self.head_shard,
            anchor=self.anchor,
            idx_mapping=idx_mapping,
            temperatures=temperatures,
            noise=self.noise,
            logits_cache=cache,
            cache_stride0=stride0,
            cache_stride1=stride1,
            draft_tokens=draft_tokens,
            peers=self.peers,
            control=self.wide_control if wide else self.control,
            base_column=self.vocab_offset if full_vocab else 0,
            rank=self.rank,
            world=self.world,
            vocab_local=self.vocab_local,
            rows=rows,
            max_rows=self.wide_rows if wide else self.narrow_rows,
            timeout_ns=self.timeout_ns,
            mailbox_offset=self.wide_offset if wide else self.mailbox_offset,
            ctas=self.ctas,
            cache_dtype=self.cache_dtype,
            # Host-static per row count, so a CUDA graph of one batch size captures one kernel.
            direct=not wide and rows <= DIRECT_MAX_ROWS,
            wide=wide,
        )


def make_sharded_sampler(
    *,
    markov_head,
    gamma: int,
    max_bs: int,
    cache_dtype: torch.dtype,
    keeps_logits: bool,
    timeout_s: float,
) -> Optional[DSparkShardedMarkovSampler]:
    """Build the sampler on every rank of the Markov head's shard group, or on none
    (reason logged). Every rank must call it: it runs CPU collectives."""
    shard = markov_head._tp_shard
    coordinator = markov_head._shard_group
    group = (
        None
        if coordinator is None or coordinator.world_size == 1
        else coordinator.cpu_group
    )
    world = 1 if group is None else coordinator.world_size
    rank = 0 if group is None else coordinator.rank_in_group
    vocab = int(markov_head.vocab_size)
    embed = markov_head.markov_w1.weight
    head = markov_head.markov_w2.weight
    reason = unsupported_reason(
        world=world,
        vocab_size=vocab,
        num_steps=gamma,
        embed_weight=embed,
        head_weight=head,
        cache_dtype=cache_dtype,
        capability=torch.cuda.get_device_capability(embed.device),
    )
    # The kernel's vocab shard must be the LM head's (its local base logits).
    if (
        reason is None
        and shard is not None
        and (
            shard.org_vocab_start != rank * vocab // world
            or shard.num_embeddings_per_partition != vocab // world
        )
    ):
        reason = "an LM-head vocab partition that is not the even split"
    if not _all_ranks_true(group, reason is None):
        if rank == 0:
            logger.info(
                "DSpark sharded Markov chain not used: %s.",
                reason or "another rank cannot use it",
            )
        return None
    sampler = None
    try:
        sampler = DSparkShardedMarkovSampler(
            embed_weight=embed,
            head_weight=head,
            vocab_size=vocab,
            num_steps=gamma,
            cache_dtype=cache_dtype,
            group=group,
            rank=rank,
            world=world,
            max_bs=max_bs,
            timeout_s=timeout_s,
            keeps_logits=keeps_logits,
        )
    except RuntimeError as error:
        logger.warning("DSpark sharded Markov chain set-up failed: %s", error)
    if not _all_ranks_true(group, sampler is not None):
        logger.warning("DSpark sharded Markov chain not used: set-up failed on a rank.")
        return None
    if rank == 0:
        logger.info(
            "DSpark sharded Markov chain installed (%d ranks, %d CTAs per rank, batches "
            "of up to %d request rows, waits bounded at %.1f s).",
            sampler.world,
            sampler.ctas,
            sampler.max_rows,
            timeout_s,
        )
    return sampler
