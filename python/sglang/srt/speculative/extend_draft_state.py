"""Request-generation ownership for deferred MTP extend inputs."""

from dataclasses import dataclass

import torch


@dataclass
class PendingExtendRows:
    slots_cpu: torch.Tensor
    valid_cpu: torch.Tensor
    indices: torch.Tensor
    hidden: torch.Tensor
    tokens: torch.Tensor
    cache: torch.Tensor
    seq_lens: torch.Tensor
    accept_lens: torch.Tensor


class PendingExtendStore:
    def __init__(self, pool, width, hidden_size, dtype, device):
        self.pool = pool
        self.width = width
        count = pool.req_generation.numel()
        self.generations = torch.full((count,), -1, dtype=torch.int64)
        self.generation_epoch = getattr(pool, "req_generation_epoch", 0)
        self.hidden = torch.zeros(
            (count, width, hidden_size), dtype=dtype, device=device
        )
        self.tokens = torch.zeros((count, width), dtype=torch.int64, device=device)
        self.cache = torch.zeros_like(self.tokens)
        self.seq_lens = torch.zeros(count, dtype=torch.int64, device=device)
        self.accept_lens = torch.ones(count, dtype=torch.int32, device=device)
        self.stored_rows = self.consumed_rows = 0

    def ready(self, slots_cpu):
        epoch = getattr(self.pool, "req_generation_epoch", 0)
        if epoch != self.generation_epoch:
            self.generations.fill_(-1)
            self.generation_epoch = epoch
        if slots_cpu is None:
            return torch.empty(0, dtype=torch.bool)
        return (slots_cpu != 0) & (
            self.generations[slots_cpu] == self.pool.req_generation[slots_cpu]
        )

    def put(self, slots_cpu, slots, hidden, tokens, cache, seq_lens, accept_lens):
        n = slots_cpu.numel()
        if n == 0:
            return
        if self.ready(slots_cpu).any():
            raise RuntimeError("pending extend overwritten before consumption")
        if (slots_cpu == 0).any() or slots_cpu.unique().numel() != n:
            raise RuntimeError("pending extend requires unique live request slots")
        # All copies and consumption run on the same forward stream, in order.
        self.hidden.index_copy_(0, slots, hidden.reshape(n, self.width, -1))
        self.tokens.index_copy_(0, slots, tokens.reshape(n, self.width).to(torch.int64))
        accepted = (
            torch.arange(self.width, device=slots.device)[None, :]
            < accept_lens[:, None]
        )
        # Rejected slots may be freed before delayed extend; never write them again.
        live_cache = torch.where(accepted, cache.reshape(n, self.width), 0)
        self.cache.index_copy_(0, slots, live_cache.to(torch.int64))
        self.seq_lens.index_copy_(0, slots, seq_lens.to(torch.int64))
        self.accept_lens.index_copy_(0, slots, accept_lens.to(torch.int32))
        self.generations[slots_cpu] = self.pool.req_generation[slots_cpu]
        self.stored_rows += n

    def take(self, slots_cpu, slots_gpu):
        valid = self.ready(slots_cpu)
        # Reuse scheduler indices; a blocking H2D here stalls the forward stream.
        indices = slots_gpu.long()
        if not valid.all():
            indices = torch.where(
                valid.to(indices.device, non_blocking=True), indices, 0
            )
        return PendingExtendRows(
            slots_cpu=slots_cpu,
            valid_cpu=valid,
            indices=indices,
            hidden=self.hidden[indices].flatten(0, 1),
            tokens=self.tokens[indices].flatten(),
            cache=self.cache[indices].flatten(),
            seq_lens=self.seq_lens[indices],
            accept_lens=self.accept_lens[indices],
        )

    def consumed(self, rows):
        slots = rows.slots_cpu[rows.valid_cpu]
        self.generations[slots] = -1
        self.consumed_rows += slots.numel()

    def discard(self, slots_cpu):
        if slots_cpu is not None:
            self.generations[slots_cpu] = -1
