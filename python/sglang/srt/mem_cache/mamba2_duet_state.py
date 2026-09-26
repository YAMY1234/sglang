"""Eager DUET Mamba-2 state policy.

The native FP32 pool remains the recurrence workspace. At a prune boundary its
contents become x_bar a^T + rank-r content, using the release reference callable.
Keeping a dense workspace deliberately trades memory for straightforward parity.
No truncation is performed inside a prompt scan. Warm starts and decode clocks
belong to physical request slots, never positions in a changing scheduler batch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class Projection:
    coefficient: torch.Tensor
    right_basis: torch.Tensor | None


class DuetMamba2State:
    def __init__(self, sink_dir: torch.Tensor, rank: int, every: int,
                 truncate_rank: Callable):
        if sink_dir.ndim != 3 or rank <= 0 or every <= 0:
            raise ValueError("Expected [layer, head, input] sink and positive rank/cadence")
        self.sink_dir = sink_dir.float()
        self.rank = int(rank)
        self.every = int(every)
        self.truncate_rank = truncate_rank
        self.steps: dict[int, int] = {}
        self.projections: dict[tuple[int, int], Projection] = {}

    def reset(self, slots: list[int]) -> None:
        """Call on allocation/cold prefill, including a recycled native pool slot."""
        self._validate_slots(slots)
        selected = set(slots)
        for key in list(self.projections):
            if key[1] in selected:
                del self.projections[key]
        for slot in slots:
            self.steps[slot] = 0

    @staticmethod
    def _validate_slots(slots):
        if len(slots) != len(set(slots)) or any(s < 0 for s in slots):
            raise ValueError("DUET requires distinct valid physical Mamba slots")

    def advance(self, slots: list[int]) -> list[int]:
        """Advance once after a full decode, including the prompt boundary token."""
        self._validate_slots(slots)
        if any(slot not in self.steps for slot in slots):
            raise RuntimeError("Decode state has no DUET prompt clock")
        due = []
        for slot in slots:
            self.steps[slot] += 1
            if self.steps[slot] % self.every == 0:
                due.append(slot)
        return due

    @torch.no_grad()
    def project(self, layer: int, pool: torch.Tensor, slots: list[int], *, warm: bool):
        self._validate_slots(slots)
        if pool.dtype != torch.float32:
            raise ValueError("DUET Mamba-2 recurrence requires an FP32 temporal pool")
        direction = self.sink_dir[layer].to(pool.device)
        if tuple(pool.shape[1:3]) != tuple(direction.shape):
            raise ValueError("Mamba-2 pool must use [slot, head, input, state] layout")
        norm2 = (direction * direction).sum(-1).clamp_min(1e-12)
        # B=1 matches a per-request reference independently of co-batching/order.
        for slot in slots:
            state = pool[slot:slot + 1]
            a = torch.einsum("bhpn,hp->bhn", state, direction) / norm2[None, :, None]
            sink = direction[None, :, :, None] * a[:, :, None, :]
            previous = self.projections.get((layer, slot)) if warm else None
            content, basis = self.truncate_rank(
                state - sink, self.rank,
                prev=previous.right_basis if previous is not None else None,
                detach_basis=True,
            )
            state.copy_(sink + content)
            self.projections[layer, slot] = Projection(a, basis)
