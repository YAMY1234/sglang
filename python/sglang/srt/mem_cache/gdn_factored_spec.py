"""Isolated NEXTN factor transactions (TwinStar stage 2, directive 427).

The persistent pool remains the checkpoint throughout verify. Working factors
use the ordinary W8 recurrence/cut; each post-input state is saved separately.
Only commit publishes a selected prefix. Conv/PLE and KV use their respective
transaction owners at the same native verify-commit hook.
"""
from dataclasses import dataclass
import os

import torch


@dataclass
class FactorVerifyTicket:
    epoch: int
    slots: torch.Tensor
    generations: torch.Tensor
    closed: bool = False


class FactoredGDNVerifyState:
    names = ("a", "U", "W", "count")

    def __init__(self, pool, max_batch_size: int, draft_tokens: int, *, direct_checkpoints=None,
                 _checkpoint_storage=True):
        if max_batch_size < 1 or draft_tokens != 4 or (pool.cfg.r, pool.cfg.m) != (8, 8):
            raise ValueError("factor verify currently requires r8/W8 and NEXTN 3/1/4")
        self.pool = pool
        self.capacity = max_batch_size
        self.draft_tokens = draft_tokens
        self.epoch = 0
        self.current = None
        self.direct_checkpoints = (os.environ.get("SGLANG_GDN_VERIFY_DIRECT_CHECKPOINT", "0") == "1"
                                   if direct_checkpoints is None else direct_checkpoints)
        self.working, self.checkpoints = {}, {}
        for name in self.names:
            src = getattr(pool, name)
            shape = (src.shape[0], max_batch_size, *src.shape[2:])
            self.working[name] = torch.zeros(shape, dtype=src.dtype, device=src.device)
            if not _checkpoint_storage:
                continue
            if self.direct_checkpoints:
                # A candidate's complete batch is contiguous for the unchanged
                # W8 kernel. Commit already honors request/step strides.
                checkpoint = torch.zeros((shape[0], draft_tokens, max_batch_size, *shape[2:]),
                                         dtype=src.dtype, device=src.device)
                self.checkpoints[name] = checkpoint.transpose(1, 2)
            else:
                self.checkpoints[name] = torch.zeros(
                    (shape[0], max_batch_size, draft_tokens, *shape[2:]), dtype=src.dtype, device=src.device)
        self.working["count"].fill_(pool.cfg.r)
        device = pool.a.device
        self.generations = torch.zeros(pool.a.shape[1], dtype=torch.int64, device=device)
        self.row_ids = torch.arange(max_batch_size, dtype=torch.int64, device=device)
        self.work_indices = torch.full_like(self.row_ids, -1)
        self.stale = torch.ones(max_batch_size, dtype=torch.int32, device=device)
        self.written = torch.zeros((pool.a.shape[0], max_batch_size, draft_tokens), dtype=torch.bool, device=device)
        self.constants = {v: torch.full((1, max_batch_size, draft_tokens, 1), v, dtype=torch.int32, device=device)
                          for v in (-1, 0, 1)}

    def bytes(self):
        tensors = [*self.working.values(), *self.checkpoints.values(), self.generations,
                   self.row_ids, self.work_indices, self.stale, self.written, *self.constants.values()]
        return sum(t.numel()*t.element_size() for t in tensors)

    def invalidate_slots(self, slots):
        if slots.numel():
            # Slot lifecycle calls use unique real indices, outside graph replay.
            self.generations[slots.long()] += 1

    def snapshot_commit(self, slots):
        """Called outside forward graph capture, before eager/graph verify alike."""
        if self.current is not None and not self.current.closed:
            raise RuntimeError("uncommitted factor verify transaction")
        if slots.ndim != 1 or not 0 < slots.numel() <= self.capacity:
            raise ValueError("invalid factor verify batch shape")
        slots = slots.long()
        torch._assert_async(torch.all((slots >= 0) & (slots < self.pool.a.shape[1])), "invalid factor slot")
        # Duplicate slots would make publication ambiguous even for a chain.
        torch._assert_async(torch.all(torch.sort(slots).values[1:] != torch.sort(slots).values[:-1]),
                            "duplicate factor slots")
        n = slots.numel()
        if (self.direct_checkpoints or getattr(self, 'snapshot_kernel', False)) and slots.is_cuda:
            from sglang.srt.layers.attention.linear.kernels.gdn_verify_io import snapshot_factors
            snapshot_factors(self.pool, self.working, slots)
        else:
            for name in self.names:
                self.working[name][:, :n].copy_(getattr(self.pool, name).index_select(1, slots))
        self.work_indices.copy_(torch.where(self.row_ids < n, self.row_ids, -1))
        self.written.zero_()
        self.epoch += 1
        ticket = FactorVerifyTicket(self.epoch, slots.clone(), self.generations[slots].clone())
        self.current = ticket
        return ticket

    def record_step(self, layer_index, step, batch_size):
        """Forward/graph body: save the state *after* its original W8 cut."""
        if not 0 <= step < self.draft_tokens or not 0 < batch_size <= self.capacity:
            raise ValueError("invalid candidate state coordinate")
        for name in self.names:
            self.checkpoints[name][layer_index, :batch_size, step].copy_(
                self.working[name][layer_index, :batch_size])
        self.written[layer_index, :batch_size, step].fill_(True)

    def _validate(self, ticket, steps):
        if ticket is not self.current or ticket.closed or ticket.epoch != self.epoch:
            raise RuntimeError("closed or stale factor verify transaction")
        if steps.shape != ticket.slots.shape:
            raise ValueError("factor commit shape differs from snapshot")
        torch._assert_async(torch.all(self.generations[ticket.slots] == ticket.generations),
                            "factor slot reused during verify")
        torch._assert_async(torch.all((steps >= 0) & (steps < self.draft_tokens)), "invalid accepted input index")
        rows = self.row_ids[:steps.numel()]
        torch._assert_async(torch.all(self.written[:, rows, steps.long()]), "candidate state not recorded")

    @staticmethod
    def _scatter(dst, src, slots, steps):
        if dst.is_cuda:
            from sglang.kernels.ops.mamba.mamba_state_scatter_triton import fused_mamba_state_scatter_with_mask
            fused_mamba_state_scatter_with_mask(dst, src, slots, steps)
        else:
            rows = torch.arange(steps.numel(), device=steps.device)
            valid = (slots >= 0) & (steps >= 0)
            dst[:, slots[valid].long()] = src[:, rows[valid], steps[valid].long()]

    def _publish(self, slots, steps):
        for name in self.names:
            self._scatter(getattr(self.pool, name), self.checkpoints[name], slots, steps)
        # Never invalidate slot zero merely because a padded/track row is -1.
        for target, value in ((self.pool.stale, 1), (self.pool.dense_of, -1),
                              (self.pool.dense_required, 0), (self.pool.prefix_valid, 0)):
            if target is not None:
                self._scatter(target.view(1, -1, 1), self.constants[value], slots, steps)

    def commit(self, ticket, last_consumed_indices, *, track_slots=None, track_steps=None):
        """Indices refer to consumed target inputs, not accepted-draft counts."""
        steps = last_consumed_indices.long()
        self._validate(ticket, steps)
        if track_slots is not None:
            if track_steps is None or track_slots.shape != steps.shape or track_steps.shape != steps.shape:
                raise ValueError("incomplete factor tracking coordinates")
            torch._assert_async(torch.all((track_steps < 0) | ((track_slots >= 0) &
                                (track_slots < self.pool.a.shape[1]) & (track_steps <= steps))),
                                "tracking beyond accepted prefix")
            rows = self.row_ids[:steps.numel()]
            safe_steps = track_steps.long().clamp_min(0)
            torch._assert_async(torch.all((track_steps < 0)[None, :] | self.written[:, rows, safe_steps]),
                                "tracking state not recorded")
            self._publish(track_slots, track_steps)
        self._publish(ticket.slots, steps)
        self.invalidate_slots(ticket.slots)
        ticket.closed = True

    def rollback(self, ticket):
        if ticket is not self.current or ticket.closed:
            raise RuntimeError("closed or stale factor rollback")
        # Persistent factors and metadata were never touched by candidates.
        ticket.closed = True
