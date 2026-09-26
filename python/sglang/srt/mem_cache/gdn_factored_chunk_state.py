"""Checkpoint-select verify transaction for factored GDN (#ssmon-opus, docs/120).

Default-off (SGLANG_GDN_VERIFY_CHUNK=1).  Verify reads the persistent pool directly (no snapshot copy, no raw-input
recording) and writes compact per-input records; commit is one launch over all layers that rebuilds the accepted
state from the untouched entry plus the records, applies the due r+m -> r cut once, and publishes the request slot
and its radix track slot.  Cut rule is the B-arm rule (no cut inside the verify window), gated by KL/GSM (#753).
"""
import os

import torch

from .gdn_factored_spec import FactorVerifyTicket


class FactoredGDNChunkState:
    names = ("a", "U", "W", "count")
    replay_inputs = True  # gdn_backend routes verify through forward_layer
    verify_window_fused = False
    batched_commit = False
    variant = "chunk"

    def __init__(self, pool, max_batch_size, draft_tokens):
        from sglang.srt.layers.attention.linear.kernels.gdn_factored_chunk import allocate_records
        if max_batch_size < 1 or draft_tokens != 4 or (pool.cfg.r, pool.cfg.m) != (8, 8) or pool.U.shape[-2] != 16:
            raise ValueError("chunk verify requires r8/W8 (RMAX 16) and NEXTN 3/1/4")
        self.pool = pool
        self.capacity = max_batch_size
        self.draft_tokens = draft_tokens
        self.epoch = 0
        self.current = None
        layers, _, hv, _, k = pool.U.shape
        device = pool.a.device
        self.records = allocate_records(layers, max_batch_size, draft_tokens, hv, k, pool.W.shape[-1], device)
        self.generations = torch.zeros(pool.a.shape[1], dtype=torch.int64, device=device)
        self.row_ids = torch.arange(max_batch_size, dtype=torch.int64, device=device)
        self.work_indices = torch.full_like(self.row_ids, -1)  # entry slot per verify row (graph-fixed address)
        self.layer_arguments = [None] * layers
        self.checks = os.environ.get("SGLANG_GDN_CHUNK_CHECKS", "0") == "1"
        # Optional: run the commit on a side stream so it overlaps the MTP draft forward (the draft never touches this
        # pool).  Every later reader/writer of factor slots joins first: the next target forward, snapshot_commit and
        # the pool's slot-level methods (wrapped in FactoredGDNPool).
        side = os.environ.get("SGLANG_GDN_CHUNK_COMMIT_STREAM", "0") == "1" and device.type == "cuda"
        self.side = torch.cuda.Stream(device=device) if side else None
        self.done = torch.cuda.Event() if side else None
        self.recorded = False
        from sglang.srt.layers.attention.linear.kernels.gdn_factored_chunk import MODE
        self.variant = "chunk-" + MODE + ("-stream" if side else "")

    def join(self):
        # Waiting on an already-completed event is free; every stream that touches factor slots must wait once.
        if self.recorded:
            torch.cuda.current_stream().wait_event(self.done)

    def bytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (*self.records.values(), self.generations, self.row_ids, self.work_indices))

    def invalidate_slots(self, slots):
        self.join()
        if slots.numel():
            self.generations[slots.long()] += 1

    def snapshot_commit(self, slots):
        """Outside graph replay: only the entry-slot row map is written; the pool itself is the checkpoint."""
        if self.current is not None and not self.current.closed:
            raise RuntimeError("uncommitted factor verify transaction")
        self.join()
        if slots.ndim != 1 or not 0 < slots.numel() <= self.capacity:
            raise ValueError("invalid factor verify batch shape")
        slots = slots.long()
        n = slots.numel()
        if self.checks:
            torch._assert_async(torch.all((slots >= 0) & (slots < self.pool.a.shape[1])), "invalid factor slot")
            ordered = torch.sort(slots).values
            torch._assert_async(torch.all(ordered[1:] != ordered[:-1]), "duplicate factor slots")
        self.work_indices[n:].fill_(-1)
        self.work_indices[:n].copy_(slots)
        self.epoch += 1
        ticket = FactorVerifyTicket(self.epoch, slots.clone(), self.generations[slots].clone())
        self.current = ticket
        return ticket

    def forward_layer(self, layer, mixed_qkv, a, b):
        from sglang.srt.layers.attention.linear.kernels.gdn_factored_chunk import verify as chunk_verify
        tokens = self.draft_tokens
        batch = mixed_qkv.shape[0] // tokens
        if mixed_qkv.shape[0] != batch * tokens or not 0 < batch <= self.capacity:
            raise ValueError("chunk verify requires fixed four-input verify rows")
        li = self.pool.layer_index(layer.layer_id)
        if self.layer_arguments[li] is None:
            self.layer_arguments[li] = (layer.A_log, layer.dt_bias)
        output = chunk_verify(
            mixed_qkv.reshape(batch, tokens, -1), a.reshape(batch, tokens, -1), b.reshape(batch, tokens, -1),
            A_log=layer.A_log, dt_bias=layer.dt_bias, vbar=self.pool.vbar[li],
            pa=self.pool.a[li], pu=self.pool.U[li], pw=self.pool.W[li], pcount=self.pool.count[li],
            indices=self.work_indices[:batch], records=self.records, layer=li,
            scale=layer.head_k_dim ** -0.5, num_q_heads=layer.num_q_heads)
        return output.reshape(1, batch * tokens, layer.num_v_heads, layer.head_v_dim)

    def commit(self, ticket, last_consumed_indices, *, track_slots=None, track_steps=None):
        """Indices refer to consumed target inputs (0..3), not accepted-draft counts."""
        from sglang.srt.layers.attention.linear.kernels.gdn_factored_chunk import commit_select
        if ticket is not self.current or ticket.closed or ticket.epoch != self.epoch:
            raise RuntimeError("closed or stale factor verify transaction")
        steps = last_consumed_indices
        if steps.shape != ticket.slots.shape:
            raise ValueError("factor commit shape differs from snapshot")
        if (track_slots is None) != (track_steps is None):
            raise ValueError("incomplete factor tracking coordinates")
        if self.checks:
            torch._assert_async(torch.all(self.generations[ticket.slots] == ticket.generations),
                                "factor slot reused during verify")
            torch._assert_async(torch.all((steps >= 0) & (steps < self.draft_tokens)), "invalid accepted input index")
            if track_slots is not None:
                torch._assert_async(torch.all((track_steps < 0) | (track_steps <= steps)),
                                    "tracking beyond accepted prefix")
        cfg = self.pool.cfg

        def launch():
            commit_select(self.pool, self.records, self.work_indices[:steps.numel()], steps.to(torch.int32),
                          None if track_slots is None else track_slots.to(torch.int64),
                          None if track_steps is None else track_steps.to(torch.int32),
                          r=cfg.r, rfull=cfg.rfull)
            if ticket.slots.numel():
                self.generations[ticket.slots] += 1

        if self.side is None:
            launch()
        else:
            self.side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.side):
                launch()
            for t in (steps, ticket.slots, track_slots, track_steps):
                if t is not None:
                    t.record_stream(self.side)
            self.done.record(self.side)
            self.recorded = True
        ticket.closed = True

    def rollback(self, ticket):
        if ticket is not self.current or ticket.closed:
            raise RuntimeError("closed or stale factor rollback")
        ticket.closed = True
