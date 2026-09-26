"""Opt-in graph of whole-layer AGG factorization, stores and publication.

The original factorization and per-layer store kernels retain their order.
Warmup/capture use negative slots and a disabled metadata publisher; only a
bound replay can modify the persistent pool. Slot generations/final radix
copies remain in the caller. No model-forward graph is captured here.
"""
from dataclasses import replace
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _publish_prefill_metadata(SLOTS, TRACK, REQUIRED, PREFIX, DENSE_REQUIRED,
                              ACTIVE, B: tl.constexpr, T: tl.constexpr,
                              HAS_PREFIX: tl.constexpr, HAS_REQUIRED: tl.constexpr):
    if tl.load(ACTIVE) == 0:
        return
    row = tl.program_id(0)
    if row < B:
        slot = tl.maximum(tl.load(SLOTS + row).to(tl.int64), 0)
        if HAS_PREFIX:
            tl.store(PREFIX + slot, 1)
        if HAS_REQUIRED:
            tl.store(DENSE_REQUIRED + slot, tl.load(REQUIRED + row))
    if HAS_PREFIX and row < T:
        slot = tl.maximum(tl.load(TRACK + row).to(tl.int64), 0)
        tl.store(PREFIX + slot, 1)


class CommitBuffers:
    def __init__(self, pool, plan, track_slots):
        self.pool, self.cfg = pool, replace(pool.cfg)
        self.dense = torch.stack([x[0] for x in plan.pending])
        self.states = tuple(self.dense.unbind(0))
        self.tracked = (torch.stack([x[1] for x in plan.pending])
                        if track_slots is not None else None)
        self.track_states = tuple(self.tracked.unbind(0)) if self.tracked is not None else None
        self.slots = torch.full_like(plan.slots, -1)
        self.ring_dst = torch.full_like(plan.ring_dst, -1)
        self.track_slots = torch.full_like(track_slots, -1) if track_slots is not None else None
        self.required = (plan.dense_required_after_commit.clone()
                         if pool.dense_required is not None else None)
        self.active = torch.zeros((), dtype=torch.int32, device=plan.slots.device)
        self.omega = self.probe(self.states[0])
        self.track_omega = self.probe(self.track_states[0]) if self.track_states is not None else None

    def probe(self, first):
        b, h, v, _ = first.shape
        generator = torch.Generator(device=first.device).manual_seed(0)
        return torch.randn(b, h, v, self.cfg.r+self.cfg.init_oversample,
                           device=first.device, generator=generator)

    def bind(self, plan, track_slots):
        torch.stack([x[0] for x in plan.pending], out=self.dense)
        self.slots.copy_(plan.slots)
        self.ring_dst.copy_(plan.ring_dst)
        if self.tracked is not None:
            torch.stack([x[1] for x in plan.pending], out=self.tracked)
            self.track_slots.copy_(track_slots)
        if self.required is not None:
            self.required.copy_(plan.dense_required_after_commit)
        self.active.fill_(1)

    def evaluate(self, factorize):
        from sglang.srt.layers.attention.linear.kernels.gdn_factored_io import store_factored
        p = self.pool
        factors = factorize(self.states, p.vbar, self.cfg, omega=self.omega)
        tracked = (factorize(self.track_states, p.vbar, self.cfg, omega=self.track_omega)
                   if self.track_states is not None else None)
        for i in range(len(self.states)):
            store_factored(*factors[i], p.a[i], p.U[i], p.W[i], p.count[i],
                p.stale, p.dense_of, self.slots, self.cfg.r, stale_value=0,
                dense=self.states[i], ring=p.dense_ring[i], ring_dst=self.ring_dst)
            if tracked is not None:
                store_factored(*tracked[i], p.a[i], p.U[i], p.W[i], p.count[i],
                    p.stale, p.dense_of, self.track_slots, self.cfg.r, stale_value=1)
        b = self.slots.numel()
        t = self.track_slots.numel() if self.track_slots is not None else 0
        _publish_prefill_metadata[(max(b, t),)](
            self.slots, self.track_slots if t else self.slots,
            self.required if self.required is not None else self.slots,
            p.prefix_valid if p.prefix_valid is not None else p.stale,
            p.dense_required if p.dense_required is not None else p.stale,
            self.active, b, t, p.prefix_valid is not None,
            p.dense_required is not None, num_warps=1)
        # Retain outputs and captured workspace ownership, even though the
        # caller only observes the already-published persistent state.
        return factors, tracked


class PrefillCommitGraph:
    MAX_INPUT_BYTES = 128 << 20  # admits the AGG 56.6 MiB whole-layer budget
    MAX_ENTRIES = 2

    def __init__(self):
        self.entries = {}
        self.stats = dict(captured=0, replayed=0, fallback=0, input_bytes=0,
                          retained_allocated_bytes=0)

    def run(self, pool, plan, track_slots, *, factorize, policy, replay_stream=None):
        states = [x[0] for x in plan.pending]
        tensors = [x for row in plan.pending for x in row if x is not None]
        size = sum(x.numel()*x.element_size() for x in tensors)+pool.vbar.nbytes
        # This path publishes prefix validity only after ALL relevant layers.
        if (not states[0].is_cuda or torch.cuda.is_current_stream_capturing()
                or len(states) != len(pool.layer_ids)
                or pool.prefix_layer_count() != len(pool.layer_ids)
                or not pool.batch_prefill_final_copy or pool.prefix_dense is not None
                or size > self.MAX_INPUT_BYTES or states[0].shape[0] > 16):
            self.stats['fallback'] += 1
            return False
        assert (track_slots is None) == (plan.pending[0][1] is None)
        backing = tuple(x.data_ptr() for x in (pool.a, pool.U, pool.W, pool.count,
            pool.stale, pool.dense_of, pool.dense_ring, pool.vbar,
            pool.prefix_valid, pool.dense_required) if x is not None)
        shapes = tuple((tuple(x.shape), x.dtype, x.device) for x in tensors)
        cfg = pool.cfg
        config = (cfg.r, cfg.rmax, cfg.dtype, cfg.init_iters, cfg.init_oversample, cfg.init_method)
        key = (backing, shapes, config, policy, factorize,
               torch.backends.cuda.matmul.allow_tf32,
               torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
               torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= self.MAX_ENTRIES:
                self.stats['fallback'] += 1
                return False
            before = torch.cuda.memory_allocated(states[0].device)
            buffers = CommitBuffers(pool, plan, track_slots)
            current = torch.cuda.current_stream(states[0].device)
            stream = torch.cuda.Stream(device=states[0].device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                buffers.evaluate(factorize)
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = buffers.evaluate(factorize)
            entry = (buffers, graph, outputs, stream)
            self.entries[key] = entry
            self.stats['captured'] += 1
            self.stats['input_bytes'] += size
            self.stats['retained_allocated_bytes'] += max(0, torch.cuda.memory_allocated(states[0].device)-before)
            logger.info('GDN whole-layer prefill commit graph captured: input_bytes=%d stats=%s', size, self.stats)
        entry[0].bind(plan, track_slots)
        if replay_stream is None:
            entry[1].replay()
        else:
            # Bind stays on the forward stream (it reads this forward's dense states); the factorisation replays on
            # the caller's side stream, whose readers (next forward, slot methods) join it first.
            replay_stream.wait_stream(torch.cuda.current_stream(states[0].device))
            with torch.cuda.stream(replay_stream):
                entry[1].replay()
        self.stats['replayed'] += 1
        return True
