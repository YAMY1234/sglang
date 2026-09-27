"""Opt-in graph of whole-layer AGG factorization, stores and publication.

The original factorization and per-layer store kernels retain their order.
Warmup/capture use negative slots and a disabled metadata publisher; only a
bound replay can modify the persistent pool. Slot generations/final radix
copies remain in the caller. No model-forward graph is captured here.
"""
from dataclasses import replace
import logging
import os

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
        self.store_layers = os.environ.get('SGLANG_GDN_PREFILL_STORE_LAYERS', '0') == '1'
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
        kwargs = dict(packed=True) if self.store_layers else {}
        factors = factorize(self.states, p.vbar, self.cfg, omega=self.omega, **kwargs)
        tracked = (factorize(self.track_states, p.vbar, self.cfg, omega=self.track_omega, **kwargs)
                   if self.track_states is not None else None)
        if self.store_layers:
            from sglang.srt.layers.attention.linear.kernels.gdn_prefill_store_layers import store_layers
            store_layers(factors, p, self.slots, stale_value=0,
                         dense=self.dense, ring_dst=self.ring_dst)
            if tracked is not None:
                store_layers(tracked, p, self.track_slots, stale_value=1)
        else:
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
        self.fast_entries = {}
        self.fast_lookup = os.environ.get('SGLANG_GDN_PREFILL_COMMIT_LOOKUP', '0') == '1'
        self.stats = dict(captured=0, replayed=0, fallback=0, input_bytes=0,
                          retained_allocated_bytes=0)

    @staticmethod
    def _policy_key(pool, factorize, policy):
        backing = tuple(x.data_ptr() for x in (pool.a, pool.U, pool.W, pool.count,
            pool.stale, pool.dense_of, pool.dense_ring, pool.vbar,
            pool.prefix_valid, pool.dense_required) if x is not None)
        cfg = pool.cfg
        config = (cfg.r, cfg.rmax, cfg.dtype, cfg.init_iters, cfg.init_oversample, cfg.init_method)
        return (backing, config, policy, factorize,
                os.environ.get('SGLANG_GDN_PREFILL_STORE_LAYERS', '0') == '1',
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)

    @classmethod
    def _full_key(cls, pool, tensors, factorize, policy):
        backing, *rest = cls._policy_key(pool, factorize, policy)
        shapes = tuple((tuple(x.shape), x.dtype, x.device) for x in tensors)
        return (backing, shapes, *rest)

    @classmethod
    def _uniform_key(cls, pool, pending, factorize, policy):
        """Compact the original key only after checking every input tensor.

        Unlike a batch-only signature, this preserves all original shape,
        dtype, device, backing-pointer and precision-policy distinctions.
        No tensors or transient state are retained by this metadata cache.
        """
        if not pending:
            return None
        dense, tracked = pending[0]
        shape, dtype, device = dense.shape, dense.dtype, dense.device
        other = ((tracked.shape, tracked.dtype, tracked.device)
                 if tracked is not None else None)
        for a, b in pending:
            if a.shape != shape or a.dtype != dtype or a.device != device:
                return None
            if other is None:
                if b is not None:
                    return None
            elif (b is None or b.shape != other[0] or b.dtype != other[1]
                  or b.device != other[2]):
                return None
        size = len(pending) * (dense.nbytes + (tracked.nbytes if tracked is not None else 0)) + pool.vbar.nbytes
        key = (len(pending), (tuple(shape), dtype, device),
               (tuple(other[0]), other[1], other[2]) if other is not None else None,
               cls._policy_key(pool, factorize, policy))
        return key, size

    @staticmethod
    def _eligible(pool, plan, first, size, maximum):
        return (first.is_cuda and len(plan.pending) == len(pool.layer_ids)
                and pool.prefix_layer_count() == len(pool.layer_ids)
                and pool.batch_prefill_final_copy and pool.prefix_dense is None
                and size <= maximum and first.shape[0] <= 16)

    def run(self, pool, plan, track_slots, *, factorize, policy):
        compact = None
        capturing = torch.cuda.is_current_stream_capturing() if plan.pending[0][0].is_cuda else False
        if self.fast_lookup and not capturing:
            compact = self._uniform_key(pool, plan.pending, factorize, policy)
            if compact is not None:
                signature, size = compact
                if self._eligible(pool, plan, plan.pending[0][0], size, self.MAX_INPUT_BYTES):
                    key = self.fast_entries.get(signature)
                    entry = self.entries.get(key)
                    if entry is not None:
                        assert (track_slots is None) == (plan.pending[0][1] is None)
                        checked = False
                        if (os.environ.get('SGLANG_GDN_PREFILL_COMMIT_LOOKUP_CHECK', '0') == '1'
                                and not getattr(self, '_lookup_checked', False)):
                            tensors = [x for row in plan.pending for x in row if x is not None]
                            assert key == self._full_key(pool, tensors, factorize, policy)
                            checked = True
                        self._replay(entry, pool, plan, track_slots)
                        self.stats['fast_replayed'] = self.stats.get('fast_replayed', 0) + 1
                        if checked:
                            import json
                            from sglang.srt.distributed import get_tensor_model_parallel_rank
                            print('SSMOFF_COMMIT_LOOKUP_CHECK ' + json.dumps(dict(
                                rank=get_tensor_model_parallel_rank(), layers=len(plan.pending),
                                batch=plan.slots.numel(), tracked=track_slots is not None,
                                matched_reference_key=True, replayed=True)), flush=True)
                            self._lookup_checked = True
                        return True
        states = [x[0] for x in plan.pending]
        tensors = [x for row in plan.pending for x in row if x is not None]
        size = sum(x.numel()*x.element_size() for x in tensors)+pool.vbar.nbytes
        # This path publishes prefix validity only after ALL relevant layers.
        if capturing or not self._eligible(pool, plan, states[0], size, self.MAX_INPUT_BYTES):
            self.stats['fallback'] += 1
            return False
        assert (track_slots is None) == (plan.pending[0][1] is None)
        key = self._full_key(pool, tensors, factorize, policy)
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
        if compact is not None:
            self.fast_entries[compact[0]] = key
        self._replay(entry, pool, plan, track_slots)
        return True

    def _replay(self, entry, pool, plan, track_slots):
        entry[0].bind(plan, track_slots)
        entry[1].replay()
        self.stats['replayed'] += 1
        if (entry[0].store_layers
                and os.environ.get('SGLANG_GDN_PREFILL_STORE_LAYERS_CHECK', '0') == '1'
                and not getattr(self, '_store_layers_checked', False)):
            import json
            from sglang.srt.distributed import get_tensor_model_parallel_rank
            print('SSMOFF_STORE_LAYERS_CHECK ' + json.dumps(dict(
                rank=get_tensor_model_parallel_rank(), layers=list(pool.layer_ids),
                batch=plan.slots.numel(), tracked=track_slots is not None,
                packed=True, replayed=True)), flush=True)
            self._store_layers_checked = True
