"""Replay the original singleton factor-to-dense read without changing math."""
from dataclasses import replace
import logging

import torch

from sglang.srt.utils.graph_capture import graph_capture_lock

logger = logging.getLogger(__name__)


class RestoreBuffers:
    """Graph body shared with the real CPU equivalence gate."""

    def __init__(self, pool, plan):
        self.pool = pool
        self.plan = replace(plan, slots=plan.slots.clone(), pending=[], stage=None)

    def bind(self, plan):
        self.plan.slots.copy_(plan.slots, non_blocking=True)

    def evaluate(self):
        # Keep each layer's original gather/mask/einsum/add shape and order.
        # Flattening L into B here would change GEMM selection/rounding.
        return torch.stack([
            self.pool._initial_dense_eager(lid, self.plan)
            for lid in self.pool.layer_ids
        ])


class PrefillRestoreGraph:
    """One fixed singleton restore graph; no lazy capture in serving.

    The captured output is never handed to a forward. A single clone makes a
    per-plan stage, so ring writes, full-N collectors and deferred publication
    cannot retain an alias to the next replay. One owning forward stream is
    required; other streams/capture/precision or backing changes fall back.
    """

    def __init__(self):
        self.entry = None
        self.stats = dict(captured=0, replayed=0, fallbacks=0)

    @staticmethod
    def _key(pool, plan):
        return (tuple(t.data_ptr() for t in
                      (pool.a, pool.U, pool.W, pool.count, pool.vbar)),
                plan.slots.dtype, plan.slots.device,
                torch.backends.cuda.matmul.allow_tf32,
                torch.get_float32_matmul_precision())

    def prewarm(self, pool, plan):
        if (not plan.slots.is_cuda or plan.slots.numel() != 1
                or torch.cuda.is_current_stream_capturing()):
            return False
        buffers = RestoreBuffers(pool, plan)
        current = torch.cuda.current_stream(plan.slots.device)
        stream = torch.cuda.Stream(device=plan.slots.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            buffers.evaluate()
        current.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with graph_capture_lock, torch.cuda.graph(
                graph, stream=stream, capture_error_mode="thread_local"):
            output = buffers.evaluate()
        self.entry = (self._key(pool, plan), buffers, graph, output, stream, None)
        self.stats['captured'] += 1
        logger.info('GDN prefix restore graph: layers=%d rows=1 bytes=%d captured=1',
                    len(pool.layer_ids), output.numel() * output.element_size())
        return True

    def run(self, pool, plan):
        if (self.entry is None or not plan.slots.is_cuda
                or torch.cuda.is_current_stream_capturing()):
            self.stats['fallbacks'] += 1
            return None
        current = torch.cuda.current_stream(plan.slots.device).cuda_stream
        key, buffers, graph, output, stream, owner = self.entry
        if key != self._key(pool, plan) or (owner is not None and owner != current):
            self.stats['fallbacks'] += 1
            return None
        if owner is None:
            self.entry = (key, buffers, graph, output, stream, current)
        buffers.bind(plan)
        graph.replay()
        self.stats['replayed'] += 1
        if self.stats['replayed'] == 1 or self.stats['replayed'] % 100 == 0:
            logger.info('GDN prefix restore graph: captured=%d replayed=%d fallbacks=%d',
                        self.stats['captured'], self.stats['replayed'], self.stats['fallbacks'])
        return output.clone()


class InitialBuffers:
    def __init__(self, pool, layer_id, plan):
        self.pool, self.layer_id = pool, layer_id
        self.plan = replace(plan, slots=plan.slots.clone())

    def bind(self, plan):
        self.plan.slots.copy_(plan.slots)

    def evaluate(self):
        return self.pool._initial_dense_eager(self.layer_id, self.plan)


class PrefillInitialGraph:
    def __init__(self):
        self.entries = {}
        self.stats = dict(captured=0, replayed=0)

    def run(self, pool, layer_id, plan):
        # The caller restricts this graph to the non-ring singleton path.
        # Only the slot changes; factor/count/vbar backing remains live.
        if (plan.slots.numel() != 1 or plan.all_fresh or plan.n_ring_src
                or pool.prefix_dense is not None):
            raise ValueError('initial graph requires singleton factor densification')
        if not plan.slots.is_cuda or torch.cuda.is_current_stream_capturing():
            return pool._initial_dense_eager(layer_id, plan)
        pointers = tuple(t.data_ptr() for t in (pool.a, pool.U, pool.W, pool.count, pool.vbar))
        key = (layer_id, pointers, plan.slots.dtype, torch.backends.cuda.matmul.allow_tf32)
        entry = self.entries.get(key)
        if entry is None:
            # Fixed pool backing and precision: at most one graph per layer.
            # Fall back rather than retaining an unbounded series of policies.
            if len(self.entries) >= len(pool.layer_ids):
                return pool._initial_dense_eager(layer_id, plan)
            buffers = InitialBuffers(pool, layer_id, plan)
            current = torch.cuda.current_stream(plan.slots.device)
            stream = torch.cuda.Stream(device=plan.slots.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                buffers.evaluate()
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with graph_capture_lock, torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
                output = buffers.evaluate()
            entry = (buffers, graph, output, stream)
            self.entries[key] = entry
            self.stats['captured'] += 1
            logger.info('GDN prefill initial graph captured: layer=%d', layer_id)
        else:
            entry[0].bind(plan)
        entry[1].replay()
        self.stats['replayed'] += 1
        # The recurrence mutates its initial state in place. Keep captured
        # output storage private, including when an audited call retains S0.
        return entry[2].clone()


class PrefillDensifyAllGraph:
    """Every layer's singleton densify in ONE replay (was one graph replay + clone per layer).  The output
    (L, 1, HV, V, K) is the forward's stage: the chunk kernel updates each layer slice in place and the commit graph
    binds it with one copy; the next prefix-hit forward's replay rewrites every element (same stream order)."""

    def __init__(self):
        self.entry = None
        self.stats = dict(captured=0, replayed=0)

    def run(self, pool, slot, densify):
        if not slot.is_cuda or torch.cuda.is_current_stream_capturing():
            return None
        if self.entry is None:
            static = slot.clone()

            def evaluate():
                safe = static.clamp(min=0)
                return torch.stack([densify(pool.a[li][safe], pool.U[li][safe], pool.W[li][safe],
                                            pool.count[li][safe], pool.vbar[li])
                                    for li in range(len(pool.layer_ids))])

            current = torch.cuda.current_stream(slot.device)
            stream = torch.cuda.Stream(device=slot.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                evaluate()
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with graph_capture_lock, torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
                output = evaluate()
            self.entry = (static, graph, output, stream)
            self.stats['captured'] += 1
            logger.info('GDN prefill densify-all graph captured: layers=%d', len(pool.layer_ids))
        else:
            self.entry[0].copy_(slot)
        self.entry[1].replay()
        self.stats['replayed'] += 1
        return self.entry[2]
