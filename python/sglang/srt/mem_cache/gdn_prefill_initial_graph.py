"""Replay the original singleton factor-to-dense read without changing math."""
from dataclasses import replace
import logging

import torch

logger = logging.getLogger(__name__)


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
            with torch.cuda.graph(graph, stream=stream):
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
