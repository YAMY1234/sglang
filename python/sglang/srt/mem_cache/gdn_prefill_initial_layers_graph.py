"""One replay for independent layer restores, retaining each layer's algebra."""
from dataclasses import replace
import logging

import torch

logger = logging.getLogger(__name__)


class InitialLayerBuffers:
    def __init__(self, pool, plan):
        self.pool = pool
        self.plan = replace(plan, slots=plan.slots.clone())

    def bind(self, plan):
        self.plan.slots.copy_(plan.slots)

    def evaluate(self):
        # Do not batch the einsum: keep the admitted per-layer reduction order.
        return torch.stack([self.pool._initial_dense_eager(lid, self.plan)
                            for lid in self.pool.layer_ids])


class PrefillInitialLayersGraph:
    MAX_ENTRIES = 2

    def __init__(self):
        self.entries = {}
        self.stats = dict(captured=0, replayed=0, fallback=0)

    def run(self, pool, plan):
        if (plan.all_fresh or plan.n_ring_src or pool.prefix_dense is not None
                or plan.slots.numel() != 1):
            raise ValueError('whole-layer restore requires singleton factor densification')
        if not plan.slots.is_cuda or torch.cuda.is_current_stream_capturing():
            self.stats['fallback'] += 1
            return torch.stack([pool._initial_dense_eager(lid, plan) for lid in pool.layer_ids])
        backing = tuple(t.data_ptr() for t in (pool.a, pool.U, pool.W, pool.count, pool.vbar))
        key = (backing, tuple(pool.layer_ids), tuple(plan.slots.shape), plan.slots.dtype,
               torch.backends.cuda.matmul.allow_tf32,
               torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
               torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= self.MAX_ENTRIES:
                self.stats['fallback'] += 1
                return torch.stack([pool._initial_dense_eager(lid, plan) for lid in pool.layer_ids])
            buffers = InitialLayerBuffers(pool, plan)
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
            logger.info('GDN all-layer initial graph captured: layers=%d bytes=%d',
                        len(pool.layer_ids), output.nbytes)
        else:
            entry[0].bind(plan)
        entry[1].replay()
        self.stats['replayed'] += 1
        # Recurrence may mutate S0, and audit callers may retain it across plans.
        # One all-layer clone preserves both contracts without L clone launches.
        return entry[2].clone()
