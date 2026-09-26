"""#ssmoff-opus: all-layer initial states restored once per extend plan into one persistent slab.

The chunk kernel reads its initial state from, and writes its final state into, `slab[layer]` (a contiguous view);
the per-layer restore call, its graph bind/replay and the private clone disappear from the per-layer host path.
The densification math is the frozen `_initial_dense_eager` of every layer, replayed from one captured graph and
copied into the slab (bitwise the same values). Fresh rows are zeros, owned-ring rows a gather. Singleton plans
covering every layer only; anything else keeps the frozen per-layer path.
"""
from dataclasses import replace
import logging

import torch

logger = logging.getLogger(__name__)


class PrefillSlab:
    MAX_ENTRIES = 2

    def __init__(self, pool):
        self.pool = pool
        L = len(pool.layer_ids)
        self.slab = torch.empty(L, 1, pool.hv, pool.v, pool.k, dtype=torch.float32, device=pool.device)
        self.graphs = {}
        self.stats = dict(fresh=0, ring=0, graph=0, captured=0, declined=0)

    def eligible(self, plan):
        p = self.pool
        return (plan.slots.numel() == 1 and plan.next_layer == 0 and plan.last_layer == len(p.layer_ids) - 1
                and p.prefix_dense is None and plan.slots.is_cuda
                and not torch.cuda.is_current_stream_capturing())

    def restore(self, plan):
        """Fill the slab for this plan; returns True when the plan now reads its initial states from the slab."""
        p = self.pool
        if not self.eligible(plan):
            self.stats['declined'] += 1
            return False
        if plan.all_fresh:
            self.slab.zero_()
            self.stats['fresh'] += 1
        elif plan.n_ring_src == 1:
            torch.index_select(p.dense_ring, 1, plan.ring_src, out=self.slab)
            self.stats['ring'] += 1
        else:
            key = (tuple(t.data_ptr() for t in (p.a, p.U, p.W, p.count, p.vbar)), plan.slots.dtype,
                   torch.backends.cuda.matmul.allow_tf32)
            entry = self.graphs.get(key)
            if entry is None:
                if len(self.graphs) >= self.MAX_ENTRIES:
                    self.stats['declined'] += 1
                    return False
                bound = replace(plan, slots=plan.slots.clone(), pending=[])

                def evaluate():
                    for li, lid in enumerate(p.layer_ids):
                        self.slab[li].copy_(p._initial_dense_eager(lid, bound))

                current = torch.cuda.current_stream(plan.slots.device)
                stream = torch.cuda.Stream(device=plan.slots.device)
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    evaluate()
                current.wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    evaluate()
                entry = self.graphs[key] = (bound, graph)
                self.stats['captured'] += 1
                logger.info('GDN prefill slab restore graph captured: layers=%d bytes=%d',
                            len(p.layer_ids), self.slab.nbytes)
            entry[0].slots.copy_(plan.slots)
            entry[1].replay()
            self.stats['graph'] += 1
        plan.opus_slab = self.slab
        return True
