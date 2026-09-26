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
        # side stream (SGLANG_GDN_OPUS_SLAB_STREAM=1): the restore runs off the main stream from plan time, so the
        # metadata/attention host syncs between plan and layer 0 do not wait for it; layer 0 waits on one event.
        import os
        self.side = torch.cuda.Stream(device=pool.device) if os.environ.get('SGLANG_GDN_OPUS_SLAB_STREAM', '0') == '1' else None
        self.batched = os.environ.get('SGLANG_GDN_OPUS_SLAB_BATCHED', '0') == '1'

    def _densify_batched(self, bound):
        """The per-layer frozen densify (gdn_factored_pool.densify) with its elementwise work done once for all
        layers; the per-layer einsum is kept on per-layer contiguous slices of the same shapes. ~90 graph nodes
        instead of ~505."""
        p = self.pool
        safe = bound.slots.clamp(min=0)
        a_all, U_all, W_all, c_all = p.a[:, safe], p.U[:, safe], p.W[:, safe], p.count[:, safe]
        rmax = U_all.shape[3]
        rows = torch.arange(rmax, device=U_all.device)[None, None, None, :] < c_all[..., None]
        Uf = U_all.float() * rows[..., None]
        Wf = W_all.float() * rows[..., None]
        S = torch.stack([torch.einsum("bhrv,bhrk->bhvk", Wf[li], Uf[li]) for li in range(len(p.layer_ids))])
        self.slab.copy_(S + p.vbar.float()[:, None, :, :, None] * a_all.float()[:, :, :, None, :])

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
        if self.side is None:
            if not self._fill(plan):
                return False
        else:
            current = torch.cuda.current_stream(plan.slots.device)
            # after every main-stream write of the source slots (COW prefix copy) and every earlier read of the slab
            self.side.wait_stream(current)
            with torch.cuda.stream(self.side):
                filled = self._fill(plan)
            if not filled:
                current.wait_stream(self.side)
                return False
            plan.opus_slab_event = self.side.record_event()
            # plan.slots is read on the side stream: keep its storage from being reused before the restore runs
            plan.slots.record_stream(self.side)
            plan.ring_src.record_stream(self.side)
        plan.opus_slab = self.slab
        return True

    def _fill(self, plan):
        p = self.pool
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
                    if self.batched:
                        self._densify_batched(bound)
                        return
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
        return True
