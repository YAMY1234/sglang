"""Replay the original singleton factor-to-dense read without changing math."""
from dataclasses import replace
import logging
import os

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


def densify_all_layers(pool, plan):
    """Batch independent heads using the original densify expression."""
    from .gdn_factored_pool import densify
    layers, batch, heads = len(pool.layer_ids), plan.slots.numel(), pool.hv
    safe = plan.slots.clamp(min=0)
    def gather(tensor):
        gathered = tensor[:, safe].transpose(0, 1)
        return gathered.reshape(batch, layers*heads, *tensor.shape[3:])
    a, u, w, count = (gather(t) for t in (pool.a, pool.U, pool.W, pool.count))
    result = densify(a, u, w, count, pool.vbar.reshape(layers*heads, pool.v))
    return result.reshape(batch, layers, heads, pool.v, pool.k).transpose(0, 1).contiguous()


class PrefillInitialBatchGraph:
    """Prepare independent layer inputs once, with per-forward ownership."""
    MAX_BYTES = 128 << 20

    def __init__(self):
        self.entries = {}
        self.stats = dict(captured=0, replayed=0, fresh=0, ring=0, fallback=0)

    def run(self, pool, plan):
        layers = len(pool.layer_ids)
        fused_layers = os.environ.get('SGLANG_GDN_PREFILL_INITIAL_FUSED_LAYERS', '0') == '1'
        size = layers * plan.slots.numel() * pool.hv * pool.v * pool.k * 4
        if (plan.next_layer != 0 or plan.last_layer != layers-1
                or plan.slots.numel() != 1 or size > self.MAX_BYTES
                or pool.prefix_layer_count() != layers):
            self.stats['fallback'] += 1
            return None
        if plan.all_fresh:
            self.stats['fresh'] += 1
            states = torch.zeros(layers, 1, pool.hv, pool.v, pool.k,
                                 dtype=torch.float32, device=pool.device)
        elif plan.n_ring_src == 1:
            self.stats['ring'] += 1
            states = pool.dense_ring[:, plan.ring_src].contiguous()
        elif pool.prefix_dense is not None:
            self.stats['fallback'] += 1
            return None
        elif not plan.slots.is_cuda or torch.cuda.is_current_stream_capturing():
            states = (densify_all_layers(pool, plan) if fused_layers else
                      torch.stack([pool._initial_dense_eager(li, plan) for li in pool.layer_ids]))
        else:
            pointers = tuple(t.data_ptr() for t in (pool.a, pool.U, pool.W, pool.count, pool.vbar))
            key = (pointers, plan.slots.dtype, torch.backends.cuda.matmul.allow_tf32, fused_layers)
            entry = self.entries.get(key)
            if entry is None:
                if self.entries:
                    self.stats['fallback'] += 1
                    return None
                bound = replace(plan, slots=plan.slots.clone(), initial_states=None)
                def evaluate():
                    # Keep every layer's original densify operand/reduction
                    # shape; only group launches and the output copy.
                    if fused_layers:
                        return densify_all_layers(pool, bound)
                    return torch.stack([pool._initial_dense_eager(li, bound)
                                        for li in pool.layer_ids])
                current = torch.cuda.current_stream(plan.slots.device)
                stream = torch.cuda.Stream(device=plan.slots.device)
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    evaluate()
                current.wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = evaluate()
                entry = (bound, graph, output, stream)
                self.entries[key] = entry
                self.stats['captured'] += 1
                logger.info('GDN whole-layer initial graph captured: bytes=%d', size)
            else:
                entry[0].slots.copy_(plan.slots)
            entry[1].replay()
            self.stats['replayed'] += 1
            # Recurrence mutates S0. Each plan owns one copy, and each layer
            # receives a disjoint contiguous view; no graph output is exposed.
            states = entry[2].clone()
        return tuple(states.unbind(0))
