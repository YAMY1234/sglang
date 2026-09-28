"""Bounded cross-layer graphs for independent radix checkpoints."""
import logging
import os

import torch

from sglang.srt.utils.graph_capture import graph_capture_lock
from .gdn_prefill_commit_graph import BATCH_BUCKETS, _publish_valid

logger = logging.getLogger(__name__)
GROUP_LAYERS = 6


def independent_slots(live, tracked, final):
    tracked = [int(s) for s in tracked if s >= 0]
    return bool(tracked and len(set(tracked)) == len(tracked)
                and not set(tracked).intersection(s for s in (*live, *final) if s >= 0))


def prepare(pool, metadata):
    plan = metadata.factored_extend
    if (os.environ.get("SGLANG_GDN_PREFILL_CHECKPOINT_GRAPH") != "1"
            or os.environ.get("SGLANG_GDN_PSIDE_COMPOSITE") == "1"
            or os.environ.get("TWINSTAR_PD_EMITTER_GRAPH") == "1"
            or pool.cfg.init_method != "k31" or not pool.cfg.factored_prefix
            or pool.prefix_dense is not None or not pool.a.is_cuda
            or not metadata.has_mamba_track_mask
            or plan.next_layer != 0 or plan.last_layer != len(pool.layer_ids) - 1
            or len(pool.layer_ids) % GROUP_LAYERS):
        return
    tracked = metadata.track_ssm_h_dst
    if (tracked is None or not 0 < tracked.numel() <= BATCH_BUCKETS[-1]
            or plan.slots.numel() > BATCH_BUCKETS[-1]):
        return
    final = metadata.track_ssm_final_dst
    live_cpu = plan.slots.tolist()
    tracked_cpu = tracked.tolist()
    final_cpu = [] if final is None else final.tolist()
    if not independent_slots(live_cpu, tracked_cpu, final_cpu):
        return
    graph = getattr(pool, "_pfactor_checkpoint_graph", None)
    if graph is None or not graph.warmed:
        return
    pool.invalidate_prefix_dense(tracked)
    plan.checkpoint_group = CheckpointGroup(pool, graph, tracked, plan.slots.numel())


class CheckpointGroup:
    def __init__(self, pool, graph, slots, normal_batch=None):
        self.pool, self.graph, self.slots = pool, graph, slots
        size = max(slots.numel(), normal_batch or slots.numel())
        self.batch = 1 if slots.numel() == 1 else next(b for b in BATCH_BUCKETS if size <= b)
        self.next_layer = 0
        self.states = []
        self.active = None

    def accepts(self, layer_index, plan):
        split = len(plan.pending) == 1 and plan.last_layer == layer_index
        if self.active is None:
            self.active = split and layer_index == 0
        if self.active and (not split or layer_index != self.next_layer):
            raise RuntimeError("active checkpoint group lost its per-layer split")
        return self.active

    def add(self, layer_index, state, slots, *, eager, policy):
        if layer_index != self.next_layer or slots.data_ptr() != self.slots.data_ptr():
            raise RuntimeError("checkpoint group changed order or destination slots")
        self.next_layer += 1
        self.states.append(state)
        if len(self.states) == GROUP_LAYERS:
            self.graph.run(self.pool, self.next_layer - GROUP_LAYERS, self.states,
                           self.slots, eager=eager, policy=policy, batch=self.batch)
            self.states.clear()


class GroupBuffers:
    def __init__(self, pool, first, states, slots, shared=None, batch=None):
        self.pool, self.first = pool, first
        batch = batch or next(b for b in BATCH_BUCKETS if states[0].shape[0] <= b)
        if shared is None:
            self.states = [s.new_zeros((batch, *s.shape[1:])) for s in states]
            self.slots = slots.new_full((batch,), -1)
        else:
            self.states, self.slots = shared.states, shared.slots
        self.vbar = pool.vbar[first:first + GROUP_LAYERS]
        self.omega = pool.init_omega(batch)
        self.bind(states, slots)

    def bind(self, states, slots):
        for dst, src in zip(self.states, states):
            dst[:src.shape[0]].copy_(src)
            dst[src.shape[0]:].zero_()
        self.slots[:slots.numel()].copy_(slots)
        self.slots[slots.numel():].fill_(-1)

    def evaluate(self, eager):
        from sglang.srt.layers.attention.linear.kernels.gdn_factored_io import store_factored
        import triton

        p = self.pool
        factors = eager(self.states, self.vbar, p.cfg, omega=self.omega)
        for offset, values in enumerate(factors):
            i = self.first + offset
            store_factored(*values, p.a[i], p.U[i], p.W[i], p.count[i],
                           p.stale, p.dense_of, self.slots, p.cfg.r, stale_value=1)
        if self.first + GROUP_LAYERS == p.prefix_layer_count():
            _publish_valid[(1,)](p.prefix_valid, self.slots, self.slots.numel(),
                                triton.next_power_of_2(self.slots.numel()))


class CheckpointGraph:
    def __init__(self):
        self.entries = {}
        self.shared = {}
        self.warmed = False
        self.stats = dict(captured=0, replayed=0)
        self.capture_pool = None
        self.stream = None

    def run(self, pool, first, states, slots, *, eager, policy, batch=None):
        batch = batch or next(b for b in BATCH_BUCKETS if states[0].shape[0] <= b)
        key = self.key(first, batch, policy, eager)
        entry = self.entries.get(key)
        if entry is None:
            if self.warmed:
                raise RuntimeError("checkpoint graph missing after complete prewarm")
            if self.stream is None:
                self.stream = torch.cuda.Stream(device=states[0].device)
                self.capture_pool = torch.cuda.graph_pool_handle()
            buffers = GroupBuffers(pool, first, states, slots, self.shared.get(batch), batch)
            self.shared.setdefault(batch, buffers)
            current = torch.cuda.current_stream(states[0].device)
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                buffers.evaluate(eager)
            current.wait_stream(self.stream)
            graph = torch.cuda.CUDAGraph()
            with graph_capture_lock, torch.cuda.graph(graph, stream=self.stream,
                    pool=self.capture_pool, capture_error_mode="thread_local"):
                buffers.evaluate(eager)
            entry = (buffers, graph)
            self.entries[key] = entry
            self.stats["captured"] += 1
            logger.info("GDN prefill checkpoint graph captured: first=%d layers=%d batch=%d",
                        first, GROUP_LAYERS, batch)
        else:
            entry[0].bind(states, slots)
        entry[1].replay()
        self.stats["replayed"] += 1

    @staticmethod
    def key(first, batch, policy, eager):
        return (first, batch, policy, eager, torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)

    def prewarm(self, pool, *, eager, policy):
        if self.warmed or len(pool.layer_ids) % GROUP_LAYERS:
            return
        expected = set()
        for batch in BATCH_BUCKETS:
            states = [pool.a.new_zeros((batch, pool.hv, pool.v, pool.k))
                      for _ in range(GROUP_LAYERS)]
            slots = torch.full((batch,), -1, dtype=torch.long, device=pool.a.device)
            for first in range(0, len(pool.layer_ids), GROUP_LAYERS):
                self.run(pool, first, states, slots, eager=eager, policy=policy)
                expected.add(self.key(first, batch, policy, eager))
        if set(self.entries) != expected:
            raise RuntimeError("checkpoint prewarm did not cover every layer group and batch")
        self.warmed = True
        logger.info("GDN prefill checkpoint prewarm complete: expected=%d captured=%d",
                    len(expected), len(self.entries))
