"""One prompt-prefix commit graph for every layer and both checkpoint branches."""
import logging
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from sglang.srt.utils.graph_capture import graph_capture_lock
from .gdn_prefill_commit_graph import BATCH_BUCKETS, _publish_valid, prewarm_shapes

logger = logging.getLogger(__name__)


class BatchCollector:
    def __init__(self, pool, plan):
        self.pool, self.plan = pool, plan
        self.controls = None

    def __enter__(self):
        p, plan = self.pool, self.plan
        if (plan.next_layer != 0 or plan.last_layer != len(p.layer_ids) - 1
                or plan.pending or getattr(plan, "batch_collector", None) is not None
                or p.prefix_layer_count() != len(p.layer_ids)):
            raise RuntimeError("whole-prefix collection requires an unused complete layer plan")
        p.pside_join()
        plan.checkpoint_group = None
        plan.batch_collector = self
        return self

    @staticmethod
    def identity(tensor):
        return None if tensor is None else (tensor.data_ptr(), tuple(tensor.shape))

    def add(self, layer_id, dense, tracked, track_slots, final_src, final_dst):
        p, plan = self.pool, self.plan
        index = p.layer_map[layer_id]
        if index != plan.next_layer:
            raise RuntimeError("whole-prefix layers arrived out of order")
        controls = (track_slots, final_src, final_dst)
        if self.controls is None:
            self.controls = controls
            p.invalidate_prefix_dense(plan.slots)
            if track_slots is not None:
                p.invalidate_prefix_dense(track_slots)
        elif tuple(map(self.identity, controls)) != tuple(map(self.identity, self.controls)):
            raise RuntimeError("whole-prefix checkpoint destinations changed between layers")
        if plan.pending and (tracked is None) != (plan.pending[0][1] is None):
            raise RuntimeError("whole-prefix tracked branch changed between layers")
        plan.pending.append((dense, tracked))
        plan.next_layer += 1

    def __exit__(self, exc_type, exc, traceback):
        from .gdn_factored_pool import factorize_layers, factorize_dense, ORTH_METHOD, ORTH_WARPS_OVERRIDE

        p, plan = self.pool, self.plan
        plan.batch_collector = None
        if exc_type is not None:
            plan.pending.clear()
            return False
        if plan.next_layer != len(p.layer_ids) or len(plan.pending) != len(p.layer_ids):
            raise RuntimeError("whole-prefix collection ended before every layer committed")
        graph = getattr(p, "_prefill_batch_graph", None)
        if graph is None or not graph.warmed:
            raise RuntimeError("whole-prefix graph must be prewarmed before model execution")
        graph.run(p, plan, plan.pending, *self.controls, eager=factorize_layers,
                  policy=(ORTH_METHOD, ORTH_WARPS_OVERRIDE, factorize_dense))
        plan.pending.clear()
        return False


@triton.jit
def _scatter_rows(SRC, DST, SLOTS, B: tl.constexpr, S: tl.constexpr,
                  M: tl.constexpr, BLOCK: tl.constexpr):
    layer, row, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(SLOTS + row).to(tl.int64)
    offset = tile * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(SRC + (layer * B + row) * M + offset, offset < M, other=0)
    tl.store(DST + (layer * S + slot) * M + offset, value, (offset < M) & (slot >= 0))


def scatter_rows(source, target, slots):
    layers, batch = source.shape[:2]
    width = source[0, 0].numel()
    _scatter_rows[(layers, batch, triton.cdiv(width, 1024))](
        source, target, slots, batch, target.shape[1], width, 1024, num_warps=4)


class BatchBuffers:
    def __init__(self, pool, batch, tracked_batch, shared=None):
        self.pool, self.batch, self.tracked_batch = pool, batch, tracked_batch
        shared = {} if shared is None else shared

        def states(role, size):
            if size is None:
                return None
            key = (role, size)
            if key not in shared:
                shape = (size, pool.hv, pool.v, pool.k)
                shared[key] = [pool.a.new_zeros(shape) for _ in pool.layer_ids]
            return shared[key]

        self.normal = states("normal", batch)
        self.tracked = states("tracked", tracked_batch)
        self.slots = torch.full((batch,), -1, dtype=torch.long, device=pool.a.device)
        self.track_slots = (None if tracked_batch is None else torch.full(
            (tracked_batch,), -1, dtype=torch.long, device=pool.a.device))
        self.ring_dst = self.slots.clone()
        self.final_src, self.final_dst = self.slots.clone(), self.slots.clone()
        self.required = torch.zeros(batch, dtype=torch.int32, device=pool.a.device)
        self.ring_pointers = torch.tensor([row.data_ptr() for row in pool.dense_ring],
                                          dtype=torch.int64, device=pool.a.device)
        self.ring_generation = getattr(pool, "ring_generation", 0)
        self.omega = pool.init_omega(batch)
        self.track_omega = None if tracked_batch is None else pool.init_omega(tracked_batch)

    @staticmethod
    def copy_padded(dst, src, fill):
        if dst is None:
            if src is not None and src.numel():
                raise RuntimeError("unexpected tracked controls")
            return
        count = 0 if src is None else src.shape[0]
        if count:
            dst[:count].copy_(src)
        if count < dst.shape[0]:
            dst[count:].fill_(fill)

    def bind(self, plan, states, track_slots, final_src, final_dst):
        if len(states) != len(self.normal):
            raise RuntimeError("whole-prefix commit is missing layers")
        for i, (normal, tracked) in enumerate(states):
            self.copy_padded(self.normal[i], normal, 0)
            if self.tracked is not None:
                if tracked is None:
                    raise RuntimeError("tracked checkpoint is missing a layer")
                self.copy_padded(self.tracked[i], tracked, 0)
        for dst, src, fill in ((self.slots, plan.slots, -1), (self.ring_dst, plan.ring_dst, -1),
                               (self.track_slots, track_slots, -1), (self.final_src, final_src, -1),
                               (self.final_dst, final_dst, -1),
                               (self.required, plan.dense_required_after_commit, 0)):
            self.copy_padded(dst, src, fill)
        generation = getattr(self.pool, "ring_generation", 0)
        if generation != self.ring_generation:
            for i, row in enumerate(self.pool.dense_ring):
                self.ring_pointers[i].fill_(row.data_ptr())
            self.ring_generation = generation

    def evaluate(self, eager):
        from sglang.srt.layers.attention.linear.kernels.gdn_factored_io import store_factored

        p = self.pool
        normal = eager(self.normal, p.vbar, p.cfg, omega=self.omega)
        tracked = None if self.tracked is None else eager(self.tracked, p.vbar, p.cfg, omega=self.track_omega)
        for i in range(len(p.layer_ids)):
            store_factored(*normal[i], p.a[i], p.U[i], p.W[i], p.count[i],
                           p.stale, p.dense_of, self.slots, p.cfg.r, stale_value=0,
                           dense=self.normal[i], ring=self.ring_pointers[i:i+1],
                           ring_dst=self.ring_dst, ring_indirect=True)
            if tracked is not None:
                store_factored(*tracked[i], p.a[i], p.U[i], p.W[i], p.count[i],
                               p.stale, p.dense_of, self.track_slots, p.cfg.r, stale_value=1)
        for slots in (self.slots, self.track_slots):
            if slots is not None:
                _publish_valid[(1,)](p.prefix_valid, slots, slots.numel(), triton.next_power_of_2(slots.numel()))
        if p.dense_required is not None:
            scatter_rows(self.required[None, :, None], p.dense_required[None, :, None], self.slots)
        # Snapshot every source before any destination write, including aliases.
        source = self.final_src.clamp_min(0)
        for tensor in (p.a, p.U, p.W, p.count):
            scatter_rows(tensor.index_select(1, source), tensor, self.final_dst)
        for tensor, value in ((p.stale, 1), (p.dense_of, -1), (p.dense_required, 0)):
            if tensor is not None:
                rows = torch.full((1, self.batch, 1), value, dtype=tensor.dtype, device=tensor.device)
                scatter_rows(rows, tensor[None, :, None], self.final_dst)
        valid = p.prefix_valid.index_select(0, source)[None, :, None]
        scatter_rows(valid, p.prefix_valid[None, :, None], self.final_dst)


class PrefillBatchGraph:
    def __init__(self):
        self.entries = {}
        self.shared = {}
        self.warmed = False
        self.stats = dict(captured=0, replayed=0)
        self.memory_pool = None
        self.stream = None

    @staticmethod
    def key(batch, tracked, eager, policy):
        return (batch, tracked, eager, policy, torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)

    def run(self, pool, plan, states, track_slots, final_src, final_dst, *, eager, policy):
        size = max(plan.slots.numel(), 0 if track_slots is None else track_slots.numel())
        batch = next((b for b in BATCH_BUCKETS if 0 < size <= b), None)
        if batch is None:
            raise ValueError("whole-prefix graph batch is outside its prewarmed buckets")
        normal_batch = 1 if plan.slots.numel() == 1 else batch
        tracked_batch = (None if states[0][1] is None else
                         1 if states[0][1].shape[0] == 1 else batch)
        key = self.key(normal_batch, tracked_batch, eager, policy)
        entry = self.entries.get(key)
        if entry is None:
            if self.warmed:
                raise RuntimeError("whole-prefix graph missing after complete prewarm")
            buffers = BatchBuffers(pool, normal_batch, tracked_batch, self.shared)
            buffers.bind(plan, states, track_slots, final_src, final_dst)
            current = torch.cuda.current_stream(pool.a.device)
            if self.stream is None:
                self.stream = torch.cuda.Stream(device=pool.a.device)
                self.memory_pool = torch.cuda.graph_pool_handle()
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                buffers.evaluate(eager)
            current.wait_stream(self.stream)
            graph = torch.cuda.CUDAGraph()
            with graph_capture_lock, torch.cuda.graph(graph, stream=self.stream,
                    pool=self.memory_pool, capture_error_mode="thread_local"):
                buffers.evaluate(eager)
            entry = self.entries[key] = (buffers, graph)
            self.stats["captured"] += 1
            logger.info("GDN prefill batch graph captured: layers=%d normal_batch=%d tracked_batch=%s",
                        len(pool.layer_ids), normal_batch, tracked_batch)
        else:
            entry[0].bind(plan, states, track_slots, final_src, final_dst)
        entry[1].replay()
        self.stats["replayed"] += 1

    def prewarm(self, pool, *, eager, policy):
        if self.warmed:
            return
        before = torch.cuda.memory_allocated(pool.a.device)
        expected = set()
        for batch, tracked_batch in prewarm_shapes():
            normal = pool.a.new_zeros((batch, pool.hv, pool.v, pool.k))
            tracked = (None if tracked_batch is None else pool.a.new_zeros(
                (tracked_batch, pool.hv, pool.v, pool.k)))
            slots = torch.full((batch,), -1, dtype=torch.long, device=pool.a.device)
            track_slots = (None if tracked_batch is None else torch.full(
                (tracked_batch,), -1, dtype=torch.long, device=pool.a.device))
            plan = SimpleNamespace(slots=slots, ring_dst=slots, dense_required_after_commit=None)
            states = [(normal, tracked) for _ in pool.layer_ids]
            self.run(pool, plan, states, track_slots, None, None, eager=eager, policy=policy)
            expected.add(self.key(batch, tracked_batch, eager, policy))
        torch.cuda.synchronize(pool.a.device)
        if set(self.entries) != expected:
            raise RuntimeError("whole-prefix prewarm did not cover both branches and every bucket")
        self.warmed = True
        owned = sum(t.numel() * t.element_size() for tensors in self.shared.values() for t in tensors)
        logger.info("GDN prefill batch prewarm complete: layers=%d expected=%d captured=%d signatures=%s "
                    "owned_state_bytes=%d retained_bytes=%d allocated_bytes=%d reserved_bytes=%d",
                    len(pool.layer_ids), len(expected), len(self.entries),
                    sorted(str(key[:2]) for key in self.entries), owned,
                    torch.cuda.memory_allocated(pool.a.device) - before,
                    torch.cuda.memory_allocated(pool.a.device), torch.cuda.memory_reserved(pool.a.device))
