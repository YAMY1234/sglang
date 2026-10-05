"""Default-off bucketed prefix factorization plus publication graph.

The pool owns this cache and every captured pool allocation. Slot and ring IDs
are copied into owned device controls before replay. The caller keeps its host
plan bookkeeping and still processes the last token only after this returns.
"""
from dataclasses import replace
import logging
from types import SimpleNamespace

import torch

from sglang.srt.utils.graph_capture import graph_capture_lock
import triton
import triton.language as tl

from sglang.srt.layers.attention.linear.kernels.gdn_prefill_reference import k31_graph_safe
from . import gdn_prefill_joint as joint

logger = logging.getLogger(__name__)


BATCH_BUCKETS = (1, 2, 4, 8, 16)


def prewarm_shapes(*, include_tracked=True):
    for batch in BATCH_BUCKETS:
        yield batch, None
        if not include_tracked:
            continue
        yield batch, batch
        if batch != 1:
            yield 1, batch
            yield batch, 1


def tracked_buffer_dtype(cfg, tracked):
    if tracked is None:
        return None
    return torch.float32 if cfg.init_method == "k31" else tracked.dtype


def batch_bucket(dense, track_dense):
    size = max(dense.shape[0], 0 if track_dense is None else track_dense.shape[0])
    return next((b for b in BATCH_BUCKETS if 0 < size <= b), None)


@triton.jit
def _publish_valid(VALID, SLOTS, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    slot = tl.load(SLOTS + row, row < N, other=-1).to(tl.int64)
    tl.store(VALID + slot, 1, (row < N) & (slot >= 0))


class CommitBuffers:
    def __init__(self, pool, layer_id, plan, dense, track_dense, track_slots, shared=None,
                 *, join_branches=False):
        self.pool = pool
        self.layer_id = layer_id
        self.li = pool.layer_map[layer_id]
        self.cfg = replace(pool.cfg)
        bucket = getattr(plan, "prefill_normal_bucket", None) or batch_bucket(dense, track_dense)
        normal_b = 1 if dense.shape[0] == 1 else bucket
        track_b = 1 if track_dense is not None and track_dense.shape[0] == 1 else bucket
        if shared is None:
            self.dense = dense.new_zeros((normal_b, *dense.shape[1:]))
            # k31 already converts its input to FP32 before any arithmetic.
            self.track_dense = (None if track_dense is None else torch.zeros(
                (track_b, *track_dense.shape[1:]), dtype=tracked_buffer_dtype(self.cfg, track_dense),
                device=track_dense.device))
            self.slots = plan.slots.new_full((normal_b,), -1)
            self.ring_dst = plan.ring_dst.new_full((normal_b,), -1)
            self.track_slots = (None if track_dense is None else
                                track_slots.new_full((track_b,), -1))
        else:
            for name in ("dense", "track_dense", "slots", "ring_dst", "track_slots"):
                setattr(self, name, getattr(shared, name))
        self.ring_generation = getattr(pool, "ring_generation", 0)
        self.ring_pointer = torch.tensor([pool.dense_ring[self.li].data_ptr()],
                                         dtype=torch.int64, device=dense.device)
        self.vbar = pool.vbar[self.li:self.li + 1]  # pool is the cache owner
        if shared is not None:
            self.omega, self.track_omega = shared.omega, shared.track_omega
            self.joint = shared.joint
            self.bind(plan, dense, track_dense, track_slots)
            return
        b, h, v, _ = self.dense.shape
        generator = torch.Generator(device=dense.device).manual_seed(0)
        self.omega = torch.randn(b, h, v, self.cfg.r + self.cfg.init_oversample,
                                 device=dense.device, generator=generator)
        fixed = pool.init_omega(b)
        if fixed is not None:
            self.omega = fixed
        self.track_omega = self.omega
        if track_dense is not None and track_b != b:
            self.track_omega = pool.init_omega(track_b)
        self.joint = None
        if join_branches:
            if not joint.eligible(self.cfg, normal_b, None if track_dense is None else track_b):
                raise ValueError("unsupported joint factorization bucket")
            self.joint = joint.JointInputs([self.dense], [self.track_dense], self.omega, self.track_omega)
            self.dense, self.track_dense = self.joint.normal[0], self.joint.tracked[0]
        self.bind(plan, dense, track_dense, track_slots)

    def bind(self, plan, dense, track_dense, track_slots):
        generation = getattr(self.pool, "ring_generation", 0)
        if generation != self.ring_generation:
            # The captured store reads this control after continuation-ring growth.
            self.ring_pointer.fill_(self.pool.dense_ring[self.li].data_ptr())
            self.ring_generation = generation
        # Clear the unused rows when a smaller batch reuses the same bucket.
        for dst, src, fill in ((self.dense, dense, 0), (self.slots, plan.slots, -1),
                               (self.ring_dst, plan.ring_dst, -1),
                               (self.track_dense, track_dense, 0),
                               (self.track_slots, track_slots, -1)):
            if dst is not None:
                dst[:src.shape[0]].copy_(src)
                if src.shape[0] < dst.shape[0]:
                    dst[src.shape[0]:].fill_(fill)

    def evaluate(self, eager):
        from sglang.srt.layers.attention.linear.kernels.gdn_factored_io import store_factored
        p, i = self.pool, self.li
        # Compute tracked factors before normal publication, exactly as the
        # original commit does; preserve normal/track alias overwrite order.
        if self.joint is None:
            factors = eager([self.dense], self.vbar, self.cfg, omega=self.omega)[0]
            tracked = None if self.track_dense is None else eager(
                [self.track_dense], self.vbar, self.cfg, omega=self.track_omega)[0]
        else:
            normal, checkpoints = self.joint.evaluate(eager, self.vbar, self.cfg)
            factors, tracked = normal[0], checkpoints[0]
        store_factored(*factors, p.a[i], p.U[i], p.W[i], p.count[i],
            p.stale, p.dense_of, self.slots, self.cfg.r, stale_value=0,
            dense=self.dense, ring=self.ring_pointer, ring_dst=self.ring_dst, ring_indirect=True)
        self.publish(self.slots)
        if tracked is not None:
            store_factored(*tracked, p.a[i], p.U[i], p.W[i], p.count[i],
                p.stale, p.dense_of, self.track_slots, self.cfg.r, stale_value=1)
            self.publish(self.track_slots)

    def publish(self, slots):
        # Padded rows must not publish slot zero or overwrite live validity.
        p = self.pool
        if p.prefix_valid is not None and self.li == p.prefix_layer_count() - 1:
            _publish_valid[(1,)](p.prefix_valid, slots, slots.numel(),
                                 triton.next_power_of_2(slots.numel()))


class PrefillCommitGraph:
    def __init__(self):
        self.entries = {}
        self.stats = dict(captured=0, replayed=0, fallback=0, joint_replayed=0)
        self.shared_buffers = {}
        self.memory_pool = None
        self.capture_stream = None
        self.prewarmed = False

    def prewarm(self, pool, *, eager, policy):
        if self.prewarmed:
            return
        expected = {(lid, normal, torch.float32, tracked,
                     None if tracked is None else torch.float32, joined) for lid in pool.layer_ids
                    for normal, tracked in prewarm_shapes(include_tracked=not pool.cfg.no_radix)
                    for joined in joint.modes(pool.cfg, normal, tracked)}
        captured = set()
        before_bytes = torch.cuda.memory_allocated(pool.device)
        for normal, tracked in prewarm_shapes(include_tracked=not pool.cfg.no_radix):
            dense = torch.zeros(normal, pool.hv, pool.v, pool.k,
                                dtype=torch.float32, device=pool.device)
            track_dense = (None if tracked is None else torch.zeros(
                tracked, pool.hv, pool.v, pool.k, dtype=torch.float32, device=pool.device))
            slots = torch.full((normal,), -1, dtype=torch.long, device=pool.device)
            track_slots = (None if tracked is None else torch.full(
                (tracked,), -1, dtype=torch.long, device=pool.device))
            plan = SimpleNamespace(slots=slots, ring_dst=slots.clone(), pending=[None])
            for lid in pool.layer_ids:
                for joined in joint.modes(pool.cfg, normal, tracked):
                    if not self.run(pool, lid, plan, dense, track_dense, track_slots,
                                    eager=eager, policy=policy, join_branches=joined):
                        raise RuntimeError(f"GDN commit prewarm rejected {(lid, normal, tracked, joined)}")
                    captured.add((lid, normal, dense.dtype, tracked,
                                  None if track_dense is None else track_dense.dtype, joined))
        torch.cuda.synchronize(pool.device)
        actual = {(key[0], key[1][0][0][0], key[1][0][1],
                   None if key[1][3] is None else key[1][3][0][0],
                   None if key[1][3] is None else key[1][3][1], key[-1]) for key in self.entries}
        if captured != expected or actual != expected:
            raise RuntimeError(f"GDN commit prewarm incomplete: missing={expected - actual}")
        self.prewarmed = True
        logger.info("GDN prefill commit prewarm complete: expected=%d captured=%d signatures=%s retained_bytes=%d",
                    len(expected), len(actual), sorted({str(item[1:]) for item in actual}),
                    torch.cuda.memory_allocated(pool.device) - before_bytes)

    def run(self, pool, layer_id, plan, dense, track_dense, track_slots, *, eager, policy,
            join_branches=None):
        # Keep the per-layer dependency; only k31 expands beyond singleton.
        bucket = getattr(plan, "prefill_normal_bucket", None) or batch_bucket(dense, track_dense)
        if ((pool.cfg.init_method == "k31" and not k31_graph_safe(dense.device))
                or not dense.is_cuda or torch.cuda.is_current_stream_capturing()
                or len(plan.pending) != 1 or bucket is None
                or (pool.cfg.init_method != "k31" and bucket != 1)
                or pool.prefix_dense is not None
                or not (pool.cfg.factored_prefix or pool.cfg.no_radix)):
            self.stats['fallback'] += 1
            return False
        tensors = (dense, plan.slots, plan.ring_dst, track_dense, track_slots)
        # Preserve singleton arithmetic: its matmul reduction differs from B>1.
        normal_b = 1 if dense.shape[0] == 1 else bucket
        track_b = 1 if track_dense is not None and track_dense.shape[0] == 1 else bucket
        if join_branches is None:
            join_branches = joint.enabled() and joint.eligible(
                pool.cfg, normal_b, None if track_dense is None else track_b)
        batches = (normal_b, normal_b, normal_b, track_b, track_b)
        cfg = pool.cfg
        dtypes = (dense.dtype, plan.slots.dtype, plan.ring_dst.dtype,
                  tracked_buffer_dtype(cfg, track_dense),
                  None if track_slots is None else track_slots.dtype)
        shapes = tuple(None if x is None else ((b, *x.shape[1:]), dtype, x.device)
                       for x, b, dtype in zip(tensors, batches, dtypes))
        config = (cfg.r, cfg.rmax, cfg.dtype, cfg.init_iters, cfg.init_oversample, cfg.init_method)
        key = (layer_id, shapes, config, policy, eager,
               torch.backends.cuda.matmul.allow_tf32,
               torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
               torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, join_branches)
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= (len(tuple(prewarm_shapes())) + int(joint.configured())) * len(pool.layer_ids):
                self.stats['fallback'] += 1
                return False
            buffers = CommitBuffers(pool, layer_id, plan, dense, track_dense, track_slots,
                                    shared=self.shared_buffers.get(key[1:]), join_branches=join_branches)
            self.shared_buffers.setdefault(key[1:], buffers)
            current = torch.cuda.current_stream(dense.device)
            if self.capture_stream is None:
                self.capture_stream = torch.cuda.Stream(device=dense.device)
            stream = self.capture_stream
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                buffers.evaluate(eager)
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            # Entries replay sequentially; only their writes to the pool survive replay.
            with graph_capture_lock, torch.cuda.graph(graph, pool=self.memory_pool, stream=stream,
                                                       capture_error_mode="thread_local"):
                buffers.evaluate(eager)
            if self.memory_pool is None:
                self.memory_pool = graph.pool()
            entry = (buffers, graph, stream)
            self.entries[key] = entry
            self.stats['captured'] += 1
            logger.info('GDN prefill commit graph captured: layer=%d tracked=%s batch=%d normal_batch=%d tracked_batch=%s normal_dtype=%s tracked_dtype=%s',
                        layer_id, track_dense is not None, bucket, normal_b,
                        None if track_dense is None else track_b, dense.dtype, dtypes[3])
        else:
            entry[0].bind(plan, dense, track_dense, track_slots)
        entry[1].replay()
        self.stats['replayed'] += 1
        self.stats['joint_replayed'] += int(join_branches)
        return True
