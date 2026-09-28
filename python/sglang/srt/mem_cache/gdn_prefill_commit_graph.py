"""Default-off bucketed prefix factorization plus publication graph.

The pool owns this cache and every captured pool allocation. Slot and ring IDs
are copied into owned device controls before replay. The caller keeps its host
plan bookkeeping and still processes the last token only after this returns.
"""
from dataclasses import replace
import logging

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.linear.kernels.gdn_prefill_reference import k31_graph_safe

logger = logging.getLogger(__name__)


BATCH_BUCKETS = (1, 2, 4, 8, 16)


def batch_bucket(dense, track_dense):
    size = max(dense.shape[0], 0 if track_dense is None else track_dense.shape[0])
    return next((b for b in BATCH_BUCKETS if 0 < size <= b), None)


@triton.jit
def _publish_valid(VALID, SLOTS, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    slot = tl.load(SLOTS + row, row < N, other=-1).to(tl.int64)
    tl.store(VALID + slot, 1, (row < N) & (slot >= 0))


class CommitBuffers:
    def __init__(self, pool, layer_id, plan, dense, track_dense, track_slots):
        self.pool = pool
        self.layer_id = layer_id
        self.li = pool.layer_map[layer_id]
        self.cfg = replace(pool.cfg)
        bucket = batch_bucket(dense, track_dense)
        self.dense = dense.new_zeros((bucket, *dense.shape[1:]))
        self.track_dense = (None if track_dense is None else
                            track_dense.new_zeros((bucket, *track_dense.shape[1:])))
        self.slots = plan.slots.new_full((bucket,), -1)
        self.ring_dst = plan.ring_dst.new_full((bucket,), -1)
        self.track_slots = (None if track_dense is None else
                            track_slots.new_full((bucket,), -1))
        self.bind(plan, dense, track_dense, track_slots)
        self.vbar = pool.vbar[self.li:self.li + 1]  # pool is the cache owner
        b, h, v, _ = self.dense.shape
        generator = torch.Generator(device=dense.device).manual_seed(0)
        self.omega = torch.randn(b, h, v, self.cfg.r + self.cfg.init_oversample,
                                 device=dense.device, generator=generator)
        fixed = pool.init_omega(b)
        if fixed is not None:
            self.omega = fixed
        self.track_omega = self.omega

    def bind(self, plan, dense, track_dense, track_slots):
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
        factors = eager([self.dense], self.vbar, self.cfg, omega=self.omega)[0]
        # Compute tracked factors before normal publication, exactly as the
        # original commit does; preserve normal/track alias overwrite order.
        tracked = None if self.track_dense is None else eager(
            [self.track_dense], self.vbar, self.cfg, omega=self.track_omega)[0]
        store_factored(*factors, p.a[i], p.U[i], p.W[i], p.count[i],
            p.stale, p.dense_of, self.slots, self.cfg.r, stale_value=0,
            dense=self.dense, ring=p.dense_ring[i], ring_dst=self.ring_dst)
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
        self.stats = dict(captured=0, replayed=0, fallback=0)

    def run(self, pool, layer_id, plan, dense, track_dense, track_slots, *, eager, policy):
        # Keep the per-layer dependency; only k31 expands beyond singleton.
        bucket = batch_bucket(dense, track_dense)
        if ((pool.cfg.init_method == "k31" and not k31_graph_safe(dense.device))
                or not dense.is_cuda or torch.cuda.is_current_stream_capturing()
                or len(plan.pending) != 1 or bucket is None
                or (pool.cfg.init_method != "k31" and bucket != 1)
                or pool.prefix_dense is not None or not pool.cfg.factored_prefix):
            self.stats['fallback'] += 1
            return False
        tensors = (dense, plan.slots, plan.ring_dst, track_dense, track_slots)
        shapes = tuple(None if x is None else ((bucket, *x.shape[1:]), x.dtype, x.device) for x in tensors)
        cfg = pool.cfg
        config = (cfg.r, cfg.rmax, cfg.dtype, cfg.init_iters, cfg.init_oversample, cfg.init_method)
        key = (layer_id, shapes, config, policy, eager,
               torch.backends.cuda.matmul.allow_tf32,
               torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
               torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= 2 * len(BATCH_BUCKETS) * len(pool.layer_ids):
                self.stats['fallback'] += 1
                return False
            buffers = CommitBuffers(pool, layer_id, plan, dense, track_dense, track_slots)
            current = torch.cuda.current_stream(dense.device)
            stream = torch.cuda.Stream(device=dense.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                buffers.evaluate(eager)
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            # PD transfer threads use independent streams and allocations.
            # Keep capture restrictions on this inference thread; global mode
            # rejects their unrelated CUDA allocation/event/sync operations.
            with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
                buffers.evaluate(eager)
            entry = (buffers, graph, stream)
            self.entries[key] = entry
            self.stats['captured'] += 1
            logger.info('GDN prefill commit graph captured: layer=%d tracked=%s batch=%d',
                        layer_id, track_dense is not None, bucket)
        else:
            entry[0].bind(plan, dense, track_dense, track_slots)
        entry[1].replay()
        self.stats['replayed'] += 1
        return True
