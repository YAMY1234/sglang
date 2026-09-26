"""Default-off singleton prefix factorization plus publication graph.

The pool owns this cache and every captured pool allocation. Slot and ring IDs
are copied into owned device controls before replay. The caller keeps its host
plan bookkeeping and still processes the last token only after this returns.
"""
from dataclasses import replace
import logging

import torch

logger = logging.getLogger(__name__)


class CommitBuffers:
    def __init__(self, pool, layer_id, plan, dense, track_dense, track_slots):
        self.pool = pool
        self.layer_id = layer_id
        self.li = pool.layer_map[layer_id]
        self.cfg = replace(pool.cfg)
        self.dense = dense.clone()
        self.track_dense = None if track_dense is None else track_dense.clone()
        self.slots = plan.slots.clone()
        self.ring_dst = plan.ring_dst.clone()
        self.track_slots = None if track_dense is None else track_slots.clone()
        self.vbar = pool.vbar[self.li:self.li + 1]  # pool is the cache owner
        b, h, v, _ = dense.shape
        generator = torch.Generator(device=dense.device).manual_seed(0)
        self.omega = torch.randn(b, h, v, self.cfg.r + self.cfg.init_oversample,
                                 device=dense.device, generator=generator)
        self.track_omega = self.omega
        if track_dense is not None and track_dense.shape[0] != b:
            generator = torch.Generator(device=dense.device).manual_seed(0)
            self.track_omega = torch.randn(track_dense.shape[0], h, v,
                self.cfg.r + self.cfg.init_oversample, device=dense.device, generator=generator)

    def bind(self, plan, dense, track_dense, track_slots):
        self.dense.copy_(dense)
        self.slots.copy_(plan.slots)
        self.ring_dst.copy_(plan.ring_dst)
        if self.track_dense is not None:
            self.track_dense.copy_(track_dense)
            self.track_slots.copy_(track_slots)

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
        # Eligibility excludes exact dense snapshots. The native final-layer
        # scalar assignment stages a CPU tensor; index_fill_ writes the same
        # ones without a host-to-device copy during capture.
        p = self.pool
        if p.prefix_valid is not None and self.li == p.prefix_layer_count() - 1:
            p.prefix_valid.index_fill_(0, slots.long().clamp_min(0), 1)


class PrefillCommitGraph:
    def __init__(self):
        self.entries = {}
        self.stats = dict(captured=0, replayed=0, fallback=0)

    def run(self, pool, layer_id, plan, dense, track_dense, track_slots, *, eager, policy):
        # A bounded per-layer singleton path. Batched layer groups, exact-prefix
        # snapshots, foreign graph capture and larger request batches stay native.
        if (not dense.is_cuda or torch.cuda.is_current_stream_capturing()
                or len(plan.pending) != 1 or dense.shape[0] != 1
                or (track_dense is not None and track_dense.shape[0] != 1)
                or pool.prefix_dense is not None or not pool.cfg.factored_prefix):
            self.stats['fallback'] += 1
            return False
        tensors = (dense, plan.slots, plan.ring_dst, track_dense, track_slots)
        shapes = tuple(None if x is None else (tuple(x.shape), x.dtype, x.device,
                                               tuple(x.stride())) for x in tensors)
        cfg = pool.cfg
        config = (cfg.r, cfg.rmax, cfg.dtype, cfg.init_iters, cfg.init_oversample, cfg.init_method)
        key = (layer_id, shapes, config, policy, eager,
               torch.backends.cuda.matmul.allow_tf32,
               torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
               torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= 2 * len(pool.layer_ids):
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
            with torch.cuda.graph(graph, stream=stream):
                buffers.evaluate(eager)
            entry = (buffers, graph, stream)
            self.entries[key] = entry
            self.stats['captured'] += 1
            logger.info('GDN prefill commit graph captured: layer=%d tracked=%s',
                        layer_id, track_dense is not None)
        else:
            entry[0].bind(plan, dense, track_dense, track_slots)
        entry[1].replay()
        self.stats['replayed'] += 1
        return True
