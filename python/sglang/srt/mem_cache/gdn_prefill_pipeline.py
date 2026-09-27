"""Prefill-only layer groups with final-group metadata publication.

The zero-vbar recipe preserves factorize_layers bitwise under grouping. Loaded
vbar, non-singleton batches, partial layer ranges and larger workspaces retain
the existing path. Each group owns its graph and bind buffers; all side-stream
work remains ordered on the existing commit stream.
"""
from dataclasses import replace
import logging
from types import SimpleNamespace

import torch

from .gdn_prefill_commit_graph import PrefillCommitGraph

logger = logging.getLogger(__name__)


class PrefillLayerCommitPipeline:
    graph_type = PrefillCommitGraph

    def __init__(self, layers):
        if layers not in (12, 18):
            raise ValueError("prefill pipeline groups must be 12 or 18 layers")
        self.layers = layers
        self.entries = {}
        self.replayed = 0
        self.logged = False

    def eligible(self, pool, plan, dense, tracked):
        # _load_vbar(None) constructs zeros. No device reduction or implicit
        # synchronization is needed to prove this immutable recipe property.
        size = dense.nbytes + (tracked.nbytes if tracked is not None else 0)
        return (pool.cfg.vbar_path is None and pool.cfg.init_method == "iter"
                and len(pool.layer_ids) == 36 and plan.next_layer == 0
                and plan.last_layer == 35 and dense.shape[0] == 1
                and (tracked is None or tracked.shape[0] == 1)
                and pool.prefix_layer_count() == 36
                and pool.prefix_dense is None and pool.batch_prefill_final_copy
                and size * 36 + pool.vbar.nbytes <= PrefillCommitGraph.MAX_INPUT_BYTES
                and size * 36 <= pool.batch_prefill_max_bytes)

    def _entry(self, pool, first, last):
        key = (id(pool), first, last)
        if key not in self.entries:
            final = last == len(pool.layer_ids) - 1
            # Slice contiguous layer ranges; no new persistent state or math.
            fields = {name: getattr(pool, name)[first:last+1] for name in
                      ("a", "U", "W", "count", "dense_ring", "vbar")}
            fields.update(cfg=pool.cfg, layer_ids=pool.layer_ids[first:last+1],
                          stale=pool.stale, dense_of=pool.dense_of,
                          prefix_valid=pool.prefix_valid if final else None,
                          dense_required=pool.dense_required if final else None,
                          batch_prefill_final_copy=True, prefix_dense=None,
                          prefix_layer_count=lambda: last-first+1)
            self.entries[key] = (SimpleNamespace(**fields), self.graph_type())
        return self.entries[key]

    def run(self, pool, plan, track_slots, *, first, last, factorize, policy, replay_stream):
        view, graph = self._entry(pool, first, last)
        group = replace(plan,
                        stage=None if plan.stage is None else plan.stage[first:last+1],
                        track_stage=None if plan.track_stage is None else plan.track_stage[first:last+1])
        ran = graph.run(view, group, track_slots, factorize=factorize,
                        policy=policy, replay_stream=replay_stream)
        if not ran:
            # If a later group's cache/shape falls back, order eager writes
            # after prior groups, including their shared slot metadata.
            if replay_stream is not None:
                torch.cuda.current_stream(plan.slots.device).wait_stream(replay_stream)
            return False
        self.replayed += 1
        if not self.logged:
            logger.info("Factored GDN prefill pipeline groups: %d", self.layers)
            self.logged = True
        return True
