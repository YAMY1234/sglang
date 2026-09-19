"""Low-bit dense content plus an fp32 watermark; K1 slot/ring lifecycle."""
import logging

import torch

from .gdn_factored_pool import FactoredGDNPool
from sglang.srt.layers.attention.linear.kernels.gdn_dense_quant import pack, unpack, packed_step, track_tensor

logger = logging.getLogger(__name__)


class DenseQuantGDNPool(FactoredGDNPool):
    def __init__(self, *, size, cache_params, mamba_layer_ids, device, cfg, tp_rank):
        # Do not allocate a factor pool (or a full-sized dense shadow).
        self.cfg, self.size, self.device = cfg, size, device
        self.layer_ids = list(mamba_layer_ids)
        self.layer_map = {lid: i for i, lid in enumerate(self.layer_ids)}
        self.hv, self.v, self.k = cache_params.shape.temporal
        assert self.v == self.k == 128
        self.mode = {"dense_fp8": 2, "dense_int8": 3, "dense_int4": 4}[cfg.precision]
        layers, slots, hv = len(self.layer_ids), size + 1, self.hv
        self.a = torch.zeros(layers, slots, hv, 128, dtype=torch.float32, device=device)
        self.content = torch.zeros(layers, slots, hv, 128, 64 if self.mode == 4 else 128,
                                   dtype=torch.uint8, device=device)
        self.scales = torch.zeros(layers, slots, hv, 128, 4 if self.mode == 4 else 1,
                                  dtype=torch.float32, device=device)
        self.stale = torch.ones(slots, dtype=torch.int32, device=device)
        self.dense_of = torch.full((slots,), -1, dtype=torch.int32, device=device)
        self.dense_ring = torch.zeros(layers, cfg.ring, hv, 128, 128, dtype=torch.float32, device=device)
        self.ring_owner = [-1] * cfg.ring
        self.ring_lru = list(range(cfg.ring))
        self.vbar = self._load_vbar(cfg.vbar_path, tp_rank)
        self.stats = {"extends": 0, "rows": 0, "ring_src": 0, "ring_miss": 0, "densified": 0}
        logger.info("Dense quantized GDN pool: %s slots=%d state_bytes_per_layer=%d allocated_bytes=%d",
                    cfg.precision, slots, cfg.state_bytes_per_layer(cache_params.shape), self.mem_usage_bytes())

    def _states(self):
        return (self.a, self.content, self.scales)

    def reset_slots(self, indices):
        for t in self._states():
            t[:, indices] = 0
        self.stale[indices] = 1
        self.dense_of[indices] = -1

    def copy_slots(self, src_index, dst_index):
        for t in self._states():
            t[:, dst_index] = t[:, src_index]
        self.stale[dst_index] = 1
        self.dense_of[dst_index] = -1

    def get_cpu_slots(self, indices):
        return tuple(t[:, indices].to("cpu", non_blocking=True) for t in self._states())

    def load_cpu_slots(self, data, indices):
        if data is None:
            return
        for target, source in zip(self._states(), data):
            target[:, indices] = source.to(self.device, non_blocking=True)
        self.stale[indices] = 1
        self.dense_of[indices] = -1

    def iter_transfer_state_entries(self):
        for lid, li in self.layer_map.items():
            for name, t in zip(("a", "content", "scales"), self._states()):
                yield ("gdn_dense_quant_" + name, t[li], 0, lid)

    def mem_usage_bytes(self):
        return sum(t.numel() * t.element_size() for t in (*self._states(), self.stale, self.dense_of,
                                                          self.dense_ring, self.vbar))

    def initial_dense(self, layer_id, plan):
        li = self.layer_map[layer_id]
        s = unpack(self.a[li], self.content[li], self.scales[li], self.vbar[li], plan.slots, self.mode)
        if plan.n_ring_src:
            s = torch.where(plan.use_ring[:, None, None, None], self.dense_ring[li][plan.ring_src], s)
        return s.contiguous()

    def commit_extend(self, layer_id, plan, s_final):
        self.write_factored_dense(layer_id, plan.slots, s_final)
        valid = plan.slots[plan.slots >= 0]
        self.stale[valid] = 0
        # plan_extend already installed the new ownership mapping.
        self.dense_of[valid] = plan.ring_dst[plan.slots >= 0].to(torch.int32)
        li = self.layer_map[layer_id]
        if plan.ring_dst_rows.numel():
            self.dense_ring[li][plan.ring_dst[plan.ring_dst_rows]] = s_final[plan.ring_dst_rows]

    def write_factored_dense(self, layer_id, slots, s_dense):
        # Name retained for the existing backend's generic extend/checkpoint interface.
        li = self.layer_map[layer_id]
        pack(s_dense.float().contiguous(), self.a[li], self.content[li], self.scales[li],
             self.vbar[li], slots, self.mode)
        valid = slots[slots >= 0]
        self.stale[valid] = 1
        self.dense_of[valid] = -1

    def copy_slots_layer(self, layer_id, src, dst):
        li = self.layer_map[layer_id]
        for t in self._states():
            t[li, dst] = t[li, src]
        self.stale[dst] = 1
        self.dense_of[dst] = -1

    def decode(self, layer, mixed, ag, bg, slots):
        li = self.layer_map[layer.layer_id]
        return packed_step(mixed, ag, bg, layer=layer, a=self.a[li], c=self.content[li], scales=self.scales[li],
                           vb=self.vbar[li], slots=slots, stale=self.stale, mode=self.mode)

    def track_copy(self, src, mask, dst):
        for t in self._states():
            track_tensor(t, src, mask, dst)
        # Copy helpers update only tracked destinations; stale/dense_of are small
        # slot metadata, handled by the same masked byte-independent kernel.
        # Avoid factor-specific assumptions: a separate metadata kernel follows.
        _track_metadata[(src.numel(),)](src, mask, dst, self.stale, self.dense_of)

    def dump_slots(self, layer_id, slots, meta, out_dir, tag):
        import os
        os.makedirs(out_dir, exist_ok=True)
        li = self.layer_map[layer_id]
        torch.save({"a": self.a[li, slots].cpu(), "content": self.content[li, slots].cpu(),
                    "scales": self.scales[li, slots].cpu(), "mode": self.mode, **meta},
                   os.path.join(out_dir, f"{tag}_L{layer_id}.pt"))


import triton
import triton.language as tl


@triton.jit
def _track_metadata(SRC, MASK, DST, STALE, DENSE_OF):
    row = tl.program_id(0)
    src, dst = tl.load(SRC + row), tl.load(DST + row)
    if tl.load(MASK + row) and src >= 0 and dst >= 0:
        tl.store(STALE + dst, 1)
        tl.store(DENSE_OF + dst, -1)
