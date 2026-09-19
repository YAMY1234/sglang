"""Per-column FP8 factor pool with bf16 active-batch K3-A scratch."""
import torch

from .gdn_factored_pool import FactoredGDNPool, factorize_dense, densify
from .gdn_dense_quant_pool import _track_metadata
from sglang.srt.layers.attention.linear.kernels.gdn_dense_quant import track_tensor
from sglang.srt.layers.attention.linear.kernels.gdn_factor_fp8 import allocate_batch, transfer


class FactorFP8GDNPool(FactoredGDNPool):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.u_scales = torch.zeros(self.U.shape[:-1], device=self.device, dtype=torch.float32)
        self.w_scales = torch.zeros(self.W.shape[:-1], device=self.device, dtype=torch.float32)

    def _all_states(self):
        return (self.a, self.U, self.W, self.count, self.u_scales, self.w_scales)

    def reset_slots(self, indices):
        super().reset_slots(indices)
        self.u_scales[:, indices] = 0
        self.w_scales[:, indices] = 0

    def copy_slots(self, src_index, dst_index):
        super().copy_slots(src_index, dst_index)
        self.u_scales[:, dst_index] = self.u_scales[:, src_index]
        self.w_scales[:, dst_index] = self.w_scales[:, src_index]

    def get_cpu_slots(self, indices):
        return tuple(t[:, indices].to("cpu", non_blocking=True) for t in self._all_states())

    def load_cpu_slots(self, data, indices):
        if data is None:
            return
        for target, source in zip(self._all_states(), data):
            target[:, indices] = source.to(self.device, non_blocking=True)
        self.stale[indices] = 1
        self.dense_of[indices] = -1

    def iter_transfer_state_entries(self):
        yield from super().iter_transfer_state_entries()
        for lid, li in self.layer_map.items():
            yield ("gdn_factor_fp8_u_scale", self.u_scales[li], 0, lid)
            yield ("gdn_factor_fp8_w_scale", self.w_scales[li], 0, lid)

    def mem_usage_bytes(self):
        base = super().mem_usage_bytes()
        return base + sum(t.numel() * t.element_size() for t in (self.u_scales, self.w_scales))

    def _transfer(self, batch, slots, store=False, li=None, **kwargs):
        tensors = self._all_states()
        if li is not None:
            tensors = tuple(t[li:li + 1] for t in tensors)
        transfer(*tensors, batch, slots, self.stale, store, **kwargs)

    def initial_dense(self, layer_id, plan):
        li = self.layer_map[layer_id]
        batch = allocate_batch(1, plan.slots.numel(), self.hv, self.cfg.rmax, self.device)
        self._transfer(batch, plan.slots, li=li)
        s = densify(*(t[0] for t in batch), self.vbar[li])
        if plan.n_ring_src:
            s = torch.where(plan.use_ring[:, None, None, None], self.dense_ring[li][plan.ring_src], s)
        return s.contiguous()

    def write_factored_dense(self, layer_id, slots, s_dense):
        li = self.layer_map[layer_id]
        a, u, w = factorize_dense(s_dense.float(), self.vbar[li], self.cfg.r, self.cfg.rmax,
                                  torch.bfloat16, iters=self.cfg.init_iters, oversample=self.cfg.init_oversample)
        count = torch.full(a.shape[:-1], self.cfg.r, device=self.device, dtype=torch.int32)
        self._transfer(tuple(t[None] for t in (a, u, w, count)), slots, store=True, li=li)
        valid = slots[slots >= 0]
        self.dense_of[valid] = -1

    def commit_extend(self, layer_id, plan, s_final):
        self.write_factored_dense(layer_id, plan.slots, s_final)
        valid = plan.slots >= 0
        self.stale[plan.slots[valid]] = 0
        self.dense_of[plan.slots[valid]] = plan.ring_dst[valid].to(torch.int32)
        li = self.layer_map[layer_id]
        if plan.ring_dst_rows.numel():
            self.dense_ring[li][plan.ring_dst[plan.ring_dst_rows]] = s_final[plan.ring_dst_rows]

    def copy_slots_layer(self, layer_id, src, dst):
        li = self.layer_map[layer_id]
        for t in self._all_states():
            t[li, dst] = t[li, src]
        self.stale[dst] = 1
        self.dense_of[dst] = -1

    def track_copy(self, src, mask, dst):
        for t in self._all_states():
            track_tensor(t, src, mask, dst)
        _track_metadata[(src.numel(),)](src, mask, dst, self.stale, self.dense_of)

    def prepare_decode_scratch(self, max_batch):
        # One shared backing allocation, reused by all graph batch sizes. Every
        # decode gathers before reading it; graph replays are serialized.
        self._scratch_capacity = int(max_batch)
        self._scratch = allocate_batch(len(self.layer_ids), self._scratch_capacity,
                                       self.hv, self.cfg.rmax, self.device)

    def decode_scratch_bytes(self):
        return sum(t.numel() * t.element_size() for t in getattr(self, "_scratch", ()))

    def _batch_view(self, b):
        if not hasattr(self, "_scratch"):
            self.prepare_decode_scratch(b)  # standalone unit tests
        if b > self._scratch_capacity:
            raise ValueError(f"FP8 factor decode batch {b} exceeds scratch capacity {self._scratch_capacity}")
        l = len(self.layer_ids)
        shapes = [(l, b, self.hv, 128), (l, b, self.hv, self.cfg.rmax, 128),
                  (l, b, self.hv, self.cfg.rmax, 128), (l, b, self.hv)]
        import math
        return tuple(t.reshape(-1)[:math.prod(shape)].view(shape) for t, shape in zip(self._scratch, shapes))

    def decode(self, layer, mixed, ag, bg, slots, single_layer=False):
        from sglang.srt.layers.attention.linear.kernels.gdn_factored import factored_packed_decode, factored_expiry_truncate_layers
        li = self.layer_map[layer.layer_id]
        b = slots.numel()
        if single_layer or li == 0:
            batch = allocate_batch(1, b, self.hv, self.cfg.rmax, self.device) if single_layer else self._batch_view(b)
            local = torch.empty(b, device=self.device, dtype=torch.int32)
            stale = torch.empty(b, device=self.device, dtype=torch.int32)
            self._transfer(batch, slots, li=li if single_layer else None, local=local,
                           batch_stale=stale, decode=True)
            if not single_layer:
                self._decode_batch = batch, local, stale
        else:
            batch, local, stale = self._decode_batch
        index = 0 if single_layer else li
        a, u, w, count = (t[index] for t in batch)
        batched = not single_layer and self.cfg.r == 16
        out = factored_packed_decode(mixed, ag, bg, A_log=layer.A_log, dt_bias=layer.dt_bias, scale=128 ** -.5,
              vbar=self.vbar[li], fa=a, fu=u, fw=w, fcount=count, stale=stale, ssm_state_indices=local,
              num_q_heads=layer.num_q_heads, num_v_heads=layer.num_v_heads, head_k_dim=128, head_v_dim=128,
              r=self.cfg.r, rfull=self.cfg.rfull, truncate=not batched, **self.cfg.kernel_kwargs())
        if single_layer or self.is_last_layer(layer.layer_id):
            if batched:
                factored_expiry_truncate_layers(batch[1], batch[2], batch[3], local, self.cfg.r, self.cfg.rfull)
            self._transfer(batch, slots, store=True, li=li if single_layer else None, decode=True)
        return out
