"""Opt-in Kimi accuracy experiment: dense fp32 pool + exact sink + rank cuts.

No compressed storage or performance claim. The auxiliary value e_0 is passed
through the SAME KDA kernel as the main state. Solver calls stay outside CUDA
graph replay; a cut after the current token's read affects the next token only.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)


class KDAStatePruner:
    def __init__(self, pool, vbar, rank, every=8, prefix_only=False):
        self.pool = pool
        self.states = pool.mamba_pool.mamba_cache.temporal
        if self.states.dtype != torch.float32 or self.states.ndim != 5:
            raise ValueError("KDA state pruning requires the dense fp32 Mamba pool")
        layers, slots, heads, values, keys = self.states.shape
        if not 0 < rank <= min(values, keys) or every < 1:
            raise ValueError("invalid KDA content rank or pruning interval")
        self.rank, self.every, self.prefix_only = rank, every, prefix_only
        self.layer_map = {int(l): pool.mamba2_layer_index(int(l)) for l in vbar}
        if sorted(self.layer_map.values()) != list(range(layers)):
            raise ValueError("calibration must cover every resident KDA layer exactly")
        self.vbar = torch.empty(
            (layers, heads, values), device=self.states.device, dtype=torch.float32
        )
        for l, index in self.layer_map.items():
            value = vbar[l] if l in vbar else vbar[str(l)]
            if tuple(value.shape) != (heads, values) or not torch.isfinite(value).all():
                raise ValueError(f"invalid vbar for layer {l}")
            self.vbar[index].copy_(value)
        # V=16 is the kernel's auxiliary value width; only row zero carries a.
        self.sinks = torch.zeros(
            (layers, slots, heads, 16, keys),
            device=self.states.device,
            dtype=torch.float32,
        )
        self.count = torch.zeros(
            (layers, slots), device=self.states.device, dtype=torch.int32
        )
        self.pending_prefix = torch.zeros_like(self.count, dtype=torch.bool)
        self.prefix_cuts = 0
        self.decode_cuts = 0
        pool.mamba_pool.register_slot_state(self)
        logger.info(
            "KDA state pruning: content r=%d W=%d prefix_only=%s dense_pool=%d bytes aux=%d bytes slots=%d",
            rank,
            every,
            prefix_only,
            self.states.numel() * 4,
            self.sinks.numel() * 4,
            slots,
        )

    @classmethod
    def from_runner(cls, runner):
        rank = int(os.environ.get("SGLANG_KDA_STATE_PRUNE_RANK", "0"))
        if rank == 0:
            return None
        from sglang.srt.runtime_context import (
            get_disagg,
            get_memory,
            get_parallel,
            get_spec,
        )

        if get_parallel().tp_size != 1 or get_spec().speculative_algorithm is not None:
            raise ValueError(
                "KDA state pruning accuracy mode requires TP1 and no speculative decoding"
            )
        if (
            not get_memory().disable_radix_cache
            or get_disagg().disaggregation_mode != "null"
        ):
            raise ValueError(
                "KDA state pruning accuracy mode requires radix off and a single AGG engine"
            )
        path = os.environ.get("SGLANG_KDA_STATE_PRUNE_CALIB")
        if not path:
            raise ValueError("SGLANG_KDA_STATE_PRUNE_CALIB is required")
        calibration = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            runner.req_to_token_pool,
            calibration["vbar"],
            rank,
            every=int(os.environ.get("SGLANG_KDA_STATE_PRUNE_EVERY", "8")),
            prefix_only=os.environ.get("SGLANG_KDA_STATE_PRUNE_PREFIX_ONLY", "0")
            == "1",
        )

    def reset_slots(self, indices):
        self.sinks[:, indices] = 0
        self.count[:, indices] = 0
        self.pending_prefix[:, indices] = False

    def copy_slots(self, src, dst):
        self.sinks[:, dst] = self.sinks[:, src]
        self.count[:, dst] = self.count[:, src]
        self.pending_prefix[:, dst] = self.pending_prefix[:, src]

    def get_cpu_slots(self, indices):
        return tuple(
            x[:, indices].cpu() for x in (self.sinks, self.count, self.pending_prefix)
        )

    def load_cpu_slots(self, data, indices):
        for target, saved in zip((self.sinks, self.count, self.pending_prefix), data):
            target[:, indices] = saved.to(target.device)

    def slots(self, batch):
        return self.pool.get_mamba_indices(
            batch.req_pool_indices[: batch.batch_size]
        ).long()

    def _cut_many(self, plan):
        # Pool layout is (slot, head, VALUE, KEY): transpose to the mathematical
        # (key, value) layout used by S = a vbar^T before projecting.
        anchors, contents = [], []
        for layer_index, slots in plan:
            state = self.states[layer_index, slots].transpose(-1, -2)
            a = self.sinks[layer_index, slots, :, 0, :]
            anchor = a[..., :, None] * self.vbar[layer_index, :, None, :]
            anchors.append(anchor)
            contents.append(state - anchor)
        content = torch.cat(contents, dim=0)
        flat = content.reshape(-1, *content.shape[-2:])
        projected = torch.empty_like(flat)
        # Batch due heads across layers, while bounding solver scratch. All
        # readouts have completed, so these independent cuts commute exactly.
        for start in range(0, flat.shape[0], 2048):
            x = flat[start : start + 2048].double()
            gram = x @ x.transpose(-1, -2)
            try:
                _, vectors = torch.linalg.eigh(gram)
            except torch.linalg.LinAlgError:
                if not torch.isfinite(gram).all():
                    raise
                logger.warning("KDA Gram-eigh convergence failure; using CPU LAPACK")
                _, vectors = torch.linalg.eigh(gram.cpu())
                vectors = vectors.to(x.device)
            u = vectors[..., -self.rank :]
            projected[start : start + 2048] = (u @ (u.transpose(-1, -2) @ x)).float()
        projected = projected.reshape_as(content)
        offset = 0
        for (layer_index, slots), anchor in zip(plan, anchors):
            count = len(slots)
            restored = anchor + projected[offset : offset + count]
            self.states[layer_index, slots] = restored.transpose(-1, -2)
            offset += count

    def flush(self, slots, *, prefix, layer_id=None):
        if not prefix and self.prefix_only:
            return
        # These host decisions occur outside capture/replay. Each slot's count
        # is independent, including asynchronous admission and slot reuse.
        slots = slots.long()
        slots = slots[slots >= 0].unique()
        indices = (
            list(range(self.states.shape[0]))
            if layer_id is None
            else [self.layer_map[layer_id]]
        )
        masks = (
            (
                self.pending_prefix[indices][:, slots]
                if prefix
                else self.count[indices][:, slots] >= self.every
            )
            .cpu()
            .tolist()
        )
        plan = []
        for index, mask in zip(indices, masks):
            chosen = [i for i, yes in enumerate(mask) if yes]
            if not chosen:
                continue
            selected = slots[torch.tensor(chosen, device=slots.device)]
            plan.append((index, selected))
        if not plan:
            return
        self._cut_many(plan)
        for index, selected in plan:
            self.count[index, selected] = 0
            if prefix:
                self.pending_prefix[index, selected] = False
                self.prefix_cuts += len(selected)
            else:
                self.decode_cuts += len(selected)

    def before_graph(self, batch):
        self.flush(self.slots(batch), prefix=True)

    def after_graph(self, batch):
        self.flush(self.slots(batch), prefix=False)

    def decode(self, dispatcher, layer, qkv, a, b, slots, query_start_loc):
        index = self.layer_map[layer.layer_id]
        q, k, _ = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        auxiliary = qkv.new_zeros((qkv.shape[0], layer.num_v_heads, 16))
        auxiliary[..., 0] = 1
        # Kimi's stock fast path has no safe-gate lower bound.
        dispatcher.packed_decode(
            mixed_qkv=torch.cat((q, k, auxiliary.flatten(1)), dim=-1),
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            scale=layer.head_k_dim**-0.5,
            ssm_states=self.sinks[index],
            cache_indices=slots,
            num_v_heads=layer.num_v_heads,
            head_v_dim=16,
            lower_bound=None,
        )
        safe_slots = slots.long().clamp_min(0)
        self.count[index, safe_slots] = self.count[index, safe_slots] + 1

    def extend(
        self,
        dispatcher,
        layer,
        batch,
        q,
        k,
        v,
        g,
        beta,
        slots,
        query_start_loc,
        beta_is_raw,
    ):
        index = self.layer_map[layer.layer_id]
        # A fresh request may reuse a slot. Continuation chunks retain the exact
        # sink; prefix cuts are deferred until the first decode consumes it.
        fresh = batch.extend_prefix_lens == 0
        selected = slots[fresh]
        self.sinks[index, selected] = 0
        auxiliary = v.new_zeros((*v.shape[:-1], 16))
        auxiliary[..., 0] = 1
        dispatcher.extend(
            q=q,
            k=k,
            v=auxiliary,
            g=g,
            beta=beta,
            ssm_states=self.sinks[index],
            cache_indices=slots,
            query_start_loc=query_start_loc,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            lower_bound=layer.lower_bound,
            beta_is_raw=beta_is_raw,
            extend_seq_lens_cpu=batch.extend_seq_lens_cpu,
        )
        self.pending_prefix[index, slots] = True
        self.count[index, slots] = 0
