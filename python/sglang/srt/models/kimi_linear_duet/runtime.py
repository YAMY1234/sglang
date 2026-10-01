"""Spec-driven TP1 Kimi DUET runtime. Production remains qualification-gated."""
import copy

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.linear.kda_state_prune import KDAStatePolicy
from sglang.srt.model_executor.forward_context import get_attn_backend, get_token_to_kv_pool

from sglang.srt.duet.state_factor import project_state
from .model import _KDAEmitter, _MLAEmitter, _rms


class EmitterGraph:
    """One bounded, shape-specific graph for the fp32 state-only emitters.

    Dynamic request slots and KV locations are inputs, not capture-time values.
    Other shapes use the same eager emitter function. No global prefill graph
    may bypass the model's CPU batch decomposition.
    """

    def __init__(self):
        self.key = None
        self.graph = None

    def run(self, model, h, fb, emit):
        kb = get_attn_backend().linear_attn_backend
        metadata = kb.forward_metadata
        key = (tuple(h.shape), h.dtype, h.device, tuple(fb.extend_seq_lens_cpu),
               id(kb.req_to_token_pool))
        if self.key is not None and key != self.key:
            emit(h, fb)
            return
        if self.graph is not None:
            self.hidden.copy_(h)
            self.batch.out_cache_loc.copy_(fb.out_cache_loc)
            self.metadata.mamba_cache_indices.copy_(metadata.mamba_cache_indices)
            self.graph.replay()
            return
        self.key = key
        self.hidden = h.clone()
        self.batch = copy.copy(fb)
        self.batch.out_cache_loc = fb.out_cache_loc.clone()
        self.metadata = copy.copy(metadata)
        self.metadata.mamba_cache_indices = metadata.mamba_cache_indices.clone()
        caller = torch.cuda.current_stream(h.device)
        stream = torch.cuda.Stream(device=h.device)
        stream.wait_stream(caller)
        kb.forward_metadata = self.metadata
        try:
            with torch.cuda.stream(stream):
                # Each emitter overwrites fresh-prefix state; warming never
                # advances a decode recurrence or consumes random state.
                emit(self.hidden, self.batch)
                emit(self.hidden, self.batch)
            caller.wait_stream(stream)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                emit(self.hidden, self.batch)
            self.graph = graph
            graph.replay()
        finally:
            kb.forward_metadata = metadata
            caller.wait_stream(stream)


def kda_state(k, v, log_decay, beta):
    """The reference scan's final state. Inputs (1,T,H,D), beta (1,T,H).

    Use the same FLA path when available, with the same torch recurrence
    fallback as the reference. This path is used only for fp32 emitters.
    """
    if k.is_cuda:
        try:
            from fla.ops.kda import chunk_kda
        except ImportError:
            pass
        else:
            _, state = chunk_kda(q=k.to(v.dtype), k=k.to(v.dtype), v=v, g=log_decay,
                                 beta=beta, initial_state=None, output_final_state=True,
                                 use_qk_l2norm_in_kernel=False)
            return state.float()
    state = k.new_zeros((k.shape[0], k.shape[2], k.shape[3], v.shape[3]), dtype=torch.float32)
    for t in range(k.shape[1]):
        state = state * log_decay[:, t].exp()[..., :, None]
        error = v[:, t].float() - torch.einsum("bhk,bhkv->bhv", k[:, t].float(), state)
        state = state + k[:, t].float()[..., :, None] * (beta[:, t, :, None] * error)[..., None, :]
    return state


class DuetKDAEmitter(_KDAEmitter):
    def emit(self, h, fb):
        kb = get_attn_backend().linear_attn_backend
        cache = kb.req_to_token_pool.mamba2_layer_cache(self.layer_id)
        if self.require_fp32_state and cache.conv[0].dtype != torch.float32:
            raise ValueError("Kimi DUET reference requires a float32 convolution pool")
        slots = kb.forward_metadata.mamba_cache_indices.long()
        start, seg = 0, self.Hl * self.Dh
        for batch_index, length in enumerate(fb.extend_seq_lens_cpu):
            length = int(length)
            x = _rms(h[start:start + length].float(), self.norm_w, self.eps)
            raw = [F.linear(x, self.qkv_w[i * seg:(i + 1) * seg]) for i in range(3)]
            convolved, tails = [], []
            for i, values in enumerate(raw):
                full = F.pad(values.T[None], (self.K - 1, 0))
                out = F.conv1d(full, self.conv_w[i * seg:(i + 1) * seg, None], groups=seg)
                convolved.append(F.silu(out).transpose(1, 2).reshape(1, length, self.Hl, self.Dh))
                tails.append(full[0, :, -(self.K - 1):].T.contiguous())
            k, v = convolved[1:]
            k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            forget = F.linear(F.linear(x, self.f_a_w), self.f_b_w).view(1, length, self.Hl, self.Dh)
            g = -self.A_log.exp() * F.softplus(forget + self.dt_bias.view(1, 1, self.Hl, self.Dh))
            beta = F.linear(x, self.b_w).sigmoid()[None]
            state = kda_state(k, v, g, beta)
            slot = slots[batch_index:batch_index + 1]
            cache.conv[0][slot] = torch.cat(tails, -1)[None].to(cache.conv[0].dtype)
            cache.temporal[slot] = state.transpose(-1, -2).to(cache.temporal.dtype)
            start += length
        pruner = kb.state_pruner
        if pruner is None and not self.prune_state:
            return
        if not isinstance(pruner, DuetStatePruner):
            raise RuntimeError("HF DUET state policy was not installed; refusing unpruned service")
        pruner.mark_prefix(self.layer_id, slots)


class DuetMLAEmitter(_MLAEmitter):
    def emit(self, h, fb, target_layer):
        c, kr = self.latent(h.float())
        attn = target_layer.self_attn
        handle = getattr(attn, "attn_mqa", None) or attn.attn_mha
        get_token_to_kv_pool().set_mla_kv_buffer(
            handle, fb.out_cache_loc, c.to(h.dtype).unsqueeze(1).contiguous(),
            kr.to(h.dtype).unsqueeze(1).contiguous())


class DuetStatePruner(KDAStatePolicy):
    """Existing pool lifecycle/hooks, reference current-state sink + warm basis.

    Functional acceptance fixes B=1, so random subspace initialisation has the
    same shape and seed as the B=1 reference. A basis belongs to a layer AND slot.
    """
    def __init__(self, pool, directions, spec, layer_ids, *, graph_safe=False):
        super().__init__(pool, {l: directions[l] for l in layer_ids},
                         spec["state_rank"], max(1, spec["state_every"]),
                         prefix_only=spec["state_every"] == 0)
        self.explicit_sink = spec.get("state_sink", "explicit") == "explicit"
        self.warm = {}
        self.graph_safe = graph_safe

    def _cut_many(self, plan):
        for layer_index, slots in plan:
            for slot in slots.tolist():
                state = self.states[layer_index, slot:slot + 1].transpose(-1, -2)
                fresh = bool(self.pending_prefix[layer_index, slot])
                key = (layer_index, slot)
                projected, basis = project_state(state, self.vbar[layer_index], self.rank,
                                                  None if fresh else self.warm.get(key),
                                                  explicit=self.explicit_sink)
                if basis is None:
                    self.warm.pop(key, None)
                else:
                    self.warm[key] = basis
                self.states[layer_index, slot:slot + 1] = projected.transpose(-1, -2)

    def reset_slots(self, indices):
        super().reset_slots(indices)
        if hasattr(self, "warm"):
            selected = set(torch.as_tensor(indices).reshape(-1).tolist())
            self.warm = {k: v for k, v in self.warm.items() if k[1] not in selected}

    def copy_slots(self, src, dst):
        super().copy_slots(src, dst)
        sources = torch.as_tensor(src).reshape(-1).tolist()
        targets = torch.as_tensor(dst).reshape(-1).tolist()
        saved = {(layer, target): self.warm.get((layer, source))
                 for source, target in zip(sources, targets) for layer in self.layer_map.values()}
        for key, value in saved.items():
            if value is None:
                self.warm.pop(key, None)
            else:
                self.warm[key] = value.clone()

    def get_cpu_slots(self, indices):
        selected = torch.as_tensor(indices).reshape(-1).tolist()
        bases = {(layer, i): self.warm[(layer, slot)].cpu()
                 for i, slot in enumerate(selected) for layer in self.layer_map.values()
                 if (layer, slot) in self.warm}
        return super().get_cpu_slots(indices), bases

    def load_cpu_slots(self, data, indices):
        self.reset_slots(indices)
        ordinary, bases = data
        super().load_cpu_slots(ordinary, indices)
        selected = torch.as_tensor(indices).reshape(-1).tolist()
        for (layer, i), value in bases.items():
            self.warm[(layer, selected[i])] = value.to(self.states.device)

    def decode(self, dispatcher, layer, qkv, a, b, slots, query_start_loc):
        # The DUET sink is projected from S at each cut; no auxiliary recurrence.
        index = self.layer_map[layer.layer_id]
        if self.graph_safe:
            # Fixed-shape operations are replayed with each graph invocation.
            # Padding contributes zero even when several -1 entries map to 0.
            self.count[index].scatter_add_(
                0, slots.long().clamp_min(0), (slots >= 0).to(self.count.dtype))
        else:
            # Keep the qualified reference path unchanged.
            valid = slots[slots >= 0].long()
            self.count[index, valid] += 1

    def mark_prefix(self, layer_id, slots):
        index = self.layer_map[layer_id]
        if self.graph_safe:
            # Advanced-index scalar assignment stages a CPU scalar. index_fill
            # passes the value directly to the device kernel during capture.
            self.pending_prefix[index].index_fill_(0, slots.long(), True)
            self.count[index].index_fill_(0, slots.long(), 0)
        else:
            self.pending_prefix[index, slots] = True
            self.count[index, slots] = 0

    def extend(self, dispatcher, layer, batch, q, k, v, g, beta, slots,
               query_start_loc, beta_is_raw):
        self.mark_prefix(layer.layer_id, slots)


def make_pruner(model, runner):
    from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel, get_spec
    if get_parallel().tp_size != 1 or get_spec().speculative_algorithm is not None:
        raise ValueError("HF DUET functional mode requires TP1 and no speculative decoding")
    if not get_memory().disable_radix_cache or get_disagg().disaggregation_mode != "null":
        raise ValueError("HF DUET functional mode requires radix off and AGG")
    if model.duet_options.decode_ssm_r == 0:
        return None
    return DuetStatePruner(runner.req_to_token_pool, model.duet_sink_dir,
                          model.duet_options.effective_spec(model.duet_report["spec"]),
                          model.config.linear_layer_ids,
                          graph_safe=model.duet_profile == "production")
