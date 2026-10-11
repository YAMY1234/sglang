"""Spec-driven TP1 Kimi DUET runtime. Production remains qualification-gated."""

import copy

import torch
import torch.nn.functional as F

from sglang.srt.duet.state_factor import project_state
from sglang.srt.layers.attention.linear.kda_state_prune import KDAStatePolicy
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)

from . import fast_state
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
        key = (
            tuple(h.shape),
            h.dtype,
            h.device,
            tuple(fb.extend_seq_lens_cpu),
            id(kb.req_to_token_pool),
        )
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


def kda_state(k, v, log_decay, beta, initial_state=None):
    """The reference scan's final state. Inputs (B,T,H,D), beta (B,T,H).

    Use the same FLA path when available, with the same torch recurrence
    fallback as the reference. This path is used only for fp32 emitters.
    """
    if k.is_cuda:
        try:
            from fla.ops.kda import chunk_kda
        except ImportError:
            pass
        else:
            _, state = chunk_kda(
                q=k.to(v.dtype),
                k=k.to(v.dtype),
                v=v,
                g=log_decay,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
            )
            return state.float()
    state = (
        k.new_zeros(
            (k.shape[0], k.shape[2], k.shape[3], v.shape[3]), dtype=torch.float32
        )
        if initial_state is None
        else initial_state.float()
    )
    for t in range(k.shape[1]):
        state = state * log_decay[:, t].exp()[..., :, None]
        error = v[:, t].float() - torch.einsum("bhk,bhkv->bhv", k[:, t].float(), state)
        state = (
            state
            + k[:, t].float()[..., :, None]
            * (beta[:, t, :, None] * error)[..., None, :]
        )
    return state


class DuetKDAEmitter(_KDAEmitter):
    def emit(self, h, fb):
        kb = get_attn_backend().linear_attn_backend
        cache = kb.req_to_token_pool.mamba2_layer_cache(self.layer_id)
        if self.require_fp32_state and cache.conv[0].dtype != torch.float32:
            raise ValueError("Kimi DUET reference requires a float32 convolution pool")
        slots = kb.forward_metadata.mamba_cache_indices.long()
        if self.require_fp32_state:
            self._emit_reference(h, fb, cache, slots)
        else:
            self._emit_batched(h, fb, cache, slots)
        pruner = kb.state_pruner
        if pruner is None and not self.prune_state:
            return
        if not isinstance(pruner, DuetStatePruner):
            raise RuntimeError(
                "HF DUET state policy was not installed; refusing unpruned service"
            )
        pruner.mark_prefix(self.layer_id, slots)

    # Total tokens across a bucket's requests, not a per-request allowance.
    # Keep non-final blocks aligned to FLA's 64-token internal chunks.
    max_emitter_tokens = 4096

    def _emitter_workspace(self, h, batch, capacity):
        """One emit-local workspace, reused across blocks, never per-layer cached."""
        seg = self.Hl * self.Dh
        tokens = batch * capacity
        return {
            "qkv": h.new_empty(tokens, 3 * seg, dtype=torch.float32),
            "forget_a": h.new_empty(tokens, self.Dh, dtype=torch.float32),
            "forget": h.new_empty(tokens, seg, dtype=torch.float32),
            "beta": h.new_empty(tokens, self.Hl, dtype=torch.float32),
            "conv": h.new_empty(batch, seg, capacity + self.K - 1, dtype=torch.float32),
            "indices": torch.empty(batch, capacity, device=h.device, dtype=torch.long),
            "positions": torch.arange(capacity, device=h.device),
        }

    @torch.no_grad()
    def _emit_batched(self, h, fb, cache, slots):
        seg = self.Hl * self.Dh
        for length, rows, starts in self._batch_plan(h, fb):
            batch = rows.shape[0]
            block = max(1, self.max_emitter_tokens // batch)
            if block >= 64:
                block = block // 64 * 64
            capacity = min(block, length)
            work = self._emitter_workspace(h, batch, capacity)
            # Separate owned tails avoid retaining a whole prior block's storage.
            tails = h.new_zeros(batch, self.K - 1, 3 * seg, dtype=torch.float32)
            state = None  # Each fresh prefix replaces, rather than advances, the pool.
            for start in range(0, length, block):
                size = min(block, length - start)
                # A tiny final GEMM can switch to a different reduction (GEMV)
                # and flip BF16 tail rounding. Pad projection rows only; conv
                # and the recurrence still consume exactly the real tokens.
                projected_size = size
                if length > block:
                    projected_size = min(capacity, (size + 63) // 64 * 64)
                tokens = work["indices"][:, :size]
                torch.add(starts[:, None], work["positions"][None, :size], out=tokens)
                tokens.add_(start)
                x = _rms(h[tokens].float(), self.norm_w, self.eps)
                if projected_size != size:
                    x = F.pad(x, (0, 0, 0, projected_size - size))
                x = x.reshape(batch * projected_size, -1)
                qkv = work["qkv"][: batch * projected_size]
                forget_a = work["forget_a"][: batch * projected_size]
                forget = work["forget"][: batch * projected_size]
                beta = work["beta"][: batch * projected_size]
                torch.mm(x, self.qkv_w.T, out=qkv)
                torch.mm(x, self.f_a_w.T, out=forget_a)
                torch.mm(forget_a, self.f_b_w.T, out=forget)
                torch.mm(x, self.b_w.T, out=beta)
                beta.sigmoid_()
                del x
                raw = qkv.view(batch, projected_size, 3 * seg)[:, :size].split(seg, -1)
                convolved = []
                full = work["conv"][:, :, : size + self.K - 1]
                for i, values in enumerate(raw):
                    tail = tails[:, :, i * seg : (i + 1) * seg]
                    full[:, :, : self.K - 1].copy_(tail.transpose(1, 2))
                    full[:, :, self.K - 1 :].copy_(values.transpose(1, 2))
                    if self.K > 1:
                        tail.copy_(full[:, :, -(self.K - 1) :].transpose(1, 2))
                    if i:  # q supplies only the pool's raw convolution tail.
                        out = F.conv1d(
                            full, self.conv_w[i * seg : (i + 1) * seg, None], groups=seg
                        )
                        convolved.append(
                            F.silu(out)
                            .transpose(1, 2)
                            .reshape(batch, size, self.Hl, self.Dh)
                        )
                        del out
                k, v = convolved
                del convolved
                k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
                f = forget.view(batch, projected_size, self.Hl, self.Dh)[:, :size]
                g = -self.A_log.exp() * F.softplus(
                    f + self.dt_bias.view(1, 1, self.Hl, self.Dh)
                )
                state = kda_state(
                    k,
                    v,
                    g,
                    beta.view(batch, projected_size, self.Hl)[:, :size],
                    initial_state=state,
                )
                # Release block outputs before reusing the projection/conv workspace.
                del (
                    k,
                    v,
                    g,
                    qkv,
                    forget_a,
                    forget,
                    beta,
                    raw,
                    full,
                    f,
                    tokens,
                    values,
                    tail,
                )
            selected = slots[rows]
            cache.conv[0][selected] = tails.to(cache.conv[0].dtype)
            cache.temporal[selected] = state.transpose(-1, -2).to(cache.temporal.dtype)
            del work, tails, state

    def _batch_plan(self, h, fb):
        key = (tuple(int(n) for n in fb.extend_seq_lens_cpu), h.device)
        if getattr(self, "_batch_index_key", None) == key:
            return self._batch_indices
        groups, start = {}, 0
        for row, length in enumerate(key[0]):
            if length:
                groups.setdefault(length, []).append((row, start))
            start += length
        plan = []
        max_batch = max(1, min(64, self.max_emitter_tokens))
        for length, members in groups.items():
            for offset in range(0, len(members), max_batch):
                group = members[offset : offset + max_batch]
                rows = torch.tensor([row for row, _ in group], device=h.device)
                starts = torch.tensor([start for _, start in group], device=h.device)
                plan.append((length, rows, starts))
        # Capture retains only small row/start metadata, not O(B*T) token indices.
        # Other shapes remain eager without invalidating the captured addresses.
        if getattr(self, "_batch_index_key", None) is None:
            self._batch_index_key, self._batch_indices = key, plan
        return plan

    def _emit_reference(self, h, fb, cache, slots):
        start, seg = 0, self.Hl * self.Dh
        for batch_index, length in enumerate(fb.extend_seq_lens_cpu):
            length = int(length)
            x = _rms(h[start : start + length].float(), self.norm_w, self.eps)
            raw = [F.linear(x, self.qkv_w[i * seg : (i + 1) * seg]) for i in range(3)]
            convolved, tails = [], []
            for i, values in enumerate(raw):
                full = F.pad(values.T[None], (self.K - 1, 0))
                out = F.conv1d(
                    full, self.conv_w[i * seg : (i + 1) * seg, None], groups=seg
                )
                convolved.append(
                    F.silu(out).transpose(1, 2).reshape(1, length, self.Hl, self.Dh)
                )
                tails.append(full[0, :, -(self.K - 1) :].T.contiguous())
            k, v = convolved[1:]
            k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            forget = F.linear(F.linear(x, self.f_a_w), self.f_b_w).view(
                1, length, self.Hl, self.Dh
            )
            g = -self.A_log.exp() * F.softplus(
                forget + self.dt_bias.view(1, 1, self.Hl, self.Dh)
            )
            beta = F.linear(x, self.b_w).sigmoid()[None]
            state = kda_state(k, v, g, beta)
            slot = slots[batch_index : batch_index + 1]
            cache.conv[0][slot] = torch.cat(tails, -1)[None].to(cache.conv[0].dtype)
            cache.temporal[slot] = state.transpose(-1, -2).to(cache.temporal.dtype)
            start += length


class DuetMLAEmitter(_MLAEmitter):
    def emit(self, h, fb, target_layer):
        c, kr = self.latent(h.float())
        attn = target_layer.self_attn
        handle = getattr(attn, "attn_mqa", None) or attn.attn_mha
        get_token_to_kv_pool().set_mla_kv_buffer(
            handle,
            fb.out_cache_loc,
            c.to(h.dtype).unsqueeze(1).contiguous(),
            kr.to(h.dtype).unsqueeze(1).contiguous(),
        )


class DuetStatePruner(KDAStatePolicy):
    """Existing pool lifecycle/hooks, reference current-state sink + warm basis.

    Functional acceptance fixes B=1, so random subspace initialisation has the
    same shape and seed as the B=1 reference. A basis belongs to a layer AND slot.
    """

    def __init__(
        self, pool, directions, spec, layer_ids, *, graph_safe=False, fastpath=None
    ):
        super().__init__(
            pool,
            {l: directions[l] for l in layer_ids},
            spec["state_rank"],
            max(1, spec["state_every"]),
            prefix_only=spec["state_every"] == 0,
        )
        self.explicit_sink = spec.get("state_sink", "explicit") == "explicit"
        self.warm = {}
        self.graph_safe = graph_safe
        self.fastpath = graph_safe if fastpath is None else fastpath
        self._omega = None

    def flush(self, slots, *, prefix, layer_id=None):
        if not self.fastpath:
            return super().flush(slots, prefix=prefix, layer_id=layer_id)
        if self.in_prefill_boundary or (not prefix and self.prefix_only):
            return
        # In eager mode all prefix states must be cut BEFORE the first KDA
        # layer reads them; cadence cuts happen AFTER the last layer reads out.
        # Graph runner calls with layer_id=None already bracket the whole graph.
        if layer_id is not None and layer_id != (
            min(self.layer_map) if prefix else max(self.layer_map)
        ):
            return
        self._flush_metadata(self._read_metadata(slots), prefix=prefix)

    def _read_metadata(self, slots):
        slots = slots.long().reshape(-1)
        safe = slots.clamp_min(0)
        return (
            torch.cat(
                (
                    slots[None],
                    self.count[:, safe].long(),
                    self.pending_prefix[:, safe].long(),
                ),
                dim=0,
            )
            .cpu()
            .tolist()
        )

    def _flush_metadata(self, metadata, *, prefix):
        layers = self.states.shape[0]
        # A single compact D2H read supplies both scheduling and warm decisions.
        plan, fresh = [], []
        for index in range(layers):
            if prefix:
                if not any(metadata[1 + layers + index]):
                    continue
            elif max(metadata[1 + index], default=0) < self.every:
                continue
            chosen, seen = [], set()
            for row, slot in enumerate(metadata[0]):
                pending = bool(metadata[1 + layers + index][row])
                due = pending if prefix else metadata[1 + index][row] >= self.every
                if slot >= 0 and slot not in seen and due:
                    chosen.append(slot)
                    fresh.append(pending)
                    seen.add(slot)
            if chosen:
                plan.append((index, torch.tensor(chosen, dtype=torch.long)))
        if not plan:
            return
        indices = self._cut_many(plan, fresh=fresh)
        # Keep a CPU snapshot coherent when before_graph reuses it after replay.
        for index, selected_cpu in plan:
            chosen = set(selected_cpu.tolist())
            for row, slot in enumerate(metadata[0]):
                if slot in chosen:
                    metadata[1 + index][row] = 0
                    if prefix:
                        metadata[1 + self.states.shape[0] + index][row] = 0
        layers, selected = indices.unbind(-1)
        flat = layers * self.states.shape[1] + selected
        self.count.view(-1).index_fill_(0, flat, 0)
        if prefix:
            self.pending_prefix.view(-1).index_fill_(0, flat, False)
            self.prefix_cuts += indices.shape[0]
        else:
            self.decode_cuts += indices.shape[0]

    def before_graph(self, batch):
        self._graph_metadata = None
        if not self.fastpath:
            return super().before_graph(batch)
        if self.in_prefill_boundary:
            return
        metadata = self._read_metadata(self.slots(batch))
        self._flush_metadata(metadata, prefix=True)
        # Production disallows speculative verification; a decode replay adds
        # exactly one count per real slot in each layer. Reuse the pre-replay
        # snapshot instead of synchronizing the GPU a second time afterwards.
        if batch.forward_mode.is_decode():
            self._graph_metadata = (batch, metadata)

    def after_graph(self, batch):
        saved = getattr(self, "_graph_metadata", None)
        self._graph_metadata = None
        if not self.fastpath or saved is None or saved[0] is not batch:
            return super().after_graph(batch)
        if self.in_prefill_boundary or self.prefix_only:
            return
        metadata = saved[1]
        multiplicity = {}
        for slot in metadata[0]:
            if slot >= 0:
                multiplicity[slot] = multiplicity.get(slot, 0) + 1
        increments = [multiplicity.get(slot, 0) for slot in metadata[0]]
        for index in range(self.states.shape[0]):
            metadata[1 + index] = [
                count + increment
                for count, increment in zip(metadata[1 + index], increments)
            ]
        self._flush_metadata(metadata, prefix=False)

    def _cut_many_fast(self, plan, fresh=None):
        keys = [(layer, slot) for layer, slots in plan for slot in slots.tolist()]
        if not keys:
            return
        if fresh is not None:
            fresh_by_key = dict(zip(keys, fresh))
        # Slot-major order permits one owned warm allocation per request,
        # preventing sparse cuts from retaining an entire historical batch.
        keys.sort(key=lambda key: (key[1], key[0]))
        if fresh is not None:
            fresh = [fresh_by_key[key] for key in keys]
        device = self.states.device
        indices = torch.tensor(keys, dtype=torch.long, device=device)
        layers, slots = indices.unbind(-1)
        state = self.states[layers, slots].transpose(-1, -2)
        if self.rank >= min(state.shape[-2:]):
            for key in keys:
                self.warm.pop(key, None)
            return indices
        if fresh is None:
            fresh = self.pending_prefix[layers, slots].cpu().tolist()
        previous = [
            None if new else self.warm.get(key) for key, new in zip(keys, fresh)
        ]
        use_previous = torch.tensor(
            [value is not None for value in previous], device=device
        )
        zero = state.new_zeros(1, state.shape[1], state.shape[-1], self.rank)
        basis = torch.cat([zero if value is None else value for value in previous])
        if self._omega is None:
            self._omega = fast_state.probes(
                state.shape[1], state.shape[-1], state.shape[-2], self.rank, device
            )
        projected, basis = fast_state.project(
            state,
            self.vbar[layers],
            self.rank,
            basis,
            use_previous,
            self._omega,
            explicit=self.explicit_sink,
        )
        self.states[layers, slots] = projected.transpose(-1, -2)
        start = 0
        while start < len(keys):
            stop = start + 1
            while stop < len(keys) and keys[stop][1] == keys[start][1]:
                stop += 1
            # Clone once per slot, rather than once per layer or retaining the
            # whole cut batch when only a single slot survives a reset.
            owned = basis[start:stop].clone()
            self.warm.update(zip(keys[start:stop], owned.split(1)))
            start = stop
        return indices

    def _cut_many(self, plan, fresh=None):
        if self.fastpath:
            return self._cut_many_fast(plan, fresh)
        for layer_index, slots in plan:
            for slot in slots.tolist():
                state = self.states[layer_index, slot : slot + 1].transpose(-1, -2)
                fresh = bool(self.pending_prefix[layer_index, slot])
                key = (layer_index, slot)
                projected, basis = project_state(
                    state,
                    self.vbar[layer_index],
                    self.rank,
                    None if fresh else self.warm.get(key),
                    explicit=self.explicit_sink,
                )
                if basis is None:
                    self.warm.pop(key, None)
                else:
                    self.warm[key] = basis
                self.states[layer_index, slot : slot + 1] = projected.transpose(-1, -2)

    def reset_slots(self, indices):
        super().reset_slots(indices)
        if hasattr(self, "warm"):
            selected = set(torch.as_tensor(indices).reshape(-1).tolist())
            self.warm = {k: v for k, v in self.warm.items() if k[1] not in selected}

    def copy_slots(self, src, dst):
        super().copy_slots(src, dst)
        sources = torch.as_tensor(src).reshape(-1).tolist()
        targets = torch.as_tensor(dst).reshape(-1).tolist()
        saved = {
            (layer, target): self.warm.get((layer, source))
            for source, target in zip(sources, targets)
            for layer in self.layer_map.values()
        }
        for key, value in saved.items():
            if value is None:
                self.warm.pop(key, None)
            else:
                self.warm[key] = value.clone()

    def get_cpu_slots(self, indices):
        selected = torch.as_tensor(indices).reshape(-1).tolist()
        bases = {
            (layer, i): self.warm[(layer, slot)].cpu()
            for i, slot in enumerate(selected)
            for layer in self.layer_map.values()
            if (layer, slot) in self.warm
        }
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
                0, slots.long().clamp_min(0), (slots >= 0).to(self.count.dtype)
            )
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
        self.mark_prefix(layer.layer_id, slots)


def make_pruner(model, runner):
    from sglang.srt.runtime_context import (
        get_disagg,
        get_memory,
        get_parallel,
        get_spec,
    )

    if get_parallel().tp_size != 1 or get_spec().speculative_algorithm is not None:
        raise ValueError(
            "HF DUET functional mode requires TP1 and no speculative decoding"
        )
    if (
        not get_memory().disable_radix_cache
        or get_disagg().disaggregation_mode != "null"
    ):
        raise ValueError("HF DUET functional mode requires radix off and AGG")
    if model.duet_options.decode_ssm_r == 0:
        return None
    return DuetStatePruner(
        runner.req_to_token_pool,
        model.duet_sink_dir,
        model.duet_options.effective_spec(model.duet_report["spec"]),
        model.config.linear_layer_ids,
        graph_safe=model.duet_profile == "production",
    )
