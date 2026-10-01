"""Functional Lightning DUET inside the native SGLang NemotronH engine.

TP=PP=1, eager execution, no prefix reuse/chunking/speculation. The scheduler,
base weights, shallow native layers, attention, MoE and HTTP interface remain
SGLang's. Only the DUET memory handoff and Mamba decode state are replaced.
"""

import copy
import logging
from types import MethodType

import torch
import torch.nn.functional as F

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_req_to_token_pool,
    get_token_to_kv_pool,
)
from sglang.srt.models.nemotron_h import NemotronHForCausalLM as StockNemotronH
from sglang.srt.runtime_context import get_server_args

from ._common import load as _load
from .boundary import mark_runner_dummy_batches, prefill_count
from .components import Components, base_rms_norm

_options = _load("options")
DuetOptions = _options.DuetOptions
LeftSinkStatePool = _load("state_pool").LeftSinkStatePool
reconstruct_boundary = _load("invariants").reconstruct_boundary

log = logging.getLogger(__name__)


class Runtime:
    def __init__(self, owner, directory):
        self.body = owner.model
        self.original_forward = self.body.forward
        # docs/167 §4.1 (P4): the two production items this adapter implements are the code matmul precision
        # and the component upload mode; graphs / batched decode stay unsupported (eager path).
        _numerics = _load("numerics")
        args = get_server_args()
        self.numerics_profile = _numerics.profile_name(args)
        self.code_precision = _options.resolve_code_precision(args)
        self.components = Components(
            directory, owner.config, self.body.embed_tokens.weight.device,
            async_h2d=_numerics.controls(self.numerics_profile)["async_h2d"],
            tf32=self.code_precision == "tf32",
        )
        self.mamba_ids = tuple(
            i
            for i, layer in enumerate(self.body.layers)
            if hasattr(layer, "_forward_mamba")
        )
        self.geometry = self.components.geometry
        self.options = DuetOptions.resolve(self.components.spec, get_server_args())
        if self.mamba_ids != self.geometry.mamba_ids:
            raise ValueError("base model memory layers differ from config")
        if self.options.decode_ssm_r >= min(
            self.geometry.head_dim, self.geometry.state_dim
        ):
            raise ValueError(
                "use decode-ssm-r=0 for exact state; compressed rank must be smaller than the state dimensions"
            )
        self.pool = None
        self.last_handoff = None
        for norm in [self.body.norm_f, *[layer.norm for layer in self.body.layers]]:
            norm.forward = MethodType(base_rms_norm, norm)
        # These are per-instance hooks; the off path never creates this Runtime.
        for layer_id in self.mamba_ids:
            layer = self.body.layers[layer_id]
            original = layer._forward_mamba

            def forward(
                layer_self, hidden_states, forward_batch, lid=layer_id, native=original
            ):
                if forward_batch.forward_mode.is_decode() and not getattr(
                    forward_batch, "_lightning_dummy_batch", False
                ):
                    return self.mamba_decode(lid, hidden_states, forward_batch)
                return native(hidden_states, forward_batch)

            layer._forward_mamba = MethodType(forward, layer)
        self.body.forward = self.forward
        log.info(
            "Lightning DUET loaded: spec=%s geometry=%s options=%s sha256=%s numerics=%s code_precision=%s async_h2d=%s "
            "graphs/batched-decode=unsupported-by-adapter",
            self.components.spec,
            vars(self.geometry),
            self.options,
            self.components.manifest["sha256"],
            self.numerics_profile,
            self.code_precision,
            self.components.async_h2d,
        )

    def ensure_pool(self):
        req_pool = get_req_to_token_pool()
        if self.pool is None:
            native = req_pool.mamba_pool
            # Native allocation remains temporary shallow-prefill scratch. This
            # functional path makes no claim of reducing reserved GPU memory.
            self.pool = LeftSinkStatePool(
                native.size + 1,
                self.mamba_ids,
                self.components.directions,
                n=self.geometry.state_dim,
                rank=self.options.decode_ssm_r,
                window=self.options.decode_ssm_w,
                conv_dim=self.geometry.conv_dim,
                conv_width=self.geometry.conv_width,
                transfer_prefix="lightning_",
            )
            native.register_slot_state(self.pool)
        return req_pool

    def slots(self, fb):
        req_pool = self.ensure_pool()
        virtual = req_pool.get_mamba_indices(fb.req_pool_indices)
        return req_pool.translate_mamba_indices(virtual).tolist()

    def slice_batch(self, fb, request, begin, end, sequence_length, decode=False):
        nb = copy.copy(fb)
        nb.forward_mode = ForwardMode.DECODE if decode else ForwardMode.EXTEND
        nb.global_forward_mode = nb.forward_mode
        nb.batch_size = 1
        nb.input_ids, nb.positions = fb.input_ids[begin:end], fb.positions[begin:end]
        nb.req_pool_indices = fb.req_pool_indices[request : request + 1]
        if getattr(fb, "req_pool_indices_cpu", None) is not None:
            nb.req_pool_indices_cpu = fb.req_pool_indices_cpu[request : request + 1]
        nb.seq_lens = fb.seq_lens.new_tensor([sequence_length])
        nb.seq_lens_cpu = torch.tensor([sequence_length], dtype=torch.int64)
        nb.seq_lens_sum = sequence_length
        nb.orig_seq_lens = nb.seq_lens.clone() if fb.orig_seq_lens is not None else None
        nb.out_cache_loc = fb.out_cache_loc[begin:end]
        if fb.out_cache_loc_virtual is not None:
            nb.out_cache_loc_virtual = fb.out_cache_loc_virtual[begin:end]
        nb.extend_num_tokens = end - begin
        nb.extend_seq_lens_cpu = [end - begin]
        nb.extend_prefix_lens_cpu = [sequence_length - (end - begin)]
        nb.extend_seq_lens = nb.seq_lens.new_tensor(nb.extend_seq_lens_cpu)
        nb.extend_prefix_lens = nb.seq_lens.new_tensor(nb.extend_prefix_lens_cpu)
        nb.extend_start_loc = nb.seq_lens.new_tensor([0])
        nb.extend_logprob_start_lens_cpu = [0]
        nb.return_logprob = False
        nb.top_logprobs_nums, nb.token_ids_logprobs = None, None
        for name in ("mamba_track_indices", "mamba_track_mask", "mamba_track_seqlens"):
            value = getattr(fb, name, None)
            if value is not None:
                setattr(nb, name, value[request : request + 1])
        nb.global_num_token_non_padded_cpu = end - begin
        for name in ("global_num_token_non_padded", "num_token_non_padded"):
            value = getattr(fb, name, None)
            if value is not None:
                setattr(nb, name, value.new_tensor(end - begin))
        nb.forward_metadata_ready = nb.forward_metadata_replan_equivalent = False
        nb.forward_metadata_planned_bs = nb.forward_metadata_planned_num_tokens = None
        nb._original_batch_size = nb._original_forward_mode = (
            nb._original_num_tokens
        ) = None
        nb.spec_info = nb.mm_inputs = nb.input_embeds = nb.mrope_positions = None
        return nb

    def shallow_handoff(self, fb, slot):
        req_pool = self.ensure_pool()
        self.pool.reset_slots(torch.tensor([slot], device=fb.input_ids.device))
        hidden = self.body.embed_tokens(fb.input_ids)
        residual = None
        get_attn_backend().init_forward_metadata(fb)
        for layer in self.body.layers[: self.geometry.k]:
            hidden, residual = layer.forward(
                hidden_states=hidden, residual=residual, forward_batch=fb
            )
        hidden = hidden if residual is None else hidden + residual
        # The common hook performs a NEW embedding lookup after shallow layers
        # mutate their residual buffer and reconstructs from the packed record.
        record, reconstructed = reconstruct_boundary(
            self.components.code,
            hidden,
            fb.input_ids,
            embedding_lookup=self.body.embed_tokens,
        )
        for layer_id in self.mamba_ids:
            if layer_id < self.geometry.k:
                cache = req_pool.mamba2_layer_cache(layer_id)
                self.pool.initialize(
                    layer_id, slot, cache.temporal[slot], cache.conv[0][slot]
                )
                cache.temporal[slot].zero_()
            else:
                emitted = self.components.emitter(layer_id, reconstructed)
                self.pool.initialize(layer_id, slot, emitted["state"], emitted["conv"])
        kv = get_token_to_kv_pool()
        for layer_id in self.components.attention_emitters:
            emitted = self.components.emitter(layer_id, reconstructed)
            attention = self.body.layers[layer_id].mixer.attn
            kv.set_kv_buffer(
                attention,
                fb.out_cache_loc,
                emitted["k"].to(hidden.dtype),
                emitted["v"].to(hidden.dtype),
            )
        self.last_handoff = {
            "tokens": len(fb.input_ids),
            "latent_bytes": record.nbytes,
            "mamba_emitters": len(self.components.mamba_emitters),
            "attention_emitters": len(self.components.attention_emitters),
        }
        log.info("Lightning DUET P/D handoff: %s", self.last_handoff)
        return hidden

    def full_handoff(self, fb, slot):
        """Accuracy-first full-depth prefill, then reference prompt-final pruning."""
        req_pool = self.ensure_pool()
        self.pool.reset_slots(torch.tensor([slot], device=fb.input_ids.device))
        get_attn_backend().init_forward_metadata(fb)
        hidden = self.original_forward(fb.input_ids, fb.positions, fb)
        for layer_id in self.mamba_ids:
            cache = req_pool.mamba2_layer_cache(layer_id)
            self.pool.initialize(
                layer_id, slot, cache.temporal[slot], cache.conv[0][slot]
            )
            cache.temporal[slot].zero_()
        self.last_handoff = {
            "tokens": len(fb.input_ids),
            "latent_bytes": 0,
            "mamba_emitters": 0,
            "attention_emitters": 0,
            "full_depth": True,
        }
        return hidden

    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        pp_proxy_tensors=None,
        inputs_embeds=None,
    ):
        fb = forward_batch
        if getattr(fb, "_lightning_dummy_batch", False):
            return self.original_forward(input_ids, positions, fb)
        if inputs_embeds is not None or pp_proxy_tensors is not None:
            raise ValueError("Lightning functional path requires token IDs and PP=1")
        if fb.forward_mode.is_idle():
            return self.original_forward(input_ids, positions, fb)
        if fb.forward_mode.is_decode():
            return self.original_forward(input_ids, positions, fb)
        if not fb.forward_mode.is_extend() or fb.spec_info is not None:
            raise ValueError("Lightning DUET supports ordinary extend/decode only")
        if any(fb.extend_prefix_lens_cpu):
            raise ValueError("Lightning DUET requires an uncached, unchunked prompt")
        out = self.body.embed_tokens.weight.new_zeros(
            (len(input_ids), self.body.config.hidden_size)
        )
        slots = self.slots(fb)
        offset = 0
        for request, length in enumerate(fb.extend_seq_lens_cpu):
            start = (
                fb.extend_logprob_start_lens_cpu[request] if fb.return_logprob else None
            )
            # A start at length means output-logprob-only: ordinary P/D split.
            start = None if start == length else start
            if self.options.prefill_layer_trim and self.geometry.k < len(
                self.body.layers
            ):
                count = prefill_count(length, start)
                shallow_count = max(1, count)
                shallow = self.slice_batch(
                    fb, request, offset, offset + shallow_count, shallow_count
                )
                cut_hidden = self.shallow_handoff(shallow, slots[request])
                if count == 0:
                    # Reference T=1 uses shallow logits and no boundary step.
                    out[offset : offset + 1] = self.body.norm_f(cut_hidden)
            else:
                shallow_count = length if start is None else start + 1
                full = self.slice_batch(
                    fb, request, offset, offset + shallow_count, shallow_count
                )
                out[offset : offset + shallow_count] = self.full_handoff(
                    full, slots[request]
                )
            for pos in range(shallow_count, length):
                step = self.slice_batch(
                    fb, request, offset + pos, offset + pos + 1, pos + 1, decode=True
                )
                get_attn_backend().init_forward_metadata(step)
                out[offset + pos : offset + pos + 1] = self.original_forward(
                    step.input_ids, step.positions, step
                )
            offset += length
        return out

    def mamba_decode(self, layer_id, hidden, fb):
        g = self.geometry
        mixer = self.body.layers[layer_id].mixer
        projected, _ = mixer.in_proj(hidden)
        gate, xbc, raw_dt = projected.split([g.intermediate, g.conv_dim, g.heads], -1)
        ys = []
        for row, slot in enumerate(self.slots(fb)):
            index = self.pool.layer_map[layer_id]
            history = self.pool.conv[index, slot]
            full = torch.cat((history, xbc[row, :, None]), dim=-1)
            convolved = (full.float() * mixer.conv1d.weight[:, 0].float()).sum(-1)
            if mixer.conv1d.bias is not None:
                convolved = convolved + mixer.conv1d.bias.float()
            x, b, c = F.silu(convolved).split([g.intermediate, g.bc_dim, g.bc_dim], -1)
            x = x.reshape(g.heads, g.head_dim)
            b, c = [
                v.reshape(g.groups, g.state_dim).repeat_interleave(
                    g.heads // g.groups, 0
                )
                for v in (b, c)
            ]
            dt = F.softplus(raw_dt[row].float() + mixer.dt_bias.float())
            state = self.pool.step(
                layer_id,
                slot,
                (dt * mixer.A).exp(),
                dt[:, None] * x,
                b,
                full[:, 1:].contiguous(),
            )
            y = torch.einsum("hpn,hn->hp", state, c) + x * mixer.D.float()[:, None]
            gated = y.reshape(g.intermediate) * F.silu(gate[row].float())
            grouped = gated.reshape(g.groups, g.intermediate // g.groups)
            grouped = grouped * torch.rsqrt(
                grouped.square().mean(-1, keepdim=True) + self.components.eps
            )
            ys.append(
                (grouped.reshape(g.intermediate) * mixer.norm.weight).to(hidden.dtype)
            )
        result, _ = mixer.out_proj(torch.stack(ys))
        return result


class NemotronHForCausalLM(StockNemotronH):
    def prepare_before_cuda_graph_capture(self, model_runner):
        # EagerRunner also invokes this before its autotune dummy decode.
        # Dummy forwards tune native MoE/attention; real requests must still
        # pass initialization checks and can never enter this branch by slot ID.
        mark_runner_dummy_batches(model_runner)

    def __init__(self, **kwargs):
        args = get_server_args()
        # docs/167 §4: the production profile is not validated on Lightning yet (P4); refuse unless the
        # validation run opted in with --duet-allow-unvalidated-profile.  Nothing is degraded either way.
        _load("numerics").require_profile("lightning", args, production_supported=False)
        if (
            args.tp_size != 1
            or args.pp_size != 1
            or kwargs.get("quant_config") is not None
        ):
            raise ValueError(
                "Lightning DUET functional release requires TP=PP=1 and BF16 weights"
            )
        if not args.disable_radix_cache or args.chunked_prefill_size > 0:
            raise ValueError(
                "Lightning DUET requires --disable-radix-cache --chunked-prefill-size -1"
            )
        if not args.disable_cuda_graph or not args.disable_overlap_schedule:
            raise ValueError(
                "Lightning DUET requires --disable-cuda-graph --disable-overlap-schedule"
            )
        if args.speculative_algorithm is not None:
            raise ValueError("Lightning DUET does not support speculative decoding")
        super().__init__(**kwargs)

    def load_weights(self, weights, is_mtp=False):
        if is_mtp:
            raise ValueError("Lightning DUET cannot load MTP weights")
        super().load_weights(weights, is_mtp=False)
        directory = _options.resolve_release(
            get_server_args(), legacy_directory="TWINSTAR_LIGHTNING_DUET_DIR"
        )
        if not directory:
            raise ValueError(
                "--duet-release / SGLANG_DUET_DIR must name the verified HF release directory"
            )
        self.lightning_runtime = Runtime(self, directory)


# Exactly the stock class when all DUET flags are off. No patched modules,
# sibling allocations or loader overrides survive this selection.
EntryClass = (
    NemotronHForCausalLM
    if _options.duet_enabled(legacy_enabled="TWINSTAR_LIGHTNING_DUET")
    else StockNemotronH
)
