"""Functional Lightning DUET inside the native SGLang NemotronH engine.

TP=PP=1, eager prefill, optional factor decode graph, no prefix reuse/chunking/speculation. The scheduler,
base weights, shallow native layers, attention, MoE and HTTP interface remain
SGLang's. Only the DUET memory handoff and Mamba decode state are replaced.
"""

import copy
import logging
import os
from types import MethodType, SimpleNamespace

import torch
import torch.nn.functional as F
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    causal_conv1d_update as bridge_conv_update,
)
from sglang.kernels.ops.mamba.triton_ops.mamba_ssm import (
    selective_state_update as bridge_state_update,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
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
DenseBridgeStatePool = _load("dense_bridge").DenseBridgeStatePool
_reference_boundary = _load("invariants").reconstruct_boundary


def _transient_boundary(code, hidden, token_ids, *, embedding_lookup):
    """docs/209 ablation: the AGG transient-code path (80da08d3cc0), no stored record.

    Same E/D, NVFP4 and spike arithmetic as encode+decode, so the decoded
    boundary is identical; nbytes counts one gap8 byte per spike (no escapes).
    """
    rank, spikes = code.encoder.shape[0], code.spikes
    nbytes = hidden.shape[0] * (rank // 2 + rank // 16 + 3 * spikes + 12) + 4
    return SimpleNamespace(nbytes=nbytes), code.reconstruct(hidden, embedding_lookup(token_ids))


reconstruct_boundary = (
    _transient_boundary
    if os.environ.get("SGLANG_LTN_BOUNDARY_CODEC", "record") == "transient"
    else _reference_boundary
)

log = logging.getLogger(__name__)


class Runtime:
    def __init__(self, owner, directory):
        self.body = owner.model
        self.original_forward = self.body.forward
        # docs/167 §4.1 (P4): the two production items this adapter implements are the code matmul precision
        # and the component upload mode; dense bridge batches decode (graph remains OFF).
        _numerics = _load("numerics")
        args = get_server_args()
        if args.mamba_backend != "triton":
            raise ValueError(
                "Lightning dense bridge requires the qualified triton Mamba backend"
            )
        self.numerics_profile = _numerics.profile_name(args)
        self.code_precision = _options.resolve_code_precision(args)
        self.components = Components(
            directory,
            owner.config,
            self.body.embed_tokens.weight.device,
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
        self.factored_decode = os.environ.get("SGLANG_LTN_FACTORED_DECODE", "0") == "1"
        self.batched_extend = os.environ.get("SGLANG_LTN_BATCHED_EXTEND", "0") == "1"
        self.native_batched_shallow = os.environ.get("SGLANG_LTN_NATIVE_BATCHED_SHALLOW", "0") == "1"
        if self.batched_extend and not self.factored_decode:
            raise ValueError("batched Lightning extend requires factor authority")
        # docs/209 §9: replay the captured decode graph for the last prompt
        # token of every row instead of an eager 52-layer step. Default off.
        self.tail_graph = os.environ.get("SGLANG_LTN_TAIL_GRAPH", "0") == "1"
        if self.tail_graph and not self.batched_extend:
            raise ValueError("Lightning tail graph requires batched extend")
        self.model_runner = None
        self.tail_logits = None
        self.extend_plan = None
        self.extend_rows = None
        self.pool = None
        self.factor_trunc_stream = None
        self.host_slots_by_request = {}
        self.decode_slots = None
        self.last_handoff = None
        if not self.factored_decode:
            for norm in [self.body.norm_f, *[layer.norm for layer in self.body.layers]]:
                norm.forward = MethodType(base_rms_norm, norm)
        # These are per-instance hooks; the off path never creates this Runtime.
        for layer_id in self.mamba_ids:
            layer = self.body.layers[layer_id]
            original = layer._forward_mamba

            def forward(
                layer_self, hidden_states, forward_batch, lid=layer_id, native=original
            ):
                if forward_batch.forward_mode.is_decode() and (
                    self.factored_decode or not getattr(
                        forward_batch, "_lightning_dummy_batch", False
                    )
                ):
                    if not self.factored_decode:
                        return self.mamba_decode(lid, hidden_states, forward_batch)
                    # Reuse the actual native mixer: only its SSU call changes.
                    mixer = layer_self.mixer
                    mixer._ltn_factor_update = lambda x, b, c, dt, out, slots: self.mamba_factor_update(
                        lid, x, b, c, dt, out, slots
                    )
                    try:
                        return native(hidden_states, forward_batch)
                    finally:
                        mixer._ltn_factor_update = None
                if not self.factored_decode:
                    return native(hidden_states, forward_batch)
                mixer = layer_self.mixer
                dummy = getattr(forward_batch, "_lightning_dummy_batch", False)
                mixer._ltn_factor_dummy = dummy
                if not dummy:
                    # The adapter serializes uncached prompts. Consume one
                    # layer's scan output immediately; no per-slot dense pool.
                    def store(state):
                        if self.extend_plan is not None:
                            self.pool.commit_extend(lid, self.extend_plan, state,
                                                    rows=self.extend_rows)
                            return
                        if state.shape[0] != 1:
                            raise ValueError(
                                "factored Lightning prefill expects one request"
                            )
                        index = self.pool.layer_map[lid]
                        self.pool.initialize(
                            lid,
                            self.prefill_slot,
                            state[0],
                            self.pool.conv[index, self.prefill_slot],
                        )

                    mixer._ltn_factor_prefill_store = store
                try:
                    return native(hidden_states, forward_batch)
                finally:
                    mixer._ltn_factor_dummy = False
                    mixer._ltn_factor_prefill_store = None

            layer._forward_mamba = MethodType(forward, layer)
        self.body.forward = self.forward
        log.info(
            "Lightning DUET loaded: spec=%s geometry=%s options=%s sha256=%s numerics=%s code_precision=%s async_h2d=%s "
            "factored-decode=%s batched-decode=on service-graphs=factor-decode-only",
            self.components.spec,
            vars(self.geometry),
            self.options,
            self.components.manifest["sha256"],
            self.numerics_profile,
            self.code_precision,
            self.components.async_h2d,
            self.factored_decode,
        )

    def ensure_pool(self, req_pool=None):
        if req_pool is None:
            req_pool = get_req_to_token_pool()
        if self.pool is None:
            native = req_pool.mamba_pool
            # Dense bridge binds native SSM; factor mode requires it empty.
            # Both share native conv storage and its lifecycle owner.
            pool_cls = (
                _load("mamba2_factored_pool").Mamba2FactoredStatePool
                if self.factored_decode
                else DenseBridgeStatePool
            )
            self.pool = pool_cls(
                native.size + 1,
                self.mamba_ids,
                self.components.directions,
                n=self.geometry.state_dim,
                rank=self.options.decode_ssm_r,
                window=self.options.decode_ssm_w,
                conv_dim=self.geometry.conv_dim,
                conv_width=self.geometry.conv_width,
                transfer_prefix="lightning_",
                native_dense=native.mamba_cache.temporal,
                native_conv=native.mamba_cache.conv[0],
            )
            native.register_slot_state(self.pool.native_sibling())
            if self.factored_decode:
                self.factor_trunc_stream = torch.cuda.Stream(device=self.pool.device)
                log.info(
                    "LTNF_POOL native_dense_bytes=%s resident_bytes=%s slots=%s per_slot=%s",
                    native.mamba_cache.temporal.numel()
                    * native.mamba_cache.temporal.element_size(),
                    self.pool.resident_bytes(),
                    self.pool.size,
                    self.pool.resident_bytes() // self.pool.size,
                )
        return req_pool

    def slots(self, fb):
        req_pool = self.ensure_pool()
        virtual = req_pool.get_mamba_indices(fb.req_pool_indices)
        return req_pool.translate_mamba_indices(virtual)

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
        # Native Mamba2 prepare_mixed uses this sentinel (not forward_mode)
        # to distinguish decode from prefill. A token sliced out of an extend
        # batch must not inherit its prefill classification.
        nb.extend_num_tokens = None if decode else end - begin
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
        self.prefill_slot = slot
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
                if self.factored_decode:
                    continue  # Native scan already handed off this layer.
                cache = req_pool.mamba2_layer_cache(layer_id)
                self.pool.initialize(
                    layer_id, slot, cache.temporal[slot], cache.conv[0][slot]
                )

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

    def packed_batch(self, fb, rows, token_indices, lengths, *, decode=False):
        """A native packed ForwardBatch view; metadata is rebuilt once per phase."""
        nb = copy.copy(fb)
        device = fb.input_ids.device
        ridx = torch.tensor(rows, device=device, dtype=torch.long)
        tidx = torch.tensor(token_indices, device=device, dtype=torch.long)
        nb.forward_mode = ForwardMode.DECODE if decode else ForwardMode.EXTEND
        nb.global_forward_mode = nb.forward_mode
        nb.batch_size = len(rows)
        nb.input_ids, nb.positions = fb.input_ids[tidx], fb.positions[tidx]
        nb.req_pool_indices = fb.req_pool_indices[ridx]
        if fb.req_pool_indices_cpu is not None:
            nb.req_pool_indices_cpu = fb.req_pool_indices_cpu[rows]
        nb.seq_lens_cpu = torch.tensor(lengths, dtype=torch.int64)
        nb.seq_lens = nb.seq_lens_cpu.to(device=device, dtype=fb.seq_lens.dtype)
        nb.seq_lens_sum = sum(lengths)
        nb.orig_seq_lens = nb.seq_lens.clone() if fb.orig_seq_lens is not None else None
        nb.out_cache_loc = fb.out_cache_loc[tidx]
        if fb.out_cache_loc_virtual is not None:
            nb.out_cache_loc_virtual = fb.out_cache_loc_virtual[tidx]
        counts = [1] * len(rows) if decode else lengths
        nb.extend_num_tokens = None if decode else len(token_indices)
        nb.extend_seq_lens_cpu = counts
        nb.extend_prefix_lens_cpu = [n - c for n, c in zip(lengths, counts)]
        nb.extend_seq_lens = nb.seq_lens.new_tensor(counts)
        nb.extend_prefix_lens = nb.seq_lens.new_tensor(nb.extend_prefix_lens_cpu)
        starts, offset = [], 0
        for count in counts:
            starts.append(offset)
            offset += count
        nb.extend_start_loc = nb.seq_lens.new_tensor(starts)
        nb.extend_logprob_start_lens_cpu = [0] * len(rows)
        nb.return_logprob = False
        nb.top_logprobs_nums = nb.token_ids_logprobs = None
        for name in ("mamba_track_indices", "mamba_track_mask", "mamba_track_seqlens"):
            value = getattr(fb, name, None)
            if value is not None:
                setattr(nb, name, value[ridx])
        nb.global_num_token_non_padded_cpu = len(token_indices)
        for name in ("global_num_token_non_padded", "num_token_non_padded"):
            value = getattr(fb, name, None)
            if value is not None:
                setattr(nb, name, value.new_tensor(len(token_indices)))
        nb.forward_metadata_ready = nb.forward_metadata_replan_equivalent = False
        nb.forward_metadata_planned_bs = nb.forward_metadata_planned_num_tokens = None
        nb._original_batch_size = nb._original_forward_mode = nb._original_num_tokens = None
        nb.spec_info = nb.mm_inputs = nb.input_embeds = nb.mrope_positions = None
        return nb

    def _batched_extend_forward(self, fb):
        from sglang.srt.layers.attention.linear.mamba2_emitter_extend import emitter_extend

        lengths = fb.extend_seq_lens_cpu
        shallow_lengths = [max(1, n - 1) for n in lengths]
        packed, tails, tail_rows, singletons, offset = [], [], [], [], 0
        for row, (length, shallow) in enumerate(zip(lengths, shallow_lengths)):
            packed.extend(range(offset, offset + shallow))
            if length > 1:
                tails.append(offset + length - 1)
                tail_rows.append(row)
            else:
                singletons.append((row, offset))
            offset += length
        shallow = self.packed_batch(fb, list(range(len(lengths))), packed, shallow_lengths)
        slots = self.slots(shallow)
        plan = self.pool.begin_extend(slots)
        self.extend_plan = plan
        try:
            backend = get_attn_backend()
            hidden_rows, restored_rows, latent_bytes, offset = [], [], 0, 0
            # Preserve the qualified request-local shallow arithmetic and codec
            # shapes. Packing unrelated prefixes into native BF16 scans changes
            # rounding/chunk boundaries; NVFP4 then amplifies those differences.
            # The emitter plan and factor commit are shared across requests;
            # qualified emitter arithmetic also retains request-local shapes.
            if self.native_batched_shallow:
                # One native backend plan and one packed forward for all requests.
                # Keep boundary codec request-local: this switch isolates shallow.
                self.extend_rows = None
                backend.init_forward_metadata(shallow)
                h, residual = self.body.embed_tokens(shallow.input_ids), None
                for layer in self.body.layers[: self.geometry.k]:
                    h, residual = layer.forward(hidden_states=h, residual=residual,
                                                forward_batch=shallow)
                h = h if residual is None else h + residual
                hidden_rows.append(h)
                for length in shallow_lengths:
                    record, restored = reconstruct_boundary(
                        self.components.code, h[offset:offset+length],
                        shallow.input_ids[offset:offset+length],
                        embedding_lookup=self.body.embed_tokens)
                    restored_rows.append(restored)
                    latent_bytes += record.nbytes
                    offset += length
            else:
                for row, length in enumerate(shallow_lengths):
                    one = self.slice_batch(shallow, row, offset, offset+length, length)
                    self.extend_rows = (row,)
                    backend.init_forward_metadata(one)
                    h, residual = self.body.embed_tokens(one.input_ids), None
                    for layer in self.body.layers[: self.geometry.k]:
                        h, residual = layer.forward(
                            hidden_states=h, residual=residual, forward_batch=one
                        )
                    h = h if residual is None else h + residual
                    record, restored = reconstruct_boundary(
                        self.components.code, h, one.input_ids,
                        embedding_lookup=self.body.embed_tokens,
                    )
                    hidden_rows.append(h)
                    restored_rows.append(restored)
                    latent_bytes += record.nbytes
                    offset += length
            self.extend_rows = None
            hidden, reconstructed = torch.cat(hidden_rows), torch.cat(restored_rows)
            backend.init_forward_metadata(shallow)
            # All emitters consume one packed backend plan; native shallow
            # states above joined the same bounded collector with explicit rows.
            metadata = backend.linear_attn_backend.forward_metadata
            kv = get_token_to_kv_pool()
            for lid in (*self.components.mamba_emitters, *self.components.attention_emitters):
                emitted = emitter_extend(self.components, lid, reconstructed, metadata, self.pool, plan)
                if emitted is not None:
                    kv.set_kv_buffer(self.body.layers[lid].mixer.attn, shallow.out_cache_loc,
                                     emitted["k"].to(hidden.dtype), emitted["v"].to(hidden.dtype))
            self.pool.finish_extend(plan)
            self.extend_plan = None
            out = hidden.new_zeros((fb.input_ids.numel(), self.body.config.hidden_size))
            if singletons:
                starts, cursor = [], 0
                for length in shallow_lengths:
                    starts.append(cursor)
                    cursor += length
                selected = [starts[row] for row, _ in singletons]
                targets = [pos for _, pos in singletons]
                out[targets] = self.body.norm_f(hidden[selected])
            tail_graph = False
            if tails:
                tail = self.packed_batch(fb, tail_rows, tails, [lengths[r] for r in tail_rows], decode=True)
                graph = self._tail_graph_runner(fb, tail, singletons)
                if graph is not None:
                    # Factors are published above; the graph's decode step is
                    # the first reader, as in ordinary decode. Its logits are
                    # the extend's (one row per request, same order).
                    logits = graph.execute(tail)
                    logits.next_token_logits = logits.next_token_logits.clone()
                    self.tail_logits, tail_graph = logits, True
                else:
                    backend.init_forward_metadata(tail)
                    out[tails] = self._decode_forward(tail.input_ids, tail.positions, tail)
            self.last_handoff = dict(tokens=len(packed), latent_bytes=latent_bytes,
                mamba_emitters=len(self.components.mamba_emitters),
                attention_emitters=len(self.components.attention_emitters),
                batched_extend=True, requests=len(lengths), commit_groups=plan.commits,
                shallow_request_local=not self.native_batched_shallow,
                prefill_commit_mode=self.pool.prefill_commit_mode,
                emitter_arithmetic="request-local-reference", emitter_native_batched=False,
                peak_pending_state_bytes=plan.peak_pending_bytes,
                boundary_codec=os.environ.get("SGLANG_LTN_BOUNDARY_CODEC", "record"),
                tail_graph=tail_graph)
            log.info("Lightning DUET batched extend: %s", self.last_handoff)
            return out
        except BaseException:
            self.tail_logits = None
            if self.extend_plan is None:
                # Publication succeeded, but a subsequent tail/logits operation
                # failed. Do not leave partially advanced rows reusable.
                self.pool.reset_slots(slots)
            else:
                self.pool.abort_extend(plan)
            raise
        finally:
            self.extend_plan = None
            self.extend_rows = None

    def _tail_graph_runner(self, fb, tail, singletons):
        """The captured decode graph, when it can stand in for the eager tail."""
        runner = getattr(self.model_runner, "decode_cuda_graph_runner", None)
        if (not self.tail_graph or runner is None or singletons or fb.return_logprob
                or getattr(fb, "capture_hidden_mode", None) not in (None, CaptureHiddenMode.NULL)):
            return None
        return runner if runner.can_run_graph(tail) else None

    def full_handoff(self, fb, slot):
        """Accuracy-first full-depth prefill, then reference prompt-final pruning."""
        req_pool = self.ensure_pool()
        self.pool.reset_slots(torch.tensor([slot], device=fb.input_ids.device))
        get_attn_backend().init_forward_metadata(fb)
        self.prefill_slot = slot
        hidden = self.original_forward(fb.input_ids, fb.positions, fb)
        for layer_id in () if self.factored_decode else self.mamba_ids:
            cache = req_pool.mamba2_layer_cache(layer_id)
            self.pool.initialize(
                layer_id, slot, cache.temporal[slot], cache.conv[0][slot]
            )

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
            if self.factored_decode and fb.forward_mode.is_decode():
                return self._decode_forward(input_ids, positions, fb)
            return self.original_forward(input_ids, positions, fb)
        if inputs_embeds is not None or pp_proxy_tensors is not None:
            raise ValueError("Lightning functional path requires token IDs and PP=1")
        if fb.forward_mode.is_idle():
            return self.original_forward(input_ids, positions, fb)
        if fb.forward_mode.is_decode():
            return self._decode_forward(input_ids, positions, fb)
        if not fb.forward_mode.is_extend() or fb.spec_info is not None:
            raise ValueError("Lightning DUET supports ordinary extend/decode only")
        if any(fb.extend_prefix_lens_cpu):
            raise ValueError("Lightning DUET requires an uncached, unchunked prompt")
        if (self.batched_extend and self.options.prefill_layer_trim
                and self.geometry.k < len(self.body.layers)
                and (not fb.return_logprob or all(
                    start == length for start, length in zip(
                        fb.extend_logprob_start_lens_cpu, fb.extend_seq_lens_cpu)))):
            return self._batched_extend_forward(fb)
        out = self.body.embed_tokens.weight.new_zeros(
            (len(input_ids), self.body.config.hidden_size)
        )
        # Prompt-only host mapping; ordinary decode never reads device IDs.
        slots = self.slots(fb).tolist()
        request_ids = getattr(fb, "req_pool_indices_cpu", None)
        if request_ids is None:
            request_ids = fb.req_pool_indices.detach().cpu()
        for request, slot in zip(request_ids.tolist(), slots):
            self.host_slots_by_request[request] = slot
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
                out[offset + pos : offset + pos + 1] = self._decode_forward(
                    step.input_ids, step.positions, step
                )
            offset += length
        return out

    def _decode_forward(self, input_ids, positions, fb):
        if getattr(self, "factored_decode", False):
            self.ensure_pool()
            # Capture the native backend's persistent physical-index buffer.
            # Replay refreshes it before execution, including -1 padding.
            # Dummy/capture metadata is poisoned outside capture by the backend.
            self.decode_slots = get_attn_backend().linear_attn_backend.forward_metadata.mamba_cache_indices
            try:
                return self.original_forward(input_ids, positions, fb)
            finally:
                # Join inside graph capture and before returning the final
                # layer: next decode / slot lifecycle must see completed cuts.
                torch.cuda.current_stream().wait_stream(self.factor_trunc_stream)
                self.decode_slots = None
        slots = self.slots(fb)
        request_ids = getattr(fb, "req_pool_indices_cpu", None)
        if request_ids is None:
            raise ValueError(
                "Lightning dense bridge needs CPU request IDs for due buckets"
            )
        host_slots = [self.host_slots_by_request[r] for r in request_ids.tolist()]
        self.decode_slots = self.pool.begin_step(host_slots, slots)
        hidden = self.original_forward(input_ids, positions, fb)
        # Current-token outputs are already consumed. All due layer-slot states
        # enter ONE reference factorize call before the next token/lifecycle op.
        self.pool.finish_step()
        self.decode_slots = None
        return hidden

    def mamba_factor_update(self, layer_id, x, b, c, raw_dt, out, slots):
        from sglang.srt.layers.attention.linear.kernels.mamba2_factored import (
            step, truncate_layers,
        )

        if self.decode_slots is None:
            raise RuntimeError("factor update outside native decode")
        mixer = self.body.layers[layer_id].mixer
        i = self.pool.layer_map[layer_id]
        result = step(
            x, b, c, raw_dt, mixer.A, mixer.D, self.pool.directions[i],
            self.pool.coeff[i], self.pool.u[i], self.pool.v[i],
            self.pool.fcount[i], slots, rank=self.pool.rank,
            window=self.pool.window, out=out, dt_bias=mixer.dt_bias,
            do_truncate=False,
        )
        if layer_id == self.mamba_ids[-1]:
            # Every layer's step has published its factors on the main stream.
            # Readout is already independent of U/V; one batched cut overlaps
            # the remaining model work. Persistent buffers outlive both streams.
            self.factor_trunc_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.factor_trunc_stream):
                truncate_layers(
                    self.pool.u, self.pool.v, self.pool.fcount, slots,
                    self.pool.rank, self.pool.window,
                )
        return result

    def mamba_decode(self, layer_id, hidden, fb):
        g = self.geometry
        mixer = self.body.layers[layer_id].mixer
        projected, _ = mixer.in_proj(hidden)
        gate, xbc, raw_dt = projected.split([g.intermediate, g.conv_dim, g.heads], -1)
        index = self.pool.layer_map[layer_id]
        slots = self.decode_slots
        if slots is None:
            raise RuntimeError("Mamba decode outside a dense bridge transaction")
        # FP32 history, products and conv output. Bias and SiLU remain explicit
        # reference operations; stock defaults and the off path are unchanged.
        convolved = bridge_conv_update(
            xbc.float(),
            self.pool.conv[index],
            mixer.conv1d.weight[:, 0].float(),
            activation=None,
            conv_state_indices=slots,
            enable_fp_fusion=False,
        )
        if mixer.conv1d.bias is not None:
            convolved = convolved + mixer.conv1d.bias.float()
        x, b, c = F.silu(convolved).split([g.intermediate, g.bc_dim, g.bc_dim], -1)
        x = x.reshape(-1, g.heads, g.head_dim)
        b = b.reshape(-1, g.groups, g.state_dim)
        c = c.reshape(-1, g.groups, g.state_dim)
        dt = F.softplus(raw_dt.float() + mixer.dt_bias.float())
        decay = (dt * mixer.A).exp()
        y = torch.zeros_like(x, dtype=torch.float32)
        bridge_state_update(
            self.pool.dense[index], x, dt[..., None].expand_as(x),
            mixer.A[:, None, None].expand(g.heads, g.head_dim, g.state_dim),
            b, c, mixer.D.float()[:, None].expand(g.heads, g.head_dim),
            state_batch_indices=slots, out=y, reference_decay=decay,
        )
        self.pool.note_layer_step(layer_id)
        # Group size is intermediate/groups (512 in the accepted geometry).
        # Keep gate-before-norm, FP32 reduction, weight-before-BF16 cast.
        gated = y.reshape(-1, g.intermediate) * F.silu(gate.float())
        grouped = gated.reshape(-1, g.groups, g.intermediate // g.groups)
        grouped = grouped * torch.rsqrt(
            grouped.square().mean(-1, keepdim=True) + self.components.eps
        )
        normalized = (grouped.reshape(-1, g.intermediate) * mixer.norm.weight).to(
            hidden.dtype
        )
        result, _ = mixer.out_proj(normalized)
        return result


class NemotronHForCausalLM(StockNemotronH):
    def prepare_before_cuda_graph_capture(self, model_runner):
        # EagerRunner also invokes this before its autotune dummy decode.
        # Dummy forwards tune native MoE/attention; real requests must still
        # pass initialization checks and can never enter this branch by slot ID.
        mark_runner_dummy_batches(model_runner)
        self.lightning_runtime.model_runner = model_runner
        if self.lightning_runtime.factored_decode:
            # Allocate/register authoritative state before warmup/capture.
            self.lightning_runtime.ensure_pool(model_runner.req_to_token_pool)

    def forward(self, input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None):
        runtime = self.lightning_runtime
        if not runtime.tail_graph or not forward_batch.forward_mode.is_extend():
            return super().forward(input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors)
        runtime.tail_logits = None
        hidden = self.model.forward(input_ids, positions, forward_batch, pp_proxy_tensors, input_embeds)
        logits, runtime.tail_logits = runtime.tail_logits, None
        if logits is not None:
            return logits  # The tail graph already produced every row's logits.
        return self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)

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
        if not args.disable_overlap_schedule:
            raise ValueError("Lightning DUET requires --disable-overlap-schedule")
        if not args.disable_cuda_graph and os.environ.get("SGLANG_LTN_FACTORED_DECODE", "0") != "1":
            raise ValueError(
                "Lightning dense bridge requires --disable-cuda-graph; only factor decode supports capture"
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
