"""Native DUET P31 dispatch, retaining the qualified P/D boundary protocol.

The external helpers still own loading, boundary tensors, native GDN splitting,
and transport. Only the native release's dense policy and transient LinearCode
path differ from the older stored-latent adapter. No field or slot ABI changes.
"""

import logging
import os
from contextlib import nullcontext

import torch

logger = logging.getLogger(__name__)


def dense_enabled(owner):
    fs = getattr(owner, "fullstack", None)
    if fs and "duet_spec" in fs:
        # Codec storage and dense recurrence are independent native controls.
        # r>0/W=0 is also dense recurrence plus prompt-end projection.
        return fs.get("gdn_rank") == 0 or fs.get("gdn_every") == 0
    from twinstar_sgl.pd_dense import enabled

    return enabled(owner)


def transient_emitter_streams(owner, streams, embeddings, fb):
    if getattr(owner, "fullstack_code", False) and owner.fullstack.get(
        "prefill_layer_trim", True
    ):
        from .serving import embedding_streams

        # Match AGG: quantization-aware emitter input without allocating or
        # publishing a latent cache. The raw h31 boundary was saved separately.
        streams = owner.latent_codec.reconstruct(
            streams, fb.positions, embedding_streams(owner, embeddings)
        )
    elif owner.fullstack.get("latent") == "on":
        raise ValueError("shallow PD requires a native transient or stored latent path")
    return streams


def attach(owner, runner):
    from twinstar_sgl.pd_shallow import BoundaryState, SplitBoundaryPhase

    if owner.pd_shallow_role is None:
        return
    from twinstar_sgl.pd_final_metadata import install as install_final_metadata

    install_final_metadata(owner, runner)
    if os.environ.get("TWINSTAR_PD_FINAL_ADDRESS_PLAN") == "1":
        from twinstar_sgl.pd_final_address_dispatch import (
            install as install_final_addresses,
        )

        install_final_addresses(owner, runner)
    if os.environ.get("TWINSTAR_PD_FINAL_PAGE_COMMIT") == "1":
        from twinstar_sgl.pd_final_pagecommit import install as install_final_pagecommit

        install_final_pagecommit(owner, runner)
    if os.environ.get("TWINSTAR_PD_FINAL_LAUNCH_AUDIT"):
        from twinstar_sgl.pd_final_launch_audit import (
            install as install_final_launch_audit,
        )

        install_final_launch_audit(owner, runner)
    from sglang.srt.disaggregation.state_handoff import HandoffKind

    pool = runner.req_to_token_pool
    if not hasattr(pool, "pd_boundary_state"):
        custom = getattr(pool.mamba_pool, "custom_mem_pool", None)
        with torch.cuda.use_mem_pool(custom) if custom is not None else nullcontext():
            state = BoundaryState(pool.mamba_pool.size, runner.device)
        pool.pd_boundary_state = state
        pool.mamba_pool.register_slot_state(state)
        if owner.pd_shallow_role == "prefill":
            from twinstar_sgl.pd_dense import attach as attach_dense

            if dense_enabled(owner):
                attach_dense(pool, state)
            else:
                handlers = pool.pd_state_handoffs
                handlers[HandoffKind.STATE_FACTOR] = SplitBoundaryPhase(
                    handlers[HandoffKind.STATE_FACTOR], state
                )
    from .pd_factor_deferred import install

    install(owner, runner)


@torch.no_grad()
def prefill_extend(owner, input_ids, positions, fb):
    """P31 full extend, with native N-1/recurrent split only inside GDN."""
    from sglang.srt.eplb.expert_distribution import (
        get_global_expert_distribution_recorder,
    )
    from sglang.srt.layers.communicator import get_attn_tp_context
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.models import qwen4_exp as stock
    from twinstar_sgl.pd_shallow import boundary_inputs, capture_extend_boundary
    from twinstar_sgl.pd_shallow_gdn import split_boundary

    if dense_enabled(owner):
        from twinstar_sgl.pd_dense import split_boundary
    if sum(map(int, fb.extend_seq_lens_cpu)) != input_ids.shape[0]:
        raise ValueError("shallow PD requires an unpadded extend")
    if fb.return_logprob or len(owner.bridges):
        raise ValueError("shallow PD supports next-token logits without bridges")
    backend = get_attn_backend()
    body = owner.model.model
    linear = backend.linear_attn_backend
    ms = owner._boundary_lens(fb)
    emitter_fb, emitter_idx = owner._sub_batch(fb, input_ids, positions, "p")
    emitter_fb.flashnext_gdn_layer_range = (0, 35)
    # One native N-1 plan is reused for all layers. It is made before any
    # recurrent append can mark a live slot stale; its exact ring sources and
    # per-layer radix checkpoint indices remain authoritative for that prefix.
    prefix_metadata = None
    if emitter_fb.batch_size:
        linear.init_forward_metadata(emitter_fb)
        prefix_metadata = linear.forward_metadata
    boundary = boundary_idx = boundary_metadata = None
    if any(ms):
        boundary, boundary_idx = boundary_inputs(owner, input_ids, positions, fb)
        linear.init_forward_metadata(boundary)
        boundary_metadata = linear.forward_metadata
    from twinstar_sgl.pd_final_metadata import initialize_shallow

    initialize_shallow(backend.full_attn_backend, fb)
    linear.forward_metadata = prefix_metadata
    from .pd_factor_deferred import transaction_for

    deferred = transaction_for(
        owner, fb, emitter_fb, boundary, prefix_metadata, boundary_metadata
    )
    prefetched = None
    if os.environ.get("SGLANG_FLASHNEXT_ARRIVAL_OVERLAP", "0") == "1":
        from sglang.srt.mem_cache.flashnext_arrival_overlap import begin
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        prefetched = begin(
            owner,
            fb,
            [i for i, length in enumerate(ms) if length],
            get_token_to_kv_pool(),
        )
    if deferred is not None:
        from sglang.srt.mem_cache.gdn_prefill_exact_tail import split_boundary
    context = (
        split_boundary(
            linear,
            emitter_fb,
            emitter_idx,
            boundary,
            boundary_idx,
            prefix_metadata,
            boundary_metadata,
        )
        if any(ms)
        else nullcontext()
    )
    with deferred if deferred is not None else nullcontext(), context:
        with get_attn_tp_context().maybe_input_scattered(fb):
            embeddings = body.embed_tokens(input_ids)
            hidden = embeddings
            ple = (
                stock._prepare_ple_batch(
                    input_ids,
                    fb,
                    ngram_size=body.ple_ngram_size,
                    ngram_eos_token_id=body.ple_ngram_eos_token_id,
                )
                if body.has_ple
                else None
            )
            residual = None
            recorder = get_global_expert_distribution_recorder()
            for layer_id in owner.p_layer_ids:
                next_ple = getattr(body.layers[layer_id + 1], "ple", None)
                if next_ple is not None:
                    next_ple.start_prefetch(ple, fb)
                with recorder.with_current_layer(layer_id):
                    hidden, residual = body.layers[layer_id](
                        positions=positions,
                        hidden_states=hidden,
                        residual=residual,
                        forward_batch=fb,
                        ple_batch=ple,
                    )
            stock._commit_ple_batch(ple, fb)
            if residual is not None:
                raise RuntimeError("unsupported non-None h31 residual")
        boundary_hidden = (
            capture_extend_boundary(boundary, boundary_idx, hidden)
            if boundary is not None
            else None
        )
        if emitter_fb.batch_size:
            backend.full_attn_backend.init_forward_metadata(emitter_fb)
            linear.forward_metadata = prefix_metadata
            streams = hidden[emitter_idx]
            with get_attn_tp_context().maybe_input_scattered(emitter_fb):
                if owner.fullstack_v3_latent:
                    from .serving import encode_prefix

                    decoded = encode_prefix(
                        owner, streams, embeddings[emitter_idx], emitter_fb
                    )
                    if owner.fullstack_final:
                        streams = decoded
                else:
                    streams = transient_emitter_streams(
                        owner, streams, embeddings[emitter_idx], emitter_fb
                    )
                from twinstar_sgl.pd_emitter_graph import try_emit

                if not try_emit(
                    owner,
                    streams,
                    emitter_fb,
                    backend=backend,
                    prefix_metadata=prefix_metadata,
                    split=any(ms),
                ):
                    for layer_id in owner.emitter_ids:
                        emitter = owner.emitters[str(layer_id)]
                        if owner.fullstack_v3_latent and (
                            not owner.fullstack_final or emitter.is_attn
                        ):
                            continue
                        emitter.emit(streams, emitter_fb)
    if owner.fullstack_v3_latent and any(ms):
        from .serving import materialize_arrivals

        materialize_arrivals(
            owner,
            fb,
            [i for i, length in enumerate(ms) if length],
            prefetched=prefetched,
        )
    if deferred is not None:
        deferred.finish_return()
    if any(ms):
        owner._publish_qsa_prefix(fb, ms)
        from twinstar_sgl.pd_shallow_audit import snapshot

        snapshot(owner, boundary, "before-deep", hidden=boundary_hidden)
        owner.pd_boundary_requests = getattr(owner, "pd_boundary_requests", 0) + sum(ms)
        logger.info(
            "Flash-Next P boundary: extra_forwards=0 extend_boundary_requests=%d wire_phase=%s",
            owner.pd_boundary_requests,
            "dense:N/N-1"
            if dense_enabled(owner)
            else (
                "r/r:shallow-S_N/deep-S_N-1"
                if deferred is not None and deferred.final
                else "9/8"
            ),
        )
    owner.n_twinstar += 1
    if any(int(p) > 0 for p in fb.extend_prefix_lens_cpu):
        owner.n_prefix += 1
    audit = getattr(owner, "pd_final_launch_audit", None)
    if audit is not None:
        audit.record(fb)
    return LogitsProcessorOutput(
        next_token_logits=torch.zeros(
            fb.batch_size,
            owner.config.vocab_size,
            dtype=torch.float32,
            device=input_ids.device,
        )
    )
