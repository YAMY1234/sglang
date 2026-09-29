"""P-only model ordering for one complete GDN prefill publication."""

import importlib
import logging
import os
from contextlib import ExitStack, contextmanager
from functools import wraps

import torch

logger = logging.getLogger(__name__)
FLAG = "SGLANG_GDN_PREFILL_BATCH_GRAPH"


def _unsupported(owner, batch, input_ids, *, factor=False, args=(), kwargs=None):
    kwargs = kwargs or {}
    mode = batch.forward_mode
    if not mode.is_extend() or mode.is_mixed() or batch.spec_info is not None:
        return "forward-mode"
    lengths = batch.extend_seq_lens_cpu
    final = getattr(batch, "twinstar_prompt_final", None)
    if lengths is None or final is None or len(final) != len(lengths):
        return "prompt-final-metadata"
    if not 1 <= batch.batch_size <= 16:
        return "batch-bucket"
    if len(lengths) != batch.batch_size or sum(lengths) != input_ids.shape[0]:
        return "padded-extend"
    if any(int(n) - int(last) <= 0 for n, last in zip(lengths, final)):
        return "empty-prefix"
    if getattr(batch, "can_run_tbo", False):
        return "two-batch-overlap"
    if any(x is not None for x in (getattr(batch, "mm_inputs", None) or ())):
        return "multimodal"
    if (getattr(batch, "input_embeds", None) is not None
            or kwargs.get("input_embeds") is not None or args
            or kwargs.get("pp_proxy_tensors") is not None
            or kwargs.get("input_deepstack_embeds") is not None
            or kwargs.get("get_embedding")):
        return "nontext-input"
    if not factor and (batch.return_logprob or len(owner.bridges)):
        return "shallow-logprobs-or-bridges"
    if os.environ.get("SGLANG_FLASHNEXT_ARRIVAL_OVERLAP") == "1":
        return "arrival-overlap"
    return None


def _fallback(owner, arm, reason):
    stats = owner._gdn_prefill_model_split_stats
    key = "fallback_" + reason
    stats[key] = stats.get(key, 0) + 1
    logger.warning("GDN prefill model split fallback: arm=%s reason=%s count=%d",
                   arm, reason, stats[key])


def _backend():
    from sglang.srt.model_executor.forward_context import get_attn_backend

    return get_attn_backend()


def _input_scope(batch):
    from sglang.srt.layers.communicator import get_attn_tp_context

    return get_attn_tp_context().maybe_input_scattered(batch)


def _observe(name, function, batch, *args):
    if not os.environ.get("TWINSTAR_CUDA_TIMELINE"):
        return function(*args)
    from twinstar_sgl.pd_boundary_observe import call

    return call(name, function, args, {}, batch=batch)


@contextmanager
def _collect(pool, plan, prefix):
    collector = pool.collect_prefill_batch(plan)
    with ExitStack() as stack:
        collector.__enter__()

        def close(*exc):
            if exc[0] is not None:
                return collector.__exit__(*exc)
            return _observe("P_all_layer_factor_commit", collector.__exit__, prefix, *exc)

        stack.push(close)
        yield


def _prefix_metadata(backend, prefix):
    linear = backend.linear_attn_backend
    pool = linear.factored
    if (pool is None or not pool.cfg.strict_chunk or pool.cfg.init_method != "k31"
            or not pool.cfg.factored_prefix or len(pool.layer_ids) != 36):
        raise RuntimeError("GDN model split requires the native 36-layer strict k31 pool")
    if pool.prefix_layer_count() != len(pool.layer_ids):
        raise RuntimeError("GDN model split requires factors for every prefix layer")
    prefix.flashnext_gdn_layer_range = (0, 35)
    prefix.pd_factor_only_full_batch = False
    linear.init_forward_metadata(prefix)
    metadata = linear.forward_metadata
    plan = metadata.factored_extend
    if plan is None:
        raise RuntimeError("GDN model split is missing its prefix plan")
    plan.last_layer = getattr(pool, "num_layers", len(pool.layer_ids)) - 1
    return pool, metadata, plan


def _tail_rids(batch, tail):
    rids = getattr(batch, "rids", None)
    if rids is not None and len(rids) == batch.batch_size:
        tail.rids = [rid for rid, final in zip(rids, batch.twinstar_prompt_final) if final]


def _shallow_hidden(owner, batch):
    from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
    from sglang.srt.models import qwen4_exp as stock

    body = owner.model.model
    embeddings = body.embed_tokens(batch.input_ids)
    hidden = embeddings
    ple = (stock._prepare_ple_batch(batch.input_ids, batch,
            ngram_size=body.ple_ngram_size, ngram_eos_token_id=body.ple_ngram_eos_token_id)
           if body.has_ple else None)
    residual = None
    recorder = get_global_expert_distribution_recorder()
    for layer_id in owner.p_layer_ids:
        next_ple = getattr(body.layers[layer_id + 1], "ple", None)
        if next_ple is not None:
            next_ple.start_prefetch(ple, batch)
        with recorder.with_current_layer(layer_id):
            hidden, residual = body.layers[layer_id](positions=batch.positions,
                hidden_states=hidden, residual=residual, forward_batch=batch, ple_batch=ple)
    stock._commit_ple_batch(ple, batch)
    if residual is not None:
        raise RuntimeError("GDN model split requires the native h31 streams")
    return hidden, embeddings


def _emit_prefix(owner, prefix, hidden, embeddings, backend, metadata):
    backend.full_attn_backend.init_forward_metadata(prefix)
    backend.linear_attn_backend.forward_metadata = metadata
    with _input_scope(prefix):
        streams = hidden
        if owner.fullstack_v3_latent:
            from twinstar_sgl.fullstack_v3_serving import encode_prefix

            decoded = encode_prefix(owner, streams, embeddings, prefix)
            if owner.fullstack_final:
                streams = decoded
        elif owner.fullstack["latent"] == "on":
            raise ValueError("GDN model split requires the served v3 latent path")
        # The old emitter graph contains its own commits and cannot collect.
        for layer_id in owner.emitter_ids:
            emitter = owner.emitters[str(layer_id)]
            if owner.fullstack_v3_latent and (not owner.fullstack_final or emitter.is_attn):
                continue
            emitter.emit(streams, prefix)


@torch.no_grad()
def _shallow_prefill(owner, input_ids, positions, batch):
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from twinstar_sgl import pd_shallow as pd
    from twinstar_sgl.pd_final_metadata import initialize_shallow

    backend = _backend()
    linear = backend.linear_attn_backend
    saved = linear.forward_metadata
    lengths = owner._boundary_lens(batch)
    prefix, _ = owner._sub_batch(batch, input_ids, positions, "p")
    tail = pd.boundary_inputs(owner, input_ids, positions, batch)[0] if any(lengths) else None
    if tail is not None:
        _tail_rids(batch, tail)
    try:
        pool, metadata, plan = _prefix_metadata(backend, prefix)
        initialize_shallow(backend.full_attn_backend, prefix)

        def forward_prefix():
            with _input_scope(prefix):
                hidden, embeddings = _shallow_hidden(owner, prefix)
            _emit_prefix(owner, prefix, hidden, embeddings, backend, metadata)

        with _collect(pool, plan, prefix):
            _observe("P_all_layer_prefix", forward_prefix, prefix)
        boundary_hidden = None
        if tail is not None:
            from sglang.srt.model_executor.gdn_prefill_tail_graph import FLAG as TAIL_FLAG, execute

            if os.environ.get(TAIL_FLAG) != "1":
                backend.init_forward_metadata(tail)

            def forward_tail():
                value = execute(owner, tail)
                if value is not None:
                    return value
                with _input_scope(tail):
                    return _shallow_hidden(owner, tail)[0]

            boundary_hidden = _observe("P_model_tail", forward_tail, tail)
            rows = torch.arange(tail.batch_size, device=boundary_hidden.device)
            pd.capture_extend_boundary(tail, rows, boundary_hidden)
            if owner.fullstack_v3_latent:
                from twinstar_sgl.fullstack_v3_serving import materialize_arrivals

                materialize_arrivals(owner, batch, [i for i, n in enumerate(lengths) if n])
            owner._publish_qsa_prefix(batch, lengths)
            from twinstar_sgl.pd_shallow_audit import snapshot

            snapshot(owner, tail, "before-deep", hidden=boundary_hidden)
            owner.pd_boundary_requests = getattr(owner, "pd_boundary_requests", 0) + sum(lengths)
        owner.n_twinstar += 1
        if any(int(p) > 0 for p in batch.extend_prefix_lens_cpu):
            owner.n_prefix += 1
        audit = getattr(owner, "pd_final_launch_audit", None)
        if audit is not None:
            audit.record(batch)
        owner._gdn_prefill_model_split_stats["1+2"] += 1
        return LogitsProcessorOutput(next_token_logits=torch.zeros(
            batch.batch_size, owner.config.vocab_size, dtype=torch.float32, device=input_ids.device))
    finally:
        linear.forward_metadata = saved


def _assemble(parts, total):
    result = None
    for indices, values in parts:
        if values is None:
            continue
        if result is None:
            result = values.new_empty((total, *values.shape[1:]))
        result[indices] = values
    return result


@contextmanager
def _split_body(owner, batch, prefix, prefix_indices, tail, tail_indices, backend):
    body = owner.model.model
    original = body.forward
    linear = backend.linear_attn_backend
    saved = linear.forward_metadata

    def run(input_ids, positions, forward_batch, input_embeds=None,
            pp_proxy_tensors=None, input_deepstack_embeds=None):
        if (forward_batch is not batch or pp_proxy_tensors is not None
                or input_deepstack_embeds is not None
                or getattr(batch, "input_embeds", None) is not None
                or any(value is not None for value in (getattr(batch, "mm_inputs", None) or ()))):
            raise ValueError("GDN full-depth split requires the original text P batch")
        total = batch.input_ids.shape[0]
        if input_embeds is not None:
            if input_ids is not None or input_embeds.ndim != 2 or input_embeds.shape[0] != total:
                raise ValueError("GDN full-depth split requires native text embeddings")
        elif input_ids is None or input_ids.shape[0] != total:
            raise ValueError("GDN full-depth split is missing the original text IDs")

        def run_part(part, indices, name):
            # The VL text path supplies embeddings but PLE still needs real IDs.
            arguments = (part.input_ids, part.positions, part)
            if input_embeds is not None:
                arguments += (input_embeds[indices],)
            return _observe(name, original, part, *arguments)

        pool, metadata, plan = _prefix_metadata(backend, prefix)
        backend.full_attn_backend.init_forward_metadata(prefix)
        linear.forward_metadata = metadata
        with _collect(pool, plan, prefix), _input_scope(prefix):
            prefix_output = run_part(prefix, prefix_indices, "P_all_layer_prefix")
        parts = [(prefix_indices, prefix_output)]
        hc_parts = [(prefix_indices, body.last_hc_hidden_states)]
        if tail is not None:
            from sglang.srt.model_executor.gdn_prefill_tail_graph import FLAG as TAIL_FLAG, execute

            if os.environ.get(TAIL_FLAG) != "1":
                backend.init_forward_metadata(tail)

            def forward_tail():
                value = execute(owner, tail)
                if value is not None:
                    return value
                arguments = (tail.input_ids, tail.positions, tail)
                if input_embeds is not None:
                    arguments += (input_embeds[tail_indices],)
                with _input_scope(tail):
                    return original(*arguments)

            output = _observe("P_model_tail", forward_tail, tail)
            parts.append((tail_indices, output))
            hc_parts.append((tail_indices, body.last_hc_hidden_states))
        body.last_hc_hidden_states = _assemble(hc_parts, total)
        return _assemble(parts, total)

    body.forward = run
    try:
        yield
    finally:
        body.forward = original
        linear.forward_metadata = saved


def _factor_prefill(owner, original, input_ids, positions, batch, *args, **kwargs):
    from twinstar_sgl.pd_factor_only import BatchSlicer
    from twinstar_sgl.pd_shallow import boundary_inputs

    slicer = BatchSlicer(owner)
    prefix, prefix_indices = slicer._sub_batch(batch, input_ids, positions, "p")
    tail = tail_indices = None
    if any(slicer._boundary_lens(batch)):
        tail, tail_indices = boundary_inputs(slicer, input_ids, positions, batch)
        tail.pd_factor_only_full_batch = False
        tail.can_run_decode_cuda_graph = False
        _tail_rids(batch, tail)
    with _split_body(owner, batch, prefix, prefix_indices, tail, tail_indices, _backend()):
        result = original(owner, input_ids, positions, batch, *args, **kwargs)
    owner._gdn_prefill_model_split_stats["factor"] += 1
    return result


def install_prefill_model_split(model):
    if os.environ.get(FLAG) != "1":
        return False
    from sglang.srt.runtime_context import get_disagg

    if get_disagg().disaggregation_mode != "prefill":
        return False
    if getattr(model, "_gdn_prefill_model_split_installed", False):
        return True
    shallow = getattr(model, "pd_shallow_role", None) == "prefill"
    factor = os.environ.get("TWINSTAR_PD_FACTOR_ONLY_TAIL") == "1"
    if not shallow and not factor:
        raise ValueError("GDN prefill model split requires an admitted P model")
    if (getattr(model, "n_layers", None) != 48
            or (shallow and list(model.p_layer_ids) != list(range(31)))
            or (factor and (model.fullstack or model.emitters))):
        raise ValueError("GDN prefill model split has an unexpected layer contract")
    model._gdn_prefill_model_split_stats = {"1+2": 0, "factor": 0}
    module = importlib.import_module("twinstar_sgl.pd_shallow" if shallow
                                     else "twinstar_sgl.pd_factor_only")
    name = "prefill_extend" if shallow else "forward"
    original = getattr(module, name)
    if shallow:
        @wraps(original)
        def dispatch(owner, input_ids, positions, batch):
            if owner is not model:
                return original(owner, input_ids, positions, batch)
            reason = _unsupported(owner, batch, input_ids)
            if reason:
                _fallback(owner, "1+2", reason)
                return original(owner, input_ids, positions, batch)
            return _shallow_prefill(owner, input_ids, positions, batch)
    else:
        @wraps(original)
        def dispatch(owner, native, input_ids, positions, batch, *args, **kwargs):
            if owner is not model:
                return original(owner, native, input_ids, positions, batch, *args, **kwargs)
            reason = _unsupported(owner, batch, input_ids, factor=True, args=args, kwargs=kwargs)
            if reason:
                _fallback(owner, "factor", reason)
                return original(owner, native, input_ids, positions, batch, *args, **kwargs)
            return _factor_prefill(owner, native, input_ids, positions, batch, *args, **kwargs)
    setattr(module, name, dispatch)
    model._gdn_prefill_model_split_installed = True
    logger.info("GDN prefill model split installed: arm=%s normal_and_tracked_layers=36",
                "1+2" if shallow else "factor")
    return True
