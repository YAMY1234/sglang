"""v3 prompt encoding and once-per-arrival private emitter materialization."""
import copy
import os
import time

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import get_attn_backend, get_token_to_kv_pool
from .latent import LatentBatch

# #905 diagnostic: row provenance stamps / checks in the shared arena (numerics unchanged).
LATENT_AUDIT = os.environ.get("SGLANG_FLASHNEXT_LATENT_AUDIT", "0") == "1"


def embedding_streams(model, embeddings):
    width = model.config.hc_count * model.config.hidden_size
    if embeddings.shape[-1] == width:
        return embeddings
    if embeddings.shape[-1] != model.config.hidden_size:
        raise ValueError("unexpected v3 embedding width")
    return embeddings.repeat(1, model.config.hc_count)


def encode_prefix(model, streams, embeddings, fb):
    pool = get_token_to_kv_pool()
    base = embedding_streams(model, embeddings)
    decoded = None
    if model.fullstack_final and getattr(pool, 'shared_arena', False):
        batch, decoded = model.latent_codec.encode_and_decode(streams, fb.positions, base)
    else:
        batch = model.latent_codec.encode(streams, fb.positions, base)
    if LATENT_AUDIT:
        pool.store_latent(fb.out_cache_loc, batch, fb.input_ids, positions=fb.positions)
    else:
        pool.store_latent(fb.out_cache_loc, batch, fb.input_ids)
    rp = pool.request_pool
    slots = rp.translate_mamba_indices(rp.get_mamba_indices(fb.req_pool_indices)).long()
    lengths = torch.tensor(fb.extend_seq_lens_cpu, dtype=torch.long, device=streams.device)
    token_rows = torch.repeat_interleave(torch.arange(fb.batch_size, device=streams.device), lengths)
    sink_slots = slots[token_rows[batch.sink_rows]]
    pool.request_state.sink[sink_slots] = batch.sink_values
    pool.request_state.sink_valid[sink_slots] = 1
    # The last shallow GDN commits its prefix checkpoint before P31's final
    # residual is encoded. Copy the sink after encoding to those same slots.
    if fb.mamba_track_mask is not None and fb.mamba_track_indices is not None:
        mask = fb.mamba_track_mask[:fb.batch_size]
        dst = fb.mamba_track_indices[:fb.batch_size][mask].long()
        src = slots[mask]
        pool.request_state.sink[dst] = pool.request_state.sink[src]
        pool.request_state.sink_valid[dst] = pool.request_state.sink_valid[src]
    if model.fullstack_final:
        # New P tokens feed all deep recurrent emitters now; their prefix slots
        # survive radix reuse. Only QSA is rebuilt on request arrival.
        return decoded if decoded is not None else model.latent_codec.decode(batch, base)


def materialization_batch(fb, row, start, stop, token_ids, private_locs, final):
    nb = copy.copy(fb)
    device = token_ids.device
    nb.forward_mode = ForwardMode.EXTEND
    nb.batch_size = 1
    nb._original_batch_size = None
    nb.input_ids = token_ids.long()
    nb.positions = torch.arange(start, stop, dtype=torch.int64, device=device)
    nb.req_pool_indices = fb.req_pool_indices[row:row+1]
    nb.req_pool_indices_cpu = (fb.req_pool_indices_cpu[row:row+1]
                               if fb.req_pool_indices_cpu is not None else None)
    nb.seq_lens = torch.tensor([stop], dtype=torch.int64, device=device)
    nb.seq_lens_cpu = torch.tensor([stop], dtype=torch.int64)
    nb.seq_lens_sum = stop
    nb.orig_seq_lens = nb.seq_lens.to(torch.int32)
    nb.out_cache_loc = private_locs.long()
    nb.out_cache_loc_virtual = None
    nb.extend_num_tokens = stop - start
    nb.extend_seq_lens = torch.tensor([stop-start], dtype=torch.int32, device=device)
    nb.extend_prefix_lens = torch.tensor([start], dtype=torch.int32, device=device)
    nb.extend_start_loc = torch.zeros(1, dtype=torch.int32, device=device)
    nb.extend_seq_lens_cpu = [stop-start]
    nb.extend_prefix_lens_cpu = [start]
    nb.extend_logprob_start_lens_cpu = [0]
    nb.twinstar_prompt_final = [final]
    nb.flashnext_gdn_layer_range = (24, 35)
    nb.flashnext_private_locations = True
    nb.return_logprob = False
    nb.is_extend_in_batch = True
    nb.can_run_decode_cuda_graph = False
    nb.spec_info = None
    # ForwardBatch.can_run_tbo is a read-only property of the split index.
    nb.tbo_split_seq_index = None
    nb.tbo_parent_token_range = None
    nb.tbo_padded_len = None
    nb.tbo_children = None
    nb.mm_inputs = nb.input_embeds = nb.mrope_positions = None
    for name in ("mamba_track_indices", "mamba_track_mask", "mamba_track_seqlens"):
        setattr(nb, name, None)
    if getattr(fb, "global_num_token_non_padded_cpu", None) is not None:
        nb.global_num_token_non_padded_cpu = stop-start
    for name in ("global_num_token_non_padded", "num_token_non_padded"):
        value = getattr(fb, name, None)
        if value is not None:
            setattr(nb, name, torch.tensor(stop-start, dtype=value.dtype, device=device))
    for name in ("forward_metadata_ready", "forward_metadata_replan_equivalent"):
        if hasattr(nb, name):
            setattr(nb, name, False)
    for name in ("forward_metadata_planned_bs", "forward_metadata_planned_num_tokens"):
        if hasattr(nb, name):
            setattr(nb, name, None)
    return nb


def arrival_metadata(fb, row, start, stop, token_ids, private_locs, pool, backend, final, implementation):
    if implementation != "legacy":
        from sglang.srt.mem_cache.flashnext_materialization import make_batch
        return make_batch(fb, row, start, stop, token_ids, private_locs, pool.deep, final,
                          implementation=implementation)
    material = materialization_batch(fb, row, start, stop, token_ids, private_locs, final)
    backend.init_forward_metadata(material)
    return material


@torch.no_grad()
def materialize_arrivals(model, fb, rows, *, prompt_lengths=None, prefetched=None):
    pool = get_token_to_kv_pool()
    rp = pool.request_pool
    backend = get_attn_backend()
    row_slots = (fb.req_pool_indices_cpu if fb.req_pool_indices_cpu is not None else fb.req_pool_indices).tolist()
    chunk = model.fullstack["materialization_chunk"]
    implementation = os.environ.get("TWINSTAR_MATERIALIZATION_IMPL", "legacy")
    if implementation not in ("legacy", "kv-only", "kv-preserve"):
        raise ValueError("unknown final materialization implementation")
    fast = (model.fullstack_final and getattr(pool, "shared_arena", False)
            and implementation != "legacy")
    starts = {}
    if prefetched is not None:
        if prefetched['batch'] is not fb or not fast:
            raise ValueError('prefetched arrival does not belong to this final forward')
        torch.cuda.current_stream(pool.device).wait_event(prefetched['event'])
        starts = prefetched['starts']
    for row in rows:
        request_slot = int(row_slots[row])
        if request_slot in pool.materialized:
            continue
        total = int(prompt_lengths[row] if prompt_lengths is not None else fb.seq_lens_cpu[row]-1)
        if total < 0:
            raise ValueError("negative latent prompt length")
        slot = rp.translate_mamba_indices(rp.get_mamba_indices(fb.req_pool_indices[row:row+1])).long()
        if total and not bool(pool.request_state.sink_valid[slot].all()):
            raise RuntimeError("cached v3 prefix has no exact sink")
        if not model.fullstack_final:
            # Historical scheme-A B rebuilt every emitter. Final retains GDN.
            for tensor in rp.mamba_pool.mamba_cache.conv:
                tensor[24:, slot] = 0
            for tensor in (rp.factored_gdn_pool.a, rp.factored_gdn_pool.U, rp.factored_gdn_pool.W):
                tensor[24:, slot] = 0
            rp.factored_gdn_pool.count[24:, slot] = 8
        timing = model.profile or model.state_audit_dir is not None
        if timing:
            torch.cuda.synchronize()
            started = time.perf_counter()
        begin = starts.get(request_slot, 0)
        if begin < 0 or begin > total or begin % chunk:
            raise ValueError('prefetched prefix is not a complete subset of this arrival')
        for start in range(begin, total, chunk):
            stop = min(total, start+chunk)
            locations = rp.req_to_token[request_slot, start:stop]
            if LATENT_AUDIT:
                payload = pool.load_latent(locations, positions=torch.arange(start, stop, device=locations.device))
            else:
                payload = pool.load_latent(locations)
            token_ids = payload.pop("token_ids").flatten().long()
            sink_rows = torch.zeros(1 if start == 0 else 0, dtype=torch.long, device=token_ids.device)
            sink_values = pool.request_state.sink[slot] if start == 0 else pool.request_state.sink[:0]
            batch_type = LatentBatch
            if model.fullstack_final:
                from sglang.srt.mem_cache.flashnext_scheme_c import SchemeCBatch
                batch_type = SchemeCBatch
            latent = batch_type(**payload, sink_rows=sink_rows, sink_values=sink_values)
            embeddings = model.model.model.embed_tokens(token_ids)
            if fast and os.environ.get('SGLANG_FLASHNEXT_REBUILD_GRAPH', '0') == '1':
                from sglang.srt.mem_cache.flashnext_rebuild_graph import RebuildGraph
                graph = getattr(model, '_pdfix_rebuild_graph', None)
                if graph is None:
                    graph = model._pdfix_rebuild_graph = RebuildGraph()
                private_locs = pool.deep_req_to_token[request_slot, start:stop]
                material = arrival_metadata(fb, row, start, stop, token_ids, private_locs,
                                            pool, backend, stop == total, implementation)
                emitters = [model.emitters[str(layer)] for layer in model.emitter_ids
                            if model.emitters[str(layer)].is_attn]
                if graph.run(model.latent_codec, emitters, latent,
                             embedding_streams(model, embeddings), material):
                    continue
            streams = model.latent_codec.decode(latent, embedding_streams(model, embeddings))
            private_locs = pool.deep_req_to_token[request_slot, start:stop]
            material = arrival_metadata(fb, row, start, stop, token_ids, private_locs,
                                        pool, backend, stop == total, implementation if fast else "legacy")
            for layer in model.emitter_ids:
                if model.fullstack_final and not model.emitters[str(layer)].is_attn:
                    continue
                model.emitters[str(layer)].emit(streams, material)
        pool.materialized.add(request_slot)
        pool.stats["materializations"] += 1
        pool.stats["materialized_tokens"] += total
        if timing:
            torch.cuda.synchronize()
            pool.stats["materialization_ms"] += (time.perf_counter() - started) * 1000


def capture_boundary(module, args, kwargs):
    """Capture P's exact residual before layer 31, including graph replay."""
    fb = kwargs["forward_batch"]
    verify = fb.forward_mode.is_target_verify()
    if not fb.forward_mode.is_decode() and not verify:
        return
    pool = get_token_to_kv_pool()
    rp = pool.request_pool
    slots = rp.translate_mamba_indices(rp.get_mamba_indices(fb.req_pool_indices)).long()
    hidden = kwargs["hidden_states"]
    residual = kwargs.get("residual")
    streams = hidden if residual is None else hidden + residual
    if verify:
        pool.request_state.record_verify_boundary(streams, kwargs["positions"])
        return
    pool.request_state.boundary[slots] = streams
    pool.request_state.boundary_position[slots] = kwargs["positions"].reshape(-1, 1)


@torch.no_grad()
def prepare_pd_arrivals(model, fb):
    """Restore new D arrivals before either eager execution or graph replay.

    P sends shallow post-boundary state. Materialization creates deep
    pre-boundary state, so replay exactly the deep suffix of that boundary.
    """
    pool = get_token_to_kv_pool()
    host_indices = fb.req_pool_indices_cpu
    if host_indices is None:
        raise RuntimeError("v3 D arrival requires the scheduler's CPU request index mirror")
    requests = host_indices.tolist()
    rows = [i for i, slot in enumerate(requests) if int(slot) not in pool.materialized]
    if not rows:
        return
    totals = [pool.prompt_lengths[int(slot)] - 1 for slot in requests]
    rp = pool.request_pool
    slots = rp.translate_mamba_indices(rp.get_mamba_indices(fb.req_pool_indices)).long()
    received_positions = pool.request_state.boundary_position[slots[rows]].flatten().tolist()
    if received_positions != [totals[i] for i in rows]:
        raise RuntimeError("v3 PD boundary phase or latent metadata was not received")
    materialize_arrivals(model, fb, rows, prompt_lengths=totals)
    backend = get_attn_backend()
    for row in rows:
        total = totals[row]
        replay = copy.copy(fb)
        replay.batch_size = 1
        replay._original_batch_size = None
        replay.req_pool_indices = fb.req_pool_indices[row:row+1]
        replay.req_pool_indices_cpu = host_indices[row:row+1]
        replay.seq_lens = torch.tensor([total+1], dtype=fb.seq_lens.dtype, device=fb.seq_lens.device)
        replay.seq_lens_cpu = torch.tensor([total+1], dtype=torch.int64)
        replay.seq_lens_sum = total+1
        replay.positions = torch.tensor([total], dtype=torch.int64, device=fb.input_ids.device)
        replay.input_ids = fb.input_ids[row:row+1]  # deep layers contain no PLE/embedding reads
        replay.out_cache_loc = pool.deep_req_to_token[requests[row], total:total+1].long()
        replay.flashnext_private_locations = True
        replay.mamba_track_indices = replay.mamba_track_mask = replay.mamba_track_seqlens = None
        replay.can_run_decode_cuda_graph = False
        replay.spec_info = None
        for name in ("forward_metadata_ready", "forward_metadata_replan_equivalent"):
            if hasattr(replay, name):
                setattr(replay, name, False)
        hidden = pool.request_state.boundary[slots[row:row+1]]
        residual = None
        backend.init_forward_metadata(replay)
        for layer in model.emitter_ids:
            hidden, residual = model.model.model.layers[layer](positions=replay.positions,
                hidden_states=hidden, residual=residual, forward_batch=replay, ple_batch=None)
