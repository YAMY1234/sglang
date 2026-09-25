"""P-only complete-chunk rebuild, queued ahead of the remaining P forward.

Only already-published latent tokens enter this stream. All TP communication
finishes on the foreground stream before compute-only background graphs start.
The caller joins the completion event before rebuilding the suffix/publishing.
"""
from copy import copy
import logging
import os

import torch

from .flashnext_rebuild_graph import RebuildGraph
from .flashnext_scheme_c import SchemeCBatch

logger = logging.getLogger(__name__)
CHUNK = 8192


def prepare_prefix(model, pool, slot, stop):
    """Original TP operations, once for all complete prefix chunks."""
    payload = pool.load_latent(pool.request_pool.req_to_token[slot, :stop])
    token_ids = payload.pop('token_ids').flatten().long()
    base = model.model.model.embed_tokens(token_ids)
    width = model.config.hc_count * model.config.hidden_size
    if base.shape[-1] != width:
        if base.shape[-1] != model.config.hidden_size:
            raise ValueError('unexpected arrival embedding width')
        base = base.repeat(1, model.config.hc_count)
    return dict(payload=payload, token_ids=token_ids, base=base, stop=stop)


class PrefixStore:
    """Stable graph inputs; one foreground copy per field and request."""
    def __init__(self, prepared, capacity):
        def reserve(value):
            return torch.empty((capacity, *value.shape[1:]), dtype=value.dtype, device=value.device)
        self.payload = {name: reserve(value) for name, value in prepared['payload'].items()}
        self.token_ids = reserve(prepared['token_ids'])
        self.base = reserve(prepared['base'])
        self.capacity = capacity

    def bind(self, prepared):
        stop = prepared['stop']
        if stop > self.capacity or set(prepared['payload']) != set(self.payload):
            raise ValueError('arrival prefix input capacity/layout changed')
        for name, value in prepared['payload'].items():
            self.payload[name][:stop].copy_(value)
        self.token_ids[:stop].copy_(prepared['token_ids'])
        self.base[:stop].copy_(prepared['base'])


def written(emitters, material):
    plan = material.flashnext_arrival_plan
    result = []
    for emitter in emitters:
        local = plan.deep._transfer_full_attention_id(emitter.layer_id)
        for getter, locations in ((plan.deep.get_key_buffer, plan.kv[local]),
                                  (plan.deep.get_value_buffer, plan.kv[local]),
                                  (plan.deep.get_qsa_compressed_k_buffer, plan.compressed[local].long())):
            result.append(getter(emitter.layer_id)[locations].clone())
    return result


def verify_prefix(model, pool, fb, handle):
    """Compare actual background-written pages against the original TP path.

    Only capture admission and explicitly selected byte requests call this;
    ordinary timed replays launch no extra reads, collectives, or synchronizes.
    The caller has joined handle.event on the current stream.
    """
    from .flashnext_materialization import make_batch
    emitters = [model.emitters[str(layer)] for layer in model.emitter_ids
                if model.emitters[str(layer)].is_attn]
    rp = pool.request_pool
    count = 0
    for slot, stop in handle['starts'].items():
        for start in range(0, stop, CHUNK):
            payload = pool.load_latent(rp.req_to_token[slot, start:start+CHUNK])
            token_ids = payload.pop('token_ids').flatten().long()
            state_slot = rp.translate_mamba_indices(rp.get_mamba_indices(fb.req_pool_indices)).long()
            latent = SchemeCBatch(**payload,
                sink_rows=torch.zeros(1 if start == 0 else 0, dtype=torch.long, device=pool.device),
                sink_values=pool.request_state.sink[state_slot] if start == 0 else pool.request_state.sink[:0])
            base = model.model.model.embed_tokens(token_ids)
            if base.shape[-1] != model.config.hc_count * model.config.hidden_size:
                if base.shape[-1] != model.config.hidden_size:
                    raise ValueError('unexpected arrival embedding width')
                base = base.repeat(1, model.config.hc_count)
            material = make_batch(fb, 0, start, start+CHUNK, token_ids,
                pool.deep_req_to_token[slot, start:start+CHUNK], pool.deep, False)
            actual = written(emitters, material)
            streams = model.latent_codec.decode(latent, base)
            for emitter in emitters:
                emitter.emit(streams, material)
            RebuildGraph.verify(actual, written(emitters, material))
            count += 1
    return count


class ArrivalBuffers:
    def __init__(self, model, pool, fb, source, *, first, count=CHUNK):
        self.model, self.pool, self.fb = model, pool, copy(fb)
        self.source, self.first, self.count = source, first, count
        self.control = torch.zeros(2, dtype=torch.int64, device=pool.device)
        self.offsets = torch.arange(count, device=pool.device)
        self.sink_rows = torch.zeros(1 if first else 0, dtype=torch.long, device=pool.device)
        self.emitters = [model.emitters[str(layer)] for layer in model.emitter_ids
                         if model.emitters[str(layer)].is_attn]
        if len(self.emitters) != 5 or count % 4:
            raise ValueError('arrival graph requires five QSA emitters and complete groups')

    def bind(self, slot, start):
        # Each asynchronous H2D source has independent pinned storage. Reusing
        # one host control row would race with queued copies on this stream.
        host = torch.tensor([slot, start], dtype=torch.int64)
        if self.control.is_cuda:
            host = host.pin_memory()
        self.control.copy_(host, non_blocking=self.control.is_cuda)
        return host

    def evaluate(self):
        from .flashnext_materialization import make_batch
        pool, model = self.pool, self.model
        rp = pool.request_pool
        positions = self.offsets + self.control[1]
        rows = self.control[:1].expand(self.count)
        payload = {name: value[positions] for name, value in self.source.payload.items()}
        token_ids = self.source.token_ids[positions]
        slots = rp.translate_mamba_indices(rp.get_mamba_indices(self.control[:1])).long()
        sink = pool.request_state.sink[slots] if self.first else pool.request_state.sink[:0]
        latent = SchemeCBatch(**payload, sink_rows=self.sink_rows, sink_values=sink)
        base = self.source.base[positions]
        private_locs = pool.deep_req_to_token[rows, positions]
        material = make_batch(self.fb, 0, 0, self.count, token_ids, private_locs,
                              pool.deep, False, implementation='kv-only')
        # Original plan math with a dynamic start; positions and rope are
        # integer additions, while physical pages already use the live slice.
        material.positions.add_(self.control[1])
        material.flashnext_arrival_plan.rope.add_(self.control[1])
        streams = model.latent_codec.decode(latent, base)
        for emitter in self.emitters:
            emitter.emit(streams, material)
        self.material = material


class ArrivalOverlap:
    def __init__(self, model, pool):
        self.model, self.pool = model, pool
        self.source = None
        self.stream = torch.cuda.Stream(device=pool.device)
        self.entries = {}
        self.stats = dict(captured=0, chunks=0, requests=0)

    def launch(self, fb, slot, stop):
        if stop <= 0 or stop % CHUNK:
            raise ValueError('overlap prefix must consist of complete 8192-token chunks')
        current = torch.cuda.current_stream(self.pool.device)
        # No collective is captured or outstanding on the background stream.
        # These original operations and all source copies are in the P window.
        prepared = prepare_prefix(self.model, self.pool, slot, stop)
        if self.source is None:
            capacity = min(self.pool.request_pool.req_to_token.shape[1], 262144)
            self.source = PrefixStore(prepared, capacity)
        self.source.bind(prepared)
        self.stream.wait_stream(current)
        hosts = []
        captured_before = self.stats['captured']
        with torch.cuda.stream(self.stream):
            for start in range(0, stop, CHUNK):
                first = start == 0
                key = (first, os.environ.get('SGLANG_FLASHNEXT_DECODE_EPILOGUE', '0') == '1',
                       self.model.latent_codec.compute_precision, torch.backends.cuda.matmul.allow_tf32)
                entry = self.entries.get(key)
                if entry is None:
                    if len(self.entries) >= 4:
                        raise RuntimeError('arrival overlap capture policy changed beyond the two registered sink shapes')
                    buffers = ArrivalBuffers(self.model, self.pool, fb, self.source, first=first)
                    hosts.append(buffers.bind(slot, start))
                    buffers.evaluate()  # warm the unchanged decode/emit/page operators
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=self.stream):
                        buffers.evaluate()
                    entry = self.entries[key] = (buffers, graph)
                    self.stats['captured'] += 1
                    logger.info('Flash-Next arrival overlap captured: first=%s', first)
                else:
                    hosts.append(entry[0].bind(slot, start))
                entry[1].replay()
                self.stats['chunks'] += 1
            done = torch.cuda.Event()
            done.record(self.stream)
        self.stats['requests'] += 1
        return dict(event=done, host_controls=hosts, starts={slot: stop}, batch=fb,
                    capture_guard=self.stats['captured'] != captured_before)


def begin(model, fb, final_rows, pool):
    """No cache-policy change: reuse only this forward's immutable old tokens."""
    if (os.environ.get('SGLANG_FLASHNEXT_ARRIVAL_OVERLAP', '0') != '1'
            or not model.fullstack_final or not model.fullstack_v3_latent
            or not getattr(pool, 'shared_arena', False) or fb.batch_size != 1
            or final_rows != [0] or fb.spec_info is not None
            or not fb.input_ids.is_cuda
            or torch.cuda.is_current_stream_capturing()):
        return None
    if os.environ.get('TWINSTAR_MATERIALIZATION_IMPL') != 'kv-only':
        raise ValueError('arrival overlap requires the production kv-only path')
    stop = int(fb.extend_prefix_lens_cpu[0]) // CHUNK * CHUNK
    if not stop or stop > 262144:
        return None
    slot = int(fb.req_pool_indices_cpu[0])
    if slot in pool.materialized:
        return None
    rp = pool.request_pool
    state_slot = rp.translate_mamba_indices(rp.get_mamba_indices(fb.req_pool_indices)).long()
    if not bool(pool.request_state.sink_valid[state_slot].all()):
        raise RuntimeError('overlap prefix has no published exact sink')
    overlap = getattr(model, '_pdfix_arrival_overlap', None)
    if overlap is None:
        overlap = model._pdfix_arrival_overlap = ArrivalOverlap(model, pool)
    handle = overlap.launch(fb, slot, stop)
    if handle['capture_guard']:
        torch.cuda.current_stream(pool.device).wait_event(handle['event'])
        chunks = verify_prefix(model, pool, fb, handle)
        logger.info('Flash-Next arrival overlap: native KV/index byte guard passed; chunks=%d', chunks)
    return handle
