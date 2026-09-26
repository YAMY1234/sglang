"""Per-request arrival segments sharing an owned batch source; compute-only graphs."""
from copy import copy
import logging
import os
import torch
from .flashnext_arrival_overlap import CHUNK, PrefixStore, verify_prefix
from .flashnext_scheme_c import SchemeCBatch

logger = logging.getLogger(__name__)


def segments(fb, final_rows, pool):
    """Host metadata only; preserve the original final-row publication order."""
    slots = fb.req_pool_indices_cpu.tolist()
    result, offset = [], 0
    for row in final_rows:
        slot = int(slots[row])
        stop = int(fb.extend_prefix_lens_cpu[row]) // CHUNK * CHUNK
        if not stop or slot in pool.materialized:
            continue
        if stop > 262144:
            raise ValueError('arrival prefix exceeds production context')
        result.append(dict(row=row, slot=slot, stop=stop, offset=offset))
        offset += stop
    if len({item['slot'] for item in result}) != len(result):
        raise ValueError('duplicate request slot in arrival batch')
    return result


def prepare(model, pool, item):
    payload = pool.load_latent(pool.request_pool.req_to_token[item['slot'], :item['stop']])
    token_ids = payload.pop('token_ids').flatten().long()
    base = model.model.model.embed_tokens(token_ids)
    if base.shape[-1] not in (model.config.hidden_size, model.config.hc_count*model.config.hidden_size):
        raise ValueError('unexpected arrival embedding width')
    # Keep the original embedding compact. Exact repeated bytes are expanded
    # inside the graph, avoiding four copies in the persistent batch source.
    return dict(payload=payload, token_ids=token_ids, base=base, stop=item['stop'])


def bind_segment(source, prepared, offset):
    stop = offset + prepared['stop']
    if offset < 0 or stop > source.capacity or set(prepared['payload']) != set(source.payload):
        raise ValueError('batch arrival source segment out of range')
    for name, value in prepared['payload'].items():
        source.payload[name][offset:stop].copy_(value)
    source.token_ids[offset:stop].copy_(prepared['token_ids'])
    source.base[offset:stop].copy_(prepared['base'])


class BatchBuffers:
    def __init__(self, model, pool, fb, source, *, first, count=CHUNK):
        self.model, self.pool, self.fb = model, pool, copy(fb)
        self.source, self.first, self.count = source, first, count
        self.control = torch.zeros(3, dtype=torch.int64, device=pool.device)
        self.offsets = torch.arange(count, device=pool.device)
        self.sink_rows = torch.zeros(1 if first else 0, dtype=torch.long, device=pool.device)
        self.fb.req_pool_indices = self.control[:1]
        self.fb.req_pool_indices_cpu = torch.zeros(1, dtype=torch.long)
        self.emitters = [model.emitters[str(layer)] for layer in model.emitter_ids if model.emitters[str(layer)].is_attn]
        if len(self.emitters) != 5 or count % 4:
            raise ValueError('five emitters and complete compression groups required')

    def bind(self, item, start):
        host = torch.tensor([item['slot'], start, item['offset']], dtype=torch.int64)
        if self.control.is_cuda: host = host.pin_memory()
        self.control.copy_(host, non_blocking=self.control.is_cuda)
        if hasattr(self, 'material'):
            self.material.req_pool_indices_cpu = host[:1]
            self.material.flashnext_arrival_plan.request_slot = item['slot']
        return host

    def bind_device(self, item, control):
        self.control.copy_(control)
        if hasattr(self, 'material'):
            self.material.flashnext_arrival_plan.request_slot = item['slot']

    def evaluate(self):
        from .flashnext_materialization import make_batch
        positions = self.offsets + self.control[1]
        packed = positions + self.control[2]
        payload = {name: value[packed] for name, value in self.source.payload.items()}
        token_ids = self.source.token_ids[packed]
        rp = self.pool.request_pool
        slots = rp.translate_mamba_indices(rp.get_mamba_indices(self.control[:1])).long()
        sink = self.pool.request_state.sink[slots] if self.first else self.pool.request_state.sink[:0]
        latent = SchemeCBatch(**payload, sink_rows=self.sink_rows, sink_values=sink)
        base = self.source.base[packed]
        if base.shape[-1] == self.model.config.hidden_size:
            base = base.repeat(1, self.model.config.hc_count)
        private = self.pool.deep_req_to_token[self.control[:1].expand(self.count), positions]
        material = make_batch(self.fb, 0, 0, self.count, token_ids, private,
                              self.pool.deep, False, implementation='kv-only')
        material.positions.add_(self.control[1])
        material.flashnext_arrival_plan.rope.add_(self.control[1])
        streams = self.model.latent_codec.decode(latent, base)
        for emitter in self.emitters: emitter.emit(streams, material)
        self.material = material


class BatchOverlap:
    def __init__(self, model, pool):
        self.model, self.pool = model, pool
        self.stream = torch.cuda.Stream(device=pool.device)
        self.source, self.entries = None, {}
        self.stats = dict(batches=0, requests=0, chunks=0, captured=0, source_bytes=0, packed=0)

    def launch(self, fb, items):
        current = torch.cuda.current_stream(self.pool.device)
        needed = sum(item['stop'] for item in items)
        packed = os.environ.get('SGLANG_FLASHNEXT_ARRIVAL_PACK', '0') == '1'
        if packed:
            from .flashnext_arrival_pack import prepare_packed, control_rows, transfer_controls
            prepared = prepare_packed(self.model, self.pool, items)
            steps = control_rows(items, CHUNK)
            control_host, controls = transfer_controls(steps, self.pool.device)
        else:
            prepared = prepare(self.model, self.pool, items[0])
        if self.source is None or self.source.capacity < needed:
            # Graph storage may still have queued uses from the preceding
            # batch. Growth is rare and explicitly synchronized/accounted.
            if self.source is not None:
                current.synchronize()
            self.entries.clear()
            self.source = None
            capacity = 1 << (needed-1).bit_length()
            self.source = PrefixStore(prepared, capacity)
            self.stats['source_bytes'] = sum(x.numel()*x.element_size() for x in
                (*self.source.payload.values(), self.source.token_ids, self.source.base))
            logger.info('Flash-Next batch arrival source: tokens=%d bytes=%d', capacity, self.stats['source_bytes'])
        if packed:
            bind_segment(self.source, prepared, 0)
        else:
            for index, item in enumerate(items):
                if index: prepared = prepare(self.model, self.pool, item)
                bind_segment(self.source, prepared, item['offset'])
        self.stream.wait_stream(current)
        hosts, captured = [], self.stats['captured']
        with torch.cuda.stream(self.stream):
            cursor = 0
            for item in items:
                for start in range(0, item['stop'], CHUNK):
                    key = (start == 0, os.environ.get('SGLANG_FLASHNEXT_DECODE_EPILOGUE', '0'),
                           self.model.latent_codec.compute_precision, torch.backends.cuda.matmul.allow_tf32)
                    entry = self.entries.get(key)
                    if entry is None:
                        if len(self.entries) >= 4: raise RuntimeError('batch arrival graph policy changed')
                        buffers = BatchBuffers(self.model, self.pool, fb, self.source, first=start == 0, count=CHUNK)
                        if packed: buffers.bind_device(item, controls[cursor])
                        else: hosts.append(buffers.bind(item, start))
                        buffers.evaluate()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=self.stream, capture_error_mode='thread_local'): buffers.evaluate()
                        entry = self.entries[key] = (buffers, graph)
                        self.stats['captured'] += 1
                    elif packed: entry[0].bind_device(item, controls[cursor])
                    else: hosts.append(entry[0].bind(item, start))
                    entry[1].replay(); self.stats['chunks'] += 1
                    cursor += 1
            done = torch.cuda.Event(); done.record(self.stream)
        if packed:
            controls.record_stream(self.stream)
            hosts.extend((control_host, controls))
            self.stats['packed'] += 1
            logger.info('Flash-Next packed arrival used: requests=%d chunks=%d', len(items), cursor)
        self.stats['batches'] += 1; self.stats['requests'] += len(items)
        return dict(event=done, host_controls=hosts, starts={x['slot']: x['stop'] for x in items},
                    rows={x['slot']: x['row'] for x in items}, batch=fb,
                    capture_guard=self.stats['captured'] != captured)


def begin(model, fb, final_rows, pool):
    if (os.environ.get("SGLANG_FLASHNEXT_ARRIVAL_OVERLAP", "0") != "1"
            or not model.fullstack_final or not model.fullstack_v3_latent
            or not getattr(pool, 'shared_arena', False) or not final_rows
            or fb.spec_info is not None or not fb.input_ids.is_cuda
            or torch.cuda.is_current_stream_capturing()):
        return None
    if os.environ.get('TWINSTAR_MATERIALIZATION_IMPL') != 'kv-only':
        raise ValueError('batch arrival overlap requires production kv-only')
    items = segments(fb, final_rows, pool)
    if not items:
        logger.info('Flash-Next batch arrival coverage: batch=%d final=%d selected=0 chunks=0', fb.batch_size, len(final_rows))
        return None
    rp = pool.request_pool
    state_slots = rp.translate_mamba_indices(rp.get_mamba_indices(
        fb.req_pool_indices[[x['row'] for x in items]])).long()
    if not bool(pool.request_state.sink_valid[state_slots].all()):
        raise RuntimeError('batch arrival prefix has no published exact sink')
    overlap = getattr(model, '_pdfix_batch_overlap', None)
    if overlap is None: overlap = model._pdfix_batch_overlap = BatchOverlap(model, pool)
    handle = overlap.launch(fb, items)
    logger.info('Flash-Next batch arrival coverage: batch=%d final=%d selected=%d chunks=%d source_bytes=%d',
        fb.batch_size, len(final_rows), len(items), sum(x['stop']//CHUNK for x in items), overlap.stats['source_bytes'])
    if handle['capture_guard']:
        torch.cuda.current_stream(pool.device).wait_event(handle['event'])
        chunks = verify_prefix(model, pool, fb, handle)
        logger.info('Flash-Next batch arrival overlap: native KV/index byte guard passed; requests=%d chunks=%d',len(items),chunks)
    return handle
