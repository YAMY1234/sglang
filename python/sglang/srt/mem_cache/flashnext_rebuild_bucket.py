"""Default-off 128-token tail buckets with dynamically masked publication.

Each request reuses one length graph regardless of its containing batch/sink.
All GEMMs keep the original N; only norm/compression/publication are padded.
Padding never publishes to KV, compressed pages or another request's ring.
"""
from copy import copy
import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _scatter_rows(SRC, DST, LOC, CONTROL, WIDTH: tl.constexpr,
                  SRC_ROW: tl.constexpr, DST_ROW: tl.constexpr,
                  DIVISOR: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = index // WIDTH
    col = index % WIDTH
    valid = row < tl.load(CONTROL) // DIVISOR
    loc = tl.load(LOC + row, valid, other=0).to(tl.int64)
    value = tl.load(SRC + row * SRC_ROW + col, valid & (col < WIDTH), other=0)
    tl.store(DST + loc * DST_ROW + col, value, valid & (col < WIDTH))


@triton.jit
def _pending(KEY, ROPE, DST, DST_ROPE, CONTROL, WIDTH: tl.constexpr,
             KEY_ROW: tl.constexpr, DST_ROW: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    count = tl.load(CONTROL)
    slot = tl.load(CONTROL + 1).to(tl.int64)
    valid = row < count % 4
    source = count // 4 * 4 + row
    value = tl.load(KEY + source * KEY_ROW + col, valid & (col < WIDTH), other=0)
    tl.store(DST + (slot * 4 + row) * DST_ROW + col, value, valid & (col < WIDTH))
    pos = tl.load(ROPE + source * 3 + col, valid & (col < 3), other=0)
    tl.store(DST_ROPE + (slot * 4 + row) * 3 + col, pos, valid & (col < 3))


def scatter_rows(source, destination, locations, control, *, divisor=1):
    width = source.shape[-1]
    if source.ndim != 2 or not destination.is_contiguous():
        raise ValueError('bucket publication requires flat source rows and contiguous banks')
    if (destination.dtype != source.dtype or source.stride(-1) != 1
            or destination[0].numel() != width):
        raise ValueError('bucket publication must preserve storage bytes')
    _scatter_rows[(triton.cdiv(source.shape[0] * width, 256),)](
        source, destination, locations, control, width, source.stride(0),
        destination.stride(0), divisor, 256)


def publish_pending(token_k, plan, layer):
    target = plan.deep.get_qsa_key_state_buffer(layer)
    key = token_k.reshape(token_k.shape[0], -1)
    _pending[(3,)](key, plan.rope, target, plan.deep.qsa_rope_position_buffer,
                  plan.bucket_control, key.shape[1], key.stride(0), target.stride(0),
                  triton.next_power_of_2(max(key.shape[1], 3)))


class BucketBuffers:
    def __init__(self, projected, fb, first_layer):
        sample_k = projected[0][0]
        count = ((sample_k.shape[0] + 127) // 128) * 128
        device = sample_k.device
        self.projected = [tuple(torch.zeros((count, *t.shape[1:]), dtype=t.dtype, device=device)
                                for t in tensors) for tensors in projected]
        self.fb = copy(fb)
        self.fb.positions = torch.zeros(count, dtype=fb.positions.dtype, device=device)
        plan = self.fb.flashnext_arrival_plan = copy(fb.flashnext_arrival_plan)
        plan.kv = torch.zeros((5, count), dtype=plan.kv.dtype, device=device)
        plan.compressed = torch.zeros((5, count // 4), dtype=plan.compressed.dtype, device=device)
        plan.group_rows = torch.arange(count, dtype=plan.group_rows.dtype, device=device).reshape(-1, 4)
        plan.rope = torch.zeros((count, 3), dtype=plan.rope.dtype, device=device)
        plan.count, plan.groups, plan.tail = count, count // 4, 0
        plan.bucket_control = torch.empty(2, dtype=torch.int64, device=device)
        sample = plan.deep.get_qsa_compressed_k_buffer(first_layer)
        plan.bucket_index = torch.empty((count // 4, sample[0].numel()), dtype=sample.dtype, device=device)
        plan.bucket_index_locs = torch.arange(count // 4, dtype=torch.int32, device=device)
        self.bind(projected, fb)

    def bind(self, projected, fb):
        n = fb.flashnext_arrival_plan.count
        for targets, tensors in zip(self.projected, projected, strict=True):
            for target, tensor in zip(targets, tensors, strict=True):
                target[:n].copy_(tensor)
        self.fb.positions[:n].copy_(fb.positions)
        plan, original = self.fb.flashnext_arrival_plan, fb.flashnext_arrival_plan
        plan.kv[:, :n].copy_(original.kv)
        plan.compressed[:, :n // 4].copy_(original.compressed)
        plan.rope[:n].copy_(original.rope)
        plan.bucket_control.copy_(torch.tensor([n, original.request_slot], dtype=torch.int64,
                                               device=plan.kv.device))
        self.original = fb

    def evaluate(self, emitters, publisher=None):
        if publisher is None:
            from .flashnext_materialization import publish_bucket
            publisher = publish_bucket
        for emitter, tensors in zip(emitters, self.projected, strict=True):
            publisher(emitter, *tensors, self.fb)

    def written(self, emitters):
        from .flashnext_rebuild_graph import RebuildBuffers
        return RebuildBuffers.written(type('View', (), {'fb': self.original})(), emitters)

    def owned_bytes(self):
        p = self.fb.flashnext_arrival_plan
        tensors = [t for group in self.projected for t in group]
        tensors += [self.fb.positions, p.kv, p.compressed, p.group_rows,
                    p.rope, p.bucket_control, p.bucket_index, p.bucket_index_locs]
        return sum(t.numel() * t.element_size() for t in tensors)


class BucketGraph:
    def __init__(self, stats):
        self.entries = {}
        self.pool = None
        self.stream = None
        self.stats = stats

    def run(self, codec, emitters, latent, base, fb, *, verify=False):
        from .flashnext_rebuild_graph import RebuildGraph
        plan = fb.flashnext_arrival_plan
        count = (plan.count + 127) // 128 * 128
        backing = tuple((plan.deep.get_key_buffer(e.layer_id).data_ptr(),
                         plan.deep.get_value_buffer(e.layer_id).data_ptr(),
                         plan.deep.get_qsa_compressed_k_buffer(e.layer_id).data_ptr(),
                         plan.deep.get_qsa_key_state_buffer(e.layer_id).data_ptr()) for e in emitters)
        key = (count, backing,
               plan.deep.qsa_rope_position_buffer.data_ptr(), id(codec),
               tuple(id(e) for e in emitters), codec.compute_precision,
               torch.backends.cuda.matmul.allow_tf32,
               os.environ.get('SGLANG_FLASHNEXT_DECODE_EPILOGUE', '0') == '1')
        entry = self.entries.get(key)
        if entry is None and (len(self.entries) >= 256
                              or self.stats['owned_bytes'] + count * 29000 > 24 * 1024**3):
            self.stats['fallback'] += 1
            return False
        from .flashnext_materialization import project_bucket
        streams = codec.decode(latent, base)
        projected = [project_bucket(e, streams) for e in emitters]
        if entry is None:
            buffers = BucketBuffers(projected, fb, emitters[0].layer_id)
            current = torch.cuda.current_stream(base.device)
            if self.stream is None:
                self.stream = torch.cuda.Stream(device=base.device)
                self.pool = torch.cuda.graph_pool_handle()
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                reference = streams
                for emitter in emitters:
                    emitter.emit(reference, fb)
                expected = buffers.written(emitters)
                buffers.evaluate(emitters)
                RebuildGraph.verify(expected, buffers.written(emitters))
            current.wait_stream(self.stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream, pool=self.pool,
                                  capture_error_mode='thread_local'):
                buffers.evaluate(emitters)
            graph.replay()
            RebuildGraph.verify(expected, buffers.written(emitters))
            self.entries[key] = entry = dict(buffers=buffers, graph=graph, rebound_checked=False)
            self.stats['captured'] += 1
            self.stats['owned_bytes'] += buffers.owned_bytes()
            logger.info('Flash-Next bucket publication graph: byte guard passed; count=%d bucket=%d',
                        plan.count, count)
        else:
            buffers = entry['buffers']
            buffers.bind(projected, fb)
            if verify or not entry['rebound_checked']:
                reference = streams
                for emitter in emitters:
                    emitter.emit(reference, fb)
                expected = buffers.written(emitters)
                entry['graph'].replay()
                RebuildGraph.verify(expected, buffers.written(emitters))
                self.stats['rebound_checked'] += 1
                entry['rebound_checked'] = True
            else:
                entry['graph'].replay()
        self.stats['replayed'] += 1
        self.stats['bucket_replayed'] = self.stats.get('bucket_replayed', 0) + 1
        self.stats['bucket_real_tokens'] = self.stats.get('bucket_real_tokens', 0) + plan.count
        self.stats['bucket_padding_tokens'] = self.stats.get('bucket_padding_tokens', 0) + count - plan.count
        return True
