"""P singleton recurrent tail graphs after this layer's prefix publication.

First/last layers keep native invalidation, cross-layer truncation and tracking.
Only original backend arithmetic is captured. Warmup restores all mutated rows.
"""
from copy import copy
import logging
import torch

logger = logging.getLogger(__name__)


class TailBuffers:
    def __init__(self, backend, layer, batch, qkv, a, b):
        self.backend, self.layer = backend, layer
        self.batch = copy(batch)
        self.metadata = copy(backend.forward_metadata)
        self.metadata.mamba_cache_indices = backend.forward_metadata.mamba_cache_indices.clone()
        start = self.metadata.query_start_loc
        if start is not None:
            self.metadata.query_start_loc = start.clone()
        self.qkv, self.a, self.b = qkv.clone(), a.clone(), b.clone()
        pool = backend.factored
        fa, fu, fw, count, _ = pool.layer_tensors(layer.layer_id)
        conv = backend.req_to_token_pool.mamba2_layer_cache(layer.layer_id).conv[0]
        self.banks = (conv, fa, fu, fw, count, pool.stale)

    def bind(self, qkv, a, b, metadata):
        self.qkv.copy_(qkv); self.a.copy_(a); self.b.copy_(b)
        self.metadata.mamba_cache_indices.copy_(metadata.mamba_cache_indices)
        if self.metadata.query_start_loc is not None:
            self.metadata.query_start_loc.copy_(metadata.query_start_loc)

    def snapshot(self):
        indices = self.metadata.mamba_cache_indices.long()
        return tuple(bank.index_select(0, indices).clone() for bank in self.banks)

    def restore(self, values):
        indices = self.metadata.mamba_cache_indices.long()
        for bank, value in zip(self.banks, values, strict=True):
            bank.index_copy_(0, indices, value)

    def evaluate(self, eager, kwargs):
        previous = self.backend.forward_metadata
        self.backend.forward_metadata = self.metadata
        try:
            return eager(self.layer, self.batch, self.qkv, self.a, self.b, **kwargs)
        finally:
            self.backend.forward_metadata = previous


class PrefillTailGraph:
    def __init__(self):
        self.entries = {}
        self.stream = None
        self.pool = None
        self.stats = dict(captured=0, replayed=0, rebound_checked=0, fallback=0, owned_bytes=0, private_reserved_bytes=0)

    @staticmethod
    def eligible(backend, layer, batch, qkv):
        pool = backend.factored
        if (not qkv.is_cuda or qkv.shape[0] != 1 or batch.batch_size != 1
                or pool is None or pool.layer_index(layer.layer_id) == 0
                or pool.is_last_layer(layer.layer_id)
                or not backend._factored_batch_trunc or backend._factored_side_stream is not None
                or torch.cuda.is_current_stream_capturing()):
            return False
        meta = backend.forward_metadata
        if meta.replayssm_write_pos is not None or meta.replayssm_force_flush is not None:
            return False
        # The original tracking function must be a host-only early return.
        if batch.mamba_track_mask is not None:
            tracked = backend._track_pools()
            if tracked is None or backend.req_to_token_pool.mamba_map.get(layer.layer_id) == tracked[2]:
                return False
        return True

    @staticmethod
    def same(expected, actual):
        if len(expected) != len(actual) or any(
                a.shape != b.shape or a.dtype != b.dtype
                or not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
                for a, b in zip(expected, actual, strict=True)):
            raise RuntimeError('P recurrent tail graph changed output or state bytes')

    def run(self, backend, layer, batch, qkv, a, b, *, eager, kwargs, verify=False):
        if not self.eligible(backend, layer, batch, qkv):
            self.stats['fallback'] += 1
            return eager(layer, batch, qkv, a, b, **kwargs)
        pool = backend.factored
        fa, fu, fw, count, vbar = pool.layer_tensors(layer.layer_id)
        conv = backend.req_to_token_pool.mamba2_layer_cache(layer.layer_id).conv[0]
        tensors = (conv, fa, fu, fw, count, pool.stale, vbar,
                   layer.conv_weights, layer.A_log, layer.dt_bias)
        key = (id(backend), layer.layer_id, tuple(t.data_ptr() for t in tensors),
               tuple((t.shape, t.dtype, t.stride()) for t in (qkv, a, b)),
               repr(pool.cfg.kernel_kwargs()),
               tuple((k, v if isinstance(v, (str,int,float,bool,type(None))) else id(v))
                     for k,v in sorted(kwargs.items())),
               backend.forward_metadata.query_start_loc is not None,
               batch.mamba_track_mask is not None)
        entry = self.entries.get(key)
        metadata = backend.forward_metadata
        if entry is None:
            # Two stable tracking variants per layer; arbitrary caller metadata
            # must not grow graph storage without bound.
            if len(self.entries) >= 2 * len(pool.layer_ids):
                self.stats['fallback'] += 1
                return eager(layer, batch, qkv, a, b, **kwargs)
            free, _ = torch.cuda.mem_get_info(qkv.device)
            if free < 512 * 1024**2:
                raise RuntimeError('P recurrent tail graph has less than 512 MiB capture headroom')
            buf = TailBuffers(backend, layer, batch, qkv, a, b)
            initial = buf.snapshot()
            # The reference uses independent QKV because Conv1D may overwrite
            # its argument. Caller-owned activations must survive warmup.
            reference = eager(layer, batch, qkv.clone(), a, b, **kwargs).clone()
            expected = (reference, *buf.snapshot())
            buf.restore(initial)
            buf.bind(qkv, a, b, metadata)
            current = torch.cuda.current_stream(qkv.device)
            if self.stream is None:
                self.stream = torch.cuda.Stream(device=qkv.device)
                self.pool = torch.cuda.graph_pool_handle()
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                buf.evaluate(eager, kwargs)
            current.wait_stream(self.stream)
            buf.restore(initial)
            buf.bind(qkv, a, b, metadata)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream, pool=self.pool,
                                  capture_error_mode='thread_local'):
                output = buf.evaluate(eager, kwargs)
            buf.restore(initial)
            buf.bind(qkv, a, b, metadata)
            graph.replay()
            self.same(expected, (output, *buf.snapshot()))
            self.entries[key] = entry = dict(buffers=buf, graph=graph, output=output,
                                              rebound_checked=False)
            self.stats['captured'] += 1
            owned = (buf.qkv, buf.a, buf.b, buf.metadata.mamba_cache_indices,
                     buf.metadata.query_start_loc)
            self.stats['owned_bytes'] += sum(t.numel()*t.element_size() for t in owned if t is not None)
            private = sum(x['total_size'] for x in torch.cuda.memory_snapshot()
                          if tuple(x['segment_pool_id']) == tuple(self.pool))
            self.stats['private_reserved_bytes'] = private
            if private + self.stats['owned_bytes'] > 128 * 1024**2:
                raise RuntimeError('P recurrent tail graph exceeded its 128 MiB total storage budget')
            logger.info('GDN prefill tail memory: owned=%d private_reserved=%d peak_allocated=%d free=%d',
                        self.stats['owned_bytes'], private, torch.cuda.max_memory_allocated(),
                        torch.cuda.mem_get_info()[0])
            logger.info('GDN prefill tail graph captured: layer=%d byte guard passed', layer.layer_id)
        else:
            buf = entry['buffers']
            buf.bind(qkv, a, b, metadata)
            if verify or not entry['rebound_checked']:
                initial = buf.snapshot()
                reference = eager(layer, batch, qkv.clone(), a, b, **kwargs).clone()
                expected = (reference, *buf.snapshot())
                buf.restore(initial)
                # Snapshot/reference calls never mutate the owned inputs.
                entry['graph'].replay()
                self.same(expected, (entry['output'], *buf.snapshot()))
                self.stats['rebound_checked'] += 1
                entry['rebound_checked'] = True
            else:
                entry['graph'].replay()
        self.stats['replayed'] += 1
        return entry['output']
