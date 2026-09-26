"""Default-off full-chunk Scheme-C decode and five-layer KV graph.

Only group-aligned full chunks enter capture. Resolved physical page indices,
positions, payload, base and sink are rebound into owned buffers on every call.
Tail pending state and request publication remain in the native caller.
"""
from copy import copy
from dataclasses import fields
import logging
import os

import torch

logger = logging.getLogger(__name__)


def unpack_payload(payload, latent_fields, *, gathered=None):
    """Keep replicated fields local; concatenate sharded bytes in TP order."""
    out = {}
    offset = 0
    sharded = ('z', 'spike_indices', 'spike_values', 'z_block_scale')
    for name, width, dtype in latent_fields:
        size = width * dtype.itemsize
        if gathered is not None and name in sharded:
            value = torch.cat((gathered[:, offset:offset+size],
                               gathered[:, 2048+offset:2048+offset+size]), -1)
        else:
            value = payload[:, offset:offset+size].contiguous()
        out[name] = value.view(dtype)
        offset += size
    if offset != 1972:
        raise ValueError('Scheme-C wire layout changed')
    if gathered is not None:
        out['z_block_scale'] = out['z_block_scale'].view(torch.float8_e4m3fn)
    return out


class RebuildBuffers:
    plan_fields = ('kv', 'compressed', 'group_rows', 'rope')

    def __init__(self, latent, base, fb):
        self.latent = type(latent)(**{f.name: getattr(latent, f.name).clone() for f in fields(latent)})
        self.base = base.clone()
        self.fb = copy(fb)
        self.fb.positions = fb.positions.clone()
        self.fb.flashnext_arrival_plan = copy(fb.flashnext_arrival_plan)
        for name in self.plan_fields:
            setattr(self.fb.flashnext_arrival_plan, name, getattr(fb.flashnext_arrival_plan, name).clone())
        plan = self.fb.flashnext_arrival_plan
        if plan.tail:
            # Capture an owned, rebound address vector, never a Python slot's
            # slice baked into the graph. Incomplete groups keep native math.
            plan.pending_locs = torch.arange(plan.tail, device=base.device) + plan.request_slot * 4

    def bind(self, latent, base, fb):
        for field in fields(latent):
            getattr(self.latent, field.name).copy_(getattr(latent, field.name))
        self.base.copy_(base)
        self.fb.positions.copy_(fb.positions)
        for name in self.plan_fields:
            getattr(self.fb.flashnext_arrival_plan, name).copy_(getattr(fb.flashnext_arrival_plan, name))
        plan = self.fb.flashnext_arrival_plan
        if plan.tail:
            plan.request_slot = fb.flashnext_arrival_plan.request_slot
            plan.pending_locs.copy_(torch.arange(plan.tail, device=base.device) + plan.request_slot * 4)

    def evaluate(self, codec, emitters):
        streams = codec.decode(self.latent, self.base)
        for emitter in emitters:
            emitter.emit(streams, self.fb)

    def written(self, emitters):
        """Read the actual KV/index publication locations for byte admission."""
        plan = self.fb.flashnext_arrival_plan
        result = []
        for emitter in emitters:
            layer = emitter.layer_id
            local = plan.deep._transfer_full_attention_id(layer)
            for getter, locations in ((plan.deep.get_key_buffer, plan.kv[local]),
                                      (plan.deep.get_value_buffer, plan.kv[local]),
                                      (plan.deep.get_qsa_compressed_k_buffer, plan.compressed[local].long())):
                result.append(getter(layer)[locations].clone())
            if plan.tail:
                slot = plan.request_slot * 4
                result.append(plan.deep.get_qsa_key_state_buffer(layer)[slot:slot+plan.tail].clone())
        if plan.tail:
            slot = plan.request_slot * 4
            result.append(plan.deep.qsa_rope_position_buffer[slot:slot+plan.tail].clone())
        return result


class RebuildGraph:
    def __init__(self):
        self.entries = {}
        self.tail_pool = None
        self.tail_stream = None
        self.stats = dict(captured=0, rebound_checked=0, replayed=0, fallback=0, owned_bytes=0)

    def run(self, codec, emitters, latent, base, fb, *, verify=False):
        plan = fb.flashnext_arrival_plan
        tails = os.environ.get('SGLANG_FLASHNEXT_REBUILD_TAIL_GRAPH', '0') == '1'
        partial = plan.count != 8192 or bool(plan.tail)
        if (not base.is_cuda or (partial and not tails) or not 0 < plan.count <= 8192 or plan.implementation != 'kv-only'
                or torch.cuda.is_current_stream_capturing()):
            self.stats['fallback'] += 1
            return False
        if len(emitters) != 5 or any(not e.is_attn for e in emitters):
            raise ValueError('rebuild graph requires exactly five QSA emitters')
        backing = tuple((plan.deep.get_key_buffer(e.layer_id).data_ptr(),
                         plan.deep.get_value_buffer(e.layer_id).data_ptr(),
                         plan.deep.get_qsa_compressed_k_buffer(e.layer_id).data_ptr()) for e in emitters)
        if plan.tail:
            backing += tuple((plan.deep.get_qsa_key_state_buffer(e.layer_id).data_ptr(),
                              plan.deep.qsa_rope_position_buffer.data_ptr()) for e in emitters)
        shapes = tuple((f.name, tuple(getattr(latent, f.name).shape), getattr(latent, f.name).dtype)
                       for f in fields(latent))
        key = (shapes, base.device, backing, id(codec), tuple(id(e) for e in emitters),
               codec.compute_precision, torch.backends.cuda.matmul.allow_tf32,
               os.environ.get('SGLANG_FLASHNEXT_DECODE_EPILOGUE', '0') == '1')
        entry = self.entries.get(key)
        if entry is None:
            # Keep the independent epilogue switch in the capture identity.
            # Each arithmetic policy still retains at most two sink shapes.
            limit = 256 if tails else 2
            if sum(k[-1] == key[-1] for k in self.entries) >= limit:
                self.stats['fallback'] += 1
                return False
            inputs = [getattr(latent, f.name) for f in fields(latent)] + [base, fb.positions]
            inputs += [getattr(plan, name) for name in RebuildBuffers.plan_fields]
            owned_bytes = sum(x.numel() * x.element_size() for x in inputs) + plan.tail * 8
            if tails and self.stats['owned_bytes'] + owned_bytes > 24 * 1024**3:
                self.stats['fallback'] += 1
                return False
            buffers = RebuildBuffers(latent, base, fb)
            current = torch.cuda.current_stream(base.device)
            if tails:
                if self.tail_stream is None:
                    self.tail_stream = torch.cuda.Stream(device=base.device)
                stream = self.tail_stream
            else:
                stream = torch.cuda.Stream(device=base.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                # Native caller plan has no pending address vector. Its exact
                # slice publication is the reference for the captured stores.
                reference = codec.decode(latent, base)
                for emitter in emitters: emitter.emit(reference, fb)
            current.wait_stream(stream)
            expected = buffers.written(emitters)
            graph = torch.cuda.CUDAGraph()
            if tails and self.tail_pool is None:
                self.tail_pool = torch.cuda.graph_pool_handle()
            # Every graph here finishes its stores before another graph uses
            # this workspace on the foreground stream; outputs stay in banks.
            with torch.cuda.graph(graph, stream=stream, pool=self.tail_pool if tails else None,
                                  capture_error_mode='thread_local'):
                buffers.evaluate(codec, emitters)
            graph.replay()
            self.verify(expected, buffers.written(emitters))
            entry = dict(buffers=buffers, graph=graph, stream=stream, rebound_checked=False)
            self.entries[key] = entry
            self.stats['captured'] += 1
            self.stats['owned_bytes'] += owned_bytes
            logger.info('Flash-Next rebuild graph: capture byte guard passed; sink_rows=%d count=%d tail=%d',
                        latent.sink_rows.numel(), plan.count, plan.tail)
        else:
            buffers = entry['buffers']
            buffers.bind(latent, base, fb)
            if verify or not entry['rebound_checked']:
                reference = codec.decode(latent, base)
                for emitter in emitters: emitter.emit(reference, fb)
                expected = buffers.written(emitters)
                entry['graph'].replay()
                self.verify(expected, buffers.written(emitters))
                if not entry['rebound_checked']:
                    entry['rebound_checked'] = True
                    self.stats['rebound_checked'] += 1
                    logger.info('Flash-Next rebuild graph: rebound page/input byte guard passed')
            else:
                entry['graph'].replay()
        self.stats['replayed'] += 1
        return True

    @staticmethod
    def verify(expected, actual):
        if any(not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
               for a, b in zip(expected, actual, strict=True)):
            raise RuntimeError('Flash-Next rebuild graph changed KV/index bytes')
