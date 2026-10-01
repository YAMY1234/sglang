"""Optional graph of the pure prefill factorization, without pool pointers.

Each entry owns input, probe, and output tensors. Live slot/ring addresses and
publication metadata never enter a capture. Callers receive independent output
storage, so a tracked-state call cannot overwrite a pending normal-state result.
"""
from dataclasses import replace
import logging

import torch

from sglang.srt.utils.graph_capture import graph_capture_lock

logger = logging.getLogger(__name__)


class FactorizeBuffers:
    def __init__(self, states, vbar, cfg, *, omega=None):
        self.states = tuple(x.clone() for x in states)
        self.vbar = vbar.clone()
        self.cfg = replace(cfg)
        b, h, v, _ = states[0].shape
        self.explicit_omega = omega is not None
        self.broadcast_omega = omega is not None and b > 1 and omega.stride(0) == 0
        if omega is None:
            generator = torch.Generator(device=states[0].device).manual_seed(0)
            self.omega_storage = torch.randn(b, h, v, cfg.r + cfg.init_oversample,
                                             device=states[0].device, generator=generator)
        else:
            if omega.shape[:3] != (b, h, v) or omega.device != states[0].device:
                raise ValueError('explicit factor graph omega shape/device mismatch')
            self.omega_storage = (omega[:1] if self.broadcast_omega else omega).clone()
        self.omega = (self.omega_storage.expand_as(omega) if self.broadcast_omega
                      else self.omega_storage)

    def bind(self, states, vbar, *, omega=None):
        from .gdn_graph_copy import bind_many
        if self.explicit_omega != (omega is not None):
            raise ValueError('factor graph omega policy changed')
        sources, destinations = (*states, vbar), (*self.states, self.vbar)
        if omega is not None:
            sources += (omega[:1] if self.broadcast_omega else omega,)
            destinations += (self.omega_storage,)
        bind_many(sources, destinations)

    def evaluate(self, eager):
        return eager(self.states, self.vbar, self.cfg, omega=self.omega)

    @staticmethod
    def independent(outputs):
        from .gdn_graph_copy import clone_many
        values = iter(clone_many(x for row in outputs for x in row))
        return [tuple(next(values) for _ in row) for row in outputs]


class PrefillFactorGraph:
    """Bounded singleton/small-batch cache; all other shapes remain eager."""
    def __init__(self, *, max_input_bytes=4*1024*1024, max_entries=4,
                 max_total_input_bytes=None, max_retained_bytes=None):
        self.entries = {}
        self.max_input_bytes = max_input_bytes
        self.max_entries = max_entries
        self.max_total_input_bytes = max_total_input_bytes
        self.max_retained_bytes = max_retained_bytes
        self.stats = dict(captured=0, replayed=0, fallback=0,
                          input_bytes=0, retained_bytes=0,
                          allocated_growth_bytes=0, reserved_growth_bytes=0)

    def run(self, states, vbar, cfg, *, eager, policy, omega=None):
        first = states[0]
        def fallback():
            self.stats['fallback'] += 1
            if omega is None:
                return eager(states, vbar, cfg)
            return eager(states, vbar, cfg, omega=omega)
        # Do not nest capture in D/MTP/model graphs. Limit retained workspaces;
        # large cross-layer and GSM batches keep their existing eager path.
        size = sum(x.numel() * x.element_size() for x in states) + vbar.numel() * vbar.element_size()
        if omega is not None:
            # An expanded batch owns only one probe row; preserve that layout.
            probe = omega[:1] if omega.shape[0] > 1 and omega.stride(0) == 0 else omega
            size += probe.numel() * probe.element_size()
        if (not first.is_cuda or torch.cuda.is_current_stream_capturing()
                or size > self.max_input_bytes or first.shape[0] > 16):
            return fallback()
        shapes = tuple((tuple(x.shape), x.dtype, x.device) for x in (*states, vbar))
        config = (cfg.r, cfg.rmax, cfg.dtype, cfg.init_iters, cfg.init_oversample, cfg.init_method)
        omega_key = (None if omega is None else
                     (tuple(omega.shape), tuple(omega.stride()), omega.dtype, omega.device))
        key = (shapes, config, policy, eager, omega_key, torch.backends.cuda.matmul.allow_tf32,
               torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
               torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        entry = self.entries.get(key)
        if entry is None:
            if (len(self.entries) >= self.max_entries or
                    (self.max_total_input_bytes is not None and
                     self.stats['input_bytes']+size > self.max_total_input_bytes)):
                return fallback()
            before_bytes = torch.cuda.memory_allocated(first.device)
            before_reserved = torch.cuda.memory_reserved(first.device)
            buffers = FactorizeBuffers(states, vbar, cfg, omega=omega)
            current = torch.cuda.current_stream(first.device)
            stream = torch.cuda.Stream(device=first.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                buffers.evaluate(eager)  # compile/initialize the unchanged ops
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with graph_capture_lock, torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
                outputs = buffers.evaluate(eager)
            allocated_growth = max(0, torch.cuda.memory_allocated(first.device)-before_bytes)
            reserved_growth = max(0, torch.cuda.memory_reserved(first.device)-before_reserved)
            # Captured intermediates may be inactive allocator blocks retained
            # by a graph's private pool. Do not charge only active tensors.
            retained = max(allocated_growth, reserved_growth)
            if (self.max_retained_bytes is not None and
                    self.stats['retained_bytes']+retained > self.max_retained_bytes):
                # No pool state was captured or published. Fail the opt-in
                # candidate rather than consume an unaccounted prefix budget.
                raise RuntimeError('factor graph retained workspace exceeds admitted budget')
            entry = (buffers, graph, outputs, stream)
            self.entries[key] = entry
            self.stats['captured'] += 1
            self.stats['input_bytes'] += size
            self.stats['retained_bytes'] += retained
            self.stats['allocated_growth_bytes'] += allocated_growth
            self.stats['reserved_growth_bytes'] += reserved_growth
            logger.info('GDN prefill factor graph captured: shapes=%s entries=%d explicit_omega=%s input_bytes=%d retained_bytes=%d',
                        shapes, len(self.entries), omega is not None,
                        self.stats['input_bytes'], self.stats['retained_bytes'])
        else:
            entry[0].bind(states, vbar, omega=omega)
        entry[1].replay()
        self.stats['replayed'] += 1
        return FactorizeBuffers.independent(entry[2])
