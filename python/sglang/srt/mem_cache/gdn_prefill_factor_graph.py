"""Optional graph of the pure prefill factorization, without pool pointers.

Each entry owns input, probe, and output tensors. Live slot/ring addresses and
publication metadata never enter a capture. Callers receive independent output
storage, so a tracked-state call cannot overwrite a pending normal-state result.
"""
from dataclasses import replace
import logging

import torch

logger = logging.getLogger(__name__)


class FactorizeBuffers:
    def __init__(self, states, vbar, cfg):
        self.states = tuple(x.clone() for x in states)
        self.vbar = vbar.clone()
        self.cfg = replace(cfg)
        b, h, v, _ = states[0].shape
        generator = torch.Generator(device=states[0].device).manual_seed(0)
        self.omega = torch.randn(b, h, v, cfg.r + cfg.init_oversample,
                                 device=states[0].device, generator=generator)

    def bind(self, states, vbar):
        from .gdn_graph_copy import bind_many
        bind_many((*states, vbar), (*self.states, self.vbar))

    def evaluate(self, eager):
        return eager(self.states, self.vbar, self.cfg, omega=self.omega)

    @staticmethod
    def independent(outputs):
        from .gdn_graph_copy import clone_many
        values = iter(clone_many(x for row in outputs for x in row))
        return [tuple(next(values) for _ in row) for row in outputs]


class PrefillFactorGraph:
    """Bounded singleton/small-batch cache; all other shapes remain eager."""
    def __init__(self):
        self.entries = {}
        self.stats = dict(captured=0, replayed=0, fallback=0)

    def run(self, states, vbar, cfg, *, eager, policy):
        first = states[0]
        # Do not nest capture in D/MTP/model graphs. Limit retained workspaces;
        # large cross-layer and GSM batches keep their existing eager path.
        size = sum(x.numel() * x.element_size() for x in states) + vbar.numel() * vbar.element_size()
        if (not first.is_cuda or torch.cuda.is_current_stream_capturing()
                or size > 4 * 1024 * 1024 or first.shape[0] > 16):
            self.stats['fallback'] += 1
            return eager(states, vbar, cfg)
        shapes = tuple((tuple(x.shape), x.dtype, x.device) for x in (*states, vbar))
        config = (cfg.r, cfg.rmax, cfg.dtype, cfg.init_iters, cfg.init_oversample, cfg.init_method)
        key = (shapes, config, policy, eager, torch.backends.cuda.matmul.allow_tf32,
               torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
               torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= 4:
                self.stats['fallback'] += 1
                return eager(states, vbar, cfg)
            buffers = FactorizeBuffers(states, vbar, cfg)
            current = torch.cuda.current_stream(first.device)
            stream = torch.cuda.Stream(device=first.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                buffers.evaluate(eager)  # compile/initialize the unchanged ops
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = buffers.evaluate(eager)
            entry = (buffers, graph, outputs, stream)
            self.entries[key] = entry
            self.stats['captured'] += 1
            logger.info('GDN prefill factor graph captured: shapes=%s entries=%d',
                        shapes, len(self.entries))
        else:
            entry[0].bind(states, vbar)
        entry[1].replay()
        self.stats['replayed'] += 1
        return FactorizeBuffers.independent(entry[2])
