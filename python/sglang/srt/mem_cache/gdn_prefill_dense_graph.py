"""Default-off replay of the unchanged singleton GDN chunk recurrence.

Live pool slots never enter the graph. Owned S0 is reset before every replay;
the caller still commits the prefix and subsequently processes its last token.
"""
from collections import OrderedDict
import logging

import torch

logger = logging.getLogger(__name__)


class DenseBuffers:
    names = ('q', 'k', 'v', 'g', 'beta', 'ssm_states', 'cache_indices', 'query_start_loc')

    def __init__(self, arguments):
        self.arguments = dict(arguments)
        for name in self.names:
            self.arguments[name] = arguments[name].clone()

    def bind(self, arguments):
        for name in self.names:
            self.arguments[name].copy_(arguments[name])

    def evaluate(self, eager):
        return eager(**self.arguments)

    def publish(self, arguments, outputs):
        arguments['ssm_states'].copy_(self.arguments['ssm_states'])
        return tuple(None if x is None else x.clone() for x in outputs)


class PrefillDenseGraph:
    def __init__(self):
        self.entries = OrderedDict()
        self.stats = dict(captured=0, replayed=0, fallback=0)

    def run(self, *, eager, **arguments):
        state, query = arguments['ssm_states'], arguments['q']
        if (not state.is_cuda or state.shape[0] != 1 or query.shape[0] != 1
                or query.shape[1] > 8192
                or arguments.get('output') is not None
                or arguments['query_start_loc'].numel() != 2
                or torch.cuda.is_current_stream_capturing()):
            self.stats['fallback'] += 1
            return eager(**arguments)
        # With exactly one sequence, cu_seqlens is [0, T]. This immutable
        # shape identity also makes FLA's cached chunk-index plan valid.
        shapes = tuple((name, tuple(arguments[name].shape), arguments[name].dtype,
                        tuple(arguments[name].stride())) for name in DenseBuffers.names)
        metadata = tuple((name, value) for name, value in arguments.items()
                         if name not in DenseBuffers.names and value is not None)
        # Triton extend ignores checkpoint kwargs and output. Only capture
        # its original signature; unknown tensor metadata stays eager.
        if any(isinstance(value, torch.Tensor) for _, value in metadata):
            self.stats['fallback'] += 1
            return eager(**arguments)
        key = (shapes, metadata, state.device, eager, torch.backends.cuda.matmul.allow_tf32)
        entry = self.entries.pop(key, None)
        if entry is None:
            if len(self.entries) >= 2:
                _, old = self.entries.popitem(last=False)
                # Capture allocations belong to a side stream; a previous
                # replay/publication may still be running on the caller stream.
                current = torch.cuda.current_stream(state.device)
                for value in (*old[0].arguments.values(), *old[2]):
                    if isinstance(value, torch.Tensor):
                        value.record_stream(current)
            buffers = DenseBuffers(arguments)
            current = torch.cuda.current_stream(state.device)
            stream = torch.cuda.Stream(device=state.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                buffers.evaluate(eager)
            current.wait_stream(stream)
            buffers.bind(arguments)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = buffers.evaluate(eager)
            entry = (buffers, graph, outputs, stream)
            self.stats['captured'] += 1
            logger.info('GDN prefill dense graph captured: T=%d', query.shape[1])
        # Warmup and capture mutate owned S0. Reset it even on the first call.
        entry[0].bind(arguments)
        entry[1].replay()
        self.entries[key] = entry
        self.stats['replayed'] += 1
        return entry[0].publish(arguments, entry[2])
