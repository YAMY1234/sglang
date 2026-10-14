"""Opt-in stage attribution; no model math changes or inter-stage synchronization."""
import contextlib
import gzip
import json
import time
import sys
from collections import defaultdict

import torch


def run(original, runner, fb, args, kwargs, config, root):
    trace = config['mode'] == 'i4-trace'
    saved, samples, stack = [], [], []
    runtime = getattr(runner.model, 'lightning_runtime', None)
    body = runtime.body if runtime is not None else runner.model.model

    @contextlib.contextmanager
    def region(name):
        start, end = (None, None) if trace else (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        sample = dict(stage=name, children=[])
        if stack:
            stack[-1]['children'].append(sample)
        stack.append(sample)
        annotation = torch.profiler.record_function('I4_STAGE/' + name) if trace else contextlib.nullcontext()
        with annotation:
            if start is not None:
                start.record()
            t = time.perf_counter()
            try:
                yield
            finally:
                sample['host_ms'] = (time.perf_counter()-t)*1000
                if end is not None:
                    end.record()
                sample['events'] = (start, end)
                samples.append(sample)
                stack.pop()

    def patch(obj, key, stage, conditional=None):
        old = getattr(obj, key)
        saved.append((obj, key, old))
        def wrapped(*a, **kw):
            if conditional is not None and not conditional():
                return old(*a, **kw)
            with region(stage):
                return old(*a, **kw)
        setattr(obj, key, wrapped)

    # Native layers only: tail is separately bounded and must not be called shallow.
    for layer in body.layers:
        patch(layer, 'forward', 'layer_trim' if runtime is not None else 'native_layers',
              (lambda: not any(s['stage'] == 'tail' for s in stack)) if runtime is not None else None)
    if runtime is not None:
        engine = sys.modules[type(runtime).__module__]
        import sglang.srt.layers.attention.linear.mamba2_emitter_extend as emitter
        patch(engine, 'reconstruct_boundary', 'boundary')
        patch(emitter, 'emitter_extend', 'emitter')
        patch(runtime, '_decode_forward', 'tail')
        # Include all flushes, also those nested in layer/emitter collector calls.
        patch(runtime.pool, 'begin_extend', 'factorize_commit')
        patch(runtime.pool, 'commit_extend', 'factorize_commit')
        patch(runtime.pool, 'finish_extend', 'factorize_commit')
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA], record_shapes=False, with_stack=False,
        profile_memory=False) if trace else contextlib.nullcontext()
    row = dict(tag=config['tag'], batch_size=fb.batch_size, input_tokens=fb.input_ids.numel(),
        lengths=fb.extend_seq_lens_cpu, layer_count=len(body.layers), mode=config['mode'])
    try:
        torch.cuda.synchronize()
        with profiler as prof:
            t = time.perf_counter()
            with region('total'):
                result = original(runner, fb, *args, **kwargs)
            submitted = time.perf_counter()
            torch.cuda.synchronize()
            row.update(completed_wall_ms=(time.perf_counter()-t)*1000,
                       host_submit_ms=(submitted-t)*1000)
        if trace:
            path = root / (config['tag'] + '.json')
            prof.export_chrome_trace(str(path))
            with path.open('rb') as src, gzip.open(str(path)+'.gz', 'wb', compresslevel=2) as out:
                out.write(src.read())
            path.unlink()
        else:
            totals = defaultdict(lambda: dict(calls=0, host_exclusive_ms=0., cuda_span_exclusive_ms=0.))
            for s in samples:  # children precede parents
                s['cuda_ms'] = s['events'][0].elapsed_time(s['events'][1])
                target = totals[s['stage']]
                target['calls'] += 1
                target['host_exclusive_ms'] += s['host_ms']-sum(c['host_ms'] for c in s['children'])
                target['cuda_span_exclusive_ms'] += s['cuda_ms']-sum(c['cuda_ms'] for c in s['children'])
            row['stages'] = dict(totals)
            row['device_span_ms'] = samples[-1]['cuda_ms']
        if runtime is not None:
            row['handoff'] = dict(runtime.last_handoff)
        with (root/'i4-stages.jsonl').open('a') as out:
            out.write(json.dumps(row)+'\n')
        return result
    finally:
        for obj, key, old in reversed(saved):
            setattr(obj, key, old)
