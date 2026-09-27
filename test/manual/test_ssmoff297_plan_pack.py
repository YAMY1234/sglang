"""Plan dtype materialization against torch and the real pool transaction planner."""
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

import torch

from sglang.srt.mem_cache.gdn_prefill_plan_pack import materialize

GPU = os.environ.get('REPLAY_TEST_DEVICE') == 'cuda'
DEVICE = 'cuda' if GPU else 'cpu'
if not GPU:
    assert os.environ.get('TRITON_INTERPRET') == '1'


def inputs(batch, required, prefix, rows):
    # Include signed values, padding and int64 -> int32 wrap boundaries.
    values = [0, -1, 1, 2, -(2**31)-1, 2**31, 2**32+1, -(2**32)+1]
    offsets, flat = {}, []
    for name, count in (('use_ring', batch), ('ring_src', batch),
                        ('ring_dst', batch), ('rows', rows),
                        ('required', batch if required else 0),
                        ('use_prefix', batch if prefix else 0)):
        begin = len(flat)
        flat.extend(values[(i+begin) % len(values)] for i in range(count))
        offsets[name] = (begin, len(flat))
    return torch.tensor(flat, dtype=torch.int64, device=DEVICE), offsets


def reference(packed, offsets, batch):
    def part(name):
        begin, end = offsets[name]
        return packed[begin:end]
    return dict(use_ring=part('use_ring').bool(), ring_dst_i32=part('ring_dst').int(),
                required=part('required').int() if offsets['required'][1] > offsets['required'][0] else None,
                use_prefix=part('use_prefix').bool() if offsets['use_prefix'][1] > offsets['use_prefix'][0] else None)


def same(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if value is None:
            assert actual[key] is None, key
        else:
            assert actual[key].dtype == value.dtype, key
            assert torch.equal(actual[key].view(torch.uint8), value.view(torch.uint8)), key


def main():
    cases = []
    for batch in (1, 2, 3, 16, 32, 64, 96):
        for required, prefix in ((False, False), (True, False), (False, True), (True, True)):
            for rows in (0, batch):
                packed, offsets = inputs(batch, required, prefix, rows)
                before = packed.clone()
                actual = materialize(packed, offsets, batch)
                same(actual, reference(packed, offsets, batch))
                # A later plan must not overwrite tensors retained by the earlier plan.
                materialize(packed + 1, offsets, batch)
                same(actual, reference(before, offsets, batch))
                assert torch.equal(packed, before)
                cases.append(dict(batch=batch, required=required, prefix=prefix, rows=rows, bitwise=True))
    path = Path(__file__).with_name('test_ssmoff297_prefill_host.py')
    spec = importlib.util.spec_from_file_location('host_reference', path)
    host = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(host)
    host.DEV = DEVICE
    os.environ['SGLANG_GDN_PREFILL_PLAN_GATHER'] = '1'
    checks = 0
    for tokens, prefix in ((256, 64), (32768, 0), (2285, 32768), (6687, 51392)):
        for seed in range(6):
            os.environ['SGLANG_GDN_PREFILL_PLAN_PACK'] = '0'
            old = host.run(True, seed, tokens, prefix)
            os.environ['SGLANG_GDN_PREFILL_PLAN_PACK'] = '1'
            new = host.run(True, seed, tokens, prefix)
            assert host.same(old, new), (tokens, prefix, seed)
            checks += len(old)
    timing = None
    if GPU:
        packed, offsets = inputs(1, True, True, 1)
        funcs = {'reference': lambda: reference(packed, offsets, 1),
                 'candidate': lambda: materialize(packed, offsets, 1)}
        def bench(fn):
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(500):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - start) * 1e6 / 500
        measurements = {name: [] for name in funcs}
        for pair in range(6):
            for name in (('reference', 'candidate') if pair % 2 == 0 else ('candidate', 'reference')):
                measurements[name].append(bench(funcs[name]))
        timing = dict(pairs_us=measurements,
                      median_us={name: statistics.median(rows) for name, rows in measurements.items()},
                      scope='B1 four dtype conversions versus one integer kernel, allocations included; excludes H2D, D2H and other planning')
    print(json.dumps(dict(complete=True, passed=True, device=DEVICE, cases=cases,
                         pool_checks=checks, timing=timing)))


if __name__ == '__main__':
    main()
