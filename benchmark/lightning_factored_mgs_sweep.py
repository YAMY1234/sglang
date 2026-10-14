"""Isolated 3/2/1 MGS diagnostic. Never changes the service configuration."""
import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('m1', source / 'test/manual/duet/test_mamba2_factored.py')
    m1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m1)
    kernel = m1.kernel
    original_iters = kernel.TRUNC_ITERS
    result = dict(scope='diagnostic only; production remains 3 iterations',
                  reference='FP64 recurrence + optimal rank16 content SVD every W16; sink retained',
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, rows=[])
    start = time.monotonic()
    try:
        for iterations in (3, 2, 1):
            kernel.TRUNC_ITERS = iterations
            for batch in (1, 8, 32):
                cases = []
                for native in (False, True):
                    numerical = m1.case(batch, 'cuda', native=native)
                    # The old same-MGS oracle is fixed at 3 rounds; deliberately
                    # do not report its errors for this sweep. Optimal SVD is independent.
                    cases.append(dict(native_dtype=native,
                        max_output_rel_f=max(x['best_rank16_rel_f'] for x in numerical['rows']),
                        max_fused_output_rel_f=max(x['best_rank16_fused_rel_f'] for x in numerical['rows']),
                        rows=[{k:v for k,v in x.items() if k == 'step' or k.startswith('best_rank16_')} for x in numerical['rows']],
                        cuts=[{k:x[k] for k in ('step', 'state_vs_best_rel_f', 'excess_residual')} for x in numerical['cuts']],
                        graph_replay=numerical['graph_replay_64_changing_inputs_slots']))
                torch.manual_seed(20261014)
                layers, heads = 23, 64
                q = torch.linalg.qr(torch.randn(64, 32, device='cuda')).Q.T
                u = q.expand(layers, batch, heads, 32, 64).contiguous()
                v = torch.randn(layers, batch, heads, 32, 128, device='cuda')
                saved_u, saved_v = u.clone(), v.clone()
                count = torch.full((layers, batch, heads), 32, device='cuda', dtype=torch.int32)
                slots = torch.arange(batch, device='cuda', dtype=torch.int32)
                kernel.truncate_layers(u, v, count, slots)
                torch.cuda.synchronize()
                # Capture only expiry; input restore/count reset is outside the events.
                graph = torch.cuda.CUDAGraph()
                count.fill_(32)
                with torch.cuda.graph(graph):
                    kernel.truncate_layers(u, v, count, slots)
                samples = []
                for _ in range(12):
                    u.copy_(saved_u); v.copy_(saved_v); count.fill_(32)
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record(); graph.replay(); end.record(); end.synchronize()
                    assert bool((count == 16).all())
                    samples.append(begin.elapsed_time(end))
                row = dict(iterations=iterations, B=batch, numerical=cases,
                           expiry_ms=statistics.median(samples[2:]), samples_ms=samples,
                           layers=layers, heads=heads, timing='all layers due, graph replay; resets excluded')
                result['rows'].append(row)
                result['seconds'] = time.monotonic() - start
                (args.out / 'mgs-sweep.json').write_text(json.dumps(result, indent=2) + '\n')
                print('MGS_SWEEP', iterations, batch, row['expiry_ms'], [r['max_output_rel_f'] for r in cases], flush=True)
                del u, v, saved_u, saved_v, count, graph
    finally:
        kernel.TRUNC_ITERS = original_iters
    result['complete'] = True
    (args.out / 'mgs-sweep.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
