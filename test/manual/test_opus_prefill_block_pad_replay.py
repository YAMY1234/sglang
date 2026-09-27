"""#ssmoff-opus: one-time cost of the first replay of a primed padded block graph (893438: first-use turns +130-250 ms
in model_forward after all buckets were captured at startup).

Prime every bucket from one short prefill's tensors, then per bucket time (host wall, synchronized) the first and the
second replay with a real-length input, plus the same after a bucket's first replay was already done. One JSON line.
"""
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
__import__('os').environ['REPLAY_TEST_DEVICE'] = 'cuda'
import test_opus_prefill_block_pad as t  # noqa: E402
m = t.m


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        print(json.dumps(dict(device='cpu', imports=True, passed=True)))
        return
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating

    def evaluate_padded(x):
        g, beta = fused_gdn_gating(x['log'], x['a'], x['b'], x['bias'])
        return m.chunk_padded(x['q'], x['k'], x['v'], g, beta, x['state'], x['rows'], x['cu'], x['real_end'])

    graph = m.PaddedBlockGraph()
    gen = torch.Generator().manual_seed(1)
    torch.cuda.synchronize()
    started = time.perf_counter()
    graph.prime(t.inputs(7, gen), evaluate_padded)
    torch.cuda.synchronize()
    prime_s = time.perf_counter() - started
    rows = []
    for padded in m.BUCKETS:
        tokens = max(1, padded - 37) if padded > 32 else padded - 1
        x = t.inputs(tokens, gen)
        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            s = time.perf_counter()
            graph.run(x, padded, evaluate_padded)
            torch.cuda.synchronize()
            times.append(round((time.perf_counter() - s) * 1e3, 3))
        rows.append(dict(bucket=padded, tokens=tokens, first_ms=times[0], second_ms=times[1], third_ms=times[2]))
    print(json.dumps(dict(prime_seconds=round(prime_s, 3), rows=rows, stats=graph.stats, passed=True)))


if __name__ == '__main__':
    main()
