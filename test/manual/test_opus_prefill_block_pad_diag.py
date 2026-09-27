"""#ssmoff-opus: diagnostic for the padded block graph at short lengths (892422: tokens=1 layer 1 output mismatch).

Same inputs, three evaluations per case: the eager unpadded call twice (eager determinism) and the padded graph from
a graph that was primed (all buckets captured from other tensors first) or captured at first use. Per case: number of
output elements whose bytes differ and their max abs difference (graph vs eager, eager vs eager), and state /
checkpoint bytewise. One JSON line.
"""
import importlib.util
import json
from pathlib import Path
import sys

import torch

p = Path(__file__).resolve().parents[2]/'python/sglang/srt/mem_cache/gdn_prefill_block_pad.py'
spec = importlib.util.spec_from_file_location('prefill_block_pad', p)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
sys.path.insert(0, str(Path(__file__).resolve().parent))
os_env = __import__('os').environ
os_env['REPLAY_TEST_DEVICE'] = 'cuda'
import test_opus_prefill_block_pad as t  # noqa: E402  (inputs(), same())


def diff(a, b):
    x = a.contiguous().view(torch.uint8).view(-1, a.element_size())
    y = b.contiguous().view(torch.uint8).view(-1, b.element_size())
    bad = (x != y).any(dim=1)
    n = int(bad.sum())
    return n, (float((a.float() - b.float()).abs().max()) if n else 0.0)


@torch.inference_mode()
def main():
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
    if not torch.cuda.is_available():  # same-image CPU pass: imports and the bucket rule only
        assert m.bucket(1) == 512 and len(m.BUCKETS) == 16
        print(json.dumps(dict(device='cpu', imports=True, passed=True)))
        return

    def evaluate(x):
        g, beta = fused_gdn_gating(x['log'], x['a'], x['b'], x['bias'])
        return chunk_gated_delta_rule(q=x['q'], k=x['k'], v=x['v'], g=g, beta=beta, initial_state=x['state'],
                                      initial_state_indices=x['rows'], cu_seqlens=x['cu'], head_first=False,
                                      use_qk_l2norm_in_kernel=True, inplace_update=True)

    def evaluate_padded(x):
        g, beta = fused_gdn_gating(x['log'], x['a'], x['b'], x['bias'])
        return m.chunk_padded(x['q'], x['k'], x['v'], g, beta, x['state'], x['rows'], x['cu'], x['real_end'])

    rows = []
    for mode in ('first_use', 'primed', 'first_use'):
        graph = m.PaddedBlockGraph()
        gen = torch.Generator().manual_seed(779)
        if mode == 'primed':
            graph.prime(t.inputs(7, torch.Generator().manual_seed(5)), evaluate_padded)
        for tokens in (1, 1, 2, 3, 7, 63, 1000):
            for layer in range(4):
                x = t.inputs(tokens, gen)
                refs = []
                for _ in range(2):
                    r = {k: v.clone() for k, v in x.items()}
                    r['cu'] = torch.tensor([0, tokens], dtype=torch.int32, device='cuda')
                    out, last, h = evaluate(r)
                    refs.append((out, r['state'] if last is None else last, h))
                got, state, h_got = graph.run({k: v.clone() for k, v in x.items()}, m.bucket(tokens), evaluate_padded)
                ne, me = diff(refs[0][0], refs[1][0])
                ng, mg = diff(got, refs[0][0])
                rows.append(dict(mode=mode, tokens=tokens, layer=layer, eager_vs_eager=[ne, me], graph_vs_eager=[ng, mg],
                                 state=t.same(state, refs[0][1]), checkpoint=t.same(h_got[:, :refs[0][2].shape[1]], refs[0][2]),
                                 elements=refs[0][0].numel()))
    torch.cuda.synchronize()
    bad = [r for r in rows if r['graph_vs_eager'][0] or r['eager_vs_eager'][0] or not r['state'] or not r['checkpoint']]
    print(json.dumps(dict(cases=len(rows), mismatches=bad, passed=not bad)))


if __name__ == '__main__':
    main()
