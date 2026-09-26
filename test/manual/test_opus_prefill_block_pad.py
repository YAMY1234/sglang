"""#ssmoff-opus #779: length-bucketed prefill block graph == eager unpadded chunk call (bytewise).

CPU (TRITON_INTERPRET=1, CUDA_VISIBLE_DEVICES=''): bucket rule and the padded bind kernel (real rows copied from
strided views, padding rows zero / -inf, parameters copied). CUDA (REPLAY_TEST_DEVICE=cuda): for irregular lengths,
the gating + chunk kernels on the real tensors (fresh copy of the initial state) against PaddedBlockGraph.run on the
bucket: output rows < L, the final state and the chunk states of the real chunks, bytewise; several layers share one
graph per bucket. One JSON line; exit 1 on any mismatch.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

GPU = os.environ.get('REPLAY_TEST_DEVICE') == 'cuda'
p = Path(__file__).resolve().parents[2]/'python/sglang/srt/mem_cache/gdn_prefill_block_pad.py'
spec = importlib.util.spec_from_file_location('prefill_block_pad', p)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
DEV = 'cuda' if GPU else 'cpu'
H, HV, D = 8, 24, 128


def same(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def inputs(tokens, gen):
    mixed = (torch.randn(1, tokens, (2 * H + HV) * D, generator=gen) * .05).to(torch.bfloat16).to(DEV)
    return dict(q=mixed[:, :, :H * D].view(1, tokens, H, D), k=mixed[:, :, H * D:2 * H * D].view(1, tokens, H, D),
                v=mixed[:, :, 2 * H * D:].view(1, tokens, HV, D),
                a=torch.randn(tokens, HV, generator=gen).to(torch.bfloat16).to(DEV),
                b=torch.randn(tokens, HV, generator=gen).to(torch.bfloat16).to(DEV),
                log=torch.randn(HV, generator=gen).to(DEV), bias=torch.randn(HV, generator=gen).to(torch.bfloat16).to(DEV),
                state=(torch.randn(1, HV, D, D, generator=gen) * .01).to(DEV),
                rows=torch.tensor([0], dtype=torch.int32, device=DEV))


def cpu_checks():
    assert m.bucket(256) is None and m.bucket(8192) is None and m.bucket(40000) is None
    assert m.bucket(1) == 512 and m.bucket(1000) == 1024 and m.bucket(8191) == 8192
    assert m.bucket(8193) == 12288 and m.bucket(32768) == 32768
    gen = torch.Generator().manual_seed(1)
    t = inputs(40, gen)
    bufs = {n: (torch.full((1, 64) + tuple(x.shape[2:]), 7, dtype=x.dtype, device=DEV) if x.ndim == 4 else
                torch.full((64,) + tuple(x.shape[1:]), 7, dtype=x.dtype, device=DEV)) if m._LAYOUT[n][0]
            else torch.empty_like(x) for n, x in t.items()}
    m.bind_padded(bufs, t, 40)
    ok = all(same(bufs[n][:, :40] if bufs[n].ndim == 4 else bufs[n][:40], t[n]) for n in ('q', 'k', 'v', 'a', 'b'))
    ok &= all(float(bufs[n][:, 40:].abs().max()) == 0 for n in ('q', 'k', 'v'))
    ok &= all(bool(torch.isneginf(bufs[n][40:].float()).all()) for n in ('a', 'b'))
    ok &= all(same(bufs[n], t[n]) for n in ('log', 'bias', 'state', 'rows'))
    return dict(bind=bool(ok))


def gpu_checks():
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating

    def evaluate(t):
        g, beta = fused_gdn_gating(t['log'], t['a'], t['b'], t['bias'])
        return chunk_gated_delta_rule(q=t['q'], k=t['k'], v=t['v'], g=g, beta=beta, initial_state=t['state'],
                                      initial_state_indices=t['rows'], cu_seqlens=t['cu'], head_first=False,
                                      use_qk_l2norm_in_kernel=True, inplace_update=True)
    graph = m.PaddedBlockGraph()
    gen = torch.Generator().manual_seed(779)
    cases = []
    for tokens in (1000, 1024, 1500, 2047, 3000, 4095, 5000, 7777, 9000, 12288, 20000, 31000):
        for layer in range(2):
            t = inputs(tokens, gen)
            ref_t = {k: v.clone() for k, v in t.items()}
            ref_t['cu'] = torch.tensor([0, tokens], dtype=torch.int32, device=DEV)
            out, last, h = evaluate(ref_t)
            final = ref_t['state'] if last is None else last
            got, state, h_got = graph.run(t, m.bucket(tokens), evaluate)
            cases.append(dict(tokens=tokens, layer=layer, padded=m.bucket(tokens), output=same(got, out),
                              state=same(state, final), checkpoint=same(h_got[:, :h.shape[1]], h)))
    torch.cuda.synchronize()
    return dict(cases=cases, stats=graph.stats,
                bitwise=all(c['output'] and c['state'] and c['checkpoint'] for c in cases))


@torch.inference_mode()
def main():
    res = dict(device=DEV, cpu=cpu_checks())
    ok = res['cpu']['bind']
    if GPU:
        res['gpu'] = gpu_checks()
        ok = ok and res['gpu']['bitwise']
    res['passed'] = bool(ok)
    print(json.dumps(res))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
