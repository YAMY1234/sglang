"""#ssmoff-opus: which stage makes the padded prefill differ from the eager unpadded call (892561: rare 1-ulp output).

No graphs: the same inputs through the eager pipeline stages at T = L (cu [0, L]) and padded to Lb (zero q/k/v,
-inf gating inputs) with the real-end state kernel and with the original state kernel. Per stage (g cumsum, w, u,
v_new, final state, output) the number of trials whose real rows differ bytewise, over many seeds. One JSON line.
"""
import json
import os

import torch

H, HV, D = 8, 24, 128


def same(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@torch.inference_mode()
def main():
    from sglang.kernels.ops.attention.fla.chunk import CHUNK_SIZE, chunk_gated_delta_rule_fwd_h
    from sglang.kernels.ops.attention.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from sglang.kernels.ops.attention.fla.chunk_o import chunk_fwd_o
    from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
    from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
    from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
    if not torch.cuda.is_available():
        print(json.dumps(dict(device='cpu', imports=True, passed=True)))
        return
    dev = 'cuda'

    def stages(q, k, v, a, b, log, bias, state, T, real_end=None):
        cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
        ci = prepare_chunk_indices(cu, CHUNK_SIZE)
        g, beta = fused_gdn_gating(log, a, b, bias)
        q, k = l2norm_fwd(q), l2norm_fwd(k)
        g = chunk_local_cumsum(g, chunk_size=CHUNK_SIZE, cu_seqlens=cu, chunk_indices=ci)
        w, u, A = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g, beta=beta, cu_seqlens=cu, chunk_indices=ci)
        s = state.clone()
        rows = torch.tensor([0], dtype=torch.int32, device=dev)
        kw = {} if real_end is None else dict(real_end=torch.tensor([real_end], dtype=torch.int32, device=dev))
        h, v_new = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=g, initial_state=s, initial_state_indices=rows,
                                                cu_seqlens=cu, chunk_indices=ci, inplace_update=True, **kw)
        o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=D ** -0.5, cu_seqlens=cu)
        return dict(g=g, w=w, u=u, v_new=v_new, state=s, o=o.to(q.dtype))

    trials = int(os.environ.get('STAGE_TRIALS', '200'))
    counts = {}
    # chunk_fwd_o tiles by BT = min(64, max(16, next_power_of_2(T))): the eager call at L <= 32 uses BT 16/32, a pad
    # to 512 uses 64; 'small' pads to 16 (L <= 16) or 32 (L <= 32), where the eager BT is the same
    variants = lambda L: (('real_end', 512, L), ('original', 512, None),
                          ('small', 16 if L <= 16 else (32 if L <= 32 else 512), L))
    for L in (1, 7, 16, 17, 31, 32, 33, 63, 100):
        c = {f'{variant}:{name}': 0 for variant, _, _ in variants(L) for name in ('g', 'w', 'u', 'v_new', 'state', 'o')}
        for seed in range(trials):
            gen = torch.Generator().manual_seed(seed)
            mixed = (torch.randn(1, L, (2 * H + HV) * D, generator=gen) * .05).to(torch.bfloat16).to(dev)
            q, k, v = (mixed[:, :, :H * D].reshape(1, L, H, D).contiguous(), mixed[:, :, H * D:2 * H * D].reshape(1, L, H, D).contiguous(),
                       mixed[:, :, 2 * H * D:].reshape(1, L, HV, D).contiguous())
            a = torch.randn(L, HV, generator=gen).to(torch.bfloat16).to(dev)
            b = torch.randn(L, HV, generator=gen).to(torch.bfloat16).to(dev)
            log = torch.randn(HV, generator=gen).to(dev)
            bias = torch.randn(HV, generator=gen).to(torch.bfloat16).to(dev)
            state = (torch.randn(1, HV, D, D, generator=gen) * .01).to(dev)

            def pad(x, value, Lb):
                out = torch.full((x.shape[0], Lb) + tuple(x.shape[2:]) if x.ndim == 4 else (Lb,) + tuple(x.shape[1:]),
                                 value, dtype=x.dtype, device=dev)
                (out[:, :L] if x.ndim == 4 else out[:L]).copy_(x)
                return out
            ref = stages(q, k, v, a, b, log, bias, state, L)
            for variant, Lb, re in variants(L):
                got = stages(pad(q, 0., Lb), pad(k, 0., Lb), pad(v, 0., Lb), pad(a, float('-inf'), Lb),
                             pad(b, float('-inf'), Lb), log, bias, state, Lb, re)
                for name in ('g', 'w', 'u', 'v_new', 'o'):
                    x, y = ref[name], got[name]
                    y = y[:, :L] if y.ndim >= 3 else y[:L]
                    c[f'{variant}:{name}'] += not same(x, y)
                c[f'{variant}:state'] += not same(ref['state'], got['state'])
        counts[f'L{L}'] = c
    torch.cuda.synchronize()
    print(json.dumps(dict(trials=trials, differing_trials=counts, passed=True)))


if __name__ == '__main__':
    main()
