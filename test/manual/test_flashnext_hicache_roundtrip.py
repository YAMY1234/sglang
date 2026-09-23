"""Audit production HiCache Mamba transport, including registered prefix siblings.

Reports partial transport as FAIL. A successful ordinary CPU-copy control does
not qualify the optimized host path or the QSA token pool. No model weights.
"""
import json
from types import SimpleNamespace as NS

import torch

from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNPool, FactoredGDNConfig
from sglang.srt.mem_cache.ple_state_pool import ShortConvPool, NGramPool
from sglang.srt.mem_cache.flashnext_latent_pool import LatentRequestState
from sglang.srt.model_executor.fullstack_policy import FULLSTACK_R8_RADIX_STATE


def equal(a, b):
    if isinstance(a, torch.Tensor): return torch.equal(a, b)
    if a is None or b is None: return a is b
    return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))


def payload(sibling, slots):
    result = sibling.get_cpu_slots(slots)
    torch.cuda.synchronize()
    return result


def run(arm, envelope):
    layers = [0, 1]
    cp = NS(shape=NS(conv=[(1024, 3)], temporal=(2, 128, 128)),
            dtype=NS(conv=torch.bfloat16, temporal=torch.float32), is_kda=False)
    pool = MambaPool(size=8, spec_state_size=0, cache_params=cp, mamba_layer_ids=layers,
                     device='cuda', empty_temporal=arm != 'stock', envelope_layout=envelope)
    for t in [*pool.mamba_cache.conv, pool.mamba_cache.temporal]:
        if t.numel(): t.normal_()
    short = ShortConvPool(size=8, state_shape=(3, 128), layer_ids=layers, dtype=torch.bfloat16, device='cuda')
    gram = NGramPool(size=8, context_len=4, eos_token_id=0, device='cuda')
    short.conv_state.normal_(); gram.context.random_(1, 100)
    named = [('ple_short_conv', short), ('ple_ngram', gram)]
    if arm != 'stock':
        factor = FactoredGDNPool(size=8, cache_params=cp, mamba_layer_ids=layers, device='cuda',
                                 cfg=FactoredGDNConfig.parse(FULLSTACK_R8_RADIX_STATE))
        for t in (factor.a, factor.U, factor.W): t.normal_(0, .01)
        factor.count.fill_(11); factor.prefix_factored_valid.fill_(1)
        named.append(('factor_a_U_W_count_prefix_valid', factor))
    if arm == 'final':
        latent = LatentRequestState(8, 'cuda')
        latent.sink.normal_(); latent.sink_valid.fill_(1)
        latent.boundary.normal_(); latent.boundary_position.fill_(100)
        named.append(('latent_sink_valid_and_boundary_reset', latent))
    for _, sibling in named: pool.register_slot_state(sibling)
    source = torch.tensor([2, 3], device='cuda'); target = torch.tensor([5, 6], device='cuda')
    host_ids = torch.tensor([1, 4], device='cuda')
    expected = {name: payload(sibling, source) for name, sibling in named}
    expected_conv = [t[:,source].clone() for t in pool.mamba_cache.conv]
    expected_temporal = pool.mamba_cache.temporal[:,source].clone()
    torch.cuda.synchronize()
    # Existing sibling serializers are the prefix-state oracle, not the HiCache path.
    control = {}
    for name, sibling in named:
        sibling.load_cpu_slots(expected[name], target)
        control[name] = equal(expected[name], payload(sibling, target))
        sibling.reset_slots(target)
    host = MambaPoolHost(pool, host_to_device_ratio=2, host_size=0, layout='page_first')
    host.backup_from_device_all_layer(pool, host_ids, source, io_backend='kernel')
    torch.cuda.synchronize()
    for t in [*pool.mamba_cache.conv, pool.mamba_cache.temporal]: t[:,target] = 0
    for layer in layers:
        host.load_to_device_per_layer(pool, host_ids, target, layer, io_backend='kernel')
    torch.cuda.synchronize()
    actual = {name: equal(expected[name], payload(sibling, target)) for name, sibling in named}
    dense = equal(expected_temporal, pool.mamba_cache.temporal[:,target])
    conv = all(equal(a, b[:,target]) for a,b in zip(expected_conv, pool.mamba_cache.conv))
    if arm == 'final':
        actual['D_boundary_not_restored'] = bool((latent.boundary[target] == 0).all() and (latent.boundary_position[target] == -1).all())
    result = dict(arm=arm, envelope=envelope, serializer_control=control,
                conv_exact=conv, temporal_exact=dense, siblings=actual,
                passed=all(control.values()) and dense and conv and all(actual.values()),
                device_conv_stride=list(pool.mamba_cache.conv[0].stride()),
                host_size_per_slot=host.size_per_token,
                scope='Mamba host path only; QSA token/index/latent pages require separate guard')
    host.destroy()
    return result


if __name__ == '__main__':
    torch.manual_seed(430)
    cases = []
    for arm,envelope in [('stock',False), ('A',False), ('final',True)]:
        case = run(arm, envelope)
        cases.append(case)
        print(json.dumps(dict(progress=case)), flush=True)
    result = dict(passed=all(c['passed'] for c in cases), gpu=torch.cuda.get_device_name(), cases=cases)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['passed'] else 1)
