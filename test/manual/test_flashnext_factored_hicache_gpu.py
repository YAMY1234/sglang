"""Native-kernel factored GDN HiCache round trip (TwinStar docs/139); no model.

Production head shapes (24 heads, K = V = 128, RMAX 16, fp16 factors), PLE
siblings; a small pool and the served pool (36 GDN layers, 480 slots). Bitwise: a, U, W, count, the P-checkpoint flag, conv and
PLE after backup -> zeroing -> restore into different slots; authority metadata
reset; untouched slots unchanged; the PD transfer manifest unchanged.
"""
import json
import os
from types import SimpleNamespace as NS

import torch

os.environ["SGLANG_FLASHNEXT_STOCK_HICACHE"] = "1"
os.environ["SGLANG_FLASHNEXT_FACTOR_HICACHE"] = "1"

from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNConfig, FactoredGDNPool
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
from sglang.srt.mem_cache.pool_host.flashnext_factored import (
    FlashNextFactoredMambaHost, payload_tensors,
)

def exact(a, b):
    return (a.dtype == b.dtype and a.shape == b.shape
            and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def case(layers, size, src, dst, hi):
    LAYERS = list(range(layers))
    cp = NS(shape=NS(conv=[(3072, 3)], temporal=(24, 128, 128),
                     disable_conv_window_dedup=False, conv_kernel=4),
            dtype=NS(conv=torch.bfloat16, temporal=torch.bfloat16), is_kda=False, layers=LAYERS)
    pool = MambaPool(size=size, spec_state_size=0, cache_params=cp, mamba_layer_ids=LAYERS,
                     device="cuda", empty_temporal=True)
    factor = FactoredGDNPool(size=size, cache_params=cp, mamba_layer_ids=LAYERS, device="cuda",
                             cfg=FactoredGDNConfig.parse(
                                 "r=8,m=8,dtype=fp16,ring=2,init_iters=2,async=1,"
                                 "strict_chunk=1,factored_prefix=1"))
    short = ShortConvPool(size=size, state_shape=(3, 128), layer_ids=LAYERS,
                          dtype=torch.bfloat16, device="cuda")
    gram = NGramPool(size=size, context_len=4, eos_token_id=0, device="cuda")
    for sibling in (factor, short, gram):
        pool.register_slot_state(sibling)
    tensors = dict(a=factor.a, U=factor.U, W=factor.W, count=factor.count,
                   conv=pool.mamba_cache.conv[0], short=short.conv_state,
                   ngram=gram.context.unsqueeze(0))
    for t in tensors.values():
        if t.is_floating_point():
            t.normal_()
        else:
            t.random_(1, 1000)
    factor.prefix_valid.random_(0, 2)
    manifest = [(n, t.data_ptr(), tuple(t.shape), a, l)
                for n, t, a, l in pool._iter_transfer_state_entries()]
    src = torch.tensor(src, device="cuda")
    dst = torch.tensor(dst, device="cuda")
    hi = torch.tensor(hi, dtype=torch.int64)  # write_back keeps host indices on CPU
    keep = sorted(set(range(1, size + 1)) - set(src.tolist()) - set(dst.tolist()))[:1]
    oracle = {n: t[:, src].clone() for n, t in tensors.items()}
    flag = factor.prefix_valid[src].clone()
    other = {n: t[:, keep].clone() for n, t in tensors.items()}
    host = FlashNextFactoredMambaHost(pool, host_to_device_ratio=2, host_size=0,
                                     layout="page_first")
    try:
        host.backup_from_device_all_layer(pool, hi, src, "kernel")
        torch.cuda.synchronize()
        for t in tensors.values():
            t[:, src] = 0
            t[:, dst] = 0
        factor.prefix_valid[dst] = 7
        factor.stale[dst] = 0
        factor.dense_of[dst] = 1
        factor.dense_required[dst] = 1
        load_hi = hi.to("cuda")
        for layer in range(len(LAYERS)):
            host.load_to_device_per_layer(pool, load_hi, dst, layer, "kernel")
        torch.cuda.synchronize()
        checks = {n: exact(oracle[n], t[:, dst]) for n, t in tensors.items()}
        checks["prefix_flag"] = exact(flag, factor.prefix_valid[dst])
        checks["authority_reset"] = bool((factor.stale[dst] == 1).all()
                                         and (factor.dense_of[dst] == -1).all()
                                         and (factor.dense_required[dst] == 0).all())
        checks["other_slots_unchanged"] = all(exact(other[n], t[:, keep]) for n, t in tensors.items())
        checks["pd_manifest_unchanged"] = manifest == [
            (n, t.data_ptr(), tuple(t.shape), a, l) for n, t, a, l in pool._iter_transfer_state_entries()]
        allocated = sum(b.numel() * b.element_size() for b in host.kv_buffer)
        checks["budget_covers_payload"] = allocated == host.size * host.size_per_token
        return dict(layers=layers, size=size, src=src.tolist(), dst=dst.tolist(), host=hi.tolist(), checks=checks,
                    host_bytes_per_slot=host.size_per_token,
                    factor_bytes_per_layer_slot=sum(t[0, 0].numel() * t.element_size()
                                                    for t in payload_tensors(factor)),
                    passed=all(checks.values()))
    finally:
        host.destroy()


if __name__ == "__main__":
    torch.manual_seed(974)
    cases = [case(3, 8, [3, 1], [7, 5], [2, 6]),
             case(36, 480, [479, 3, 250], [1, 480, 17], [900, 0, 451])]
    for c in cases:
        print(json.dumps(dict(progress=c)), flush=True)
    passed = all(c["passed"] for c in cases)
    print(json.dumps(dict(passed=passed, cases=cases, gpu=torch.cuda.get_device_name(),
                          scope="factored GDN + conv + PLE host transport; model gates separate")),
          flush=True)
    raise SystemExit(0 if passed else 1)
