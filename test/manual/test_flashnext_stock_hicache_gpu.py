"""Production-kernel stock HiCache byte round trips; no model or fake I/O.

This qualifies transport only. A full-model K1 and paired AgentX measurement
are separate requirements before H224 enters a reported serving recipe.
"""
import json
import os

import torch

os.environ['SGLANG_FLASHNEXT_STOCK_HICACHE'] = '1'

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateDType, Mamba2StateShape
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
from sglang.srt.mem_cache.pool_host.flashnext_stock import (
    FlashNextStockMambaHost, FlashNextStockQSAHost,
)
from sglang.srt.mem_cache.pool_host.mha import get_mha_host_pool_cls
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool


def exact(a, b):
    return (a.dtype == b.dtype and a.shape == b.shape
            and torch.equal(a.contiguous().view(torch.uint8),
                            b.contiguous().view(torch.uint8)))


def make_mamba(dtype, device):
    shape = Mamba2StateShape(conv=[(1024, 3)], temporal=(2, 128, 128),
        intermediate_size=512, conv_dim=1024, ssm_state_size=128,
        num_heads=2, head_dim=128, state_size=128, conv_kernel=4)
    cp = Mamba2CacheParams(shape=shape, layers=[0, 1],
        dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=dtype))
    return MambaPool(size=8, spec_state_size=4, cache_params=cp,
                     mamba_layer_ids=[0, 1], device=device,
                     enable_linear_replayssm_spec=True,
                     speculative_num_draft_tokens=4)


def mamba(dtype):
    pool = make_mamba(dtype, 'cuda')
    short = ShortConvPool(size=8, state_shape=(3, 128), layer_ids=[0, 1],
                          dtype=torch.bfloat16, device='cuda')
    gram = NGramPool(size=8, context_len=4, eos_token_id=0, device='cuda')
    pool.register_slot_state(short); pool.register_slot_state(gram)
    tensors = dict(temporal=pool.mamba_cache.temporal,
                   conv=pool.mamba_cache.conv[0], short=short.conv_state,
                   ngram=gram.context.unsqueeze(0))
    for t in tensors.values():
        if t.is_floating_point(): t.normal_()
        else: t.random_(1, 1000)
    src = torch.tensor([3, 1], device='cuda')
    dst = torch.tensor([7, 5], device='cuda')
    # Real write_back dispatch may leave host indices on CPU.
    hi = torch.tensor([2, 6], dtype=torch.int64)
    oracle = {name: t[:, src].clone() for name, t in tensors.items()}
    untouched = {name: t[:, 2].clone() for name, t in tensors.items()}
    host = FlashNextStockMambaHost(pool, host_to_device_ratio=2,
                                  host_size=0, layout='page_first')
    try:
        host.backup_from_device_all_layer(pool, hi, src, 'kernel')
        torch.cuda.synchronize()
        for t in tensors.values(): t[:, src] = 0; t[:, dst] = 0
        # Spec replay windows are request-private scratch, not radix payload.
        # Their cursors must not be indexed using persistent Mamba slot IDs.
        assert pool.replayssm_write_pos is None
        pool.replayssm_spec_write_pos.fill_(3)
        for layer in range(2):
            host.load_to_device_per_layer(pool, hi, dst, layer, 'kernel')
        torch.cuda.synchronize()
        checks = {name: exact(oracle[name], t[:, dst]) for name, t in tensors.items()}
        checks['other_slots_unchanged'] = all(exact(untouched[n], t[:, 2]) for n, t in tensors.items())
        checks['spec_cursor_unchanged'] = bool((pool.replayssm_spec_write_pos == 3).all())
        return pool, dict(dtype=str(dtype), checks=checks,
                          host_bytes=host.size * host.size_per_token,
                          passed=all(checks.values()))
    finally:
        host.destroy()


def pages(values, device):
    return torch.cat([torch.arange(v * 64, (v + 1) * 64, device=device)
                      for v in values]).to(torch.int64)


def qsa(pool):
    def make(layers):
        return QSATokenToKVPool(size=512, dtype=torch.bfloat16, page_size=64,
            head_num=2, head_dim=128, full_attention_layer_ids=layers,
            device='cuda', mamba_pool=pool, qsa_index_kv_heads=1,
            qsa_index_head_dim=128, qsa_compress_ratio=16,
            qsa_token_topk=64, num_request_slots=9)
    target, draft = make([0, 1]), make([0])
    assert get_mha_host_pool_cls(target.full_kv_pool) is FlashNextStockQSAHost
    host = FlashNextStockQSAHost(target.full_kv_pool,
        host_to_device_ratio=2, host_size=0, page_size=64, layout='page_first',
        mtp_draft_device_pools=(draft.full_kv_pool,))
    src, dst, hi = pages([3, 1], 'cuda'), pages([6, 4], 'cuda'), pages([2, 5], 'cpu')
    entries = []
    for role, owner in [('target', target), ('draft', draft)]:
        for kind, tensors in [('K', owner.full_kv_pool.k_buffer),
                              ('V', owner.full_kv_pool.v_buffer),
                              ('QSA', owner.qsa_compressed_k_buffer_pool)]:
            si, di = (src[::16] // 16, dst[::16] // 16) if kind == 'QSA' else (src, dst)
            for layer, t in enumerate(tensors):
                t.normal_()
                entries.append((f'{role}-{kind}-{layer}', t, si, di,
                                t[si].clone(), t[0].clone()))
    try:
        host.backup_from_device_all_layer(target.full_kv_pool, hi, src, 'kernel')
        torch.cuda.synchronize()
        for _, t, si, di, _, _ in entries: t[si] = 0; t[di] = 0
        for layer in range(target.full_kv_pool.layer_num):
            host.load_to_device_per_layer(target.full_kv_pool, hi, dst, layer, 'kernel')
        host.load_to_device_per_layer(draft.full_kv_pool, hi, dst,
                                     target.full_kv_pool.layer_num, 'kernel', is_draft=True)
        torch.cuda.synchronize()
        checks = {name: exact(expected, t[di]) and exact(other, t[0])
                  for name, t, _, di, expected, other in entries}
        checks['budget_covers_payload'] = (host.kv_buffer.numel() * host.kv_buffer.element_size()
            + host.index_buffer.numel() * host.index_buffer.element_size()
            == host.size * host.size_per_token)
        return dict(checks=checks, passed=all(checks.values()),
                    host_bytes=host.size * host.size_per_token,
                    page_order=[3, 1], restore_page_order=[6, 4])
    finally:
        host.destroy()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-constructor', action='store_true')
    args = parser.parse_args()
    if args.cpu_constructor:
        assert not torch.cuda.is_available()
        for dtype in (torch.float32, torch.bfloat16):
            pool = make_mamba(dtype, 'cpu')
            assert pool.mamba_cache.temporal.shape == (2, 9, 2, 128, 128)
            assert pool.mamba_cache.conv[0].shape == (2, 9, 1024, 3)
            assert pool.replayssm_write_pos is None
            assert pool.replayssm_spec_write_pos.shape == (5,)
        print(json.dumps(dict(passed=True, device='CPU', cases=2,
            scope='real MambaPool constructor; GPU byte transport still required')))
        raise SystemExit(0)
    torch.manual_seed(457)
    cases = []
    for dtype in (torch.float32, torch.bfloat16):
        pool, state = mamba(dtype)
        case = dict(mamba=state, qsa=qsa(pool))
        cases.append(case)
        print(json.dumps(dict(progress=case)), flush=True)
    passed = all(c['mamba']['passed'] and c['qsa']['passed'] for c in cases)
    print(json.dumps(dict(passed=passed, cases=cases,
                         gpu=torch.cuda.get_device_name(),
                         scope='stock committed dense/conv/PLE/target+draft QSA byte transport; full-model K1 separate')), flush=True)
    raise SystemExit(0 if passed else 1)
