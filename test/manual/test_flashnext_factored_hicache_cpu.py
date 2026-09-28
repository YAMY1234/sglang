"""CPU tests of the factored GDN HiCache payload (TwinStar docs/139).

The production transfer kernels are replaced by a byte-level emulation of their
address arithmetic (transfer_mamba.cuh), so page layout, the flattened
(slot, layer) restore and the flag staging are checked exactly. The GPU test
runs the same round trip with the native kernels.
"""
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch

os.environ.setdefault("SGLANG_FLASHNEXT_FACTOR_HICACHE", "1")

from sglang.srt.mem_cache import memory_pool
from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNConfig, FactoredGDNPool
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import _split_hicache_size
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
from sglang.srt.mem_cache.pool_host import flashnext_factored as ff
from sglang.srt.mem_cache.pool_host import flashnext_stock, mamba as mamba_host

ROOT = Path(__file__).resolve().parents[2]
TENSORS = {}  # data_ptr -> per-layer tensor, for the pointer-table kernel


def _rows(t, item):
    return t.contiguous().view(torch.uint8).view(-1, item) if t.is_contiguous() else None


def lf_pf(src_ptrs, dst, src_indices, dst_indices, item_size, dst_layout_dim, num_layers):
    out = dst.view(torch.uint8).view(-1, dst_layout_dim)
    for layer, ptr in enumerate(src_ptrs.tolist()):
        src = TENSORS[ptr].view(torch.uint8).view(-1, item_size)
        for s, d in zip(src_indices.tolist(), dst_indices.tolist()):
            out[d, layer * item_size:(layer + 1) * item_size] = src[s]


def pf_lf(src, dst, src_indices, dst_indices, layer_id, item_size, src_layout_dim):
    inp = src.view(torch.uint8).view(-1, src_layout_dim)
    out = dst.view(torch.uint8).view(-1, item_size)
    for s, d in zip(src_indices.tolist(), dst_indices.tolist()):
        out[d] = inp[s, layer_id * item_size:(layer_id + 1) * item_size]


for module in (ff, mamba_host):
    module.transfer_kv_mamba_lf_pf = lf_pf
    module.transfer_kv_mamba_pf_lf = pf_lf
mamba_host.host_memory_budget_bytes = lambda: 1 << 40  # the budget is a host fact, not under test


def register(*tensors):
    for tensor in tensors:
        for layer in tensor:
            TENSORS[layer.data_ptr()] = layer


LAYERS, SIZE, HV, V, K = [0, 1, 2], 8, 4, 16, 16


def cache_params():
    return NS(shape=NS(conv=[(64, 3)], temporal=(HV, V, K), disable_conv_window_dedup=False,
                       conv_kernel=4),
              dtype=NS(conv=torch.bfloat16, temporal=torch.bfloat16), is_kda=False,
              layers=LAYERS)


def build(cfg="r=8,m=8,dtype=fp16,ring=2,strict_chunk=1,factored_prefix=1", *, ple=True, factor=True):
    cp = cache_params()
    pool = MambaPool(size=SIZE, spec_state_size=0, cache_params=cp, mamba_layer_ids=LAYERS,
                     device="cpu", **({"empty_temporal": True} if factor else {}))
    parts = NS(pool=pool, factor=None, short=None, gram=None)
    if factor:
        parts.factor = FactoredGDNPool(size=SIZE, cache_params=cp, mamba_layer_ids=LAYERS,
                                       device="cpu", cfg=FactoredGDNConfig.parse(cfg))
        pool.register_slot_state(parts.factor)
    if ple:
        parts.short = ShortConvPool(size=SIZE, state_shape=(3, 32), layer_ids=LAYERS,
                                    dtype=torch.bfloat16, device="cpu")
        parts.gram = NGramPool(size=SIZE, context_len=4, eos_token_id=0, device="cpu")
        pool.register_slot_state(parts.short)
        pool.register_slot_state(parts.gram)
    return parts


def fill(*tensors):
    for t in tensors:
        if t.is_floating_point():
            t.normal_()
        else:
            t.random_(1, 1000)


def exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def host_for(parts, **kw):
    host = ff.FlashNextFactoredMambaHost(parts.pool, host_to_device_ratio=2, host_size=0,
                                         pin_memory=False, layout="page_first", **kw)
    register(*ff.payload_tensors(parts.factor), *parts.pool.mamba_cache.conv,
             parts.short.conv_state, parts.gram.context.unsqueeze(0))
    TENSORS[host.flag_out.data_ptr()] = host.flag_out
    return host


class FactoredHiCacheCPU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(974)
        os.environ["SGLANG_FLASHNEXT_FACTOR_HICACHE"] = "1"

    def payload(self, parts):
        f = parts.factor
        return dict(a=f.a, U=f.U, W=f.W, count=f.count, conv=parts.pool.mamba_cache.conv[0],
                    short=parts.short.conv_state, ngram=parts.gram.context.unsqueeze(0))

    def test_round_trip_is_bitwise_and_restores_authority(self):
        parts = build()
        f = parts.factor
        tensors = self.payload(parts)
        fill(*tensors.values())
        f.prefix_valid.copy_(torch.tensor([0, 1, 0, 1, 0, 0, 1, 0, 1], dtype=torch.int32))
        src, dst, hi = torch.tensor([3, 1]), torch.tensor([7, 5]), torch.tensor([2, 6])
        oracle = {n: t[:, src].clone() for n, t in tensors.items()}
        flag = f.prefix_valid[src].clone()
        untouched = {n: t[:, 2].clone() for n, t in tensors.items()}
        host = host_for(parts)
        host.backup_from_device_all_layer(parts.pool, hi, src, "kernel")
        for t in tensors.values():
            t[:, src] = 0
            t[:, dst] = 0
        f.prefix_valid[dst] = 7
        f.stale[dst] = 0
        f.dense_of[dst] = 1
        f.dense_required[dst] = 1
        for layer in range(len(LAYERS)):
            host.load_to_device_per_layer(parts.pool, hi, dst, layer, "kernel")
        for name, t in tensors.items():
            self.assertTrue(exact(oracle[name], t[:, dst]), name)
            self.assertTrue(exact(untouched[name], t[:, 2]), name)
        self.assertTrue(exact(flag, f.prefix_valid[dst]))
        self.assertEqual(f.stale[dst].tolist(), [1, 1])
        self.assertEqual(f.dense_of[dst].tolist(), [-1, -1])
        self.assertEqual(f.dense_required[dst].tolist(), [0, 0])
        self.assertEqual(host.flag_host[hi, 0, 0, 1:].abs().sum().item(), 0)

    def test_every_factor_layer_lands_with_target_layer_zero(self):
        parts = build()
        f = parts.factor
        fill(f.a, f.U, f.W, f.count)
        host = host_for(parts)
        src, dst, hi = torch.tensor([4]), torch.tensor([6]), torch.tensor([0])
        host.backup_from_device_all_layer(parts.pool, hi, src, "kernel")
        before = f.a[:, dst].clone()
        for layer in (2, 1):
            host.load_to_device_per_layer(parts.pool, hi, dst, layer, "kernel")
        self.assertTrue(exact(before, f.a[:, dst]))
        host.load_to_device_per_layer(parts.pool, hi, dst, 0, "kernel")
        for t in ff.payload_tensors(f):
            self.assertTrue(exact(t[:, src], t[:, dst]))

    def test_host_bytes_cover_payload_exactly(self):
        parts = build()
        host = host_for(parts)
        f = parts.factor
        factor = sum(t[:, 0].numel() * t.element_size() for t in ff.payload_tensors(f))
        stock = flashnext_stock.FlashNextStockMambaHost.get_size_per_token(host)
        conv = sum(t[:, 0].numel() * t.element_size() for t in parts.pool.mamba_cache.conv)
        ple = sum(t[:, 0].numel() * t.element_size() for _, t in host.ple)
        self.assertEqual(stock, conv + ple)
        self.assertEqual(host.get_size_per_token(), conv + ple + factor + 16)
        allocated = sum(b.numel() * b.element_size() for b in host.kv_buffer)
        self.assertEqual(allocated, host.size * host.size_per_token)

    def test_qualification_keeps_stock_rejections(self):
        parts = build()
        os.environ["SGLANG_FLASHNEXT_FACTOR_HICACHE"] = "0"
        self.assertIsNone(ff.qualify(parts.pool))
        with self.assertRaisesRegex(ValueError, "factor/latent"):
            flashnext_stock.ple_tensors(parts.pool)
        os.environ["SGLANG_FLASHNEXT_FACTOR_HICACHE"] = "1"
        self.assertIs(ff.qualify(parts.pool), parts.factor)
        parts.pool._slot_siblings.append(object())
        with self.assertRaisesRegex(ValueError, "factor/latent"):
            ff.qualify(parts.pool)
        parts.pool._slot_siblings.pop()
        parts.pool.prefix_layer_limit = 2
        with self.assertRaisesRegex(ValueError, "partial-layer"):
            ff.qualify(parts.pool)
        with self.assertRaisesRegex(ValueError, "factored_prefix"):
            ff.qualify(build("r=8,m=8,dtype=fp16,ring=2,strict_chunk=1").pool)
        with self.assertRaisesRegex(ValueError, "factored_prefix"):
            ff.qualify(build("r=8,m=8,dtype=fp16,ring=2,strict_chunk=1,exact_prefix=1").pool)

    def test_stock_and_dense_pools_keep_the_stock_path(self):
        dense = build(factor=False)
        self.assertIsNone(ff.qualify(dense.pool))
        stock = flashnext_stock.FlashNextStockMambaHost(
            dense.pool, host_to_device_ratio=2, host_size=0, pin_memory=False, layout="page_first")
        self.assertEqual([n for n, _ in stock.ple], ["short_conv", "ngram"])
        self.assertFalse(hasattr(dense.pool.mamba_cache, "factor_host"))

    def test_d_side_transfer_manifest_is_unchanged(self):
        parts = build()
        def manifest():
            return [(n, t.data_ptr(), tuple(t.shape), t.dtype, axis, lid)
                    for n, t, axis, lid in parts.pool._iter_transfer_state_entries()]
        before = manifest()
        siblings = list(parts.pool._slot_siblings)
        host = host_for(parts)
        host.backup_from_device_all_layer(parts.pool, torch.tensor([1]), torch.tensor([2]), "kernel")
        host.load_to_device_per_layer(parts.pool, torch.tensor([1]), torch.tensor([3]), 0, "kernel")
        self.assertEqual(manifest(), before)
        self.assertEqual(parts.pool._slot_siblings, siblings)

    def test_budget_split_counts_factor_payload_like_dense_state(self):
        meta = lambda *s, dtype=torch.float32: torch.empty(*s, dtype=dtype, device="meta")
        L, S = 36, 481
        factor = NS(a=meta(L, S, 24, 128), U=meta(L, S, 24, 16, 128, dtype=torch.float16),
                    W=meta(L, S, 24, 16, 128, dtype=torch.float16),
                    count=meta(L, S, 24, dtype=torch.int32), prefix_valid=meta(S, dtype=torch.int32))
        self.assertEqual(ff.payload_bytes(factor), S * (7_523_712 + 4))
        conv = int(0.50 * 2**30)
        pool = NS(get_kv_size_bytes=lambda: conv)
        kv = NS(get_kv_size_bytes=lambda: 89_806_152_457)
        _, unfixed = _split_hicache_size(224, (kv, pool))
        _, fixed = _split_hicache_size(224, (kv, ff.StateBytes(pool, factor)))
        per_slot = 7_523_712 + 16 + conv / S
        self.assertEqual(int(unfixed * 1e9 // per_slot), 154)
        self.assertEqual(int(fixed * 1e9 // per_slot), 1146)

    def test_backup_refuses_an_unjoined_deferred_p_commit(self):
        parts = build()
        host = host_for(parts)
        parts.factor._pside_deferred_commit = ("pending",)
        with self.assertRaisesRegex(RuntimeError, "deferred P commit"):
            host.backup_from_device_all_layer(parts.pool, torch.tensor([1]), torch.tensor([2]), "kernel")

    def test_restore_invalidates_verify_windows(self):
        parts = build()
        host = host_for(parts)
        seen = []
        parts.factor.spec_state = NS(invalidate_slots=lambda s: seen.append(s.tolist()))
        host.load_to_device_per_layer(parts.pool, torch.tensor([1, 0]), torch.tensor([4, 2]), 0, "kernel")
        self.assertEqual(seen, [[4, 2]])

    @unittest.skipUnless(hasattr(flashnext_stock, "is_pd_boundary_state"), "fork has no PD boundary state")
    def test_pd_boundary_is_invalidated_while_factors_restore(self):
        from twinstar_sgl.pd_shallow import BoundaryState

        parts = build()
        state = BoundaryState(SIZE, "cpu")
        state.valid.fill_(1)
        state.position.fill_(8191)
        parts.pool.register_slot_state(state)
        self.assertIs(ff.qualify(parts.pool), parts.factor)
        host = host_for(parts)
        self.assertEqual(host.pd_boundaries, (state,))
        fill(*ff.payload_tensors(parts.factor))
        src, dst, hi = torch.tensor([3, 1]), torch.tensor([7, 5]), torch.tensor([2, 6])
        host.backup_from_device_all_layer(parts.pool, hi, src, "kernel")
        host.load_to_device_per_layer(parts.pool, hi, dst, 0, "kernel")
        self.assertEqual(state.valid[dst].sum().item(), 0)
        self.assertEqual(state.position[dst].tolist(), [[-1], [-1]])
        self.assertEqual(state.valid.sum().item(), SIZE + 1 - 2)
        for t in ff.payload_tensors(parts.factor):
            self.assertTrue(exact(t[:, src], t[:, dst]))

    def test_plan_waits_for_layer_zero_restore(self):
        order = []
        counter = NS(wait_until=lambda layer: order.append(("wait", layer)))
        factor = NS()
        rtp = NS(layer_transfer_counter=None)
        ff.install_restore_wait(factor, rtp)
        factor.hicache_wait_restore()
        self.assertEqual(order, [])
        rtp.layer_transfer_counter = counter
        factor.hicache_wait_restore()
        self.assertEqual(order, [("wait", 0)])
        source = (ROOT / "python/sglang/srt/layers/attention/linear/gdn_backend.py").read_text()
        body = source[source.index("def init_forward_metadata(self, forward_batch"):]
        self.assertLess(body.index("hicache_wait_restore"), body.index("self.factored.plan_extend("))


if __name__ == "__main__":
    unittest.main()
