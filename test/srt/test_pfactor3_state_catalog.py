"""Run real catalog/transport signatures and Triton byte copies on CPU."""
import inspect
import os
os.environ.setdefault("TRITON_INTERPRET", "1")
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.flashnext_staging import Catalog, Entry
from sglang.srt.disaggregation.flashnext_staging_kernels import copy_payload
from sglang.srt.disaggregation.flashnext_staging_manifest import Manifest
from sglang.srt.disaggregation.pfactor3_state_catalog import CompactFactorCatalog, LayerRows, make_catalog


def factor_records(seed, layers, slots):
    gen = torch.Generator().manual_seed(seed)
    backing = {}
    for kind, shape, dtype in (
        ('a', (2, 8), torch.float32), ('u', (2, 2, 8), torch.float16),
        ('w', (2, 2, 8), torch.float16), ('count', (2,), torch.int32),
    ):
        backing[kind] = torch.randint(0, 120, (len(layers), slots, *shape),
                                     generator=gen, dtype=torch.int64).to(dtype)
    records = [(f'gdn_factored_{kind}', tensor[i], 0, layer)
               for i, layer in enumerate(layers) for kind, tensor in backing.items()]
    return records, backing


def fixture(seed, layers, slots, *, compact=True):
    records, backing = factor_records(seed, layers, slots)
    pool = SimpleNamespace(mamba_pool=SimpleNamespace(_iter_transfer_state_entries=lambda: iter(records)))
    args = SimpleNamespace(page_size=64, num_draft_entries=0,
        kv_layer_ids=[], kv_data_ptrs=[], kv_item_lens=[], state_types=[StateType.MAMBA],
        state_layer_ids=[[r[3] for r in records]],
        state_data_ptrs=[[r[1].data_ptr() for r in records]],
        state_item_lens=[[r[1][0].nbytes for r in records]],
        state_conv_shard_groups=[[None] * len(records)])
    # The byte-copy tests exercise the production default selection, too.
    return make_catalog(args=args, pool=pool, mode=None if compact else "0"), backing


def source(catalog, slots, *, last=True, shallow=False):
    kwargs = dict(room=19, generation=1, source_rank=0, source_tp=2,
        prompt_tokens=257, token_start=0, token_end=257,
        kv_indices=np.array([1, 2, 3, 4, 5]), kv_by_entry=None,
        state_indices=[np.asarray(slots)], chunk_index=0,
        last_chunk=last, shallow_boundary=shallow)
    inspect.signature(Catalog.source_payload).bind(catalog, **kwargs)
    return catalog.source_payload(**kwargs)


class CompactCatalogTest(unittest.TestCase):
    def test_opt_in_shallow_keeps_boundary_phase_and_every_destination_byte(self):
        layers=tuple(i for i in range(48) if i%4!=3)
        def shallow(seed,slots,enabled):
            base,backing=fixture(seed,layers,slots,compact=False)
            records=list(base.pool.mamba_pool._iter_transfer_state_entries())
            for i,layer in enumerate(layers):backing['count'][i].fill_(9 if layer<31 else 8)
            boundary=torch.arange(slots*10,dtype=torch.int64).reshape(slots,10)+seed
            backing['h31']=boundary
            records.append(('pd_h31',boundary,None,4294967290))
            base.pool.mamba_pool._iter_transfer_state_entries=lambda:iter(records)
            base.pool.mamba_pool.prefix_layer_limit=31
            base.args.state_layer_ids=[[r[3] for r in records]]
            base.args.state_data_ptrs=[[r[1].data_ptr() for r in records]]
            base.args.state_item_lens=[[r[1][0].nbytes for r in records]]
            base.args.state_conv_shard_groups=[[None]*len(records)]
            with patch.dict(os.environ,SGLANG_PFACTOR4_COMPACT_SHALLOW=str(int(enabled))):
                catalog=make_catalog(args=base.args,pool=base.pool)
            self.assertIs(type(catalog),CompactFactorCatalog if enabled else Catalog)
            return catalog,backing
        for batch in (1,8):
            p,values=shallow(1,11,True)
            old_p,_=shallow(1,11,False)
            d,new=shallow(2,17,True)
            old_d,old=shallow(2,17,False)
            src=list(range(1,batch+1));dst=list(range(16-batch,16))
            compact,local=source(p,src,shallow=True)
            legacy,old_local=source(old_p,src,shallow=True)
            self.assertEqual((compact.shallow_count,compact.deep_count),(9,8))
            self.assertEqual((legacy.shallow_count,legacy.deep_count),(9,8))
            self.assertEqual(len(compact.fields),5)
            self.assertEqual(len(legacy.fields),145)
            for catalog,mapping,manifest in ((d,local,compact),(old_d,old_local,legacy)):
                wire=Manifest.from_bytes(manifest.to_bytes())
                target=catalog.destination_payload(manifest=wire,kv_indices=np.arange(5),
                    state_indices=[np.array(dst)],decode_prefix_tokens=0,
                    destination_rank=0,destination_tp=2)
                buf=torch.full((manifest.nbytes,),237,dtype=torch.uint8)
                copy_payload(manifest=manifest,local=mapping,staging=buf,gather=True)
                copy_payload(manifest=wire,local=target,staging=buf.clone(),gather=False)
            for kind in old:
                self.assertTrue(torch.equal(new[kind].view(torch.uint8),old[kind].view(torch.uint8)),kind)
            self.assertTrue(torch.equal(new['h31'][dst],values['h31'][src]))
            self.assertTrue(torch.all(new['count'][:24,dst]==9))
            self.assertTrue(torch.all(new['count'][24:,dst]==8))

    def test_default_and_explicit_on_have_identical_wire_bytes(self):
        p, _ = fixture(1, tuple(range(36)), 11, compact=False)
        default = make_catalog(args=p.args, pool=p.pool)
        explicit = make_catalog(args=p.args, pool=p.pool, mode="1")
        automatic = make_catalog(args=p.args, pool=p.pool, mode="auto")
        rollback = make_catalog(args=p.args, pool=p.pool, mode="0")
        self.assertIs(type(default), CompactFactorCatalog)
        self.assertIs(type(rollback), Catalog)
        for batch in (1, 8):
            slots = list(range(1, batch + 1))
            wire = source(default, slots)[0].to_bytes()
            self.assertEqual(wire, source(explicit, slots)[0].to_bytes())
            self.assertEqual(wire, source(automatic, slots)[0].to_bytes())
            self.assertEqual(len(source(rollback, slots)[0].fields), 144)

    def test_default_preserves_dense_and_shallow_catalogs(self):
        empty = SimpleNamespace(page_size=64, num_draft_entries=0, kv_layer_ids=[],
            kv_data_ptrs=[], kv_item_lens=[], state_types=[])
        pool = SimpleNamespace(mamba_pool=SimpleNamespace(_iter_transfer_state_entries=lambda: iter(())))
        self.assertIs(type(make_catalog(args=empty, pool=pool)), Catalog)
        from test_flashnext_staging import StagingTest
        dense = StagingTest().make_catalog(21, factor=False)
        self.assertIs(type(make_catalog(args=dense.args, pool=dense.pool)), Catalog)
        for attribute, value in (("shared_arena", True), ("prefix_layer_limit", 31)):
            p, _ = fixture(1, (0, 2, 4), 11, compact=False)
            target = p.pool if attribute == "shared_arena" else p.pool.mamba_pool
            setattr(target, attribute, value)
            self.assertIs(type(make_catalog(args=p.args, pool=p.pool)), Catalog)
        p, _ = fixture(1, (0, 2, 4), 11, compact=False)
        p.pool.mamba_pool.prefix_layer_limit = None
        self.assertIs(type(make_catalog(args=p.args, pool=p.pool)), CompactFactorCatalog)
        records = [(r.name.split('.')[1], r.tensor, 0, r.layer) for r in p.entries]
        records[0] = ("pd_h31", *records[0][1:])
        p.pool.mamba_pool._iter_transfer_state_entries = lambda: iter(records)
        self.assertIs(type(make_catalog(args=p.args, pool=p.pool)), Catalog)
        with self.assertRaisesRegex(ValueError, "must be 0, 1 or auto"):
            make_catalog(args=p.args, pool=p.pool, mode="invalid")

    def test_layer_slot_mapping_and_real_copy_all_bytes(self):
        for count in (3, 36):
            layers = tuple(2 * i for i in range(count))
            for batch in (1, 8):
                with self.subTest(layers=count, batch=batch):
                    p, a = fixture(1, layers, 11)
                    d, b = fixture(2, layers, 17)
                    before = {k: v.clone() for k, v in b.items()}
                    src = list(range(1, batch + 1))
                    dst = list(range(16 - batch, 16))
                    m, local = source(p, src)
                    self.assertEqual(len(m.fields), 4)
                    self.assertEqual((m.shallow_count, m.deep_count), (0, 0))
                    wire = Manifest.from_bytes(m.to_bytes())
                    kwargs = dict(manifest=wire, kv_indices=np.array([1, 2, 3, 4, 5]),
                        state_indices=[np.array(dst)], decode_prefix_tokens=0,
                        destination_rank=0, destination_tp=2)
                    inspect.signature(Catalog.destination_payload).bind(d, **kwargs)
                    target = d.destination_payload(**kwargs)
                    buf = torch.full((m.nbytes,), 237, dtype=torch.uint8)
                    copy_payload(manifest=m, local=local, staging=buf, gather=True)
                    copy_payload(manifest=wire, local=target, staging=buf.clone(), gather=False)
                    for kind in a:
                        self.assertTrue(torch.equal(a[kind][:, src].view(torch.uint8),
                                                    b[kind][:, dst].view(torch.uint8)))
                        other = [i for i in range(17) if i not in dst]
                        self.assertTrue(torch.equal(before[kind][:, other], b[kind][:, other]))
                    dense_catalog, _ = fixture(1, layers, 11, compact=False)
                    legacy, legacy_local = source(dense_catalog, src)
                    self.assertEqual(len(legacy.fields), 4 * count)
                    legacy_d, legacy_b = fixture(2, layers, 17, compact=False)
                    legacy_wire = Manifest.from_bytes(legacy.to_bytes())
                    legacy_target = legacy_d.destination_payload(
                        **dict(kwargs, manifest=legacy_wire))
                    legacy_buffer = torch.full((legacy.nbytes,), 237, dtype=torch.uint8)
                    copy_payload(manifest=legacy, local=legacy_local,
                                 staging=legacy_buffer, gather=True)
                    copy_payload(manifest=legacy_wire, local=legacy_target,
                                 staging=legacy_buffer.clone(), gather=False)
                    for kind in b:
                        # Compare the entire D allocation after BOTH real copy
                        # paths, including every untouched destination slot.
                        self.assertTrue(torch.equal(b[kind].view(torch.uint8),
                                                    legacy_b[kind].view(torch.uint8)))
                    # Canonical logical byte sequence per component matches
                    # the original independent layer fields, including B8.
                    for field in m.fields:
                        view = local[field.key]
                        component = field.name.split('.layers=')[0]
                        old = [legacy_local[f.key].tensor[legacy_local[f.key].rows]
                               for f in legacy.fields if f.name == component]
                        self.assertTrue(torch.equal(torch.cat(old).view(torch.uint8),
                                                    view.tensor[view.rows].view(torch.uint8)))
                    self.assertEqual(p.fixed_transfer_bytes,
                        sum(e.tensor[0].nbytes for e in dense_catalog.entries))

    def test_invalid_slot_fails_before_layer_expansion(self):
        p, _ = fixture(1, (0, 2, 4), 11)
        for slots in ([-1], [11], [1, 11]):
            with self.assertRaisesRegex(ValueError, 'factor slot'):
                source(p, slots)

    def test_peer_layer_order_mismatch_fails(self):
        p, _ = fixture(1, (0, 2, 4), 11)
        d, _ = fixture(2, (2, 0, 4), 17)
        m, _ = source(p, [1])
        with self.assertRaisesRegex(ValueError, 'unknown destination'):
            d.destination_payload(manifest=m, kv_indices=np.array([1, 2, 3, 4, 5]),
                state_indices=[np.array([1])], decode_prefix_tokens=0,
                destination_rank=0, destination_tp=2)

    def test_capacity_counts_all_36_layers_after_coalescing(self):
        p, _ = fixture(1, tuple(range(36)), 11)
        kv = torch.zeros((13, 64, 1, 8), dtype=torch.bfloat16)
        p._add(Entry(3, 'K', kv, -1, 0, 64, slice_axis=2))
        per_page = kv[0].nbytes
        header_bound = (len(p.entries) + 4) * 256
        slot_bytes = p.fixed_transfer_bytes + header_bound + 2 * per_page
        safe_pages = (slot_bytes - p.fixed_transfer_bytes - header_bound) // per_page
        naive_fixed = sum(e.tensor[0].nbytes for e in p.entries if not e.tokens_per_row)
        self.assertEqual(safe_pages, 2)
        self.assertGreater((slot_bytes - naive_fixed - header_bound) // per_page, safe_pages)
        for pages in (1, 2):
            tokens = pages * 64 - 1
            m, _ = p.source_payload(room=19, generation=1, source_rank=0, source_tp=2,
                prompt_tokens=tokens, token_start=0, token_end=tokens,
                kv_indices=np.arange(1, pages + 1), kv_by_entry=None,
                state_indices=[np.array([2])], chunk_index=0,
                last_chunk=True, shallow_boundary=False)
            self.assertLessEqual(m.nbytes, slot_bytes)

    def test_separate_layer_allocations_are_rejected(self):
        p, _ = fixture(1, (0, 2, 4), 11, compact=False)
        records = [(r.name.split('.')[1], r.tensor.clone(), 0, r.layer) for r in p.entries]
        p.pool.mamba_pool._iter_transfer_state_entries = lambda: iter(records)
        p.args.state_data_ptrs = [[r[1].data_ptr() for r in records]]
        with self.assertRaisesRegex(ValueError, 'contiguous registered allocation'):
            CompactFactorCatalog(args=p.args, pool=p.pool)
        with self.assertRaisesRegex(ValueError, 'contiguous registered allocation'):
            make_catalog(args=p.args, pool=p.pool)

    def test_stock_and_shallow_are_rejected_when_explicitly_enabled(self):
        p, _ = fixture(1, (0, 2, 4), 11, compact=False)
        p.pool.shared_arena = True
        with self.assertRaisesRegex(ValueError, 'P48'):
            CompactFactorCatalog(args=p.args, pool=p.pool)
        empty = SimpleNamespace(page_size=64, num_draft_entries=0, kv_layer_ids=[],
            kv_data_ptrs=[], kv_item_lens=[], state_types=[])
        pool = SimpleNamespace(mamba_pool=SimpleNamespace(_iter_transfer_state_entries=lambda: iter(())))
        with self.assertRaisesRegex(ValueError, 'multiple distinct factor layers'):
            CompactFactorCatalog(args=empty, pool=pool)

    def test_native_transport_chunk_capacity_and_completion(self):
        # Reuse the native worker/Endpoint test, supplying genuine contiguous
        # factors with a different slot count on P and D. The test drives the
        # real source/destination methods, byte kernels, bulk segmentation,
        # scatter ACK, abort drain, and lease-generation guards.
        from test_flashnext_staging import StagingTest

        class PackedTransport(StagingTest):
            def make_catalog(self, seed, factor=True):
                base = super().make_catalog(seed, factor)
                records, _ = factor_records(seed, (0, 2, 4), 11 if seed == 21 else 17)
                base.pool.mamba_pool._iter_transfer_state_entries = lambda: iter(records)
                base.args.state_layer_ids[0] = [r[3] for r in records]
                base.args.state_data_ptrs[0] = [r[1].data_ptr() for r in records]
                base.args.state_item_lens[0] = [r[1][0].nbytes for r in records]
                base.args.state_conv_shard_groups[0] = [None] * len(records)
                return CompactFactorCatalog(args=base.args, pool=base.pool)

        PackedTransport().test_native_mooncake_worker_bulk_chunks_and_scatter_ack()


if __name__ == '__main__':
    unittest.main()
