"""Run real catalog/transport signatures and Triton byte copies on CPU."""
import inspect
import os
os.environ.setdefault("TRITON_INTERPRET", "1")
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.flashnext_staging import Catalog
from sglang.srt.disaggregation.flashnext_staging_kernels import copy_payload
from sglang.srt.disaggregation.flashnext_staging_manifest import Manifest
from sglang.srt.disaggregation.pfactor3_state_catalog import CompactFactorCatalog, LayerRows


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
    cls = CompactFactorCatalog if compact else Catalog
    return cls(args=args, pool=pool), backing


def source(catalog, slots, *, last=True):
    kwargs = dict(room=19, generation=1, source_rank=0, source_tp=2,
        prompt_tokens=257, token_start=0, token_end=257,
        kv_indices=np.array([1, 2, 3, 4, 5]), kv_by_entry=None,
        state_indices=[np.asarray(slots)], chunk_index=0,
        last_chunk=last, shallow_boundary=False)
    inspect.signature(Catalog.source_payload).bind(catalog, **kwargs)
    return catalog.source_payload(**kwargs)


class CompactCatalogTest(unittest.TestCase):
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

    def test_invalid_slot_and_peer_layer_order_fail(self):
        p, _ = fixture(1, (0, 2, 4), 11)
        for slots in ([-1], [11], [1, 11]):
            with self.assertRaisesRegex(ValueError, 'factor slot'):
                source(p, slots)
        d, _ = fixture(2, (2, 0, 4), 17)
        m, _ = source(p, [1])
        with self.assertRaisesRegex(ValueError, 'unknown destination'):
            d.destination_payload(manifest=m, kv_indices=np.array([1, 2, 3, 4, 5]),
                state_indices=[np.array([1])], decode_prefix_tokens=0,
                destination_rank=0, destination_tp=2)

    def test_separate_layer_allocations_are_rejected(self):
        p, _ = fixture(1, (0, 2, 4), 11, compact=False)
        records = [(r.name.split('.')[1], r.tensor.clone(), 0, r.layer) for r in p.entries]
        p.pool.mamba_pool._iter_transfer_state_entries = lambda: iter(records)
        p.args.state_data_ptrs = [[r[1].data_ptr() for r in records]]
        with self.assertRaisesRegex(ValueError, 'contiguous registered allocation'):
            CompactFactorCatalog(args=p.args, pool=p.pool)

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
