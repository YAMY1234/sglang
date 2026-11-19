"""Unit tests for external-linker device pool assembly."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
    _build_deepseek_v4_device_pool_group,
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDevicePoolEntry(CustomTestCase):
    def test_sparse_multi_component_layer_ranges(self):
        k0 = torch.zeros((8, 3), dtype=torch.uint8)
        k2 = torch.zeros((8, 5), dtype=torch.uint8)
        v0 = torch.zeros((8, 7), dtype=torch.uint8)
        v2 = torch.zeros((8, 11), dtype=torch.uint8)
        pool = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[[k0, k2], [v0, v2]],
            layer_mapping={0: 0, 2: 1},
            page_size=2,
            rows_are_pages=False,
            packed=False,
        )

        indices = torch.tensor([0, 1, 4, 5])
        locations = pool.prepare_locations(indices)
        self.assertEqual(locations, [0, 4])
        pointers, sizes = pool.get_page_buffer_meta(indices)
        self.assertEqual(
            pointers,
            [
                buffer[row].data_ptr()
                for row in locations
                for buffer in (k0, k2, v0, v2)
            ],
        )
        self.assertEqual(sizes, [6, 10, 14, 22] * 2)
        self.assertIsNone(pool.get_prepared_layer_range_meta(locations, 1))

        pointers, sizes, offsets = pool.get_prepared_layer_range_meta(locations, 2)
        self.assertEqual(
            pointers,
            [
                [k2[0].data_ptr()],
                [v2[0].data_ptr()],
                [k2[4].data_ptr()],
                [v2[4].data_ptr()],
            ],
        )
        self.assertEqual(sizes, [[10], [22], [10], [22]])
        self.assertEqual(offsets, [[6], [14], [6], [14]])

    def test_merged_page_entry_stores_kv_then_indexer_in_one_object(self):
        # Rhea 60076 / Hecate H112: two objects per page and stage (KV, DSA
        # indexer) doubled the master's per-object work on lookups and loads.
        from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
            MergedPageEntry,
        )

        kv0 = torch.zeros((8, 4), dtype=torch.uint8)
        kv2 = torch.zeros((8, 4), dtype=torch.uint8)
        ix = torch.zeros((4, 3), dtype=torch.uint8)  # one row per page, layer 2 only
        kv = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[[kv0, kv2]],
            layer_mapping={0: 0, 2: 1},
            page_size=2,
            rows_are_pages=False,
        )
        indexer = DevicePoolEntry(
            name=PoolName.INDEXER,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[[ix]],
            layer_mapping={2: 0},
            page_size=2,
            rows_are_pages=True,
        )
        pool = MergedPageEntry(kv, indexer, object_name="ki")
        self.assertEqual((pool.name, pool.object_name), (PoolName.KV, "ki"))

        indices = torch.tensor([0, 1, 4, 5])  # pages 0 and 2
        locations = pool.prepare_locations(indices)
        self.assertEqual(locations, [(0, 0), (4, 2)])
        pointers, sizes = pool.get_page_buffer_meta(indices)
        self.assertEqual(
            pointers,
            [kv0[0].data_ptr(), kv2[0].data_ptr(), ix[0].data_ptr()]
            + [kv0[4].data_ptr(), kv2[4].data_ptr(), ix[2].data_ptr()],
        )
        self.assertEqual(sizes, [8, 8, 3] * 2)  # each object: the KV page, then the indexer row

        pointers, sizes, offsets = pool.get_prepared_layer_range_meta(locations, 0)
        self.assertEqual(pointers, [[kv0[0].data_ptr()], [kv0[4].data_ptr()]])
        self.assertEqual(offsets, [[0], [0]])
        pointers, sizes, offsets = pool.get_prepared_layer_range_meta(locations, 2)
        self.assertEqual(
            pointers, [[kv2[0].data_ptr(), ix[0].data_ptr()], [kv2[4].data_ptr(), ix[2].data_ptr()]]
        )
        self.assertEqual(sizes, [[8, 3], [8, 3]])
        self.assertEqual(offsets, [[8, 16], [8, 16]])  # the indexer starts after both KV layers
        self.assertIsNone(pool.get_prepared_layer_range_meta(locations, 1))

    def test_vector_key_range_meta_matches_per_layer_loop(self):
        # SGLANG_LINKER_VECTOR_META (GB300 Lk20f: the read threads held the GIL
        # ~45% building these lists layer by layer) must build the same lists.
        from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
            MergedPageEntry,
            vector_key_range_meta,
        )

        def per_layer(pool, locations, num_layers):
            entries = [([], [], []) for _ in locations]
            for layer in range(num_layers):
                meta = pool.get_prepared_layer_range_meta(locations, layer)
                if meta is None:
                    continue
                for entry, *parts in zip(entries, *meta):
                    for dst, part in zip(entry, parts):
                        dst.extend(part)
            return tuple([e[i] for e in entries] for i in range(3))

        kv = [torch.zeros((16, 4), dtype=torch.uint8) for _ in range(3)]
        vk = [torch.zeros((16, 6), dtype=torch.uint8) for _ in range(3)]
        ix = torch.zeros((8, 5), dtype=torch.uint8)
        kv_pool = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[kv, vk],
            layer_mapping={0: 0, 1: [1, 2], 3: 2},
            page_size=2,
            rows_are_pages=False,
        )
        ix_pool = DevicePoolEntry(
            name=PoolName.INDEXER,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[[ix]],
            layer_mapping={1: 0},
            page_size=2,
            rows_are_pages=True,
        )
        merged = MergedPageEntry(kv_pool, ix_pool, object_name="ki")
        indices = torch.tensor([0, 1, 6, 7, 14, 15])
        for pool in (kv_pool, merged):
            locations = pool.prepare_locations(indices)
            with self.subTest(pool=type(pool).__name__):
                self.assertEqual(
                    vector_key_range_meta(pool, locations, 4),
                    per_layer(pool, locations, 4),
                )
        unpacked = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[kv],
            layer_mapping={0: 0},
            page_size=2,
            rows_are_pages=False,
            packed=False,
        )
        self.assertIsNone(vector_key_range_meta(unpacked, [0], 1))

    def test_coalesce_and_vector_put_meta_match_reference(self):
        # SGLANG_LINKER_COALESCE_RANGES merges ranges contiguous in the object and
        # in device memory, and is a no-op on separate tensors; SGLANG_LINKER_VECTOR_PUT
        # builds the store's packed per-page buffer lists with numpy.
        from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
            MergedPageEntry,
            vector_key_range_meta,
            vector_page_buffer_meta,
        )
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        big = torch.zeros((16, 8), dtype=torch.uint8)
        halves = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[[big[:, 0:4], big[:, 4:8]]],
            layer_mapping={0: [0, 1]},
            page_size=2,
            rows_are_pages=True,
        )
        locations = halves.prepare_locations(torch.tensor([0, 1, 6, 7]))
        ptrs, sizes, offsets = vector_key_range_meta(halves, locations, 1, coalesce=True)
        self.assertEqual(sizes, [[8], [8]])
        self.assertEqual(offsets, [[0], [0]])
        self.assertEqual(ptrs, [[big[r].data_ptr()] for r in locations])

        kv = [torch.zeros((16, 4), dtype=torch.uint8) for _ in range(3)]
        ix = torch.zeros((8, 5), dtype=torch.uint8)
        kv_pool = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[kv],
            layer_mapping={0: 0, 1: 1, 2: 2},
            page_size=2,
            rows_are_pages=False,
        )
        ix_pool = DevicePoolEntry(
            name=PoolName.INDEXER,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[[ix]],
            layer_mapping={1: 0},
            page_size=2,
            rows_are_pages=True,
        )
        merged = MergedPageEntry(kv_pool, ix_pool, object_name="ki")
        single = DevicePoolEntry(
            name=PoolName.INDEXER,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[[ix]],
            layer_mapping={0: 0},
            page_size=2,
            rows_are_pages=True,
        )
        indices = torch.tensor([0, 1, 6, 7, 14, 15])
        for pool in (kv_pool, merged, single):
            with self.subTest(pool=type(pool).__name__):
                locations = pool.prepare_locations(indices)
                self.assertEqual(
                    vector_key_range_meta(pool, locations, 3, coalesce=True),
                    vector_key_range_meta(pool, locations, 3),
                )
                ptr_list, size_list = pool.get_page_buffer_meta(indices)
                keys = [f"k{i}" for i in range(len(locations))]
                expected = MooncakeStore._pack_multi_buffer_meta(keys, ptr_list, size_list)
                self.assertEqual(vector_page_buffer_meta(pool, indices), expected)

        store = MooncakeStore.__new__(MooncakeStore)
        results = [0, -1, 5, 0]
        for flag in (False, True):
            store._vector_put_flag = flag
            for is_set in (False, True):
                got = store._batch_postprocess(results, is_set_operate=is_set, key_multiplier=1)
                self.assertEqual(got, [r == 0 if is_set else r > 0 for r in results])

    def test_store_arrays_match_the_range_lists(self):
        # SGLANG_LINKER_STORE_ARRAYS hands the store flat uint64 arrays plus a CSR
        # indptr; they must hold exactly the per-key lists of the list call.
        from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
            MergedPageEntry,
            RangeArrays,
            vector_key_range_meta,
        )

        kv = [torch.zeros((16, 4), dtype=torch.uint8) for _ in range(3)]
        ix = torch.zeros((8, 5), dtype=torch.uint8)
        kv_pool = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[kv],
            layer_mapping={0: 0, 1: 1, 2: 2},
            page_size=2,
            rows_are_pages=False,
        )
        ix_pool = DevicePoolEntry(
            name=PoolName.INDEXER,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[[ix]],
            layer_mapping={1: 0},
            page_size=2,
            rows_are_pages=True,
        )
        merged = MergedPageEntry(kv_pool, ix_pool, object_name="ki")
        indices = torch.tensor([0, 1, 6, 7, 14, 15])
        for pool in (kv_pool, merged):
            for coalesce in (False, True):
                with self.subTest(pool=type(pool).__name__, coalesce=coalesce):
                    locations = pool.prepare_locations(indices)
                    lists = vector_key_range_meta(pool, locations, 3, coalesce)
                    arrays = vector_key_range_meta(
                        pool, locations, 3, coalesce, arrays=True
                    )
                    self.assertIsInstance(arrays, RangeArrays)
                    for field in arrays:
                        self.assertEqual(field.dtype, np.uint64)
                        self.assertTrue(field.flags.c_contiguous)
                    self.assertEqual(arrays.to_lists(), lists)
                    self.assertEqual(
                        RangeArrays.from_lists(*lists).to_lists(), lists
                    )
                    self.assertEqual(arrays.expected(), [sum(s) for s in lists[1]])

        ragged = ([[1, 2], [], [3]], [[10, 20], [], [30]], [[0, 10], [], [0]])
        parts = RangeArrays.from_lists(*ragged)
        self.assertEqual(parts.indptr.tolist(), [0, 2, 2, 3])
        self.assertEqual(parts.expected(), [30, 0, 30])
        self.assertEqual(
            parts.select([2, 0]).to_lists(),
            ([[3], [1, 2]], [[30], [10, 20]], [[0], [0, 10]]),
        )
        both = RangeArrays.concat([parts, RangeArrays.from_lists([[4]], [[40]], [[5]])])
        self.assertEqual(
            both.to_lists(),
            tuple(a + b for a, b in zip(ragged, ([[4]], [[40]], [[5]]))),
        )

    def test_cpu_indices_keep_device_copies_off_linker_threads(self):
        # SGLANG_LINKER_CPU_INDICES: prefetch slots (allocator) and offload node
        # values are device tensors; the scheduler thread copies them to the CPU
        # so _rows on the read/offload threads never D2H-copies against the forward.
        from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
            attach_cpu_indices,
        )

        host = torch.tensor([0, 1, 4, 5])
        device = Mock(is_cuda=True, shape=host.shape, dtype=host.dtype)
        pinned = Mock()
        with patch(
            "sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler.torch.empty",
            return_value=pinned,
        ) as empty:
            out = attach_cpu_indices(
                [
                    PoolTransfer(name=PoolName.KV, keys=["a", "b"], device_indices=device),
                    PoolTransfer(name=PoolName.MAMBA, keys=["s"], device_indices=host),
                ]
            )
        empty.assert_called_once_with(host.shape, dtype=host.dtype, pin_memory=True)
        pinned.copy_.assert_called_once_with(device, non_blocking=True)
        self.assertIs(out[0].cpu_indices, pinned)
        self.assertIsNone(out[1].cpu_indices)  # already on the CPU

        pool = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[[torch.zeros((8, 3), dtype=torch.uint8)]],
            layer_mapping={0: 0},
            page_size=2,
            rows_are_pages=False,
        )
        before = DevicePoolEntry.cuda_index_copies
        pool.prepare_locations(host)
        self.assertEqual(DevicePoolEntry.cuda_index_copies, before)

    def test_page_checksum_detects_a_corrupted_page(self):
        # SGLANG_LINKER_PAGE_CHECKSUM: a page copied intact checksums equal on both
        # sides; one flipped byte in the destination shows as one mismatch.
        import collections
        import threading

        from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
            MergedPageEntry,
            page_checksums,
        )
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
            MooncakeDirectLinker,
        )

        def pools(seed):
            g = torch.Generator().manual_seed(seed)
            kv = [torch.randint(0, 256, (16, 4), dtype=torch.uint8, generator=g) for _ in range(2)]
            ix = torch.randint(0, 256, (8, 5), dtype=torch.uint8, generator=g)
            kv_pool = DevicePoolEntry(
                name=PoolName.KV, indices_from_pool=PoolName.KV, device_pool=None,
                components=[kv], layer_mapping={0: 0, 1: 1}, page_size=2, rows_are_pages=False,
            )
            ix_pool = DevicePoolEntry(
                name=PoolName.INDEXER, indices_from_pool=PoolName.KV, device_pool=None,
                components=[[ix]], layer_mapping={1: 0}, page_size=2, rows_are_pages=True,
            )
            return MergedPageEntry(kv_pool, ix_pool, object_name="ki"), kv, ix

        src, src_kv, src_ix = pools(0)
        dst, dst_kv, dst_ix = pools(1)
        src_loc = src.prepare_locations(torch.tensor([2, 3, 10, 11]))
        dst_loc = dst.prepare_locations(torch.tensor([6, 7, 12, 13]))
        for (s_tok, s_row), (d_tok, d_row) in zip(src_loc, dst_loc):  # the "read"
            for a, b in zip(src_kv, dst_kv):
                b[d_tok : d_tok + 2] = a[s_tok : s_tok + 2]
            dst_ix[d_row] = src_ix[s_row]
        self.assertEqual(page_checksums(src, src_loc), page_checksums(dst, dst_loc))

        linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        linker.page_checksum_rate = 1.0
        linker.page_checksums = collections.OrderedDict(
            zip(["a", "b"], page_checksums(src, src_loc))
        )
        linker.page_checksum_counts = collections.Counter()
        linker.page_checksum_lock = threading.Lock()
        linker._checksum_stream = None
        linker.pools = {PoolName.KV: dst}
        linker._verify_page_checksums({PoolName.KV: (["a", "b"], dst_loc)})
        self.assertEqual(linker.page_checksum_counts["mismatched"], 0)
        self.assertEqual(linker.page_checksum_counts["checked"], 2)
        dst_kv[1][dst_loc[1][0], 3] ^= 0xFF
        linker._verify_page_checksums({PoolName.KV: (["a", "b", "c"], dst_loc + dst_loc[:1])})
        self.assertEqual(linker.page_checksum_counts["mismatched"], 1)
        self.assertEqual(linker.page_checksum_counts["missing"], 1)

    def test_rejects_invalid_pages_and_empty_buffers(self):
        with self.assertRaisesRegex(ValueError, "has no storage buffers"):
            DevicePoolEntry(
                name=PoolName.KV,
                indices_from_pool=PoolName.KV,
                device_pool=None,
                components=[],
                layer_mapping={},
                page_size=2,
                rows_are_pages=False,
            )

        pool = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=None,
            components=[[torch.zeros((8, 3), dtype=torch.uint8)]],
            layer_mapping={0: 0},
            page_size=2,
            rows_are_pages=False,
        )
        for indices, error in (
            (torch.tensor([0]), "multiple of page_size"),
            (torch.tensor([1, 2]), "aligned contiguous pages"),
            (torch.tensor([0, 2]), "aligned contiguous pages"),
            (torch.tensor([8, 9]), "exceeds buffer shapes"),
        ):
            with self.subTest(indices=indices.tolist()):
                with self.assertRaisesRegex(ValueError, error):
                    pool.prepare_locations(indices)


class TestDevicePoolGroup(CustomTestCase):
    def test_resolve_transfers_expands_physical_pools(self):
        entries = [
            SimpleNamespace(
                name=PoolName.KV,
                indices_from_pool=PoolName.KV,
                translate_indices=lambda indices: indices,
            ),
            SimpleNamespace(
                name=PoolName.INDEXER,
                indices_from_pool=PoolName.KV,
                translate_indices=lambda indices: indices + 100,
            ),
        ]
        group = DevicePoolGroup(entries, num_layers=2, page_size=2)
        transfer = PoolTransfer(
            name=PoolName.KV,
            keys=["a", "b"],
            device_indices=torch.tensor([0, 1, 4, 5]),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        resolved = group.resolve_transfers([transfer])

        self.assertEqual(
            [item.name for item in resolved], [PoolName.KV, PoolName.INDEXER]
        )
        self.assertEqual(resolved[0].host_indices.tolist(), [0, 1, 4, 5])
        self.assertEqual(resolved[1].host_indices.tolist(), [100, 101, 104, 105])
        self.assertTrue(
            all(item.hit_policy == PoolHitPolicy.ALL_PAGES for item in resolved)
        )

    def test_partial_side_pool_requires_explicit_opt_in(self):
        entry = SimpleNamespace(
            name=PoolName.SWA,
            indices_from_pool=PoolName.SWA,
            translate_indices=lambda indices: indices + 100,
        )
        group = DevicePoolGroup([entry], num_layers=1, page_size=2)
        transfer = PoolTransfer(
            name=PoolName.SWA,
            keys=["b", "d"],
            device_indices=torch.tensor([20, 21, 24, 25]),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        self.assertEqual(group.resolve_transfers([transfer]), [])
        resolved = group.resolve_transfers(
            [transfer], allow_partial=True, allow_missing_kv=True
        )

        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].name, PoolName.SWA)
        self.assertEqual(resolved[0].keys, ["b", "d"])
        self.assertEqual(resolved[0].host_indices.tolist(), [120, 121, 124, 125])
        self.assertEqual(resolved[0].hit_policy, PoolHitPolicy.TRAILING_PAGES)


class TestHybridDevicePoolAssembler(CustomTestCase):
    def test_deepseek_v4_maps_sparse_sidecars(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4LayerItem,
            DeepSeekV4TokenToKVPool,
        )

        def state_pool():
            return SimpleNamespace(
                ring_size=2,
                kv_score_buffer=SimpleNamespace(kv_score=torch.zeros((8, 3))),
            )

        kvcache = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        kvcache._unified_kv = False
        kvcache.start_layer = 1
        kvcache.end_layer = 4
        kvcache.swa_page_size = 2
        kvcache.swa_kv_pool = SimpleNamespace(
            page_size=2,
            kv_buffer=[torch.zeros((8, 3), dtype=torch.uint8) for _ in range(3)],
        )
        kvcache.c4_kv_pool = SimpleNamespace(
            kv_buffer=[torch.zeros((8, 5), dtype=torch.uint8) for _ in range(2)],
            bytes_per_page_padded=5,
        )
        kvcache.c4_indexer_kv_pool = SimpleNamespace(
            index_k_with_scale_buffer=[
                torch.zeros((8, 7), dtype=torch.uint8) for _ in range(2)
            ]
        )
        kvcache.c128_kv_pool = SimpleNamespace(
            kv_buffer=[torch.zeros((8, 11), dtype=torch.uint8)],
            bytes_per_page_padded=11,
        )
        kvcache.layer_mapping = [
            DeepSeekV4LayerItem(0, -1),
            DeepSeekV4LayerItem(4, 0),
            DeepSeekV4LayerItem(128, 0),
            DeepSeekV4LayerItem(4, 1),
        ]
        kvcache.compress_state_pools = [None, state_pool(), None, state_pool()]
        kvcache.indexer_compress_state_pools = [
            None,
            state_pool(),
            None,
            state_pool(),
        ]
        draft_swa_buffers = [
            torch.zeros((8, 13), dtype=torch.uint8),
            torch.zeros((8, 17), dtype=torch.uint8),
        ]

        group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=2,
            params=SimpleNamespace(
                mtp_draft_device_pools=(
                    SimpleNamespace(
                        swa_kv_pool=SimpleNamespace(kv_buffer=draft_swa_buffers)
                    ),
                )
            ),
            components={ComponentType.FULL, ComponentType.SWA},
        )

        self.assertEqual(group.num_layers, 3)
        self.assertTrue(group.rank_replicated)
        self.assertEqual(
            set(group.entry_map),
            {
                PoolName.SWA,
                PoolName.DEEPSEEK_V4_C4,
                PoolName.DEEPSEEK_V4_C4_INDEXER,
                PoolName.DEEPSEEK_V4_C128,
                PoolName.DEEPSEEK_V4_C4_STATE,
                PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            },
        )
        self.assertEqual(group.sources[PoolName.DEEPSEEK_V4_C4], PoolName.KV)
        self.assertEqual(group.sources[PoolName.DEEPSEEK_V4_C4_STATE], PoolName.SWA)

        c4_pool = group.entry_map[PoolName.DEEPSEEK_V4_C4]
        pointers, sizes = c4_pool.get_page_buffer_meta(torch.tensor([0, 1]))
        self.assertEqual(len(pointers), 2)
        self.assertEqual(sizes, [5, 5])
        _, sizes, offsets = c4_pool.get_prepared_layer_range_meta([0], 2)
        self.assertEqual(sizes, [[5]])
        self.assertEqual(offsets, [[5]])
        self.assertIsNone(c4_pool.get_prepared_layer_range_meta([0], 1))

        swa_pool = group.entry_map[PoolName.SWA]
        _, sizes, offsets = swa_pool.get_prepared_layer_range_meta([0], 0)
        self.assertEqual(sizes, [[3, 13]])
        self.assertEqual(offsets, [[0, 9]])
        _, sizes, offsets = swa_pool.get_prepared_layer_range_meta([0], 1)
        self.assertEqual(sizes, [[3, 17]])
        self.assertEqual(offsets, [[3, 22]])

    def test_deepseek_v41_shared_kv_and_split_indexer_pages(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4LayerItem,
            DeepSeekV4TokenToKVPool,
        )

        def index_pool(layers, rows, width):
            return SimpleNamespace(
                page_size=2,
                index_k_with_scale_buffer=None,
                index_k_payload_buffer=[
                    torch.arange(rows * width, dtype=torch.uint8).reshape(rows, width)
                    for _ in range(layers)
                ],
                index_k_scale_buffer=[
                    torch.zeros((rows, 2), dtype=torch.uint8) for _ in range(layers)
                ],
            )

        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = False
        pool.start_layer, pool.end_layer = 1, 7
        pool.swa_page_size = 8
        pool.swa_kv_pool = SimpleNamespace(
            page_size=8,
            kv_buffer=[torch.zeros((5, 3), dtype=torch.uint8) for _ in range(6)],
        )
        pool.c4_kv_pool = pool.c128_kv_pool = pool.c4_indexer_kv_pool = None
        pool.sources_by_ratio = {1: [4], 2: [2, 6]}
        pool.kv_pools = {
            1: SimpleNamespace(kv_buffer=[torch.zeros((5, 7), dtype=torch.uint8)]),
            2: SimpleNamespace(
                kv_buffer=[torch.zeros((5, 11), dtype=torch.uint8) for _ in range(2)]
            ),
        }
        pool.index_pools = {1: index_pool(1, 23, 4), 2: index_pool(2, 13, 6)}
        pool.layer_mapping = [
            DeepSeekV4LayerItem(ratio, layer)
            for ratio, layer in [
                (0, -1),
                (0, -1),
                (2, 0),
                (2, 0),
                (1, 0),
                (1, 0),
                (2, 1),
            ]
        ]
        pool.compress_state_pools = [None] * 7
        pool.indexer_compress_state_pools = [None] * 7
        draft = SimpleNamespace(
            swa_kv_pool=SimpleNamespace(
                kv_buffer=[torch.zeros((5, 13), dtype=torch.uint8) for _ in range(2)]
            )
        )
        group = resolve_hybrid_device_pool_group(
            kvcache=pool,
            page_size=8,
            params=SimpleNamespace(mtp_draft_device_pools=(draft,)),
            components={ComponentType.FULL, ComponentType.SWA},
        )
        self.assertEqual(
            set(group.entry_map),
            {
                PoolName.SWA,
                PoolName.DEEPSEEK_V4_C1,
                PoolName.DEEPSEEK_V4_C1_INDEXER,
                PoolName.DEEPSEEK_V4_C1_INDEXER_SCALE,
                PoolName.DEEPSEEK_V4_C2,
                PoolName.DEEPSEEK_V4_C2_INDEXER,
                PoolName.DEEPSEEK_V4_C2_INDEXER_SCALE,
            },
        )
        self.assertEqual(group.entry_map[PoolName.DEEPSEEK_V4_C1].layer_mapping, {3: 0})
        c2 = group.entry_map[PoolName.DEEPSEEK_V4_C2]
        self.assertEqual(c2.layer_mapping, {1: 0, 5: 1})
        self.assertIsNone(c2.get_prepared_layer_range_meta([1], 2))
        self.assertEqual(
            c2.get_prepared_layer_range_meta([1], 5),
            ([[pool.kv_pools[2].kv_buffer[1][1].data_ptr()]], [[11]], [[11]]),
        )
        full = PoolTransfer(
            name=PoolName.KV, keys=["page"], device_indices=torch.arange(8, 16)
        )
        swa = PoolTransfer(
            name=PoolName.SWA,
            keys=["tail"],
            device_indices=torch.arange(8, 16),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        transfers = {t.name: t for t in group.resolve_transfers([full, swa])}
        self.assertEqual(transfers[PoolName.DEEPSEEK_V4_C2].keys, ["page"])
        self.assertEqual(
            transfers[PoolName.DEEPSEEK_V4_C2].hit_policy, PoolHitPolicy.ALL_PAGES
        )
        self.assertEqual(transfers[PoolName.SWA].keys, ["tail"])
        for ratio, name, width, raw_row in [
            (1, PoolName.DEEPSEEK_V4_C1_INDEXER, 16, 4),
            (2, PoolName.DEEPSEEK_V4_C2_INDEXER, 12, 2),
        ]:
            entry = group.entry_map[name]
            pointers, sizes = entry.get_page_buffer_meta(torch.arange(8, 16))
            original = pool.index_pools[ratio].index_k_payload_buffer
            self.assertEqual(pointers, [b[raw_row].data_ptr() for b in original])
            self.assertEqual(sizes, [width] * len(original))
            self.assertTrue(
                torch.equal(
                    entry.components[0][0][1],
                    original[0][raw_row : raw_row * 2].flatten(),
                )
            )
        self.assertEqual(
            group.entry_map[PoolName.DEEPSEEK_V4_C1_INDEXER_SCALE].get_page_buffer_meta(
                torch.arange(8, 16)
            )[1],
            [8],
        )
        self.assertEqual(
            group.entry_map[PoolName.DEEPSEEK_V4_C2_INDEXER_SCALE].get_page_buffer_meta(
                torch.arange(8, 16)
            )[1],
            [4, 4],
        )
        _, sizes, offsets = group.entry_map[PoolName.SWA].get_prepared_layer_range_meta(
            [1], 0
        )
        self.assertEqual(sizes, [[3, 13]])
        self.assertEqual(offsets, [[0, 18]])

    def test_deepseek_v41_request_window_drops_swa(self):
        """Encoder replay leaves no target SWA pool: skip SWA and the draft SWA
        rather than dereference them, and refuse pools that take SWA indices."""
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4LayerItem,
            DeepSeekV4TokenToKVPool,
        )

        def state_pool():
            return SimpleNamespace(
                ring_size=2,
                kv_score_buffer=SimpleNamespace(kv_score=torch.zeros((4, 3))),
            )

        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = False
        pool.start_layer, pool.end_layer = 0, 2
        pool.swa_page_size = 8
        pool.swa_kv_pool = None
        pool.c4_kv_pool = pool.c128_kv_pool = pool.c4_indexer_kv_pool = None
        pool.sources_by_ratio = {1: [1]}
        pool.kv_pools = {
            1: SimpleNamespace(kv_buffer=[torch.zeros((3, 7), dtype=torch.uint8)])
        }
        pool.index_pools = {
            1: SimpleNamespace(
                page_size=2,
                index_k_with_scale_buffer=[torch.zeros((9, 4), dtype=torch.uint8)],
            )
        }
        pool.layer_mapping = [DeepSeekV4LayerItem(0, -1), DeepSeekV4LayerItem(1, 0)]
        pool.compress_state_pools = [None, None]
        pool.indexer_compress_state_pools = [None, None]
        draft = SimpleNamespace(
            swa_kv_pool=SimpleNamespace(
                kv_buffer=[torch.zeros((3, 5), dtype=torch.uint8)]
            )
        )
        params = SimpleNamespace(mtp_draft_device_pools=(draft,))
        group = resolve_hybrid_device_pool_group(
            kvcache=pool,
            page_size=8,
            params=params,
            components={ComponentType.FULL, ComponentType.SWA},
        )
        self.assertEqual(
            set(group.entry_map),
            {PoolName.DEEPSEEK_V4_C1, PoolName.DEEPSEEK_V4_C1_INDEXER},
        )
        indexer = group.entry_map[PoolName.DEEPSEEK_V4_C1_INDEXER]
        _, sizes = indexer.get_page_buffer_meta(torch.arange(8, 16))
        self.assertEqual(sizes, [16])

        pool.layer_mapping[0] = DeepSeekV4LayerItem(4, 0)
        pool.c4_kv_pool = SimpleNamespace(
            kv_buffer=[torch.zeros((3, 5), dtype=torch.uint8)],
            bytes_per_page_padded=5,
        )
        pool.c4_indexer_kv_pool = SimpleNamespace(
            index_k_with_scale_buffer=[torch.zeros((3, 6), dtype=torch.uint8)]
        )
        pool.compress_state_pools[0] = state_pool()
        pool.indexer_compress_state_pools[0] = state_pool()
        with self.assertRaisesRegex(ValueError, "SWA page indices"):
            resolve_hybrid_device_pool_group(
                kvcache=pool,
                page_size=8,
                params=params,
                components={ComponentType.FULL, ComponentType.SWA},
            )

    def test_unified_deepseek_v4_uses_only_compressed_pools(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4LayerItem

        for split_indexer in (False, True):
            with self.subTest(split_indexer=split_indexer):
                c4 = [torch.zeros((4, width), dtype=torch.uint8) for width in (5, 7)]
                c128 = [torch.zeros((4, 11), dtype=torch.uint8)]
                expected = {
                    PoolName.DEEPSEEK_V4_C4: c4,
                    PoolName.DEEPSEEK_V4_C128: c128,
                }
                if split_indexer:
                    payload = [
                        torch.zeros((4, 1, 4, 1, 16), dtype=torch.uint8)
                        for _ in range(2)
                    ]
                    scale = [
                        torch.zeros((4, 1, 4, 1), dtype=torch.uint8) for _ in range(2)
                    ]
                    indexer = SimpleNamespace(
                        index_k_with_scale_buffer=None,
                        index_k_payload_buffer=payload,
                        index_k_scale_buffer=scale,
                    )
                    expected[PoolName.DEEPSEEK_V4_C4_INDEXER] = [
                        b.flatten(1) for b in payload
                    ]
                    expected[PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE] = [
                        b.flatten(1) for b in scale
                    ]
                else:
                    buffers = [
                        torch.zeros((4, width), dtype=torch.uint8) for width in (13, 17)
                    ]
                    indexer = SimpleNamespace(index_k_with_scale_buffer=buffers)
                    expected[PoolName.DEEPSEEK_V4_C4_INDEXER] = buffers
                regions = {4: (c4, 7), 128: (c128, 11)}
                kvcache = SimpleNamespace(
                    _unified_kv=True,
                    start_layer=0,
                    end_layer=3,
                    layer_mapping=[
                        DeepSeekV4LayerItem(4, 1),
                        DeepSeekV4LayerItem(128, 0),
                        DeepSeekV4LayerItem(4, 0),
                    ],
                    # Unified KV has neither paged KV nor an index-addressed SWA pool.
                    swa_kv_pool=None,
                    c4_kv_pool=None,
                    c128_kv_pool=None,
                    swa_page_size=3,
                    c4_indexer_kv_pool=indexer,
                    unified_region_buffers=Mock(side_effect=regions.__getitem__),
                    # bf16 unified rows are a single pool: no rope sibling.
                    unified_rope_region_buffers=Mock(return_value=None),
                )
                group = _build_deepseek_v4_device_pool_group(kvcache, page_size=2)

                # expected carries no *_ROPE entry, so this also pins that the
                # bf16 layout ships exactly what it did before.
                self.assertEqual(set(group.entry_map), set(expected))
                self.assertEqual(set(group.sources.values()), {PoolName.KV})
                self.assertTrue(group.rank_replicated)
                for name, buffers in expected.items():
                    entry = group.entry_map[name]
                    actual = entry.components[0]
                    self.assertEqual(len(actual), len(buffers))
                    for got, want in zip(actual, buffers):
                        self.assertEqual(got.data_ptr(), want.data_ptr())
                        self.assertEqual(got.shape, want.shape)
                self.assertEqual(
                    kvcache.unified_region_buffers.call_args_list, [call(4), call(128)]
                )
                resolved = group.resolve_transfers(
                    [
                        PoolTransfer(
                            name=PoolName.KV,
                            keys=["page-0"],
                            device_indices=torch.tensor([0, 1]),
                        )
                    ]
                )
                self.assertEqual({t.name for t in resolved}, set(expected))

    def test_unified_deepseek_v4_fp8_ships_the_rope_half(self):
        """fp8 rows are two pools; shipping nope alone leaves stale rope."""
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4LayerItem

        c4 = [torch.zeros((4, width), dtype=torch.uint8) for width in (5, 7)]
        c128 = [torch.zeros((4, 11), dtype=torch.uint8)]
        c4_rope = [torch.zeros((4, 3), dtype=torch.uint8) for _ in range(2)]
        c128_rope = [torch.zeros((4, 3), dtype=torch.uint8)]
        indexer = [torch.zeros((4, width), dtype=torch.uint8) for width in (13, 17)]
        regions = {4: (c4, 7), 128: (c128, 11)}
        rope_regions = {4: (c4_rope, 3), 128: (c128_rope, 3)}
        kvcache = SimpleNamespace(
            _unified_kv=True,
            start_layer=0,
            end_layer=3,
            layer_mapping=[
                DeepSeekV4LayerItem(4, 1),
                DeepSeekV4LayerItem(128, 0),
                DeepSeekV4LayerItem(4, 0),
            ],
            swa_kv_pool=None,
            c4_kv_pool=None,
            c128_kv_pool=None,
            swa_page_size=3,
            c4_indexer_kv_pool=SimpleNamespace(index_k_with_scale_buffer=indexer),
            unified_region_buffers=Mock(side_effect=regions.__getitem__),
            unified_rope_region_buffers=Mock(side_effect=rope_regions.__getitem__),
        )
        group = _build_deepseek_v4_device_pool_group(kvcache, page_size=2)

        expected_rope = {
            PoolName.DEEPSEEK_V4_C4_ROPE: c4_rope,
            PoolName.DEEPSEEK_V4_C128_ROPE: c128_rope,
        }
        self.assertTrue(set(expected_rope).issubset(group.entry_map))
        for name, buffers in expected_rope.items():
            actual = group.entry_map[name].components[0]
            self.assertEqual(len(actual), len(buffers))
            for got, want in zip(actual, buffers):
                self.assertEqual(got.data_ptr(), want.data_ptr())
                self.assertEqual(got.shape, want.shape)
        self.assertEqual(
            kvcache.unified_rope_region_buffers.call_args_list, [call(4), call(128)]
        )

        # One row index addresses both halves: same layer mapping, same source.
        for rope_name, nope_name in (
            (PoolName.DEEPSEEK_V4_C4_ROPE, PoolName.DEEPSEEK_V4_C4),
            (PoolName.DEEPSEEK_V4_C128_ROPE, PoolName.DEEPSEEK_V4_C128),
        ):
            self.assertEqual(
                group.entry_map[rope_name].layer_mapping,
                group.entry_map[nope_name].layer_mapping,
            )
            self.assertEqual(group.sources[rope_name], group.sources[nope_name])

        # One KV transfer must expand to both halves.
        resolved = group.resolve_transfers(
            [
                PoolTransfer(
                    name=PoolName.KV,
                    keys=["page-0"],
                    device_indices=torch.tensor([0, 1]),
                )
            ]
        )
        self.assertTrue(set(expected_rope).issubset({t.name for t in resolved}))

    def test_deepseek_v4_still_rejects_hisparse(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import HiSparseC4DevicePool

        kvcache = SimpleNamespace(
            c4_kv_pool=HiSparseC4DevicePool.__new__(HiSparseC4DevicePool)
        )
        with self.assertRaisesRegex(ValueError, "does not support HiSparse"):
            _build_deepseek_v4_device_pool_group(kvcache, 2)

    def test_dsa_uses_hybrid_assembler_strategy(self):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        def dsa_pool(kv_width, index_width):
            pool = DSATokenToKVPool.__new__(DSATokenToKVPool)
            pool.page_size = 2
            pool.layer_num = 1
            pool.kv_buffer = [torch.zeros((8, kv_width), dtype=torch.uint8)]
            pool.index_key_cache = SimpleNamespace(
                buffer=[torch.zeros((4, index_width), dtype=torch.uint8)]
            )
            return pool

        kvcache = dsa_pool(3, 7)
        kvcache.layer_num = 2
        kvcache.kv_buffer.append(torch.zeros((8, 5), dtype=torch.uint8))
        kvcache.index_key_cache.buffer.append(torch.zeros((4, 11), dtype=torch.uint8))
        draft_pools = (dsa_pool(13, 17), dsa_pool(19, 23))

        group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=2,
            params=SimpleNamespace(mtp_draft_device_pools=draft_pools),
            components={ComponentType.FULL},
        )
        with self.assertRaisesRegex(ValueError, "draft KV"):
            resolve_hybrid_device_pool_group(
                kvcache=kvcache,
                page_size=2,
                params=SimpleNamespace(
                    mtp_draft_device_pools=(), sidecar_draft_device_pools=(object(),)
                ),
                components={ComponentType.FULL},
            )

        self.assertEqual(group.num_layers, 2)
        self.assertTrue(group.rank_replicated)
        self.assertEqual(set(group.entry_map), {PoolName.KV, PoolName.INDEXER})
        self.assertEqual(
            group.sources,
            {
                PoolName.KV: PoolName.KV,
                PoolName.INDEXER: PoolName.KV,
            },
        )
        _, sizes, offsets = group.entry_map[PoolName.KV].get_prepared_layer_range_meta(
            [0], 0
        )
        self.assertEqual(sizes, [[6, 26]])
        self.assertEqual(offsets, [[0, 16]])
        _, sizes, offsets = group.entry_map[
            PoolName.INDEXER
        ].get_prepared_layer_range_meta([0], 0)
        self.assertEqual(sizes, [[7, 17]])
        self.assertEqual(offsets, [[0, 18]])
        _, sizes, offsets = group.entry_map[
            PoolName.INDEXER
        ].get_prepared_layer_range_meta([0], 1)
        self.assertEqual(sizes, [[11, 23]])
        self.assertEqual(offsets, [[7, 35]])

    def test_draft_plan_packs_only_nextn_mtp(self):
        from sglang.srt.speculative import base_spec_worker as spec
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        draft = SimpleNamespace(
            token_to_kv_pool=object(),
            req_to_token_pool=None,
            model_config=SimpleNamespace(
                num_nextn_predict_layers=0,
                hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
            ),
        )
        target = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.EAGLE, req_to_token_pool=None
        )
        worker = SimpleNamespace(
            target_worker=SimpleNamespace(model_runner=target),
            _draft_model_runners=lambda: (draft,),
        )
        for linker_enabled, nextn_layers in (
            (False, 0),
            (True, 0),
            (False, 1),
            (True, 1),
        ):
            draft.model_config.num_nextn_predict_layers = nextn_layers
            with (
                self.subTest(linker=linker_enabled, nextn=nextn_layers),
                patch.object(
                    spec,
                    "get_memory",
                    return_value=SimpleNamespace(
                        enable_hierarchical_cache=not linker_enabled,
                        enable_unified_cache_external_linker=linker_enabled,
                    ),
                ),
            ):
                plan = spec.BaseSpecWorker._build_hicache_draft_plan(worker)
                self.assertEqual(
                    plan.mode,
                    spec.HiCacheDraftMode.PACKED
                    if nextn_layers
                    else spec.HiCacheDraftMode.SIDECAR,
                )
                self.assertEqual(plan.device_pools, (draft.token_to_kv_pool,))
                self.assertEqual(
                    target.mtp_draft_device_pools,
                    plan.device_pools if nextn_layers else (),
                )

    def test_mla_mamba_maps_kv_pages_and_state_slots(self):
        from sglang.srt.mem_cache.hybrid_cache import linker_pool_assembler
        from sglang.srt.mem_cache.memory_pool import (
            DSATokenToKVPool,
            HybridLinearKVPool,
            MHATokenToKVPool,
            MLATokenToKVPool,
        )

        kvcache = HybridLinearKVPool.__new__(HybridLinearKVPool)
        kvcache.start_layer = 0
        kvcache.full_attention_layer_id_mapping = {1: 0, 3: 1}
        kvcache.full_kv_pool = MLATokenToKVPool.__new__(MLATokenToKVPool)
        kvcache.full_kv_pool.kv_buffer = [
            torch.zeros((8, 1, 5), dtype=torch.uint8) for _ in range(2)
        ]
        mamba_cache = SimpleNamespace(
            temporal=torch.zeros((2, 4, 3), dtype=torch.uint8),
            conv=[torch.zeros((2, 4, 2), dtype=torch.uint8)],
        )
        draft = MHATokenToKVPool.__new__(MHATokenToKVPool)
        draft.page_size, draft.layer_num = 2, 1
        draft.k_scale_buffer = draft.native_k_scale_buffer = None
        draft.k_buffer = [torch.zeros((8, 3), dtype=torch.uint8)]
        draft.v_buffer = [torch.zeros((8, 3), dtype=torch.uint8)]
        params = SimpleNamespace(
            mtp_draft_device_pools=(),
            sidecar_draft_device_pools=(draft,),
            req_to_token_pool=SimpleNamespace(
                mamba_pool=SimpleNamespace(mamba_cache=mamba_cache, num_mamba_layers=2),
                mamba_map={0: 0, 2: 1},
                mamba_ckpt_pool=None,
                short_conv_pool=SimpleNamespace(enabled=False),
                ngram_pool=SimpleNamespace(enabled=False),
            ),
        )

        def resolve(dcp_size=1):
            with (
                patch.object(
                    linker_pool_assembler,
                    "get_memory",
                    return_value=SimpleNamespace(enable_unified_memory=False),
                ),
                patch.object(
                    linker_pool_assembler,
                    "get_parallel",
                    return_value=SimpleNamespace(
                        attn_dcp_size=dcp_size, attn_dcp_rank=1
                    ),
                ),
                patch.object(MLATokenToKVPool, "rows_are_dcp_collapsed", new=True),
            ):
                return resolve_hybrid_device_pool_group(
                    kvcache=kvcache,
                    page_size=2,
                    params=params,
                    components={ComponentType.FULL, ComponentType.MAMBA},
                )

        group = resolve()
        self.assertEqual(group.num_layers, 4)
        self.assertFalse(group.rank_replicated)
        kv, mamba = group.entry_map[PoolName.KV], group.entry_map[PoolName.MAMBA]
        self.assertEqual(kv.layer_mapping, {1: 0, 3: 1})
        self.assertEqual(mamba.layer_mapping, {0: 0, 2: 1})
        self.assertEqual(kv.prepare_locations(torch.tensor([2, 3])), [2])
        self.assertEqual(mamba.prepare_locations(torch.tensor([3])), [3])
        self.assertEqual(group.sources[PoolName.DRAFT], PoolName.KV)
        self.assertEqual(group.entry_map[PoolName.DRAFT].layer_mapping, {0: (0, 1)})

        dsa = DSATokenToKVPool.__new__(DSATokenToKVPool)
        dsa.page_size = dsa.index_page_size = 1
        dsa.kv_buffer = kvcache.full_kv_pool.kv_buffer
        dsa.index_key_cache = SimpleNamespace(
            buffer=[torch.zeros((4, 3), dtype=torch.uint8) for _ in range(2)]
        )
        kvcache.full_kv_pool = dsa
        group = resolve(dcp_size=2)
        for name in (PoolName.KV, PoolName.INDEXER):
            entry = group.entry_map[name]
            self.assertTrue(entry.dcp_sharded, name)
            locations = entry.prepare_locations(
                entry.translate_indices(torch.tensor([4, 5]))
            )
            self.assertEqual(locations, [2], name)
        from sglang.srt.environ import envs

        with envs.SGLANG_LINKER_MERGE_INDEXER.override(True):
            group = resolve(dcp_size=2)
        self.assertNotIn(PoolName.INDEXER, group.entry_map)
        merged = group.entry_map[PoolName.KV]
        self.assertEqual(merged.object_name, "ki")
        self.assertEqual(
            merged.prepare_locations(merged.translate_indices(torch.tensor([4, 5]))),
            [(2, 2)],
        )
        params.mtp_draft_device_pools = (object(),)
        with self.assertRaisesRegex(ValueError, "plain MLA"):
            resolve()
        params.mtp_draft_device_pools = ()
        kvcache.full_kv_pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
        with self.assertRaisesRegex(ValueError, "plain MLA"):
            resolve()

    def test_unsupported_strategy_fails_with_context(self):
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        kvcache = SWAKVPool.__new__(SWAKVPool)
        with self.assertRaisesRegex(
            ValueError,
            "does not support the direct external linker: _SwaStrategy",
        ):
            resolve_hybrid_device_pool_group(
                kvcache=kvcache,
                page_size=2,
                params=SimpleNamespace(),
                components={ComponentType.FULL, ComponentType.SWA},
            )



class FakeHelperStore:
    """Dict-backed store for the linker helper tests (runs in the helper)."""

    def __init__(self):
        self.objects = {}

    def batch_get_session_start(self, keys):
        return [0] * len(keys)

    def batch_get_session_end(self, keys):
        return [0] * len(keys)

    def batch_is_exist(self, keys):
        return [1 if key in self.objects else 0 for key in keys]

    def batch_put_from_multi_buffers(self, keys, ptrs, sizes):
        import ctypes

        for key, p, n in zip(keys, ptrs, sizes):
            self.objects[key] = b"".join(ctypes.string_at(a, b) for a, b in zip(p, n))
        return [0] * len(keys)

    def batch_get_into_multi_buffer_ranges_arrays(self, keys, ptrs, sizes, offsets, indptr):
        import ctypes

        out = []
        for i, key in enumerate(keys):
            obj = self.objects.get(key)
            if obj is None:
                out.append(-704)
                continue
            got = 0
            for j in range(int(indptr[i]), int(indptr[i + 1])):
                off, n = int(offsets[j]), int(sizes[j])
                ctypes.memmove(int(ptrs[j]), obj[off : off + n], n)
                got += n
            out.append(got)
        return out


def _merged_pool(make):
    """KV pages (2 layers) plus a per-page indexer row on layer 2, as one object."""
    from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import MergedPageEntry

    kv0, kv2, ix = make((8, 4)), make((8, 4)), make((4, 3))
    kv = DevicePoolEntry(
        name=PoolName.KV, indices_from_pool=PoolName.KV, device_pool=None,
        components=[[kv0, kv2]], layer_mapping={0: 0, 2: 1}, page_size=2, rows_are_pages=False,
    )
    indexer = DevicePoolEntry(
        name=PoolName.INDEXER, indices_from_pool=PoolName.KV, device_pool=None,
        components=[[ix]], layer_mapping={2: 0}, page_size=2, rows_are_pages=True,
    )
    return MergedPageEntry(kv, indexer, object_name="ki"), (kv0, kv2, ix)


class TestLinkerHelper(CustomTestCase):
    """SGLANG_LINKER_HELPER: the helper rebuilds keys and ranges from the pool
    tables and runs the store calls in its own process."""

    def test_helper_rebuilds_the_in_process_ranges_and_put_meta(self):
        from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
            vector_key_range_meta,
            vector_page_buffer_meta,
        )
        from sglang.srt.mem_cache.storage.mooncake_store.linker_helper import (
            HelperInit,
            HelperWorker,
            pool_table,
        )

        pool, _ = _merged_pool(lambda shape: torch.zeros(shape, dtype=torch.uint8))
        table = pool_table(PoolName.KV, pool, "_r0_ki", num_layers=3)
        indices = torch.tensor([2, 3, 6, 7])  # pages 1 and 3
        for coalesce in (False, True):
            worker = HelperWorker(
                HelperInit(tables={"kv": table}, prefix="tag", coalesce=coalesce), FakeHelperStore(), []
            )
            want = vector_key_range_meta(
                pool, pool.prepare_locations(indices), 3, coalesce=coalesce, arrays=True
            )
            got = worker.read_ranges("kv", indices.numpy())
            for field in ("ptrs", "sizes", "offsets", "indptr"):
                np.testing.assert_array_equal(getattr(got, field), getattr(want, field), field)
        ptrs, sizes = worker.put_meta("kv", indices.numpy())
        want_ptrs, want_sizes = vector_page_buffer_meta(pool, indices)
        self.assertEqual((ptrs, sizes), (want_ptrs, want_sizes))
        self.assertEqual(worker._keys("kv", ["h1", "h2"]), ["tag_h1_r0_ki", "tag_h2_r0_ki"])
        with self.assertRaises(ValueError):
            worker.read_ranges("kv", np.array([1, 2]))  # not page aligned

    def test_helper_process_round_trip_restores_the_bytes(self):
        from multiprocessing import shared_memory

        from sglang.srt.mem_cache.storage.mooncake_store.linker_helper import (
            HelperClient,
            HelperInit,
            pool_table,
        )

        shm = shared_memory.SharedMemory(create=True, size=4096)
        try:
            arena = torch.frombuffer(shm.buf, dtype=torch.uint8)
            cursor = [0]

            def make(shape):
                n = shape[0] * shape[1]
                view = arena[cursor[0] : cursor[0] + n].view(shape)
                cursor[0] += (n + 63) // 64 * 64
                return view

            pool, (kv0, kv2, ix) = _merged_pool(make)
            for i, t in enumerate((kv0, kv2, ix)):
                t.copy_(torch.randint(0, 255, t.shape, dtype=torch.uint8) + i)
            src = torch.tensor([2, 3, 6, 7])  # pages 1 and 3
            dst = torch.tensor([0, 1, 4, 5])  # pages 0 and 2
            dst_rows = ([0, 1, 4, 5], [0, 2])
            init = HelperInit(
                tables={"kv": pool_table(PoolName.KV, pool, "_r0_ki", num_layers=3)},
                prefix="tag",
                exports=[("shm", arena.data_ptr(), shm.size, shm.name)],
                store_factory=FakeHelperStore,
                threads=2,
            )
            client = HelperClient(init, start_timeout_s=120)
            try:
                results, info = client.submit("write", [("kv", ["p1", "p3"], src.numpy())]).result(60)
                self.assertEqual(results, {"kv": [True, True]})
                self.assertEqual(info["put_objects"], {"kv": 2})
                want = [kv0[[2, 3, 6, 7]].clone(), kv2[[2, 3, 6, 7]].clone(), ix[[1, 3]].clone()]
                for t, rows in ((kv0, dst_rows[0]), (kv2, dst_rows[0]), (ix, dst_rows[1])):
                    t[rows] = 0
                status, info = client.submit("read", [("kv", ["p1", "p3"], dst.numpy())]).result(60)
                self.assertIs(status, True)
                self.assertEqual(info["objects"], 2)
                self.assertTrue(torch.equal(kv0[dst_rows[0]], want[0]))
                self.assertTrue(torch.equal(kv2[dst_rows[0]], want[1]))
                self.assertTrue(torch.equal(ix[dst_rows[1]], want[2]))
                # A second write of the same pages: the read's keys skip the exist query and the put.
                _, info = client.submit("write", [("kv", ["p1", "p3"], src.numpy())]).result(60)
                self.assertEqual((info["put_objects"], info["exist_skipped"]), ({}, 2))
                # A new key namespace (cold reset): the old objects are not found.
                client.submit("prefix", "tag-e1").result(60)
                status, _ = client.submit("read", [("kv", ["p1"], dst[:2].numpy())]).result(60)
                self.assertIn("-704", status)
                self.assertEqual(client.submit("stats").result(60)["reads"], 1)
            finally:
                client.close()
            self.assertFalse(client.proc.is_alive())
        finally:
            shm.close()
            shm.unlink()



class TestLinkerStaging(CustomTestCase):
    """SGLANG_LINKER_EARLY_READS: pinned-host staging of looked-up objects, the
    scatter into a request's device rows, and the per-stage entry lifecycle."""

    def test_scatter_plan_restores_the_device_rows_of_staged_objects(self):
        import ctypes

        from sglang.srt.mem_cache.storage.mooncake_store.linker_staging import (
            StagingSlab,
            cpu_scatter,
            scatter_plan,
        )

        pool, (kv0, kv2, ix) = _merged_pool(lambda shape: torch.zeros(shape, dtype=torch.uint8))
        for i, t in enumerate((kv0, kv2, ix)):
            t.copy_(torch.randint(0, 255, t.shape, dtype=torch.uint8) + i)
        table = pool.key_range_table(3)
        obj_bytes = int((table[:, 3] + table[:, 4]).max())
        src_rows = pool.key_range_rows(pool.prepare_locations(torch.tensor([2, 3, 6, 7])))
        dst_idx = torch.tensor([0, 1, 4, 5])
        dst_rows = pool.key_range_rows(pool.prepare_locations(dst_idx))
        want = [kv0[[2, 3, 6, 7]].clone(), kv2[[2, 3, 6, 7]].clone(), ix[[1, 3]].clone()]
        buf = np.zeros(StagingSlab.size_of({"kv": (obj_bytes, 4)}), dtype=np.uint8)
        slab = StagingSlab({"kv": (obj_bytes, 4)}, buf, buf.ctypes.data)
        slots = slab.alloc({"kv": 2})["kv"]
        ptrs = slab.ptrs("kv", slots)
        # The store object of each page, as the per-key read lays it out: item j at its offset.
        _, base, stride, size, offset = table.T
        for row, ptr in zip(src_rows, ptrs):
            for j in range(len(table)):
                ctypes.memmove(int(ptr + offset[j]), int(base[j] + row[table[j, 0]] * stride[j]), int(size[j]))
        for t, rows in ((kv0, [0, 1, 4, 5]), (kv2, [0, 1, 4, 5]), (ix, [0, 2])):
            t[rows] = 0
        cpu_scatter(*scatter_plan(table, dst_rows, ptrs))
        self.assertTrue(torch.equal(kv0[[0, 1, 4, 5]], want[0]))
        self.assertTrue(torch.equal(kv2[[0, 1, 4, 5]], want[1]))
        self.assertTrue(torch.equal(ix[[0, 2]], want[2]))
        self.assertIsNone(slab.alloc({"kv": 3}))  # all or nothing: 2 of 4 slots left
        slab.release({"kv": slots})
        self.assertEqual(len(slab.free["kv"]), 4)

    def test_slab_pools_give_every_stage_the_same_page_slots_within_the_cap(self):
        from sglang.srt.mem_cache.storage.mooncake_store.linker_staging import slab_pools

        gb = 2**30
        # Hecate ab4 object sizes: PP0-5 KV 135,936 B, PP7 KV 181,248 B; states 58.6 / 39.1 MB.
        pp0 = slab_pools({"kv": 135936, "mamba": 58613760}, "mamba", 40000, 32, 9 * gb)
        pp7 = slab_pools({"kv": 181248, "mamba": 39075840}, "mamba", 40000, 32, 9 * gb)
        self.assertEqual((pp0["kv"][1], pp7["kv"][1]), (40000, 40000))
        self.assertEqual((pp0["mamba"][1], pp7["mamba"][1]), (32, 32))
        # A tighter cap shrinks the page slots to fit; the state slots stay.
        tight = slab_pools({"kv": 181248, "mamba": 39075840}, "mamba", 40000, 32, 6 * gb)
        self.assertEqual(tight["mamba"][1], 32)
        self.assertEqual(tight["kv"][1], (6 * gb - 32 * 39075840) // 181248)
        self.assertLessEqual(tight["kv"][1] * 181248 + 32 * 39075840, 6 * gb)

    def test_early_reads_vote_once_never_block_and_release_every_slot(self):
        from concurrent.futures import Future

        from sglang.srt.mem_cache.storage.mooncake_store.linker_staging import (
            EarlyMessage,
            EarlyReads,
            StagingSlab,
        )

        buf = np.zeros(StagingSlab.size_of({"kv": (8, 4)}), dtype=np.uint8)
        slab = StagingSlab({"kv": (8, 4)}, buf, buf.ctypes.data)
        reads = []

        def read(keys, ptrs):
            reads.append(Future())
            return reads[-1]

        early = EarlyReads(slab, lambda hashes: {"kv": ([f"k{h}" for h in hashes], 8)}, read)
        early.start(EarlyMessage(0, 1, "a", ["h1", "h2"]))
        early.start(EarlyMessage(0, 2, "b", ["h3", "h4"]))
        early.start(EarlyMessage(0, 3, "c", ["h5"]))  # slab full: not staged, no read
        self.assertEqual((len(reads), early.counts["full"]), (2, 1))
        reads[0].set_result(True)
        self.assertEqual(early.take_landed(), [(0, 1)])
        self.assertEqual(early.take_landed(), [])  # voted once
        self.assertIsNone(early.take(0, 1, 3))  # a voted read asked for other pages: a counted near miss
        self.assertEqual(early.counts["voted_unused"], 1)
        # A load of the same request under another key reads directly: counted.
        self.assertIsNone(early.take(0, 99, 2, rid="a"))
        self.assertEqual(early.counts["voted_direct"], 1)
        entry = early.take(0, 1, 2)
        self.assertIsNotNone(entry)
        # Cycle 2 is still reading when its load applies: today's read is used, and
        # the staged read, landing later, is released and never voted.
        self.assertIsNone(early.take(0, 2, 2))
        reads[1].set_result(True)
        self.assertEqual(early.take_landed(), [])
        early.release(entry)
        self.assertEqual(len(slab.free["kv"]), 4)
        # A failed read is not voted; a reset drops older epochs.
        early.start(EarlyMessage(0, 4, "d", ["h6"]))
        reads[2].set_result(False)
        self.assertEqual((early.take_landed(), early.counts["failed"]), ([], 1))
        early.start(EarlyMessage(0, 5, "e", ["h7"]))
        reads[3].set_result(True)
        early.reset(1)
        self.assertEqual((early.take_landed(), len(slab.free["kv"])), ([], 4))
        early.start(EarlyMessage(0, 6, "f", ["h8"]))  # an older epoch's message is ignored
        self.assertEqual(len(reads), 4)

    def test_coalesce_ranges_merges_only_ranges_contiguous_on_both_sides(self):
        from sglang.srt.mem_cache.storage.mooncake_store.linker_staging import coalesce_ranges, cpu_scatter

        src_buf = np.random.randint(0, 255, 64, dtype=np.uint8)
        dst_buf = np.zeros(64, dtype=np.uint8)
        s, d = src_buf.ctypes.data, dst_buf.ctypes.data
        # 0-4 and 4-8 join; 8-12 is contiguous in dst only; 20-24 in src only.
        dst = np.array([d, d + 4, d + 8, d + 20, d + 40])
        src = np.array([s, s + 4, s + 16, s + 20, s + 24])
        sizes = np.array([4, 4, 4, 4, 4])
        cd, cs, cn = coalesce_ranges(dst, src, sizes)
        self.assertEqual((cd - d).tolist(), [0, 8, 20, 40])
        self.assertEqual((cs - s).tolist(), [0, 16, 20, 24])
        self.assertEqual(cn.tolist(), [8, 4, 4, 4])
        cpu_scatter(cd, cs, cn)
        want = np.zeros(64, dtype=np.uint8)
        for a, b, n in zip(dst - d, src - s, sizes):
            want[a : a + n] = src_buf[b : b + n]
        self.assertTrue(np.array_equal(dst_buf, want))

    def test_copy_engine_entries_per_request_at_ab4_sizes(self):
        """Entries one staged load hands cudaMemcpyBatchAsync on the K3.1 PP8 rig,
        and the driver time at GLM's measured 0.34 us per entry."""
        from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
            MergedPageEntry,
            coalesce_key_range_table,
        )
        from sglang.srt.mem_cache.storage.mooncake_store.linker_staging import coalesce_ranges, scatter_plan

        page, slab_pages = 64, 64
        # ab4 PP0 cached pages per load (5,082 loads): p50 996, p99 5599. Every stage loads the
        # same pages. KDA state: 9 layers (PP7: 6) x (temporal + conv) items, one object per load.
        loads = {"p50": 996, "p99": 5599}
        stages = {"PP0-5": (3, 18), "PP6": (2, 18), "PP7": (4, 12)}
        report = {}
        for stage, (attn_layers, state_items) in stages.items():
            kvs = [torch.zeros((page * slab_pages, 576), dtype=torch.uint8) for _ in range(attn_layers)]
            ixs = [torch.zeros((slab_pages, page * 132), dtype=torch.uint8) for _ in range(attn_layers)]
            layers = {i: i for i in range(attn_layers)}
            kv = DevicePoolEntry(
                name=PoolName.KV, indices_from_pool=PoolName.KV, device_pool=None,
                components=[kvs], layer_mapping=layers, page_size=page, rows_are_pages=False,
            )
            indexer = DevicePoolEntry(
                name=PoolName.INDEXER, indices_from_pool=PoolName.KV, device_pool=None,
                components=[ixs], layer_mapping=layers, page_size=page, rows_are_pages=True,
            )
            pool = MergedPageEntry(kv, indexer, object_name="ki")
            table = coalesce_key_range_table(pool.key_range_table(attn_layers))
            obj_bytes = int((table[:, 3] + table[:, 4]).max())
            self.assertEqual(obj_bytes, attn_layers * 45312)
            for name, pages in loads.items():
                # Consecutive device pages and slots: the layout most favourable to coalescing.
                tokens = torch.arange(pages * page) % (slab_pages * page)
                rows = pool.key_range_rows(pool.prepare_locations(tokens))
                ptrs = 4096 + np.arange(pages, dtype=np.int64) * obj_bytes
                entries = len(coalesce_ranges(*scatter_plan(table, rows, ptrs))[0]) + state_items
                # Objects are page-major in staging, device rows layer-major: nothing joins.
                self.assertEqual(entries, pages * 2 * attn_layers + state_items)
                report[(stage, name)] = (entries, entries * 0.34e-3)
        for (stage, name), (entries, ms) in report.items():
            print(f"{stage} {name}: {entries} entries per load, {ms:.1f} ms of driver time")
        self.assertLess(max(ms for _, ms in report.values()), 20.0)

    def test_staged_copies_use_torchs_libcudart_by_path_only(self):
        from sglang.srt.mem_cache.storage.mooncake_store.linker_staging_kernel import torch_cudart_path

        sp = "/opt/sglang/lib/python3.12/site-packages"
        torch_rt = f"{sp}/nvidia/cu13/lib/libcudart.so.13"
        vision_rt = f"{sp}/torchvision.libs/libcudart.a6a4ec3c.so.13"  # mapped beside it in every process
        self.assertEqual(torch_cudart_path([vision_rt, torch_rt], sp), torch_rt)
        self.assertEqual(torch_cudart_path([f"{sp}/torch/lib/../../nvidia/cu13/lib/libcudart.so.13", vision_rt], sp), torch_rt)
        # Not torch's, or ambiguous: no runtime, so the kernel scatter runs.
        self.assertIsNone(torch_cudart_path([vision_rt, "/usr/local/cuda/lib64/libcudart.so.13"], sp))
        self.assertIsNone(torch_cudart_path([torch_rt, f"{sp}/nvidia/cu12/lib/libcudart.so.12"], sp))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_memcpy_batch_scatters_pinned_ranges_into_device_rows(self):
        import time

        from sglang.srt.mem_cache.storage.mooncake_store.linker_staging import copy_engine_scatter

        n, size = 33612, 256  # PP0's p99 load
        host = torch.randint(0, 255, (n * size,), dtype=torch.uint8).pin_memory()
        dev = torch.zeros(2 * n * size, dtype=torch.uint8, device="cuda")
        perm = torch.randperm(2 * n)[:n].numpy().astype(np.int64)
        dst = dev.data_ptr() + perm * size
        src = host.data_ptr() + np.arange(n, dtype=np.int64) * size
        sizes = np.full(n, size, dtype=np.int64)
        stream = torch.cuda.Stream()
        copy_engine_scatter([(dst[:16], src[:16], sizes[:16])], stream).synchronize()  # warm
        t = time.perf_counter()
        done = copy_engine_scatter([(dst, src, sizes)], stream)
        call_ms = (time.perf_counter() - t) * 1e3
        done.synchronize()
        got = dev.view(2 * n, size)[torch.from_numpy(perm).cuda()].cpu()
        self.assertTrue(torch.equal(got, host.view(n, size)))
        print(f"memcpy_batch: {n} entries, {call_ms:.1f} ms on the calling thread ({call_ms / n * 1e3:.2f} us/entry)")

    def test_early_channel_delivers_from_pp0_to_a_stage(self):
        import threading
        import time

        from sglang.srt.mem_cache.storage.mooncake_store.linker_staging import EarlyChannel, EarlyMessage

        got, seen = [], threading.Event()
        info = {}

        class _Group:
            def __init__(self, rank):
                self.rank = rank

            def broadcast_object(self, obj, src=0):
                if self.rank == 0:
                    info["addr"] = obj
                return info["addr"]

        with patch("sglang.srt.utils.network.get_local_ip_auto", return_value="127.0.0.1"):
            pp0 = EarlyChannel(0, _Group(0))
            pp1 = EarlyChannel(1, _Group(1), on_message=lambda m: (got.append(m), seen.set()))
        try:
            msg = EarlyMessage(3, 7, "rid", ["h1", "h2"])
            for _ in range(50):  # a SUB socket joins asynchronously
                pp0.send(msg)
                if seen.wait(0.1):
                    break
            self.assertEqual(got[0], msg)
        finally:
            pp0.close()
            pp1.close()


if __name__ == "__main__":
    unittest.main()
