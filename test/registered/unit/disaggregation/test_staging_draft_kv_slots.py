"""Staging slot ids stay aligned once a draft KV pool is registered.

The staging gather writes every k_buffer and then every v_buffer, while
kv_data_ptrs (and therefore kv_layer_ids) is ordered
[K target, V target, K draft, V draft]. Labelling slots with kv_layer_ids
silently pairs a layer's KV with another layer's staging slot as soon as a
draft pool exists.
"""

import unittest

from sglang.srt.disaggregation.utils import (
    build_staging_slot_metadata,
    build_staging_entry_metadata,
    is_mla_backend,
    build_transfer_entry_pairs,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
import torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Pool(MHATokenToKVPool):
    def __init__(self, tag, layer_ids):
        self.k_buffer = [f"{tag}K{i}" for i in layer_ids]
        self.v_buffer = [f"{tag}V{i}" for i in layer_ids]


class _Wrapper(HybridLinearKVPool):
    def __init__(self, inner):
        self.full_kv_pool = inner


def _kv_layer_ids(target_ids, draft_ids):
    """kv_data_ptrs order: K target, V target, K draft, V draft."""
    return list(target_ids) + list(target_ids) + list(draft_ids) + list(draft_ids)


class TestStagingDraftKvSlots(CustomTestCase):
    def test_draft_slots_follow_gather_order(self):
        target, draft = [87, 91], [92]
        k_buffers, v_buffers, slot_ids = build_staging_slot_metadata(
            kv_layer_ids=_kv_layer_ids(target, draft),
            num_draft_entries=2,
            kv_pool=_Pool("t", target),
            draft_kv_pool=_Pool("d", draft),
        )
        self.assertEqual(k_buffers, ["tK87", "tK91", "dK92"])
        self.assertEqual(v_buffers, ["tV87", "tV91", "dV92"])
        self.assertEqual(slot_ids, [87, 91, 92, 87, 91, 92])
        self.assertNotEqual(slot_ids, _kv_layer_ids(target, draft))

    def test_without_draft_matches_kv_layer_ids(self):
        # The two orders coincide with no draft pool, so every deployment that
        # predates draft KV must keep its exact slot labelling.
        target = [3, 7]
        _, _, slot_ids = build_staging_slot_metadata(
            kv_layer_ids=_kv_layer_ids(target, []),
            num_draft_entries=0,
            kv_pool=_Pool("t", target),
            draft_kv_pool=None,
        )
        self.assertEqual(slot_ids, _kv_layer_ids(target, []))

    def test_pp_stage_pairs_against_full_decode(self):
        # A prefill stage holds a slice of the layers while decode holds them
        # all, so the ids -- not the positions -- have to drive the pairing.
        src = build_staging_slot_metadata(
            kv_layer_ids=_kv_layer_ids([87, 91], [92]),
            num_draft_entries=2,
            kv_pool=_Pool("t", [87, 91]),
            draft_kv_pool=_Pool("d", [92]),
        )[2]
        decode_target = [3, 7, 11, 87, 91]
        dst = build_staging_slot_metadata(
            kv_layer_ids=_kv_layer_ids(decode_target, [92]),
            num_draft_entries=2,
            kv_pool=_Pool("t", decode_target),
            draft_kv_pool=_Pool("d", [92]),
        )[2]
        pairs = build_transfer_entry_pairs(src, dst, len(src), len(dst))
        self.assertEqual(len(pairs), len(src))
        for i, j in pairs:
            self.assertEqual(src[i], dst[j])
        self.assertEqual(len({j for _, j in pairs}), len(pairs))

    def test_hybrid_wrapper_pools_are_unwrapped(self):
        # A hybrid draft pool that is left wrapped looks exactly like a draft
        # pool with no buffers, which drops draft KV out of staging.
        target, draft = [87, 91], [92]
        k_buffers, _, slot_ids = build_staging_slot_metadata(
            kv_layer_ids=_kv_layer_ids(target, draft),
            num_draft_entries=2,
            kv_pool=_Wrapper(_Pool("t", target)),
            draft_kv_pool=_Wrapper(_Pool("d", draft)),
        )
        self.assertEqual(k_buffers, ["tK87", "tK91", "dK92"])
        self.assertEqual(slot_ids, [87, 91, 92, 87, 91, 92])

    def test_undescribable_draft_still_yields_target_buffers(self):
        # Returning nothing here left the caller skipping set_kv_buffer_tensors
        # entirely, and staging then came up with no buffers at all.
        class _NoBuffers:
            pass

        k_buffers, v_buffers, slot_ids = build_staging_slot_metadata(
            kv_layer_ids=_kv_layer_ids([87], [92]),
            num_draft_entries=2,
            kv_pool=_Pool("t", [87]),
            draft_kv_pool=_NoBuffers(),
        )
        self.assertEqual(k_buffers, ["tK87"])
        self.assertEqual(v_buffers, ["tV87"])
        self.assertEqual(slot_ids, [])

    def test_pool_without_contiguous_tensors_is_declined(self):
        # MLA pools have no k_buffer/v_buffer to stage; the caller relies on None
        # to skip the registration rather than register empty lists.
        class _NoBuffers:
            pass

        self.assertIsNone(
            build_staging_slot_metadata(
                kv_layer_ids=[],
                num_draft_entries=0,
                kv_pool=_NoBuffers(),
                draft_kv_pool=None,
            )
        )


class TestStagingV2PoolMetadata(CustomTestCase):
    @staticmethod
    def mla(ids, widths, hybrid=True):
        pool = object.__new__(MLATokenToKVPool)
        pool.page_size, pool.layer_num, pool.start_layer = (
            4,
            len(ids),
            (ids[0] if ids else 0),
        )
        pool.kv_buffer = [
            torch.zeros(32, 1, width, dtype=torch.bfloat16) for width in widths
        ]
        if not hybrid:
            return pool
        wrapper = object.__new__(HybridLinearKVPool)
        wrapper.full_kv_pool = pool
        wrapper.use_mla = True
        wrapper.full_attention_layer_id_mapping = {
            layer: i for i, layer in enumerate(ids)
        }
        return wrapper

    def test_sparse_hybrid_and_plain_global_layers(self):
        for pool, ids in [
            (self.mla([3, 19, 47], [7, 9, 13]), [3, 19, 47]),
            (self.mla([17, 18], [11, 11], False), [17, 18]),
            (self.mla([], []), []),
        ]:
            self.assertTrue(is_mla_backend(pool))
            buffers, entries = build_staging_entry_metadata(
                kv_pool=pool, draft_kv_pool=None, num_hidden_layers=93
            )
            self.assertEqual([entry.global_layer_id for entry in entries], ids)
            self.assertEqual(
                [entry.copy_width_bytes for entry in entries],
                [tensor.shape[-1] * 2 for tensor in buffers],
            )
            self.assertEqual([entry.index for entry in entries], list(range(len(ids))))
            self.assertTrue(all(entry.kind == "mla_latent" for entry in entries))

    def test_mha_draft_uses_independent_widths_and_reserved_band(self):
        draft = object.__new__(MHATokenToKVPool)
        draft.page_size, draft.size, draft.layer_num, draft.start_layer = 4, 28, 1, 0
        draft.store_dtype, draft.use_hnd = torch.bfloat16, False
        draft.k_buffer = [torch.zeros(32, 4, 3, dtype=torch.bfloat16)]
        draft.v_buffer = [torch.zeros(32, 4, 5, dtype=torch.bfloat16)]
        draft._kv_buffer_descs = draft._build_kv_buffer_descs()
        _, entries = build_staging_entry_metadata(
            kv_pool=self.mla([47], [11]),
            draft_kv_pool=draft,
            num_hidden_layers=93,
            draft_total_heads=8,
        )
        self.assertEqual(
            [
                (e.global_layer_id, e.kind, e.copy_width_bytes, e.total_heads)
                for e in entries
            ],
            [(47, "mla_latent", 22, 0), (93, "mha_k", 24, 8), (93, "mha_v", 40, 8)],
        )

    def test_mla_draft_and_unsupported_storage(self):
        _, entries = build_staging_entry_metadata(
            kv_pool=self.mla([3], [11]),
            draft_kv_pool=self.mla([0], [7], False),
            num_hidden_layers=93,
        )
        self.assertEqual(entries[-1].key, ("draft", 93, "mla_latent"))
        pool = self.mla([0], [7], False)
        pool.kv_buffer[0] = pool.kv_buffer[0].float()
        with self.assertRaisesRegex(ValueError, "BF16/FP16"):
            build_staging_entry_metadata(
                kv_pool=pool, draft_kv_pool=None, num_hidden_layers=93
            )

        class Quantized(MLATokenToKVPool):
            pass

        with self.assertRaisesRegex(ValueError, "Unsupported"):
            build_staging_entry_metadata(
                kv_pool=object.__new__(Quantized),
                draft_kv_pool=None,
                num_hidden_layers=93,
            )


if __name__ == "__main__":
    unittest.main()
