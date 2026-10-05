"""Unit tests for full-attention KV transfer with prefill pp_size > 1 on
hybrid-linear models (HybridLinearKVPool)."""

import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.ascend.conn import AscendKVManager
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.prefill import _transfer_start_layer
from sglang.srt.disaggregation.utils import (
    build_kv_layer_ids,
    build_transfer_entry_pairs,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.runtime_context import get_memory, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _full_attention_ids(*, num_layers: int, interval: int) -> list:
    return [i for i in range(num_layers) if i % interval == interval - 1]


def _hybrid_pool(*, start_layer: int) -> HybridLinearKVPool:
    pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
    pool.start_layer = start_layer
    return pool


class TestTransferStartLayer(CustomTestCase):
    """Bug regression: with prefill pp_size=2 on a 60-layer hybrid-linear model
    (full_attention_interval=4), stage 1's pool.start_layer is 30 — a global
    layer index counting linear layers. The decode peer's KV pointer list is
    dense over the 15 full-attention layers only, so slicing dst[30:38] yielded
    [] and an IndexError in mooncake send_kvcache_slice. The transfer offset
    must be the count of full-attention layers before the stage boundary."""

    def test_hybrid_stage1_translates_to_full_attention_offset(self):
        cfg = SimpleNamespace(
            full_attention_layer_ids=_full_attention_ids(num_layers=60, interval=4)
        )
        self.assertEqual(
            _transfer_start_layer(
                pool=_hybrid_pool(start_layer=30), hf_text_config=cfg
            ),
            7,
        )

    def test_hybrid_stage0_is_zero(self):
        cfg = SimpleNamespace(
            full_attention_layer_ids=_full_attention_ids(num_layers=60, interval=4)
        )
        self.assertEqual(
            _transfer_start_layer(pool=_hybrid_pool(start_layer=0), hf_text_config=cfg),
            0,
        )

    def test_non_hybrid_pool_keeps_global_start_layer(self):
        cfg = SimpleNamespace(full_attention_layer_ids=[])
        self.assertEqual(
            _transfer_start_layer(
                pool=SimpleNamespace(start_layer=30), hf_text_config=cfg
            ),
            30,
        )


class _RecordingKVManager:
    get_mha_kv_ptrs_with_pp = CommonKVManager.get_mha_kv_ptrs_with_pp
    get_mla_kv_ptrs_with_pp = CommonKVManager.get_mla_kv_ptrs_with_pp

    def __init__(self, *, prefill_start_layer: int, pp_size: int):
        self.is_mla_backend = False
        self.is_hybrid_mla_backend = False
        self.enable_custom_mem_pool = False
        self.max_transfer_batch_indices = 0
        self.pp_size = pp_size
        self.kv_args = SimpleNamespace(prefill_start_layer=prefill_start_layer)
        self.blocks = []

    def _transfer_data(self, mooncake_session_id, transfer_blocks):
        self.blocks.extend(transfer_blocks)
        return 0


class TestHybridSendUsesLayerIdPairing(CustomTestCase):
    """Bug regression: a hybrid-linear (non-MLA-flagged) backend fell into the
    positional MHA slicing path of _send_kvcache_generic even when both peers
    published layer ids. For a stage with F full-attention layers against a
    decode peer with N (F < N, F not dividing N), the draft-KV modulo heuristic
    silently placed the V block at F * (N // F) instead of N — wrong layers
    transferred, no error. With layer ids published on both sides the pairing
    must be exact."""

    def _run_case(
        self, *, model_full_ids: list, stage_full_ids: list, start_offset: int
    ):
        num_stage = len(stage_full_ids)
        num_model = len(model_full_ids)
        src_ptrs = [1000 + i for i in range(2 * num_stage)]
        dst_ptrs = [2000 + i for i in range(2 * num_model)]
        item_lens = [10 + i for i in range(2 * num_stage)]
        manager = _RecordingKVManager(prefill_start_layer=start_offset, pp_size=2)
        rc = MooncakeKVManager._send_kvcache_generic(
            manager,
            mooncake_session_id="session",
            src_data_ptrs=src_ptrs,
            dst_data_ptrs=dst_ptrs,
            item_lens=item_lens,
            prefill_data_indices=np.array([0], dtype=np.int32),
            dst_data_indices=np.array([0], dtype=np.int32),
            executor=None,
            src_layer_ids=stage_full_ids * 2,
            dst_layer_ids=model_full_ids * 2,
        )
        self.assertEqual(rc, 0)
        expected = [
            (src_ptrs[i], dst_ptrs[start_offset + i], item_lens[i])
            for i in range(num_stage)
        ] + [
            (
                src_ptrs[num_stage + i],
                dst_ptrs[num_model + start_offset + i],
                item_lens[num_stage + i],
            )
            for i in range(num_stage)
        ]
        self.assertEqual(manager.blocks, expected)

    def test_stage1_f8_of_n15(self):
        ids = _full_attention_ids(num_layers=60, interval=4)
        self._run_case(model_full_ids=ids, stage_full_ids=ids[7:], start_offset=7)

    def test_stage0_f7_of_n15(self):
        ids = _full_attention_ids(num_layers=60, interval=4)
        self._run_case(model_full_ids=ids, stage_full_ids=ids[:7], start_offset=0)

    def test_f5_of_n12(self):
        ids = _full_attention_ids(num_layers=48, interval=4)
        self._run_case(model_full_ids=ids, stage_full_ids=ids[:5], start_offset=0)


class TestSingleRegionSWATransfer(CustomTestCase):
    def test_one_region_full_generates_transfer_block(self):
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.addCleanup(reset_context)
        manager = _RecordingKVManager(prefill_start_layer=0, pp_size=1)
        manager.kv_args.kv_data_ptrs = [1000]
        manager.kv_args.kv_item_lens = [64]
        manager.kv_args.kv_layer_ids = []
        manager._validate_envelope_kv_layout = (
            MooncakeKVManager._validate_envelope_kv_layout.__get__(manager)
        )
        manager._send_kvcache_generic = MooncakeKVManager._send_kvcache_generic.__get__(
            manager
        )
        with get_memory().override(enable_unified_memory=True):
            rc = MooncakeKVManager.send_kvcache(
                manager,
                mooncake_session_id="session",
                prefill_kv_indices=np.array([3, 4], dtype=np.int32),
                dst_kv_ptrs=[2000],
                dst_kv_indices=np.array([7, 8], dtype=np.int32),
                dst_kv_item_len=64,
                executor=None,
            )
        self.assertEqual(rc, 0)
        self.assertEqual(manager.blocks, [(1192, 2448, 128)])

    def test_one_region_swa_generates_transfer_block(self):
        manager = _RecordingKVManager(prefill_start_layer=0, pp_size=1)
        rc = MooncakeKVManager._send_kvcache_generic(
            manager,
            mooncake_session_id="session",
            src_data_ptrs=[1000],
            dst_data_ptrs=[2000],
            item_lens=[64],
            prefill_data_indices=np.array([3, 4], dtype=np.int32),
            dst_data_indices=np.array([7, 8], dtype=np.int32),
            executor=None,
            state_type=StateType.SWA,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(manager.blocks, [(1000 + 3 * 64, 2000 + 7 * 64, 2 * 64)])


class _RecordingAscendManager:
    def __init__(self):
        self.is_hybrid_mla_backend = True
        self.pp_size = 2
        self.kv_args = SimpleNamespace(
            kv_data_ptrs=[101, 102, 103, 104],
            kv_item_lens=[11, 12, 13, 14],
            kv_layer_ids=[31, 35, 31, 35],
        )
        self.generic_call = None

    def _validate_envelope_kv_layout(self, *args):
        pass

    def _send_kvcache_generic(self, **kwargs):
        self.generic_call = kwargs
        return 7


class TestAscendHybridPpDispatch(CustomTestCase):
    def test_hybrid_pp_uses_layer_id_pairing_path(self):
        manager = _RecordingAscendManager()
        dst_layer_ids = [3, 7, 31, 35, 3, 7, 31, 35]
        rc = AscendKVManager.send_kvcache(
            manager,
            mooncake_session_id="session",
            prefill_kv_indices=np.array([1], dtype=np.int32),
            dst_kv_ptrs=list(range(8)),
            dst_kv_indices=np.array([2], dtype=np.int32),
            executor=None,
            dst_layer_ids=dst_layer_ids,
        )

        self.assertEqual(rc, 7)
        self.assertEqual(manager.generic_call["src_layer_ids"], [31, 35, 31, 35])
        self.assertEqual(manager.generic_call["dst_layer_ids"], dst_layer_ids)


class TestMambaSlotTransfer(CustomTestCase):
    def test_stage1_uses_paired_slot_sizes_and_offsets(self):
        manager = _RecordingKVManager(prefill_start_layer=1, pp_size=2)
        MooncakeKVManager._send_mamba_state(
            manager,
            req=SimpleNamespace(mooncake_session_id="session"),
            prefill_mamba_index=[2],
            src_state_data_ptrs=[1000],
            src_state_item_lens=[16],
            dst_state_data_ptrs=[2000, 3000],
            dst_mamba_index=[3],
            src_layer_ids=[7],
            dst_layer_ids=[3, 7],
            dst_state_item_lens=[8, 16],
        )
        self.assertEqual(manager.blocks, [(1032, 3048, 16)])

    def test_slot_size_mismatch_rejected_before_transfer(self):
        manager = _RecordingKVManager(prefill_start_layer=0, pp_size=1)
        with self.assertRaisesRegex(RuntimeError, "Mamba slot size mismatch"):
            MooncakeKVManager._send_mamba_state(
                manager,
                req=SimpleNamespace(mooncake_session_id="session"),
                prefill_mamba_index=[1],
                src_state_data_ptrs=[1000],
                src_state_item_lens=[16],
                dst_state_data_ptrs=[2000],
                dst_mamba_index=[1],
                dst_state_item_lens=[32],
            )
        self.assertEqual(manager.blocks, [])


class TestGetMhaKvPtrsWithPp(CustomTestCase):
    """Derived property: the modulo heuristic in get_mha_kv_ptrs_with_pp exists
    for the decode-has-draft-KV layout [K_main, V_main, draft_K, draft_V]. Pin
    that geometry (15 main + 1 draft layer) so a future rewrite of the
    heuristic (e.g. to fix the plain-MHA pp>1 F-not-dividing-N case) keeps the
    draft case intact."""

    def test_draft_kv_geometry_selects_main_v_block(self):
        manager = SimpleNamespace(kv_args=SimpleNamespace(prefill_start_layer=0))
        src_kv_ptrs = list(range(30))
        dst_kv_ptrs = list(range(100, 132))
        src_k, src_v, dst_k, dst_v, num_layers = (
            CommonKVManager.get_mha_kv_ptrs_with_pp(manager, src_kv_ptrs, dst_kv_ptrs)
        )
        self.assertEqual(src_k, src_kv_ptrs[:15])
        self.assertEqual(src_v, src_kv_ptrs[15:])
        self.assertEqual(dst_k, dst_kv_ptrs[:15])
        self.assertEqual(dst_v, dst_kv_ptrs[15:30])
        self.assertEqual(num_layers, 15)


class TestBuildTransferEntryPairsDuplicateIds(CustomTestCase):
    """Derived property: layer ids repeat across the K and V tensor groups, so
    pairing must consume dst occurrences in order (K with K, V with V) rather
    than by plain id lookup."""

    def test_k_then_v_occurrence_ordering(self):
        pairs = build_transfer_entry_pairs(
            src_layer_ids=[3, 7, 3, 7],
            dst_layer_ids=[3, 7, 11, 3, 7, 11],
            n_src=4,
            n_dst=6,
            allow_positional_fallback=False,
        )
        self.assertEqual(pairs, [(0, 0), (1, 1), (2, 3), (3, 4)])


def _hybrid_pool_with_ids(*, layer_ids: list) -> HybridLinearKVPool:
    pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
    pool.full_attention_layer_id_mapping = layer_ids
    pool.use_mla = False
    return pool


class TestBuildKvLayerIds(CustomTestCase):
    """Bug regression: enabling EAGLE appended draft KV buffers to kv_data_ptrs
    while kv_layer_ids described only the target's entries, so the ids were
    suppressed entirely and the transfer fell back to positional slicing. Under
    prefill pp_size > 1 that slices the wrong layers -- prefill pp=2 + EAGLE
    produced garbled decode output while pp=1 + EAGLE did not."""

    def _stage1_ids(self) -> list:
        full = _full_attention_ids(num_layers=60, interval=4)
        return [lid for lid in full if lid >= 30]

    def test_draft_entries_get_a_reserved_band_above_the_target_range(self):
        """A draft pool that only reports a layer count, not ids."""
        ids = build_kv_layer_ids(
            token_to_kv_pool=_hybrid_pool_with_ids(layer_ids=self._stage1_ids()),
            draft_token_to_kv_pool=SimpleNamespace(layer_num=1),
            num_draft_entries=2,
            num_hidden_layers=60,
        )
        stage1 = self._stage1_ids()
        # k0..k(L-1) then v0..v(L-1) per pool, and the pools are concatenated --
        # so the band repeats per group after the target's ids, not interleaved.
        self.assertEqual(ids, stage1 + stage1 + [60, 60])

    def test_hybrid_draft_pool_is_remapped_out_of_the_target_range(self):
        """The EAGLE draft pool for a hybrid-linear model is itself a
        HybridLinearKVPool that numbers its single MTP layer from zero, so its
        raw ids collide with target layer 0 and must be remapped into the band."""
        ids = build_kv_layer_ids(
            token_to_kv_pool=_hybrid_pool_with_ids(layer_ids=self._stage1_ids()),
            draft_token_to_kv_pool=_hybrid_pool_with_ids(layer_ids=[0]),
            num_draft_entries=2,
            num_hidden_layers=60,
        )
        stage1 = self._stage1_ids()
        self.assertEqual(ids, stage1 + stage1 + [60, 60])

    def test_non_hybrid_pool_publishes_nothing(self):
        self.assertEqual(
            build_kv_layer_ids(
                token_to_kv_pool=SimpleNamespace(),
                draft_token_to_kv_pool=None,
                num_draft_entries=0,
                num_hidden_layers=60,
            ),
            [],
        )

    def test_ragged_draft_registration_is_rejected(self):
        with self.assertRaises(RuntimeError):
            build_kv_layer_ids(
                token_to_kv_pool=_hybrid_pool_with_ids(layer_ids=self._stage1_ids()),
                draft_token_to_kv_pool=SimpleNamespace(layer_num=2),
                num_draft_entries=3,
                num_hidden_layers=60,
            )


class TestDraftBandPairsAcrossPipelineStages(CustomTestCase):
    """Derived property: a pp=2 prefill stage and a pp=1 decode peer, both with
    an EAGLE draft pool, must pair on layer id -- the stage's 8 full-attention
    layers land on the decode peer's matching K and V entries, and the draft
    band lands on the decode peer's draft entries rather than on layer 0."""

    def test_stage1_pairs_onto_the_decode_layout(self):
        full = _full_attention_ids(num_layers=60, interval=4)
        stage1 = [lid for lid in full if lid >= 30]
        src = build_kv_layer_ids(
            token_to_kv_pool=_hybrid_pool_with_ids(layer_ids=stage1),
            draft_token_to_kv_pool=_hybrid_pool_with_ids(layer_ids=[0]),
            num_draft_entries=2,
            num_hidden_layers=60,
        )
        dst = build_kv_layer_ids(
            token_to_kv_pool=_hybrid_pool_with_ids(layer_ids=full),
            draft_token_to_kv_pool=_hybrid_pool_with_ids(layer_ids=[0]),
            num_draft_entries=2,
            num_hidden_layers=60,
        )
        pairs = build_transfer_entry_pairs(
            src, dst, len(src), len(dst), allow_positional_fallback=False
        )
        k_offset = len(full) - len(stage1)
        self.assertEqual(
            pairs,
            # K block, then V block, then the two draft entries at the tail.
            [(i, k_offset + i) for i in range(len(stage1))]
            + [(len(stage1) + i, len(full) + k_offset + i) for i in range(len(stage1))]
            + [
                (2 * len(stage1), 2 * len(full)),
                (2 * len(stage1) + 1, 2 * len(full) + 1),
            ],
        )


class TestMixedMlaDraftPayload(CustomTestCase):
    """An MLA target must not lend its row width or replica policy to MHA draft."""

    @staticmethod
    def _pools(tp, heads):
        import torch
        from sglang.srt.mem_cache.memory_pool import (
            MHATokenToKVPool,
            MLATokenToKVPool,
        )

        latent = MLATokenToKVPool.__new__(MLATokenToKVPool)
        latent.page_size, latent.layer_num, latent.start_layer = 4, 1, 47
        latent.kv_buffer = [torch.zeros(64, 1, 10, dtype=torch.float16)]
        target = HybridLinearKVPool.__new__(HybridLinearKVPool)
        target.full_kv_pool = latent
        target.full_attention_layer_id_mapping = {47: 0}
        target.use_mla = True
        draft = MHATokenToKVPool.__new__(MHATokenToKVPool)
        draft.page_size, draft.size, draft.layer_num = 4, 60, 1
        draft.store_dtype = torch.float16
        draft.use_hnd = False
        draft.k_buffer = [torch.zeros(64, max(1, heads // tp), 2, dtype=torch.float16)]
        draft.v_buffer = [torch.zeros(64, max(1, heads // tp), 3, dtype=torch.float16)]
        draft._kv_buffer_descs = draft._build_kv_buffer_descs()
        return target, draft

    @staticmethod
    def _registration(target, draft):
        from sglang.srt.disaggregation.base.conn import KVArgs

        args = KVArgs()
        tp, tl, ti = target.get_contiguous_buf_infos()
        dp, dl, di = draft.get_contiguous_buf_infos()
        args.kv_data_ptrs, args.kv_data_lens, args.kv_item_lens = (
            tp + dp,
            tl + dl,
            ti + di,
        )
        args.num_draft_entries, args.page_size = len(dp), 4
        args.kv_layer_ids = build_kv_layer_ids(
            token_to_kv_pool=target,
            draft_token_to_kv_pool=draft,
            num_draft_entries=len(dp),
            num_hidden_layers=93,
        )
        return args

    def test_head_shards_use_their_own_registered_strides(self):
        self._check_head_shards(pure=False)

    def test_pure_mla_with_mha_draft_uses_the_same_direct_baseline(self):
        self._check_head_shards(pure=True)

    def _check_head_shards(self, pure):
        import concurrent.futures
        import ctypes
        import torch
        from sglang.srt.runtime_context import get_context

        for src_tp, dst_tp, heads in ((2, 8, 8), (8, 2, 8), (8, 8, 8), (2, 8, 4)):
            for dst_rank in range(dst_tp):
                with self.subTest(
                    src_tp=src_tp, dst_tp=dst_tp, heads=heads, dst=dst_rank
                ):
                    target, draft = self._pools(dst_tp, heads)
                    dest = self._registration(
                        target.full_kv_pool if pure else target, draft
                    )
                    buffers = (
                        target.full_kv_pool.kv_buffer + draft.k_buffer + draft.v_buffer
                    )
                    for tensor in buffers:
                        tensor.fill_(-1)
                    source_ranks = (
                        range(
                            dst_rank * src_tp // dst_tp,
                            (dst_rank + 1) * src_tp // dst_tp,
                        )
                        if src_tp >= dst_tp
                        else [dst_rank * src_tp // dst_tp]
                    )
                    dst_rows, src_rows = (
                        [8, 9, 10, 11, 0, 1, 2, 3],
                        [12, 13, 14, 15, 4, 5, 6, 7],
                    )
                    for src_rank in source_ranks:
                        source, source_draft = self._pools(src_tp, heads)
                        args = self._registration(
                            source.full_kv_pool if pure else source, source_draft
                        )
                        args.draft_total_kv_head_num = heads
                        args.engine_rank = src_rank + 3 * src_tp
                        args.prefill_start_layer = 0
                        sources = (
                            source.full_kv_pool.kv_buffer
                            + source_draft.k_buffer
                            + source_draft.v_buffer
                        )
                        source.full_kv_pool.kv_buffer[0].fill_(47)
                        head_start = (src_rank // max(1, src_tp // heads)) * max(
                            1, heads // src_tp
                        )
                        for kind, tensor in enumerate(sources[1:], 1):
                            for row in range(64):
                                for head in range(tensor.shape[1]):
                                    tensor[row, head] = (
                                        kind * 500 + row * 8 + head_start + head
                                    )
                        src_bounds = dict(
                            zip(args.kv_data_ptrs, args.kv_data_lens, strict=True)
                        )
                        dst_bounds = dict(
                            zip(dest.kv_data_ptrs, dest.kv_data_lens, strict=True)
                        )

                        def copy(
                            session,
                            srcs,
                            dsts,
                            lengths,
                            src_bounds=src_bounds,
                            dst_bounds=dst_bounds,
                        ):
                            for src, dst, length in zip(
                                srcs, dsts, lengths, strict=True
                            ):
                                for addr, bounds in (
                                    (src, src_bounds),
                                    (dst, dst_bounds),
                                ):
                                    self.assertTrue(
                                        any(
                                            base <= addr
                                            and addr + length <= base + size
                                            for base, size in bounds.items()
                                        )
                                    )
                                ctypes.memmove(dst, src, length)
                            return 0

                        manager = MooncakeKVManager.__new__(MooncakeKVManager)
                        manager.kv_args, manager.attn_tp_size, manager.pp_size = (
                            args,
                            src_tp,
                            4,
                        )
                        manager.is_mla_backend, manager.is_hybrid_mla_backend = (
                            True,
                            not pure,
                        )
                        manager.enable_custom_mem_pool = (
                            manager.enable_deferred_decode_kv_release
                        ) = False
                        manager.max_transfer_batch_indices = 0
                        manager.engine = SimpleNamespace(batch_transfer_sync=copy)
                        with (
                            get_context().override_server_args(
                                enable_unified_memory=False
                            ),
                            concurrent.futures.ThreadPoolExecutor(
                                max_workers=1
                            ) as executor,
                        ):
                            self.assertEqual(
                                manager.send_kvcache(
                                    "cpu",
                                    np.array([3, 1]),
                                    dest.kv_data_ptrs,
                                    np.array([2, 0]),
                                    executor,
                                    dst_layer_ids=dest.kv_layer_ids,
                                    dst_kv_item_len=dest.kv_item_lens[0],
                                    dst_attn_tp_size=dst_tp,
                                    dst_kv_item_lens=dest.kv_item_lens,
                                    dst_tp_rank=dst_rank,
                                ),
                                0,
                            )
                    first_head = (dst_rank // max(1, dst_tp // heads)) * max(
                        1, heads // dst_tp
                    )
                    for kind, tensor in enumerate(buffers):
                        for row in range(64):
                            for head in range(tensor.shape[1]):
                                expected = -1
                                if row in dst_rows:
                                    expected = (
                                        47
                                        if kind == 0
                                        else kind * 500
                                        + src_rows[dst_rows.index(row)] * 8
                                        + first_head
                                        + head
                                    )
                                self.assertTrue(
                                    torch.all(tensor[row, head] == expected),
                                    (kind, row, head, expected, tensor[row, head]),
                                )


class TestStagingV2KdaState(CustomTestCase):
    def test_69_layers_keep_every_tp_shard_and_both_persistent_fields(self):
        import ctypes
        from sglang.srt.disaggregation.base.conn import KVArgs

        # The complement of the 24 sparse target layers: no token-ring storage.
        layers = [layer for layer in range(93) if layer % 4]
        self.assertEqual(len(layers), 69)

        def buffers(ids, tp, rank, populated):
            result, dims, groups, outers = [], [], [], []
            for kind in (0, 1):
                full_groups = [8, 8, 16] if kind == 0 else [8]
                channels, base = [], 0
                for width in full_groups:
                    channels.extend(
                        range(
                            base + rank * width // tp, base + (rank + 1) * width // tp
                        )
                    )
                    base += width
                outer = 2 if kind == 0 else 1
                for layer in ids:
                    tensor = np.full((4, outer, len(channels)), -1, dtype=np.int32)
                    if populated:
                        for row in range(outer):
                            tensor[1, row] = [
                                layer * 1000 + kind * 500 + row * 64 + channel
                                for channel in channels
                            ]
                    result.append(tensor)
                    dims.append(len(channels))
                    groups.append(full_groups if kind == 0 else None)
                    outers.append(outer)
            return result, dims, groups, outers

        for src_tp, dst_tp in ((8, 8), (2, 8), (8, 2)):
            for dst_rank in range(dst_tp):
                dst, dst_dims, _, _ = buffers(layers, dst_tp, dst_rank, False)
                expected, _, _, _ = buffers(layers, dst_tp, dst_rank, True)
                source_ranks = (
                    range(
                        dst_rank * src_tp // dst_tp, (dst_rank + 1) * src_tp // dst_tp
                    )
                    if src_tp >= dst_tp
                    else [dst_rank * src_tp // dst_tp]
                )
                for pp_rank in range(4):
                    ids = layers[pp_rank * 69 // 4 : (pp_rank + 1) * 69 // 4]
                    for src_rank in source_ranks:
                        src, dims, groups, outers = buffers(ids, src_tp, src_rank, True)
                        mgr = object.__new__(MooncakeKVManager)
                        args = KVArgs()
                        args.engine_rank = src_rank
                        args.state_types = [StateType.MAMBA]
                        args.state_data_ptrs = [[x.ctypes.data for x in src]]
                        args.state_item_lens = [[x[0].nbytes for x in src]]
                        args.state_dim_per_tensor = [dims]
                        args.state_conv_shard_groups = [groups]
                        args.state_slice_outer_counts = [outers]
                        args.state_layer_ids = [ids * 2]
                        mgr.kv_args, mgr.attn_tp_size, mgr.pp_size = args, src_tp, 4
                        peer = SimpleNamespace(
                            dst_state_data_ptrs=[[x.ctypes.data for x in dst]],
                            dst_state_item_lens=[[x[0].nbytes for x in dst]],
                            dst_state_dim_per_tensor=[dst_dims],
                            dst_state_layer_ids=[layers * 2],
                            dst_attn_tp_size=dst_tp,
                            dst_tp_rank=dst_rank,
                        )
                        transferred = []

                        def transfer(session, blocks):
                            for source, destination, size in blocks:
                                self.assertTrue(
                                    any(
                                        x.ctypes.data <= source
                                        and source + size <= x.ctypes.data + x.nbytes
                                        for x in src
                                    )
                                )
                                self.assertTrue(
                                    any(
                                        x.ctypes.data <= destination
                                        and destination + size
                                        <= x.ctypes.data + x.nbytes
                                        for x in dst
                                    )
                                )
                                ctypes.memmove(destination, source, size)
                            transferred.extend(blocks)
                            return 0

                        mgr._transfer_data = transfer
                        expected_bytes = mgr._validate_staging_v2_state(peer)
                        req = SimpleNamespace(
                            mooncake_session_id="decode", dst_state_indices=[[2]]
                        )
                        self.assertEqual(
                            mgr.maybe_send_extra(req, [[1]], None, peer), 0
                        )
                        self.assertEqual(
                            sum(n for _, _, n in transferred), expected_bytes
                        )
                        saved = args.state_dim_per_tensor
                        args.state_dim_per_tensor = [[]]
                        with self.assertRaisesRegex(ValueError, "complete Mamba"):
                            mgr._validate_staging_v2_state(peer)
                        args.state_dim_per_tensor = saved
                for actual, oracle in zip(dst, expected, strict=True):
                    np.testing.assert_array_equal(actual[2], oracle[1])
                    np.testing.assert_array_equal(actual[[0, 1, 3]], -1)


if __name__ == "__main__":
    unittest.main()
