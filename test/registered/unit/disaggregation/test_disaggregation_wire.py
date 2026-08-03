import threading
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.common.staging_handler import (
    DecodeStagingHandler,
    handle_staging_req,
)
from sglang.srt.disaggregation.common.utils import (
    group_concurrent_contiguous,
    pack_int_lists,
    pack_list_of_buffers,
    unpack_int_lists,
    unpack_list_of_buffers,
)
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    TransferInfo,
)
from sglang.srt.disaggregation.utils import (
    MetadataBuffers,
    TransferBackend,
    get_dsv4_c128_state_indices,
    setup_state_kv_args,
)
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.speculative.eagle_disaggregation import (
    build_eagle_disagg_draft_input,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDisaggregationWire(unittest.TestCase):
    def test_int_lists_roundtrip(self):
        cases = [
            ("Q", [[1, 2, 3], [4]]),
            ("I", [[10, 20], [30, 40, 50]]),
            ("i", [[-1, 2], [3, -4, 5]]),
        ]
        for fmt, sample in cases:
            packed = pack_int_lists(sample, fmt)
            self.assertEqual(unpack_int_lists(packed, fmt), sample, msg=fmt)

    def test_pack_accepts_ndarray(self):
        arrs = [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5], dtype=np.int32),
        ]
        packed = pack_int_lists(arrs, "i")
        self.assertEqual(unpack_int_lists(packed, "i"), [[1, 2, 3], [4, 5]])

    def test_empty_outer_list(self):
        self.assertEqual(pack_int_lists([], "Q"), b"")
        self.assertEqual(unpack_int_lists(b"", "Q"), [])

    def test_empty_inner_list(self):
        packed = pack_int_lists([[]], "I")
        self.assertEqual(unpack_int_lists(packed, "I"), [[]])

    def test_list_of_buffers_roundtrip(self):
        bufs = [b"abc", b"", b"de", b"x" * 17]
        self.assertEqual(unpack_list_of_buffers(pack_list_of_buffers(bufs)), bufs)


class TestGroupConcurrentContiguous(unittest.TestCase):
    @staticmethod
    def _arr(values):
        return np.array(values, dtype=np.int32)

    def test_single_contiguous_group(self):
        src = self._arr([10, 11, 12])
        dst = self._arr([5, 6, 7])
        self.assertEqual(
            group_concurrent_contiguous(src, dst),
            ([[10, 11, 12]], [[5, 6, 7]]),
        )

    def test_splits_on_discontiguous_indices(self):
        src = self._arr([10, 11, 20])
        dst = self._arr([5, 6, 7])
        self.assertEqual(
            group_concurrent_contiguous(src, dst),
            ([[10, 11], [20]], [[5, 6], [7]]),
        )

    def test_empty_src_nonempty_dst(self):
        self.assertEqual(
            group_concurrent_contiguous(self._arr([]), self._arr([1, 2])), ([], [])
        )

    def test_nonempty_src_empty_dst(self):
        # Regression: a non-empty source paired with an empty destination must not
        # raise a NumPy broadcast error (observed transferring DSA sparse-attention
        # state on a disaggregated GLM deployment when decode registered zero dst indices).
        self.assertEqual(
            group_concurrent_contiguous(self._arr([1, 2]), self._arr([])), ([], [])
        )

    def test_mismatched_nonempty_lengths_raise(self):
        with self.assertRaises(ValueError):
            group_concurrent_contiguous(self._arr([1, 2, 3]), self._arr([1, 2]))


class TestMooncakePPStaging(unittest.TestCase):
    def test_mha_dcp_relayout_requires_enabled_staging(self):
        manager = object.__new__(MooncakeKVManager)
        manager.dcp_size = 1
        manager.dcp_rank = 0
        manager.is_mla_backend = False
        manager.is_hybrid_mla_backend = False
        manager.enable_staging = True

        self.assertTrue(manager.requires_dcp_relayout(4, 0))

        manager.enable_staging = False
        with self.assertRaisesRegex(RuntimeError, "Unsupported PD DCP topology"):
            manager.requires_dcp_relayout(4, 0)

    @patch("sglang.srt.disaggregation.common.conn.requests.get")
    def test_mha_decode_dcp_bootstrap_requires_enabled_staging(self, get):
        response = Mock(status_code=200)
        response.json.return_value = {
            "attn_tp_size": 1,
            "attn_cp_size": 1,
            "dp_size": 1,
            "pp_size": 4,
            "page_size": 64,
            "kv_cache_dtype": "fp8_e4m3",
            "follow_bootstrap_room": False,
            "dcp_size": 1,
        }
        get.return_value = response

        manager = object.__new__(MooncakeKVManager)
        manager.prefill_info_table = {}
        manager.kv_args = SimpleNamespace(page_size=64)
        manager.kv_cache_dtype_str = "fp8_e4m3"
        manager.dcp_size = 4
        manager.is_mla_backend = False
        manager.is_hybrid_mla_backend = False
        manager.enable_staging = True
        manager._resolve_rank_mapping = Mock()

        self.assertTrue(manager.try_ensure_parallel_info("prefill"))
        manager._resolve_rank_mapping.assert_called_once()

        manager.prefill_info_table = {}
        manager.enable_staging = False
        with self.assertRaisesRegex(RuntimeError, "staging enabled"):
            manager.try_ensure_parallel_info("prefill")

    def test_transfer_info_carries_exact_kv_token_count(self):
        dst_indices = np.array([3, 4], dtype=np.int32)
        info = TransferInfo.from_zmq(
            [
                b"7",
                b"127.0.0.1",
                b"1234",
                b"peer",
                dst_indices.tobytes(),
                b"5",
                pack_int_lists([[6]], "i"),
                b"1",
                b"256",
                b"8191",
            ]
        )

        self.assertEqual(info.decode_prefix_len, 256)
        self.assertEqual(info.num_kv_tokens, 8191)

    def test_decode_waits_for_dcp_staging_only_when_relayout_is_required(self):
        def make_receiver(prefill_dcp_size):
            prefill_info = SimpleNamespace(
                attn_tp_size=16,
                dcp_size=prefill_dcp_size,
                target_tp_rank=0,
                target_tp_ranks=[0],
                target_cp_ranks=[0],
                target_pp_ranks=[0],
                required_dst_info_num=1,
                required_prefill_response_num=1,
            )
            manager = SimpleNamespace(
                prefill_info_table={"prefill": prefill_info},
                required_prefill_response_num_table={},
                enable_staging=True,
                supports_dcp_staging=True,
                attn_tp_size=16,
                dcp_size=4,
                update_status=Mock(),
            )
            receiver = object.__new__(MooncakeKVReceiver)
            receiver.bootstrap_addr = "prefill"
            receiver.bootstrap_room = 7
            receiver.kv_mgr = manager
            receiver.conclude_state = None
            receiver._setup_bootstrap_infos = Mock()
            receiver.init(0)
            return receiver

        self.assertTrue(make_receiver(prefill_dcp_size=1).require_staging)
        self.assertFalse(make_receiver(prefill_dcp_size=4).require_staging)

    def test_dcp_relayout_uses_staging_when_available(self):
        manager = object.__new__(MooncakeKVManager)
        manager.enable_staging = True
        manager.kv_buffer_tensors = object()
        manager.attn_tp_size = 16
        target = SimpleNamespace(
            staging=object(),
            requires_dcp_relayout=True,
            dst_attn_tp_size=16,
        )

        self.assertTrue(manager._should_use_staging_transfer(target, object()))

        target.requires_dcp_relayout = False
        self.assertFalse(manager._should_use_staging_transfer(target, object()))
        target.requires_dcp_relayout = True
        target.staging = None
        self.assertFalse(manager._should_use_staging_transfer(target, object()))

    def test_dcp_staging_registration_allows_different_kv_geometry(self):
        manager = object.__new__(MooncakeKVManager)
        manager.enable_staging = True
        manager.kv_buffer_tensors = object()
        manager.attn_tp_size = 1
        manager.kv_args = SimpleNamespace(kv_item_lens=[32768] * 8)
        manager.requires_dcp_relayout = Mock(return_value=True)
        manager.prepare_dcp_token_item_lens = Mock(
            side_effect=RuntimeError("geometry differs")
        )
        target = SimpleNamespace(
            staging=object(),
            requires_dcp_relayout=False,
            dcp_token_item_lens=None,
            dst_attn_tp_size=4,
            dst_dcp_size=2,
            dst_dcp_rank=1,
            dst_kv_item_len=16384,
        )

        manager._configure_dcp_registration(target)

        self.assertTrue(target.requires_dcp_relayout)
        self.assertIsNone(target.dcp_token_item_lens)
        manager.prepare_dcp_token_item_lens.assert_not_called()

        target.staging = None
        with self.assertRaisesRegex(RuntimeError, "geometry differs"):
            manager._configure_dcp_registration(target)
        manager.prepare_dcp_token_item_lens.assert_called_once_with([16384] * 8)

    @patch("sglang.srt.disaggregation.decode.setup_state_kv_args")
    @patch("sglang.srt.disaggregation.decode.get_parallel")
    @patch("sglang.srt.disaggregation.decode.get_kv_class")
    def test_decode_staging_uses_model_total_kv_heads(
        self, get_kv_class, get_parallel, _setup_state_kv_args
    ):
        get_parallel.return_value = SimpleNamespace(attn_tp_size=16)
        manager = Mock()
        manager_class = Mock(return_value=manager)
        get_kv_class.side_effect = [KVArgs, manager_class]

        full_kv_pool = SimpleNamespace(
            head_num=1,
            k_buffer=[object()],
            v_buffer=[object()],
            page_size=64,
        )
        token_pool = SimpleNamespace(
            full_kv_pool=full_kv_pool,
            page_size=64,
            get_contiguous_buf_infos=lambda: ([1, 2], [3, 4], [5, 6]),
            get_kv_layer_ids=lambda: [7, 7],
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.tp_rank = 0
        queue.pp_rank = 0
        queue.transfer_backend = TransferBackend.MOONCAKE
        queue.token_to_kv_pool = token_pool
        queue.draft_token_to_kv_pool = None
        queue.metadata_buffers = SimpleNamespace(
            get_buf_infos=lambda: ([10], [20], [30])
        )
        queue.is_mla_backend = False
        queue.enable_staging = True
        queue.scheduler = SimpleNamespace(
            enable_hisparse=False,
            model_config=SimpleNamespace(
                num_hidden_layers=60,
                get_total_num_kv_heads=lambda: 2,
            ),
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(kv_cache_dtype_str="fp8_e4m3")
            ),
            server_args=SimpleNamespace(disaggregation_ib_device=None),
            ps=SimpleNamespace(gpu_id=0, dp_rank=0),
        )

        queue._init_kv_manager()

        kv_args = manager_class.call_args.args[0]
        self.assertEqual(kv_args.kv_head_num, 1)
        self.assertEqual(kv_args.total_kv_head_num, 2)

    @patch("sglang.srt.disaggregation.common.conn._get_bootstrap_session")
    def test_bootstrap_info_records_target_pp_rank(self, get_session):
        response = Mock(status_code=200)
        response.json.return_value = {"rank_ip": "127.0.0.1", "rank_port": 1234}
        get_session.return_value.get.return_value = response
        receiver = object.__new__(MooncakeKVReceiver)
        receiver.bootstrap_addr = "127.0.0.1:5678"

        info = receiver._get_bootstrap_info_from_server(0, 0, 0, 3)

        self.assertEqual(info["pp_rank"], 3)

    @patch("sglang.srt.disaggregation.common.staging_handler.prefetch_staging_reqs")
    def test_pp_rank_requests_its_allocation_response(self, prefetch):
        manager = object.__new__(MooncakeKVManager)
        manager.enable_staging = True
        manager.kv_buffer_tensors = {"page_size": 64}
        manager.pp_size = 16
        manager.pp_rank = 1
        manager.attn_tp_size = 1
        manager.transfer_infos = {
            7: {"peer": SimpleNamespace(is_dummy=False, mooncake_session_id="peer")}
        }
        manager.decode_kv_args_table = {"peer": SimpleNamespace(dst_attn_tp_size=16)}
        manager.server_args = SimpleNamespace(chunked_prefill_size=8192)
        manager._staging_ctx = SimpleNamespace(
            prefetch_requested=set(), prefetch_sockets={}
        )

        with patch(
            "sglang.srt.disaggregation.mooncake.conn.get_schedule",
            return_value=SimpleNamespace(chunked_prefill_size=8192),
        ):
            manager._prefetch_staging_reqs(7)
        prefetch.assert_called_once_with(
            7,
            manager.transfer_infos,
            manager.kv_buffer_tensors,
            8192,
            manager._staging_ctx.prefetch_requested,
            manager._staging_ctx.prefetch_sockets,
            requester_pp_rank=1,
        )

    def test_staging_response_targets_requesting_pp_rank(self):
        sock = Mock()
        receiver = SimpleNamespace(
            chunk_staging_infos=[],
            _connect_to_bootstrap_server=Mock(return_value=(sock, threading.Lock())),
        )
        allocator = SimpleNamespace(
            assign=Mock(return_value=(3, 128, 0)), total_size=1 << 20
        )
        kv_args = SimpleNamespace(
            page_size=64,
            kv_item_lens=[4096, 4096],
            total_kv_head_num=4,
            engine_rank=0,
        )
        target = {"pp_rank": 3}

        handle_staging_req(
            [b"STAGING_REQ", b"7", b"0", b"1", b"peer", b"3"],
            allocator,
            kv_args,
            attn_tp_size=16,
            prefill_attn_tp_size=1,
            kv_buffer_tensors=None,
            room_receivers={7: receiver},
            room_bootstrap={7: [{"pp_rank": 2}, target]},
        )

        receiver._connect_to_bootstrap_server.assert_called_once_with(target)
        sock.send_multipart.assert_called_once()

    @patch("sglang.srt.disaggregation.common.staging_handler.prefetch_staging_reqs")
    def test_dcp_relayout_prefetches_staging(self, prefetch):
        manager = object.__new__(MooncakeKVManager)
        manager.enable_staging = True
        manager.kv_buffer_tensors = {"page_size": 64}
        manager.pp_size = 16
        manager.pp_rank = 1
        manager.attn_tp_size = 16
        manager.transfer_infos = {
            7: {"peer": SimpleNamespace(is_dummy=False, mooncake_session_id="peer")}
        }
        manager.decode_kv_args_table = {
            "peer": SimpleNamespace(dst_attn_tp_size=16, requires_dcp_relayout=True)
        }
        manager._staging_ctx = SimpleNamespace(
            prefetch_requested=set(), prefetch_sockets={}
        )

        with patch(
            "sglang.srt.disaggregation.mooncake.conn.get_schedule",
            return_value=SimpleNamespace(chunked_prefill_size=8192),
        ):
            manager._prefetch_staging_reqs(7)

        prefetch.assert_called_once_with(
            7,
            manager.transfer_infos,
            manager.kv_buffer_tensors,
            8192,
            manager._staging_ctx.prefetch_requested,
            manager._staging_ctx.prefetch_sockets,
            requester_pp_rank=1,
        )

    def test_dcp_staging_allocation_counts_owned_tokens(self):
        sock = Mock()
        receiver = SimpleNamespace(
            chunk_staging_infos=[],
            chunk_staging_num_tokens=[],
            decode_prefix_len=0,
            _connect_to_bootstrap_server=Mock(return_value=(sock, threading.Lock())),
        )
        allocator = SimpleNamespace(
            assign=Mock(return_value=(3, 128, 0)), total_size=1 << 20
        )
        kv_args = SimpleNamespace(
            page_size=64,
            kv_item_lens=[4096, 4096],
            total_kv_head_num=4,
            engine_rank=2,
        )
        target = {"pp_rank": 0}

        handle_staging_req(
            [b"STAGING_REQ", b"7", b"0", b"1", b"peer", b"0", b"5", b"0"],
            allocator,
            kv_args,
            attn_tp_size=16,
            prefill_attn_tp_size=1,
            kv_buffer_tensors=None,
            room_receivers={7: receiver},
            room_bootstrap={7: [target]},
            dcp_size=4,
            dcp_rank=2,
        )

        allocator.assign.assert_called_once_with(128)
        self.assertEqual(receiver.chunk_staging_num_tokens, [5])

    @patch(
        "sglang.srt.disaggregation.common.staging_buffer.gather_all_layers_to_staging"
    )
    def test_pp_stage_writes_its_global_layer_slots(self, gather):
        manager = object.__new__(MooncakeKVManager)
        tensor = SimpleNamespace(shape=(1, 1, 8), element_size=lambda: 2)
        manager.kv_buffer_tensors = {
            "k_buffers": [tensor],
            "v_buffers": [tensor],
            "page_size": 2,
        }
        manager.attn_tp_size = 1
        manager.pp_size = 16
        manager.kv_args = SimpleNamespace(
            engine_rank=0,
            gpu_id=0,
            total_kv_head_num=4,
            kv_head_num=4,
            kv_layer_ids=[7, 7],
        )
        manager._transfer_data = Mock(return_value=0)
        staging = SimpleNamespace(fits=lambda size: True, get_ptr=lambda: 0x9000)

        ret = manager.send_kvcache_staged(
            "peer",
            np.array([1, 2], dtype=np.int32),
            dst_staging_ptr=0x100000,
            dst_staging_size=1 << 20,
            dst_tp_rank=0,
            dst_attn_tp_size=16,
            dst_kv_item_len=128,
            dst_layer_ids=[3, 7, 11, 3, 7, 11],
            staging_buffer=staging,
        )

        self.assertEqual(ret, 0)
        gather.assert_called_once()
        manager._transfer_data.assert_called_once_with(
            "peer",
            [
                (0x9000, 0x100000 + 64, 64),
                (0x9000 + 64, 0x100000 + 4 * 64, 64),
            ],
        )

    @patch(
        "sglang.srt.disaggregation.common.staging_buffer.gather_all_layers_to_staging"
    )
    def test_dcp_staging_gathers_only_owned_tokens(self, gather):
        manager = object.__new__(MooncakeKVManager)
        tensor = SimpleNamespace(shape=(1, 4, 8), element_size=lambda: 2)
        manager.kv_buffer_tensors = {
            "k_buffers": [tensor],
            "v_buffers": [tensor],
            "page_size": 2,
        }
        manager.attn_tp_size = 1
        manager.pp_size = 1
        manager.kv_args = SimpleNamespace(
            engine_rank=0,
            gpu_id=0,
            total_kv_head_num=4,
            kv_head_num=4,
            kv_layer_ids=[],
        )
        manager._transfer_data = Mock(return_value=0)
        staging = SimpleNamespace(fits=lambda size: True, get_ptr=lambda: 0x9000)

        cases = [
            (2, 1, 1, [21, 41], 0, 2),
            (2, 1, 3, [21, 41], 2, 2),
            (4, 1, 1, [21], 0, 4),
        ]
        for dcp_size, dcp_rank, tp_rank, owned, head_start, num_heads in cases:
            with self.subTest(dcp_size=dcp_size, tp_rank=tp_rank):
                gather.reset_mock()
                manager._transfer_data.reset_mock()
                ret = manager.send_kvcache_staged(
                    "peer",
                    np.array([10, 20], dtype=np.int32),
                    dst_staging_ptr=0x100000,
                    dst_staging_size=1 << 20,
                    dst_tp_rank=tp_rank,
                    dst_attn_tp_size=4,
                    dst_kv_item_len=32,
                    dst_layer_ids=[],
                    staging_buffer=staging,
                    dst_kv_indices=np.array([3], dtype=np.int32),
                    dst_dcp_size=dcp_size,
                    dst_dcp_rank=dcp_rank,
                    num_kv_tokens=4,
                )

                self.assertEqual(ret, 0)
                gather.assert_called_once()
                np.testing.assert_array_equal(
                    gather.call_args.args[2], np.array(owned, dtype=np.int64)
                )
                self.assertEqual(gather.call_args.args[4], head_start)
                self.assertEqual(gather.call_args.args[5], num_heads)
                self.assertEqual(gather.call_args.args[6], 1)
                expected_bytes = len(owned) * num_heads * 8 * 2 * 2
                manager._transfer_data.assert_called_once_with(
                    "peer", [(0x9000, 0x100000, expected_bytes)]
                )

    @patch("sglang.srt.disaggregation.common.staging_buffer.scatter_staging_to_kv")
    @patch("torch.cuda.stream", return_value=nullcontext())
    @patch("torch.cuda.set_device")
    def test_dcp_scatter_maps_virtual_slots(self, _set_device, _cuda_stream, scatter):
        handler = object.__new__(DecodeStagingHandler)
        handler.kv_buffer_info = {
            "k_buffers": [torch.empty((20, 1, 8))],
            "v_buffers": [torch.empty((20, 1, 8))],
            "page_size": 2,
        }
        handler.staging_allocator = SimpleNamespace(
            _scatter_stream=object(),
            buffer=SimpleNamespace(buffer=torch.empty(1024, dtype=torch.uint8)),
        )
        handler.decode_tp = 16
        handler.total_kv_heads = 4
        handler.kv_manager = SimpleNamespace(
            dcp_size=4,
            dcp_rank=2,
            kv_args=SimpleNamespace(engine_rank=2),
        )
        handler.scheduler = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.tensor([list(range(40, 48))])
            )
        )
        decode_req = SimpleNamespace(
            req=SimpleNamespace(req_pool_idx=0),
            kv_receiver=SimpleNamespace(
                decode_prefix_len=0,
                prefill_info=SimpleNamespace(attn_tp_size=1),
            ),
        )

        self.assertTrue(handler._scatter_region(0, 0, 8, decode_req))

        scatter.assert_called_once()
        np.testing.assert_array_equal(
            scatter.call_args.args[3].numpy(), np.array([10, 11])
        )
        self.assertEqual(scatter.call_args.args[4], 1)
        self.assertEqual(scatter.call_args.args[6], 4)
        self.assertEqual(scatter.call_args.args[7], 0)

    def test_intermediate_chunk_waits_for_all_pp_writers(self):
        handler = object.__new__(DecodeStagingHandler)
        handler.decode_tp = 16
        handler.kv_manager = SimpleNamespace(pp_size=1)
        decode_req = SimpleNamespace(
            kv_receiver=SimpleNamespace(
                prefill_info=SimpleNamespace(attn_tp_size=1, pp_size=16)
            )
        )

        self.assertEqual(handler.num_writers_for(decode_req), 16)

        handler.kv_manager.pp_size = 16
        self.assertEqual(handler.num_writers_for(decode_req), 1)

    def test_last_scatter_consumes_allocation(self):
        handler = object.__new__(DecodeStagingHandler)
        handler.scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(page_size=64)
        )
        handler._scatter_region = Mock(return_value=True)
        receiver = SimpleNamespace(
            chunk_staging_infos=[(7, 128, 0, 256, 2)],
            chunk_staging_num_tokens=[128],
            decode_prefix_len=0,
            num_kv_tokens=128,
            prefill_info=SimpleNamespace(page_size=64),
        )
        decode_req = SimpleNamespace(
            kv_receiver=receiver,
            req=SimpleNamespace(origin_input_ids=[0] * 128),
        )

        self.assertEqual(handler._submit_last_scatter(decode_req), 7)
        handler._scatter_region.assert_called_once_with(128, 0, 128, decode_req)
        self.assertEqual(receiver.chunk_staging_infos, [(-1, -1, 0, -1, 0)])

    def test_last_scatter_preserves_legacy_page_fallback(self):
        handler = object.__new__(DecodeStagingHandler)
        handler.scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(page_size=64)
        )
        handler._scatter_region = Mock(return_value=True)
        receiver = SimpleNamespace(
            chunk_staging_infos=[(7, 128, 0, 256, 1)],
            chunk_staging_num_tokens=[],
            prefill_info=SimpleNamespace(page_size=64),
        )
        decode_req = SimpleNamespace(
            kv_receiver=receiver,
            req=SimpleNamespace(origin_input_ids=[0] * 65),
        )

        self.assertEqual(handler._submit_last_scatter(decode_req), 7)
        handler._scatter_region.assert_called_once_with(128, 64, 64, decode_req)


class TestEagleDsaSeedTransfer(unittest.TestCase):
    @staticmethod
    def _make_req(seed, metadata_buffer_index=0):
        return SimpleNamespace(
            metadata_buffer_index=metadata_buffer_index,
            output_ids=[101],
            cached_tokens=0,
            cached_tokens_device=0,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            multimodal_inputs=None,
            return_logprob=False,
            return_sampling_mask=False,
            hidden_states_tensor=torch.tensor([1.0, 2.0]),
            output_topk_p=torch.tensor([1.0]),
            output_topk_index=torch.tensor([7]),
            output_dsa_topk_indices=seed,
            bootstrap_room=9,
        )

    def test_metadata_buffer_copies_seed_and_uses_invalid_sentinel(self):
        buffers = MetadataBuffers(
            size=2,
            hidden_size=2,
            hidden_states_dtype=torch.float32,
            output_dsa_topk_indices_dim=3,
        )
        seed = torch.tensor([4, 5, 6], dtype=torch.int32)
        buffers.set_buf(self._make_req(seed))
        buffers.set_buf(self._make_req(None, metadata_buffer_index=1))

        self.assertTrue(torch.equal(buffers.output_dsa_topk_indices[0], seed))
        self.assertEqual(buffers.output_dsa_topk_indices[1].tolist(), [-1, -1, -1])
        ptrs, data_lens, item_lens = buffers.get_buf_infos()
        self.assertEqual(ptrs[-2], buffers.output_dsa_topk_indices.data_ptr())
        self.assertEqual(data_lens[-2], buffers.output_dsa_topk_indices.nbytes)
        self.assertEqual(item_lens[-2], buffers.output_dsa_topk_indices[0].nbytes)

    def test_decode_input_requires_valid_seed_for_every_request(self):
        seeds = (
            torch.tensor([1, 2, 3], dtype=torch.int32),
            torch.tensor([4, 5, 6], dtype=torch.int32),
        )
        batch = SimpleNamespace(
            reqs=[self._make_req(seed) for seed in seeds],
            device="cpu",
            enable_overlap=False,
        )
        server_args = SimpleNamespace(
            speculative_eagle_topk=1,
            speculative_num_steps=5,
            enable_multi_layer_eagle=False,
        )
        last_tokens = torch.tensor([11, 12], dtype=torch.int64)

        draft_input = build_eagle_disagg_draft_input(
            batch, server_args, last_tokens, None
        )
        self.assertTrue(torch.equal(draft_input.dsa_topk_indices, torch.stack(seeds)))

        for invalid_seed in (
            None,
            torch.full((3,), -1, dtype=torch.int32),
        ):
            batch.reqs[1].output_dsa_topk_indices = invalid_seed
            draft_input = build_eagle_disagg_draft_input(
                batch, server_args, last_tokens, None
            )
            self.assertIsNone(draft_input.dsa_topk_indices)

    def test_future_map_initializes_seed_buffer_after_seedless_payload(self):
        future_map = object.__new__(FutureMap)
        future_map.dsa_topk_indices_buf = None
        future_map.req_pool_size = 4
        future_map.device = "cpu"
        future_map._maybe_init_dsa_topk_indices_buf(
            RelayPayload(bonus_tokens=torch.zeros((2,), dtype=torch.int64))
        )
        self.assertIsNone(future_map.dsa_topk_indices_buf)

        seeds = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
        future_map._maybe_init_dsa_topk_indices_buf(
            RelayPayload(
                bonus_tokens=torch.zeros((2,), dtype=torch.int64),
                dsa_topk_indices=seeds,
            )
        )
        self.assertEqual(future_map.dsa_topk_indices_buf.shape, (4, 3))
        self.assertEqual(future_map.dsa_topk_indices_buf.dtype, torch.int32)


class TestDSV4C128StateIndices(unittest.TestCase):
    def test_online_aligned_boundary_has_no_partial_state(self):
        np.testing.assert_array_equal(
            get_dsv4_c128_state_indices(7, 256, online=True, ring_size=1),
            np.empty((0,), dtype=np.int32),
        )

    def test_online_partial_boundary_uses_request_slot(self):
        np.testing.assert_array_equal(
            get_dsv4_c128_state_indices(7, 257, online=True, ring_size=1),
            np.array([7], dtype=np.int32),
        )

    def test_offline_aligned_boundary_has_no_partial_state(self):
        np.testing.assert_array_equal(
            get_dsv4_c128_state_indices(7, 256, online=False, ring_size=128),
            np.empty((0,), dtype=np.int32),
        )

    def test_offline_partial_boundary_uses_request_local_page(self):
        np.testing.assert_array_equal(
            get_dsv4_c128_state_indices(7, 129, online=False, ring_size=256),
            np.array([15], dtype=np.int32),
        )


def _buf_infos(*ptrs):
    return list(ptrs), [ptr + 100 for ptr in ptrs], [ptr + 200 for ptr in ptrs]


def _make_dsv4_target(*, unified, mapping=None):
    pool = object.__new__(DeepSeekV4TokenToKVPool)
    pool._unified_kv = unified
    pool.page_size = 256
    pool.sliding_window = 128
    pool.full_to_swa_index_mapping = mapping
    pool.unified_swa_window = 128
    pool.unified_swa_ring_size = 131
    pool.unified_swa_pages = 524
    pool.get_state_buf_infos = lambda: _buf_infos(11)
    pool.get_unified_swa_ring_buf_infos = lambda: (
        _buf_infos(12) if unified else ([], [], [])
    )
    pool.get_c128_state_buf_infos = lambda: ([], [], [])
    return pool


def _make_dsv4_draft(*, unified, mapping=None):
    pool = object.__new__(DeepSeekV4TokenToKVPool)
    pool._unified_kv = unified
    pool.compression_ratios = [0]
    pool.page_size = 256
    pool.sliding_window = 128
    pool.full_to_swa_index_mapping = mapping
    pool.unified_swa_window = 128
    pool.unified_swa_ring_size = 131
    pool.unified_swa_pages = 524
    pool.compress_state_pools = [None]
    pool.indexer_compress_state_pools = [None]
    if unified:
        pool.unified_kv_pool = SimpleNamespace(
            swa_pages=524,
            kv_buffer=[torch.empty((524, 16), dtype=torch.uint8)],
        )
    else:
        pool.swa_kv_pool = SimpleNamespace(
            kv_buffer=[torch.empty((2, 16), dtype=torch.uint8)]
        )
    return pool


class TestDSV4DraftStateRegistration(unittest.TestCase):
    def test_draft_state_is_a_separate_component(self):
        mapping = torch.arange(16)
        cases = [
            (
                "paged",
                _make_dsv4_target(unified=False, mapping=mapping),
                _make_dsv4_draft(unified=False, mapping=mapping),
                [StateType.SWA, StateType.SWA],
                [[11]],
            ),
            (
                "unified",
                _make_dsv4_target(unified=True),
                _make_dsv4_draft(unified=True),
                [StateType.SWA, StateType.SWA_RING, StateType.SWA_RING],
                [[11], [12]],
            ),
        ]

        for name, target, draft, expected_types, target_ptrs in cases:
            with self.subTest(name=name):
                if draft._unified_kv:
                    expected_infos = draft.get_unified_swa_ring_buf_infos()
                else:
                    expected_infos = draft.get_state_buf_infos()
                kv_args = KVArgs()

                setup_state_kv_args(kv_args, target, draft)

                self.assertEqual(kv_args.state_types, expected_types)
                self.assertEqual(kv_args.state_data_ptrs[:-1], target_ptrs)
                self.assertEqual(kv_args.state_data_ptrs[-1], expected_infos[0])
                self.assertEqual(kv_args.state_data_lens[-1], expected_infos[1])
                self.assertEqual(kv_args.state_item_lens[-1], expected_infos[2])


if __name__ == "__main__":
    unittest.main()
