import concurrent.futures
import ctypes
import unittest
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMooncakeTransferQueueSharding(CustomTestCase):
    @staticmethod
    def _make_manager(rooms, queue_count=4):
        queued_rooms = [[] for _ in range(queue_count)]
        queues = [
            SimpleNamespace(
                put=lambda chunk, queue_index=queue_index: queued_rooms[
                    queue_index
                ].append((chunk.room, chunk.index_slice.start))
            )
            for queue_index in range(queue_count)
        ]
        sessions = {
            f"decode-host:{port}": object() for port in (15001, 15002, 15003, 15004)
        }
        manager = SimpleNamespace(
            staging_version=0,
            disaggregation_mode=DisaggregationMode.PREFILL,
            request_status={room: KVPoll.WaitingForInput for room in rooms},
            transfer_infos={room: sessions for room in rooms},
            transfer_queues=queues,
        )
        manager.check_status = lambda room: manager.request_status[room]
        return manager, queued_rooms

    @staticmethod
    def _enqueue(manager, room, chunk_start=0):
        MooncakeKVManager.add_transfer_request(
            manager,
            bootstrap_room=room,
            kv_indices=np.array([room], dtype=np.int32),
            index_slice=slice(chunk_start, chunk_start + 1),
            is_last_chunk=False,
        )

    def test_congruent_rooms_distribute_across_queues(self):
        room_count = 32
        queue_count = 4
        # These rooms model one DP shard: their low queue bits are identical.
        rooms = range(100, 100 + room_count * queue_count, queue_count)
        manager, queued_rooms = self._make_manager(rooms, queue_count)

        for room in rooms:
            self._enqueue(manager, room)

        used_queues = [items for items in queued_rooms if items]
        self.assertEqual(len(used_queues), min(room_count, queue_count))

    def test_chunks_from_same_room_stay_on_one_queue(self):
        room = 100
        manager, queued_rooms = self._make_manager([room])

        self._enqueue(manager, room, 0)
        self._enqueue(manager, room, 1)

        used_queues = [items for items in queued_rooms if items]
        self.assertEqual(len(used_queues), 1)
        self.assertEqual(used_queues[0], [(room, 0), (room, 1)])


class TestMooncakeTransferBatching(unittest.TestCase):
    @staticmethod
    def _make_manager(
        side_effect=None, enable_custom_mem_pool=False, max_batch_indices=0
    ):
        engine = MagicMock()
        if side_effect is None:
            engine.batch_transfer_sync.return_value = 0
        else:
            engine.batch_transfer_sync.side_effect = side_effect
        manager = SimpleNamespace(
            engine=engine,
            is_mla_backend=True,
            is_hybrid_mla_backend=False,
            pp_size=1,
            enable_custom_mem_pool=enable_custom_mem_pool,
            custom_mem_pool_type="NVLINK" if enable_custom_mem_pool else None,
            enable_deferred_decode_kv_release=False,
            max_transfer_batch_indices=max_batch_indices,
            get_mla_kv_ptrs_with_pp=MagicMock(
                return_value=([1000, 2000], [5000, 6000], 2)
            ),
        )
        manager._transfer_data = lambda session, blocks: (
            MooncakeKVManager._transfer_data(manager, session, blocks)
        )
        manager._await_transfer_futures = lambda futures: (
            MooncakeKVManager._await_transfer_futures(manager, futures)
        )
        return manager

    @staticmethod
    def _send(
        manager,
        dst_device_data_indices=None,
        dst_device_data_ptrs=None,
    ):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return MooncakeKVManager._send_kvcache_generic(
                manager,
                mooncake_session_id="session",
                src_data_ptrs=[1000, 2000],
                dst_data_ptrs=[5000, 6000],
                item_lens=[10, 20],
                prefill_data_indices=np.array([0, 1, 2, 3, 4], dtype=np.int32),
                dst_data_indices=np.array([10, 11, 12, 13, 14], dtype=np.int32),
                executor=executor,
                dst_device_data_indices=dst_device_data_indices,
                dst_device_data_ptrs=dst_device_data_ptrs,
            )

    def test_slices_index_arrays_before_forming_transfer_ranges(self):
        manager = self._make_manager(max_batch_indices=2)
        ret = self._send(manager)

        self.assertEqual(ret, 0)
        self.assertEqual(
            manager.engine.batch_transfer_sync.call_args_list,
            [
                call("session", [1000, 2000], [5100, 6200], [20, 40]),
                call("session", [1020, 2040], [5120, 6240], [20, 40]),
                call("session", [1040, 2080], [5140, 6280], [10, 20]),
            ],
        )

    def test_preserves_legacy_single_batch_path_for_short_transfers(self):
        for max_batch_indices in (0, 5, 6):
            with self.subTest(max_batch_indices=max_batch_indices):
                manager = self._make_manager(max_batch_indices=max_batch_indices)
                ret = self._send(manager)

                self.assertEqual(ret, 0)
                manager.engine.batch_transfer_sync.assert_called_once_with(
                    "session",
                    [1000, 2000],
                    [5100, 6200],
                    [50, 100],
                )

    def test_stops_after_first_failed_index_batch(self):
        manager = self._make_manager(side_effect=[0, -1], max_batch_indices=2)
        ret = self._send(manager)

        self.assertEqual(ret, -1)
        self.assertEqual(manager.engine.batch_transfer_sync.call_count, 2)

    def test_uses_device_page_indices_in_batched_path(self):
        manager = self._make_manager(max_batch_indices=2)
        ret = self._send(
            manager,
            dst_device_data_indices=np.array([20, 21, 22, 23, 24], dtype=np.int32),
            dst_device_data_ptrs={6000},
        )

        self.assertEqual(ret, 0)
        self.assertEqual(
            manager.engine.batch_transfer_sync.call_args_list,
            [
                call("session", [1000, 2000], [5100, 6400], [20, 40]),
                call("session", [1020, 2040], [5120, 6440], [20, 40]),
                call("session", [1040, 2080], [5140, 6480], [10, 20]),
            ],
        )

    def test_preserves_one_transfer_per_layer_for_custom_mem_pool(self):
        manager = self._make_manager(enable_custom_mem_pool=True, max_batch_indices=2)
        ret = self._send(manager)

        self.assertEqual(ret, 0)
        self.assertEqual(manager.engine.batch_transfer_sync.call_count, 2)
        manager.engine.batch_transfer_sync.assert_has_calls(
            [
                call("session", [1000], [5100], [50]),
                call("session", [2000], [6200], [100]),
            ],
            any_order=True,
        )


class TestMiniMaxStateTransfer(CustomTestCase):
    def test_index_truncates_but_dense_rejects_mismatched_page_lists(self):
        """Legacy index transfers copy the common prefix; incomplete dense KV must fail."""

        def copy_bytes(session, sources, destinations, lengths):
            for src, dst, length in zip(sources, destinations, lengths, strict=True):
                ctypes.memmove(dst, src, length)
            return 0

        for state in (StateType.MINIMAX_INDEX_K, StateType.MINIMAX_DENSE_KV):
            for src_pages, dst_pages in (([1], [0]), ([1, 2], [0]), ([1], [0, 2])):
                with self.subTest(state=state, src=src_pages, dst=dst_pages):
                    src = np.arange(3, dtype=np.int32)
                    dst = np.full(3, -1, dtype=np.int32)
                    manager = MooncakeKVManager.__new__(MooncakeKVManager)
                    manager.kv_args = SimpleNamespace(
                        state_types=[state],
                        state_data_ptrs=[[src.ctypes.data]],
                        state_item_lens=[[src.itemsize]],
                        state_dim_per_tensor=[[]],
                        state_layer_ids=[[]],
                    )
                    manager.engine = SimpleNamespace(batch_transfer_sync=copy_bytes)
                    manager.pp_size = manager.attn_tp_size = 1
                    manager.is_mla_backend = manager.is_hybrid_mla_backend = False
                    manager.enable_custom_mem_pool = False
                    manager.max_transfer_batch_indices = 0
                    peer = SimpleNamespace(
                        dst_state_data_ptrs=[[dst.ctypes.data]],
                        dst_state_item_lens=[[dst.itemsize]],
                        dst_state_dim_per_tensor=[[]],
                        dst_state_layer_ids=[[]],
                        dst_attn_tp_size=1,
                    )
                    kwargs = dict(
                        req=SimpleNamespace(
                            mooncake_session_id="cpu", dst_state_indices=[dst_pages]
                        ),
                        prefill_state_indices=[src_pages],
                        executor=None,
                        target_rank_registration_info=peer,
                    )
                    if state == StateType.MINIMAX_DENSE_KV and len(src_pages) != len(
                        dst_pages
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, "state index length mismatch"
                        ):
                            manager.maybe_send_extra(**kwargs)
                        np.testing.assert_array_equal(dst, [-1, -1, -1])
                    else:
                        self.assertEqual(manager.maybe_send_extra(**kwargs), 0)
                        np.testing.assert_array_equal(dst, [1, -1, -1])


class TestDcpDraftHeadTransfer(unittest.TestCase):
    def test_transfers_draft_heads_to_logical_destination_rows(self):
        for src_tp, dst_tp in ((4, 8), (8, 4), (8, 8), (4, 32), (32, 4)):
            for custom_pool in (False, True):
                for batch_size in (0, 37):
                    with self.subTest(
                        src_tp=src_tp,
                        dst_tp=dst_tp,
                        custom_pool=custom_pool,
                        batch_size=batch_size,
                    ):
                        self._check_transfer(src_tp, dst_tp, custom_pool, batch_size)

    def test_pure_mla_with_sharded_draft_retains_all_source_heads(self):
        # Bootstrap now retains every source when any draft entry is present.
        for src_tp, dst_tp in ((4, 8), (8, 4)):
            with self.subTest(src_tp=src_tp, dst_tp=dst_tp):
                self._check_transfer(src_tp, dst_tp, False, 37, pure_mla=True)

    def test_sliced_draft_stops_after_failed_batch(self):
        self._check_transfer(4, 8, False, 37, fail_draft=True)

    def _check_transfer(
        self, src_tp, dst_tp, custom_pool, batch_size, fail_draft=False, pure_mla=False
    ):
        page_size, tokens, heads, head_bytes = 64, 249, 16, 4
        src_width, dst_width = (
            max(1, heads // src_tp) * head_bytes,
            max(1, heads // dst_tp) * head_bytes,
        )
        src_pages = np.array([1, 3, 4, 7], dtype=np.int32)
        logical = np.arange(tokens)
        src_rows = src_pages[logical // page_size] * page_size + logical % page_size
        expected = (
            np.arange(tokens * heads * head_bytes, dtype=np.int64)
            .reshape(tokens, heads, head_bytes)
            .astype(np.uint8)
        )
        for dst_rank in range(dst_tp):
            dst_buffers = {
                base: np.zeros(16384 * max(8, dst_width), dtype=np.uint8)
                for base in (1000000, 2000000, 3000000, 4000000)
            }
            source_ranks = (
                range(dst_rank * src_tp // dst_tp, (dst_rank + 1) * src_tp // dst_tp)
                if src_tp >= dst_tp
                else [dst_rank * src_tp // dst_tp]
            )
            for src_rank in source_ranks:
                src_head_start = (src_rank // max(1, src_tp // heads)) * max(
                    1, heads // src_tp
                )
                source = np.zeros(1024 * src_width, dtype=np.uint8)
                source.reshape(-1, src_width)[src_rows] = expected[
                    :, src_head_start : src_head_start + max(1, heads // src_tp)
                ].reshape(tokens, src_width)
                target = np.zeros(1024 * 8, dtype=np.uint8)
                target.reshape(-1, 8)[src_rows] = (
                    np.arange(tokens * 8).reshape(tokens, 8).astype(np.uint8)
                )
                src_buffers = {10000: target, 100000: source, 200000: source}

                failed_batches = []

                def transfer(
                    session, blocks, src_buffers=src_buffers, dst_buffers=dst_buffers
                ):
                    draft_blocks = [block for block in blocks if block[1] >= 3000000]
                    if fail_draft and draft_blocks:
                        failed_batches.append(draft_blocks)
                        return 17
                    if batch_size and src_width != dst_width:
                        self.assertLessEqual(
                            len(draft_blocks), batch_size * (1 if custom_pool else 2)
                        )
                    for src, dst, size in blocks:
                        src_base = max(base for base in src_buffers if base <= src)
                        dst_base = max(base for base in dst_buffers if base <= dst)
                        dst_buffers[dst_base][
                            dst - dst_base : dst - dst_base + size
                        ] = src_buffers[src_base][
                            src - src_base : src - src_base + size
                        ]
                    return 0

                manager = SimpleNamespace(
                    is_mla_backend=pure_mla,
                    kv_args=SimpleNamespace(
                        page_size=page_size,
                        kv_layer_ids=[47, 93, 93],
                        kv_data_ptrs=[10000, 100000, 200000],
                        num_draft_entries=2,
                        engine_rank=src_rank + 2 * src_tp,
                    ),
                    attn_tp_size=src_tp,
                    max_transfer_batch_indices=batch_size,
                    enable_custom_mem_pool=custom_pool,
                    _transfer_data=transfer,
                    _await_transfer_futures=lambda futures: max(
                        f.result() for f in futures
                    ),
                )
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    result = MooncakeKVManager.send_kvcache_dcp(
                        manager,
                        "session",
                        src_pages,
                        [1000000, 2000000, 3000000, 4000000],
                        np.array([2], dtype=np.int32),
                        dcp_token_item_lens=[8, src_width, src_width],
                        dst_dcp_size=dst_tp,
                        dst_dcp_rank=dst_rank,
                        src_page_offset=0,
                        decode_prefix_len=0,
                        num_kv_tokens=tokens,
                        executor=executor,
                        dst_layer_ids=[3, 47, 93, 93],
                        dst_kv_item_lens=[
                            page_size * 8,
                            page_size * 8,
                            page_size * dst_tp * dst_width,
                            page_size * dst_tp * dst_width,
                        ],
                        dst_tp_rank=dst_rank,
                        dst_attn_tp_size=dst_tp,
                    )
                if fail_draft:
                    self.assertEqual(result, 17)
                    self.assertEqual(len(failed_batches), 1)
                    return
                self.assertEqual(result, 0)
            dst_head_start = (dst_rank // max(1, dst_tp // heads)) * max(
                1, heads // dst_tp
            )
            for base in (3000000, 4000000):
                actual = dst_buffers[base].reshape(-1, dst_width)[
                    2 * page_size * dst_tp + logical
                ]
                np.testing.assert_array_equal(
                    actual,
                    expected[
                        :,
                        dst_head_start : dst_head_start + max(1, heads // dst_tp),
                    ].reshape(tokens, dst_width),
                )
            owned = np.arange(dst_rank, tokens, dst_tp)
            actual_target = dst_buffers[2000000].reshape(-1, 8)[
                2 * page_size + owned // dst_tp
            ]
            np.testing.assert_array_equal(
                actual_target,
                np.arange(tokens * 8).reshape(tokens, 8).astype(np.uint8)[owned],
            )
            self.assertFalse(dst_buffers[1000000].any())


class TestDcpPackLifetime(CustomTestCase):
    def test_failed_transfer_drains_before_pack_buffer_reuse(self):
        """A failed layer must not release the pack buffer while another transfer reads it."""
        manager = TestMooncakeTransferBatching._make_manager(
            enable_custom_mem_pool=True
        )
        manager.kv_args = SimpleNamespace(
            page_size=1, kv_layer_ids=[], kv_data_ptrs=[1000, 2000], num_draft_entries=0
        )
        source = np.array([11], dtype=np.uint8)
        observed = []
        running, release = Event(), Event()

        def transfer(session, blocks):
            if blocks[0][0] == 1000:
                self.assertTrue(running.wait(10))
                return 17
            running.set()
            self.assertTrue(release.wait(10))
            observed.append(int(source[0]))
            return 0

        def send(executor):
            result = MooncakeKVManager.send_kvcache_dcp(
                manager,
                "session",
                np.array([0, 1], dtype=np.int32),
                [5000, 6000],
                np.array([0], dtype=np.int32),
                dcp_token_item_lens=[1, 1],
                dst_dcp_size=2,
                dst_dcp_rank=0,
                src_page_offset=0,
                decode_prefix_len=0,
                num_kv_tokens=2,
                executor=executor,
                dst_layer_ids=[],
                pack_buffer=object(),
            )
            source[0] = 22
            return result

        manager._transfer_data = transfer
        with (
            patch(
                "sglang.srt.disaggregation.common.dcp_pack.try_pack_dcp_src",
                return_value=([1000, 2000], np.array([0], dtype=np.int64)),
            ),
            concurrent.futures.ThreadPoolExecutor(max_workers=2) as transfers,
            concurrent.futures.ThreadPoolExecutor(max_workers=1) as worker,
        ):
            future = worker.submit(send, transfers)
            try:
                self.assertTrue(running.wait(10))
                with self.assertRaises(concurrent.futures.TimeoutError):
                    future.result(timeout=1)
            finally:
                release.set()
            self.assertEqual(future.result(timeout=10), 17)
        self.assertEqual(observed, [11])


class TestStagingV2Payload(CustomTestCase):
    """Real worker routing, real byte copy and an independent row/head oracle."""

    @staticmethod
    def entries(layer_ids, tp, heads, page, draft):
        from sglang.srt.disaggregation.common.staging_layout import StagingEntry

        specs = [
            ("target", layer, "mla_latent", (layer % 3 + 3) * 2, 0)
            for layer in layer_ids
        ]
        if draft:
            specs += [
                ("draft", 93, kind, max(1, heads // tp) * width, heads)
                for kind, width in (("mha_k", 4), ("mha_v", 6))
            ]
        return tuple(
            StagingEntry(
                component, layer, kind, i, "float16", page, width, width, count
            )
            for i, (component, layer, kind, width, count) in enumerate(specs)
        )

    @staticmethod
    def tensors(entries, tp, rank, page, sentinel=False):
        import torch

        result = []
        for e in entries:
            tensor = torch.empty(
                8 * page, 1, e.copy_width_bytes // 2, dtype=torch.float16
            )
            raw = tensor.view(torch.uint8).reshape(8 * page, -1)
            if sentinel:
                raw.fill_(253)
            else:
                head_width = {"mha_k": 4, "mha_v": 6}.get(e.kind, e.copy_width_bytes)
                first_head = (
                    rank // max(1, tp // e.total_heads) * max(1, e.total_heads // tp)
                    if e.total_heads
                    else 0
                )
                rows = torch.arange(8 * page)[:, None]
                columns = torch.arange(e.copy_width_bytes)[None, :]
                raw.copy_(
                    (
                        e.global_layer_id * 17
                        + rows * 3
                        + (first_head + columns // head_width) * 7
                        + columns % head_width
                        + (81 if e.kind == "mha_v" else 0)
                    )
                    % 251
                )
            result.append(tensor)
        return result

    def run_payload(
        self, src_tp, dst_tp, pp_size, heads, page, full=False, staged=True
    ):
        import json
        import threading
        from collections import defaultdict, deque
        import torch
        from sglang.kernels.ops.kvcache import scatter_staging
        from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
        from sglang.srt.disaggregation.common.staging_handler import (
            PrefillStagingContext,
            StagingRegisterInfo,
        )
        from sglang.srt.disaggregation.common.staging_layout import (
            WriterLayout,
            encode_manifest,
            plan_chunk,
        )
        from sglang.srt.disaggregation.common.utils import TransferKVChunk
        from sglang.srt.disaggregation.base.conn import KVTransferMetric
        from sglang.srt.disaggregation.mooncake.conn import TransferInfo
        from sglang.srt.runtime_context import get_context

        layer_ids = list(range(0, 93, 4))
        partitions = (
            [layer_ids]
            if pp_size == 1
            else [layer_ids[:6], layer_ids[6:11], layer_ids[11:17], layer_ids[17:]]
        )
        tokens = 2 * page if full else page + 1
        source_pages = np.array([5, 2], dtype=np.int32)
        dst_pages = np.array([4, 1], dtype=np.int32)
        # These are final request rows after a nonzero prefix, not source pages.
        final_table = torch.tensor(
            [6 * page + j for j in range(page)]
            + [int(dst_pages[j // page]) * page + j % page for j in range(tokens)]
        )
        calls = 0
        for dst_rank in range(dst_tp):
            src_ranks = (
                list(
                    range(
                        dst_rank * src_tp // dst_tp, (dst_rank + 1) * src_tp // dst_tp
                    )
                )
                if src_tp >= dst_tp
                else [dst_rank * src_tp // dst_tp]
            )
            writers = tuple(
                WriterLayout(
                    f"p{pp}t{rank}",
                    pp,
                    rank,
                    src_tp,
                    self.entries(ids, src_tp, heads, page, pp == pp_size - 1),
                )
                for pp, ids in enumerate(partitions)
                for rank in src_ranks
            )
            destination = WriterLayout(
                "decode",
                0,
                dst_rank,
                dst_tp,
                self.entries(layer_ids, dst_tp, heads, page, True),
            )
            plan = plan_chunk(writers, destination, tokens)
            ring = StagingBuffer(max(256, plan.total_bytes), "cpu", 0)
            ring.buffer.fill_(254)
            outputs = self.tensors(destination.entries, dst_tp, dst_rank, page, True)
            peer = SimpleNamespace(
                staging=StagingRegisterInfo(
                    ring.get_ptr(),
                    ring.get_size(),
                    manifest=encode_manifest(writers, destination),
                ),
                dst_attn_tp_size=dst_tp,
                dst_tp_rank=dst_rank,
                requires_dcp_relayout=False,
            )
            for writer in reversed(writers):
                source = self.tensors(writer.entries, src_tp, writer.tp_rank, page)
                region = plan.region_for(writer.writer_id)
                staging = StagingBuffer(
                    max(256, region.length if region else 0), "cpu", 0
                )
                mgr = object.__new__(MooncakeKVManager)
                mgr.staging_version, mgr.staging_layout = (2 if staged else 0), writer
                mgr.kv_buffer_tensors = {"entries": source, "page_size": page}
                mgr.kv_args = SimpleNamespace(
                    page_size=page,
                    kv_data_ptrs=[x.data_ptr() for x in source],
                    kv_item_lens=[e.copy_width_bytes * page for e in writer.entries],
                    kv_layer_ids=[e.global_layer_id for e in writer.entries],
                    num_draft_entries=2 if writer.pp_rank == pp_size - 1 else 0,
                    draft_total_kv_head_num=heads,
                    engine_rank=writer.tp_rank,
                    prefill_start_layer=0,
                )
                mgr.attn_tp_size, mgr.pp_size = src_tp, pp_size
                mgr.attn_tp_rank = writer.tp_rank
                mgr._deferred_ack_targets, mgr._deferred_ack_poisoned_rooms = {}, set()
                mgr.is_mla_backend, mgr.is_hybrid_mla_backend = True, True
                mgr._staging_ctx = PrefillStagingContext()
                mgr.enable_staging, mgr.enable_trace = staged, False
                mgr.enable_custom_mem_pool = False
                mgr.enable_deferred_decode_kv_release = False
                mgr.defer_decode_allocation = False
                mgr._staging_outstanding = defaultdict(int)
                mgr.session_lock = threading.Lock()
                mgr.state_layout_rejections, mgr.failed_sessions = {}, set()
                mgr.session_failures = defaultdict(int)
                mgr.state_strides_validated = set()
                mgr.max_transfer_batch_indices = 0
                mgr.attn_cp_size, mgr.attn_cp_rank, mgr.pp_rank = 1, 0, writer.pp_rank
                mgr.kv_args.attn_tp_size = src_tp
                mgr.request_status = {7: KVPoll.Transferring}
                mgr.decode_kv_args_table = {"decode": peer}
                req = TransferInfo(
                    7,
                    "127.0.0.1",
                    9999,
                    "decode",
                    dst_pages,
                    0,
                    [],
                    1,
                    False,
                    page,
                    staging_generation="generation",
                )
                mgr.transfer_infos = {7: {"decode": req}}
                mgr.req_to_decode_prefix_len = {7: page}
                ready = []

                def send_message(address, message, **kwargs):
                    document = json.loads(message[1])
                    if message[0] == b"STAGING_V2_REQ":
                        mgr._handle_staging_v2_rsp(
                            dict(
                                document,
                                alloc_id=9,
                                offset=0,
                                round=0,
                                end=plan.total_bytes,
                            )
                        )
                    else:
                        ready.append(document)

                mgr._send_multipart_locked = send_message
                source_bounds = (
                    [(staging.get_ptr(), staging.get_size())]
                    if staged
                    else [(t.data_ptr(), t.numel() * t.element_size()) for t in source]
                )
                dest_bounds = (
                    [(ring.get_ptr(), ring.get_size())]
                    if staged
                    else [(t.data_ptr(), t.numel() * t.element_size()) for t in outputs]
                )

                def bulk(session, srcs, dsts, sizes):
                    nonlocal calls
                    calls += 1
                    if staged:
                        self.assertEqual(len(srcs), 1)
                    for src, dst, size in zip(srcs, dsts, sizes, strict=True):
                        for addr, bounds in ((src, source_bounds), (dst, dest_bounds)):
                            self.assertTrue(
                                any(
                                    base <= addr and addr + size <= base + length
                                    for base, length in bounds
                                )
                            )
                        ctypes.memmove(dst, src, size)
                    return 0

                mgr.engine = SimpleNamespace(batch_transfer_sync=bulk)
                work = TransferKVChunk(
                    7,
                    source_pages,
                    slice(0, 2),
                    False,
                    None,
                    None,
                    num_kv_tokens=tokens,
                    transfer_metric=KVTransferMetric(),
                )
                work_queue = deque([work])
                queue = SimpleNamespace(
                    get=lambda: work_queue.popleft() if work_queue else None,
                    put=work_queue.append,
                )
                if staged:
                    mgr.transfer_worker(queue, None, staging)
                    self.assertEqual(len(ready), 1)
                    self.assertEqual(
                        work.transfer_metric.transfer_total_bytes,
                        region.length if region else 0,
                    )
                    # A retry may not rewrite an already scattered ring allocation.
                    self.assertEqual(
                        mgr._do_staging_transfer_v2(work, req, peer, staging, queue),
                        (0, False),
                    )
                else:
                    # Legacy direct transfers operate in pages; use full chunks.
                    with (
                        get_context().override_server_args(enable_unified_memory=False),
                        concurrent.futures.ThreadPoolExecutor(
                            max_workers=1
                        ) as executor,
                    ):
                        mgr.send_kvcache(
                            "decode",
                            source_pages,
                            [t.data_ptr() for t in outputs],
                            dst_pages,
                            executor,
                            dst_layer_ids=[
                                e.global_layer_id for e in destination.entries
                            ],
                            dst_attn_tp_size=dst_tp,
                            dst_kv_item_len=destination.entries[0].copy_width_bytes
                            * page,
                            dst_kv_item_lens=[
                                e.copy_width_bytes * page for e in destination.entries
                            ],
                            dst_tp_rank=dst_rank,
                        )
            if staged:
                scatter_staging(ring.buffer, outputs, final_table[page:], plan)
                for region in plan.regions:
                    for i, copy in enumerate(region.entries):
                        end = (
                            region.entries[i + 1].offset
                            if i + 1 < len(region.entries)
                            else region.length
                        )
                        self.assertEqual(
                            ring.buffer[
                                region.offset
                                + copy.offset
                                + copy.length : region.offset + end
                            ].tolist(),
                            [0] * (end - copy.offset - copy.length),
                        )
            oracle = self.tensors(destination.entries, dst_tp, dst_rank, page)
            for output, expected_source in zip(outputs, oracle, strict=True):
                expected = torch.full_like(output.view(torch.uint8), 253)
                for j in range(tokens):
                    source_row = int(source_pages[j // page]) * page + j % page
                    expected[int(final_table[page + j])] = expected_source.view(
                        torch.uint8
                    )[source_row]
                self.assertTrue(
                    torch.equal(output.view(torch.uint8), expected),
                    (src_tp, dst_tp, pp_size, page, dst_rank),
                )
        return calls

    def test_actual_worker_routes_tep8_and_pp4_through_one_bulk_per_writer(self):
        self.assertEqual(self.run_payload(8, 8, 1, 8, 4, full=True), 8)
        self.assertEqual(self.run_payload(2, 8, 4, 8, 4, full=True), 32)
        self.run_payload(8, 8, 1, 8, 4, full=True, staged=False)
        self.run_payload(2, 8, 4, 8, 4, full=True, staged=False)

    def test_partial_pages_reverse_tp_and_replicated_draft_heads(self):
        for page in (1, 4, 64):
            for src_tp, dst_tp, heads in (
                (8, 8, 8),
                (2, 8, 8),
                (8, 2, 8),
                (8, 2, 4),
                (2, 8, 4),
            ):
                with self.subTest(page=page, src_tp=src_tp, dst_tp=dst_tp, heads=heads):
                    self.run_payload(src_tp, dst_tp, 4, heads, page)


if __name__ == "__main__":
    unittest.main()
