"""Late bootstrap must request staging before a parked prefill can transfer.

Exercise the real scheduler, sender, queue and staging protocol on CPU. Socket
delivery, the decode allocator and the GPU copy are substituted explicitly.
"""

import threading
import unittest
from array import array
from queue import SimpleQueue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.staging_handler import (
    PrefillStagingContext,
    PrefillStagingStrategy,
    handle_staging_req,
    handle_staging_rsp,
)
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVSender,
    TransferInfo,
)
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestMooncakeOptimisticStaging(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", device="cpu"), role="tokenizer")
        enter_scope(self, get_context().override_server_args(chunked_prefill_size=512))
        enter_scope(
            self,
            patch.dict(
                "os.environ",
                {
                    "SGLANG_PERFPF_STAGING_LAZY": "1",
                    "SGLANG_PERFPF_STAGING_GATHER_EVENT": "0",
                    "SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB": "0",
                },
            ),
        )
        enter_scope(
            self,
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                side_effect=lambda senders, *groups: [sender.poll() for sender in senders],
            ),
        )
        enter_scope(self, patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache"))
        self.messages = []
        self.room, self.session = 8478244407566187271, "127.0.0.1:10000"
        self.manager = manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.request_status = {self.room: KVPoll.Bootstrapping}
        manager.transfer_infos = {}
        manager.req_to_decode_prefix_len = {}
        manager.deferred_bootstrap = None
        manager.defer_decode_allocation = False
        manager.enable_trace = False
        manager.is_dummy_cp_rank = False
        manager.enable_all_cp_ranks_for_transfer = False
        manager.bootstrap_timeout = 600
        manager._dcp_pack_max_tokens = None
        manager._staging_outstanding = {}
        manager.enable_staging = True
        manager.attn_tp_size, manager.pp_rank = 2, 0
        manager.kv_args = SimpleNamespace(page_size=256, state_types=[])
        manager.kv_buffer_tensors = {"page_size": 256}
        manager._staging_ctx = PrefillStagingContext()
        manager._staging_ctx.prefetch_sockets["tcp://127.0.0.1:10000"] = SimpleNamespace(
            send_multipart=lambda msg: self.messages.append(msg)
        )
        manager.decode_kv_args_table = {
            self.session: SimpleNamespace(dst_attn_tp_size=8)
        }
        manager.transfer_queues = [SimpleQueue()]
        self.sender = MooncakeKVSender(manager, "127.0.0.1:8998", self.room)
        self.scheduler = scheduler = SchedulerDisaggregationPrefillMixin()
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=256, translate_kv_indices_for_transfer=lambda indices: indices
        )
        scheduler.kv_checksum_computer = None
        scheduler.enable_staging = True
        scheduler.disagg_metadata_buffers = Mock()
        scheduler.disagg_prefill_pending_chunk_rids = set()
        scheduler.scheduler_stage_metrics = None
        scheduler.attn_cp_cpu_group = scheduler.attn_tp_cpu_group = object()
        scheduler.tree_cache = Mock()
        scheduler.output_streamer = Mock()
        scheduler.waiting_queue = []
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.scheduler, queue.kv_manager = scheduler, manager
        scheduler.disagg_prefill_bootstrap_queue = queue

    def request(self, input_ids, prefix=0):
        """Also used by the external replay with checkpoint-tokenized requests."""
        req = Req(
            "late-bootstrap", "", array("q", input_ids), SamplingParams(),
            bootstrap_room=self.room,
        )
        req.disagg_kv_sender = self.sender
        req.pending_bootstrap, req.prefill_attempt_count = True, 1
        req.metadata_buffer_index, req.kv.req_pool_idx = 3, 0
        req.extend_end = len(input_ids)
        self.scheduler.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(len(input_ids), dtype=torch.int64).reshape(1, -1),
            available_size=lambda: 1,
        )
        self.scheduler.disagg_prefill_inflight_queue = [req]
        self.prefix = prefix
        return req

    def metadata_arrives(self, req):
        pages = (len(req.origin_input_ids) - self.prefix + 255) // 256
        info = TransferInfo(
            room=self.room, endpoint="127.0.0.1", dst_port=10000,
            mooncake_session_id=self.session, dst_kv_indices=np.arange(pages),
            dst_aux_index=3, dst_state_indices=[], required_dst_info_num=1,
            is_dummy=False, decode_prefix_len=self.prefix,
        )
        self.manager.transfer_infos[self.room] = {self.session: info}
        self.manager.req_to_decode_prefix_len[self.room] = self.prefix
        self.manager.update_status(self.room, KVPoll.WaitingForInput)
        return info

    def complete_staging(self, info, chunk):
        """Run actual request/response handlers, then the worker's ready/send path."""
        strategy = PrefillStagingStrategy(self.manager, object())
        start, count = chunk.index_slice.start, len(chunk.prefill_kv_indices)
        self.assertFalse(strategy.check_ready(info, start, count)[0])
        responses = []
        receiver = SimpleNamespace(
            chunk_staging_infos=[],
            _connect_to_bootstrap_server=lambda bi: (
                SimpleNamespace(send_multipart=lambda msg: responses.append(msg)),
                threading.Lock(),
            ),
        )
        allocator = Mock()
        allocator.assign.return_value = (0, 0, 0)
        decode_args = SimpleNamespace(
            page_size=256, kv_item_lens=[256 * 16, 256 * 16],
            total_kv_head_num=8, engine_rank=0,
        )
        for msg in self.messages:
            handle_staging_req(
                msg, allocator, decode_args, 8, 2, None,
                {self.room: receiver}, {self.room: [{"pp_rank": 0}]},
            )
        self.assertEqual(len(responses), len(self.messages))
        for msg in responses:
            handle_staging_rsp(msg, self.manager.transfer_infos)
        self.assertTrue(strategy.check_ready(info, start, count)[0])
        self.manager.send_kvcache_staged = Mock(return_value=0)
        ready_messages = []
        self.manager._send_multipart_locked = lambda ep, msg, **kw: ready_messages.append(msg)
        target = SimpleNamespace(
            staging_base_ptr=4096, staging_total_size=1 << 20,
            dst_tp_rank=0, dst_attn_tp_size=8, dst_kv_item_len=4096,
            dst_kv_layer_ids=None, staging=None,
        )
        self.assertEqual(
            self.manager._do_staging_transfer(
                strategy, chunk, info, target, info.dst_kv_indices,
                None, self.manager.transfer_queues[0], 0,
            ),
            (0, False),
        )
        self.assertEqual(ready_messages[0][0], b"CHUNK_READY")
        self.assertEqual(int(ready_messages[0][1]), self.room)

    def exercise_parked_request(self, input_ids):
        req = self.request(input_ids)
        self.scheduler.maybe_prefetch_staging_for_batch(SimpleNamespace(reqs=[req]))
        self.assertEqual(self.messages, [])  # No decode metadata before forward.
        self.scheduler.process_disagg_prefill_inflight_queue()
        self.assertTrue(self.manager.transfer_queues[0].empty())
        self.assertTrue(req.pending_bootstrap)
        info = self.metadata_arrives(req)
        self.scheduler.process_disagg_prefill_inflight_queue()
        self.assertFalse(req.pending_bootstrap)
        self.assertEqual(self.sender.num_kv_indices, 1)
        chunk = self.manager.transfer_queues[0].get_nowait()
        self.assertEqual(chunk.index_slice, slice(0, 1))
        self.assertTrue(chunk.is_last_chunk)
        self.assertEqual(chunk.prefill_aux_index, 3)
        self.assertEqual(chunk.num_kv_tokens, len(input_ids))
        self.assertEqual(len(self.messages), 1, "late bootstrap never requested staging")
        self.assertEqual(self.messages[0][:4], [
            b"STAGING_REQ", str(self.room).encode(), b"0", b"1",
        ])
        self.complete_staging(info, chunk)
        return req, chunk

    def test_parked_15_token_request(self):
        self.exercise_parked_request(list(range(15)))

    def test_lazy_suffix_prefix_and_page_rounding(self):
        # 512 cached tokens, then 512 + 512 + 1 suffix tokens. Requesting with
        # an absolute end would reserve chunk 1 when only chunk 0 can be sent.
        req = self.request(list(range(1537)), prefix=512)
        self.metadata_arrives(req)
        self.assertTrue(self.scheduler.handle_pending_bootstrap(req, KVPoll.WaitingForInput))
        for index, end in enumerate((1024, 1536, 1537)):
            self.scheduler.send_kv_chunk(req, last_chunk=index == 2, end_idx=end)
            self.assertEqual(len(self.messages), index + 1)
            chunk = self.manager.transfer_queues[0].get_nowait()
            self.assertEqual(chunk.index_slice, (slice(0, 2), slice(2, 4), slice(4, 5))[index])
            self.assertEqual(chunk.is_last_chunk, index == 2)
            self.assertEqual(int(self.messages[-1][2]), index)
            self.assertEqual(int(self.messages[-1][3]), (2, 2, 1)[index])
            np.testing.assert_array_equal(chunk.prefill_kv_indices, ([2, 3], [4, 5], [6])[index])
        self.assertEqual(self.sender.num_kv_indices, 5)
        self.assertEqual(req.start_send_idx, 1537)

    def test_existing_prefetch_is_not_requested_twice(self):
        req = self.request(list(range(15)))
        self.metadata_arrives(req)
        self.scheduler.maybe_prefetch_staging_for_batch(SimpleNamespace(reqs=[req]))
        before = list(self.messages)
        self.scheduler.process_disagg_prefill_inflight_queue()
        self.assertEqual(len(before), 1)
        self.assertEqual(self.messages, before)

    def test_nonlazy_preserves_full_grid_prefetch(self):
        req = self.request(list(range(1025)))
        self.metadata_arrives(req)
        self.scheduler.handle_pending_bootstrap(req, KVPoll.WaitingForInput)
        with patch.dict("os.environ", {"SGLANG_PERFPF_STAGING_LAZY": "0"}):
            self.scheduler.send_kv_chunk(req, end_idx=512)
        self.assertEqual([int(msg[2]) for msg in self.messages], [0, 1, 2])

    def test_disabled_or_same_tp_does_not_allocate_staging(self):
        for enabled, dst_tp in ((False, 8), (True, 2)):
            with self.subTest(enabled=enabled, dst_tp=dst_tp):
                self.manager.enable_staging = enabled
                self.manager.decode_kv_args_table[self.session].dst_attn_tp_size = dst_tp
                self.manager.transfer_infos[self.room] = {self.session: SimpleNamespace(
                    is_dummy=False, mooncake_session_id=self.session,
                )}
                self.manager.add_transfer_request(self.room, [0], slice(0, 1), False)
                self.assertEqual(self.messages, [])
                self.assertFalse(self.manager.transfer_queues[0].empty())

    def test_failed_or_missing_room_does_not_allocate_staging(self):
        req = self.request(list(range(15)))
        self.metadata_arrives(req)
        self.manager.update_status(self.room, KVPoll.Failed)
        self.manager.add_transfer_request(self.room, [0], slice(0, 1), False)
        self.manager.request_status[self.room] = KVPoll.WaitingForInput
        self.manager.transfer_infos.clear()
        self.manager.add_transfer_request(self.room, [0], slice(0, 1), False)
        self.assertEqual(self.messages, [])
        self.assertTrue(self.manager.transfer_queues[0].empty())


if __name__ == "__main__":
    unittest.main()
