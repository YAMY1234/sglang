import unittest
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    HiCacheRestoreResult,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.decode_hicache_mixin import (
    HiCacheRestoreGatedKVReceiver,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=13, suite="base-a-test-cpu")


def _run_as_single_rank(test: unittest.TestCase) -> None:
    # The transfer queue sizes its release collectives from the gloo group;
    # these tests have no process group, so answer as one rank.
    world_size = patch("torch.distributed.get_world_size", return_value=1)
    world_size.start()
    test.addCleanup(world_size.stop)


class TestDecodeQueueCleanup(CustomTestCase):
    def setUp(self):
        _run_as_single_rank(self)
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def test_ensure_prefill_info_tolerates_cleared_receiver(self):
        # A req whose kv_receiver was already cleared must not crash on .abort().
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._max_ensure_retries = 1
        queue._ensure_retry_interval = 0
        queue._ensure_retry_count = {"127.0.0.1:11500": 0}
        queue._ensure_last_attempt_time = {}
        queue.kv_manager = MagicMock()
        queue.kv_manager.try_ensure_parallel_info.return_value = False

        cleared_req = SimpleNamespace(
            req=SimpleNamespace(rid="cleared"), kv_receiver=None
        )
        addr_to_reqs = {"127.0.0.1:11500": [cleared_req]}

        ready, remaining = queue._ensure_prefill_info(addr_to_reqs)

        self.assertEqual(ready, {})
        self.assertEqual(remaining, [])

    def test_prefetches_prefill_dp_rank_query(self):
        addr = "127.0.0.1:11500"
        executor = MagicMock()
        future = Future()
        future.set_result({"7": 1})
        executor.submit.return_value = future

        def decode_req(room):
            return SimpleNamespace(
                req=SimpleNamespace(
                    bootstrap_host="127.0.0.1",
                    bootstrap_port=11500,
                    bootstrap_room=room,
                ),
                kv_receiver=MagicMock(),
            )

        first = decode_req(7)
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pending_reqs = [first]
        queue._prefill_dp_rank_queries = {}
        queue.kv_manager = SimpleNamespace(
            prefill_info_table={addr: object()},
            _ensure_prefill_recompute_executor=lambda: executor,
        )
        queue._resolve_prefill_dp_rank = MagicMock(return_value=None)
        queue._ensure_prefill_info = lambda groups: (groups, [])

        queue.prefetch_prefill_dp_rank_queries()
        tail = decode_req(8)
        queue.pending_reqs.append(tail)
        with patch(
            "sglang.srt.disaggregation.decode.CommonKVReceiver.query_prefill_dp_ranks",
            return_value={"8": 2},
        ) as query:
            queue._resolve_pending_reqs()

        _, called_addr, called_rooms = executor.submit.call_args.args
        self.assertEqual((called_addr, called_rooms), (addr, [7]))
        query.assert_called_once_with(addr, [8])
        first.kv_receiver.init.assert_called_once_with(1)
        tail.kv_receiver.init.assert_called_once_with(2)
        self.assertEqual(queue.pending_reqs, [])


class TestDeferredPreallocMetadata(CustomTestCase):
    def test_metadata_send_waits_for_nonblocking_index_copies(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.alloc.return_value = 7
        queue.req_to_token_pool = SimpleNamespace(
            req_index_to_mamba_index_mapping=torch.tensor(
                [[9], [11], [13]], dtype=torch.int32
            ),
            translate_mamba_indices=lambda indices: indices,
        )
        queue.transfer_queue = SimpleNamespace(enable_staging=False)
        queue._deferred_sends = deque()
        queue._num_published_destinations = 0
        event = MagicMock()
        event.query.return_value = False
        current_stream = object()
        queue.scheduler = SimpleNamespace(
            device_module=SimpleNamespace(
                Event=MagicMock(return_value=event),
                current_stream=MagicMock(return_value=current_stream),
            )
        )

        receiver = SimpleNamespace(require_staging=False, send_metadata=MagicMock())
        decode_req = SimpleNamespace(
            req=SimpleNamespace(
                bootstrap_room=3,
                finished_reason=None,
                kv=SimpleNamespace(req_pool_idx=1),
                time_stats=MagicMock(),
            ),
            kv_receiver=receiver,
            metadata_buffer_index=-1,
            is_rebootstrap=False,
        )
        kv_indices = torch.arange(16, 24, dtype=torch.int64)
        with patch(
            "sglang.srt.disaggregation.decode.async_d2h",
            side_effect=lambda tensor: tensor.cpu(),
        ) as copy_to_host:
            queue._defer_kv_metadata(
                decode_req,
                kv_indices,
                4,
                has_state=True,
                metadata_kwargs={"decode_prefix_len": 5},
            )

        self.assertEqual(decode_req.metadata_buffer_index, 7)
        self.assertEqual(queue._num_published_destinations, 1)
        decode_req.req.time_stats.set_decode_transfer_queue_entry_time.assert_called_once()
        self.assertEqual(copy_to_host.call_count, 2)
        queue.scheduler.device_module.Event.assert_called_once_with()
        queue.scheduler.device_module.current_stream.assert_called_once_with()
        event.record.assert_called_once_with(current_stream)

        queue.flush_deferred_sends()
        receiver.send_metadata.assert_not_called()
        self.assertEqual(len(queue._deferred_sends), 1)

        event.query.return_value = True
        queue.flush_deferred_sends()
        self.assertEqual(len(queue._deferred_sends), 0)
        self.assertFalse(decode_req.metadata_send_pending)
        self.assertFalse(decode_req.metadata_send_cancelled)
        args, kwargs = receiver.send_metadata.call_args
        np.testing.assert_array_equal(args[0], np.array([4, 5], dtype=np.int32))
        self.assertEqual(args[1], 7)
        np.testing.assert_array_equal(args[2][0][0], np.array([11], dtype=np.int32))
        self.assertEqual(kwargs, {"decode_prefix_len": 5})

    def test_abort_before_copy_completion_cancels_metadata_send(self):
        event = MagicMock()
        event.query.return_value = False
        receiver = SimpleNamespace(
            conclude_state=KVPoll.Failed,
            send_metadata=MagicMock(),
        )
        decode_req = SimpleNamespace(
            req=SimpleNamespace(finished_reason=None),
            kv_receiver=receiver,
            metadata_buffer_index=7,
            metadata_send_pending=True,
            metadata_send_cancelled=False,
            is_rebootstrap=False,
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._deferred_sends = deque(
            [(decode_req, [torch.tensor([4, 5])], False, {}, event)]
        )

        queue.flush_deferred_sends()

        receiver.send_metadata.assert_not_called()
        self.assertEqual(len(queue._deferred_sends), 1)
        self.assertTrue(decode_req.metadata_send_pending)
        self.assertTrue(decode_req.metadata_send_cancelled)

        event.query.return_value = True
        queue.flush_deferred_sends()

        receiver.send_metadata.assert_not_called()
        self.assertFalse(queue._deferred_sends)
        self.assertFalse(decode_req.metadata_send_pending)
        self.assertTrue(decode_req.metadata_send_cancelled)


class TestDeferredPdWorkScheduling(unittest.TestCase):
    def test_pp_drains_deferred_work_before_transfer_poll(self):
        order = []
        scheduler = SchedulerPPMixin()
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            flush_deferred_sends=lambda: order.append("metadata")
        )
        scheduler._flush_staging_scatters = lambda: order.append("scatter")
        scheduler.pp_group = SimpleNamespace(is_first_rank=True)
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])

        def get_rids(*_args, **_kwargs):
            order.append("poll")
            return []

        scheduler.get_rids = get_rids

        self.assertEqual(scheduler._pp_pd_get_decode_transferred_ids(), [])
        self.assertEqual(order, ["metadata", "scatter", "poll"])

    def test_staging_scatter_uses_the_stream_that_runs_the_forward(self):
        cases = (
            (False, 1, "schedule"),
            (True, 1, "forward"),
            (False, 2, "forward"),
        )
        for enable_overlap, pp_size, expected in cases:
            with self.subTest(enable_overlap=enable_overlap, pp_size=pp_size):
                handler = MagicMock()
                scheduler = SimpleNamespace(
                    enable_overlap=enable_overlap,
                    forward_stream="forward",
                    schedule_stream="schedule",
                    disagg_decode_transfer_queue=SimpleNamespace(
                        staging_handler=handler
                    ),
                )
                with patch(
                    "sglang.srt.disaggregation.decode.get_parallel",
                    return_value=SimpleNamespace(pp_size=pp_size),
                ):
                    SchedulerDisaggregationDecodeMixin._flush_staging_scatters(
                        scheduler
                    )
                handler.flush_pending_scatters.assert_called_once_with(expected)


class _GateReceiver:
    def __init__(self, poll):
        self._poll = poll
        self.require_staging = False

    def poll(self):
        return self._poll

    def clear(self):
        pass

    def failure_exception(self):
        raise KVTransferError(
            7, "Failed due to an unknown reason from another rank", True
        )


def _gated_decode_req(poll, restore_status):
    return SimpleNamespace(
        req=SimpleNamespace(rid="r", bootstrap_room=7, return_logprob=False),
        kv_receiver=_GateReceiver(poll),
        metadata_buffer_index=3,
        hicache_restore_status=restore_status,
    )


class TestDecodeHiCacheRestoreGate(CustomTestCase):
    def setUp(self):
        super().setUp()
        _run_as_single_rank(self)

    def test_gate_folds_restore_outcome_into_terminal_poll_only(self):
        cases = [
            (KVPoll.Success, HiCacheRestoreResult.READY, KVPoll.Success),
            (KVPoll.Success, HiCacheRestoreResult.PENDING, KVPoll.Transferring),
            (KVPoll.Success, HiCacheRestoreResult.FAILED, KVPoll.Failed),
            (KVPoll.Transferring, HiCacheRestoreResult.FAILED, KVPoll.Transferring),
            (
                KVPoll.WaitingForInput,
                HiCacheRestoreResult.FAILED,
                KVPoll.WaitingForInput,
            ),
            (KVPoll.Failed, HiCacheRestoreResult.PENDING, KVPoll.Failed),
        ]
        for poll, status, expected in cases:
            with self.subTest(poll=poll, status=status):
                gated = HiCacheRestoreGatedKVReceiver(_gated_decode_req(poll, status))
                self.assertEqual(gated.poll(), expected)

    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    def test_cancelled_deferred_metadata_releases_without_drain_wait(
        self, _prepare_abort
    ):
        receiver = MagicMock()
        receiver.failure_exception.side_effect = KVTransferError(
            7, "aborted before metadata send", False
        )
        receiver.kv_mgr.enable_deferred_decode_kv_release = True
        receiver.abort_notified = True
        decode_req = SimpleNamespace(
            req=SimpleNamespace(
                rid="cancelled",
                bootstrap_room=7,
                return_logprob=False,
            ),
            kv_receiver=receiver,
            metadata_buffer_index=3,
            metadata_send_cancelled=True,
            hicache_restore_status=HiCacheRestoreResult.PENDING,
            host_staged=False,
        )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.enable_staging = False
        queue.enable_host_receive = False
        queue.enable_deferred_kv_release = True
        queue.gloo_group = MagicMock()
        queue.tp_rank = 0
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.metadata_buffers = MagicMock()
        queue._poll_with_metadata_gate = MagicMock(return_value=[KVPoll.Failed])
        queue._clean_hicache_prefetch_resources = MagicMock()
        queue._record_transfer_failure = MagicMock()
        queue._release_request = MagicMock()
        queue._defer_release = MagicMock()
        queue.scheduler = SimpleNamespace(
            enable_decode_hicache=False,
            enable_hisparse=False,
            output_streamer=MagicMock(),
        )

        self.assertEqual(queue.pop_transferred(), [])

        queue._defer_release.assert_not_called()
        queue._release_request.assert_called_once_with(decode_req)
        receiver.clear.assert_called_once()
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        self.assertEqual(queue.queue, [])

    def test_peer_held_cancelled_metadata_requires_no_abort_ack(self):
        receiver = SimpleNamespace(
            abort_notified=True,
            bootstrap_infos=[object(), object()],
        )
        decode_req = SimpleNamespace(
            kv_receiver=receiver,
            metadata_buffer_index=3,
            metadata_send_cancelled=True,
        )
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.deferred_kv_release_timeout = 30
        queue._deferred_releases = []
        queue._is_host_destination = MagicMock(return_value=False)

        queue._defer_release(decode_req)

        self.assertEqual(queue._deferred_releases[0][4], 0)

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    def test_pop_transferred_branches_on_reduced_poll_only(
        self, mock_prepare_abort, mock_release_kv_cache
    ):
        def make_queue(decode_req, peer_poll):
            queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
            queue.queue = [decode_req]
            queue.enable_staging = False
            queue.enable_host_receive = False
            queue.enable_deferred_kv_release = False
            queue.gloo_group = MagicMock()
            queue.req_to_metadata_buffer_idx_allocator = MagicMock()
            queue.tp_rank = 0
            queue.tree_cache = MagicMock()
            queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
            queue.metadata_buffers.clear_slot = lambda idx: (
                queue.metadata_buffers.bootstrap_room.__setitem__(idx, 0)
            )
            queue._clean_hicache_prefetch_resources = MagicMock()
            queue._commit_transfer_to_req = MagicMock()
            queue._record_transfer_failure = MagicMock()
            scheduler = MagicMock()
            scheduler.enable_decode_hicache = True
            scheduler.enable_hisparse = False
            scheduler.metrics_reporter.enable_metrics = False
            queue.scheduler = scheduler

            def reduce_with_peer(
                pollers, group, decode_reqs=None, metadata_buffers=None
            ):
                return [min(int(p.poll()), int(peer_poll)) for p in pollers]

            patcher = patch(
                "sglang.srt.disaggregation.decode.poll_and_all_reduce",
                side_effect=reduce_with_peer,
            )
            patcher.start()
            self.addCleanup(patcher.stop)
            return queue

        decode_req = _gated_decode_req(KVPoll.Success, HiCacheRestoreResult.FAILED)
        queue = make_queue(decode_req, peer_poll=KVPoll.Failed)
        self.assertEqual(queue.pop_transferred(), [])
        self.assertEqual(queue.queue, [])
        queue._commit_transfer_to_req.assert_not_called()
        queue._record_transfer_failure.assert_called_once_with("HiCache restore failed")
        mock_prepare_abort.assert_called_once()

        mock_prepare_abort.reset_mock()
        decode_req = _gated_decode_req(KVPoll.Success, HiCacheRestoreResult.READY)
        queue = make_queue(decode_req, peer_poll=KVPoll.Failed)
        self.assertEqual(queue.pop_transferred(), [])
        queue._commit_transfer_to_req.assert_not_called()
        queue._record_transfer_failure.assert_called_once_with(
            "Failed due to an unknown reason from another rank"
        )
        mock_prepare_abort.assert_called_once()

        mock_prepare_abort.reset_mock()
        mock_release_kv_cache.reset_mock()
        decode_req = _gated_decode_req(KVPoll.Transferring, HiCacheRestoreResult.FAILED)
        queue = make_queue(decode_req, peer_poll=KVPoll.Success)
        self.assertEqual(queue.pop_transferred(), [])
        self.assertEqual(queue.queue, [decode_req])
        mock_prepare_abort.assert_not_called()
        mock_release_kv_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
