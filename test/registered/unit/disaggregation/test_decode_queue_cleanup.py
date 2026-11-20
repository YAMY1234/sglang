import os
import unittest
from collections import Counter
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.decode_hicache_mixin import (
    HiCacheRestoreGatedKVReceiver,
)
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_override

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


def _waiting_decode_req(rid, input_len=8):
    req = MagicMock()
    req.rid = rid
    req.origin_input_ids = list(range(input_len))
    req.output_ids = [99]
    req.finished_reason = None
    req.pd_rebootstrap_in_progress = False
    req.sampling_params.max_new_tokens = 16
    return SimpleNamespace(
        req=req, waiting_for_input=True, is_rebootstrap=False, kv_receiver=None
    )


def _reserved(extend_end, host_staged=False):
    return SimpleNamespace(
        req=SimpleNamespace(extend_end=extend_end), host_staged=host_staged
    )


class TestDecodePreallocCap(CustomTestCase):
    """SGLANG_DISAGG_DECODE_PREALLOC_CAP bounds the KV reserved for requests
    whose prefill KV has not arrived; the rest wait without a reservation."""

    def setUp(self):
        _run_as_single_rank(self)
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        mamba = patch(
            "sglang.srt.disaggregation.decode.ensure_mamba_capacity",
            return_value=True,
        )
        mamba.start()
        self.addCleanup(mamba.stop)

    def _queue(self, decode_reqs, reserved, max_total_num_tokens=100):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = list(decode_reqs)
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue.max_total_num_tokens = max_total_num_tokens
        queue.num_reserved_decode_tokens = 0
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue._uses_swa_reservation = MagicMock(return_value=False)
        queue._allocatable_token_budgets = MagicMock(return_value=10**9)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        queue._prealloc_reservation_fits = MagicMock(return_value=True)
        queue._pre_alloc = MagicMock(return_value=torch.arange(8))
        queue._send_kv_metadata = MagicMock()
        queue.transfer_queue = SimpleNamespace(queue=list(reserved))
        queue.tree_cache = MagicMock()
        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 8
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 8
        queue.token_to_kv_pool = MagicMock()
        queue.token_to_kv_pool_allocator = MagicMock(page_size=1)
        queue.kv_manager = SimpleNamespace(kv_args=SimpleNamespace(state_types=[]))
        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_hisparse = False
        scheduler.enable_lora = False
        scheduler.enable_decode_hicache = False
        scheduler.enable_priority_scheduling = False
        queue.scheduler = scheduler
        return queue

    def _pop(self, queue, cap):
        env = {} if cap is None else {"SGLANG_DISAGG_DECODE_PREALLOC_CAP": cap}
        with patch.dict(os.environ, env):
            return queue.pop_preallocated()

    def test_unset_cap_reserves_every_ready_request(self):
        reqs = [_waiting_decode_req("a"), _waiting_decode_req("b")]
        queue = self._queue(reqs, reserved=[_reserved(90)])

        preallocated, failed = self._pop(queue, cap=None)

        self.assertEqual(preallocated, reqs)
        self.assertEqual(failed, [])
        self.assertEqual(queue.queue, [])

    def test_cap_counts_reservations_made_in_the_same_pass(self):
        first, second = _waiting_decode_req("a"), _waiting_decode_req("b")
        queue = self._queue([first, second], reserved=[])

        # 8 + 8 tokens exceed 0.1 x 100; the head is admitted with nothing held.
        preallocated, failed = self._pop(queue, cap="0.1")

        self.assertEqual(preallocated, [first])
        self.assertEqual(failed, [])
        self.assertEqual(queue.queue, [second])
        queue._pre_alloc.assert_called_once()

    def test_cap_counts_device_reservations_in_the_transfer_queue(self):
        first, second = _waiting_decode_req("a"), _waiting_decode_req("b")
        # Host-staged receives hold no device KV and are not counted.
        reserved = [_reserved(30), _reserved(1000, host_staged=True)]
        queue = self._queue([first, second], reserved=reserved)

        # 30 + 8 fits 0.4 x 100; 30 + 8 + 8 does not.
        preallocated, _ = self._pop(queue, cap="0.4")

        self.assertEqual(preallocated, [first])
        self.assertEqual(queue.queue, [second])

    def test_cap_waits_before_matching_or_locking_a_prefix(self):
        enter_override(
            self,
            get_context().override_server_args(
                disaggregation_decode_enable_radix_cache=True
            ),
        )
        waiting = _waiting_decode_req("a")
        queue = self._queue([waiting], reserved=[_reserved(60)])
        queue._match_prefix_and_lock = MagicMock(
            side_effect=AssertionError("capped request must not lock a prefix")
        )

        preallocated, failed = self._pop(queue, cap="0.5")

        self.assertEqual((preallocated, failed), ([], []))
        self.assertEqual(queue.queue, [waiting])
        queue._pre_alloc.assert_not_called()
        queue.tree_cache.unlock.assert_not_called()

    def test_break_diag_counts_each_stop_reason(self):
        first, second = _waiting_decode_req("a"), _waiting_decode_req("b")
        queue = self._queue([first, second], reserved=[])
        queue.prealloc_break_counts = Counter()  # SGLANG_DISAGG_DECODE_PREALLOC_BREAK_DIAG

        self._pop(queue, cap="0.1")  # the second waits on the cap
        queue._prealloc_reservation_fits.return_value = False
        self._pop(queue, cap=None)  # then on the KV pool
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 0
        self._pop(queue, cap=None)  # then on the metadata buffers

        self.assertEqual(queue.queue, [second])
        self.assertEqual(
            queue.prealloc_break_counts, Counter(cap=1, kv_fit=1, metadata=1)
        )

    def test_break_diag_off_keeps_no_counts(self):
        queue = self._queue([_waiting_decode_req("a")], reserved=[])
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 0

        self.assertEqual(self._pop(queue, cap=None), ([], []))
        self.assertIsNone(queue.prealloc_break_counts)


if __name__ == "__main__":
    unittest.main()
