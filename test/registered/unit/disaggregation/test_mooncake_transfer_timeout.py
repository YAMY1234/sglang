"""SGLANG_DISAGGREGATION_TRANSFER_TIMEOUT in the Mooncake transfer worker: a room whose
CUDA wait never completes must not hold every room queued behind it."""

import threading
import unittest
from collections import defaultdict
from unittest.mock import Mock

import numpy as np
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.utils import FastQueue, TransferKVChunk
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, TransferInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

STUCK, BEHIND = 11, 12


class _NeverDoneEvent:
    """A CUDA event whose work never finishes until the test releases it."""

    def __init__(self):
        self.release = threading.Event()

    def query(self) -> bool:
        return self.release.is_set()

    def synchronize(self) -> None:
        self.release.wait()


def _manager(timeout: float) -> MooncakeKVManager:
    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager.transfer_timeout = timeout
    manager.room_rids = {STUCK: "r-stuck", BEHIND: "r-behind"}
    manager._timed_out_rooms = set()
    manager.pp_rank = 3
    manager.enable_trace = manager.enable_staging = False
    manager.enable_deferred_decode_kv_release = False
    manager.settle_diag = None
    manager.deferred_bootstrap = None
    manager.failure_lock = threading.Lock()
    manager.failure_records = {}
    manager._staging_outstanding = defaultdict(int)
    manager.req_to_decode_prefix_len = {}
    manager.request_status = {STUCK: KVPoll.Transferring, BEHIND: KVPoll.Transferring}
    manager.transfer_infos = {
        room: {
            "dummy": TransferInfo(
                room=room,
                endpoint="10.0.0.2",
                dst_port=1234,
                mooncake_session_id="10.0.0.2:15000",
                dst_kv_indices=np.array([], dtype=np.int32),
                dst_aux_index=0,
                dst_state_indices=None,
                required_dst_info_num=1,
                is_dummy=True,
            )
        }
        for room in (STUCK, BEHIND)
    }
    manager.send_kv_status_message = Mock()
    return manager


def _chunk(room: int, wait_event=None) -> TransferKVChunk:
    chunk = TransferKVChunk(
        room=room,
        prefill_kv_indices=np.array([], dtype=np.int32),
        index_slice=slice(0, 0),
        is_last_chunk=True,
        prefill_aux_index=None,
        state_indices=None,
    )
    chunk.wait_event = wait_event
    return chunk


def _start_worker(manager, event) -> threading.Thread:
    queue = FastQueue()
    queue.put(_chunk(STUCK, wait_event=event))
    queue.put(_chunk(BEHIND))
    queue.put(None)
    worker = threading.Thread(
        target=manager.transfer_worker,
        args=(queue,),
        kwargs={"executor": None},
        daemon=True,
    )
    worker.start()
    return worker


class TestMooncakeTransferTimeout(CustomTestCase):
    def test_without_the_timeout_a_stuck_room_holds_the_rooms_behind_it(self):
        manager = _manager(timeout=0)
        event = _NeverDoneEvent()
        worker = _start_worker(manager, event)
        worker.join(0.3)
        self.assertTrue(worker.is_alive())
        self.assertEqual(manager.request_status[BEHIND], KVPoll.Transferring)
        event.release.set()  # the old behaviour only recovers if the wait ever ends
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(manager.request_status[STUCK], KVPoll.Success)
        self.assertEqual(manager.request_status[BEHIND], KVPoll.Success)

    def test_the_timeout_fails_only_the_stuck_room_and_the_rest_complete(self):
        manager = _manager(timeout=0.2)
        event = _NeverDoneEvent()
        with self.assertLogs(
            "sglang.srt.disaggregation.common.conn", level="ERROR"
        ) as logs:
            worker = _start_worker(manager, event)
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(manager.request_status[STUCK], KVPoll.Failed)
        self.assertEqual(manager.request_status[BEHIND], KVPoll.Success)
        self.assertIn("timed out", manager.failure_records[STUCK])
        self.assertNotIn(BEHIND, manager.failure_records)
        (line,) = [r for r in logs.output if "PD transfer timeout" in r]
        self.assertIn(
            f"room {STUCK} rid r-stuck stage 3 stuck in transfer worker wait_event",
            line,
        )
        # Decode is told once, for the stuck room only.
        failed = [
            c
            for c in manager.send_kv_status_message.call_args_list
            if c.kwargs.get("status") == KVPoll.Failed
        ]
        self.assertEqual([c.kwargs["bootstrap_room"] for c in failed], [STUCK])
        # A second timeout report for the same room is ignored.
        manager.fail_room_on_timeout(STUCK, "transfer worker wait_event", 9.9)
        self.assertEqual(
            len(manager.send_kv_status_message.call_args_list), len(failed)
        )
        event.release.set()


if __name__ == "__main__":
    unittest.main()
