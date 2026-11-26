"""Decode late binding: handoff target rules, eligibility and the D->D protocol."""

import pickle
import time
import unittest
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import msgspec
import zmq

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.late_binding import (
    DecodeLateBinding,
    LateBindingAck,
    LateBindingCancel,
    LateBindingHandoff,
    choose_handoff_target,
    choose_handoff_target_dominant,
)
from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.scheduler_components.dp_attn import (
    LATE_BINDING_INACTIVE_ROW,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _rows(*committed, kv=0, target=-1):
    return [(c, kv, target) for c in committed]


class _RecvReq(msgspec.Struct):
    rid: str
    disagg_prefill_dp_rank: Optional[int] = None


class TestChooseHandoffTarget(CustomTestCase):
    def test_threshold_gap_to_least_committed_rank(self):
        rows = _rows(5, 3, 4)
        self.assertIsNone(choose_handoff_target(rows, 0, threshold=3))
        self.assertEqual(choose_handoff_target(rows, 0, threshold=2), 1)
        # Only a rank at least `threshold` above the minimum is a source.
        self.assertIsNone(choose_handoff_target(rows, 2, threshold=2))
        self.assertIsNone(choose_handoff_target(rows, 1, threshold=2))

    def test_concurrent_sources_take_distinct_targets(self):
        rows = _rows(9, 8, 1, 2)
        self.assertEqual(choose_handoff_target(rows, 0, threshold=2), 2)
        self.assertEqual(choose_handoff_target(rows, 1, threshold=2), 3)

    def test_announced_handoff_counts_on_its_target(self):
        rows = [(4, 0, -1), (2, 0, -1), (3, 0, 1)]
        # Rank 2 announced a handoff to rank 1: 4 vs 3 is below the threshold.
        self.assertIsNone(choose_handoff_target(rows, 0, threshold=2))

    def test_inactive_rank_is_never_source_or_target(self):
        rows = _rows(5, 3) + [LATE_BINDING_INACTIVE_ROW]
        self.assertEqual(choose_handoff_target(rows, 0, threshold=2), 1)
        self.assertIsNone(choose_handoff_target(rows, 2, threshold=2))
        self.assertIsNone(
            choose_handoff_target(_rows(5) + [LATE_BINDING_INACTIVE_ROW], 0, 1)
        )

    def test_disabled_threshold_or_unknown_rank(self):
        self.assertIsNone(choose_handoff_target(_rows(9, 0), 0, threshold=0))
        self.assertIsNone(choose_handoff_target(_rows(9, 0), 2, threshold=1))


class TestChooseHandoffTargetDominant(CustomTestCase):
    def test_light_kv_matches_count_rule(self):
        for self_rank in range(3):
            rows = _rows(9, 3, 5, kv=10)
            self.assertEqual(
                choose_handoff_target_dominant(rows, self_rank, 2, cap=100),
                choose_handoff_target(rows, self_rank, 2),
            )

    def test_kv_bound_rank_hands_off_at_equal_counts(self):
        rows = [(4, 900, -1), (4, 100, -1)]
        self.assertIsNone(choose_handoff_target(rows, 0, threshold=2))
        # Scores: max(4, 0.9 * 10) = 9 and max(4, 0.1 * 10) = 4.
        self.assertEqual(choose_handoff_target_dominant(rows, 0, 2, cap=10), 1)

    def test_move_must_leave_target_below_source(self):
        rows = [(4, 900, -1), (4, 100, -1)]
        # The request takes 80% of the target's pool: (0.1 + 0.8) * 10 >= 9.
        self.assertIsNone(
            choose_handoff_target_dominant(rows, 0, 2, cap=10, req_kv_frac=0.8)
        )


class TestLateBindingEligibility(CustomTestCase):
    def _eligible(self, session_radix=False, is_rebootstrap=False, **req_fields):
        req = SimpleNamespace(
            bootstrap_host="prefill",
            late_binding_recv_req=object(),
            multimodal_inputs=None,
            finished_reason=None,
            session=None,
            session_id=None,
        )
        for key, value in req_fields.items():
            setattr(req, key, value)
        queue = SimpleNamespace(
            scheduler=SimpleNamespace(enable_session_radix_cache=session_radix)
        )
        decode_req = SimpleNamespace(req=req, is_rebootstrap=is_rebootstrap)
        return DecodePreallocQueue._late_binding_eligible(queue, decode_req)

    def test_plain_request_is_eligible(self):
        self.assertTrue(self._eligible())
        # A session id alone keeps no state on a decode without radix sessions.
        self.assertTrue(self._eligible(session_id="s"))

    def test_requests_that_stay_on_their_rank(self):
        for case in (
            dict(is_rebootstrap=True),
            dict(bootstrap_host=FAKE_BOOTSTRAP_HOST),
            dict(late_binding_recv_req=None),
            dict(multimodal_inputs=object()),
            dict(finished_reason=FINISH_ABORT()),
            dict(session=object()),
            dict(session_id="s", session_radix=True),
        ):
            with self.subTest(case=case):
                self.assertFalse(self._eligible(**case))


class TestDecodeLateBindingProtocol(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.ctx = zmq.Context()
        self.addCleanup(self.ctx.destroy, linger=0)
        self.source = self._rank(0)
        self.target = self._rank(1)

    def _rank(self, dp_rank):
        late_binding = DecodeLateBinding(
            dp_rank=dp_rank,
            num_dp_ranks=2,
            threshold=2,
            local_ip="127.0.0.1",
            ack_timeout=10.0,
            context=self.ctx,
        )
        late_binding.sent = []

        def send_bytes(peer, data):
            late_binding.sent.append((peer, pickle.loads(data)))
            return True

        late_binding._send_bytes = send_bytes
        return late_binding

    def _handoff(self, rid="r1"):
        decode_req = SimpleNamespace(late_binding_hops=0)
        handoff = LateBindingHandoff(
            recv_req=_RecvReq(rid=rid),
            room=7,
            old_endpoint=("decode", 2),
            source_dp_rank=0,
            target_dp_rank=1,
            sent_at=time.time(),
            handoff_id=self.source.new_handoff_id(7),
        )
        self.assertTrue(self.source.send_handoff(handoff, decode_req))
        return handoff, decode_req

    def _admit(self, handoff, present=False, enqueued=True):
        enqueue = Mock(
            side_effect=lambda recv_req: (
                enqueued and self.target.incoming.pop(recv_req.rid)
            )
        )
        abort = Mock()
        admitted = self.target.admit(
            handoff, rid_present=lambda rid: present, enqueue=enqueue, abort=abort
        )
        return admitted, enqueue, abort

    def test_decide_once_per_snapshot_and_once_per_request(self):
        decode_req = SimpleNamespace(late_binding_hops=0)
        self.assertIsNone(self.source.decide(decode_req))
        self.source.on_rows([[5, 0, -1], [1, 0, -1]])
        self.assertEqual(self.source.decide(decode_req), 1)
        self.assertIsNone(self.source.decide(decode_req))
        self.source.on_rows([[5, 0, -1], [1, 0, -1]])
        self.assertIsNone(self.source.decide(SimpleNamespace(late_binding_hops=1)))
        self.assertEqual(self.source.decide(decode_req), 1)

    def test_handoff_is_announced_in_next_row(self):
        self.assertEqual(self.source.local_row(3, 0.25), (3, 250, -1))
        self._handoff()
        self.assertEqual(self.source.local_row(3, 0.25), (3, 250, 1))
        self.source.on_rows([[3, 250, 1], [1, 0, -1]])
        self.assertEqual(self.source.local_row(3, 0.25), (3, 250, -1))

    def test_ack_settles_handoff(self):
        handoff, _ = self._handoff()
        admitted, enqueue, abort = self._admit(handoff)
        self.assertTrue(admitted)
        enqueue.assert_called_once_with(handoff.recv_req)
        abort.assert_not_called()
        [(peer, ack)] = self.target.sent
        self.assertEqual(peer, 0)
        self.assertEqual(ack, LateBindingAck("r1", 7, handoff.handoff_id, True))
        self.source.on_timeout = Mock()
        self.source._on_ack(ack)
        self.assertEqual(self.source.pending_acks, {})
        self.source.check_timeouts(now=time.monotonic() + 100)
        self.source.on_timeout.assert_not_called()

    def test_target_refuses_duplicate_or_unqueued_request(self):
        for present, enqueued in ((True, True), (False, False)):
            with self.subTest(present=present, enqueued=enqueued):
                handoff, _ = self._handoff()
                self.target.sent.clear()
                admitted, _, _ = self._admit(handoff, present, enqueued)
                self.assertFalse(admitted)
                self.assertFalse(self.target.sent[0][1].ok)
                self.assertEqual(self.target.incoming, {})

    def _timed_out_handoff(self):
        handoff, decode_req = self._handoff()
        self.source.on_timeout = Mock()
        self.source.check_timeouts(now=time.monotonic() + 11)
        self.source.on_timeout.assert_called_once_with(handoff, decode_req)
        self.assertEqual(self.source.pending_acks, {})
        [(peer, cancel)] = self.source.sent[1:]
        self.assertEqual(peer, 1)
        return handoff, cancel

    def test_timeout_fails_source_and_cancels_target(self):
        handoff, cancel = self._timed_out_handoff()
        self.assertEqual(cancel, LateBindingCancel("r1", 7, handoff.handoff_id, 0))
        # A late ACK no longer changes the outcome.
        self.source._on_ack(LateBindingAck("r1", 7, handoff.handoff_id, True))
        self.assertEqual(self.source.pending_acks, {})

    def test_cancel_before_arrival_fences_handoff(self):
        handoff, cancel = self._timed_out_handoff()
        self.target.arrivals.append(handoff)
        self.target.on_cancel(cancel)
        self.assertEqual(self.target.take_arrivals(), [])
        admitted, enqueue, _ = self._admit(handoff)
        self.assertFalse(admitted)
        enqueue.assert_not_called()

    def test_cancel_after_enqueue_aborts_target_copy(self):
        handoff, cancel = self._timed_out_handoff()
        self.assertTrue(self._admit(handoff)[0])
        self.target.on_cancel(cancel)
        [abort] = self.target.take_arrivals()
        self.assertIsInstance(abort, AbortReq)
        self.assertEqual(abort.rid, "r1")

    def test_abort_seen_in_transit_is_replayed_on_admit(self):
        handoff, _ = self._handoff()
        early = AbortReq(rid="r1")
        self.target.recent_aborts.append((early, handoff.sent_at - 1))
        self.assertIsNone(self.target.aborted_in_transit(handoff))
        late = AbortReq(rid="r1")
        self.target.note_abort(late)
        admitted, _, abort = self._admit(handoff)
        self.assertTrue(admitted)
        abort.assert_called_once_with(late)


if __name__ == "__main__":
    unittest.main()
