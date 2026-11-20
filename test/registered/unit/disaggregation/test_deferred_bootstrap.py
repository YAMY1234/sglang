"""Lifecycle tests for the shared pre-allocation bootstrap rendezvous."""

import itertools
import random
import time
import unittest

from sglang.srt.disaggregation.common.bootstrap import DeferredBootstrap
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDeferredBootstrap(unittest.TestCase):
    def setUp(self):
        self.table = DeferredBootstrap(timeout=10)
        self.owner = object()
        self.endpoint = ("decode", 1234)

    def test_cancel_at_every_point_prevents_later_ready(self):
        for order in itertools.permutations(("open", "register", "cancel")):
            with self.subTest(order=order):
                table = DeferredBootstrap(10)
                notifications = []
                for action in order:
                    if action == "open":
                        table.open(1, self.owner)
                    elif action == "register":
                        notifications.append(table.register(1, self.endpoint))
                    else:
                        notifications.append(table.fail(1))
                self.assertTrue(table.rooms[1].failed)
                self.assertIsNone(table.complete(1, self.owner))
                self.assertIn((self.endpoint, True), notifications)

    def test_duplicate_source_cannot_replace_owner(self):
        self.table.open(1, self.owner)
        other = object()
        self.assertIsNone(self.table.open(1, other))
        self.assertIsNone(self.table.complete(1, other))
        self.assertIsNone(self.table.close(1, other))
        self.assertIs(self.table.rooms[1].owner, self.owner)

    def test_endpoint_registration_is_idempotent_not_replaceable(self):
        self.table.open(1, self.owner)
        self.table.complete(1, self.owner)
        self.assertEqual(self.table.register(1, self.endpoint), (self.endpoint, False))
        self.assertIsNone(self.table.register(1, self.endpoint))
        self.assertIsNone(self.table.complete(1, self.owner))
        other = ("other", 9999)
        self.assertEqual(self.table.register(1, other), (other, True))
        self.assertEqual(self.table.rooms[1].endpoint, self.endpoint)

    def test_success_resets_room_for_either_arrival_order(self):
        for receiver_first in (True, False):
            with self.subTest(receiver_first=receiver_first):
                self.table.open(1, self.owner)
                self.table.register(1, self.endpoint)
                self.table.complete(1, self.owner)
                # The transfer worker, not the release, tells decode of Success.
                self.assertIsNone(self.table.close(1, self.owner, success=True))
                self.assertNotIn(1, self.table.rooms)
                new_owner = object()
                if receiver_first:
                    self.assertIsNone(self.table.register(1, self.endpoint))
                state = self.table.open(1, new_owner)
                self.assertFalse(state.ready)
                self.assertFalse(state.failed)
                if not receiver_first:
                    self.assertIsNone(self.table.register(1, self.endpoint))
                # A repeated cleanup from the retired sender cannot close this one.
                self.assertIsNone(self.table.close(1, self.owner, success=True))
                self.assertEqual(
                    self.table.complete(1, new_owner), (self.endpoint, False)
                )
                self.table.close(1, new_owner, success=True)

    def test_next_request_registers_before_release(self):
        """Once decode hears Success it may resume the room's request before this
        side releases it. Its registration must start the next room, not attach
        to the finished one, whose readiness was already sent."""
        self.table.open(1, self.owner)
        self.table.register(1, self.endpoint)
        self.assertEqual(self.table.complete(1, self.owner), (self.endpoint, False))
        self.table.conclude(1)
        self.assertTrue(self.table.awaits_release(1))
        self.assertIsNone(self.table.register(1, self.endpoint))
        self.assertIsNone(self.table.open(1, object()))
        self.assertIsNone(self.table.close(1, self.owner, success=True))
        self.assertFalse(self.table.awaits_release(1))
        new_owner = object()
        state = self.table.open(1, new_owner)
        self.assertFalse(state.ready or state.failed)
        self.assertEqual(self.table.complete(1, new_owner), (self.endpoint, False))

    def test_next_request_cancelled_before_release_stays_cancelled(self):
        self.table.open(1, self.owner)
        self.table.register(1, self.endpoint)
        self.table.complete(1, self.owner)
        self.table.conclude(1)
        self.assertIsNone(self.table.register(1, self.endpoint))
        self.table.cancel(1)
        self.table.close(1, self.owner, success=True)
        self.assertTrue(self.table.open(1, object()).failed)

    def test_success_cannot_reset_cancelled_room(self):
        self.table.open(1, self.owner)
        self.table.fail(1)
        self.table.close(1, self.owner, success=True)
        self.assertEqual(self.table.register(1, self.endpoint), (self.endpoint, True))

    def test_no_lock_escapes_any_operation(self):
        for operation in (
            lambda: self.table.open(1, self.owner),
            lambda: self.table.register(1, self.endpoint),
            lambda: self.table.complete(1, self.owner),
            lambda: self.table.fail(1),
            lambda: self.table.cancel(1),
            lambda: self.table.close(1, self.owner),
        ):
            operation()
            self.assertTrue(self.table.lock.acquire(blocking=False))
            self.table.lock.release()


class TestDeferredBootstrapMultiEndpoint(unittest.TestCase):
    """A room of a TP1 prefill rank takes every rank of a wider decode attention
    TP group (SGLANG_PREFILL_COMPLETE_ALLOW_MULTI_ENDPOINT)."""

    N = 4

    def setUp(self):
        self.table = DeferredBootstrap(timeout=10)
        self.owner = object()
        self.endpoints = [("decode", 2000 + rank) for rank in range(self.N)]

    def register_all(self, room=1, endpoints=None):
        return [
            self.table.register(room, endpoint, self.N)
            for endpoint in endpoints or self.endpoints
        ]

    def test_every_endpoint_gets_one_readiness_in_any_order(self):
        actions = ["open", "complete"] + [f"r{rank}" for rank in range(self.N)]
        for seed in range(24):
            order = list(actions)
            random.Random(seed).shuffle(order)
            if order.index("complete") < order.index("open"):
                order.remove("open")
                order.insert(0, "open")
            with self.subTest(order=order):
                table, owner, ready = DeferredBootstrap(10), object(), []
                for action in order:
                    if action == "open":
                        note = table.open(1, owner) and None
                    elif action == "complete":
                        note = table.complete(1, owner)
                    else:
                        note = table.register(
                            1, self.endpoints[int(action[1:])], self.N
                        )
                    ready.extend(note or [])
                self.assertEqual(
                    sorted(ready), sorted((e, False) for e in self.endpoints)
                )
                # Repeats send nothing more.
                self.assertIsNone(table.complete(1, owner))
                for endpoint in self.endpoints:
                    self.assertIsNone(table.register(1, endpoint, self.N))

    def test_extra_or_mismatched_endpoint_fails_alone(self):
        self.table.open(1, self.owner)
        self.register_all()
        extra = ("decode", 2999)
        self.assertEqual(self.table.register(1, extra, self.N), [(extra, True)])
        self.assertEqual(self.table.register(1, extra, 2), [(extra, True)])
        self.assertEqual(self.table.rooms[1].endpoints, self.endpoints)
        self.assertFalse(self.table.rooms[1].failed)
        self.assertEqual(
            sorted(self.table.complete(1, self.owner)),
            sorted((e, False) for e in self.endpoints),
        )
        # A one-endpoint registrant of a room expecting four fails alone too.
        table = DeferredBootstrap(10)
        table.register(2, self.endpoints[0], self.N)
        self.assertEqual(table.register(2, extra), [(extra, True)])

    def test_failure_reaches_every_endpoint_and_late_registrants(self):
        for when in ("before_complete", "after_complete", "cancel_then_close"):
            with self.subTest(when=when):
                table, owner = DeferredBootstrap(10), object()
                table.open(1, owner)
                for endpoint in self.endpoints[:2]:
                    table.register(1, endpoint, self.N)
                if when == "after_complete":
                    table.complete(1, owner)
                if when == "cancel_then_close":
                    table.cancel(1)
                    note = table.close(1, owner)
                    self.assertIsNone(note)  # cancel already recorded the failure
                else:
                    note = table.fail(1)
                    self.assertEqual(
                        sorted(note), sorted((e, True) for e in self.endpoints[:2])
                    )
                    self.assertIsNone(table.fail(1))
                for endpoint in self.endpoints[2:]:
                    self.assertEqual(
                        table.register(1, endpoint, self.N), [(endpoint, True)]
                    )

    def test_release_failure_tells_every_endpoint_once(self):
        self.table.open(1, self.owner)
        self.register_all()
        note = self.table.close(1, self.owner)
        self.assertEqual(sorted(note), sorted((e, True) for e in self.endpoints))
        self.assertIsNone(self.table.close(1, self.owner))

    def test_next_request_registrations_wait_for_release(self):
        self.table.open(1, self.owner)
        self.register_all()
        self.table.complete(1, self.owner)
        self.assertTrue(self.table.conclude(1))
        # Part of the next request's group registers before this one is released.
        for endpoint in self.endpoints[:3]:
            self.assertIsNone(self.table.register(1, endpoint, self.N))
        self.assertIsNone(self.table.close(1, self.owner, success=True))
        state = self.table.rooms[1]
        self.assertEqual(state.endpoints, self.endpoints[:3])
        self.assertEqual(state.expected, self.N)
        new_owner = object()
        self.table.open(1, new_owner)
        self.assertIsNone(self.table.register(1, self.endpoints[3], self.N))
        self.assertEqual(
            sorted(self.table.complete(1, new_owner)),
            sorted((e, False) for e in self.endpoints),
        )

    def test_partial_registration_expires_without_owner(self):
        table = DeferredBootstrap(0.01)
        table.register(1, self.endpoints[0], self.N)
        time.sleep(0.02)
        table._next_cleanup = 0.0
        table.register(2, self.endpoints[0])
        self.assertNotIn(1, table.rooms)


if __name__ == "__main__":
    unittest.main()
