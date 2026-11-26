"""Lifecycle tests for the shared pre-allocation bootstrap rendezvous."""

import itertools
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
                self.assertEqual(
                    self.table.close(1, self.owner, success=True),
                    (self.endpoint, False),
                )
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

    def test_success_cannot_reset_cancelled_room(self):
        self.table.open(1, self.owner)
        self.table.fail(1)
        self.table.close(1, self.owner, success=True)
        self.assertEqual(self.table.register(1, self.endpoint), (self.endpoint, True))

    def test_rebind_renotifies_ready_room_at_new_endpoint(self):
        new = ("decode", 5678)
        self.table.open(1, self.owner)
        self.table.register(1, self.endpoint)
        self.assertEqual(self.table.complete(1, self.owner), (self.endpoint, False))
        self.assertEqual(self.table.rebind(1, new, self.endpoint), (new, False))
        self.assertEqual(self.table.rooms[1].endpoint, new)
        # A repeated REBIND neither notifies again nor changes the room.
        self.assertIsNone(self.table.rebind(1, new, self.endpoint))
        self.assertFalse(self.table.rooms[1].failed)

    def test_rebind_before_ready_waits_for_completion(self):
        new = ("decode", 5678)
        for registered in (True, False):
            with self.subTest(registered=registered):
                table = DeferredBootstrap(10)
                table.open(1, self.owner)
                if registered:
                    table.register(1, self.endpoint)
                self.assertIsNone(table.rebind(1, new, self.endpoint))
                self.assertEqual(table.complete(1, self.owner), (new, False))

    def test_rebind_of_failed_or_foreign_room_fails_new_endpoint(self):
        new, other = ("decode", 5678), ("decode", 9999)
        failed = DeferredBootstrap(10)
        failed.open(1, self.owner)
        failed.register(1, self.endpoint)
        failed.fail(1)
        self.assertEqual(failed.rebind(1, new, self.endpoint), (new, True))
        self.table.open(1, self.owner)
        self.table.register(1, other)
        self.assertEqual(self.table.rebind(1, new, self.endpoint), (new, True))
        self.assertEqual(self.table.rooms[1].endpoint, other)
        self.assertFalse(self.table.rooms[1].failed)

    def test_cancel_from_rebound_endpoint_is_ignored(self):
        new = ("decode", 5678)
        self.table.open(1, self.owner)
        self.table.register(1, self.endpoint)
        self.table.complete(1, self.owner)
        self.table.rebind(1, new, self.endpoint)
        self.assertFalse(self.table.cancel(1, self.endpoint))
        self.assertFalse(self.table.rooms[1].failed)
        self.assertTrue(self.table.cancel(1, new))
        self.assertTrue(self.table.rooms[1].failed)
        self.assertIsNone(self.table.complete(1, self.owner))

    def test_cancel_without_rebind_fails_room_like_fail(self):
        for opened in (True, False):
            with self.subTest(opened=opened):
                table = DeferredBootstrap(10)
                if opened:
                    table.open(1, self.owner)
                    table.register(1, self.endpoint)
                # Any endpoint cancels a room nobody was rebound away from.
                self.assertTrue(table.cancel(1, ("other", 1)))
                self.assertTrue(table.rooms[1].failed)
                self.assertEqual(
                    table.register(1, self.endpoint), (self.endpoint, True)
                )

    def test_no_lock_escapes_any_operation(self):
        for operation in (
            lambda: self.table.open(1, self.owner),
            lambda: self.table.register(1, self.endpoint),
            lambda: self.table.complete(1, self.owner),
            lambda: self.table.rebind(1, ("decode", 5678), self.endpoint),
            lambda: self.table.cancel(1, self.endpoint),
            lambda: self.table.fail(1),
            lambda: self.table.close(1, self.owner),
        ):
            operation()
            self.assertTrue(self.table.lock.acquire(blocking=False))
            self.table.lock.release()


if __name__ == "__main__":
    unittest.main()
