"""Bookkeeping for bootstrap rendezvous before decode has reserved KV.

The transfer backend still owns the destination-pointer handshake and abort
draining. This table only joins a completed optimistic prefill with decode's
control endpoint. It performs no I/O, including while holding its lock.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

Endpoint = tuple[str, int]
# One (endpoint, failed) message; a list of them for a room that expects more
# than one decode endpoint (decode attention TP over prefill attention TP).
BootstrapNotification = (
    tuple[Endpoint, bool] | list[tuple[Endpoint, bool]] | None
)


@dataclass
class BootstrapRoom:
    owner: object | None = None
    endpoints: list[Endpoint] = field(default_factory=list)
    # Decode endpoints the room's request registers.
    expected: int = 1
    ready: bool = False
    # Endpoints that were sent this room's readiness.
    notified: set[Endpoint] = field(default_factory=set)
    failed: bool = False
    expires_at: float = 0.0
    # The transfer succeeded and decode has been told; the owner has not
    # released the room yet.
    concluded: bool = False
    # Decode's registrations of the room's next request, which arrived while
    # this one was concluded but not released.
    next_endpoints: list[Endpoint] = field(default_factory=list)
    next_expected: int = 1

    @property
    def endpoint(self) -> Endpoint | None:
        return self.endpoints[0] if self.endpoints else None


class DeferredBootstrap:
    def __init__(self, timeout: float):
        self.timeout = timeout
        self.rooms: dict[int, BootstrapRoom] = {}
        self.lock = threading.Lock()
        self._next_cleanup = 0.0

    def _room(self, room: int) -> BootstrapRoom:
        now = time.monotonic()
        if now >= self._next_cleanup:
            # Active senders own their lifetime. Unmatched endpoints and closed
            # rooms expire, including cancellation before the sender exists.
            self.rooms = {
                key: state
                for key, state in self.rooms.items()
                if state.owner is not None or state.expires_at > now
            }
            self._next_cleanup = now + min(1.0, self.timeout)
        return self.rooms.setdefault(room, BootstrapRoom(expires_at=now + self.timeout))

    @staticmethod
    def _notification(
        state: BootstrapRoom, endpoints: list[Endpoint] | None = None
    ) -> BootstrapNotification:
        notes = []
        for endpoint in state.endpoints if endpoints is None else endpoints:
            if state.failed:
                notes.append((endpoint, True))
            elif state.ready and endpoint not in state.notified:
                # One readiness message per endpoint, transfer and PP stage, sent
                # before that stage's Success; so none can trail the room's
                # Success and accidentally admit the next transfer using this room.
                state.notified.add(endpoint)
                notes.append((endpoint, False))
        return DeferredBootstrap._out(state.expected, notes)

    @staticmethod
    def _out(expected: int, notes: list) -> BootstrapNotification:
        if not notes:
            return None
        return notes if expected > 1 else notes[0]

    def open(self, room: int, owner: object) -> BootstrapRoom | None:
        with self.lock:
            state = self._room(room)
            if state.owner is not None:
                return None
            state.owner = owner
            return state

    def register(
        self, room: int, endpoint: Endpoint, expected: int = 1
    ) -> BootstrapNotification:
        """``expected``: the decode endpoints of the room's request (every rank of
        one decode attention TP group registers when prefill's is smaller)."""
        with self.lock:
            state = self._room(room)
            if state.concluded:
                # Decode heard Success, so this registers the room's next
                # request; it applies once this one is released.
                if not state.next_endpoints:
                    state.next_expected = expected
                if (
                    endpoint not in state.next_endpoints
                    and len(state.next_endpoints) < state.next_expected
                ):
                    state.next_endpoints.append(endpoint)
                return None
            if endpoint in state.endpoints:
                return self._notification(state, [endpoint])
            if state.endpoints and (
                expected != state.expected or len(state.endpoints) >= state.expected
            ):
                # A room identifies one request, not a reusable allocation slot.
                # Reject another receiver without replacing the registered ones.
                return self._out(max(expected, state.expected), [(endpoint, True)])
            if not state.endpoints:
                state.expected = expected
            state.endpoints.append(endpoint)
            return self._notification(state, [endpoint])

    def complete(self, room: int, owner: object) -> BootstrapNotification:
        with self.lock:
            state = self.rooms.get(room)
            if state is None or state.owner is not owner or state.failed:
                return None
            state.ready = True
            return self._notification(state)

    def conclude(self, room: int) -> bool:
        """Commit the room's transfer success before decode is told. False when
        a failure or cancellation won, which then owns decode's notification."""
        with self.lock:
            state = self.rooms.get(room)
            if state is None or state.owner is None or state.failed:
                return False
            state.concluded = True
            return True

    def awaits_release(self, room: int) -> bool:
        """Whether the room's previous request succeeded but is not released;
        its next request must not open the room until then."""
        with self.lock:
            state = self.rooms.get(room)
            return state is not None and state.owner is not None and state.concluded

    def fail(self, room: int) -> BootstrapNotification:
        with self.lock:
            state = self._room(room)
            if state.failed or state.concluded:
                # Decode already holds a concluded transfer's Success.
                return None
            state.failed = True
            return self._notification(state)

    def cancel(self, room: int) -> None:
        """Record decode's ABORT; on a concluded room it cancels the room's
        next request, the only one decode can still abort."""
        with self.lock:
            self._room(room).failed = True

    def close(
        self, room: int, owner: object, *, success: bool = False
    ) -> BootstrapNotification:
        with self.lock:
            state = self.rooms.get(room)
            if state is None or state.owner is not owner:
                return None
            if not state.failed and (success or state.concluded):
                # The transfer worker already told decode. Registrations of the
                # room's next request that arrived since start that room.
                if not state.next_endpoints:
                    del self.rooms[room]
                else:
                    self.rooms[room] = BootstrapRoom(
                        endpoints=list(state.next_endpoints),
                        expected=state.next_expected,
                        expires_at=time.monotonic() + self.timeout,
                    )
                return None
            state.owner = None
            state.concluded = False
            already_failed = state.failed
            state.failed = True
            state.expires_at = time.monotonic() + self.timeout
            # Retain a bounded tombstone: a late endpoint must not wait forever
            # or attach to a second request using the same bootstrap room.
            return None if already_failed else self._notification(state)
