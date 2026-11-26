"""Bookkeeping for bootstrap rendezvous before decode has reserved KV.

The transfer backend still owns the destination-pointer handshake and abort
draining. This table only joins a completed optimistic prefill with decode's
control endpoint. It sends nothing, and performs no I/O while holding its lock.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

BootstrapNotification = tuple[tuple[str, int], bool] | None


@dataclass
class BootstrapRoom:
    owner: object | None = None
    endpoint: tuple[str, int] | None = None
    ready: bool = False
    ready_notified: bool = False
    failed: bool = False
    expires_at: float = 0.0
    # Decode endpoints the request was moved away from (REBIND); their
    # BOOTSTRAP, ABORT and destinations no longer touch the room.
    stale_endpoints: set[tuple[str, int]] = field(default_factory=set)
    # Decode sent a destination for the room: it can no longer be rebound.
    has_destination: bool = False


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
    def _notification(state: BootstrapRoom) -> BootstrapNotification:
        if state.endpoint is not None:
            if state.failed:
                return state.endpoint, True
            if state.ready and not state.ready_notified:
                # One readiness message per transfer. Decode must consume it
                # before publishing destinations, so none can trail Success
                # and accidentally admit the next transfer using this room.
                state.ready_notified = True
                return state.endpoint, False
        return None

    def open(self, room: int, owner: object) -> BootstrapRoom | None:
        with self.lock:
            state = self._room(room)
            if state.owner is not None:
                return None
            state.owner = owner
            return state

    def register(self, room: int, endpoint: tuple[str, int]) -> BootstrapNotification:
        with self.lock:
            state = self._room(room)
            current = state.endpoint
            if endpoint not in state.stale_endpoints:
                if current is not None and current != endpoint:
                    # A room identifies one request, not a reusable allocation
                    # slot. Reject another receiver without replacing the first.
                    return endpoint, True
                state.endpoint = endpoint
                return self._notification(state)
        # The old rank's BOOTSTRAP reached this prefill stage after the REBIND
        # that moved its request away (a PP stage can see them in that order).
        logger.warning(
            f"Dropped BOOTSTRAP of room {room} from {endpoint}, which it was "
            f"rebound away from (now bound to {current})"
        )
        return None

    def complete(self, room: int, owner: object) -> BootstrapNotification:
        with self.lock:
            state = self.rooms.get(room)
            if state is None or state.owner is not owner or state.failed:
                return None
            state.ready = True
            return self._notification(state)

    def rebind(
        self,
        room: int,
        endpoint: tuple[str, int],
        old_endpoint: tuple[str, int],
    ) -> BootstrapNotification:
        """Move the room to the decode endpoint its request was handed to before
        decode sent any destination. A ready room is notified again at the new
        endpoint; a failed room, one with a destination, or one bound to a third
        endpoint notifies the new endpoint Failed."""
        with self.lock:
            state = self._room(room)
            if state.endpoint == endpoint and old_endpoint in state.stale_endpoints:
                return None
            if (
                state.failed
                or state.has_destination
                or state.endpoint not in (None, old_endpoint)
            ):
                return endpoint, True
            # The old endpoint may already have been notified; the new one not.
            state.stale_endpoints.add(old_endpoint)
            state.endpoint = endpoint
            state.ready_notified = False
            return self._notification(state)

    def note_metadata(self, room: int, endpoint: tuple[str, int]) -> bool:
        """Record decode's destination for the room. False for one from an
        endpoint the room was rebound away from, which the caller drops."""
        with self.lock:
            state = self.rooms.get(room)
            if state is None:
                return True
            if endpoint in state.stale_endpoints:
                return False
            state.has_destination = True
            return True

    def cancel(self, room: int, endpoint: tuple[str, int]) -> bool:
        """Record decode's ABORT. Returns False, changing nothing, for an ABORT
        from an endpoint the room was rebound away from."""
        with self.lock:
            state = self._room(room)
            if endpoint in state.stale_endpoints:
                return False
            state.failed = True
            return True

    def fail(self, room: int) -> BootstrapNotification:
        with self.lock:
            state = self._room(room)
            if state.failed:
                return None
            state.failed = True
            return self._notification(state)

    def close(
        self, room: int, owner: object, *, success: bool = False
    ) -> BootstrapNotification:
        with self.lock:
            state = self.rooms.get(room)
            if state is None or state.owner is not owner:
                return None
            if success and not state.failed:
                # The caller clears transport state before sending Success.
                # Only then can decode reuse this room for recomputation.
                del self.rooms[room]
                return (state.endpoint, False) if state.endpoint is not None else None
            state.owner = None
            already_failed = state.failed
            state.failed = True
            state.expires_at = time.monotonic() + self.timeout
            # Retain a bounded tombstone: a late endpoint must not wait forever
            # or attach to a second request using the same bootstrap room.
            return None if already_failed else self._notification(state)
