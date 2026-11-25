"""Bookkeeping for bootstrap rendezvous before decode has reserved KV.

The transfer backend still owns the destination-pointer handshake and abort
draining. This table only joins a completed optimistic prefill with decode's
control endpoint. It performs no I/O, including while holding its lock.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Per-room prefill-done -> first-destination wait line on P (both arms of the
# late-binding experiment); off unless set, so P logs nothing new by default.
_LOG_P_KV_WAIT = os.environ.get("SGLANG_DISAGG_LATE_BINDING_P_LOG", "0") == "1"

BootstrapNotification = tuple[tuple[str, int], bool] | None


@dataclass
class BootstrapRoom:
    owner: object | None = None
    endpoint: tuple[str, int] | None = None
    ready: bool = False
    ready_notified: bool = False
    failed: bool = False
    expires_at: float = 0.0
    # The transfer succeeded and decode has been told; the owner has not
    # released the room yet.
    concluded: bool = False
    # Decode's registration of the room's next request, which arrived while
    # this one was concluded but not released.
    next_endpoint: tuple[str, int] | None = None
    # Late binding: decode endpoints this request was moved away from (REBIND).
    # Their late BOOTSTRAP/ABORT/metadata must not touch the room.
    stale_endpoints: set = field(default_factory=set)
    # time.time_ns() of prefill completion and of decode's first destination.
    ready_ns: int = 0
    metadata_ns: int = 0


class DeferredBootstrap:
    def __init__(self, timeout: float):
        self.timeout = timeout
        self.rooms: dict[int, BootstrapRoom] = {}
        self.lock = threading.Lock()
        self._next_cleanup = 0.0
        # Late-binding counters (rebind outcomes, stale-endpoint races) and the
        # prefill-done -> first-destination wait, all rooms and rebound rooms.
        self.counters: Counter = Counter()
        self._next_summary = time.monotonic() + 30.0

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
                # One readiness message per transfer and PP stage, sent before
                # that stage's Success; so none can trail the room's Success
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
            if state.concluded:
                # Decode heard Success, so this registers the room's next
                # request; it applies once this one is released.
                if state.next_endpoint is None:
                    state.next_endpoint = endpoint
                return None
            if endpoint in state.stale_endpoints:
                # The handed-off rank's BOOTSTRAP arrived after the REBIND.
                self.counters["late_bootstrap_from_stale"] += 1
                logger.warning(
                    "Late binding: dropped BOOTSTRAP for room %s from stale "
                    "endpoint %s (room now bound to %s)",
                    room,
                    endpoint,
                    state.endpoint,
                )
                return None
            if state.endpoint is not None and state.endpoint != endpoint:
                # A room identifies one request, not a reusable allocation slot.
                # Reject another receiver without replacing the first endpoint.
                return endpoint, True
            state.endpoint = endpoint
            return self._notification(state)

    def complete(self, room: int, owner: object) -> BootstrapNotification:
        with self.lock:
            state = self.rooms.get(room)
            if state is None or state.owner is not owner or state.failed:
                return None
            state.ready = True
            if state.ready_ns == 0:
                state.ready_ns = time.time_ns()
            return self._notification(state)

    def rebind(
        self,
        room: int,
        new_endpoint: tuple[str, int],
        old_endpoint: tuple[str, int],
    ) -> tuple[BootstrapNotification, str]:
        """Move a room's decode endpoint before decode has sent destinations.

        Returns (notification, verdict). Accepted only while the room is not
        failed, not concluded and has no destination yet; a ready room is
        re-notified to the new endpoint. Every reject notifies the new endpoint
        Failed, so the request fails loudly instead of waiting.
        """
        with self.lock:
            state = self._room(room)
            if state.endpoint == new_endpoint and old_endpoint in state.stale_endpoints:
                verdict, notification = "rebind_duplicate", None
            elif state.failed:
                verdict, notification = "rebind_reject_failed", (new_endpoint, True)
            elif state.concluded or state.metadata_ns:
                verdict, notification = "rebind_reject_bound", (new_endpoint, True)
            elif state.endpoint is not None and state.endpoint != old_endpoint:
                verdict, notification = "rebind_reject_mismatch", (new_endpoint, True)
            else:
                # endpoint None: this PP stage has not seen the old BOOTSTRAP yet.
                verdict = "rebind_ok" if state.endpoint is not None else "rebind_ok_early"
                state.stale_endpoints.add(old_endpoint)
                state.endpoint = new_endpoint
                # The old endpoint may have had readiness; the new one has not.
                state.ready_notified = False
                notification = self._notification(state)
            self.counters[verdict] += 1
        if verdict.startswith("rebind_reject"):
            logger.error(
                "Late binding: %s room=%s new=%s old=%s",
                verdict,
                room,
                new_endpoint,
                old_endpoint,
            )
        return notification, verdict

    def note_metadata(self, room: int, endpoint: tuple[str, int]) -> bool:
        """Record decode's first destination for a room (the hard binding).
        Returns False for a destination from an endpoint the room was rebound
        away from: the caller drops it before the transfer backend."""
        line = None
        with self.lock:
            state = self.rooms.get(room)
            if state is None:
                return True
            if endpoint in state.stale_endpoints:
                self.counters["metadata_from_stale"] += 1
                logger.error(
                    "Late binding: rejected destination for room %s from stale "
                    "endpoint %s",
                    room,
                    endpoint,
                )
                return False
            if state.metadata_ns:
                return True
            state.metadata_ns = time.time_ns()
            if state.ready_ns:
                wait_ms = (state.metadata_ns - state.ready_ns) / 1e6
                rebound = int(bool(state.stale_endpoints))
                self.counters["wait_n"] += 1
                self.counters["wait_us_sum"] += int(wait_ms * 1000)
                if rebound:
                    self.counters["wait_rebound_n"] += 1
                    self.counters["wait_rebound_us_sum"] += int(wait_ms * 1000)
                line = (state.ready_ns, state.metadata_ns, wait_ms, rebound)
        if line is not None and _LOG_P_KV_WAIT:
            logger.info(
                "LATEBIND_P_KV_WAIT room=%d prefill_done_ns=%d send_metadata_ns=%d "
                "wait_ms=%.3f rebound=%d",
                room,
                *line,
            )
        self.maybe_log_summary()
        return True

    def maybe_log_summary(self, force: bool = False) -> None:
        now = time.monotonic()
        if not (_LOG_P_KV_WAIT or force) or (not force and now < self._next_summary):
            return
        self._next_summary = now + 30.0
        with self.lock:
            counters = dict(self.counters)
        logger.info("LATEBIND_P_SUMMARY %s", counters)

    def conclude(self, room: int) -> bool:
        """Commit the room's transfer success before decode is told. False when
        a failure or cancellation won, which then owns decode's notification."""
        with self.lock:
            state = self.rooms.get(room)
            if state is None or state.owner is None or state.failed:
                return False
            state.concluded = True
            # Stale marks belong to this request's registration; the room's
            # next request may legitimately register from the old endpoint.
            state.stale_endpoints.clear()
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

    def cancel(self, room: int, endpoint: tuple[str, int] | None = None) -> bool:
        """Record decode's ABORT; on a concluded room it cancels the room's
        next request, the only one decode can still abort.

        Returns False, without cancelling, for an ABORT from an endpoint the
        room was rebound away from: that rank no longer owns the request.
        """
        with self.lock:
            state = self._room(room)
            if endpoint is not None and endpoint in state.stale_endpoints:
                self.counters["abort_from_stale"] += 1
                logger.error(
                    "Late binding: ignored ABORT for room %s from stale endpoint %s",
                    room,
                    endpoint,
                )
                return False
            state.failed = True
            return True

    def close(
        self, room: int, owner: object, *, success: bool = False
    ) -> BootstrapNotification:
        with self.lock:
            state = self.rooms.get(room)
            if state is None or state.owner is not owner:
                return None
            if not state.failed and (success or state.concluded):
                # The transfer worker already told decode. A registration of the
                # room's next request that arrived since starts that room.
                if state.next_endpoint is None:
                    del self.rooms[room]
                else:
                    self.rooms[room] = BootstrapRoom(
                        endpoint=state.next_endpoint,
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
