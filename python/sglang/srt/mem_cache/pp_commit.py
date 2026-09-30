"""CPU state machine for the opt-in PP cache logical commit boundary.

This module does no communication, allocation, or queue draining. Physical ACKs
become local prepared effects; only a leader frame can apply them. Missing
operations stop the contiguous frontier, never manufacture a NOOP from a count.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class OperationId:
    """Matching identity, independent of local node IDs and physical addresses.

    ``key`` includes kind-specific content/request, pool and logical range.
    Generation distinguishes repeated operations on the same immutable key.
    The leader subsequently assigns the common (epoch, seq) log position.
    """

    epoch: int
    kind: str
    key: str
    generation: int

    def wire(self):
        return [self.epoch, self.kind, self.key, self.generation]


@dataclass
class PreparedEffect:
    identity: OperationId
    payload_digest: str
    apply: Callable[[], None]
    pinned_bytes: int = 0
    created_at: float = 0.0


class CommitBoundary:
    def __init__(
        self,
        rank: int,
        size: int,
        *,
        epoch: int = 0,
        max_pending: int = 8192,
        max_pinned_bytes: int = 8 << 30,
        stall_seconds: float = 120.0,
        clock=time.monotonic,
    ):
        self.rank, self.size, self.epoch = rank, size, epoch
        self.max_pending, self.max_pinned_bytes = max_pending, max_pinned_bytes
        self.stall_seconds, self.clock = stall_seconds, clock
        self.round = self.confirmed = self.committed = self.last_seq = 0
        self.prepared: OrderedDict[OperationId, PreparedEffect] = OrderedDict()
        self.manifest: dict[int, tuple[OperationId, str]] = {}
        self.assigned: dict[OperationId, int] = {}
        self.completed: dict[tuple[str, str], tuple[int, set[int]]] = {}
        self.hashes = {0: digest([epoch])}
        self.pinned_bytes = 0
        self.pending_since = None
        self.no_progress_rounds = 0
        self.warned = False
        self.stats = {
            "prepared": 0,
            "applied": 0,
            "duplicates": 0,
            "commit_stall": 0,
            "peak_pending": 0,
            "peak_pinned_bytes": 0,
            "max_wait_s": 0.0,
        }

    @property
    def admission_open(self):
        return (
            len(self.prepared) < self.max_pending
            and self.pinned_bytes < self.max_pinned_bytes
        )

    def stage(self, identity: OperationId, payload, apply, pinned_bytes=0):
        if identity.epoch != self.epoch:
            raise RuntimeError("PP commit stale epoch ACK")
        payload_digest = digest(payload)
        old = self.prepared.get(identity)
        if old is not None:
            if old.payload_digest != payload_digest:
                raise RuntimeError("PP commit conflicting duplicate ACK")
            self.stats["duplicates"] += 1
            return
        watermark, later = self.completed.get(
            (identity.kind, identity.key), (-1, set())
        )
        if identity.generation <= watermark or identity.generation in later:
            self.stats["duplicates"] += 1
            return
        # Callers gate NEW cache admission before this bound. Already-owned ACK
        # resources must remain pinned; fail explicitly rather than free early.
        if (
            not self.admission_open
            or self.pinned_bytes + pinned_bytes > self.max_pinned_bytes
        ):
            raise RuntimeError("PP commit prepared resource bound exceeded")
        self.prepared[identity] = PreparedEffect(
            identity, payload_digest, apply, pinned_bytes, self.clock()
        )
        self.pinned_bytes += pinned_bytes
        self.stats["prepared"] += 1
        self.stats["peak_pending"] = max(self.stats["peak_pending"], len(self.prepared))
        self.stats["peak_pinned_bytes"] = max(
            self.stats["peak_pinned_bytes"], self.pinned_bytes
        )
        if self.pending_since is None:
            self.pending_since = self.clock()
        self.refresh_confirmed()

    def refresh_confirmed(self):
        while self.confirmed + 1 in self.manifest:
            identity, expected = self.manifest[self.confirmed + 1]
            effect = self.prepared.get(identity)
            if effect is None:
                break
            if effect.payload_digest != expected:
                raise RuntimeError(f"PP commit payload mismatch at {identity}")
            self.confirmed += 1
        return self.confirmed

    def ready(self):
        return {
            "epoch": self.epoch,
            "round": self.round,
            "confirmed": self.confirmed,
            "digest": self.hashes[self.confirmed],
            "applied": self.committed,
            "applied_digest": self.hashes[self.committed],
        }

    def leader_frame(
        self,
        reports: dict[int, dict],
        *,
        limit=128,
        local_confirmed=None,
        proposals=(),
        kind_limits=None,
    ):
        if self.rank != 0:
            raise RuntimeError("Only PP0 assigns commit sequence numbers")
        next_round = self.round + 1
        frontier = self.confirmed if local_confirmed is None else local_confirmed
        for rank in range(1, self.size):
            report = reports.get(rank)
            if report is None:
                frontier = min(frontier, self.committed)
                continue
            if report["epoch"] != self.epoch or report["round"] >= next_round:
                raise RuntimeError("PP commit READY epoch/round mismatch")
            seq = report["confirmed"]
            if seq < self.committed:
                frontier = min(frontier, self.committed)
                continue  # a coalesced, delayed snapshot cannot undo a commit
            if seq > self.last_seq or report["digest"] != self.hashes.get(seq):
                raise RuntimeError("PP commit READY manifest digest mismatch")
            frontier = min(frontier, seq)
        entries = []
        candidates = {
            identity: effect.payload_digest
            for identity, effect in self.prepared.items()
        }
        for wire, payload_hash in proposals:
            identity = OperationId(*wire)
            if identity.epoch != self.epoch or identity.kind not in (
                "backup",
                "release",
            ):
                raise RuntimeError("PP commit invalid physical proposal")
            old = candidates.get(identity)
            if old is not None and old != payload_hash:
                raise RuntimeError("PP commit conflicting physical proposal")
            watermark, later = self.completed.get(
                (identity.kind, identity.key), (-1, set())
            )
            if identity.generation > watermark and identity.generation not in later:
                candidates[identity] = payload_hash
        used_kinds = {}
        for identity, payload_hash in candidates.items():
            if identity not in self.assigned:
                category = (
                    "belief" if identity.kind.startswith("belief_") else "physical"
                )
                if kind_limits and used_kinds.get(category, 0) >= kind_limits[category]:
                    continue
                used_kinds[category] = used_kinds.get(category, 0) + 1
                entries.append(
                    [
                        self.last_seq + len(entries) + 1,
                        identity.wire(),
                        payload_hash,
                    ]
                )
                if len(entries) == limit:
                    break
        return {
            "epoch": self.epoch,
            "round": next_round,
            "commit": frontier,
            "entries": entries,
        }

    def accept_frame(self, frame):
        if frame["epoch"] != self.epoch or frame["round"] != self.round + 1:
            raise RuntimeError("PP commit frame sequence/epoch mismatch")
        self.round = frame["round"]
        for seq, wire, payload_digest in frame["entries"]:
            identity = OperationId(*wire)
            if (
                identity.epoch != self.epoch
                or seq != self.last_seq + 1
                or identity in self.assigned
            ):
                raise RuntimeError("PP commit noncontiguous or duplicate manifest")
            self.manifest[seq] = (identity, payload_digest)
            self.assigned[identity] = seq
            self.hashes[seq] = digest(
                [self.hashes[self.last_seq], seq, wire, payload_digest]
            )
            self.last_seq = seq
        self.refresh_confirmed()
        frontier = frame["commit"]
        if not self.committed <= frontier <= self.confirmed:
            raise RuntimeError("PP commit attempts unconfirmed logical effects")
        previous = self.committed
        for seq in range(previous + 1, frontier + 1):
            identity, _ = self.manifest[seq]
            effect = self.prepared[identity]
            effect.apply()  # exceptions are fatal: never acknowledge a partial apply
            self.pinned_bytes -= effect.pinned_bytes
            key = (identity.kind, identity.key)
            watermark, later = self.completed.get(key, (-1, set()))
            later.add(identity.generation)
            while watermark + 1 in later:
                watermark += 1
                later.remove(watermark)
            self.completed[key] = (watermark, later)
            del self.prepared[identity], self.assigned[identity], self.manifest[seq]
            self.committed = seq
            self.stats["applied"] += 1
        # Keep the committed hash and all uncommitted hashes, not an unbounded
        # manifest history. Stale READY snapshots are ignored above.
        for seq in range(previous, self.committed):
            self.hashes.pop(seq, None)
        pending = bool(self.prepared or self.manifest)
        if self.prepared:
            oldest = min(effect.created_at for effect in self.prepared.values())
            if self.clock() - oldest >= self.stall_seconds:
                self.stats["commit_stall"] += 1
                raise RuntimeError(
                    f"PP commit frontier stalled: aged local ACK {self.snapshot()}"
                )
        if self.committed != previous or not pending:
            if self.pending_since is not None:
                self.stats["max_wait_s"] = max(
                    self.stats["max_wait_s"], self.clock() - self.pending_since
                )
            self.pending_since = self.clock() if pending else None
            self.no_progress_rounds = 0
            self.warned = False
        elif pending:
            if self.pending_since is None:
                self.pending_since = self.clock()
            self.no_progress_rounds += 1
            if self.no_progress_rounds >= 8 and not self.warned:
                logger.warning(
                    "PP commit waiting rank=%d snapshot=%s", self.rank, self.snapshot()
                )
                self.warned = True
            if self.clock() - self.pending_since >= self.stall_seconds:
                self.stats["commit_stall"] += 1
                raise RuntimeError(f"PP commit frontier stalled: {self.snapshot()}")

    def snapshot(self):
        return dict(
            rank=self.rank,
            epoch=self.epoch,
            round=self.round,
            confirmed=self.confirmed,
            committed=self.committed,
            pending=len(self.prepared),
            pinned_bytes=self.pinned_bytes,
            missing=[
                (seq, op.wire())
                for seq, (op, _) in list(self.manifest.items())[:4]
                if op not in self.prepared
            ],
            unassigned=[
                op.wire() for op in list(self.prepared)[:4] if op not in self.assigned
            ],
            no_progress_rounds=self.no_progress_rounds,
            **self.stats,
        )

    def reset(self, epoch):
        if self.prepared or self.manifest:
            raise RuntimeError(
                "Cannot reset PP commit with pinned uncommitted operations"
            )
        if epoch <= self.epoch:
            raise RuntimeError("PP commit epoch must increase")
        self.__init__(
            self.rank,
            self.size,
            epoch=epoch,
            max_pending=self.max_pending,
            max_pinned_bytes=self.max_pinned_bytes,
            stall_seconds=self.stall_seconds,
            clock=self.clock,
        )
