"""CPU control contracts for opt-in, four-rank prefill CP borrowing.

This module has no model, torch, or process-group dependency. A plan describes
one native prefill request; it never changes the original scheduler batch.
"""

from dataclasses import dataclass, field
from enum import Enum, IntEnum, IntFlag
from typing import Sequence, TypeVar

T = TypeVar("T")
BASE_WIDTH = 13
BORROW_WIDTH = 21
WORLD_SIZE = 4


class ProtocolError(RuntimeError):
    """Rank disagreement requires stopping, not unilateral DP fallback."""


class Phase(IntEnum):
    PROPOSE = 0
    RESERVE = 1
    READY = 2


class Flag(IntFlag):
    CANDIDATE = 1
    IDLE_ACCEPT = 2
    RESERVED = 4
    READY = 8


class State(Enum):
    PROPOSED = "proposed"
    RESERVED = "reserved"
    COMMITTED = "committed"
    REJECTED = "rejected"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass(frozen=True)
class Vote:
    epoch: int
    phase: Phase = Phase.PROPOSE
    flags: Flag = Flag(0)
    ticket: int = 0
    prefix: int = 0
    chunk: int = 0
    reject: int = 0

    def suffix(self) -> tuple[int, ...]:
        return (1, self.epoch, int(self.phase), int(self.flags), self.ticket,
                self.prefix, self.chunk, self.reject)


@dataclass(frozen=True)
class Interleave:
    n: int

    def __post_init__(self):
        if self.n < WORLD_SIZE:
            raise ValueError("Borrowing requires at least four new tokens")

    @property
    def counts(self) -> tuple[int, ...]:
        return tuple((self.n + WORLD_SIZE - 1 - r) // WORLD_SIZE for r in range(WORLD_SIZE))

    def indices(self, rank: int) -> range:
        if not 0 <= rank < WORLD_SIZE:
            raise ValueError("Invalid borrow rank")
        return range(rank, self.n, WORLD_SIZE)

    def split(self, values: Sequence[T]) -> tuple[tuple[T, ...], ...]:
        if len(values) != self.n:
            raise ValueError("Input length does not match the committed chunk")
        return tuple(tuple(values[j] for j in self.indices(r)) for r in range(WORLD_SIZE))

    def restore(self, shards: Sequence[Sequence[T]]) -> tuple[T, ...]:
        if len(shards) != WORLD_SIZE or tuple(map(len, shards)) != self.counts:
            raise ValueError("Expected exact live shards without communication padding")
        return tuple(shards[j % WORLD_SIZE][j // WORLD_SIZE] for j in range(self.n))


@dataclass(frozen=True)
class Plan:
    epoch: int
    owner: int
    ticket: int
    prefix: int
    n: int
    chunk: int
    logical_rows: tuple[tuple[int, ...], ...] = field(repr=False)

    @property
    def key(self) -> tuple[int, ...]:
        return self.epoch, self.owner, self.ticket, self.chunk

    @property
    def logical_counts(self) -> tuple[int, ...]:
        return tuple(self.n if r == self.owner else 0 for r in range(WORLD_SIZE))

    @property
    def execution_counts(self) -> tuple[int, ...]:
        return Interleave(self.n).counts

    @property
    def final_rank(self) -> int:
        return (self.n - 1) % WORLD_SIZE

    def positions(self, rank: int) -> tuple[int, ...]:
        return tuple(self.prefix + j for j in Interleave(self.n).indices(rank))

    def attention_pairs(self, rank: int) -> int:
        m = len(Interleave(self.n).indices(rank))
        return m * (self.prefix + rank + 1) + WORLD_SIZE * m * (m - 1) // 2


def _check_rows(rows: Sequence[Sequence[int]], phase: Phase, epoch=None) -> int:
    if len(rows) != WORLD_SIZE or any(len(r) != BORROW_WIDTH for r in rows):
        raise ProtocolError("Borrow protocol requires four 21-column rows")
    if epoch is None:
        epoch = rows[0][14]
    for row in rows:
        if row[13] != 1 or row[14] != epoch or row[15] != phase:
            raise ProtocolError("Borrow protocol, epoch, or phase mismatch")
        if any(not isinstance(v, int) for v in row):
            raise ProtocolError("Borrow wire fields must be integers")
        if epoch < 0 or min(row[0], row[17], row[18], row[19], row[20]) < 0:
            raise ProtocolError("Negative borrow metadata")
        if row[16] < 0 or row[16] & ~int(Flag.CANDIDATE | Flag.IDLE_ACCEPT | Flag.RESERVED | Flag.READY):
            raise ProtocolError("Unknown borrow flags")
    return epoch


def choose_plan(
    rows: Sequence[Sequence[int]], *, min_new_tokens: int,
    extend_mode: int, idle_mode: int,
) -> Plan | None:
    """Every rank derives the same proposal from the existing sync result."""
    if min_new_tokens < WORLD_SIZE:
        raise ValueError("Borrow threshold must be at least four tokens")
    epoch = _check_rows(rows, Phase.PROPOSE)
    if any(r[16] & int(Flag.RESERVED | Flag.READY) for r in rows):
        raise ProtocolError("Proposal contains later-phase flags")
    candidates = [r for r in range(WORLD_SIZE) if rows[r][16] & Flag.CANDIDATE]
    if len(candidates) != 1:
        return None
    owner = candidates[0]
    source = rows[owner]
    if source[17] == 0:
        raise ProtocolError("A candidate needs an immutable request ticket")
    if source[16] != Flag.CANDIDATE or source[5] != extend_mode or not source[3]:
        return None
    if source[0] < min_new_tokens:
        return None
    for rank, row in enumerate(rows):
        if row[9] or row[11] or row[12] or row[20]:
            return None
        if rank != owner and (
            row[16] != Flag.IDLE_ACCEPT or row[0] != 0 or row[5] != idle_mode
        ):
            return None
    return Plan(epoch, owner, source[17], source[18], source[0], source[19],
                tuple(tuple(r[:BASE_WIDTH]) for r in rows))


def _phase_accepted(plan: Plan, rows: Sequence[Sequence[int]], phase: Phase) -> bool:
    _check_rows(rows, phase, plan.epoch)
    accepted = True
    for rank, row in enumerate(rows):
        if tuple(row[:BASE_WIDTH]) != plan.logical_rows[rank]:
            raise ProtocolError("Original scheduler batch changed during borrowing")
        if (row[17], row[18], row[19]) != (plan.ticket, plan.prefix, plan.chunk):
            raise ProtocolError("Borrow request identity or prefix changed")
        role = Flag.CANDIDATE if rank == plan.owner else Flag.IDLE_ACCEPT
        if row[16] & int(Flag.CANDIDATE | Flag.IDLE_ACCEPT) != role:
            raise ProtocolError("Borrow owner or helper role changed")
        required = Flag.RESERVED
        if phase == Phase.READY:
            required |= Flag.READY
        elif row[16] & Flag.READY:
            raise ProtocolError("Reservation contains a premature ready vote")
        accepted &= not row[20] and row[16] & required == required
    return bool(accepted)


@dataclass
class Transaction:
    plan: Plan
    state: State = field(default=State.PROPOSED, init=False)

    @property
    def can_fallback(self) -> bool:
        return self.state in (State.PROPOSED, State.RESERVED, State.REJECTED)

    def _advance(self, rows, expected: State, phase: Phase, target: State) -> bool:
        if self.state != expected:
            raise ProtocolError("Borrow phase replay or invalid transition")
        try:
            accepted = _phase_accepted(self.plan, rows, phase)
        except ProtocolError:
            self.state = State.FAILED
            raise
        self.state = target if accepted else State.REJECTED
        return accepted

    def reserve(self, rows: Sequence[Sequence[int]]) -> bool:
        return self._advance(rows, State.PROPOSED, Phase.RESERVE, State.RESERVED)

    def commit(self, rows: Sequence[Sequence[int]]) -> bool:
        return self._advance(rows, State.RESERVED, Phase.READY, State.COMMITTED)

    def finish(self):
        if self.state != State.COMMITTED:
            raise ProtocolError("Cannot finish an uncommitted borrow")
        self.state = State.FINISHED

    def fail(self):
        if self.state == State.FINISHED:
            raise ProtocolError("Cannot fail a completed borrow")
        self.state = State.FAILED


@dataclass(frozen=True)
class PrivateKVMap:
    """Local token locations, never remote physical page identifiers."""

    req_slot: int
    prefix_locs: tuple[int, ...]
    suffix_locs: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "prefix_locs", tuple(self.prefix_locs))
        object.__setattr__(self, "suffix_locs", tuple(self.suffix_locs))
        locations = self.prefix_locs + self.suffix_locs
        if self.req_slot < 0 or any(p < 0 for p in locations):
            raise ValueError("Invalid local request or token location")
        if len(set(locations)) != len(locations):
            raise ValueError("Private prefix and suffix locations must not alias")

    def prefix_copy_pairs(self, owner: "PrivateKVMap") -> tuple[tuple[int, int], ...]:
        if len(owner.prefix_locs) != len(self.prefix_locs):
            raise ValueError("Prefix lengths disagree")
        return tuple(zip(owner.prefix_locs, self.prefix_locs))


@dataclass(frozen=True)
class PrefixCopyOp:
    kind: str
    offset: int
    size: int
    slot: int
    helper: int | None
    event: str
    wait_for: tuple[str, ...]


def prefix_copy_schedule(
    nbytes: int, slab_bytes: int, helpers: tuple[int, ...]
) -> tuple[PrefixCopyOp, ...]:
    """Double-buffer dependencies, executable on copy/scatter CUDA streams.

    Each helper has two destination slabs; the owner has two source slabs.
    Reusing a source waits for every reader. Reusing a destination waits for
    that helper's scatter, not the other slot's scatter. No kernel is implied
    here: an executor uses the existing pack/copy/index_copy operations.
    """
    if nbytes < 0 or slab_bytes <= 0 or not helpers or len(set(helpers)) != len(helpers):
        raise ValueError("Invalid prefix size, slab capacity, or helper set")
    ops = []
    previous_copies = [(), ()]
    previous_scatters = {h: [None, None] for h in helpers}
    for chunk, offset in enumerate(range(0, nbytes, slab_bytes)):
        slot = chunk % 2
        size = min(slab_bytes, nbytes - offset)
        gather_event = f"gather-{chunk}"
        ops.append(PrefixCopyOp("gather", offset, size, slot, None,
                                gather_event, previous_copies[slot]))
        copies = []
        for helper in helpers:
            copy_event = f"copy-{chunk}-{helper}"
            scatter_event = f"scatter-{chunk}-{helper}"
            old_scatter = previous_scatters[helper][slot]
            deps = (gather_event,) + ((old_scatter,) if old_scatter else ())
            ops.append(PrefixCopyOp("copy", offset, size, slot, helper,
                                    copy_event, deps))
            ops.append(PrefixCopyOp("scatter", offset, size, slot, helper,
                                    scatter_event, (copy_event,)))
            copies.append(copy_event)
            previous_scatters[helper][slot] = scatter_event
        previous_copies[slot] = tuple(copies)
    return tuple(ops)
