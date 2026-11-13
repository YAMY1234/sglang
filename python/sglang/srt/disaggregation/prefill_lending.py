"""Prefill lending between the attention-DP ranks of one PD prefill worker.

With DP attention and EP MoE, every forward of a prefill worker runs in lockstep across
its DP ranks, so the forward costs what its busiest rank carries. Requests are pinned to
a rank by the router (session affinity), so a rank can hold queued work for several
forwards while a peer runs short. A rank with work beyond its next batch (the owner)
lends whole waiting requests to an under-filled peer (the borrower):

1. The owner locks the request's cached prefix, allocates pages for the request's full
   uncached pages, and sends a LendRequest naming both page lists.
2. The borrower allocates pages, copies the prefix from the owner, inserts it into its
   tree with the window component marked evicted (bounded replay rebuilds it, as for any
   window-evicted hit), and admits the request into the budget its own queue leaves.
3. After the forward the borrower copies the full pages past the prefix into the
   owner's pages and replies LendDone.
4. The owner inserts those pages into its tree (window evicted) and requeues the
   request at its own position. Its next batch computes only the partial last page and
   the replayed rows, samples, and sends to decode as for any request, so decode, the
   bootstrap room and the session's rank do not change.

This module holds the parts that need no GPU: the configuration, the per-rank status,
the lend policy, the owner's and borrower's ledgers, the page-run planning for copies
and the wire messages. The ledgers keep two invariants:
- the owner's prefix stays locked from the lend until a final reply (done, failed or
  cancel-acked);
- the owner never reuses its destination pages before a final reply, because the
  borrower writes into them only before that reply.
"""

from __future__ import annotations

import enum
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import msgspec
import msgspec.msgpack


class LendingConfig(msgspec.Struct, frozen=True):
    enabled: bool = False
    max_inflight_per_borrower: int = 4
    min_uncached_tokens: int = 1024
    margin_tokens: int = 1024
    status_max_age_s: float = 1.0
    timeout_s: float = 30.0
    force: bool = False

    @classmethod
    def from_env(cls) -> "LendingConfig":
        from sglang.srt.environ import envs

        return cls(
            enabled=envs.SGLANG_ENABLE_DP_PREFILL_LENDING.get(),
            max_inflight_per_borrower=envs.SGLANG_DP_PREFILL_LENDING_MAX_INFLIGHT.get(),
            min_uncached_tokens=envs.SGLANG_DP_PREFILL_LENDING_MIN_UNCACHED_TOKENS.get(),
            margin_tokens=envs.SGLANG_DP_PREFILL_LENDING_MARGIN_TOKENS.get(),
            status_max_age_s=envs.SGLANG_DP_PREFILL_LENDING_STATUS_MAX_AGE_S.get(),
            timeout_s=envs.SGLANG_DP_PREFILL_LENDING_TIMEOUT_S.get(),
            force=envs.SGLANG_DP_PREFILL_LENDING_FORCE.get(),
        )


# ---------------------------------------------------------------- status and policy


class RankStatus(msgspec.Struct, frozen=True, tag=True):
    """What a rank publishes after it forms a batch."""

    rank: int
    seq: int
    # Uncached tokens queued beyond what its next batch takes.
    surplus_tokens: int
    # Budget its next batch leaves after its own queue.
    spare_tokens: int
    borrowed_inflight: int
    stamp: float


class LendCandidate(msgspec.Struct, frozen=True):
    """A waiting request past the owner's next batch, in the owner's FCFS order."""

    rid: str
    uncached_tokens: int
    # The matched prefix is fully on the device (no host or storage hit pending).
    prefix_on_device: bool


def split_surplus(
    uncached_in_fcfs_order: Sequence[Tuple[str, int, bool]], next_batch_budget: int
) -> Tuple[List[LendCandidate], int]:
    """Split the waiting queue (rid, uncached tokens, prefix on device) after the batch
    just formed into what the next batch takes and the surplus behind it. Returns the
    surplus candidates and the surplus token count. A request the next batch would only
    start (cut by the budget) stays: lending takes whole requests only."""
    taken = 0
    out: List[LendCandidate] = []
    surplus = 0
    for rid, uncached, on_device in uncached_in_fcfs_order:
        if taken < next_batch_budget:
            taken += uncached
            continue
        out.append(LendCandidate(rid, uncached, on_device))
        surplus += uncached
    return out, surplus


def plan_lends(
    candidates: Sequence[LendCandidate],
    own_rank: int,
    peers: Mapping[int, RankStatus],
    cfg: LendingConfig,
    now: float,
    lent_to: Optional[Mapping[int, int]] = None,
) -> List[Tuple[str, int]]:
    """Pick (rid, borrower) pairs.

    A borrower is a fresh, non-donor peer (no surplus) whose spare budget takes the whole
    request plus a margin, with an in-flight slot left. Candidates go in the owner's FCFS
    order; one that fits no borrower stays home and later ones may still go, since a
    lend only moves work into idle capacity and never delays the requests that stay.
    The borrower with the most spare budget takes each request (worst fit spreads load).
    ``lent_to`` counts this owner's lends already in flight per borrower.
    """
    lent_to = dict(lent_to or {})
    spare: Dict[int, int] = {}
    slots: Dict[int, int] = {}
    for r, st in peers.items():
        if r == own_rank or now - st.stamp > cfg.status_max_age_s:
            continue
        if not cfg.force and st.surplus_tokens > 0:
            continue
        free = cfg.max_inflight_per_borrower - max(st.borrowed_inflight, lent_to.get(r, 0))
        if free <= 0:
            continue
        spare[r] = st.spare_tokens
        slots[r] = free
    plan: List[Tuple[str, int]] = []
    for c in candidates:
        if not c.prefix_on_device:
            continue
        if not cfg.force and c.uncached_tokens < cfg.min_uncached_tokens:
            continue
        need = c.uncached_tokens + cfg.margin_tokens
        fits = [r for r in spare if slots[r] > 0 and (cfg.force or spare[r] >= need)]
        if not fits:
            continue
        r = max(fits, key=lambda x: (spare[x], -x))
        plan.append((c.rid, r))
        spare[r] -= need
        slots[r] -= 1
    return plan


# ---------------------------------------------------------------- page-run planning


def page_runs(src_pages: Sequence[int], dst_pages: Sequence[int]) -> List[Tuple[int, int, int]]:
    """Coalesce page pairs into (src_start, dst_start, count) runs where both sides are
    consecutive, so each run is one copy per region."""
    if len(src_pages) != len(dst_pages):
        raise ValueError(f"page lists differ: {len(src_pages)} vs {len(dst_pages)}")
    runs: List[Tuple[int, int, int]] = []
    for s, d in zip(src_pages, dst_pages):
        if runs:
            s0, d0, n = runs[-1]
            if s == s0 + n and d == d0 + n:
                runs[-1] = (s0, d0, n + 1)
                continue
        runs.append((int(s), int(d), 1))
    return runs


def region_copies(
    src_ptrs: Sequence[int],
    dst_ptrs: Sequence[int],
    item_lens: Sequence[int],
    runs: Sequence[Tuple[int, int, int]],
) -> Tuple[List[int], List[int], List[int]]:
    """(src addresses, dst addresses, lengths) for every region and run. Both ranks
    share one pool layout, so a region's per-page bytes are the same on both sides."""
    if not (len(src_ptrs) == len(dst_ptrs) == len(item_lens)):
        raise ValueError("region lists differ in length")
    srcs: List[int] = []
    dsts: List[int] = []
    lens: List[int] = []
    for sp, dp, il in zip(src_ptrs, dst_ptrs, item_lens):
        for s, d, n in runs:
            srcs.append(sp + s * il)
            dsts.append(dp + d * il)
            lens.append(n * il)
    return srcs, dsts, lens


def full_page_split(prefix_tokens: int, input_tokens: int, page_size: int) -> Tuple[int, int]:
    """(prefix pages, pages the borrower computes and returns): the full pages past the
    prefix. The owner keeps the partial last page, which it recomputes with the replay
    rows. The prefix is page aligned (tree matches are)."""
    if prefix_tokens % page_size:
        raise ValueError(f"prefix {prefix_tokens} is not page aligned ({page_size})")
    # The last input token is always computed by the owner, which samples it.
    full = (input_tokens - 1) // page_size
    pre = prefix_tokens // page_size
    return pre, max(0, full - pre)


# ---------------------------------------------------------------- wire messages


class LendRequest(msgspec.Struct, tag=True):
    lend_id: int
    owner: int
    rid: str
    input_ids: List[int]
    prefix_tokens: int
    # Owner's pages: the prefix (to copy from) and the destination of the returned pages.
    owner_prefix_pages: List[int]
    owner_dst_pages: List[int]
    extra_key: Optional[str] = None


class LendDone(msgspec.Struct, tag=True):
    lend_id: int
    borrower: int
    ok: bool
    reason: str = ""
    # Seconds from receipt to the returned pages landing, for the log line.
    elapsed_s: float = 0.0


class LendCancel(msgspec.Struct, tag=True):
    lend_id: int
    owner: int


class LendCancelAck(msgspec.Struct, tag=True):
    lend_id: int
    borrower: int


class PeerRegions(msgspec.Struct, tag=True):
    """A rank's pool regions for copies, exchanged once at startup."""

    rank: int
    session_id: str
    kv_ptrs: List[int]
    kv_item_lens: List[int]
    page_size: int


class SeedReady(msgspec.Struct, tag=True):
    """Borrower to owner: pages allocated for the prefix; write them."""

    lend_id: int
    borrower: int
    borrower_prefix_pages: List[int]


class SeedDone(msgspec.Struct, tag=True):
    """Owner to borrower: the prefix pages have landed (ok) or the lend is off."""

    lend_id: int
    owner: int
    ok: bool


LendMessage = (
    RankStatus | PeerRegions | LendRequest | SeedReady | SeedDone | LendDone | LendCancel | LendCancelAck
)
_ENCODER = msgspec.msgpack.Encoder()
_DECODER = msgspec.msgpack.Decoder(LendMessage)


def encode_message(msg: LendMessage) -> bytes:
    return _ENCODER.encode(msg)


def decode_message(data: bytes) -> LendMessage:
    return _DECODER.decode(data)


# ---------------------------------------------------------------- owner ledger


class OwnerAction(enum.Enum):
    INSERT_AND_REQUEUE = "insert_and_requeue"  # pages arrived: insert, unlock, requeue
    REQUEUE_KEEP_QUARANTINE = "requeue_keep_quarantine"  # timed out: run it at home now
    FREE_QUARANTINE = "free_quarantine"  # final reply after a timeout or abort
    REQUEUE_AND_FREE = "requeue_and_free"  # borrower failed before writing: run at home
    RETIRE_AND_FREE = "retire_and_free"  # aborted request, final reply: free all
    IGNORE = "ignore"


class LendState(enum.Enum):
    LENT = "lent"
    REQUEUED_AWAITING_FINAL = "requeued_awaiting_final"  # timed out, pages quarantined
    ABORTED_AWAITING_FINAL = "aborted_awaiting_final"


class OwnerEntry(msgspec.Struct):
    lend_id: int
    rid: str
    borrower: int
    sent_at: float
    state: LendState = LendState.LENT


class OwnerLedger:
    """The owner's lends. A lend ends only on a final reply (done, failed or cancel-ack)
    from its borrower; until then the destination pages stay allocated (quarantined if the
    request already ran at home), so a late write can never land in reused pages."""

    def __init__(self, timeout_s: float):
        self.timeout_s = timeout_s
        self._next = 0
        self.entries: Dict[int, OwnerEntry] = {}
        self.by_rid: Dict[str, int] = {}

    def new_lend(self, rid: str, borrower: int, now: float) -> int:
        if rid in self.by_rid:
            raise ValueError(f"{rid} is already lent")
        self._next += 1
        e = OwnerEntry(self._next, rid, borrower, now)
        self.entries[e.lend_id] = e
        self.by_rid[rid] = e.lend_id
        return e.lend_id

    def inflight_by_borrower(self) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for e in self.entries.values():
            out[e.borrower] = out.get(e.borrower, 0) + 1
        return out

    def is_lent(self, rid: str) -> bool:
        lid = self.by_rid.get(rid)
        return lid is not None and self.entries[lid].state == LendState.LENT

    def _close(self, e: OwnerEntry) -> None:
        del self.entries[e.lend_id]
        if self.by_rid.get(e.rid) == e.lend_id:
            del self.by_rid[e.rid]

    def on_done(self, msg: LendDone) -> Tuple[OwnerAction, Optional[OwnerEntry]]:
        e = self.entries.get(msg.lend_id)
        if e is None or e.borrower != msg.borrower:
            return OwnerAction.IGNORE, None
        self._close(e)
        if e.state == LendState.LENT:
            return (OwnerAction.INSERT_AND_REQUEUE if msg.ok else OwnerAction.REQUEUE_AND_FREE), e
        if e.state == LendState.REQUEUED_AWAITING_FINAL:
            return OwnerAction.FREE_QUARANTINE, e
        return OwnerAction.RETIRE_AND_FREE, e

    def on_cancel_ack(self, msg: LendCancelAck) -> Tuple[OwnerAction, Optional[OwnerEntry]]:
        e = self.entries.get(msg.lend_id)
        if e is None or e.borrower != msg.borrower:
            return OwnerAction.IGNORE, None
        self._close(e)
        if e.state == LendState.ABORTED_AWAITING_FINAL:
            return OwnerAction.RETIRE_AND_FREE, e
        return OwnerAction.FREE_QUARANTINE, e

    def expire(self, now: float) -> List[OwnerEntry]:
        """Lends past the timeout: the request runs at home with fresh pages; the lent
        pages wait for the borrower's final reply. Returns those to requeue and cancel."""
        out = []
        for e in self.entries.values():
            if e.state == LendState.LENT and now - e.sent_at > self.timeout_s:
                e.state = LendState.REQUEUED_AWAITING_FINAL
                if self.by_rid.get(e.rid) == e.lend_id:
                    del self.by_rid[e.rid]
                out.append(e)
        return out

    def on_abort(self, rid: str) -> Optional[OwnerEntry]:
        """The client aborted a lent request: cancel it and wait for the final reply."""
        lid = self.by_rid.get(rid)
        if lid is None:
            return None
        e = self.entries[lid]
        if e.state != LendState.LENT:
            return None
        e.state = LendState.ABORTED_AWAITING_FINAL
        del self.by_rid[rid]
        return e


# ---------------------------------------------------------------- borrower ledger


class BorrowState(enum.Enum):
    RECEIVED = "received"
    SEEDING = "seeding"  # prefix copy in flight
    QUEUED = "queued"  # in the side queue or the waiting queue
    RETURNING = "returning"  # computed pages being copied to the owner
    CANCELLED = "cancelled"  # cancel arrived; ack once nothing of ours is in flight


class BorrowAction(enum.Enum):
    ACK_NOW = "ack_now"
    ACK_AFTER_COPY = "ack_after_copy"  # a copy into the owner's pages is in flight
    DROP_REQUEST = "drop_request"  # remove the local request, then ack
    IGNORE = "ignore"


class BorrowEntry(msgspec.Struct):
    lend: LendRequest
    received_at: float
    state: BorrowState = BorrowState.RECEIVED
    local_rid: str = ""


class BorrowerLedger:
    """The borrower's loans. It writes into the owner's pages only in RETURNING, and
    acknowledges a cancel only when no such write is in flight."""

    def __init__(self, rank: int):
        self.rank = rank
        self.entries: Dict[int, BorrowEntry] = {}

    def key(self, lend: LendRequest) -> Tuple[int, int]:
        return (lend.owner, lend.lend_id)

    def add(self, lend: LendRequest, now: float) -> BorrowEntry:
        k = self.key(lend)
        if k in self.entries:
            raise ValueError(f"duplicate lend {k}")
        e = BorrowEntry(lend, now, local_rid=f"lend-{lend.owner}-{lend.lend_id}-{lend.rid}")
        self.entries[k] = e
        return e

    def inflight(self) -> int:
        return sum(1 for e in self.entries.values() if e.state != BorrowState.CANCELLED)

    def advance(self, owner: int, lend_id: int, to: BorrowState) -> BorrowEntry:
        e = self.entries[(owner, lend_id)]
        order = [BorrowState.RECEIVED, BorrowState.SEEDING, BorrowState.QUEUED, BorrowState.RETURNING]
        if e.state == BorrowState.CANCELLED:
            raise ValueError(f"lend {(owner, lend_id)} was cancelled")
        if order.index(to) <= order.index(e.state):
            raise ValueError(f"lend {(owner, lend_id)}: {e.state} -> {to} goes backwards")
        e.state = to
        return e

    def finish(self, owner: int, lend_id: int) -> Optional[BorrowEntry]:
        return self.entries.pop((owner, lend_id), None)

    def on_cancel(self, msg: LendCancel) -> Tuple[BorrowAction, Optional[BorrowEntry]]:
        e = self.entries.get((msg.owner, msg.lend_id))
        if e is None:
            # Already finished: the owner gets (or got) LendDone, which is final.
            return BorrowAction.IGNORE, None
        prev = e.state
        e.state = BorrowState.CANCELLED
        if prev == BorrowState.RETURNING:
            return BorrowAction.ACK_AFTER_COPY, e
        if prev in (BorrowState.SEEDING, BorrowState.QUEUED):
            return BorrowAction.DROP_REQUEST, e
        return BorrowAction.ACK_NOW, e


def lending_counters_line(c: Mapping[str, float]) -> str:
    keys = ("lent", "done", "failed", "timed_out", "borrowed", "returned", "copy_ms_p50", "copy_ms_p99")
    return "dp prefill lending: " + " ".join(f"{k}={c.get(k, 0)}" for k in keys)


def percentile(xs: Iterable[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return 0.0
    return s[min(len(s) - 1, int(q * len(s)))]


def now() -> float:
    return time.monotonic()
