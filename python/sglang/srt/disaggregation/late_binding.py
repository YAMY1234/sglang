"""Late binding of the decode DP rank (prototype, --disaggregation-decode-late-binding).

Under the prefill_complete allocation policy a request reaches a decode DP rank at
arrival but allocates KV only after prefill reports readiness. At that moment,
before any allocation, the rank may hand the request to a less committed DP rank
of the same decode server; the target re-binds the prefill room with REBIND.

Decision input: each rank's committed count, KV usage and last handoff target,
carried as three extra columns on the existing per-step MLP-sync gather, so every
rank decides from the same snapshot without a new collective.
Channel: a direct ZMQ PUSH/PULL between the decode schedulers (attn-TP rank 0 of
each DP group), addresses exchanged once at startup over the TP CPU group.
"""

from __future__ import annotations

import itertools
import logging
import os
import zlib
import pickle
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import zmq

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeRequest

logger = logging.getLogger(__name__)
# Handoff target by each rank's binding resource, max(committed, kv usage x cap) in request units, not committed alone.
_DOMINANT = os.environ.get("SGLANG_D_LATEBIND_DOMINANT", "0") == "1"
# Same sample as D_TIMELINE: no-move decisions are logged for 1 in N rids (crc32); every handoff is logged.
_DECIDE_SAMPLE_N = int(os.environ.get("SGLANG_D_REQ_TIMELINE", "0"))

# Gathered row: (committed, kv_usage_permille, handoff_target_since_last_sync).
NUM_LOAD_COLUMNS = 3
# Fallback row for an inactive / missing DP slot: never a target, never a source.
INACTIVE_LOAD_ROW = (1 << 40, 1000, -1)
SUMMARY_INTERVAL_S = 30.0
# Bounded send queue per peer: a full queue fails the send (non-blocking).
HANDOFF_SNDHWM = 64
# Bounded fences / records on the target, and per-peer ACK/CANCEL outbox.
MAX_FENCED_IDS = 4096
MAX_OUTBOX = 4096
# A CANCEL is retried this many ack timeouts before it is dropped (fail-loud).
CANCEL_LIFETIME_FACTOR = 10
# Bounded startup wait for the pre-connected D->D sockets' handshakes.
PRECONNECT_WAIT_S = 5.0
_CONNECTED_EVENT = getattr(zmq, "EVENT_HANDSHAKE_SUCCEEDED", zmq.EVENT_CONNECTED)
# Always printed in LATEBIND_D_SUMMARY, zero included.
ALWAYS_REPORTED = (
    "handoff_out",
    "handoff_acked",
    "handoff_in",
    "handoff_send_fail",
    "peer_not_connected_skip",
    "ack_retry",
    "ack_send_fail",
    "cancel_retry",
    "handoff_timeout_abort",
    "late_handoff_dropped",
    "cancel_after_enqueue",
    "cancel_send_fail",
    "ack_unknown",
    "handoff_refused",
    "double_handoff",
    "abort_in_transit",
    "eligible",
    "eligible_with_session_id",
    "decide_target",
    "decide_no_target",
    "inelig_rebootstrap",
    "inelig_fake",
    "inelig_no_recv_req",
    "inelig_session_state",
    "inelig_no_receiver",
    "inelig_aborted",
    "inelig_cannot_detach",
)


def choose_handoff_target(
    rows: Sequence[Sequence[int]], self_rank: int, threshold: int
) -> Optional[int]:
    """Return the DP rank this rank should hand one ready request to, or None.

    rows[r] = (committed, kv_permille, handoff_target) from the same gather on
    every rank. A handoff announced in the snapshot counts +1 on its target.
    Rule ("late_count"): this rank is a source if committed >= min(committed)
    + threshold; the most committed source gets the minimum-count rank (tie:
    lowest rank id). Further sources (by count desc, then rank id) take the
    next-lowest ranks in order, so concurrent sources never pick the same
    target from one snapshot; a pair is used only if it still differs by
    >= threshold. KV usage (column 1) is logged, not used.
    """
    if threshold < 1 or not (0 <= self_rank < len(rows)):
        return None
    committed = [int(row[0]) for row in rows]
    for row in rows:
        if 0 <= int(row[2]) < len(rows):
            committed[int(row[2])] += 1
    active = [r for r in range(len(rows)) if int(rows[r][0]) < INACTIVE_LOAD_ROW[0]]
    if self_rank not in active or len(active) < 2:
        return None
    floor = min(committed[r] for r in active)
    sources = sorted(
        (r for r in active if committed[r] >= floor + threshold),
        key=lambda r: (-committed[r], r),
    )
    if self_rank not in sources:
        return None
    targets = sorted(active, key=lambda r: (committed[r], r))
    i = sources.index(self_rank)
    if i >= len(targets):
        return None
    target = targets[i]
    if target == self_rank or committed[self_rank] < committed[target] + threshold:
        return None
    return target


def choose_handoff_target_dominant(
    rows: Sequence[Sequence[int]], self_rank: int, threshold: int, cap: int, req_kv_frac: float = 0.0
) -> Optional[int]:
    """choose_handoff_target with each rank scored by its binding resource in request units,
    max(committed, kv_permille / 1000 * cap) (both columns are already in the gathered row), so with KV light it is
    exactly the count rule. Margin `threshold`; the move must also leave the target, after taking this request
    (+1 committed, +req_kv_frac of the pool), below the source."""
    if threshold < 1 or cap < 1 or not (0 <= self_rank < len(rows)):
        return None
    committed = [int(row[0]) for row in rows]
    kv = [int(row[1]) / 1000.0 for row in rows]
    for row in rows:
        if 0 <= int(row[2]) < len(rows):
            committed[int(row[2])] += 1
    active = [r for r in range(len(rows)) if int(rows[r][0]) < INACTIVE_LOAD_ROW[0]]
    if self_rank not in active or len(active) < 2:
        return None
    score = {r: max(committed[r], kv[r] * cap) for r in active}
    margin = threshold
    floor = min(score.values())
    sources = sorted((r for r in active if score[r] >= floor + margin), key=lambda r: (-score[r], r))
    if self_rank not in sources:
        return None
    targets = sorted(active, key=lambda r: (score[r], r))
    i = sources.index(self_rank)
    if i >= len(targets):
        return None
    target = targets[i]
    if target == self_rank or score[self_rank] < score[target] + margin:
        return None
    if max(committed[target] + 1, (kv[target] + req_kv_frac) * cap) >= score[self_rank]:
        return None
    return target


@dataclass
class LateBindingHandoff:
    """Source -> target: the original tokenized request plus the old endpoint."""

    recv_req: Any  # TokenizedGenerateReqInput as received by the source
    room: int
    old_endpoint: Tuple[str, int]
    source_dp_rank: int
    target_dp_rank: int
    sent_at: float  # time.time() on the source, just before the send
    handoff_id: str


@dataclass
class LateBindingAck:
    """Target -> source: the handoff was enqueued (ok) or refused."""

    rid: str
    room: int
    handoff_id: str
    ok: bool
    reason: str = ""


@dataclass
class LateBindingCancel:
    """Source -> target after an ack timeout: fence or abort this handoff."""

    rid: str
    room: int
    handoff_id: str
    target_dp_rank: int
    source_dp_rank: int = -1


class DecodeLateBinding:
    """Per decode scheduler state for late binding (one per DP-rank process).

    Ownership rule: the source owns a handed-off request until the target's ack
    arrives. If no ack arrives within ack_timeout, the source terminally fails
    the request (client error, ABORT to P from the old endpoint) and sends
    CANCEL. The target drops a handoff whose id was cancelled; a CANCEL for an
    id it already enqueued aborts its copy. Either way one terminal owner.
    """

    def __init__(
        self,
        *,
        dp_rank: int,
        num_dp_ranks: int,
        threshold: int,
        is_sender: bool,
        local_ip: str,
        kv_manager=None,
        context: Optional[zmq.Context] = None,
        ack_timeout: float = 10.0,
        get_limits: Optional[Callable[[], Tuple[int, int]]] = None,
    ):
        self.dp_rank = dp_rank
        self.get_limits = get_limits  # () -> (running cap, KV pool tokens) of this rank, read at decide time
        self.num_dp_ranks = num_dp_ranks
        self.threshold = threshold
        self.ack_timeout = ack_timeout
        self.is_sender = is_sender
        self.stats: Counter = Counter()
        self.rows: Optional[List[Tuple[int, int, int]]] = None
        self._rows_version = 0
        self._decided_version = -1
        # Target of the handoff made since the last gather (-1 none).
        self.sent_to = -1
        self._ids = itertools.count()
        # handoff_id -> (handoff, monotonic sent, owner context) awaiting ack.
        self.pending_acks: Dict[str, Tuple[LateBindingHandoff, float, Any]] = {}
        # Per-peer FIFO of (deadline, kind, handoff_id, bytes): ACKs and CANCELs
        # not yet accepted by the socket, retried non-blockingly every poll.
        self.outbox: Dict[int, deque] = {}
        # Peers whose PUSH socket finished the ZMTP handshake (socket monitor).
        self.connected: set = set()
        self._monitors: Dict[int, zmq.Socket] = {}
        # Received handoffs not yet carried by a scheduler receive pass.
        self.arrivals: List[LateBindingHandoff] = []
        # rid -> handoff, consumed by _create_receiver_and_enqueue on the target.
        self.incoming: Dict[str, LateBindingHandoff] = {}
        # Target fences: cancelled ids, and ids enqueued here (-> rid).
        self.cancelled: "OrderedDict[str, float]" = OrderedDict()
        self.enqueued: "OrderedDict[str, str]" = OrderedDict()
        # (AbortReq, time.time()) recently seen, for requests aborted in transit.
        self.recent_aborts: List[Tuple[Any, float]] = []
        # Source: terminal failure of a timed-out handoff (handoff, owner ctx).
        self.on_timeout: Optional[Callable[[LateBindingHandoff, Any], None]] = None
        # Target: abort an enqueued copy whose handoff was cancelled (rid).
        self.on_cancel_enqueued: Optional[Callable[[str], None]] = None
        self._next_summary = time.monotonic() + SUMMARY_INTERVAL_S
        self.peers: List[Optional[str]] = []
        self._push: Dict[int, zmq.Socket] = {}
        self._ctx = context
        self.pull = None
        self.address = None
        if is_sender:
            from sglang.srt.utils.network import get_zmq_socket_on_host

            self._ctx = context or zmq.Context()
            port, self.pull = get_zmq_socket_on_host(
                self._ctx, zmq.PULL, host=local_ip, restrict_peers=True
            )
            self.address = f"tcp://{local_ip}:{port}"
        if kv_manager is not None:
            kv_manager.late_binding_stats = self.stats
            kv_manager.late_binding_handed_off = {}
        self.kv_manager = kv_manager

    # -- startup -------------------------------------------------------------
    def exchange_addresses(self, group, ranks_per_dp: int) -> None:
        """Collective over the TP CPU group (all ranks of the decode server)."""
        import torch.distributed as dist

        gathered: List[Optional[str]] = [None] * dist.get_world_size(group)
        dist.all_gather_object(gathered, self.address, group=group)
        # Group rank g belongs to DP rank g // ranks_per_dp; take its first rank.
        self.peers = [gathered[r * ranks_per_dp] for r in range(self.num_dp_ranks)]
        if any(peer is None for peer in self.peers):
            raise RuntimeError(f"Late binding: missing handoff address {self.peers}")

    # -- MLP-sync columns ----------------------------------------------------
    def local_row(self, committed: int, kv_usage: float) -> Tuple[int, int, int]:
        return (int(committed), int(round(kv_usage * 1000)), int(self.sent_to))

    def on_rows(self, rows: List[List[int]]) -> None:
        self.rows = [tuple(int(v) for v in row) for row in rows]
        self._rows_version += 1
        self.sent_to = -1

    # -- decision ------------------------------------------------------------
    def decide(self, decode_req: "DecodeRequest") -> Optional[int]:
        """Called when a prealloc request first becomes ready, before allocation."""
        if decode_req.late_binding_hops > 0:
            return None  # cap: one handoff per request
        if self.rows is None or self._decided_version == self._rows_version:
            return None  # at most one handoff attempt per gathered snapshot
        if len(self.rows) != self.num_dp_ranks:
            return None
        cap, pool = self.get_limits() if (_DOMINANT and self.get_limits is not None) else (0, 0)
        frac = len(decode_req.req.origin_input_ids) / pool if pool > 0 else 0.0
        dominant = cap > 0 and pool > 0
        if dominant:
            target = choose_handoff_target_dominant(self.rows, self.dp_rank, self.threshold, cap, frac)
        else:
            target = choose_handoff_target(self.rows, self.dp_rank, self.threshold)
        rid = decode_req.req.rid
        if target is not None or (_DECIDE_SAMPLE_N and zlib.crc32(rid.encode()) % _DECIDE_SAMPLE_N == 0):
            logger.info(
                "LATEBIND_DECIDE rid=%s dp=%d rule=%s rows=%s cap=%d pool=%d req_kv_frac=%.5f threshold=%d target=%s",
                rid, self.dp_rank, "dominant" if dominant else "count", list(self.rows), cap, pool, frac,
                self.threshold, target,
            )
        if target is not None:
            self._decided_version = self._rows_version
        return target

    def new_handoff_id(self, room: int) -> str:
        return f"{self.dp_rank}-{next(self._ids)}-{room}"

    # -- transport -----------------------------------------------------------
    def _socket_to(self, dp_rank: int) -> zmq.Socket:
        sock = self._push.get(dp_rank)
        if sock is None:
            sock = self._ctx.socket(zmq.PUSH)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.SNDHWM, HANDOFF_SNDHWM)
            # Queue only to a connected peer: no peer -> EAGAIN, not a hidden backlog.
            sock.setsockopt(zmq.IMMEDIATE, 1)
            self._monitors[dp_rank] = sock.get_monitor_socket(
                _CONNECTED_EVENT | zmq.EVENT_DISCONNECTED
            )
            sock.connect(self.peers[dp_rank])
            self._push[dp_rank] = sock
        return sock

    def _poll_monitors(self) -> None:
        from zmq.utils.monitor import recv_monitor_message

        for dp_rank, monitor in self._monitors.items():
            while True:
                try:
                    event = recv_monitor_message(monitor, zmq.NOBLOCK)
                except zmq.Again:
                    break
                if event["event"] == _CONNECTED_EVENT:
                    self.connected.add(dp_rank)
                elif event["event"] == zmq.EVENT_DISCONNECTED:
                    self.connected.discard(dp_rank)

    def connect_peers(self, wait_s: float = PRECONNECT_WAIT_S) -> None:
        """Startup: open every D->D PUSH socket and wait (bounded) for the
        handshakes, so first handoffs/ACKs do not race the connection."""
        others = [r for r in range(len(self.peers)) if r != self.dp_rank]
        for r in others:
            self._socket_to(r)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            self._poll_monitors()
            if all(r in self.connected for r in others):
                break
            time.sleep(0.01)
        missing = [r for r in others if r not in self.connected]
        if missing:
            logger.warning("Late binding: DP%d not yet connected to %s", self.dp_rank, missing)

    def _send_bytes(self, dp_rank: int, data: bytes) -> bool:
        """Never blocks: a full queue, an unconnected peer or any socket error
        returns False."""
        try:
            self._socket_to(dp_rank).send(data, zmq.NOBLOCK)
            return True
        except zmq.ZMQError:
            return False

    def _try_send(self, dp_rank: int, obj) -> bool:
        return self._send_bytes(
            dp_rank, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        )

    def send_handoff(self, handoff: LateBindingHandoff, owner: Any = None) -> bool:
        """True only if the payload is queued; the caller detaches only then.
        Not retried: on failure the request simply stays on this rank."""
        self._poll_monitors()
        if handoff.target_dp_rank not in self.connected:
            self.stats["peer_not_connected_skip"] += 1
            return False
        if not self._try_send(handoff.target_dp_rank, handoff):
            self.stats["handoff_send_fail"] += 1
            return False
        self.sent_to = handoff.target_dp_rank
        self.stats["handoff_out"] += 1
        self.pending_acks[handoff.handoff_id] = (handoff, time.monotonic(), owner)
        return True

    def _enqueue_out(self, dp_rank: int, kind: str, handoff_id: str, obj, lifetime: float) -> None:
        box = self.outbox.setdefault(dp_rank, deque())
        if len(box) >= MAX_OUTBOX:
            _, old_kind, old_id, _ = box.popleft()
            self.stats[f"{old_kind}_send_fail"] += 1
            logger.error("Late binding: outbox to DP%d full, dropped %s %s", dp_rank, old_kind, old_id)
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        box.append((time.monotonic() + lifetime, kind, handoff_id, data))
        self.flush_outbox()

    def flush_outbox(self, now: Optional[float] = None) -> None:
        """Retry queued ACKs/CANCELs in order, without blocking; drop expired
        ones loudly. Called on every poll (every scheduler loop)."""
        now = time.monotonic() if now is None else now
        for dp_rank, box in self.outbox.items():
            while box:
                deadline, kind, handoff_id, data = box[0]
                if now > deadline:
                    box.popleft()
                    self.stats[f"{kind}_send_fail"] += 1
                    logger.error(
                        "Late binding: %s for handoff %s to DP%d never delivered",
                        kind,
                        handoff_id,
                        dp_rank,
                    )
                    continue
                if not self._send_bytes(dp_rank, data):
                    self.stats[f"{kind}_retry"] += 1
                    break
                box.popleft()

    def _drop_queued_ack(self, handoff_id: str, dp_rank: int) -> None:
        box = self.outbox.get(dp_rank)
        if box:
            self.outbox[dp_rank] = deque(
                e for e in box if not (e[1] == "ack" and e[2] == handoff_id)
            )

    def send_ack(self, handoff: LateBindingHandoff, ok: bool, reason: str = "") -> None:
        """Queued and retried until delivered or the source's ack timeout
        passed (after which the source has decided and ignores it)."""
        ack = LateBindingAck(
            handoff.recv_req.rid, handoff.room, handoff.handoff_id, ok, reason
        )
        self._enqueue_out(
            handoff.source_dp_rank, "ack", handoff.handoff_id, ack, self.ack_timeout
        )

    def poll(self) -> List[LateBindingHandoff]:
        """Drain the PULL socket: returns handoffs to ingest; consumes acks and
        cancels; retries the outbox; enforces ack timeouts. Never blocks."""
        arrivals: List[LateBindingHandoff] = []
        if self.pull is not None:
            while True:
                try:
                    data = self.pull.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
                msg = pickle.loads(data)
                if isinstance(msg, LateBindingAck):
                    self._on_ack(msg)
                elif isinstance(msg, LateBindingCancel):
                    self.on_cancel(msg)
                elif self.is_fenced(msg):
                    pass
                else:
                    arrivals.append(msg)
        self._poll_monitors()
        self.flush_outbox()
        self.check_timeouts()
        return arrivals

    def _on_ack(self, ack: LateBindingAck) -> None:
        entry = self.pending_acks.pop(ack.handoff_id, None)
        if entry is None:
            # Arrived after the timeout: the CANCEL already fences/aborts it.
            self.stats["ack_unknown"] += 1
            logger.error(
                "Late binding: ack for unknown/expired handoff %s rid=%s",
                ack.handoff_id,
                ack.rid,
            )
            return
        if ack.ok:
            self.stats["handoff_acked"] += 1
            self.stats["handoff_ack_us_sum"] += int(
                (time.monotonic() - entry[1]) * 1e6
            )
        else:
            self.stats["handoff_refused"] += 1
            logger.error(
                "Late binding: target refused rid=%s room=%s: %s",
                ack.rid,
                ack.room,
                ack.reason,
            )

    def check_timeouts(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        for handoff_id, (handoff, sent, owner) in list(self.pending_acks.items()):
            if now - sent <= self.ack_timeout:
                continue
            del self.pending_acks[handoff_id]
            self.stats["handoff_timeout_abort"] += 1
            logger.warning(
                "LATEBIND_HANDOFF_TIMEOUT_ABORT rid=%s room=%s handoff_id=%s "
                "source=%d target=%d waited_ms=%.1f",
                handoff.recv_req.rid,
                handoff.room,
                handoff_id,
                handoff.source_dp_rank,
                handoff.target_dp_rank,
                (now - sent) * 1000,
            )
            # The timeout is final for this handoff: a later ACK is ignored
            # (_on_ack), and the CANCEL fences or aborts the target's copy.
            self._enqueue_out(
                handoff.target_dp_rank,
                "cancel",
                handoff_id,
                LateBindingCancel(
                    handoff.recv_req.rid,
                    handoff.room,
                    handoff_id,
                    handoff.target_dp_rank,
                    handoff.source_dp_rank,
                ),
                CANCEL_LIFETIME_FACTOR * self.ack_timeout,
            )
            if self.on_timeout is not None:
                self.on_timeout(handoff, owner)
        if self.kv_manager is not None:
            handed = self.kv_manager.late_binding_handed_off
            for room, at in list(handed.items()):
                if now - at > 10 * self.ack_timeout:
                    del handed[room]

    # -- target fences -------------------------------------------------------
    def _remember(self, table: "OrderedDict", key: str, value) -> None:
        table[key] = value
        while len(table) > MAX_FENCED_IDS:
            table.popitem(last=False)

    def is_fenced(self, handoff: LateBindingHandoff) -> bool:
        if handoff.handoff_id in self.cancelled:
            self.stats["late_handoff_dropped"] += 1
            logger.error(
                "Late binding: dropped handoff %s rid=%s that arrived after its CANCEL",
                handoff.handoff_id,
                handoff.recv_req.rid,
            )
            return True
        return False

    def note_enqueued(self, handoff: LateBindingHandoff) -> None:
        self._remember(self.enqueued, handoff.handoff_id, handoff.recv_req.rid)

    def on_cancel(self, cancel: LateBindingCancel) -> None:
        rid = self.enqueued.pop(cancel.handoff_id, None)
        if rid is None:
            # Not enqueued (yet): fence it; buffered arrivals are dropped too.
            self._remember(self.cancelled, cancel.handoff_id, time.monotonic())
            before = len(self.arrivals)
            self.arrivals = [a for a in self.arrivals if not self.is_fenced(a)]
            if before == len(self.arrivals):
                self.stats["cancel_fenced"] += 1
            return
        # Ack was lost or late: the source already failed the request, so the
        # copy here must stop too (exactly one terminal state). An ACK still
        # queued here is now moot; drop it.
        self.stats["cancel_after_enqueue"] += 1
        self._drop_queued_ack(cancel.handoff_id, cancel.source_dp_rank)
        logger.error(
            "Late binding: CANCEL after enqueue, aborting copy rid=%s handoff %s",
            rid,
            cancel.handoff_id,
        )
        if self.on_cancel_enqueued is not None:
            self.on_cancel_enqueued(rid)

    def admit(
        self,
        handoff: LateBindingHandoff,
        *,
        rid_present: Callable[[str], bool],
        enqueue: Callable[[Any], None],
        abort: Callable[[Any], None],
    ) -> bool:
        """Target: enqueue a received handoff, ack it, replay a transit abort."""
        if self.is_fenced(handoff):
            return False
        rid = handoff.recv_req.rid
        self.stats["handoff_in"] += 1
        if rid_present(rid):
            self.stats["double_handoff"] += 1
            logger.error("Late binding: double handoff, rid=%s already here", rid)
            self.send_ack(handoff, False, "duplicate rid on target")
            return False
        self.incoming[rid] = handoff
        enqueue(handoff.recv_req)
        if self.incoming.pop(rid, None) is not None:
            self.stats["handoff_in_not_enqueued"] += 1
            logger.error("Late binding: handoff rid=%s was not enqueued", rid)
            self.send_ack(handoff, False, "not enqueued")
            return False
        self.note_enqueued(handoff)
        self.send_ack(handoff, True)
        abort_req = self.aborted_in_transit(handoff)
        if abort_req is not None:
            self.stats["abort_in_transit"] += 1
            abort(abort_req)
        return True

    # -- aborts in transit ---------------------------------------------------
    def note_abort(self, abort_req) -> None:
        self._prune_aborts()
        self.recent_aborts.append((abort_req, time.time()))

    def _prune_aborts(self) -> None:
        horizon = time.time() - 2 * self.ack_timeout
        self.recent_aborts = [(a, t) for a, t in self.recent_aborts if t >= horizon]

    def aborted_in_transit(self, handoff: LateBindingHandoff) -> Optional[Any]:
        """An abort seen here after the handoff was sent and matching its rid
        (the source applies earlier aborts itself). Wall clocks of the source
        and target are compared; on one node they are the same clock."""
        self._prune_aborts()
        rid = handoff.recv_req.rid
        for abort_req, at in self.recent_aborts:
            if at >= handoff.sent_at and abort_req.matches(rid):
                return abort_req
        return None

    # -- logging -------------------------------------------------------------
    def maybe_log_summary(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now < self._next_summary:
            return
        self._next_summary = now + SUMMARY_INTERVAL_S
        counters = {key: self.stats.get(key, 0) for key in ALWAYS_REPORTED}
        counters.update(self.stats)
        logger.info(
            "LATEBIND_D_SUMMARY dp=%d pending_acks=%d rows=%s %s",
            self.dp_rank,
            len(self.pending_acks),
            self.rows,
            counters,
        )
