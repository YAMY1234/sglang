"""Decode late binding (--disaggregation-decode-late-binding).

Under the prefill_complete allocation policy a request reaches a decode DP rank
at arrival but allocates KV only after prefill reports readiness. At that
moment, before any allocation, the rank may hand the request to a less committed
DP rank of the same decode server; that rank re-binds the prefill bootstrap room
with REBIND instead of BOOTSTRAP.

Decision input: each rank's committed request count, KV usage and last handoff
target, carried as extra columns of the per-step MLP-sync gather, so every rank
decides from the same snapshot without a new collective.
Channel: ZMQ PUSH/PULL between the decode schedulers, addresses exchanged once
at startup over the TP CPU group. Received handoffs enter the scheduler with
its other inputs, through the request receiver.
"""

from __future__ import annotations

import itertools
import logging
import pickle
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch.distributed as dist
import zmq
from zmq.utils.monitor import recv_monitor_message

from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.scheduler_components.dp_attn import (
    LATE_BINDING_INACTIVE_ROW,
)
from sglang.srt.utils.network import NetworkAddress, get_zmq_socket_on_host

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeRequest

logger = logging.getLogger(__name__)

# Bounded send queue per peer: a full queue fails the send instead of blocking.
HANDOFF_SNDHWM = 64
# Bounded fences on the target, and per-peer ACK/CANCEL outbox.
MAX_FENCED_IDS = 4096
MAX_OUTBOX = 4096
# A CANCEL is retried for this many ack timeouts before it is dropped.
CANCEL_LIFETIME_FACTOR = 10
# Startup wait for the D->D sockets' handshakes. With IMMEDIATE a send fails
# until the ZMTP handshake ends, so first handoffs and ACKs would be lost.
PRECONNECT_WAIT_S = 5.0
_CONNECTED_EVENT = getattr(zmq, "EVENT_HANDSHAKE_SUCCEEDED", zmq.EVENT_CONNECTED)


def _active_ranks(rows: Sequence[Sequence[int]]) -> List[int]:
    return [r for r in range(len(rows)) if rows[r][0] < LATE_BINDING_INACTIVE_ROW[0]]


def _pick_target(
    score: Dict[int, float], self_rank: int, threshold: int
) -> Optional[int]:
    """Sources (score >= min + threshold, highest first) take the lowest
    scored ranks in order, so concurrent sources never pick the same target
    from one snapshot; a pair is used only if it still differs by threshold."""
    floor = min(score.values())
    sources = sorted(
        (r for r in score if score[r] >= floor + threshold),
        key=lambda r: (-score[r], r),
    )
    if self_rank not in sources:
        return None
    targets = sorted(score, key=lambda r: (score[r], r))
    target = targets[sources.index(self_rank)]
    if target == self_rank or score[self_rank] < score[target] + threshold:
        return None
    return target


def _committed(rows: Sequence[Sequence[int]]) -> List[int]:
    """Committed counts, with a handoff announced in the snapshot counted +1
    on its target."""
    committed = [int(row[0]) for row in rows]
    for row in rows:
        if 0 <= int(row[2]) < len(rows):
            committed[int(row[2])] += 1
    return committed


def choose_handoff_target(
    rows: Sequence[Sequence[int]], self_rank: int, threshold: int
) -> Optional[int]:
    """The DP rank this rank should hand one ready request to, or None.

    rows[r] = (committed, kv_usage_permille, handoff_target), the same gathered
    snapshot on every rank. This rank is a source if its committed count is at
    least the minimum plus ``threshold``.
    """
    if threshold < 1 or not (0 <= self_rank < len(rows)):
        return None
    active = _active_ranks(rows)
    if self_rank not in active or len(active) < 2:
        return None
    committed = _committed(rows)
    return _pick_target({r: committed[r] for r in active}, self_rank, threshold)


def choose_handoff_target_dominant(
    rows: Sequence[Sequence[int]],
    self_rank: int,
    threshold: int,
    cap: int,
    req_kv_frac: float = 0.0,
) -> Optional[int]:
    """choose_handoff_target with each rank scored by its binding resource in
    request units, max(committed, kv_usage * cap), where cap is the running
    request limit; with KV usage low this is the count rule. The move must also
    leave the target, after taking this request (+1 committed, +req_kv_frac of
    the pool), below the source."""
    if threshold < 1 or cap < 1 or not (0 <= self_rank < len(rows)):
        return None
    active = _active_ranks(rows)
    if self_rank not in active or len(active) < 2:
        return None
    committed = _committed(rows)
    kv = [int(row[1]) / 1000.0 for row in rows]
    score = {r: max(committed[r], kv[r] * cap) for r in active}
    target = _pick_target(score, self_rank, threshold)
    if target is None:
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
    source_dp_rank: int


class DecodeLateBinding:
    """Late-binding state of one decode DP-rank scheduler.

    The source owns a handed-off request until the target's ack arrives. If no
    ack arrives within ack_timeout, the source fails the request (client error,
    ABORT to prefill from the old endpoint) and sends CANCEL. The target drops a
    handoff whose id was cancelled; a CANCEL for an id it already enqueued
    aborts its copy. Either way the request has one terminal owner.
    """

    def __init__(
        self,
        *,
        dp_rank: int,
        num_dp_ranks: int,
        threshold: int,
        local_ip: str,
        ack_timeout: float,
        get_limits: Optional[Callable[[], Tuple[int, int]]] = None,
        context: Optional[zmq.Context] = None,
    ):
        self.dp_rank = dp_rank
        self.num_dp_ranks = num_dp_ranks
        self.threshold = threshold
        self.ack_timeout = ack_timeout
        # () -> (running request cap, KV pool tokens); set for the dominant rule.
        self.get_limits = get_limits
        self.rows: Optional[List[Tuple[int, int, int]]] = None
        self._rows_version = 0
        self._decided_version = -1
        # Target of the handoff made since the last gather (-1 none).
        self.sent_to = -1
        self._ids = itertools.count()
        # handoff_id -> (handoff, monotonic send time, owner) awaiting an ack.
        self.pending_acks: Dict[str, Tuple[LateBindingHandoff, float, Any]] = {}
        # Per-peer FIFO of (deadline, kind, handoff_id, bytes): ACKs and CANCELs
        # not yet accepted by the socket, retried without blocking every poll.
        self.outbox: Dict[int, deque] = {}
        # Received handoffs, and aborts of cancelled copies, not yet carried by
        # a receive pass.
        self.arrivals: List[Union[LateBindingHandoff, AbortReq]] = []
        # rid -> handoff, consumed when the target creates the receiver.
        self.incoming: Dict[str, LateBindingHandoff] = {}
        # Target fences: cancelled ids, and ids enqueued here (-> rid).
        self.cancelled: OrderedDict[str, float] = OrderedDict()
        self.enqueued: OrderedDict[str, str] = OrderedDict()
        # (AbortReq, time.time()) recently seen, for requests aborted in transit.
        self.recent_aborts: List[Tuple[Any, float]] = []
        # Source: fail a timed-out handoff (handoff, owner).
        self.on_timeout: Optional[Callable[[LateBindingHandoff, Any], None]] = None
        self.peers: List[str] = []
        self._push: Dict[int, zmq.Socket] = {}
        # Peers whose PUSH socket finished the handshake (socket monitors).
        self.connected: set = set()
        self._monitors: Dict[int, zmq.Socket] = {}
        self._ctx = context or zmq.Context()
        port, self.pull = get_zmq_socket_on_host(
            self._ctx, zmq.PULL, host=local_ip, restrict_peers=True
        )
        address = NetworkAddress(local_ip, port)
        self.address = address.to_tcp()
        self._ipv6 = address.is_ipv6

    # -- startup -------------------------------------------------------------
    def connect_peers(self, group, wait_s: float = PRECONNECT_WAIT_S) -> None:
        """Collective over the TP CPU group (one rank per DP rank): exchange the
        PULL addresses, connect a PUSH socket to every other rank and wait,
        bounded, for the handshakes."""
        gathered: List[Optional[str]] = [None] * dist.get_world_size(group)
        dist.all_gather_object(gathered, self.address, group=group)
        if len(gathered) != self.num_dp_ranks or None in gathered:
            raise RuntimeError(f"Late binding: bad handoff addresses {gathered}")
        self.peers = gathered
        others = [r for r in range(self.num_dp_ranks) if r != self.dp_rank]
        for dp_rank in others:
            self._socket_to(dp_rank)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            self._poll_monitors()
            if self.connected.issuperset(others):
                return
            time.sleep(0.01)
        missing = sorted(set(others) - self.connected)
        logger.warning(
            f"Late binding: DP{self.dp_rank} not yet connected to DP ranks {missing}"
        )

    # -- MLP-sync columns ----------------------------------------------------
    def local_row(self, committed: int, kv_usage: float) -> Tuple[int, int, int]:
        return (int(committed), int(round(kv_usage * 1000)), int(self.sent_to))

    def on_rows(self, rows: List[List[int]]) -> None:
        self.rows = [tuple(int(v) for v in row) for row in rows]
        self._rows_version += 1
        self.sent_to = -1

    # -- decision ------------------------------------------------------------
    def decide(self, decode_req: DecodeRequest) -> Optional[int]:
        """Called when a queued request first becomes ready, before allocation."""
        if decode_req.late_binding_hops > 0:
            return None  # one handoff per request
        if self.rows is None or self._decided_version == self._rows_version:
            return None  # at most one handoff per gathered snapshot
        if len(self.rows) != self.num_dp_ranks:
            return None
        cap, pool = self.get_limits() if self.get_limits is not None else (0, 0)
        if cap > 0 and pool > 0:
            frac = len(decode_req.req.origin_input_ids) / pool
            target = choose_handoff_target_dominant(
                self.rows, self.dp_rank, self.threshold, cap, frac
            )
        else:
            target = choose_handoff_target(self.rows, self.dp_rank, self.threshold)
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
            if self._ipv6:
                sock.setsockopt(zmq.IPV6, 1)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.SNDHWM, HANDOFF_SNDHWM)
            # Queue only to a connected peer: no peer fails the send at once.
            sock.setsockopt(zmq.IMMEDIATE, 1)
            self._monitors[dp_rank] = sock.get_monitor_socket(
                _CONNECTED_EVENT | zmq.EVENT_DISCONNECTED
            )
            sock.connect(self.peers[dp_rank])
            self._push[dp_rank] = sock
        return sock

    def _poll_monitors(self) -> None:
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

    def _send_bytes(self, dp_rank: int, data: bytes) -> bool:
        """Never blocks: a full queue, an unconnected peer or a socket error
        returns False."""
        try:
            self._socket_to(dp_rank).send(data, zmq.NOBLOCK)
            return True
        except zmq.ZMQError:
            return False

    def send_handoff(self, handoff: LateBindingHandoff, owner: Any) -> bool:
        """True only if the payload is queued; the caller detaches only then.
        Not retried: on failure, or to a peer whose handshake has not finished,
        the request stays on this rank."""
        self._poll_monitors()
        if handoff.target_dp_rank not in self.connected:
            return False
        data = pickle.dumps(handoff, protocol=pickle.HIGHEST_PROTOCOL)
        if not self._send_bytes(handoff.target_dp_rank, data):
            return False
        self.sent_to = handoff.target_dp_rank
        self.pending_acks[handoff.handoff_id] = (handoff, time.monotonic(), owner)
        return True

    def _enqueue_out(
        self, dp_rank: int, kind: str, handoff_id: str, obj, lifetime: float
    ) -> None:
        box = self.outbox.setdefault(dp_rank, deque())
        if len(box) >= MAX_OUTBOX:
            _, old_kind, old_id, _ = box.popleft()
            logger.error(
                f"Late binding: outbox to DP{dp_rank} full, dropped {old_kind} {old_id}"
            )
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        box.append((time.monotonic() + lifetime, kind, handoff_id, data))
        self.flush_outbox()

    def flush_outbox(self) -> None:
        """Send queued ACKs/CANCELs in order without blocking; drop expired
        ones loudly."""
        now = time.monotonic()
        for dp_rank, box in self.outbox.items():
            while box:
                deadline, kind, handoff_id, data = box[0]
                if now > deadline:
                    box.popleft()
                    logger.error(
                        f"Late binding: {kind} for handoff {handoff_id} to "
                        f"DP{dp_rank} never delivered"
                    )
                    continue
                if not self._send_bytes(dp_rank, data):
                    break
                box.popleft()

    def _drop_queued_ack(self, handoff_id: str, dp_rank: int) -> None:
        box = self.outbox.get(dp_rank)
        if box:
            self.outbox[dp_rank] = deque(
                e for e in box if not (e[1] == "ack" and e[2] == handoff_id)
            )

    def send_ack(self, handoff: LateBindingHandoff, ok: bool, reason: str = "") -> None:
        """Retried until delivered or the source's ack timeout passed, after
        which the source has decided and ignores it."""
        ack = LateBindingAck(
            handoff.recv_req.rid, handoff.room, handoff.handoff_id, ok, reason
        )
        self._enqueue_out(
            handoff.source_dp_rank, "ack", handoff.handoff_id, ack, self.ack_timeout
        )

    def poll(self) -> None:
        """Drain the PULL socket into ``arrivals``; consume ACKs and CANCELs;
        retry the outbox; enforce ack timeouts. Never blocks; called on every
        receive pass, skipped or not."""
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
            elif not self.is_fenced(msg):
                self.arrivals.append(msg)
        self._poll_monitors()
        self.flush_outbox()
        self.check_timeouts()

    def take_arrivals(self) -> List[Union[LateBindingHandoff, AbortReq]]:
        arrivals, self.arrivals = self.arrivals, []
        return arrivals

    def _on_ack(self, ack: LateBindingAck) -> None:
        if self.pending_acks.pop(ack.handoff_id, None) is None:
            # After the timeout: the CANCEL already fences or aborts it.
            logger.error(
                f"Late binding: ack for expired handoff {ack.handoff_id} rid={ack.rid}"
            )
        elif not ack.ok:
            logger.error(
                f"Late binding: target refused rid={ack.rid} room={ack.room}: "
                f"{ack.reason}"
            )

    def check_timeouts(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        for handoff_id, (handoff, sent, owner) in list(self.pending_acks.items()):
            if now - sent <= self.ack_timeout:
                continue
            del self.pending_acks[handoff_id]
            logger.warning(
                f"Late binding: handoff {handoff_id} rid={handoff.recv_req.rid} "
                f"to DP{handoff.target_dp_rank} not acknowledged in "
                f"{self.ack_timeout}s, failing the request"
            )
            # Final for this handoff: a later ACK is ignored, and the CANCEL
            # fences or aborts the target's copy.
            self._enqueue_out(
                handoff.target_dp_rank,
                "cancel",
                handoff_id,
                LateBindingCancel(
                    handoff.recv_req.rid,
                    handoff.room,
                    handoff_id,
                    handoff.source_dp_rank,
                ),
                CANCEL_LIFETIME_FACTOR * self.ack_timeout,
            )
            if self.on_timeout is not None:
                self.on_timeout(handoff, owner)

    # -- target fences -------------------------------------------------------
    def _remember(self, table: OrderedDict, key: str, value) -> None:
        table[key] = value
        while len(table) > MAX_FENCED_IDS:
            table.popitem(last=False)

    def is_fenced(self, handoff: LateBindingHandoff) -> bool:
        if handoff.handoff_id in self.cancelled:
            logger.error(
                f"Late binding: dropped handoff {handoff.handoff_id} "
                f"rid={handoff.recv_req.rid} that arrived after its CANCEL"
            )
            return True
        return False

    def on_cancel(self, cancel: LateBindingCancel) -> None:
        rid = self.enqueued.pop(cancel.handoff_id, None)
        if rid is None:
            # Not enqueued (yet): fence it; buffered arrivals are dropped too.
            self._remember(self.cancelled, cancel.handoff_id, time.monotonic())
            self.arrivals = [
                a
                for a in self.arrivals
                if not (isinstance(a, LateBindingHandoff) and self.is_fenced(a))
            ]
            return
        # The ack was lost or late and the source already failed the request,
        # so the copy here must stop too; an ACK still queued here is moot.
        self._drop_queued_ack(cancel.handoff_id, cancel.source_dp_rank)
        logger.error(
            f"Late binding: CANCEL after enqueue, aborting rid={rid} "
            f"(handoff {cancel.handoff_id})"
        )
        self.arrivals.append(
            AbortReq(rid=rid, abort_message="Late binding handoff cancelled")
        )

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
        if rid_present(rid):
            logger.error(f"Late binding: handoff of rid={rid} already here")
            self.send_ack(handoff, False, "duplicate rid on target")
            return False
        self.incoming[rid] = handoff
        enqueue(handoff.recv_req)
        if self.incoming.pop(rid, None) is not None:
            logger.error(f"Late binding: handoff rid={rid} was not enqueued")
            self.send_ack(handoff, False, "not enqueued")
            return False
        self._remember(self.enqueued, handoff.handoff_id, rid)
        self.send_ack(handoff, True)
        abort_req = self.aborted_in_transit(handoff)
        if abort_req is not None:
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
        """An abort seen here after the handoff was sent that matches its rid
        (the source applies earlier aborts itself). This compares the wall
        clocks of the source and the target."""
        self._prune_aborts()
        rid = handoff.recv_req.rid
        for abort_req, at in self.recent_aborts:
            if at >= handoff.sent_at and abort_req.matches(rid):
                return abort_req
        return None
