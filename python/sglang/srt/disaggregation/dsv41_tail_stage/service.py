"""The tail server's serving loop: stage-0 messages, slots, batches, sends to decode.

One thread, one ROUTER socket. Each pass reads every waiting message, expires slots,
heartbeats, polls the PP-rank-1 senders (decode's transfer info in, transfers out),
and runs a batch when the policy says so.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, deque
from typing import Callable, Dict, List, Optional

import msgspec
import requests
import zmq
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.dsv41_tail_stage import protocol as P
from sglang.srt.disaggregation.dsv41_tail_stage.batcher import TailBatchPolicy
from sglang.srt.disaggregation.dsv41_tail_stage.registration import (
    Stage0Peers,
    expected_registrations,
    stage1_registration,
)
from sglang.srt.disaggregation.dsv41_tail_stage.slots import (
    Reply,
    SlotState,
    SlotTable,
    TailSlot,
)
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Arbitrary: how often the loop logs its counters.
LOG_PERIOD_S = 5.0
# Arbitrary: how often senders without a pending send are polled, for decode-side failures.
SENDER_SWEEP_S = 0.2
# Forwards queued on the GPU at once: the next is launched while one runs, if it is full.
MAX_INFLIGHT_BATCHES = 2


class Backend(msgspec.Struct):
    """What the loop needs beyond the slot logic, so tests can run it without a GPU
    or Mooncake: the forward, the decode-side transport and the stage-0 PUT."""

    # (stage 0's wire) -> (rank_ip, rank_port, Welcome); called at the first Hello.
    start_transport: Callable[[List[list]], tuple]
    # (bootstrap_addr, room) -> a PP-rank-1 sender for the room.
    make_sender: Callable[[str, int], object]
    send: Callable[[object, TailSlot], None]
    # Launches the forward; returns None once it has run, else a check that is True
    # once the slots' outputs can be sent.
    run: Callable[[List[TailSlot]], Optional[Callable[[], bool]]]
    put: Callable[[str, dict], bool]


class TailStageService:
    def __init__(
        self,
        *,
        backend: Backend,
        socket: zmq.Socket,
        table: SlotTable,
        peers: Stage0Peers,
        policy: TailBatchPolicy,
        heartbeat_s: float,
        heartbeat_timeout_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.backend = backend
        self.socket = socket
        self.table = table
        self.peers = peers
        self.policy = policy
        self.heartbeat_s = heartbeat_s
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.clock = clock
        self.transport: Optional[tuple] = None
        self.senders: Dict[int, object] = {}
        # Rooms whose forward ran and whose send waits for decode's transfer info.
        self.ready_to_send: Dict[int, TailSlot] = {}
        # Rooms whose send to decode was issued and has not ended.
        self._sent: set = set()
        # Forwards launched and not yet finished, oldest first: (tails, done check).
        self._inflight: deque = deque()
        self._last_sweep = 0.0
        self.identities: Dict[str, bytes] = {}
        self.stats = Counter()
        self._last_heartbeat = 0.0
        self._last_log = 0.0
        self._tails_done_at_log = 0
        # Slots held (granted to sent) per loop step, and SlotDefers, since the last log.
        self._busy_samples: List[int] = []
        self._defers_at_log = 0

    def serve_forever(self) -> None:
        while True:
            self.step(poll_ms=1)

    def step(self, *, poll_ms: int) -> None:
        # While a forward runs, the loop does not sleep in the socket poll.
        self._receive(0 if self._inflight else poll_ms)
        now = self.clock()
        silent = self.peers.silent(now=now, timeout_s=self.heartbeat_timeout_s)
        self._expire(silent)
        self._heartbeat(now)
        self._finish_batches()
        self._poll_senders(now)
        self._run_batch()
        # This step's grants (from requests, expiries and finished sends) at once.
        self.table.pages.end_round()
        self._busy_samples.append(self.table.num_slots - len(self.table.free_slots))
        self._log(now)

    def _receive(self, poll_ms: int) -> None:
        while self.socket.poll(poll_ms, zmq.POLLIN):
            poll_ms = 0
            identity, data = self.socket.recv_multipart()
            stage0_id = identity.decode()
            try:
                msg = P.decode(data)
            except P.ProtocolError as exc:
                self._send(identity, P.Error(reason=str(exc)))
                continue
            self.identities[stage0_id] = identity
            self.peers.touch(stage0_id, now=self.clock())
            self._handle(stage0_id, msg)

    def _handle(self, stage0_id: str, msg) -> None:
        if isinstance(msg, P.Hello):
            self._hello(stage0_id, msg)
        elif stage0_id not in self.peers.peers:
            self._deliver([Reply(stage0_id, P.Error(reason="no Hello"))])
        elif isinstance(msg, P.SlotRequest):
            self._slot_request(stage0_id, msg)
        elif isinstance(msg, P.TailReady):
            self._deliver(self.table.ready(stage0_id, msg))
        elif isinstance(msg, P.TailAbort):
            self._abort(msg.room, stage0_id=stage0_id)
        elif not isinstance(msg, P.Heartbeat):
            logger.warning("DSV4.1 tail stage: unexpected %s from %s", msg, stage0_id)

    def _hello(self, stage0_id: str, hello: P.Hello) -> None:
        err = self.peers.check(hello)
        if err is not None:
            logger.error("DSV4.1 tail stage: refused %s: %s", stage0_id, err)
            self._deliver([Reply(stage0_id, P.Error(reason=err))])
            return
        self.peers.add(hello, now=self.clock())
        if self.transport is None:
            self.transport = self.backend.start_transport(self.peers.stage0_wire)
        rank_ip, rank_port, welcome = self.transport
        payload = stage1_registration(
            hello.registration,
            rank_ip=rank_ip,
            rank_port=rank_port,
            state_wire=self.peers.wire(),
        )
        if not self.backend.put(hello.bootstrap_addr, payload):
            reason = f"PP-rank-1 registration with {hello.bootstrap_addr} failed"
            self._deliver([Reply(stage0_id, P.Error(reason=reason))])
            return
        logger.info(
            "DSV4.1 tail stage: registered as PP rank 1 with %s (%d/%d)",
            hello.bootstrap_addr,
            len(self.peers.peers),
            expected_registrations(payload),
        )
        self._deliver([Reply(stage0_id, welcome)])

    def _slot_request(self, stage0_id: str, req: P.SlotRequest) -> None:
        replies = self.table.request(stage0_id, req)
        self.stats["defers"] += sum(isinstance(r.msg, P.SlotDefer) for r in replies)
        refused = any(
            isinstance(r.msg, P.Error) and r.msg.room == req.room for r in replies
        )
        if req.room not in self.senders and not refused:
            # Decode's transfer info for the room may already be here.
            bootstrap_addr = self.peers.peers[stage0_id].bootstrap_addr
            self.senders[req.room] = self.backend.make_sender(bootstrap_addr, req.room)
        if req.room in self.table.passthrough:
            self.ready_to_send[req.room] = TailSlot(
                room=req.room,
                stage0_id=stage0_id,
                rid=req.rid,
                s=req.s,
                e=0,
                state=SlotState.SENDING,
                t_request=self.clock(),
            )
        self._deliver(replies)

    def _abort(self, room: int, *, stage0_id: Optional[str]) -> None:
        self._deliver(self.table.abort(room, stage0_id=stage0_id))

    def _expire(self, silent: List[str]) -> None:
        replies = self.table.expire(silent_stage0s=silent)
        for reply in replies:
            msg = reply.msg
            if isinstance(msg, P.TailDone) and msg.status != "ok":
                # Stage 0 fails the request to decode; nothing is sent from here.
                self.senders.pop(msg.room, None)
                self.ready_to_send.pop(msg.room, None)
                self._sent.discard(msg.room)
                self.stats["tails_failed"] += 1
        self._deliver(replies)

    def _run_batch(self) -> None:
        if self._inflight and (
            len(self._inflight) >= MAX_INFLIGHT_BATCHES
            or self.table.num_ready() < self.policy.max_tails
        ):
            # The GPU is busy: queue the next forward behind it only when it is full.
            return
        n = self.policy.take_count(
            ready=self.table.num_ready(),
            oldest_wait_ms=self.table.oldest_ready_wait_ms(),
        )
        if n == 0:
            return
        tails = self.table.take(n)
        done = self.backend.run(tails)
        self.stats["batches"] += 1
        self.stats[f"batch_{len(tails)}"] += 1
        self._inflight.append((tails, done))
        self._finish_batches()

    def _finish_batches(self) -> None:
        while self._inflight:
            tails, done = self._inflight[0]
            if done is not None and not done():
                return
            self._inflight.popleft()
            for tail in tails:
                self.table.sending(tail.room)
                self.ready_to_send[tail.room] = tail

    def _poll_senders(self, now: float) -> None:
        """Each step, the senders with a send pending or issued; every SENDER_SWEEP_S,
        all of them, for rooms decode gave up on before their forward."""
        if now - self._last_sweep >= SENDER_SWEEP_S:
            self._last_sweep = now
            rooms = list(self.senders)
        else:
            rooms = [*self.ready_to_send, *self._sent]
        for room in rooms:
            sender = self.senders.get(room)
            if sender is None:
                continue
            status = sender.poll()
            if status == KVPoll.WaitingForInput and room in self.ready_to_send:
                self.backend.send(sender, self.ready_to_send.pop(room))
                self._sent.add(room)
            elif status in (KVPoll.Success, KVPoll.Failed):
                self._conclude(room, ok=status == KVPoll.Success)

    def _conclude(self, room: int, *, ok: bool) -> None:
        self.senders.pop(room)
        self.ready_to_send.pop(room, None)
        self._sent.discard(room)
        tail = self.table.tails.get(room)
        if tail is not None and tail.state in (SlotState.GRANTED, SlotState.FAILED):
            # Stage 0 may still write the slot: it frees at TailReady.
            self._deliver(self.table.abort(room))
            return
        status = "ok" if ok else "failed"
        kind = "tails" if tail is not None else "passthrough"
        self.stats[f"{kind}_{status}"] += 1
        self._deliver(self.table.finish(room, status=status, reason=status))

    def _heartbeat(self, now: float) -> None:
        if now - self._last_heartbeat < self.heartbeat_s:
            return
        self._last_heartbeat = now
        counts = self.table.counts()
        beat = P.Heartbeat(
            free_slots=counts["free_slots"],
            queued=counts["queued"],
            ready=counts["ready"],
        )
        self._deliver([Reply(stage0_id, beat) for stage0_id in self.peers.peers])

    def _deliver(self, replies: List[Reply]) -> None:
        for reply in replies:
            identity = self.identities.get(reply.stage0_id)
            if identity is not None:
                self._send(identity, reply.msg)

    def _send(self, identity: bytes, msg) -> None:
        self.socket.send_multipart([identity, P.encode(msg)])

    def _log(self, now: float) -> None:
        if now - self._last_log < LOG_PERIOD_S:
            return
        done = self.stats["tails_ok"]
        rate = (done - self._tails_done_at_log) / max(now - self._last_log, 1e-9)
        self._last_log, self._tails_done_at_log = now, done
        counts = self.table.counts()
        logger.info(
            "DSV4.1 tail stage: %.1f tails/s, batches %s, ready %d, granted %d, "
            "queued %d, free slots %d, done %d, failed %d",
            rate,
            {k: v for k, v in self.stats.items() if k.startswith("batch_")},
            counts["ready"],
            counts["granted"],
            counts["queued"],
            counts["free_slots"],
            done,
            self.stats["tails_failed"],
        )
        busy = sorted(self._busy_samples) or [0]
        logger.info(
            "DSV4.1 tail stage slots held p50/p90/max %d/%d/%d of %d, %d SlotDefer",
            busy[len(busy) // 2],
            busy[(len(busy) * 9) // 10],
            busy[-1],
            self.table.num_slots,
            self.stats["defers"] - self._defers_at_log,
        )
        self._busy_samples = []
        self._defers_at_log = self.stats["defers"]


def put_registration(bootstrap_addr: str, payload: dict) -> bool:
    for attempt in range(5):
        try:
            response = requests.put(
                f"http://{bootstrap_addr}/route", json=payload, timeout=5
            )
            if response.status_code == 200:
                return True
            logger.error(
                "DSV4.1 tail stage: PUT to %s answered %s: %s",
                bootstrap_addr,
                response.status_code,
                response.text[:300],
            )
            return False
        except requests.RequestException as exc:
            logger.warning("DSV4.1 tail stage: PUT to %s: %s", bootstrap_addr, exc)
            time.sleep(min(2**attempt, 10))
    return False


def bind_router(port: int) -> zmq.Socket:
    socket = zmq.Context.instance().socket(zmq.ROUTER)
    socket.setsockopt(zmq.ROUTER_MANDATORY, 0)
    # A ROUTER drops what a peer's full queue cannot take: a stage 0 stalled in a cold
    # forward must still get every grant.
    socket.setsockopt(zmq.SNDHWM, 0)
    socket.setsockopt(zmq.RCVHWM, 0)
    socket.bind(f"tcp://*:{port}")
    return socket


def build_service(scheduler, fwd) -> TailStageService:
    """The serving loop over the scheduler's forward and Mooncake."""
    from sglang.srt.disaggregation.dsv41_tail_stage.transport import (
        MooncakeTailBackend,
        NullDecodeSender,
    )

    mooncake = MooncakeTailBackend(fwd, scheduler.server_args)
    null_decode = envs.SGLANG_DSV41_TAIL_STAGE_NULL_DECODE.get()
    if null_decode:
        # No stage 0 writes the slots: give the late layers non-degenerate input.
        fwd.residual.normal_()
        logger.warning("DSV4.1 tail stage: NULL_DECODE, a rate benchmark only")
    backend = Backend(
        start_transport=mooncake.start,
        make_sender=(lambda addr, room: NullDecodeSender())
        if null_decode
        else mooncake.make_sender,
        send=mooncake.send,
        run=mooncake.run,
        put=(lambda addr, payload: True) if null_decode else put_registration,
    )
    table = SlotTable(
        num_slots=fwd.residual.shape[0],
        pages=fwd.memory,
        page_size=fwd.page_size,
        window=fwd.window,
        queue_cap=envs.SGLANG_DSV41_TAIL_STAGE_QUEUE_CAP.get(),
        ready_timeout_s=envs.SGLANG_DSV41_TAIL_STAGE_READY_TIMEOUT_S.get(),
    )
    max_tails = envs.SGLANG_DSV41_TAIL_STAGE_MAX_TAILS.get()
    policy = TailBatchPolicy(
        min_tails=envs.SGLANG_DSV41_TAIL_STAGE_MIN_TAILS.get(),
        max_tails=max_tails,
        wait_ms=envs.SGLANG_DSV41_TAIL_STAGE_WAIT_MS.get(),
        buckets=(max_tails,),
    )
    peers = Stage0Peers(
        page_size=fwd.page_size, window=fwd.window, draft_aux=fwd.aux is not None
    )
    return TailStageService(
        backend=backend,
        socket=bind_router(envs.SGLANG_DSV41_TAIL_STAGE_PORT.get()),
        table=table,
        peers=peers,
        policy=policy,
        heartbeat_s=1.0,
        heartbeat_timeout_s=envs.SGLANG_DSV41_TAIL_STAGE_HEARTBEAT_TIMEOUT_S.get(),
    )
