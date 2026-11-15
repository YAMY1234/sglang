"""The tail server's serving loop: stage-0 messages, slots, batches, sends to decode.

One thread, one ROUTER socket. Each pass reads every waiting message, expires slots,
heartbeats, polls the PP-rank-1 senders (decode's transfer info in, transfers out),
and runs a batch when the policy says so.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
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


class Backend(msgspec.Struct):
    """What the loop needs beyond the slot logic, so tests can run it without a GPU
    or Mooncake: the forward, the decode-side transport and the stage-0 PUT."""

    # (stage 0's wire) -> (rank_ip, rank_port, Welcome); called at the first Hello.
    start_transport: Callable[[List[list]], tuple]
    # (bootstrap_addr, room) -> a PP-rank-1 sender for the room.
    make_sender: Callable[[str, int], object]
    send: Callable[[object, TailSlot], None]
    run: Callable[[List[TailSlot]], None]
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
        self.identities: Dict[str, bytes] = {}
        self.stats = Counter()
        self._last_heartbeat = 0.0
        self._last_log = 0.0
        self._tails_done_at_log = 0

    def serve_forever(self) -> None:
        while True:
            self.step(poll_ms=1)

    def step(self, *, poll_ms: int) -> None:
        self._receive(poll_ms)
        now = self.clock()
        silent = self.peers.silent(now=now, timeout_s=self.heartbeat_timeout_s)
        self._deliver(self.table.expire(silent_stage0s=silent))
        self._heartbeat(now)
        self._poll_senders()
        self._run_batch()
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

    def _run_batch(self) -> None:
        n = self.policy.take_count(
            ready=self.table.num_ready(),
            oldest_wait_ms=self.table.oldest_ready_wait_ms(),
        )
        if n == 0:
            return
        tails = self.table.take(n)
        self.backend.run(tails)
        self.stats["batches"] += 1
        self.stats[f"batch_{len(tails)}"] += 1
        for tail in tails:
            self.table.sending(tail.room)
            self.ready_to_send[tail.room] = tail

    def _poll_senders(self) -> None:
        for room, sender in list(self.senders.items()):
            status = sender.poll()
            if status == KVPoll.WaitingForInput and room in self.ready_to_send:
                self.backend.send(sender, self.ready_to_send.pop(room))
            elif status in (KVPoll.Success, KVPoll.Failed):
                self._conclude(room, ok=status == KVPoll.Success)

    def _conclude(self, room: int, *, ok: bool) -> None:
        self.senders.pop(room)
        self.ready_to_send.pop(room, None)
        tail = self.table.tails.get(room)
        if tail is not None and tail.state == SlotState.GRANTED:
            # Stage 0 may still write the slot: it frees at TailReady.
            self._deliver(self.table.abort(room))
            return
        if ok:
            self.stats["tails_ok"] += 1
        else:
            self.stats["tails_failed"] += 1
        status = "ok" if ok else "failed"
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
    socket.bind(f"tcp://*:{port}")
    return socket


def build_service(scheduler, fwd) -> TailStageService:
    """The serving loop over the scheduler's forward and Mooncake."""
    from sglang.srt.disaggregation.dsv41_tail_stage.transport import (
        MooncakeTailBackend,
    )

    pages = fwd.page_source()
    mooncake = MooncakeTailBackend(fwd, pages, scheduler.server_args)
    backend = Backend(
        start_transport=mooncake.start,
        make_sender=mooncake.make_sender,
        send=mooncake.send,
        run=mooncake.run,
        put=put_registration,
    )
    table = SlotTable(
        num_slots=fwd.residual.shape[0],
        pages=pages,
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
