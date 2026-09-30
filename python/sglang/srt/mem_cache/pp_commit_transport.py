"""Previous-round PP READY reports on an independent startup-created group.

Only background threads wait for network progress. The scheduler publishes a
coalescible snapshot and reads a mailbox; these works must NEVER be appended to
UnifiedRadixCache.work_list (whose drain waits inside get_new_batch_prefill).
"""

from __future__ import annotations

import json
import threading

import torch
import torch.distributed as dist


class PreviousRoundReports:
    TAG = int.from_bytes(b"PcRy", "big")
    FRAME_BYTES = 2048

    def __init__(self, group):
        self.group = group
        self.rank = dist.get_rank(group)
        self.size = dist.get_world_size(group)
        self._condition = threading.Condition()
        self._latest = {}
        self._pending = None
        self._closing = False
        self._error = None
        self._threads = []
        self.stats = {"sent": 0, "received": 0, "coalesced": 0}
        peers = range(1, self.size) if self.rank == 0 else [0]
        for peer in peers:
            target = self._receiver if self.rank == 0 else self._sender
            thread = threading.Thread(
                target=self._run,
                args=(target, peer),
                daemon=True,
                name=f"pp-cache-ready-{self.rank}-{peer}",
            )
            self._threads.append(thread)
            thread.start()

    @classmethod
    def encode(cls, value):
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        if len(payload) > cls.FRAME_BYTES - 4:
            raise RuntimeError("PP READY bounded report frame exceeded")
        buffer = bytearray(cls.FRAME_BYTES)
        buffer[:4] = len(payload).to_bytes(4, "big")
        buffer[4 : 4 + len(payload)] = payload
        return torch.frombuffer(buffer, dtype=torch.uint8)

    @classmethod
    def decode(cls, tensor):
        raw = tensor.numpy().tobytes()
        size = int.from_bytes(raw[:4], "big")
        if not 0 < size <= cls.FRAME_BYTES - 4:
            raise RuntimeError("PP READY malformed report length")
        return json.loads(raw[4 : 4 + size])

    def _run(self, target, peer):
        try:
            target(peer)
        except Exception as error:  # noqa: BLE001 -- re-raised on the scheduler
            with self._condition:
                self._error = error
                self._condition.notify_all()

    def _receiver(self, peer):
        expected = 1
        while True:
            tensor = torch.empty(self.FRAME_BYTES, dtype=torch.uint8)
            dist.recv(tensor, group=self.group, group_src=peer, tag=self.TAG)
            message = self.decode(tensor)
            if message["wire_seq"] != expected:
                raise RuntimeError(
                    f"PP READY sequence mismatch peer={peer}: expect {expected} got {message['wire_seq']}"
                )
            expected += 1
            if message.get("closed"):
                return
            with self._condition:
                self._latest[peer] = message["report"]
                self.stats["received"] += 1

    def _sender(self, peer):
        wire_seq = 0
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._pending is not None or self._closing
                )
                report, self._pending = self._pending, None
                closed = report is None and self._closing
            wire_seq += 1
            # A single thread owns at most one in-flight buffer, plus one
            # coalesced pending snapshot. Never poll Work from the scheduler.
            tensor = self.encode(
                {"wire_seq": wire_seq, "report": report, "closed": closed}
            )
            dist.send(tensor, group=self.group, group_dst=peer, tag=self.TAG)
            with self._condition:
                self.stats["sent"] += 1
            if closed:
                return

    def _check(self):
        if self._error is not None:
            raise RuntimeError(
                "PP READY background control transport failed"
            ) from self._error

    def publish(self, report):
        if self.rank == 0:
            return
        with self._condition:
            self._check()
            if self._closing:
                raise RuntimeError("PP READY publish after shutdown")
            if self._pending is not None:
                self.stats["coalesced"] += 1
            self._pending = dict(report)
            self._condition.notify()

    def poll(self):
        with self._condition:
            self._check()
            return dict(self._latest)

    def close(self, timeout=0.1):
        """Best-effort shutdown, never wait indefinitely for a failed peer."""
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        for thread in self._threads:
            thread.join(timeout)
        return not any(thread.is_alive() for thread in self._threads)
