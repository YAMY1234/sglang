"""Previous-round PP READY reports on an independent startup-created group.

Only background threads wait for network progress. The scheduler publishes a
coalescible snapshot and reads a mailbox; these works must NEVER be appended to
UnifiedRadixCache.work_list (whose drain waits inside get_new_batch_prefill).
"""

from __future__ import annotations

import json
import struct
import threading
from collections import defaultdict, deque
from itertools import islice

import torch
import torch.distributed as dist


class PreviousRoundReports:
    TAG = int.from_bytes(b"PcRy", "big")
    # Metadata may coalesce. Proposal payloads live in separate bounded FIFOs,
    # never in a replaceable snapshot. Only the background workers may wait.
    FRAME_BYTES = 32768
    PAYLOAD_LIMIT = 1536
    BATCH_LIMIT = 64

    def __init__(self, group):
        self.group = group
        self.rank = dist.get_rank(group)
        self.size = dist.get_world_size(group)
        self._condition = threading.Condition()
        self._latest = {}
        self._pending = None
        self._outgoing = deque()
        self._incoming = defaultdict(deque)
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
        value = dict(value)
        blob = bytearray()
        if value.get("report") is not None and "belief_proposals" in value["report"]:
            report = value["report"] = dict(value["report"])
            compact = []
            for item in report["belief_proposals"]:
                compact.append(
                    [
                        item[k]
                        for k in (
                            "epoch",
                            "origin",
                            "serial",
                            "action",
                            "pool",
                            "origin_site",
                        )
                    ]
                    + [len(item["hashes"])]
                )
                for key in item["hashes"]:
                    is_hash = len(key) == 64 and all(
                        c in "0123456789abcdef" for c in key
                    )
                    raw = bytes.fromhex(key) if is_hash else key.encode()
                    blob += struct.pack("!BH", int(is_hash), len(raw)) + raw
            report["belief_proposals"] = compact
            value["proposal_codec"] = 1
        meta = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        payload = len(meta).to_bytes(4, "big") + meta + blob
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
        meta_size = int.from_bytes(raw[4:8], "big")
        if not 4 < size <= cls.FRAME_BYTES - 4 or not 0 < meta_size <= size - 4:
            raise RuntimeError("PP READY malformed report length")
        value = json.loads(raw[8 : 8 + meta_size])
        offset, end = 8 + meta_size, size + 4
        if value.pop("proposal_codec", None) == 1:
            items = []
            for fields in value["report"]["belief_proposals"]:
                item = dict(
                    zip(
                        ("epoch", "origin", "serial", "action", "pool", "origin_site"),
                        fields[:6],
                    )
                )
                hashes = []
                if len(fields) != 7 or not 0 < fields[6] <= 8:
                    raise RuntimeError("PP READY malformed proposal")
                for _ in range(fields[6]):
                    if offset + 3 > end:
                        raise RuntimeError("PP READY malformed proposal hash")
                    is_hash, n = struct.unpack("!BH", raw[offset : offset + 3])
                    offset += 3
                    if (
                        offset + n > end
                        or is_hash not in (0, 1)
                        or (is_hash and n != 32)
                    ):
                        raise RuntimeError("PP READY malformed proposal hash")
                    key = raw[offset : offset + n]
                    offset += n
                    hashes.append(key.hex() if is_hash else key.decode())
                item["hashes"] = hashes
                items.append(item)
            value["report"]["belief_proposals"] = items
        if offset != end:
            raise RuntimeError("PP READY malformed trailing payload")
        return value

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
                report = dict(message["report"])
                items = report.pop("belief_proposals", [])
                if len(items) > self.BATCH_LIMIT:
                    raise RuntimeError("PP READY proposal batch bound exceeded")
                # Backpressure is on this background Gloo reader, never on the
                # scheduler. A consumed prefix wakes it; no payload is replaced.
                count = len(items)
                self._condition.wait_for(
                    lambda count=count: (
                        len(self._incoming[peer]) + count <= self.PAYLOAD_LIMIT
                        or self._closing
                    )
                )
                if self._closing:
                    return
                self._incoming[peer].extend(items)
                self._latest[peer] = report
                self.stats["received"] += 1

    def _sender(self, peer):
        wire_seq = 0
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._pending is not None or self._outgoing or self._closing
                )
                report, self._pending = self._pending, None
                items = list(islice(self._outgoing, self.BATCH_LIMIT))
                for _ in items:
                    self._outgoing.popleft()
                closed = report is None and not items and self._closing
                if items:
                    report = dict(report or self._last_published)
                    report["belief_proposals"] = items
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
            return True
        with self._condition:
            self._check()
            if self._closing:
                raise RuntimeError("PP READY publish after shutdown")
            metadata = dict(report)
            items = metadata.pop("belief_proposals", [])
            if len(items) > self.BATCH_LIMIT:
                raise RuntimeError("PP READY proposal batch bound exceeded")
            accepted = len(self._outgoing) + len(items) <= self.PAYLOAD_LIMIT
            if accepted:
                self._outgoing.extend(items)
            else:
                self.stats["backpressure"] = self.stats.get("backpressure", 0) + 1
            if self._pending is not None:
                self.stats["coalesced"] += 1
            self._pending = self._last_published = metadata
            self._condition.notify()
            return accepted

    def poll(self):
        with self._condition:
            self._check()
            return {
                peer: dict(
                    report,
                    belief_proposals=list(
                        islice(self._incoming[peer], self.BATCH_LIMIT)
                    ),
                )
                for peer, report in self._latest.items()
            }

    def acknowledge(self, peer, epoch, serial):
        """Remove only the prefix accepted by PP0, not an overwritten READY."""
        with self._condition:
            queue = self._incoming[peer]
            while queue and (
                queue[0]["epoch"] < epoch
                or (queue[0]["epoch"] == epoch and queue[0]["serial"] <= serial)
            ):
                queue.popleft()
            self._condition.notify_all()

    def close(self, timeout=0.1):
        """Best-effort shutdown, never wait indefinitely for a failed peer."""
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        for thread in self._threads:
            thread.join(timeout)
        return not any(thread.is_alive() for thread in self._threads)
