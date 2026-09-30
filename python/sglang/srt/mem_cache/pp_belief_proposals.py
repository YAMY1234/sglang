"""Bounded, per-origin belief proposals; only PP0 assigns common operation IDs."""

import time
from collections import OrderedDict, defaultdict


class BeliefProposals:
    CHUNK_KEYS = 8
    BATCH_LIMIT = 64
    WINDOW_LIMIT = 1536

    def __init__(self, rank, epoch=0, limit=8192, timeout=120.0):
        self.rank, self.epoch = rank, epoch
        self.limit, self.timeout = limit, timeout
        self.serial = 0
        self.pending = OrderedDict()
        self.by_effect = {}
        self.last_by_key = {}
        self.positive = defaultdict(int)
        self.assigned = set()
        self.peak = 0
        self.completed = 0
        self.sent = 0
        self.age_peak = 0.0
        self.interval_age_peak = 0.0
        self.last_committed = 0
        self.origin_sites = {}

    def propose(self, action, pool, hashes, belief, origin_site):
        pool = str(pool)
        # Stable compact diagnostic code on both wire directions. Keep the full
        # stack label locally; it is not part of an operation's causal identity.
        site = (
            4
            if "_invalidate_absent_from_hit_query" in origin_site
            else 1
            if action == "add"
            else 2
        )
        self.origin_sites[site] = origin_site
        hashes = tuple(dict.fromkeys(hashes))
        if action == "delete":
            hashes = tuple(
                sorted(
                    h
                    for h in hashes
                    if belief.peek_present(pool, h) or self.positive.get((pool, h), 0)
                )
            )
        elif action != "add":
            raise RuntimeError(
                "Runtime belief clear requires an idle coordinated reset"
            )
        for start in range(0, len(hashes), self.CHUNK_KEYS):
            chunk = hashes[start : start + self.CHUNK_KEYS]
            effect = action, pool, chunk
            old = self.by_effect.get(effect)
            if old is not None and all(
                self.last_by_key.get((pool, h)) == old for h in chunk
            ):
                continue
            if len(self.pending) >= self.limit:
                raise RuntimeError("PP commit belief proposal bound exceeded")
            self.serial += 1
            item = {
                "epoch": self.epoch,
                "origin": self.rank,
                "serial": self.serial,
                "action": action,
                "pool": pool,
                "hashes": list(chunk),
                "origin_site": site,
            }
            self.pending[self.serial] = (item, time.monotonic())
            self.by_effect[effect] = self.serial
            for h in chunk:
                self.last_by_key[pool, h] = self.serial
                if action == "add":
                    self.positive[pool, h] += 1
            self.peak = max(self.peak, len(self.pending))

    def head(self):
        return next(iter(self.pending.values()))[0] if self.pending else None

    def batch(self):
        """Peek only the unsent prefix; credit is returned at common commit."""
        unsent = self.serial - self.sent
        quota = 16 if unsent <= 16 else 32 if unsent <= 32 else self.BATCH_LIMIT
        count = min(quota, self.WINDOW_LIMIT - (self.sent - self.last_committed))
        return [
            self.pending[s][0]
            for s in range(self.sent + 1, min(self.serial, self.sent + count) + 1)
        ]

    def mark_sent(self, items):
        for item in items:
            serial = item["serial"]
            if serial != self.sent + 1 or self.pending[serial][0] != item:
                raise RuntimeError(
                    "PP commit outgoing proposal sequence/payload mismatch"
                )
            self.sent = serial
        if self.sent - self.last_committed > self.WINDOW_LIMIT:
            raise RuntimeError("PP commit outgoing proposal credit bound exceeded")

    def mark_assigned(self, item):
        if item["origin"] != self.rank:
            return
        local = self.pending.get(item["serial"])
        if local is None or local[0] != item:
            raise RuntimeError("PP commit proposal identity/payload mismatch")
        if self.rank == 0 and item["serial"] > self.sent:
            self.mark_sent([item])
        self.assigned.add(item["serial"])

    def complete(self, item):
        if item["origin"] != self.rank:
            return
        serial = item["serial"]
        if serial != self.last_committed + 1:
            raise RuntimeError("PP commit belief completion sequence gap")
        local, born = self.pending.pop(serial)
        self.age_peak = max(self.age_peak, time.monotonic() - born)
        self.interval_age_peak = max(self.interval_age_peak, time.monotonic() - born)
        self.last_committed = serial
        if local != item or serial not in self.assigned:
            raise RuntimeError("PP commit unassigned proposal completion")
        action, pool, hashes = item["action"], item["pool"], tuple(item["hashes"])
        effect = action, pool, hashes
        if self.by_effect.get(effect) == serial:
            del self.by_effect[effect]
        for h in hashes:
            if self.last_by_key.get((pool, h)) == serial:
                del self.last_by_key[pool, h]
            if action == "add":
                self.positive[pool, h] -= 1
                if not self.positive[pool, h]:
                    del self.positive[pool, h]
        self.assigned.remove(serial)
        self.completed += 1

    def check_age(self):
        if self.pending:
            _item, born = next(iter(self.pending.values()))
            self.age_peak = max(self.age_peak, time.monotonic() - born)
            self.interval_age_peak = max(
                self.interval_age_peak, time.monotonic() - born
            )
            if time.monotonic() - born >= self.timeout:
                raise RuntimeError(
                    f"PP commit unassigned/unfinished belief proposal: {self.snapshot()}"
                )

    def snapshot(self):
        return {
            "pending": len(self.pending),
            "arrived": self.serial,
            "sent": self.sent,
            "inflight": self.sent - self.last_committed,
            "window_limit": self.WINDOW_LIMIT,
            "oldest_age_s": time.monotonic() - next(iter(self.pending.values()))[1]
            if self.pending
            else 0.0,
            "peak_age_s": self.age_peak,
            "interval_peak_age_s": self.interval_age_peak,
            "origin_sites": dict(self.origin_sites),
            "unassigned": len(self.pending) - len(self.assigned),
            "peak": self.peak,
            "completed": self.completed,
            "batch_limit": self.BATCH_LIMIT,
            "batch_overflow": max(0, len(self.pending) - self.BATCH_LIMIT),
            "peak_batch_overflow": max(0, self.peak - self.BATCH_LIMIT),
            "head": [item for item, _ in list(self.pending.values())[:4]],
        }
