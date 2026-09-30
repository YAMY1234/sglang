"""Bounded, per-origin belief proposals; only PP0 assigns common operation IDs."""

import time
from collections import OrderedDict, defaultdict
from itertools import islice


class BeliefProposals:
    CHUNK_KEYS = 8
    BATCH_LIMIT = 16

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

    def propose(self, action, pool, hashes, belief, origin_site):
        pool = str(pool)
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
                "origin_site": origin_site,
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
        # Retransmit the oldest uncommitted prefix, including assigned entries.
        # Coalescing READY snapshots cannot lose a proposal or reorder a key's
        # add/delete sequence. Removal still happens only at common commit.
        return [item for item, _ in islice(self.pending.values(), self.BATCH_LIMIT)]

    def mark_assigned(self, item):
        if item["origin"] != self.rank:
            return
        local = self.pending.get(item["serial"])
        if local is None or local[0] != item:
            raise RuntimeError("PP commit proposal identity/payload mismatch")
        self.assigned.add(item["serial"])

    def complete(self, item):
        if item["origin"] != self.rank:
            return
        serial = item["serial"]
        local, _ = self.pending.pop(serial)
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
            if time.monotonic() - born >= self.timeout:
                raise RuntimeError(
                    f"PP commit unassigned/unfinished belief proposal: {self.snapshot()}"
                )

    def snapshot(self):
        return {
            "pending": len(self.pending),
            "unassigned": len(self.pending) - len(self.assigned),
            "peak": self.peak,
            "completed": self.completed,
            "batch_limit": self.BATCH_LIMIT,
            "batch_overflow": max(0, len(self.pending) - self.BATCH_LIMIT),
            "peak_batch_overflow": max(0, self.peak - self.BATCH_LIMIT),
            "head": [item for item, _ in list(self.pending.values())[:4]],
        }
