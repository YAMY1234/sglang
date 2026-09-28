"""Experimental session-sticky admission before Dynamo's KV router.

Only a new session is assigned. No KV state or existing request is migrated.
This module deliberately has no SGLang, CUDA or HTTP dependencies.
"""

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Target:
    worker_id: int
    dp_rank: int


@dataclass
class Admission:
    target: Target | None
    inject: bool
    reason: str
    released: bool = False


class LoadBalancer:
    def __init__(self, workers=2, ranks=4, max_age=2.0):
        self.workers = workers
        self.ranks = ranks
        self.max_age = max_age
        self.mode = "stock"
        self.epoch = "initial"
        self.inflight = 0
        self.local = Counter()
        self.sessions = {}  # None means Dynamo owns this session; never take over.
        self.loads = {}
        self.external = {}
        self.received = {}
        self.counts = Counter()
        self.tie_cursor = 0

    def update_loads(self, worker_id, loads, now):
        fresh = {}
        for load in loads:
            rank = int(load["dp_rank"])
            running = int(load["num_running_reqs"])
            waiting = int(load["num_waiting_reqs"])
            timestamp = float(load["timestamp"])
            if rank not in range(self.ranks) or min(running, waiting) < 0:
                raise ValueError("Invalid rank or request count")
            target = Target(int(worker_id), rank)
            if target in fresh:
                raise ValueError("Duplicate DP rank")
            fresh[target] = (running + waiting, timestamp, running, waiting)
        if len(fresh) != self.ranks:
            raise ValueError("Incomplete DP snapshot")
        # Worker restarts require an explicit new experiment: do not silently
        # reuse session pins against a new connection ID.
        if worker_id not in self.received and len(self.received) >= self.workers:
            raise ValueError("Worker identity changed")
        self.loads.update(fresh)
        for target, values in fresh.items():
            # Reconcile the engine gauge against requests already accounted
            # locally. Arrivals after this snapshot increase the score at once.
            self.external[target] = max(0, values[0] - self.local[target])
        self.received[worker_id] = now

    def fresh(self, now):
        if len(self.received) != self.workers:
            return False
        if any(now - t > self.max_age or t > now + 1 for t in self.received.values()):
            return False
        for count, timestamp, _, _ in self.loads.values():
            # The scheduler sleeps after publishing an idle zero snapshot.
            # A fresh HTTP reply verifies the engine process is still alive;
            # local reservations cover arrivals since that idle publication.
            if timestamp > now + 1 or (count and now - timestamp > self.max_age):
                return False
        return True

    def set_policy(self, mode, epoch):
        if mode not in ("stock", "balanced") or not epoch:
            raise ValueError("Expected stock/balanced and a nonempty epoch")
        if self.inflight:
            raise RuntimeError("Cannot switch policy with requests in flight")
        self.mode, self.epoch = mode, epoch
        self.counts.clear()

    def choose(self, session, parent, constrained, now):
        target, inject = None, False
        if constrained:
            reason = "caller_constraint"
            if session is not None:
                self.sessions.setdefault(session, None)
        elif self.mode == "stock":
            reason = "stock"
            if session is not None:
                target = self.sessions.setdefault(session, None)
        elif session is not None and session in self.sessions:
            target = self.sessions[session]
            inject = target is not None
            reason = "sticky" if inject else "dynamo_owned"
        elif parent is not None:
            target = self.sessions.get(parent)
            inject = target is not None
            reason = "parent_sticky" if inject else "unknown_parent"
            if session is not None:
                self.sessions[session] = target
        elif not self.fresh(now):
            reason = "stale_loads"
            if session is not None:
                self.sessions[session] = None
        else:
            targets = sorted(self.loads)
            rotated = targets[self.tie_cursor :] + targets[: self.tie_cursor]
            target = min(rotated, key=lambda t: self.external[t] + self.local[t])
            self.tie_cursor = (targets.index(target) + 1) % len(targets)
            inject, reason = True, "new_balanced"
            if session is not None:
                self.sessions[session] = target
        self.inflight += 1
        if target is not None:
            self.local[target] += 1
        self.counts[reason] += 1
        return Admission(target, inject, reason)

    def release(self, admission):
        if admission.released:
            raise RuntimeError("Admission released twice")
        admission.released = True
        self.inflight -= 1
        if admission.target is not None:
            self.local[admission.target] -= 1
        assert self.inflight >= 0 and all(n >= 0 for n in self.local.values())

    def status(self, now):
        return {
            "mode": self.mode,
            "epoch": self.epoch,
            "inflight": self.inflight,
            "sessions": len(self.sessions),
            "fresh": self.fresh(now),
            "counts": dict(self.counts),
            "ranks": [
                {
                    "worker_id": t.worker_id,
                    "dp_rank": t.dp_rank,
                    "engine_requests": values[0],
                    "timestamp": values[1],
                    "running": values[2],
                    "waiting": values[3],
                    "local_requests": self.local[t],
                    "score": self.external[t] + self.local[t],
                }
                for t, values in sorted(self.loads.items())
            ],
        }
