"""Opt-in phase-one integration: ACK-owned resources and belief mutations.

Prefetch result/tree insertion and lookup LRU touches deliberately stay outside
this protocol. Their local effects are measured, not silently called consensus.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.distributed as dist
from sglang.srt.distributed.communication_tags import P2PTag
from sglang.srt.mem_cache.pp_belief_proposals import BeliefProposals
from sglang.srt.mem_cache.pp_commit import (
    CommitBoundary,
    OperationId,
    canonical,
    digest,
)
from sglang.srt.mem_cache.pp_commit_transport import PreviousRoundReports

logger = logging.getLogger(__name__)


def key_summary(hashes):
    hashes = tuple(hashes or ())
    return [len(hashes), hashes[:1], hashes[-1:], digest(hashes)]


def backup_key(operation):
    return [["kv", key_summary(operation.hash_value)]] + [
        [str(transfer.name), key_summary(transfer.keys)]
        for transfer in operation.pool_transfers or ()
    ]


@dataclass
class ReleaseAck:
    identity: OperationId
    indices: torch.Tensor
    pool: str
    part: int
    parts: int


class PPCommitBridge:
    def __init__(self, cache, control_group):
        self.cache = cache
        self.state = CommitBoundary(
            cache.pp_rank, cache.pp_size, max_pinned_bytes=512 << 30
        )
        self.reports = PreviousRoundReports(control_group)
        self.generations = defaultdict(int)
        self._id_lock = threading.RLock()
        self.applying = False
        self.previous_confirmed = 0
        self.previous_admit = True
        self._belief_snapshots = {}
        self._belief_mismatch_seen = set()
        self.belief_mismatches = 0
        self.admission_open = True
        self.issued_backups = {}
        self.pending_backup_keys = defaultdict(int)
        self.issued_releases = {}
        self.belief_proposals = BeliefProposals(cache.pp_rank)
        self.leader_proposals_seen = defaultdict(int)
        self.belief_effects = {}
        self.committed_by_kind = defaultdict(int)
        self.releases_by_origin = defaultdict(int)

    def identity(self, kind, key):
        key = canonical(key).decode()
        with self._id_lock:
            generation = self.generations[kind, key]
            self.generations[kind, key] += 1
        return OperationId(self.state.epoch, kind, key, generation)

    def pinned_bytes(self, pool, indices):
        group = self.cache.cache_controller.mem_pool_host
        entry = group.entry_map[pool]
        return len(indices) * entry.host_pool.get_size_per_token()

    def stage_backup(self, operation, apply):
        if getattr(operation, "pp_commit_staged", False):
            return
        operation.pp_commit_staged = True
        if not hasattr(operation, "pp_commit_identity"):
            raise RuntimeError("PP commit backup ACK lacks producer identity")
        size = self.pinned_bytes("kv", operation.host_indices)
        for transfer in operation.pool_transfers or ():
            if transfer.host_indices is not None and transfer.indices_from_pool is None:
                size += self.pinned_bytes(transfer.name, transfer.host_indices)

        def commit():
            apply()
            self.committed_by_kind["backup"] += 1
            if operation.pp_commit_keys:
                self.committed_by_kind["belief_add"] += 1
            self.issued_backups.pop(operation.id, None)
            for key in operation.pp_commit_keys:
                self.pending_backup_keys[key] -= 1
                if not self.pending_backup_keys[key]:
                    del self.pending_backup_keys[key]

        self.state.stage(
            operation.pp_commit_identity, backup_key(operation), commit, size
        )

    def note_backup(self, operation):
        operation.pp_commit_identity = self.identity("backup", backup_key(operation))
        keys = [("kv", key) for key in operation.hash_value]
        keys += [
            (str(transfer.name), key)
            for transfer in operation.pool_transfers or ()
            for key in transfer.keys or ()
        ]
        operation.pp_commit_keys = keys
        self.issued_backups[operation.id] = operation
        for key in keys:
            self.pending_backup_keys[key] += 1

    def backup_pending(self, pool, hashes):
        """Private duplicate-write suppression, never an existence/match belief."""
        return all(self.pending_backup_keys.get((str(pool), key), 0) for key in hashes)

    def stage_release(self, ack):
        if not isinstance(ack, ReleaseAck):
            raise TypeError("PP commit release ACK lacks producer identity")
        with self._id_lock:
            record = self.issued_releases.get(ack.identity)
        if (
            record is None
            or record["parts"] != ack.parts
            or not 0 <= ack.part < ack.parts
        ):
            raise RuntimeError("PP commit release fragment identity mismatch")
        record["received"].add(ack.part)
        if len(record["received"]) != ack.parts:
            return
        pool, indices = ack.pool, record["indices"]

        def commit():
            self.cache.cache_controller.mem_pool_host.free(indices, pool=pool)
            self.committed_by_kind["release:" + str(pool)] += 1
            self.releases_by_origin[record["origin"]] += 1
            with self._id_lock:
                del self.issued_releases[ack.identity]

        self.state.stage(
            ack.identity,
            [pool, len(indices)],
            commit,
            self.pinned_bytes(pool, indices),
        )

    def note_release(self, key, pool, indices, page_size):
        with self._id_lock:
            if len(self.issued_releases) >= self.state.max_pending:
                raise RuntimeError("PP commit physical release bound exceeded")
            identity = self.identity("release", [key, str(pool), len(indices)])
            self.issued_releases[identity] = {
                "indices": indices,
                "origin": str(key[1]) if len(key) > 1 else "unspecified",
                "parts": (len(indices) + page_size - 1) // page_size,
                "received": set(),
                "created_at": time.monotonic(),
            }
        return identity

    def defer_belief(self, action, pool, hashes):
        if self.applying:
            return False
        frame = sys._getframe(2)
        origin = f"{frame.f_code.co_name}:{frame.f_lineno} <- {frame.f_back.f_code.co_name}:{frame.f_back.f_lineno}"
        self.belief_proposals.propose(
            action, pool, hashes, self.cache.storage_existence_cache, origin
        )
        return True

    def _prepare_belief(self, identity, proposal):
        if identity in self.state.prepared:
            return
        action, pool, hashes = proposal["action"], proposal["pool"], proposal["hashes"]
        self.belief_proposals.mark_assigned(proposal)

        def apply():
            belief = self.cache.storage_existence_cache
            if action == "add":
                belief.add(pool, hashes)
            else:
                belief.invalidate_beyond(pool, hashes, 0)
            self.committed_by_kind["belief_" + action] += 1
            self.belief_proposals.complete(proposal)
            self.belief_effects.pop(identity, None)

        self.state.stage(identity, [action, pool, hashes], apply)

    def _assign_belief_proposals(self, reports):
        candidates = {
            rank: report.get("belief_proposal") for rank, report in reports.items()
        }
        candidates[0] = self.belief_proposals.head()
        for rank, proposal in sorted(candidates.items()):
            if proposal is None:
                continue
            if proposal["epoch"] != self.state.epoch or proposal["origin"] != rank:
                raise RuntimeError("PP commit belief proposal epoch/origin mismatch")
            if proposal["serial"] <= self.leader_proposals_seen[rank]:
                continue  # repeated READY head until its common commit
            if proposal["serial"] != self.leader_proposals_seen[rank] + 1:
                raise RuntimeError("PP commit belief proposal sequence gap")
            if (
                proposal["action"] not in ("add", "delete")
                or not 0 < len(proposal["hashes"]) <= BeliefProposals.CHUNK_KEYS
            ):
                raise RuntimeError("PP commit malformed belief proposal")
            # Only the leader assigns the canonical operation, including origin.
            identity = self.identity(
                "belief_" + proposal["action"],
                [
                    rank,
                    proposal["serial"],
                    proposal["pool"],
                    key_summary(proposal["hashes"]),
                ],
            )
            self.belief_effects[identity] = proposal
            self._prepare_belief(identity, proposal)
            self.leader_proposals_seen[rank] = proposal["serial"]

    def _frame(self, value):
        cache = self.cache
        # Two fixed sites on every logical round, including empty manifests.
        # Length is first broadcast within PP0's TP group, then along PP; the
        # payload is packed int64 words to retain v3.2 header+sequence checking.
        data = None
        length = torch.zeros(1, dtype=torch.int64)
        tp_root = (
            dist.get_process_group_ranks(cache.tp_group)[0]
            if cache.tp_world_size > 1
            else None
        )
        if cache.pp_rank == 0:
            if tp_root is None or dist.get_rank() == tp_root:
                raw = canonical(value)
                raw += b"\x00" * ((-len(raw)) % 8)
                data = torch.frombuffer(bytearray(raw), dtype=torch.int64)
                length[0] = len(data)
            if tp_root is not None:
                dist.broadcast(length, src=tp_root, group=cache.tp_group)
        cache._pp_sync(length, site=P2PTag.HIRADIX_PP_COMMIT_LENGTH)
        count = int(length.item())
        if not 0 < count <= 32768:
            raise RuntimeError("PP commit bounded manifest frame exceeded")
        if data is None:
            data = torch.empty(count, dtype=torch.int64)
        if cache.pp_rank == 0 and tp_root is not None:
            dist.broadcast(data, src=tp_root, group=cache.tp_group)
        cache._pp_sync(data, site=P2PTag.HIRADIX_PP_COMMIT_FRAME)
        return json.loads(data.numpy().tobytes().rstrip(b"\x00"))

    def tick(self):
        cache = self.cache
        # Downstream-only ACKs can remain outside PP0's physical-drain budget.
        # Advancing other commits must not hide an orphan in that input queue.
        for operation in self.issued_backups.values():
            ack_at = getattr(operation, "pp_commit_ack_at", None)
            if (
                ack_at is not None
                and time.monotonic() - ack_at >= self.state.stall_seconds
            ):
                raise RuntimeError(
                    f"PP commit orphan backup ACK rank={cache.pp_rank} identity={operation.pp_commit_identity}"
                )
        with self._id_lock:
            releases = list(self.issued_releases.items())
        for identity, record in releases:
            if time.monotonic() - record["created_at"] >= self.state.stall_seconds:
                raise RuntimeError(
                    f"PP commit orphan release ACK rank={cache.pp_rank} identity={identity} parts={len(record['received'])}/{record['parts']}"
                )
        self.belief_proposals.check_age()
        reports = {
            rank: report
            for rank, report in self.reports.poll().items()
            if report["epoch"] >= self.state.epoch
        }
        leader_tp = (
            getattr(cache, "tp_world_size", 1) == 1
            or dist.get_rank() == dist.get_process_group_ranks(cache.tp_group)[0]
        )
        if cache.pp_rank == 0 and leader_tp:
            self._assign_belief_proposals(reports)
        physical = [
            report["physical_proposal"]
            for report in reports.values()
            if report.get("physical_proposal")
        ]
        frame = (
            self.state.leader_frame(
                reports, local_confirmed=self.previous_confirmed, proposals=physical
            )
            if cache.pp_rank == 0
            else None
        )
        if frame is not None:
            frame["belief_entries"] = [
                [wire, self.belief_effects[OperationId(*wire)]]
                for _, wire, _ in frame["entries"]
                if OperationId(*wire) in self.belief_effects
            ]
            frame["admit"] = self.previous_admit and all(
                report.get("admit", True) for report in reports.values()
            )
        frame = self._frame(frame)
        self.admission_open = frame["admit"]
        for wire, proposal in frame["belief_entries"]:
            self._prepare_belief(OperationId(*wire), proposal)
        self.applying = True
        try:
            self.state.accept_frame(frame)
        finally:
            self.applying = False
        ready = self.state.ready()
        ready["belief_proposal"] = self.belief_proposals.head()
        ready["physical_proposal"] = next(
            (
                [identity.wire(), effect.payload_digest]
                for identity, effect in self.state.prepared.items()
                if identity.kind in ("backup", "release")
                and identity not in self.state.assigned
            ),
            None,
        )
        ready["admit"] = (
            len(self.state.prepared) < self.state.max_pending * 3 // 4
            and self.state.pinned_bytes < self.state.max_pinned_bytes * 3 // 4
        )
        # Same manifest on all TP ranks; only a contiguous prefix prepared on
        # every TP rank can be advertised. Local queue counts alone are not IDs.
        belief = cache.storage_existence_cache
        words = [
            (belief.commit_digest >> (32 * part)) & 0xFFFFFFFF for part in range(4)
        ]
        confirmed = torch.tensor(
            [self.state.confirmed, int(ready["admit"]), *words], dtype=torch.int64
        )
        maximum = confirmed.clone()
        cache._all_reduce_attn_groups(confirmed, dist.ReduceOp.MIN)
        cache._all_reduce_attn_groups(maximum, dist.ReduceOp.MAX)
        if confirmed[2:].tolist() != maximum[2:].tolist():
            self.belief_mismatches += 1
            if self.belief_mismatches <= 3:
                logger.warning(
                    "TP committed belief mismatch PP=%s C=%s; LRU touches remain phase2",
                    cache.pp_rank,
                    self.state.committed,
                )
        self.previous_confirmed = ready["confirmed"] = int(confirmed[0].item())
        self.previous_admit = ready["admit"] = bool(confirmed[1].item())
        ready["digest"] = self.state.hashes[self.previous_confirmed]
        ready["belief_set"] = belief.commit_digest
        ready["belief_count"] = len(belief)
        ready["lru_evictions"] = belief.lru_evictions
        ready["touches"] = belief.local_touches
        self._belief_snapshots[self.state.committed] = dict(ready)
        if cache.pp_rank == 0:
            for rank, report in reports.items():
                local = self._belief_snapshots.get(report["applied"])
                key = (rank, report["applied"])
                if (
                    local
                    and local["belief_set"] != report.get("belief_set")
                    and key not in self._belief_mismatch_seen
                ):
                    self._belief_mismatch_seen.add(key)
                    self.belief_mismatches += 1
                    if (
                        self.belief_mismatches <= 3
                        or self.belief_mismatches % 1024 == 0
                    ):
                        logger.warning(
                            "PP committed belief mismatch peer=%s local=%s remote=%s; phase2 LRU/prefetch not covered",
                            rank,
                            local,
                            report,
                        )
        # Keep a bounded history on every stage for delayed comparisons.
        while len(self._belief_snapshots) > 4096:
            del self._belief_snapshots[next(iter(self._belief_snapshots))]
        self.reports.publish(ready)
        cache._l3_tier_stats["pp_common_commit"] = self.state.snapshot()
        cache._l3_tier_stats["pp_common_commit"]["belief_proposals"] = (
            self.belief_proposals.snapshot()
        )
        cache._l3_tier_stats["pp_common_commit"]["belief_mismatch"] = (
            self.belief_mismatches
        )
        cache._l3_tier_stats["pp_common_commit"]["committed_by_kind"] = dict(
            self.committed_by_kind
        )
        cache._l3_tier_stats["pp_common_commit"]["releases_by_origin"] = dict(
            self.releases_by_origin
        )
        cache._l3_tier_stats["pp_common_commit"]["physical_backups_pending"] = len(
            self.issued_backups
        )
        cache._l3_tier_stats["pp_common_commit"]["physical_releases_pending"] = len(
            self.issued_releases
        )

    def reset(self):
        with self._id_lock:
            if self.issued_backups or self.issued_releases:
                raise RuntimeError("Cannot reset with physical ACK ownership in flight")
            if self.belief_proposals.pending:
                raise RuntimeError("Cannot reset with pending belief proposals")
            self.state.reset(self.state.epoch + 1)
            self.generations.clear()
            self.previous_confirmed = 0
            self.previous_admit = True
            self._belief_snapshots.clear()
            self.pending_backup_keys.clear()
            self.belief_proposals = BeliefProposals(self.cache.pp_rank, self.state.epoch)
            self.leader_proposals_seen.clear()
            self.belief_effects.clear()

    def close(self):
        return self.reports.close()
