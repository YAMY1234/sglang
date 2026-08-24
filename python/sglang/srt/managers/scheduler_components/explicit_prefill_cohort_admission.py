"""Restore explicit producer-cohort atomicity after PD KV bootstrap."""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import torch

from sglang.srt.distributed import get_world_group

logger = logging.getLogger(__name__)

_MAX_CLOSED_COHORTS = 4096


@dataclass
class _LocalCohort:
    expected_local_size: int
    reqs: OrderedDict[str, object] = field(default_factory=OrderedDict)
    ready_rids: set[str] = field(default_factory=set)
    failed: bool = False


class ExplicitPrefillCohortAdmissionCoordinator:
    """Hold a DP-balanced cohort until every rank's KV bootstrap is ready.

    Ordinary requests and partial-DP cohorts bypass coordination.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        dp_size: int,
        gather_status: Optional[Callable[[int, int], list[tuple[int, int]]]] = None,
    ) -> None:
        self.enabled = enabled
        self.dp_size = dp_size
        self._gather_status = gather_status or self._distributed_gather_status
        self._cohorts: OrderedDict[str, _LocalCohort] = OrderedDict()
        self._closed: OrderedDict[str, str] = OrderedDict()

    def _eligible(self, req: object) -> bool:
        return bool(
            self.enabled
            and getattr(req, "prefill_cohort_id", None)
            and isinstance(getattr(req, "prefill_cohort_size", None), int)
            and req.prefill_cohort_size >= self.dp_size
            and req.prefill_cohort_size % self.dp_size == 0
            and getattr(req, "prefill_cohort_index", None) is not None
            and getattr(req, "routed_dp_rank", None) is not None
        )

    def register(self, req: object) -> None:
        if not self._eligible(req):
            return
        cohort_id = req.prefill_cohort_id
        if cohort_id in self._closed:
            return
        expected_local_size = req.prefill_cohort_size // self.dp_size
        state = self._cohorts.setdefault(
            cohort_id, _LocalCohort(expected_local_size=expected_local_size)
        )
        if state.expected_local_size != expected_local_size:
            raise ValueError(
                f"local prefill cohort {cohort_id!r} size changed from "
                f"{state.expected_local_size} to {expected_local_size}"
            )
        rid = req.rid
        if rid in state.reqs:
            raise ValueError(
                f"duplicate local member {rid!r} for prefill cohort {cohort_id!r}"
            )
        if len(state.reqs) >= expected_local_size:
            raise ValueError(
                f"too many local members for prefill cohort {cohort_id!r}: "
                f"expected {expected_local_size}"
            )
        state.reqs[rid] = req

    def mark_failed(self, req: object) -> None:
        cohort_id = getattr(req, "prefill_cohort_id", None)
        state = self._cohorts.get(cohort_id)
        if state is not None:
            state.failed = True

    def stage_and_release(
        self, ready_reqs: Iterable[object], failed_reqs: Iterable[object]
    ) -> list[object]:
        passthrough = []
        for req in ready_reqs:
            cohort_id = getattr(req, "prefill_cohort_id", None)
            state = self._cohorts.get(cohort_id)
            if state is None:
                passthrough.append(req)
                continue
            rid = req.rid
            if state.reqs.get(rid) is not req:
                raise ValueError(f"request identity changed in cohort {cohort_id!r}")
            state.ready_rids.add(rid)

        for req in failed_reqs:
            cohort_id = getattr(req, "prefill_cohort_id", None)
            state = self._cohorts.get(cohort_id)
            if state is not None:
                state.failed = True

        if not self._cohorts:
            return passthrough

        cohort_id, state = next(iter(self._cohorts.items()))
        local_status = -1 if state.failed else len(state.ready_rids)
        gathered = self._gather_status(self._cohort_hash(cohort_id), local_status)
        hashes = {cohort_hash for cohort_hash, _ in gathered}
        if len(hashes) != 1:
            raise RuntimeError(
                "explicit prefill cohort order diverged across scheduler ranks: "
                f"local={cohort_id!r} gathered_hashes={sorted(hashes)}"
            )
        statuses = [status for _, status in gathered]
        if any(status < 0 for status in statuses):
            self._close(cohort_id, "bootstrap_failed")
            passthrough.extend(
                req for rid, req in state.reqs.items() if rid in state.ready_rids
            )
            logger.warning(
                "Explicit prefill cohort bootstrap failed id=%s; releasing "
                "surviving members without atomic admission",
                cohort_id,
            )
        elif all(status >= state.expected_local_size for status in statuses):
            if (
                len(state.reqs) != state.expected_local_size
                or len(state.ready_rids) != state.expected_local_size
            ):
                raise RuntimeError(
                    f"global cohort {cohort_id!r} ready without all local requests: "
                    f"registered={len(state.reqs)}, ready={len(state.ready_rids)}, "
                    f"expected={state.expected_local_size}"
                )
            passthrough.extend(state.reqs.values())
            self._close(cohort_id, "released")
            logger.info(
                "Explicit prefill cohort bootstrap admission released id=%s "
                "local_size=%d",
                cohort_id,
                state.expected_local_size,
            )

        return passthrough

    def _close(self, cohort_id: str, reason: str) -> None:
        self._cohorts.pop(cohort_id, None)
        self._closed[cohort_id] = reason
        self._closed.move_to_end(cohort_id)
        while len(self._closed) > _MAX_CLOSED_COHORTS:
            self._closed.popitem(last=False)

    @staticmethod
    def _cohort_hash(cohort_id: str) -> int:
        value = int.from_bytes(
            hashlib.sha256(cohort_id.encode("utf-8")).digest()[:8], "big"
        ) & ((1 << 63) - 1)
        return value or 1

    @staticmethod
    def _distributed_gather_status(
        cohort_hash: int, local_status: int
    ) -> list[tuple[int, int]]:
        group = get_world_group().cpu_group
        local = torch.tensor([cohort_hash, local_status], dtype=torch.int64)
        gathered = [
            torch.empty_like(local)
            for _ in range(torch.distributed.get_world_size(group))
        ]
        torch.distributed.all_gather(gathered, local, group=group)
        return [(int(item[0].item()), int(item[1].item())) for item in gathered]
