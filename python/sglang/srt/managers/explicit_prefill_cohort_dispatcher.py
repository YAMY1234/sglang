# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Dispatch an explicitly identified producer prefill cohort atomically."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from sglang.srt.managers.scheduler_input_blocker import input_blocker_guard_region

logger = logging.getLogger(__name__)

_MAX_CLOSED_COHORTS = 4096


@dataclass
class _Member:
    rid: str
    request: Any = None
    future: Optional[asyncio.Future[None]] = None


@dataclass
class _Cohort:
    expected_size: int
    members: dict[int, _Member] = field(default_factory=dict)


class ExplicitPrefillCohortDispatcher:
    """Wait for producer-declared membership, then globally unblock.

    Ordinary requests bypass coordination; the producer owns cohort closure.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        dispatch_one: Callable[[Any], None],
        dispatch_control: Callable[[Any], None],
        dp_size: Optional[int] = None,
        assign_balanced_dp_ranks: bool = False,
    ) -> None:
        if assign_balanced_dp_ranks and dp_size is None:
            raise ValueError("balanced cohort routing requires dp_size")
        self.enabled = enabled
        self._dispatch_one = dispatch_one
        self._dispatch_control = dispatch_control
        self._dp_size = dp_size
        self._assign_balanced_dp_ranks = assign_balanced_dp_ranks
        self._cohorts: dict[str, _Cohort] = {}
        self._rid_to_cohort: dict[str, tuple[str, int]] = {}
        self._failed_rids: dict[str, str] = {}
        self._closed: OrderedDict[str, str] = OrderedDict()

    @staticmethod
    def _validate_spec(
        cohort_id: Optional[str],
        cohort_size: Optional[int],
        cohort_index: Optional[int],
    ) -> bool:
        values = (cohort_id, cohort_size, cohort_index)
        if all(value is None for value in values):
            return False
        if any(value is None for value in values):
            raise ValueError(
                "prefill_cohort_id, prefill_cohort_size, and "
                "prefill_cohort_index must be provided together"
            )
        if not isinstance(cohort_id, str) or not cohort_id:
            raise ValueError("prefill_cohort_id must be a non-empty string")
        if not isinstance(cohort_size, int) or isinstance(cohort_size, bool):
            raise ValueError("prefill_cohort_size must be an integer")
        if cohort_size <= 0:
            raise ValueError("prefill_cohort_size must be positive")
        if not isinstance(cohort_index, int) or isinstance(cohort_index, bool):
            raise ValueError("prefill_cohort_index must be an integer")
        if cohort_index < 0 or cohort_index >= cohort_size:
            raise ValueError(
                "prefill_cohort_index must be in "
                f"[0, prefill_cohort_size), got {cohort_index}"
            )
        return True

    def register(
        self,
        *,
        rid: str,
        cohort_id: Optional[str],
        cohort_size: Optional[int],
        cohort_index: Optional[int],
    ) -> bool:
        """Register producer membership before tokenization begins."""

        has_spec = self._validate_spec(cohort_id, cohort_size, cohort_index)
        if not has_spec:
            return False
        if not self.enabled:
            raise ValueError(
                "explicit prefill cohorts require "
                "SGLANG_ENABLE_COLOCATED_BATCH_GEN=true on a DP-attention "
                "disaggregated prefill worker"
            )

        assert cohort_id is not None
        assert cohort_size is not None
        assert cohort_index is not None
        if cohort_id in self._closed:
            raise ValueError(
                f"prefill cohort {cohort_id!r} is already closed: "
                f"{self._closed[cohort_id]}"
            )
        if rid in self._rid_to_cohort or rid in self._failed_rids:
            raise ValueError(f"duplicate prefill cohort request id: {rid}")

        cohort = self._cohorts.setdefault(cohort_id, _Cohort(expected_size=cohort_size))
        if cohort.expected_size != cohort_size:
            raise ValueError(
                f"prefill cohort {cohort_id!r} size changed from "
                f"{cohort.expected_size} to {cohort_size}"
            )
        if cohort_index in cohort.members:
            raise ValueError(
                f"duplicate prefill cohort index {cohort_index} for {cohort_id!r}"
            )

        cohort.members[cohort_index] = _Member(rid=rid)
        self._rid_to_cohort[rid] = (cohort_id, cohort_index)
        return True

    async def dispatch(self, rid: str, request: Any) -> None:
        """Mark one registered member ready and await atomic dispatch."""

        failed = self._failed_rids.pop(rid, None)
        if failed is not None:
            raise RuntimeError(failed)

        location = self._rid_to_cohort.get(rid)
        if location is None:
            self._dispatch_one(request)
            return

        cohort_id, cohort_index = location
        cohort = self._cohorts[cohort_id]
        member = cohort.members[cohort_index]
        if member.future is not None:
            raise ValueError(f"prefill cohort request is already ready: {rid}")

        member.request = request
        member.future = asyncio.get_running_loop().create_future()
        self._flush_if_complete(cohort_id, cohort)
        try:
            await member.future
        except asyncio.CancelledError as exc:
            self.abort(rid, exc)
            raise

    def abort(self, rid: str, exc: BaseException) -> None:
        """Abort an incomplete cohort and wake every already-ready member."""

        location = self._rid_to_cohort.get(rid)
        if location is None:
            return
        cohort_id, _ = location
        cohort = self._cohorts.pop(cohort_id)
        message = f"prefill cohort {cohort_id!r} aborted: {exc}"
        for member in cohort.members.values():
            self._rid_to_cohort.pop(member.rid, None)
            if member.rid != rid and member.future is None:
                self._failed_rids[member.rid] = message
            if member.future is not None and not member.future.done():
                member.future.set_exception(RuntimeError(message))
        self._remember_closed(cohort_id, message)

    def _flush_if_complete(self, cohort_id: str, cohort: _Cohort) -> None:
        if len(cohort.members) != cohort.expected_size:
            return
        if any(member.future is None for member in cohort.members.values()):
            return

        members = [cohort.members[index] for index in range(cohort.expected_size)]
        try:
            if self._dp_size is not None:
                if self._assign_balanced_dp_ranks:
                    self._route_balanced(members)
                routed_dp_ranks = [
                    getattr(member.request, "routed_dp_rank", None)
                    for member in members
                ]
                requests_per_rank, remainder = divmod(
                    cohort.expected_size, self._dp_size
                )
                if (
                    remainder != 0
                    or requests_per_rank <= 0
                    or not all(isinstance(rank, int) for rank in routed_dp_ranks)
                    or Counter(routed_dp_ranks)
                    != Counter(
                        {rank: requests_per_rank for rank in range(self._dp_size)}
                    )
                ):
                    raise ValueError(
                        "a DP-balanced explicit prefill cohort must contain the "
                        "same positive number of requests for every DP rank; "
                        f"size={cohort.expected_size}, dp_size={self._dp_size}, "
                        f"routed_dp_ranks={routed_dp_ranks}"
                    )
            with input_blocker_guard_region(self._dispatch_control):
                for member in members:
                    self._dispatch_one(member.request)
        except BaseException as exc:
            for member in members:
                if member.future is not None and not member.future.done():
                    member.future.set_exception(exc)
        else:
            ranks = [
                getattr(member.request, "routed_dp_rank", None) for member in members
            ]
            logger.info(
                "Explicit prefill cohort dispatched id=%s size=%d routed_dp_ranks=%s",
                cohort_id,
                cohort.expected_size,
                ranks,
            )
            for member in members:
                if member.future is not None and not member.future.done():
                    member.future.set_result(None)
        finally:
            self._cohorts.pop(cohort_id, None)
            for member in members:
                self._rid_to_cohort.pop(member.rid, None)
            self._remember_closed(cohort_id, "dispatched")

    def _route_balanced(self, members: list[_Member]) -> None:
        """Assign a closed cohort by causal prompt work with equal rank counts."""

        assert self._dp_size is not None
        requests_per_rank, remainder = divmod(len(members), self._dp_size)
        if remainder != 0 or requests_per_rank <= 0:
            raise ValueError(
                "a DP-balanced explicit prefill cohort must contain the same "
                "positive number of requests for every DP rank; "
                f"size={len(members)}, dp_size={self._dp_size}"
            )

        rank_counts = [0] * self._dp_size
        rank_work = [0] * self._dp_size
        ordered = sorted(
            enumerate(members),
            key=lambda item: (
                -self._causal_prompt_work(item[1].request),
                item[0],
            ),
        )
        for _, member in ordered:
            available = [
                rank
                for rank in range(self._dp_size)
                if rank_counts[rank] < requests_per_rank
            ]
            rank = min(
                available,
                key=lambda candidate: (rank_work[candidate], candidate),
            )
            member.request.routed_dp_rank = rank
            rank_counts[rank] += 1
            rank_work[rank] += self._causal_prompt_work(member.request)

    @staticmethod
    def _causal_prompt_work(request: Any) -> int:
        input_length = len(request.input_ids)
        return input_length * (input_length + 1) // 2

    def _remember_closed(self, cohort_id: str, reason: str) -> None:
        self._closed[cohort_id] = reason
        self._closed.move_to_end(cohort_id)
        while len(self._closed) > _MAX_CLOSED_COHORTS:
            self._closed.popitem(last=False)
