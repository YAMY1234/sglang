"""Default-off workload controls for attention-DP measurement.

The shadow suffix deliberately preserves the original rid as a prefix, so the
existing AbortReq.matches contract also cancels every replica without a map.
"""

import copy
import os

DADP_ROUTE = os.environ.get("SGLANG_DADP_ROUTE", "")
DADP_MAX_NEW_TOKENS = os.environ.get("SGLANG_DADP_MAX_NEW_TOKENS", "")
DADP_SHADOW_SUFFIX = "__dadp_shadow_rank_"


class DadpReplicaInbox:
    """Hold rank-0 envelopes until an existing MLP-sync notifies every rank."""

    def __init__(self):
        self.envelopes = []
        self.aborts = []
        self.ready = False

    def defer(self, envelopes, aborts):
        self.envelopes.extend(envelopes)
        if self.envelopes:
            # An abort can arrive before the deferred request is admitted.
            # Replay it after admission as well as on the normal control path.
            self.aborts.extend(aborts)

    def take(self):
        result = self.envelopes + self.aborts
        self.envelopes = []
        self.aborts = []
        self.ready = False
        return result


DADP_REPLICA_INBOX = DadpReplicaInbox()


def is_dadp_shadow(rid: str) -> bool:
    _, marker, rank = rid.rpartition(DADP_SHADOW_SUFFIX)
    return bool(marker and rank.isdecimal())


def validate_dadp_config() -> None:
    if DADP_ROUTE not in ("", "replicate", "heavy_to_rank0"):
        raise ValueError(f"Invalid SGLANG_DADP_ROUTE={DADP_ROUTE!r}")
    if DADP_MAX_NEW_TOKENS not in ("", "1"):
        raise ValueError("SGLANG_DADP_MAX_NEW_TOKENS must be unset or 1")
    if DADP_ROUTE == "heavy_to_rank0":
        if int(os.environ["SGLANG_DADP_HEAVY_ISL"]) < 1:
            raise ValueError("SGLANG_DADP_HEAVY_ISL must be positive")


def clamp_dadp_max_new_tokens(params) -> None:
    # Preserve the caller's parameters (including zero-token requests).
    params.max_new_tokens = (
        1 if params.max_new_tokens is None else min(params.max_new_tokens, 1)
    )


def make_dadp_replicas(req, active_ranks, primary_rank):
    if DADP_SHADOW_SUFFIX in req.rid:
        raise ValueError("Request rid contains the reserved DADP shadow suffix")
    replicas = []
    for rank in active_ranks:
        replica = copy.copy(req)
        replica.sampling_params = copy.copy(req.sampling_params)
        replica.routed_dp_rank = rank
        if rank != primary_rank:
            replica.rid = f"{req.rid}{DADP_SHADOW_SUFFIX}{rank}"
            if req.session_params is not None:
                replica.session_params = copy.copy(req.session_params)
                if replica.session_params.rid is not None:
                    replica.session_params.rid += f"{DADP_SHADOW_SUFFIX}{rank}"
        replicas.append(replica)
    return replicas
