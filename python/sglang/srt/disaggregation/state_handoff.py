"""Local lifecycle dispatch for state transferred alongside P/D KV entries.

Transport registration stays with each pool. These hooks run before the sender
publishes a final chunk, before destination registration, and after successful
transfer/metadata validation. Local cache metadata must never travel over RDMA.
"""
from enum import Enum
from typing import Protocol


class HandoffKind(str, Enum):
    STATE_FACTOR = "state-factor"
    LATENT = "latent"
    CODED_KV = "coded-kv"


class HandoffHandler(Protocol):
    def before_send(self, req): ...
    def prepare_receive(self, req): ...
    def commit_receive(self, req): ...


def register_handoff(pool, kind: HandoffKind, handler: HandoffHandler):
    """Register one implementation; latent/coded-kv have no default no-op."""
    kind = HandoffKind(kind)
    handlers = getattr(pool, "pd_state_handoffs", None)
    if handlers is None:
        handlers = pool.pd_state_handoffs = {}
    if kind in handlers:
        raise ValueError(f"duplicate P/D handoff kind: {kind.value}")
    for phase in ("before_send", "prepare_receive", "commit_receive"):
        if not callable(getattr(handler, phase, None)):
            raise TypeError(f"{kind.value} requires {phase}")
    handlers[kind] = handler


def dispatch_handoff(pool, phase, req):
    if phase not in ("before_send", "prepare_receive", "commit_receive"):
        raise ValueError(f"unknown P/D handoff phase: {phase}")
    # Flag-off pools have no handlers and perform no state writes.
    for handler in getattr(pool, "pd_state_handoffs", {}).values():
        getattr(handler, phase)(req)


class FactorStateHandoff:
    """a/U/W/count use the existing MAMBA slot-entry transport registration."""
    def __init__(self, factor_pool):
        self.pool = factor_pool

    def before_send(self, req):
        slot = req.kv.mamba_pool_idx
        if slot is None:
            raise RuntimeError("factor P/D send without a mamba slot")
        # commit_extend[_batched] truncates once at the final prefill. Reading
        # count also synchronizes the producer stream before RDMA publication.
        if not bool((self.pool.count[:, slot] == self.pool.cfg.r).all().item()):
            raise RuntimeError("factor P/D send before final prefill r truncation")

    def prepare_receive(self, req):
        slot = req.kv.mamba_pool_idx
        if slot is None:
            raise RuntimeError("factor P/D receive without a mamba slot")
        self.pool.mark_transferred_slots(slot.reshape(-1))
        tracks = req.kv.mamba_ping_pong_track_buffer
        if tracks is not None:
            self.pool.mark_transferred_slots(tracks[tracks >= 0])
        req.kv.mamba_last_track_idx = None
        req.kv.mamba_last_track_seqlen = None

    def commit_receive(self, req):
        # Idempotent on retry. Never clear or refactor the received tensors.
        self.pool.mark_transferred_slots(req.kv.mamba_pool_idx.reshape(-1))
        req.kv.mamba_cow_src_index = None
        req.kv.mamba_needs_clear = False
