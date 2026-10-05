"""Local lifecycle dispatch for state transferred alongside P/D KV entries.

Transport registration stays with each pool. These hooks run before the sender
publishes a final chunk, before destination registration, and after successful
transfer/metadata validation. Local cache metadata must never travel over RDMA.
"""
import logging
from contextlib import nullcontext
from enum import Enum
from threading import Lock, local
from typing import Protocol

logger = logging.getLogger(__name__)
_warned_sender_types = set()
_sender_warning_lock = Lock()


def supports_state_handoff_fence(sender):
    """Probe a sender class at P startup, or a request's actual sender.

    Missing optional support uses the original synchronous handoff. Never
    substitute a no-op fence for a backend that cannot drain producer events.
    """
    if callable(getattr(sender, "set_state_handoff_fence", None)):
        return True
    sender_type = sender if isinstance(sender, type) else type(sender)
    with _sender_warning_lock:
        if sender_type in _warned_sender_types:
            return False
        _warned_sender_types.add(sender_type)
    logger.warning(
        "PD publish join offload: sender %s.%s lacks callable "
        "set_state_handoff_fence; falling back to synchronous join and count validation",
        sender_type.__module__, sender_type.__qualname__,
    )
    return False


class HandoffKind(str, Enum):
    STATE_FACTOR = "state-factor"
    DENSE_BOUNDARY = "dense-shallow-boundary"
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
        publication = getattr(self.pool, "_pd_batch_publication", None)
        records = publication.records if publication is not None else None
        record = records.for_request(req) if records is not None else None
        offload = (publication is not None and publication.offload_join
                   and (record is not None or getattr(req, "_pfactor_agg_contract", False)))
        if offload:
            sender = getattr(req, "disagg_kv_sender", None)
            offload = supports_state_handoff_fence(sender)
        if not offload:
            if record is not None:
                for event in (record.producer_done, record.publication_done):
                    if event is not None:
                        event.synchronize()
            else:
                join = getattr(self.pool, "pside_join", None)
                if join is not None:
                    join()
        slot = req.kv.mamba_pool_idx
        if slot is None:
            raise RuntimeError("factor P/D send without a mamba slot")
        # The delivered P31 service truncates after shallow P/emitter work,
        # then executes one ordinary boundary decode on P. Preserve that exact
        # update across RDMA; truncating a second time would differ from AGG.
        # Only the model-aware scheduler may set this local phase marker.
        steps = getattr(req, "factored_prefill_boundary_steps", 0)
        if steps not in (0, 1) or (steps and not self.pool.cfg.strict_chunk):
            raise RuntimeError("invalid factor prefill boundary phase")
        expected = self.pool.cfg.r + steps
        if offload:
            sender.set_state_handoff_fence(FactorTransferFence(
                record.publication_done if record is not None else publication.transfer_event(),
                self.pool.count, slot, expected, record=record,
                request_index=int(req.kv.req_pool_idx) if record is not None else None))
            return
        # Reading count also synchronizes the producer before publication.
        if not bool((self.pool.count[:, slot] == expected).all().item()):
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


_transfer_local = local()


class FactorTransferFence:
    """Local-only ownership of producer events and the original count read.

    The scheduler never reads count. The transfer worker waits on both the
    publication and the queue-time producer event before *any* RDMA, then
    validates on its own CUDA stream so .item() cannot wait for a later trunk.
    A failed event or count check never permits a send or a success status.
    """

    def __init__(self, publication_done, count, slot, expected, *, record=None, request_index=None):
        self.publication_done = publication_done
        self.count, self.slot, self.expected = count, slot, expected
        self.validated = False
        self.record, self.request_index = record, request_index
        self.generation = (int(record.owner.request_pool.req_generation[request_index])
                           if record is not None else None)

    def wait(self, producer_done):
        if self.validated:
            return
        if producer_done is None:
            raise RuntimeError("factor transfer is missing its producer event")
        if self.record is not None and int(self.record.owner.request_pool.req_generation[
                self.request_index]) != self.generation:
            raise RuntimeError("PD publication generation changed before worker send")
        own_producer = self.record.producer_done if self.record is not None else None
        for event in (self.publication_done, own_producer, producer_done):
            if event is not None:
                event.synchronize()
                if not event.query():
                    raise RuntimeError("factor transfer event incomplete before RDMA")
        import torch

        context = nullcontext()
        if self.count.is_cuda:
            streams = getattr(_transfer_local, "streams", None)
            if streams is None:
                streams = _transfer_local.streams = {}
            key = self.count.device
            if key not in streams:
                streams[key] = torch.cuda.Stream(device=key)
            context = torch.cuda.stream(streams[key])
        with context:
            if not bool((self.count[:, self.slot] == self.expected).all().item()):
                raise RuntimeError("factor P/D send before final prefill r truncation")
        if self.record is not None:
            if int(self.record.owner.request_pool.req_generation[self.request_index]) != self.generation:
                raise RuntimeError("PD publication generation changed while worker waited")
            self.record.owner.stats["transfer_waits"] += 1
            self.record.owner.edges.append((self.record.batch_id, "transfer", "producer+publication+queue"))
        self.validated = True
