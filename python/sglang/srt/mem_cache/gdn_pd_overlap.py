"""Opt-in PD publication identities carried by the result FIFO.

This module owns no numerical operation. A publication event belongs to its
forward result, never to the pool's most recently bound graph bank.
"""
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import torch


@dataclass(frozen=True)
class PDPublicationRecord:
    batch_id: int
    requests: tuple  # (CPU request index, allocation generation)
    slots: frozenset
    producer_done: Any
    publication_done: Any
    keep_alive: tuple = field(repr=False)
    owner: Any = field(repr=False, compare=False)

    def validate_request(self, request_pool, req, batch_id):
        if self.batch_id != batch_id:
            raise RuntimeError("PD publication record belongs to another result batch")
        index = int(req.kv.req_pool_idx)
        generation = int(request_pool.req_generation[index])
        if (index, generation) not in self.requests:
            raise RuntimeError("PD publication request generation changed before send")


class PDPublicationRecords:
    def __init__(self, publication, request_pool):
        self.publication = publication
        self.request_pool = request_pool
        self.records = {}
        self.local_batch_id = 0
        self.states = {}
        self.deferred_frees = []
        self.draining = False
        self.in_forward = False
        self.stats = dict(publications=0, retired=0, schedule_waits=0, forward_waits=0,
                          publication_edges=0, transfer_waits=0, deferred_frees=0,
                          released_frees=0, implicit_d2h_added=0)
        self.edges = []
        self.install_allocator_leases()

    def install_allocator_leases(self):
        """Instance-local leases: S/D allocators keep their original methods.

        A CUDA free-index tensor is retained, not copied to the host to decide
        membership. Such frees conservatively hold the current record set.
        CPU request-row frees can cheaply select exactly the intersecting IDs.
        The free-list is updated only after result retirement AND device done.
        """
        rp = self.request_pool
        allocator = rp.mamba_allocator
        self.allocator = allocator
        self.free_mamba = allocator.free
        self.free_rows = rp.free_rows

        def hold(value, *, rows):
            if self.in_forward:
                raise RuntimeError("cannot release allocator slots during a PD publication forward")
            self.reap()
            states = tuple(state for state in self.states.values()
                           if not rows or any(i in value for i, _ in state.record.requests))
            if not states:
                return False
            # Snapshot a possibly reused index tensor on its producing stream.
            snapshot = tuple(value) if rows else value.clone()
            ready = None
            if not rows and value.is_cuda:
                ready = torch.cuda.Event()
                ready.record()
            self.deferred_frees.append((snapshot, ready, states, rows))
            self.stats["deferred_frees"] += 1
            return True

        @wraps(self.free_mamba)
        def free_mamba(indices):
            if indices.numel() and hold(indices, rows=False):
                return
            return self.free_mamba(indices)

        @wraps(self.free_rows)
        def free_rows(indices):
            if indices and hold(indices, rows=True):
                return
            return self.free_rows(indices)

        allocator.free, rp.free_rows = free_mamba, free_rows
        for target, names in ((allocator, ("alloc", "_do_alloc", "available_size")),
                              (rp, ("alloc_rows", "available_size"))):
            for name in names:
                original = getattr(target, name)
                @wraps(original)
                def allocate(*args, _original=original, **kwargs):
                    self.reap()
                    return _original(*args, **kwargs)
                setattr(target, name, allocate)
        original_free = rp.free
        @wraps(original_free)
        def free_request(req):
            self.release_request(req)
            return original_free(req)
        rp.free = free_request
        for target in (allocator, rp):
            original_clear = target.clear
            @wraps(original_clear)
            def clear(*args, _original=original_clear, **kwargs):
                self.reap()
                if self.in_forward or self.states or self.deferred_frees:
                    raise RuntimeError("cannot clear allocator with live PD publication leases")
                return _original(*args, **kwargs)
            target.clear = clear

    def reap(self):
        if self.draining:
            return
        self.draining = True
        try:
            for batch_id, state in list(self.states.items()):
                if state.consumed and not state.remaining and state.device_done():
                    state.retired = True
                    del self.states[batch_id]
                    del self.records[batch_id]
                    self.stats["retired"] += 1
            pending = []
            for value, event, states, rows in self.deferred_frees:
                if all(state.retired for state in states) and (event is None or event.query()):
                    (self.free_rows if rows else self.free_mamba)(list(value) if rows else value)
                    self.stats["released_frees"] += 1
                else:
                    pending.append((value, event, states, rows))
            self.deferred_frees = pending
            if self.stats["publications"] != self.stats["retired"] + len(self.states):
                raise RuntimeError("PD publication lifetime accounting mismatch")
        finally:
            self.draining = False

    def release_request(self, req):
        index = req.kv.req_pool_idx
        if index is None:
            return
        identity = (int(index), int(self.request_pool.req_generation[int(index)]))
        for state in self.states.values():
            state.remaining.discard(identity)
        req.pd_publication_record = None
        req.pd_publication_batch_id = None
        self.reap()

    def consumed(self, record, reqs):
        state = self.states[record.batch_id]
        state.consumed = True
        for req in reqs:
            previous = getattr(req, "pd_publication_record", None)
            if previous is not None and previous is not record and previous.batch_id in self.states:
                # An earlier, non-final chunk no longer has a state sender.
                old = self.states[previous.batch_id]
                if old.transfer_attached:
                    raise RuntimeError("request advanced after attaching a final PD state sender")
                index = int(req.kv.req_pool_idx)
                old.remaining.discard((index, int(self.request_pool.req_generation[index])))

    def wait(self, record, lane):
        """Register an edge on THIS reader stream; never consume another edge."""
        state = self.states.get(record.batch_id)
        if state is None:
            if not all(e is None or e.query() for e in (record.producer_done, record.publication_done)):
                raise RuntimeError("retired PD record still has device work")
            return
        stream = torch.cuda.current_stream(self.publication.pool.a.device)
        key = (lane, stream.cuda_stream if isinstance(stream.cuda_stream, int) else id(stream))
        if key in state.waited:
            return
        if self.publication.runtime is None:
            from .gdn_pd_publication import CudaPublicationRuntime
            self.publication.runtime = CudaPublicationRuntime(self.publication.pool.a.device)
        for event in (record.producer_done, record.publication_done):
            if event is not None:
                self.publication.runtime.join(event)
        state.waited.add(key)
        self.stats[lane + "_waits"] += 1
        self.edges.append((record.batch_id, lane, "producer+publication"))

    def wait_slots(self, slots, lane):
        self.reap()
        # No added D2H for schedule/allocator readers lacking a host closure.
        # Forward's existing host selection is reused. Unknown means all.
        if isinstance(slots, torch.Tensor):
            slots = None if slots.is_cuda else frozenset(map(int, slots.reshape(-1).tolist()))
        if slots is not None and not isinstance(slots, (set, frozenset)):
            slots = None
        for state in tuple(self.states.values()):
            if slots is None or not state.record.slots or not state.record.slots.isdisjoint(slots):
                self.wait(state.record, lane)

    def reserved_slots(self):
        stream = torch.cuda.current_stream(self.publication.pool.a.device)
        stream_id = stream.cuda_stream if isinstance(stream.cuda_stream, int) else id(stream)
        return frozenset(s for state in self.states.values() if not state.device_done()
                         and not any(key[1] == stream_id for key in state.waited)
                         for s in state.record.slots)

    def after_forward(self, batch, submitted_before):
        # ModelRunner-only warmup calls have no scheduler iteration. Negative
        # IDs keep these distinct from the positive scheduler forward_iter.
        batch_id = getattr(batch, "pd_publication_batch_id", None)
        if batch_id is None:
            self.local_batch_id -= 1
            batch_id = self.local_batch_id
        ids = batch.req_pool_indices_cpu
        if isinstance(ids, torch.Tensor) and ids.device.type != "cpu":
            raise RuntimeError("PD publication identities require CPU request indices")
        requests = tuple((int(i), int(self.request_pool.req_generation[int(i)])) for i in ids)
        publication = self.publication
        published = publication.stats["submitted"] != submitted_before
        producer_done = torch.cuda.Event()
        producer_done.record()
        record = PDPublicationRecord(
            batch_id, requests,
            frozenset(publication.pending_slots or ()) if published else frozenset(),
            producer_done, publication.ticket if published else None,
            (publication.pending,) if published else (),
            self,
        )
        if batch_id in self.records:
            raise RuntimeError("duplicate PD publication batch identity")
        self.records[batch_id] = record
        self.states[batch_id] = PublicationLifetime(record, set(requests))
        self.stats["publications"] += 1
        if published:
            self.stats["publication_edges"] += 1
            self.edges.append((batch_id, "publication", "forward-ready"))
        batch.pd_publication_record = record
        return record

    def for_request(self, req):
        record = getattr(req, "pd_publication_record", None)
        batch_id = getattr(req, "pd_publication_batch_id", None)
        if record is None:
            raise RuntimeError("PD send is missing its result FIFO publication record")
        record.validate_request(self.request_pool, req, batch_id)
        state = self.states.get(record.batch_id)
        if state is not None:
            state.transfer_attached = True
        return record


@dataclass
class PublicationLifetime:
    record: PDPublicationRecord
    remaining: set
    consumed: bool = False
    retired: bool = False
    transfer_attached: bool = False
    waited: set = field(default_factory=set)

    def device_done(self):
        return all(event is None or event.query()
                   for event in (self.record.producer_done, self.record.publication_done))


def bind_result_record(batch, result):
    """Consume the record attached to this FIFO result, before cache/send work."""
    record = result.pd_publication_record
    if record is None:
        return
    if record.batch_id != batch.forward_iter:
        raise RuntimeError("PD result FIFO batch/publication identity mismatch")
    record.owner.consumed(record, batch.reqs)
    record.owner.wait(record, "schedule")
    for req in batch.reqs:
        if req.kv.req_pool_idx is None:
            continue  # cancelled and released; it must never acquire a sender
        record.validate_request(batch.req_to_token_pool, req, batch.forward_iter)
        req.pd_publication_record = record
        req.pd_publication_batch_id = batch.forward_iter


def enable_records(publication, request_pool):
    from sglang.srt.environ import envs

    if envs.SGLANG_GDN_PD_PUBLISH_OVERLAP_OK.get():
        publication.records = PDPublicationRecords(publication, request_pool)
