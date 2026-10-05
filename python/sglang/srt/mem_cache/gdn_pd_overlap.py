"""Opt-in PD publication identities carried by the result FIFO.

This module owns no numerical operation. A publication event belongs to its
forward result, never to the pool's most recently bound graph bank.
"""
from dataclasses import dataclass, field
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
        )
        if batch_id in self.records:
            raise RuntimeError("duplicate PD publication batch identity")
        self.records[batch_id] = record
        batch.pd_publication_record = record
        return record

    def for_request(self, req):
        record = getattr(req, "pd_publication_record", None)
        batch_id = getattr(req, "pd_publication_batch_id", None)
        if record is None:
            raise RuntimeError("PD send is missing its result FIFO publication record")
        record.validate_request(self.request_pool, req, batch_id)
        return record


def bind_result_record(batch, result):
    """Consume the record attached to this FIFO result, before cache/send work."""
    record = result.pd_publication_record
    if record is None:
        return
    if record.batch_id != batch.forward_iter:
        raise RuntimeError("PD result FIFO batch/publication identity mismatch")
    for req in batch.reqs:
        record.validate_request(batch.req_to_token_pool, req, batch.forward_iter)
        req.pd_publication_record = record
        req.pd_publication_batch_id = batch.forward_iter


def enable_records(publication, request_pool):
    from sglang.srt.environ import envs

    if envs.SGLANG_GDN_PD_PUBLISH_OVERLAP_OK.get():
        publication.records = PDPublicationRecords(publication, request_pool)
