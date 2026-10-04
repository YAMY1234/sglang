"""Batch-owned full-N state and local prefill fences for AGG overlap.

No tensor-to-host reads are introduced here. Publication stays on the forward
stream. The scheduler drains only a completed full-N result before its next
planning/eviction pass; ordinary decode overlap keeps its existing order.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)
TRACK_FIELDS = ("mamba_last_track_idx", "mamba_next_track_idx", "mamba_last_track_seqlen")
_ABSENT = object()


@contextmanager
def track_selection(req, selected):
    """The native tracking helper reads this marker only during this call."""
    previous = getattr(req, "_pfactor_agg_contract", _ABSENT)
    req._pfactor_agg_contract = selected
    try:
        yield
    finally:
        if previous is _ABSENT:
            del req._pfactor_agg_contract
        else:
            req._pfactor_agg_contract = previous


@dataclass(frozen=True)
class TrackSnapshot:
    req: object
    before: tuple
    after: tuple
    selected: bool


@dataclass
class FullNBatchRecord:
    controller: object
    source: object
    serial: int
    selected: bool
    tracks: dict = field(default_factory=dict)
    identities: tuple = ()
    frozen_tracks: tuple = ()
    plan: object = None
    publication_done: object = None
    sealed: bool = False
    consumed: bool = False
    result_validated: bool = False
    prior_forward_done: object = None

    def seal(self, selected):
        if self.sealed:
            raise RuntimeError("full-N batch selected twice")
        self.selected = selected
        self.frozen_tracks = tuple(self.tracks.values())
        self.identities = tuple((req, req.kv.req_pool_idx) for req in self.source.reqs)
        self.sealed = True

    def before_result(self):
        # Called *after* the existing copy_done.synchronize(), whose copy stream
        # waits for this forward. query checks that ordering, without a second
        # host synchronization or an implicit D2H.
        if (not self.sealed or not self.selected or self.consumed
                or self.publication_done is None
                or not self.controller.runtime.complete(self.publication_done)):
            raise RuntimeError("full-N result consumed before publication completed")
        for req, slot in self.identities:
            if req.kv.req_pool_idx != slot:
                raise RuntimeError("full-N request slot reused before result consumption")
        for snapshot in self.frozen_tracks:
            if tuple(getattr(snapshot.req.kv, name) for name in TRACK_FIELDS) != snapshot.after:
                raise RuntimeError("full-N track ownership changed before result consumption")
        self.result_validated = True


class CudaBoundaryRuntime:
    def __init__(self, device):
        self.device = device
        self.forward_stream = None
        self.schedule_stream = None

    def attach(self, scheduler):
        self.forward_stream = scheduler.forward_stream
        self.schedule_stream = scheduler.schedule_stream

    def fence_before_plan(self):
        import torch

        if self.forward_stream is None:
            raise RuntimeError("full-N overlap scheduler was not attached before planning")
        prior = torch.cuda.Event()
        prior.record(self.forward_stream)
        self.schedule_stream.wait_event(prior)
        return prior

    def published(self):
        import torch

        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(self.device))
        return done

    def wait_for_result(self, done):
        self.schedule_stream.wait_event(done)

    @staticmethod
    def complete(done):
        return done.query()


class FullNOverlap:
    def __init__(self, device, runtime=None):
        self.runtime = runtime if runtime is not None else CudaBoundaryRuntime(device)
        self.serial = 0
        self.pending = None
        self.stats = dict(plan_events=0, publication_events=0, result_waits=0, drained=0)
        logger.info("GDN full-N overlap: enabled=1 waits=before-checkpoint-plan,"
                    "before-result-and-next-slot-plan; publication=forward-stream "
                    "host_wait=existing-copy_done; implicit_d2h_added=0 events=0")

    def begin(self, batch, selected):
        if self.pending is not None:
            raise RuntimeError("full-N next plan reached before prior result drain")
        self.serial += 1
        record = FullNBatchRecord(self, batch, self.serial, selected)
        batch.fulln_overlap_record = record
        if selected:
            # Keep the event alive with the batch until its publication finishes.
            record.prior_forward_done = self.runtime.fence_before_plan()
            self.stats["plan_events"] += 1
        return record

    def publish(self, record):
        if not record.sealed or not record.selected or record.plan is None:
            raise RuntimeError("full-N publication missing its immutable batch plan")
        if self.pending is not None or record.publication_done is not None:
            raise RuntimeError("full-N publication/slab reused before result drain")
        record.publication_done = self.runtime.published()
        self.pending = record
        self.stats["publication_events"] += 1

    def drain_before_planning(self, scheduler, pop_and_process):
        if self.pending is None:
            return False
        record = self.pending
        if (len(scheduler.result_queue) != 1
                or scheduler.result_queue[0][0].fulln_overlap_record is not record):
            raise RuntimeError("full-N pending result lost scheduler queue ownership")
        self.runtime.wait_for_result(record.publication_done)
        self.stats["result_waits"] += 1
        # The existing result-copy event supplies the only host wait, before
        # radix insertion/release; batch result processing calls before_result.
        pop_and_process()
        if not record.result_validated or not self.runtime.complete(record.publication_done):
            raise RuntimeError("full-N publication did not complete at result drain")
        record.consumed = True
        record.source.fulln_overlap_record = None
        record.plan = None
        self.pending = None
        self.stats["drained"] += 1
        if self.stats["drained"] == 1 or self.stats["drained"] % 100 == 0:
            logger.info("GDN full-N overlap: plan_events=%d publication_events=%d "
                        "result_waits=%d drained=%d implicit_d2h_added=0",
                        *self.stats.values())
        return True


def scheduler_controller(scheduler):
    # Non-GDN and non-full-N models have no controller. This is resolved once
    # at loop entry, with no environment reads in the decode hot path.
    pool = getattr(scheduler.req_to_token_pool, "factored_gdn_pool", None)
    controller = getattr(pool, "_agg_fulln_overlap", None)
    if controller is not None:
        controller.runtime.attach(scheduler)
    return controller
