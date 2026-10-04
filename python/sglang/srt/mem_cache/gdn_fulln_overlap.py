"""Batch-owned full-N state and local prefill fences for AGG overlap.

No tensor-to-host reads are introduced here. Publication stays on the forward
stream. The scheduler consumes results early only at a CPU ownership reader. Decode
state reads stay ordered on the original forward stream; result FIFO processing
keeps its normal post-launch position.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
from time import perf_counter_ns
from collections import defaultdict

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

    def published(self):
        import torch

        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(self.device))
        return done

    def wait_for_result(self, done):
        import torch

        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(self.schedule_stream)
        self.schedule_stream.wait_event(done)
        end.record(self.schedule_stream)
        return start, end

    @staticmethod
    def elapsed_us(timing):
        start, end = timing
        return start.elapsed_time(end) * 1000 if end.query() else None

    @staticmethod
    def complete(done):
        return done.query()


def request_slots(batch):
    """CPU request-slot ownership keys; never materialize device mamba indices.

    Active requests exclusively own their recurrent and ping-pong destinations.
    Published radix ancestors are protected by the existing prefix locks/WAR;
    an unpublished full-N checkpoint cannot be looked up in radix yet.
    """
    return frozenset(req.kv.req_pool_idx for req in batch.reqs
                     if req.kv.req_pool_idx is not None) if batch is not None else frozenset()


def iteration_kind(batch):
    if batch is None:
        return "idle"
    mode = batch.forward_mode
    if mode.is_mixed():
        return "mixed"
    if mode.is_decode():
        return "decode"
    return "fulln_prefill" if batch.fulln_overlap_record is not None else "other_prefill"


class FullNOverlap:
    POINTS = ("before-checkpoint-plan", "before-result-and-next-slot-plan")

    def __init__(self, device, runtime=None):
        self.runtime = runtime if runtime is not None else CudaBoundaryRuntime(device)
        self.serial = 0
        self.publications = {}
        self.stats = dict(plan_events=0, publication_events=0, result_waits=0, drained=0)
        self.iterations = 0
        self.wait_stats = defaultdict(lambda: dict(checks=0, wait_count=0,
            pending_sum=0, pending_max=0, intersecting_sum=0, enqueue_us=0.0,
            result_host_us=0.0, gpu_wait_us=0.0, gpu_samples=0, gpu_pending=0))
        self._iteration_samples = []
        self._timings = []
        self._wait_details = []
        self.scheduler = None
        self.pop_and_process = None
        self.drained_this_iteration = False
        logger.info("GDN full-N overlap: enabled=1 waits=before-checkpoint-plan,"
                    "before-result-and-next-slot-plan; publication=forward-stream "
                    "scope=actual-cpu-owner-readers decode=forward-stream-ordered host_wait=existing-copy_done; "
                    "implicit_d2h_added=0 events=0")

    @property
    def pending(self):
        # Compatibility/read-only diagnostic view. Ownership is per publication.
        return next(iter(self.publications.values()), None)

    def intersecting(self, slots):
        return [record for record in self.publications.values()
                if any(slot in slots for _, slot in record.identities)]

    def wait_for_slots(self, point, slots):
        records = self.intersecting(slots) if self.publications else []
        sample = dict(point=point, pending=len(self.publications), records=len(records),
                      enqueue_us=0.0, result_host_us=0.0, timings=[])
        for record in records:
            start = perf_counter_ns()
            timing = self.runtime.wait_for_result(record.publication_done)
            sample["enqueue_us"] += (perf_counter_ns() - start) / 1000
            sample["timings"].append(timing)
        self._iteration_samples.append(sample)
        return records, sample

    def start_iteration(self, scheduler, pop_and_process):
        self.scheduler, self.pop_and_process = scheduler, pop_and_process
        self.drained_this_iteration = False
        # The queued ScheduleBatch.copy owns the prefill result record. The
        # mutable scheduler batch can become decode or merge into another batch;
        # it must not copy the previous publication into that next result.
        last = scheduler.last_batch
        if last is not None and last.fulln_overlap_record is not None:
            record = last.fulln_overlap_record
            if self.publications.get(record.serial) is not record:
                raise RuntimeError("full-N last batch lost its publication")
            last.fulln_overlap_record = None

    def read_owners(self, reqs, *, point=None):
        if not self.publications:
            return False
        slots = frozenset(req.kv.req_pool_idx for req in reqs
                          if req.kv.req_pool_idx is not None)
        if not self.intersecting(slots):
            return False
        if self.scheduler is None or self.pop_and_process is None:
            raise RuntimeError("full-N checkpoint ownership needs prior result drain")
        return self.drain_before_planning(self.scheduler, self.pop_and_process,
            touched_slots=slots, point=point)

    def begin(self, batch, selected):
        # Both selected and fallback checkpoint plans mutate Req track ownership.
        # A decode step has no checkpoint plan and never calls this reader gate.
        self.read_owners(batch.reqs, point=self.POINTS[0])
        self.serial += 1
        record = FullNBatchRecord(self, batch, self.serial, selected)
        batch.fulln_overlap_record = record
        return record

    def publish(self, record):
        if not record.sealed or not record.selected or record.plan is None:
            raise RuntimeError("full-N publication missing its immutable batch plan")
        if record.publication_done is not None or self.intersecting(
                frozenset(slot for _, slot in record.identities)):
            raise RuntimeError("full-N publication slot reused before result drain")
        # Shared graph inputs are rebound on this same forward stream. Disjoint
        # publications may remain in the FIFO; their batch plans are immutable.
        record.publication_done = self.runtime.published()
        self.publications[record.serial] = record
        self.stats["publication_events"] += 1

    def result_consumed(self, batch):
        record = batch.fulln_overlap_record
        if record is None:
            return
        if (self.publications.get(record.serial) is not record
                or not record.result_validated
                or not self.runtime.complete(record.publication_done)):
            raise RuntimeError("full-N publication did not complete at result drain")
        record.consumed = True
        if record.source.fulln_overlap_record is record:
            record.source.fulln_overlap_record = None
        record.plan = None
        del self.publications[record.serial]
        self.stats["drained"] += 1

    def drain_before_planning(self, scheduler, pop_and_process, touched_slots=(), point=None):
        if not self.publications:
            return False
        # Callers supply the slots of a real CPU ownership reader, never the
        # union of every currently running/last batch. GPU decode dependencies
        # are already ordered on the publication's forward stream.
        point = self.POINTS[1] if point is None else point
        matches, sample = self.wait_for_slots(point, touched_slots)
        if not matches:
            return False
        self.stats["result_waits"] += len(matches)
        if point == self.POINTS[0]:
            self.stats["plan_events"] += len(matches)
        # The native overlap queue has one prior result at this boundary. Never
        # skip an unrelated FIFO head to consume a later publication.
        if (len(scheduler.result_queue) != 1
                or not any(scheduler.result_queue[0][0].fulln_overlap_record is r for r in matches)):
            raise RuntimeError("full-N pending result lost scheduler queue ownership")
        record = scheduler.result_queue[0][0].fulln_overlap_record
        start = perf_counter_ns()
        pop_and_process()
        sample["result_host_us"] += (perf_counter_ns() - start) / 1000
        if not record.consumed:
            raise RuntimeError("full-N publication did not complete at result drain")
        self.drained_this_iteration = True
        return True

    def note_iteration(self, batch):
        kind = iteration_kind(batch)
        self.iterations += 1
        samples, self._iteration_samples = self._iteration_samples, []
        # Every iteration reports both points, including explicit zero waits.
        for point in self.POINTS:
            key = point, kind
            row = self.wait_stats[key]
            row["checks"] += 1
            for sample in samples:
                if sample["point"] != point:
                    continue
                row["wait_count"] += sample["records"]
                row["pending_sum"] += sample["pending"]
                row["pending_max"] = max(row["pending_max"], sample["pending"])
                row["intersecting_sum"] += sample["records"]
                row["enqueue_us"] += sample["enqueue_us"]
                row["result_host_us"] += sample["result_host_us"]
                if sample["records"]:
                    self._wait_details.append((self.iterations, point, kind,
                                               sample["pending"], sample["records"]))
                for timing in sample["timings"]:
                    self._timings.append((key, timing))
                    row["gpu_pending"] += 1
        if self.iterations == 1 or self.iterations % 100 == 0:
            self.log_waits()

    def log_waits(self):
        unfinished = []
        for key, timing in self._timings:
            elapsed = self.runtime.elapsed_us(timing)
            if elapsed is None:
                unfinished.append((key, timing))
            else:
                row = self.wait_stats[key]
                row["gpu_wait_us"] += elapsed
                row["gpu_samples"] += 1
                row["gpu_pending"] -= 1
        self._timings = unfinished
        for (point, kind), row in sorted(self.wait_stats.items()):
            logger.info("GDN full-N wait: point=%s kind=%s iterations=%d "
                        "checks=%d wait_count=%d pending_sum=%d pending_max=%d "
                        "intersecting_sum=%d enqueue_us=%.3f result_host_us=%.3f "
                        "gpu_wait_us=%.3f gpu_samples=%d gpu_pending=%d "
                        "implicit_d2h_added=0", point, kind, self.iterations,
                        *row.values())
        if self._wait_details:
            logger.info("GDN full-N wait samples: fields=iteration,point,kind,pending,intersecting values=%s",
                        self._wait_details)
            self._wait_details.clear()
        logger.info("GDN full-N overlap: plan_events=%d publication_events=%d "
                    "result_waits=%d drained=%d pending=%d implicit_d2h_added=0",
                    *self.stats.values(), len(self.publications))


def scheduler_controller(scheduler):
    # Non-GDN and non-full-N models have no controller. This is resolved once
    # at loop entry, with no environment reads in the decode hot path.
    pool = getattr(scheduler.req_to_token_pool, "factored_gdn_pool", None)
    controller = getattr(pool, "_agg_fulln_overlap", None)
    if controller is not None:
        controller.runtime.attach(scheduler)
    return controller
