"""Default-off k31 publication stream and request-owned decode admission fences.

Numerical work remains the original prewarmed graph. Its inputs are bound on
the producer stream; only replay moves, after the completed model forward.
Readers can drain an earlier dependency (notably PC's last prompt token).
"""
import copy
import logging
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from time import perf_counter_ns

import torch

from .gdn_pd_publication import slot_ids

logger = logging.getLogger(__name__)


class CudaRuntime:
    def __init__(self, device):
        self.device = device
        self.stream = torch.cuda.Stream(device=device, priority=0)

    def launch(self, operations, inputs):
        producer = torch.cuda.Event()
        producer.record(torch.cuda.current_stream(self.device))
        done = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(producer)
            try:
                for operation in operations:
                    operation()
            finally:
                for tensor in inputs:
                    if tensor is not None and tensor.is_cuda:
                        tensor.record_stream(self.stream)
                done.record(self.stream)
        return done

    @staticmethod
    def complete(done):
        return done.query()

    def wait(self, done, host):
        if host:
            done.synchronize()
            return None
        stream = torch.cuda.current_stream(self.device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
        stream.wait_event(done)
        end.record(stream)
        return start, end

    @staticmethod
    def elapsed_us(timing):
        start, end = timing
        return start.elapsed_time(end) * 1000 if end.query() else None


@dataclass
class Publication:
    requests: frozenset
    slots: frozenset
    operations: list
    inputs: list
    done: object = None


class CompressionSideStream:
    WAIT_BOUNDS_US = (0, 10, 100, 1000, 10000, 100000)

    def __init__(self, pool, runtime=None):
        self.pool, self.runtime = pool, runtime
        self.scope = None
        self.publications = []
        self.parked = []
        self.cache_queue = {}
        self.stats = dict(submitted=0, after_forward=0, early_reader=0,
                          skipped=0, admitted=0, reader_waits=0, bank_waits=0,
                          host_waits=0, wait_gpu_samples=0, wait_gpu_pending=0)
        self.wait_hist = [0] * (len(self.WAIT_BOUNDS_US) + 1)
        self.host_wait_hist = [0] * (len(self.WAIT_BOUNDS_US) + 1)
        self.timings = []
        self.failed = None

    def check(self):
        if self.failed is not None:
            raise RuntimeError("compression failed; request state is not publishable") from self.failed

    def get_runtime(self):
        if self.runtime is None:
            self.runtime = CudaRuntime(self.pool.a.device)
        return self.runtime

    @contextmanager
    def forward_scope(self, requests, slots, extend):
        self.check()
        if self.scope is not None:
            raise RuntimeError("nested compression forward ownership")
        # Compact workspace snapshots alias the publication graph's inputs.
        # Wait only at reuse, before the next prefill writes any of that bank.
        if extend and getattr(self.pool, "_agg_fulln_workspace_limits", None) is not None:
            self.wait_bank()
        self.wait_slots(slots)
        self.scope = requests, slots, extend
        try:
            yield
            self.launch_pending("after_forward")
        except BaseException as error:
            self.failed = error
            raise
        finally:
            self.scope = None

    def defer(self, replay, inputs):
        self.check()
        if self.scope is None or not self.scope[2]:
            raise RuntimeError("side compression needs a prefill ownership scope")
        requests, slots, _ = self.scope
        self.publications.append(Publication(requests, slots, [replay], list(inputs)))
        self.stats["submitted"] += 1

    def launch_pending(self, reason):
        self.check()
        for publication in self.publications:
            if publication.done is None:
                try:
                    publication.done = self.get_runtime().launch(publication.operations, publication.inputs)
                except BaseException as error:
                    self.failed = error
                    raise
                self.stats[reason] += 1

    def wait_bank(self):
        self.launch_pending("early_reader")
        for publication in self.publications:
            self.wait(publication, host=False, bank=True)

    def wait_slots(self, slots=None, *, host=False, forward_local=False):
        self.check()
        if forward_local and self.scope is not None:
            slots = self.scope[1]
        elif slots is not None:
            slots = slot_ids(slots)
        relevant = [p for p in self.publications if slots is None or not p.slots.isdisjoint(slots)]
        if not relevant:
            return
        self.launch_pending("early_reader")
        for publication in relevant:
            self.wait(publication, host=host)

    def wait(self, publication, *, host, bank=False):
        begin = perf_counter_ns()
        timing = self.get_runtime().wait(publication.done, host)
        self.stats["bank_waits" if bank else "reader_waits"] += 1
        if host:
            self.stats["host_waits"] += 1
            self.histogram(self.host_wait_hist, (perf_counter_ns() - begin) / 1000)
        elif timing is not None:
            self.timings.append(timing)

    def histogram(self, histogram, value):
        index = next((i for i, bound in enumerate(self.WAIT_BOUNDS_US) if value <= bound),
                     len(self.WAIT_BOUNDS_US))
        histogram[index] += 1

    def log(self, skipped=0, admitted=0):
        pending = []
        for timing in self.timings:
            elapsed = self.get_runtime().elapsed_us(timing)
            if elapsed is None:
                pending.append(timing)
            else:
                self.histogram(self.wait_hist, elapsed)
                self.stats["wait_gpu_samples"] += 1
        self.timings = pending
        self.stats["wait_gpu_pending"] = len(pending)
        logger.info("GDN compress side stream: skipped=%d admitted=%d skipped_total=%d "
                    "admitted_total=%d pending_publications=%d after_forward=%d early_reader=%d "
                    "event_wait_us_bounds=%s event_wait_hist=%s event_wait_pending=%d "
                    "host_wait_hist=%s reader_waits=%d bank_waits=%d",
                    skipped, admitted, self.stats["skipped"], self.stats["admitted"],
                    len(self.publications), self.stats["after_forward"], self.stats["early_reader"],
                    self.WAIT_BOUNDS_US, self.wait_hist, len(pending), self.host_wait_hist,
                    self.stats["reader_waits"], self.stats["bank_waits"])

    def request_ready(self, request_id):
        self.check()
        return all(p.done is not None and self.get_runtime().complete(p.done)
                   for p in self.publications if request_id in p.requests)

    def reserved_slots(self):
        self.check()
        reading = self.scope[1] if self.scope is not None else frozenset()
        return frozenset(slot for publication in self.publications
                         if publication.done is None or not self.get_runtime().complete(publication.done)
                         for slot in publication.slots if slot not in reading)

    def readiness(self, reqs, scheduler):
        ready = [self.request_ready(req.kv.req_pool_idx) for req in reqs]
        return self.consensus(ready, scheduler)

    @staticmethod
    def consensus(ready, scheduler):
        # All TP workers must construct the same batch even when their CUDA
        # events finish at different times. This collective uses the CPU group.
        if scheduler.tp_group.world_size > 1:
            values = torch.tensor(ready, dtype=torch.int32, device="cpu")
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MIN,
                                         group=scheduler.tp_group.cpu_group)
            ready = values.tolist()
        return ready

    def prepare_scheduler(self, scheduler, running):
        self.check()
        # Publication callbacks retain their request and prefix locks. Run the
        # original cache path before admitting that request into decode.
        queued = list(self.cache_queue.items())
        flags = self.readiness([item[1][0] for item in queued], scheduler) if queued else []
        for (rid, (req, callback)), ready in zip(queued, flags):
            if ready:
                callback()
                del self.cache_queue[rid]
        remaining, admitted, skipped = [], 0, 0
        for batch in self.parked:
            readiness = self.readiness(batch.reqs, scheduler)
            keep = [i for i, req in enumerate(batch.reqs) if readiness[i] or req.finished()]
            ready, waiting = split_batch(batch, keep)
            if not ready.is_empty():
                admitted += sum(not req.finished() for req in ready.reqs)
                if running.is_empty():
                    running = ready
                else:
                    running.merge_batch(ready)
            if not waiting.is_empty():
                remaining.append(waiting)
                skipped += len(waiting.reqs)
        self.parked = remaining
        self.stats["admitted"] += admitted
        self.stats["skipped"] += skipped
        # Only retire an event once no queued publisher or parked request owns
        # it. Slot-reader fences remain available until actual completion.
        if self.publications:
            completed = self.consensus([
                p.done is not None and self.get_runtime().complete(p.done)
                for p in self.publications], scheduler)
            self.publications = [p for p, done in zip(self.publications, completed) if not done]
        self.log(skipped=skipped, admitted=admitted)
        return running

    def park_pending(self, batch, scheduler):
        if batch.is_empty():
            return batch
        readiness = self.readiness(batch.reqs, scheduler)
        keep = [i for i, req in enumerate(batch.reqs) if readiness[i] or req.finished()]
        if len(keep) == len(batch.reqs):
            return batch
        ready, waiting = split_batch(batch, keep)
        waiting.fulln_overlap_record = None
        self.parked.append(waiting)
        # Pending requests remain allocated, with their native relay and
        # sampling state intact. They are never re-added to the prefill queue.
        skipped = len(waiting.reqs)
        self.stats["skipped"] += skipped
        self.log(skipped=skipped)
        return ready


def split_batch(batch, keep):
    """Use native filtering with independent penalty ownership for both views."""
    waiting = copy.copy(batch)
    waiting.reqs = batch.reqs[:]
    waiting.sampling_info = copy.copy(batch.sampling_info)
    original = batch.sampling_info.penalizer_orchestrator
    cloned = copy.copy(original)
    cloned.batch = waiting
    cloned.penalizers = {kind: copy.copy(value) for kind, value in original.penalizers.items()}
    for value in cloned.penalizers.values():
        value._orchestrator_ref = weakref.ref(cloned)
    waiting.sampling_info.penalizer_orchestrator = cloned
    remainder = [i for i in range(len(batch.reqs)) if i not in keep]
    # Native filter rebinds tensor fields; neither side mutates shared rows.
    waiting.filter_batch(keep_indices=remainder)
    batch.filter_batch(keep_indices=keep)
    return batch, waiting


def install(runner):
    from sglang.srt.environ import envs

    if not envs.SGLANG_GDN_COMPRESS_SIDE_STREAM.get():
        return
    pool = getattr(runner.req_to_token_pool, "factored_gdn_pool", None)
    args = runner.server_args
    if (pool is None or args.disaggregation_mode != "null" or args.disable_radix_cache
            or args.dp_size != 1 or args.pp_size != 1 or args.speculative_algorithm
            or getattr(args, "enable_hisparse", False) or args.is_embedding
            or getattr(runner.req_to_token_pool, "mamba_v2p_table", None) is not None
            or pool.cfg.init_method != "k31" or not pool.cfg.factored_prefix
            or not pool.host_sync_free or not pool.batch_prefill_final_copy
            or not pool.batch_prefill or pool.prefix_dense is not None
            or pool._k31_batch_graph is None or not pool._k31_batch_graph.warmed
            or envs.SGLANG_GDN_PD_BATCH_PUBLISH_DEFERRED.get()):
        raise ValueError("compression side stream requires AGG C+/PC+, radix, DP1 PP1, "
                         "HOST_SYNC_FREE and prewarmed whole-prefix k31 graph; no spec/HiSparse/unified pool")
    for name in ("SGLANG_GDN_K31_MIXED_EIGH", "SGLANG_GDN_PREFILL_RESTORE_GRAPH"):
        import os
        if name not in os.environ:
            raise ValueError("compression side stream launch must explicitly set " + name)
    if getattr(pool, "_compress_side_stream", None) is not None:
        return
    if (not getattr(pool, "_agg_prefill_enabled", False)
            or not getattr(pool, "_agg_prefill_graph", None)
            or not pool._agg_prefill_graph.warmed):
        raise ValueError("this compression side stream capsule admits C+ full-N only; "
                         "PC+ needs the deferred prompt-boundary capsule")
    controller = pool._compress_side_stream = CompressionSideStream(pool)
    original = runner.forward

    @wraps(original)
    def forward(forward_batch, *a, **kw):
        from sglang.srt.model_executor.runner import get_is_capture_mode

        batch = forward_batch
        if get_is_capture_mode():
            return original(batch, *a, **kw)
        if (getattr(batch, "spec_info", None) is not None or batch.forward_mode.is_mixed()
                or getattr(batch, "can_run_tbo", False)):
            raise RuntimeError("compression side stream rejects mixed/spec/TBO forward")
        ids = getattr(batch, "req_pool_indices_cpu", None)
        if ids is None:
            raise RuntimeError("compression side stream needs host request ownership")
        requests = frozenset(int(i) for i in ids)
        rp = runner.req_to_token_pool
        tensors = [rp.req_index_to_mamba_index_mapping[batch.req_pool_indices]]
        pingpong = getattr(rp, "req_index_to_mamba_ping_pong_track_buffer_mapping", None)
        if pingpong is not None:
            tensors.append(pingpong[batch.req_pool_indices])
        tensors.extend(getattr(batch, name, None) for name in (
            "mamba_track_indices", "mamba_cow_src_indices", "mamba_cow_dst_indices", "mamba_clear_indices"))
        slots = slot_ids(tensors)
        with controller.forward_scope(requests, slots, batch.forward_mode.is_extend()):
            return original(batch, *a, **kw)

    runner.forward = forward
    # The slot must not enter the free list before its publication finishes.
    allocator = getattr(runner.req_to_token_pool, "mamba_allocator", None)
    if allocator is not None:
        free, clear = allocator.free, allocator.clear

        @wraps(free)
        def fenced_free(indices):
            controller.wait_slots(indices, host=True)
            return free(indices)

        @wraps(clear)
        def fenced_clear():
            controller.wait_slots(host=True)
            return clear()

        allocator.free, allocator.clear = fenced_free, fenced_clear
    logger.info("GDN compress side stream installed: enabled=1 priority=0 "
                "decode=pending-skip complete-admit math=original-graph")


def scheduler_controller(scheduler):
    runner = scheduler.tp_worker.model_runner
    pool = getattr(runner.req_to_token_pool, "factored_gdn_pool", None)
    controller = getattr(pool, "_compress_side_stream", None)
    if controller is None:
        raise RuntimeError("compression side stream requested but runner was not installed")
    if not getattr(scheduler, "_compress_side_stream_attached", False):
        cache = scheduler.tree_cache
        unfinished, finished, insert = cache.cache_unfinished_req, cache.cache_finished_req, cache.insert

        @wraps(unfinished)
        def cache_unfinished(req, *a, **kw):
            rid = req.kv.req_pool_idx
            if any(rid in publication.requests for publication in controller.publications):
                if rid in controller.cache_queue:
                    raise RuntimeError("pending compression publisher overwritten")
                controller.cache_queue[rid] = req, lambda: unfinished(req, *a, **kw)
                return
            return unfinished(req, *a, **kw)

        @wraps(finished)
        def cache_finished(req, *a, **kw):
            controller.cache_queue.pop(req.kv.req_pool_idx, None)
            # Finishing may donate or release slots. Join before CPU ownership
            # publication as well as before device readers.
            owned = frozenset(slot for publication in controller.publications
                              if req.kv.req_pool_idx in publication.requests
                              for slot in publication.slots)
            controller.wait_slots(owned, host=True)
            return finished(req, *a, **kw)

        @wraps(insert)
        def cache_insert(params):
            states = getattr(params, "mamba_value", None)
            if states is not None:
                controller.wait_slots(states, host=True)
            return insert(params)

        cache.cache_unfinished_req, cache.cache_finished_req, cache.insert = cache_unfinished, cache_finished, cache_insert
        scheduler._compress_side_stream_attached = True
    return controller
