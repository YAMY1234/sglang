"""Opt-in P full-N publication after its forward, with mandatory reader joins.

Only one publication may own the graph's static inputs at a time. This does not
reorder the PD scheduler or wait indefinitely for a future request: an early
reader drains the queued work immediately.
"""
import logging
from contextlib import contextmanager
from functools import wraps

import torch

logger = logging.getLogger(__name__)


class CudaPublicationRuntime:
    def __init__(self, device):
        self.device = device
        # CUDA's lowest priority is 0. A default main stream may be equal;
        # ordering comes from events, not from a priority/preemption assumption.
        self.stream = torch.cuda.Stream(device=device, priority=0)

    def launch_after_forward(self, operation, inputs):
        forward_done = torch.cuda.Event()
        forward_done.record(torch.cuda.current_stream(self.device))
        done = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(forward_done)
            try:
                operation()
            finally:
                # Even failed enqueue paths must not release live inputs while
                # already submitted device work is still using their storage.
                for tensor in inputs:
                    if tensor is not None and tensor.is_cuda:
                        tensor.record_stream(self.stream)
                done.record(self.stream)
        return done

    def join(self, done):
        torch.cuda.current_stream(self.device).wait_event(done)


class PDBatchPublication:
    def __init__(self, pool, runtime=None, *, offload_join=False):
        self.pool = pool
        self.runtime = runtime
        self.pending = None
        self.ticket = None
        self.failed = None
        self.offload_join = offload_join
        self.forward_slots = None
        self.pending_slots = None
        self.records = None
        self.stats = dict(submitted=0, launched=0, joins=0, early_reader=0, rows=0)
        self.stats.update(disjoint_forwards=0, dependent_forwards=0, transfer_fences=0)

    def submit(self, graph, plan, states, controls, *, eager, policy):
        if self.offload_join:
            # Called after the next trunk has been enqueued. Only now order
            # reuse of the publication graph's shared static input buffers.
            self.join()
        if self.pending is not None:
            raise RuntimeError("PD publication inputs reused before reader join")
        if graph.include_tail or not graph.warmed:
            raise RuntimeError("PD deferred publication requires a prewarmed full-N graph")
        states, controls = tuple(states), tuple(controls)
        self.pending = graph, plan, states, controls, eager, policy
        self.pending_slots = self.forward_slots
        self.stats["submitted"] += 1
        self.stats["rows"] += plan.slots.numel()

    def start_after_forward(self):
        if self.failed is not None:
            raise RuntimeError("PD publication failed; state is not publishable") from self.failed
        if self.pending is None or self.ticket is not None:
            return
        graph, plan, states, controls, eager, policy = self.pending
        if self.runtime is None:
            self.runtime = CudaPublicationRuntime(self.pool.a.device)
        inputs = [t for pair in states for t in pair]
        inputs.extend(controls)
        inputs.extend((plan.slots, plan.ring_dst, plan.dense_required_after_commit))
        try:
            if self.offload_join:
                # Snapshot the model's possibly reusable static outputs on
                # the producer stream. Main-stream order protects that source
                # reuse without waiting for the much longer factorization.
                self.ticket = graph.run(
                    self.pool, plan, states, *controls, eager=eager, policy=policy,
                    launch_replay=lambda replay: self.runtime.launch_after_forward(replay, inputs))
            else:
                self.ticket = self.runtime.launch_after_forward(
                    lambda: graph.run(self.pool, plan, states, *controls,
                                      eager=eager, policy=policy), inputs)
        except BaseException as error:
            self.failed = error
            # Retain all inputs and fail every subsequent reader closed.
            raise
        self.stats["launched"] += 1

    def join(self):
        if self.records is not None:
            self.records.wait_slots(None, "forward" if self.records.in_forward else "schedule")
        if self.pending is None:
            return
        if self.ticket is None:
            self.stats["early_reader"] += 1
            self.start_after_forward()
        self.runtime.join(self.ticket)
        self.stats["joins"] += 1
        self.pending = self.ticket = None
        self.pending_slots = None

    def transfer_event(self):
        """Immutable per-publication event; a worker may retain it after join."""
        if self.failed is not None:
            raise RuntimeError("PD publication failed; state is not publishable") from self.failed
        if self.pending is not None and self.ticket is None:
            self.stats["early_reader"] += 1
            self.start_after_forward()
        self.stats["transfer_fences"] += 1
        return self.ticket

    def join_reader(self, slots=None, *, forward_local=False):
        if self.failed is not None:
            raise RuntimeError("PD publication failed; state is not publishable") from self.failed
        if self.records is not None:
            selected = self.forward_slots if forward_local and self.forward_slots is not None else slots
            lane = "forward" if self.records.in_forward else "schedule"
            self.records.wait_slots(selected, lane)
            return
        if not self.offload_join or self.pending_slots is None:
            return self.join()
        # The runner snapshots a conservative closure before metadata/COW or
        # graph replay. Pool operations within it touch only those batch slots.
        if forward_local and self.forward_slots is not None:
            slots = self.forward_slots
        elif slots is not None:
            slots = slot_ids(slots)
        if slots is None or not self.pending_slots.isdisjoint(slots):
            self.join()

    @contextmanager
    def forward_scope(self, slots):
        if self.failed is not None:
            raise RuntimeError("PD publication failed; state is not publishable") from self.failed
        if self.forward_slots is not None:
            raise RuntimeError("nested PD publication forward scope")
        if self.records is not None:
            if self.records.in_forward:
                raise RuntimeError("nested PD publication record forward")
            self.records.in_forward = True
        self.forward_slots = slots
        try:
            if self.records is not None:
                self.records.wait_slots(slots, "forward")
            if self.pending is not None:
                disjoint = (slots is not None and self.pending_slots is not None
                            and self.pending_slots.isdisjoint(slots))
                self.stats["disjoint_forwards" if disjoint else "dependent_forwards"] += 1
                if not disjoint:
                    self.join()
            yield
        finally:
            self.forward_slots = None
            if self.records is not None:
                self.records.in_forward = False

    def reserved_slots(self):
        if self.records is not None:
            return self.records.reserved_slots()
        return self.pending_slots if self.offload_join and self.pending is not None else None


def slot_ids(tensors):
    """Read indices only, never unpublished state; outside CUDA capture."""
    if isinstance(tensors, (set, frozenset)):
        return tensors
    if isinstance(tensors, torch.Tensor):
        tensors = (tensors,)
    tensors = [t.reshape(-1).to(dtype=torch.long) for t in tensors if t is not None]
    if not tensors:
        return frozenset()
    return frozenset(int(s) for s in torch.cat(tensors).cpu().tolist() if s >= 0)


def forward_slot_ids(runner, batch):
    # Unsupported dispatches retain the global barrier. Full-N prefill and
    # ordinary decode (including CUDA replay) have this complete slot closure.
    from sglang.srt.model_executor.runner import get_is_capture_mode

    if get_is_capture_mode():
        return None
    mode = getattr(batch, "forward_mode", None)
    if getattr(runner.req_to_token_pool, "mamba_v2p_table", None) is not None:
        # Unified-pool relocation has additional ownership not represented by
        # ordinary request slots; retain the original barrier there.
        return None
    if (mode is None or not (mode.is_decode() or (
            mode.is_extend() and getattr(batch, "_pfactor_agg_contract", False)))
            or mode.is_mixed() or getattr(batch, "spec_info", None) is not None
            or getattr(batch, "_pfactor_legacy_mixed", False)
            or getattr(batch, "can_run_tbo", False)
            or getattr(batch, "tbo_split_seq_index", None) is not None
            or getattr(batch, "tbo_parent_token_range", None) is not None):
        return None
    mapping = runner.req_to_token_pool.req_index_to_mamba_index_mapping
    tensors = [mapping[batch.req_pool_indices]]
    tensors.extend(getattr(batch, name, None) for name in (
        "mamba_track_indices", "mamba_cow_src_indices", "mamba_cow_dst_indices",
        "mamba_clear_indices"))
    return slot_ids(tensors)


def install_forward_join(runner, pool):
    # CUDA decode replay bypasses layer_tensors. The opt-in route checks the
    # complete batch slot closure before either replay or eager dispatch;
    # the original route retains its unconditional pre-forward barrier.
    original = runner.forward

    @wraps(original)
    def forward(*args, **kwargs):
        publication = getattr(pool, "_pd_batch_publication", None)
        if publication is not None and publication.records is not None:
            from sglang.srt.model_executor.runner import get_is_capture_mode

            if get_is_capture_mode():
                return original(*args, **kwargs)
        if publication is not None and publication.offload_join:
            batch = args[0] if args else kwargs["forward_batch"]
            with publication.forward_scope(forward_slot_ids(runner, batch)):
                submitted_before = publication.stats["submitted"]
                result = original(*args, **kwargs)
                if publication.records is not None:
                    publication.records.after_forward(batch, submitted_before)
                return result
        pool.pside_join()
        return original(*args, **kwargs)

    runner.forward = forward


def install(pool, runner):
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_schedule
    from .gdn_pd_overlap import enable_records, protocol_ready

    offload = envs.SGLANG_GDN_PD_PUBLISH_JOIN_OFFLOAD.get()
    if offload and (not pool.host_sync_free or (
            not get_schedule().disable_overlap_schedule and not protocol_ready(runner))):
        raise ValueError("PD publish join offload requires HOST_SYNC_FREE=1 and disabled overlap scheduling")
    pool._pd_batch_publication = PDBatchPublication(pool, offload_join=offload)
    enable_records(pool._pd_batch_publication, runner.req_to_token_pool)
    install_forward_join(runner, pool)
    logger.info("PD full-N deferred publication enabled: after-forward event, "
                "side priority=0; join_offload=%s; early reader drains immediately", offload)
