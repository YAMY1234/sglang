"""Opt-in T-only publication. Live F states stay on the original stream.

A publication belongs to physical slots, not a request lifetime: cancellation
must not forget an in-flight writer. Readers/writers fence only intersecting
slots; events remain visible to every reader stream until CUDA reports done.
"""
from dataclasses import dataclass, field
import logging
import threading

import torch
import triton
import triton.language as tl

from sglang.srt.utils.graph_capture import graph_capture_lock

logger = logging.getLogger(__name__)


def tensor_version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        # Serving inference tensors have no version counter. Only immutable
        # per-forward index outputs receive hints; never reusable graph inputs.
        return None


def remember_slots(tensor, values):
    # Hint comes from an EXISTING CPU metadata snapshot, never an extra D2H.
    tensor._gdn_slot_snapshot = (tensor_version(tensor), tuple(int(x) for x in values))
    return tensor


def slot_hint(tensor):
    if tensor is None:
        return ()
    snapshot = getattr(tensor, "_gdn_slot_snapshot", None)
    if snapshot is not None and snapshot[0] == tensor_version(tensor):
        return snapshot[1]
    if not tensor.is_cuda:
        return tuple(tensor.reshape(-1).tolist())
    return None


@dataclass(eq=False)
class Publication:
    event: object
    slots: tuple
    waited_streams: set = field(default_factory=set)


class SlotPublications:
    def __init__(self):
        self.pending = {}
        self.lock = threading.RLock()
        self.waits = 0
        self.readbacks = 0
        self.publications = 0

    def reap(self):
        with self.lock:
            for record in set(self.pending.values()):
                if record.event.query():
                    for slot in record.slots:
                        if self.pending.get(slot) is record:
                            del self.pending[slot]

    def publish(self, slots, event):
        with self.lock:
            # Caller has fenced the old generation before issuing the new writer.
            record = Publication(event, tuple(s for s in slots if s >= 0))
            for slot in record.slots:
                self.pending[slot] = record
            self.publications += 1
            return record

    def wait(self, slots, stream):
        with self.lock:
            self.reap()
            selected = {self.pending[s] for s in slots if s in self.pending}
            for record in selected:
                if stream.cuda_stream not in record.waited_streams:
                    stream.wait_event(record.event)
                    record.waited_streams.add(stream.cuda_stream)
                    self.waits += 1
            return len(selected)

    def wait_tensors(self, tensors, device):
        with self.lock:
            self.reap()
            if not self.pending:
                return 0
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("tracked slot reader must be fenced before capture")
        slots = []
        for tensor in tensors:
            if tensor is None or tensor.numel() == 0:
                continue
            values = slot_hint(tensor)
            if values is None:
                # Never hold the registry lock over D2H. That copy may wait for
                # a producer which needs the lock to register its publication.
                # Count the rare exact readback; no whole-batch wait substitutes.
                with self.lock:
                    self.readbacks += 1
                values = tuple(tensor.reshape(-1).tolist())
                remember_slots(tensor, values)
            slots.extend(values)
        # Re-read the latest generation after any host copy, under the lock.
        return self.wait(slots, torch.cuda.current_stream(device))


def wait_slots(pool, *tensors):
    registry = getattr(pool, "_tracked_slot_publications", None)
    if registry is not None:
        return registry.wait_tensors(tensors, pool.a.device)
    return 0


def alias_reason(live, tracked, src, dst):
    live, tracked, src, dst = ([x for x in values if x >= 0]
                               for values in (live, tracked, src, dst))
    if not tracked or len(set(tracked)) != len(tracked):
        return "empty_or_aliased_tracked"
    if set(tracked).intersection(live + src + dst):
        return "tracked_aliases_final"
    return None


@triton.jit
def _invalidate_masked(VALID, SLOTS, MASK, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    active = tl.load(MASK + i, i < N, other=0) != 0
    slot = tl.load(SLOTS + i, (i < N) & active, other=-1).to(tl.int64)
    tl.store(VALID + slot, 0, (i < N) & active & (slot >= 0))


@triton.jit
def _count_failures(COUNT, OK, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = tl.load(OK + i, i < N, other=1)
    delta = tl.sum(((i < N) & ~ok).to(tl.int64), 0)
    tl.atomic_add(COUNT, delta, sem="relaxed")


def count_failures(count, ok):
    # The original count += sum was a shared read/modify/write across F/T.
    # Integer atomic accounting changes no factor arithmetic or fallback choice.
    _count_failures[(triton.cdiv(ok.numel(), 256),)](count, ok, ok.numel(), 256)


def invalidate_masked(valid, slots, mask):
    # Prompt-only decode has a false mask. Do not issue a read/modify/write to
    # its cached T slot while another stream publishes prefix_valid=1.
    if slots.numel():
        _invalidate_masked[(1,)](valid, slots, mask, slots.numel(),
                                triton.next_power_of_2(slots.numel()))


def eligible(pool, *, role, enabled, include_tail):
    return bool(enabled and role == "null" and not include_tail
                and pool.cfg.init_method == "k31" and pool.cfg.strict_chunk
                and pool.cfg.factored_prefix and pool._generic_prompt_only_state_cache
                and pool.prefix_dense is None and pool.warm_v is None
                and pool.prefix_layer_count() == len(pool.layer_ids))


class TrackedSlotSide:
    def __init__(self, pool, whole):
        self.pool, self.whole = pool, whole
        self.registry = pool._tracked_slot_publications
        self.stream = torch.cuda.Stream(device=pool.a.device, priority=0)
        # Distinct capture streams also isolate stream-keyed library workspaces
        # (e.g. cuBLAS), in addition to the explicit graph allocator arenas.
        self.capture_streams = tuple(torch.cuda.Stream(device=pool.a.device) for _ in range(2))
        self.arenas = (torch.cuda.graph_pool_handle(), torch.cuda.graph_pool_handle())
        self.entries = ({}, {})
        # Both banks have their own T inputs/controls. Normal inputs remain
        # ordered on the producer stream. F/T temporaries use distinct arenas.
        self.done = [None, None]
        self.bank = 0
        self.fallbacks = {}
        self.replayed = 0
        self.buffer_waits = 0

    def prewarm(self, eager):
        from .gdn_prefill_batch_graph import BatchBuffers
        current = torch.cuda.current_stream(self.pool.a.device)
        before = torch.cuda.memory_allocated(self.pool.a.device)
        # Two bounded T slabs, with bucket views; no sum of per-bucket buffers.
        sizes = {key[1] for key in self.whole.entries if key[1] is not None and not key[-1]}
        if not sizes:
            return
        capacity = max(sizes)
        shared = [{key: value for key, value in self.whole.shared.items()
                   if key[0] == "normal"} for _ in range(2)]
        for bank in range(2):
            slab = self.pool.a.new_zeros((len(self.pool.layer_ids), capacity,
                                          self.pool.hv, self.pool.v, self.pool.k))
            shared[bank].update({("tracked", size): [layer[:size] for layer in slab.unbind(0)]
                                 for size in sizes})
        for key, (base, _) in self.whole.entries.items():
            if key[1] is None or key[-1]:
                continue
            for bank in range(2):
                # Neither T bank aliases the collector/whole-graph input slab.
                # The next full-N forward may snapshot into that slab immediately.
                buffers = BatchBuffers(self.pool, base.batch, base.tracked_batch,
                                       shared[bank], include_tail=False)
                graphs = []
                for branch, arena, capture_stream in zip(
                    ("normal", "tracked"), self.arenas, self.capture_streams
                ):
                    capture_stream.wait_stream(current)
                    with torch.cuda.stream(capture_stream):
                        buffers.evaluate(eager, branch=branch)
                    current.wait_stream(capture_stream)
                    graph = torch.cuda.CUDAGraph()
                    with graph_capture_lock, torch.cuda.graph(
                        graph, stream=capture_stream, pool=arena,
                        capture_error_mode="thread_local"
                    ):
                        buffers.evaluate(eager, branch=branch)
                    graphs.append(graph)
                self.entries[bank][key] = (buffers, *graphs)
        torch.cuda.synchronize(self.pool.a.device)
        logger.info("GDN tracked slot side: enabled=1 role=null banks=2 "
                    "signatures=%d side_priority=%d readers=slot-intersection "
                    "T_capacity=%d extra_retained_bytes=%d",
                    len(self.entries[0]), self.stream.priority, capacity,
                    torch.cuda.memory_allocated(self.pool.a.device) - before)

    def protect_bank(self, bank):
        event = self.done[bank]
        if event is not None and not event.query():
            torch.cuda.current_stream(self.pool.a.device).wait_event(event)
            self.buffer_waits += 1

    def fallback(self, reason):
        self.fallbacks[reason] = self.fallbacks.get(reason, 0) + 1
        total = sum(self.fallbacks.values())
        if total == 1 or total % 100 == 0:
            logger.info("GDN tracked slot side fallback: reason=%s counts=%s", reason, self.fallbacks)
        # Both T input banks are private; the ordinary graph cannot overwrite them.
        return False

    def run(self, key, plan, states, track_slots, final_src, final_dst):
        if torch.cuda.is_current_stream_capturing():
            return self.fallback("capture")
        if not getattr(plan, "tracked_side_allowed", True):
            return self.fallback("mixed_or_tbo")
        if key not in self.entries[0]:
            return self.fallback("no_split_signature")
        values = tuple(slot_hint(t) for t in (plan.slots, track_slots, final_src, final_dst))
        if any(v is None for v in values):
            return self.fallback("missing_host_slots")
        reason = alias_reason(*values)
        if reason is not None:
            return self.fallback(reason)
        current = torch.cuda.current_stream(self.pool.a.device)
        self.registry.wait([s for rows in values for s in rows], current)
        bank = self.bank
        self.protect_bank(bank)
        buffers, final_graph, tracked_graph = self.entries[bank][key]
        buffers.bind(plan, states, track_slots, final_src, final_dst)
        final_graph.replay()  # F and the final-slot copy remain on the main stream.
        bound = torch.cuda.Event()
        bound.record(current)
        self.stream.wait_event(bound)
        done = torch.cuda.Event()  # Immutable event per generation, never re-record.
        with self.registry.lock:
            with torch.cuda.stream(self.stream):
                tracked_graph.replay()  # T factorization AND its store/publication.
                done.record(self.stream)
            self.done[bank] = done
            self.registry.publish(values[1], done)
        self.bank = 1 - bank
        self.replayed += 1
        if self.replayed == 1 or self.replayed % 100 == 0:
            logger.info("GDN tracked slot side: replays=%d slots_pending=%d "
                        "slot_waits=%d reader_d2h=%d buffer_waits=%d fallbacks=%s",
                        self.replayed, len(self.registry.pending), self.registry.waits,
                        self.registry.readbacks, self.buffer_waits, self.fallbacks)
        return True


def install(pool, whole, eager):
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_disagg
    enabled = envs.SGLANG_GDN_TRACKED_SLOT_SIDE_STREAM.get()
    if not enabled:
        return
    role = get_disagg().disaggregation_mode
    if not eligible(pool, role=role, enabled=enabled, include_tail=whole.include_tail):
        logger.warning("GDN tracked slot side rejected: requires AGG k31 prompt-only "
                       "complete factored prefix, without exact-tail graph; role=%s", role)
        return
    if pool._tracked_slot_publications is None:
        pool._tracked_slot_publications = SlotPublications()
    side = TrackedSlotSide(pool, whole)
    side.prewarm(eager)
    whole.tracked_side = side
