"""Opt-in P full-N publication after its forward, with mandatory reader joins.

Only one publication may own the graph's static inputs at a time. This does not
reorder the PD scheduler or wait indefinitely for a future request: an early
reader drains the queued work immediately.
"""
import json
import logging
import os
import time
from contextlib import contextmanager
from functools import wraps

import torch

logger = logging.getLogger(__name__)


class J1Split:
    """docs/223 §16 J1 (SGLANG_GDN_J1_SPLIT_PUBLICATION, AGG deferred publication only).

    The publication replays its normal half (request slots) and records an event before the tracked half (P-1
    checkpoint k31).  Label B: a plain decode forward joins only that event; every other reader still joins the
    whole publication.  Label A: today's whole join (same two graphs).  Mode 1 = B always; ab = per publication
    A B B A, logged to <TWINSTAR_CUDA_TIMELINE>/j1-<pid>.jsonl."""

    def __init__(self, mode):
        self.mode = mode
        self.j = 0
        self.sink = None
        self.prompt_only = None
        self.logged_active = False

    def next_label(self, rows):
        label = "B" if self.mode == "1" else "ABBA"[self.j % 4]
        self.j += 1
        directory = os.environ.get("TWINSTAR_CUDA_TIMELINE")
        if self.mode == "ab" and directory:
            if self.sink is None:
                self.sink = open(os.path.join(directory, f"j1-{os.getpid()}.jsonl"), "a", buffering=1)
            self.sink.write(json.dumps(dict(j=self.j - 1, l=label, t=time.time_ns(), rows=rows)) + "\n")
        return label

    def decode_only(self, runner, batch):
        """A forward that reads only its own request slots: a plain decode (graph replay or eager), no pool copy /
        clear, and no decode-time radix tracking (prompt-only state cache: the decode track mask is all-false)."""
        from sglang.srt.model_executor.runner import get_is_capture_mode

        if get_is_capture_mode():
            return False
        mode = getattr(batch, "forward_mode", None)
        if (mode is None or not mode.is_decode() or getattr(batch, "spec_info", None) is not None
                or getattr(batch, "can_run_tbo", False)):
            return False
        for name in ("mamba_cow_src_indices", "mamba_cow_dst_indices", "mamba_clear_indices"):
            tensor = getattr(batch, name, None)
            if tensor is not None and tensor.numel():
                return False
        if self.prompt_only is None:
            policy = getattr(runner.req_to_token_pool, "_prompt_only_state_cache", None)
            if policy is None:
                return False
            self.prompt_only = bool(policy)
            if not self.prompt_only:
                logger.warning("J1_SPLIT_PUBLICATION inert: decode-time radix tracking is on (no prompt-only state "
                               "cache), every decode keeps the whole join")
        return self.prompt_only


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
    def __init__(self, pool, runtime=None, *, offload_join=False, agg=False, pool_route=False):
        self.pool = pool
        self.agg = agg
        # K4.2 on the non-full-N AGG route: commit_extend_batched submits the
        # prompt-end k31 graph; the ModelRunner.forward hook launches and joins.
        self.pool_route = pool_route
        # Debug (docs/223 K4.2 checksum replay): runs on the side stream after
        # the factorization and before the done event.
        self.after_replay = None
        self.runtime = runtime
        self.pending = None
        self.ticket = None
        self.failed = None
        self.offload_join = offload_join
        self.forward_slots = None
        self.pending_slots = None
        self.stats = dict(submitted=0, launched=0, joins=0, early_reader=0, rows=0)
        self.stats.update(disjoint_forwards=0, dependent_forwards=0, transfer_fences=0)
        # docs/223 §16 J1 (AGG only; set by install)
        self.j1 = None
        self.ticket_normal = None
        self.label = None
        self.normal_joined = False

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
            if self.j1 is not None:
                # docs/223 §16 J1: one side-stream launch; the event between the halves is the decode join point
                normal_done = torch.cuda.Event()
                hook = self.after_replay

                def replay_split():
                    graph.run(self.pool, plan, states, *controls, eager=eager, policy=policy, split=True,
                              between=lambda: normal_done.record(torch.cuda.current_stream(self.pool.a.device)))
                    if hook is not None:
                        hook()
                self.label = self.j1.next_label(plan.slots.numel())
                self.ticket = self.runtime.launch_after_forward(replay_split, inputs)
                self.ticket_normal = normal_done
                self.normal_joined = False
            elif self.offload_join:
                # Snapshot the model's possibly reusable static outputs on
                # the producer stream. Main-stream order protects that source
                # reuse without waiting for the much longer factorization.
                self.ticket = graph.run(
                    self.pool, plan, states, *controls, eager=eager, policy=policy,
                    launch_replay=lambda replay: self.runtime.launch_after_forward(replay, inputs))
            elif self.after_replay is not None:
                hook = self.after_replay

                def replay_and_hook():
                    graph.run(self.pool, plan, states, *controls, eager=eager, policy=policy)
                    hook()
                self.ticket = self.runtime.launch_after_forward(replay_and_hook, inputs)
            else:
                self.ticket = self.runtime.launch_after_forward(
                    lambda: graph.run(self.pool, plan, states, *controls,
                                      eager=eager, policy=policy), inputs)
        except BaseException as error:
            self.failed = error
            # Retain all inputs and fail every subsequent reader closed.
            raise
        self.stats["launched"] += 1
        if self.agg and self.stats["launched"] == 1:
            logger.info("K42_AGG_PUBLISH_DEFERRED_ACTIVE rows=%d: prompt-end k31 replays on the "
                        "publication side stream after logits; next reader joins", plan.slots.numel())

    def join(self):
        if self.pending is None:
            return
        if self.ticket is None:
            self.stats["early_reader"] += 1
            self.start_after_forward()
        self.runtime.join(self.ticket)
        self.stats["joins"] += 1
        self.pending = self.ticket = None
        self.pending_slots = None
        self.ticket_normal = self.label = None
        self.normal_joined = False

    def join_decode(self):
        """docs/223 §16 J1: a plain decode forward (J1Split.decode_only) under label B waits for the normal half
        only -- its request slots, ring, prefix_valid, dense_required and the aligned checkpoints' final copies are
        written; the tracked half (P-1 checkpoint k31 into track slots this forward never reads) keeps running and
        the next other reader joins the whole publication.  Label A: today's whole join."""
        if self.failed is not None:
            raise RuntimeError("PD publication failed; state is not publishable") from self.failed
        if self.pending is None:
            return
        if self.ticket is None:
            self.stats["early_reader"] += 1
            self.start_after_forward()
        if self.label != "B" or self.ticket_normal is None:
            return self.join()
        if not self.normal_joined:
            self.runtime.join(self.ticket_normal)
            self.normal_joined = True
            self.stats["normal_joins"] = self.stats.get("normal_joins", 0) + 1
            if not self.j1.logged_active:
                self.j1.logged_active = True
                logger.info("J1_SPLIT_PUBLICATION_ACTIVE mode=%s: decode joined the normal half; the tracked "
                            "checkpoint k31 continues on the publication stream", self.j1.mode)

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
        if self.pending is not None:
            disjoint = (slots is not None and self.pending_slots is not None
                        and self.pending_slots.isdisjoint(slots))
            self.stats["disjoint_forwards" if disjoint else "dependent_forwards"] += 1
            if not disjoint:
                self.join()
        self.forward_slots = slots
        try:
            yield
        finally:
            self.forward_slots = None

    def reserved_slots(self):
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
        if publication is not None and publication.offload_join:
            batch = args[0] if args else kwargs["forward_batch"]
            with publication.forward_scope(forward_slot_ids(runner, batch)):
                return original(*args, **kwargs)
        if (publication is not None and publication.j1 is not None and publication.pending is not None
                and getattr(pool, "_pside_deferred_commit", None) is None
                and publication.j1.decode_only(runner, args[0] if args else kwargs["forward_batch"])):
            publication.join_decode()
        else:
            pool.pside_join()
        return original(*args, **kwargs)

    runner.forward = forward


def install(pool, runner, *, agg=False):
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_schedule

    if agg:
        # K4.2 (docs/223): AGG keeps the unconditional pre-forward join; the
        # PD-only slot-closure offload is never enabled here.
        pool._pd_batch_publication = PDBatchPublication(pool, agg=True)
        from .gdn_prefill_batch_graph import SPLIT_FLAG, split_mode

        mode = split_mode()
        if mode != "0":
            pool._pd_batch_publication.j1 = J1Split(mode)
            logger.info("J1_SPLIT_PUBLICATION_INSTALLED %s=%s: two-graph publication (normal, tracked); plain "
                        "decodes under label B join the normal half only", SPLIT_FLAG, mode)
        install_forward_join(runner, pool)
        logger.info("AGG full-N deferred publication enabled: after-forward event, side priority=0; "
                    "join before every runner.forward and pool reader; join_offload=False")
        return
    offload = envs.SGLANG_GDN_PD_PUBLISH_JOIN_OFFLOAD.get()
    if offload and (not pool.host_sync_free or not get_schedule().disable_overlap_schedule):
        raise ValueError("PD publish join offload requires HOST_SYNC_FREE=1 and disabled overlap scheduling")
    pool._pd_batch_publication = PDBatchPublication(pool, offload_join=offload)
    install_forward_join(runner, pool)
    logger.info("PD full-N deferred publication enabled: after-forward event, "
                "side priority=0; join_offload=%s; early reader drains immediately", offload)


AGG_DEFERRED_FLAG = "SGLANG_GDN_AGG_BATCH_PUBLISH_DEFERRED"


def install_runner_hook():
    """Class-level ModelRunner.forward hook for the non-full-N AGG route (K4.2).

    Joins every pending publication before a forward (decode replay included)
    and launches a submitted one after the forward returns, i.e. after logits
    and before sampling. Inert for runners whose pool has no pool-route
    publication.
    """
    from sglang.srt.model_executor.model_runner import ModelRunner

    if getattr(ModelRunner, "_k42_pool_route_hook", False):
        return
    original = ModelRunner.forward

    @wraps(original)
    def forward(self, *args, **kwargs):
        pool = getattr(getattr(self, "req_to_token_pool", None), "factored_gdn_pool", None)
        publication = getattr(pool, "_pd_batch_publication", None)
        if publication is None or not publication.pool_route:
            return original(self, *args, **kwargs)
        pool.pside_join()
        output = original(self, *args, **kwargs)
        publication.start_after_forward()
        return output

    ModelRunner.forward = forward
    ModelRunner._k42_pool_route_hook = True


def install_pool_route(pool):
    """Called at the end of FactoredGDNPool.prewarm_k31_batch_graph (switch on)."""
    import os

    from sglang.srt.runtime_context import get_disagg, get_memory

    if getattr(pool, "_pd_batch_publication", None) is not None:
        return  # full-N contract owns publication on this pool
    # docs/223 §3.5 correction: on the original N-1 route the boundary decode
    # (final-token decode in the same forward) reads the k31 factors, so the
    # k31 cannot leave that forward. Deferral needs the full-N route.
    logger.warning("%s=1 has no effect on the N-1 + boundary route (the boundary decode reads the k31 "
                   "factors in the same forward); use SGLANG_GDN_AGG_FULLN_PREFILL=1", AGG_DEFERRED_FLAG)
    return
    if get_disagg().disaggregation_mode != "null":
        return
    if get_memory().enable_hierarchical_cache:
        logger.warning("%s=1 ignored with --enable-hierarchical-cache: k31 publication stays "
                       "on the forward stream", AGG_DEFERRED_FLAG)
        return
    pool._pd_batch_publication = PDBatchPublication(pool, agg=True, pool_route=True)
    if os.environ.get("SGLANG_GDN_K42_CHECKSUM") == "1":
        from .gdn_prefill_agg_contract import FactorChecksum

        pool._k42_checksum = FactorChecksum(pool)
    install_runner_hook()
    logger.info("AGG k31 deferred publication enabled (pool route): after-forward event, side priority=0; "
                "join before every runner.forward and pool reader; join_offload=False")


def commit_k31_pool_route(pool, graph, plan, controls, *, eager, policy):
    """The k31 branch of commit_extend_batched when a pool-route publication exists."""
    import os

    publication = pool._pd_batch_publication
    flags = None
    control = os.environ.get("SGLANG_GDN_K42_CONTROL")
    if control:
        try:
            with open(control) as stream:
                flags = stream.read().strip()
        except OSError:
            flags = ""
    checksum = getattr(pool, "_k42_checksum", None)
    if checksum is not None and flags is not None and "c" not in flags:
        checksum = None
    written = (plan.slots, controls[0])
    rows = plan.slots.numel()
    if flags is not None and flags.startswith("0"):
        # Debug in-process A/B: today's forward-stream replay for this forward.
        graph.run(pool, plan, plan.pending, *controls, eager=eager, policy=policy)
        if checksum is not None:
            checksum.drain()
            checksum.enqueue(written, "forward", rows)
        return
    if checksum is not None:
        checksum.drain()
        publication.after_replay = lambda: checksum.enqueue(written, "side", rows)
    else:
        publication.after_replay = None
    publication.submit(graph, plan, plan.pending, controls, eager=eager, policy=policy)

