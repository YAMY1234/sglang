"""Opt-in P31 exact-tail publication using the existing PD stream/fence.

No P48/full-N marker is installed: shallow live factors retain r+1/r, tracked
prefixes retain r, and D still receives the original N-1 boundary contract.
"""
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from functools import wraps
from types import MappingProxyType, SimpleNamespace

import torch

from .gdn_pd_publication import (
    PDBatchPublication, forward_slot_ids, slot_ids,
)

logger = logging.getLogger(__name__)
FLAG = "SGLANG_GDN_PD_SHALLOW_PUBLISH_DEFERRED"
GDN_IDS = tuple(i for i in range(48) if i % 4 != 3)
TAIL_IDS = tuple(i for i in GDN_IDS if i < 31)


@dataclass(frozen=True)
class ShallowSelection:
    slots: frozenset
    request_indices: tuple


@dataclass(frozen=True)
class ShallowPlan:
    slots: torch.Tensor
    ring_dst: torch.Tensor
    dense_required_after_commit: object
    exact_tail_inputs: object


def select_batch(runner, batch):
    """Snapshot this full P31 batch before metadata/COW plans can mutate it."""
    from sglang.srt.model_executor.runner import get_is_capture_mode

    mode = batch.forward_mode
    if (get_is_capture_mode() or not mode.is_extend() or mode.is_mixed()
            or not 1 <= batch.batch_size <= 16
            or getattr(batch, "_pfactor_legacy_mixed", False)
            or getattr(batch, "_pfactor_agg_contract", False)
            or getattr(batch, "can_run_tbo", False)
            or getattr(batch, "tbo_split_seq_index", None) is not None
            or getattr(batch, "tbo_parent_token_range", None) is not None
            or getattr(batch, "spec_info", None) is not None
            or getattr(runner.req_to_token_pool, "mamba_v2p_table", None) is not None):
        return None
    lengths = getattr(batch, "extend_seq_lens_cpu", None)
    ids = getattr(batch, "req_pool_indices_cpu", None)
    finals = getattr(batch, "twinstar_prompt_final", None)
    if (lengths is None or ids is None or finals is None
            or len(lengths) != batch.batch_size or len(ids) != batch.batch_size
            or len(finals) != batch.batch_size or any(int(n) <= 1 for n in lengths)):
        # Empty-prefix/single-token special tails use the old synchronous path.
        return None
    if isinstance(ids, torch.Tensor) and ids.device.type != "cpu":
        return None
    host_ids = tuple(map(int, ids))
    if len(set(host_ids)) != len(host_ids):
        return None
    pool = runner.req_to_token_pool
    indices = [pool.req_index_to_mamba_index_mapping[batch.req_pool_indices]]
    indices.extend(getattr(batch, name, None) for name in (
        "mamba_track_indices", "mamba_cow_src_indices", "mamba_cow_dst_indices",
        "mamba_clear_indices"))
    return ShallowSelection(slot_ids(indices), host_ids)


class ShallowPublication(PDBatchPublication):
    """Only the payload/eligibility differs from PR47; its lifecycle is reused."""

    def selected(self, batch):
        selection = getattr(batch, "_pd_shallow_publication_selection", None)
        return isinstance(selection, ShallowSelection)

    def submit_exact(self, transaction, *, eager, policy):
        if not self.selected(transaction.batch):
            return False
        if transaction.plan is None or transaction.empty_slots is not None:
            return False
        graph = self.pool._prefill_batch_graph
        if (not graph.warmed or not graph.include_tail
                or tuple(layer.layer_id for layer in self.pool._exact_tail_layers) != TAIL_IDS
                or tuple(self.pool.layer_ids) != GDN_IDS
                or len(transaction.states) != len(GDN_IDS)):
            raise RuntimeError("shallow deferred publication requires the complete 36/24 exact-tail graph")
        if transaction.tail_expected and set(transaction.tails) != set(TAIL_IDS):
            raise RuntimeError("shallow deferred publication is missing native tail inputs")
        selection = transaction.batch._pd_shallow_publication_selection
        if tuple(transaction.request_indices.tolist()) != selection.request_indices:
            raise RuntimeError("shallow publication request snapshot changed")
        # The original exact-tail validation already read these destinations;
        # keep that host snapshot instead of adding a second control D2H.
        targets = transaction._pd_shallow_targets
        if not targets.issubset(selection.slots):
            raise RuntimeError("shallow publication destinations escape the pre-forward slot closure")
        self.join()  # same one-bank ordering as PR47, after the next trunk
        if self.pending is not None:
            raise RuntimeError("shallow publication reused a live bank")
        clone = lambda tensor: None if tensor is None else tensor.clone()
        plan = ShallowPlan(
            transaction.controls[0][1], clone(transaction.plan.ring_dst),
            clone(transaction.plan.dense_required_after_commit),
            MappingProxyType({lid: tuple(values) for lid, values in transaction.tails.items()}),
        )
        # Original transaction.__exit__ clears its containers. This queue owns
        # separate containers plus the already-owned dense/tail/control tensors.
        states = tuple(tuple(pair) for pair in transaction.states)
        controls = tuple(item[1] for item in transaction.controls[1:])
        self.pending = graph, plan, states, controls, eager, policy
        self.pending_slots = selection.slots
        self.stats["submitted"] += 1
        self.stats["rows"] += plan.slots.numel()
        return True

    def start_after_forward(self):
        before = self.stats["launched"]
        try:
            result = super().start_after_forward()
            if self.stats["launched"] != before and (before == 0 or self.stats["launched"] % 100 == 0):
                logger.info("PD shallow publication: submitted=%d launched=%d rows=%d "
                            "joins=%d disjoint_forwards=%d dependent_forwards=%d transfer_fences=%d",
                            *(self.stats[key] for key in ("submitted", "launched", "rows", "joins",
                              "disjoint_forwards", "dependent_forwards", "transfer_fences")))
            return result
        finally:
            # The static graph reads the producer-bound copies. Retain the
            # private tail snapshots on the publication stream as well, just
            # as PR47 retains its normal/tracked source tensors.
            if self.pending is not None and self.runtime is not None and (
                    self.stats["launched"] != before or self.failed is not None):
                for values in self.pending[1].exact_tail_inputs.values():
                    for tensor in values:
                        if tensor.is_cuda:
                            tensor.record_stream(self.runtime.stream)


class ShallowTransferFence:
    """PR47 event/count fence plus the unchanged shallow validator and epoch."""

    def __init__(self, publication_done, handler, request_pool, req, expected, record=None):
        from sglang.srt.disaggregation.state_handoff import FactorTransferFence

        self.handler, self.request_pool = handler, request_pool
        self.record = record
        self.count = handler.original.pool.count
        slot = req.kv.mamba_pool_idx
        if slot is None:
            raise RuntimeError("shallow P/D send without a mamba slot")
        # Request slots may be scalar tensors. The CPU identity and ownership
        # epoch are frozen; sender lifetime retains the real slot and boundary.
        self.request_index = int(req.kv.req_pool_idx)
        self.generation = int(request_pool.req_generation[self.request_index])
        self.req = SimpleNamespace(kv=SimpleNamespace(mamba_pool_idx=slot))
        index_ndim = slot.ndim if isinstance(slot, torch.Tensor) else 0
        expected = expected.view(len(GDN_IDS), *([1] * (self.count.ndim - 2 + index_ndim)))
        self.inner = FactorTransferFence(publication_done, self.count, slot, expected)
        self.validated = False

    def wait(self, producer_done):
        if self.validated:
            return
        if int(self.request_pool.req_generation[self.request_index]) != self.generation:
            raise RuntimeError("shallow transfer request generation changed before send")
        if self.record is not None:
            self.record.producer_done.synchronize()
            if not self.record.producer_done.query():
                raise RuntimeError("PD publication producer incomplete before send")
            self.record.owner.stats["transfer_waits"] += 1
            self.record.owner.edges.append((self.record.batch_id, "transfer", "producer+publication+queue"))
        self.inner.wait(producer_done)
        if int(self.request_pool.req_generation[self.request_index]) != self.generation:
            raise RuntimeError("shallow transfer request generation changed while waiting")
        # Keep the helper's actual valid + per-layer mixed-phase check. Run it
        # on the worker's stream, never behind a subsequent P trunk on default.
        from sglang.srt.disaggregation.state_handoff import _transfer_local

        context = nullcontext()
        if self.count.is_cuda:
            context = torch.cuda.stream(_transfer_local.streams[self.count.device])
        with context:
            self.handler.before_send(self.req)
        self.validated = True


class ShallowHandoff:
    def __init__(self, original, pool, request_pool, publication):
        self.original, self.pool = original, pool
        self.request_pool, self.publication = request_pool, publication
        self.expected = torch.tensor([pool.cfg.r + int(lid < 31) for lid in GDN_IDS],
                                     dtype=pool.count.dtype, device=pool.count.device)

    def before_send(self, req):
        from sglang.srt.disaggregation.state_handoff import supports_state_handoff_fence

        sender = getattr(req, "disagg_kv_sender", None)
        record = (self.publication.records.for_request(req)
                  if self.publication.records is not None else None)
        if not supports_state_handoff_fence(sender):
            if record is None:
                self.publication.join()
            else:
                record.producer_done.synchronize()
                if record.publication_done is not None:
                    record.publication_done.synchronize()
            return self.original.before_send(req)
        sender.set_state_handoff_fence(ShallowTransferFence(
            record.publication_done if record is not None else self.publication.transfer_event(),
            self.original, self.request_pool, req, self.expected, record=record))

    def prepare_receive(self, req):
        return self.original.prepare_receive(req)

    def commit_receive(self, req):
        return self.original.commit_receive(req)


def install(runner):
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_schedule
    from sglang.srt.disaggregation.state_handoff import HandoffKind
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from twinstar_sgl.pd_shallow import SplitBoundaryPhase

    if not envs.SGLANG_GDN_PD_SHALLOW_PUBLISH_DEFERRED.get():
        return False
    owner = runner.model
    rp = runner.req_to_token_pool
    pool = getattr(rp, "factored_gdn_pool", None)
    fs = getattr(owner, "fullstack", None)
    if (runner.server_args.disaggregation_mode != "prefill"
            or getattr(owner, "pd_shallow_role", None) != "prefill"
            or not fs or not fs.get("prefill_layer_trim")
            or fs.get("gdn_rank", 0) <= 0 or fs.get("gdn_every", 0) <= 0
            or pool is None):
        return False  # S/P/C and all D/AGG paths remain untouched.
    if getattr(pool, "_pd_shallow_publication", None) is not None:
        return True
    import os

    required = ("SGLANG_GDN_PREFILL_EXACT_TAIL_BATCH", "SGLANG_GDN_PREFILL_COMMIT_GRAPH",
                "SGLANG_GDN_PD_BATCH_PUBLISH_DEFERRED", "SGLANG_GDN_PD_PUBLISH_JOIN_OFFLOAD")
    args = runner.server_args
    if (any(os.environ.get(key) != "1" for key in required)
            or not pool.host_sync_free or not get_schedule().disable_overlap_schedule
            or args.pp_size != 1 or args.speculative_algorithm or args.is_embedding
            or os.environ.get("TWINSTAR_PD_FACTOR_ONLY_TAIL") == "1"
            or not getattr(owner, "_exact_tail_installed", False)
            or not pool.cfg.strict_chunk or not pool.cfg.factored_prefix
            or pool.cfg.init_method != "k31" or pool.prefix_dense is not None
            or owner.n_layers != 48 or list(owner.p_layer_ids) != list(range(31))
            or list(owner.emitter_ids) != list(range(31, 48))
            or not pool.batch_prefill or tuple(pool.layer_ids) != GDN_IDS
            or tuple(layer.layer_id for layer in pool._exact_tail_layers) != TAIL_IDS
            or not pool._prefill_batch_graph.warmed
            or not pool._prefill_batch_graph.include_tail):
        raise ValueError("shallow deferred publication requires isolated native PC P31/17 PP1, "
                         "EXACT_TAIL_BATCH=1 COMMIT_GRAPH=1 PD_BATCH_PUBLISH_DEFERRED=1 "
                         "PD_PUBLISH_JOIN_OFFLOAD=1 HOST_SYNC_FREE=1, and the warmed 36/24 graph")
    original_handler = rp.pd_state_handoffs[HandoffKind.STATE_FACTOR]
    if not isinstance(original_handler, SplitBoundaryPhase):
        raise TypeError("shallow deferred publication requires the original SplitBoundaryPhase")
    if getattr(pool, "_pd_batch_publication", None) is not None:
        raise RuntimeError("shallow publication cannot replace a P48 publication transaction")
    publication = ShallowPublication(pool, offload_join=True)
    from .gdn_pd_overlap import enable_records

    enable_records(publication, rp)
    pool._pd_shallow_publication = pool._pd_batch_publication = publication
    rp.pd_state_handoffs[HandoffKind.STATE_FACTOR] = ShallowHandoff(
        original_handler, pool, rp, publication)
    original = runner.forward

    @wraps(original)
    def forward(*args, **kwargs):
        batch = args[0] if args else kwargs["forward_batch"]
        if get_is_capture_mode():
            return original(*args, **kwargs)
        selection = select_batch(runner, batch)
        batch._pd_shallow_publication_selection = selection
        slots = selection.slots if selection is not None else forward_slot_ids(runner, batch)
        try:
            with publication.forward_scope(slots):
                submitted_before = publication.stats["submitted"]
                result = original(*args, **kwargs)
                publication.start_after_forward()
                if publication.records is not None:
                    publication.records.after_forward(batch, submitted_before)
                return result
        finally:
            batch._pd_shallow_publication_selection = None

    runner.forward = forward
    logger.info("PD shallow deferred publication installed: P31/17, GDN36/tail24, "
                "checkpoint=N-1; live=r+1/r; after-forward side replay; "
                "slot readers and sender publication+producer fence; default warmup supported")
    return True
