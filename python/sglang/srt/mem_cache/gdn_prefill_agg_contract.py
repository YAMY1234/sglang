"""P48 uses the native AGG full-prompt state; P31 keeps its boundary adapter."""
import logging
import os
import inspect
from functools import wraps

import torch

from .gdn_prefill_batch_graph import BatchCollector, PrefillBatchGraph

FLAG = "SGLANG_GDN_PREFILL_AGG_CONTRACT"
AGG_FLAG = "SGLANG_GDN_AGG_FULLN_PREFILL"
logger = logging.getLogger(__name__)
_TRACK_FIELDS = ("mamba_last_track_idx", "mamba_next_track_idx", "mamba_last_track_seqlen")


def enabled():
    if os.environ.get(FLAG, "1") != "1":
        return False
    marker = os.environ.get(FLAG + "_FILE")
    return not marker or os.path.exists(marker)


def agg_enabled():
    from sglang.srt.environ import envs

    return envs.SGLANG_GDN_AGG_FULLN_PREFILL.get()


def eligible(batch):
    mode = batch.forward_mode
    if (not mode.is_extend() or mode.is_mixed()
            or getattr(batch, "_pfactor_legacy_mixed", False)
            or getattr(batch, "spec_info", None) is not None
            or getattr(batch, "can_run_tbo", False)
            or getattr(batch, "tbo_split_seq_index", None) is not None):
        return False
    if hasattr(batch, "reqs"):
        rows = len(batch.reqs)
        lengths = [req.extend_range.length for req in batch.reqs]
    else:
        rows, lengths = batch.batch_size, batch.extend_seq_lens_cpu
        if lengths is None or sum(lengths) != batch.input_ids.shape[0]:
            return False
    return 0 < rows <= 16 and len(lengths) == rows and all(int(n) > 1 for n in lengths)


def agg_eligible(batch):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner import get_is_capture_mode

    return (not get_is_capture_mode() and batch.forward_mode == ForwardMode.EXTEND
            and getattr(batch, "input_embeds", None) is None
            and getattr(batch, "tbo_parent_token_range", None) is None
            and eligible(batch))


def install_contracts(forward_cls, schedule_cls, backend_cls, handoff_cls, *, agg_mode=False):
    """Select the track extent before backend metadata and ring ownership exist."""
    if getattr(forward_cls, "_pfactor_agg_contract_installed", False):
        return
    legacy_track = schedule_cls._mamba_radix_cache_v2_req_prepare_for_extend
    native_track = inspect.unwrap(legacy_track)
    legacy_init = forward_cls.init_new.__func__
    legacy_metadata = backend_cls.init_forward_metadata
    legacy_send = handoff_cls.before_send
    native_send = inspect.unwrap(legacy_send)
    if not agg_mode and (native_track is legacy_track or native_send is legacy_send):
        raise ValueError("P48 AGG contract requires the frozen factor-only wrappers")

    def select(batch):
        return (agg_enabled() and agg_eligible(batch) if agg_mode
                else enabled() and eligible(batch))

    def prepare_track(batch, req, selected):
        # Native DUET also applies N-1 inside prompt_p_extent. Unwrapping the
        # external factor-only adapter alone does not select full-N tracking.
        req._pfactor_agg_contract = selected
        return (native_track if selected else legacy_track)(batch, req)

    @wraps(legacy_track)
    def track(self, req):
        selected = select(self)
        before = tuple(getattr(req.kv, name) for name in _TRACK_FIELDS)
        result = prepare_track(self, req, selected)
        req._pfactor_track_before = self, before, selected
        return result

    @classmethod
    @wraps(legacy_init)
    def initialize(cls, batch, model_runner, **kwargs):
        result = legacy_init(cls, batch, model_runner, **kwargs)
        selected = select(result)
        changes = []
        for i, req in enumerate(batch.reqs):
            snapshot = getattr(req, "_pfactor_track_before", None)
            if snapshot is not None:
                source, before, previous = snapshot
                if previous != selected:
                    for name, value in zip(_TRACK_FIELDS, before, strict=True):
                        setattr(req.kv, name, value)
                    entry = prepare_track(source, req, selected)
                    changes.append((i, entry))
                del req._pfactor_track_before
            req._pfactor_agg_contract = selected
            if selected:
                req.factored_prefill_boundary_steps = 0
        if changes:
            # TBO/mixed decisions can arrive after checkpoint preparation.
            for name, field in (("mamba_track_mask", "track_mask"),
                                ("mamba_track_indices", "track_index"),
                                ("mamba_track_seqlens", "track_seqlen")):
                source = getattr(batch, name).clone()
                for i, entry in changes:
                    source[i] = getattr(entry, field)
                setattr(batch, name, source)
                setattr(result, name, source.to(model_runner.device))
        result._pfactor_agg_contract = selected
        if not agg_mode and result.forward_mode.is_extend():
            result.pd_factor_only_full_batch = True
            result._pfactor_legacy_mixed = result.forward_mode.is_mixed()
            if result._pfactor_legacy_mixed:
                # Eager normalizes MIXED to EXTEND after checkpoint selection.
                result.twinstar_prompt_final = [
                    req.extend_range.end >= len(req.origin_input_ids)
                    for req in batch.reqs
                ]
        return result

    @wraps(legacy_metadata)
    def metadata(self, batch):
        if not getattr(batch, "_pfactor_agg_contract", False):
            return legacy_metadata(self, batch)
        previous = getattr(batch, "pd_factor_only_full_batch", False)
        batch.pd_factor_only_full_batch = False
        try:
            return legacy_metadata(self, batch)
        finally:
            batch.pd_factor_only_full_batch = previous

    @wraps(legacy_send)
    def before_send(self, req):
        if not getattr(req, "_pfactor_agg_contract", False):
            return legacy_send(self, req)
        req.factored_prefill_boundary_steps = 0
        return native_send(self, req)

    schedule_cls._mamba_radix_cache_v2_req_prepare_for_extend = track
    forward_cls.init_new = initialize
    backend_cls.init_forward_metadata = metadata
    if not agg_mode:
        handoff_cls.before_send = before_send
    forward_cls._pfactor_agg_contract_installed = True


def install(runner):
    owner = runner.model
    role = runner.server_args.disaggregation_mode
    agg_mode = role == "null"
    if agg_mode:
        if not agg_enabled():
            return
    elif (role != "prefill" or os.environ.get(FLAG, "1") != "1"
          or os.environ.get("TWINSTAR_PD_FACTOR_ONLY_TAIL") != "1"
          or getattr(owner, "pd_shallow_role", None) == "prefill"):
        return
    if getattr(owner, "_pfactor_agg_installed", False):
        return
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
    from sglang.srt.disaggregation.state_handoff import FactorStateHandoff
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from sglang.srt.runtime_context import get_schedule

    pool = runner.req_to_token_pool.factored_gdn_pool
    if (getattr(owner, "n_layers", None) != 48
            or list(pool.layer_ids) != [i for i in range(48) if i % 4 != 3]
            or not pool.cfg.strict_chunk
            or not pool.cfg.factored_prefix or pool.cfg.init_method != "k31"
            or pool.prefix_dense is not None or not pool.batch_prefill
            or (not agg_mode and not getattr(owner, "_exact_tail_installed", False))):
        raise ValueError("AGG contract requires the full-depth strict k31 P48 recipe")
    if owner.fullstack:
        from sglang.srt.models.flash_next_duet.pd_shallow_install import factor_only_contract

        factor_only_contract(owner)
    elif owner.emitters or getattr(owner, "twinstar", None) is not None:
        raise ValueError("AGG contract requires native DUET P48 or legacy factor-only P48")
    if agg_mode:
        if (not owner.fullstack or not get_schedule().disable_overlap_schedule
                or runner.server_args.is_embedding or runner.server_args.pp_size != 1
                or runner.server_args.speculative_algorithm
                or os.environ.get("SGLANG_GDN_PREFILL_COMMIT_GRAPH") != "1"
                or os.environ.get("TWINSTAR_PD_FACTOR_ONLY_TAIL") == "1"
                or os.environ.get("SGLANG_GDN_PREFILL_EXACT_TAIL_BATCH") == "1"):
            raise ValueError("AGG full-N requires native P48 generation PP1, isolated scheduling, "
                             "COMMIT_GRAPH=1 and no PD factor-only/exact-tail adapter")
    # The flag-off external model delegates here; wrapper depth is not an ABI.
    native_forward = owner.model.forward
    install_contracts(ForwardBatch, ScheduleBatch, GDNAttnBackend, FactorStateHandoff,
                      agg_mode=agg_mode)
    legacy_forward = owner.forward

    if agg_mode:
        legacy_prepare = owner.prepare_forward_batch

        @wraps(legacy_prepare)
        def prepare(batch):
            if getattr(batch, "_pfactor_agg_contract", False) and not get_is_capture_mode():
                # #17 otherwise defers the plan for its N-1 sub-batch. This
                # route needs exactly one full-N plan before entering layers.
                batch._twinstar_defer_factor_plan = False
                batch._twinstar_prefill_eligible = False
                return
            return legacy_prepare(batch)

        owner.prepare_forward_batch = prepare

    @wraps(legacy_forward)
    def forward(input_ids, positions, forward_batch, *args, **kwargs):
        if (get_is_capture_mode()
                or not getattr(forward_batch, "_pfactor_agg_contract", False)):
            return legacy_forward(input_ids, positions, forward_batch, *args, **kwargs)
        if not (agg_eligible(forward_batch) if agg_mode else eligible(forward_batch)):
            raise RuntimeError("AGG batch contract changed after checkpoint planning")
        linear = get_attn_backend().linear_attn_backend
        plan = linear.forward_metadata.factored_extend
        if plan is None or getattr(pool, "_exact_tail_transaction", None) is not None:
            raise RuntimeError("AGG prefill needs one native full-N plan without a tail transaction")
        with BatchCollector(pool, plan, graph=pool._agg_prefill_graph):
            output = native_forward(input_ids, positions, forward_batch, *args, **kwargs)
            if linear.forward_metadata.factored_extend is not plan:
                raise RuntimeError("native AGG forward replaced its full-N state plan")
        # The graph publishes r, with no boundary update or manual count edit.
        forward_batch.factored_prefill_boundary_steps = 0
        owner._agg_fulln_prefills += 1
        if owner._agg_fulln_prefills % 500 == 1:
            logger.info("GDN full-N prefill: role=%s forwards=%d rows=%d phase=0 count=%d",
                        role, owner._agg_fulln_prefills, forward_batch.batch_size, pool.cfg.r)
        return output

    owner.forward = forward
    owner._pfactor_agg_installed = True
    owner._agg_fulln_prefills = 0
    pool._agg_prefill_enabled = True
    logger.info("GDN P48 AGG contract installed: role=%s full-N, deferred commit, phase=0 count=%d; "
                "original fallback for mixed/TBO/empty-prefix batches", role, pool.cfg.r)


def prewarm(pool):
    if not getattr(pool, "_agg_prefill_enabled", False):
        return
    from .gdn_factored_pool import (
        ORTH_METHOD, ORTH_WARPS_OVERRIDE, factorize_dense, factorize_layers,
    )
    tail_graph = getattr(pool, "_prefill_batch_graph", None)
    graph = PrefillBatchGraph(include_tail=False,
                              shared=None if tail_graph is None else tail_graph.shared)
    graph.prewarm(pool, eager=factorize_layers,
                  policy=(ORTH_METHOD, ORTH_WARPS_OVERRIDE, factorize_dense))
    pool._agg_prefill_graph = graph
