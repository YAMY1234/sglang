"""P48 uses the native AGG full-prompt state; P31 keeps its boundary adapter."""
import logging
import os
import inspect
from functools import wraps

import torch

from .gdn_prefill_batch_graph import BatchCollector, PrefillBatchGraph
from .gdn_prefill_recipe_guard import warn_install_rejection

FLAG = "SGLANG_GDN_PREFILL_AGG_CONTRACT"
logger = logging.getLogger(__name__)
_TRACK_FIELDS = ("mamba_last_track_idx", "mamba_next_track_idx", "mamba_last_track_seqlen")


def enabled():
    if os.environ.get(FLAG, "1") != "1":
        return False
    marker = os.environ.get(FLAG + "_FILE")
    return not marker or os.path.exists(marker)


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


def install_contracts(forward_cls, schedule_cls, backend_cls, handoff_cls):
    """Select the track extent before backend metadata and ring ownership exist."""
    if getattr(forward_cls, "_pfactor_agg_contract_installed", False):
        return
    legacy_track = schedule_cls._mamba_radix_cache_v2_req_prepare_for_extend
    native_track = inspect.unwrap(legacy_track)
    legacy_init = forward_cls.init_new.__func__
    legacy_metadata = backend_cls.init_forward_metadata
    legacy_send = handoff_cls.before_send
    native_send = inspect.unwrap(legacy_send)
    if native_track is legacy_track or native_send is legacy_send:
        raise ValueError("P48 AGG contract requires the frozen factor-only wrappers")

    @wraps(legacy_track)
    def track(self, req):
        selected = enabled() and eligible(self)
        before = tuple(getattr(req.kv, name) for name in _TRACK_FIELDS)
        req._pfactor_agg_contract = selected
        result = (native_track if selected else legacy_track)(self, req)
        req._pfactor_track_before = self, before, selected
        return result

    @classmethod
    @wraps(legacy_init)
    def initialize(cls, batch, model_runner, **kwargs):
        result = legacy_init(cls, batch, model_runner, **kwargs)
        selected = enabled() and eligible(result)
        changes = []
        for i, req in enumerate(batch.reqs):
            snapshot = getattr(req, "_pfactor_track_before", None)
            if snapshot is not None:
                source, before, previous = snapshot
                if previous != selected:
                    for name, value in zip(_TRACK_FIELDS, before, strict=True):
                        setattr(req.kv, name, value)
                    req._pfactor_agg_contract = selected
                    entry = (native_track if selected else legacy_track)(source, req)
                    changes.append((i, entry))
                del req._pfactor_track_before
            req._pfactor_agg_contract = selected
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
        if result.forward_mode.is_extend():
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
    handoff_cls.before_send = before_send
    forward_cls._pfactor_agg_contract_installed = True


@warn_install_rejection("full-N")
def install(runner):
    owner = runner.model
    if (os.environ.get(FLAG, "1") != "1"
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

    pool = runner.req_to_token_pool.factored_gdn_pool
    if (runner.server_args.disaggregation_mode != "prefill"
            or getattr(owner, "n_layers", None) != 48
            or len(pool.layer_ids) != 36 or not pool.cfg.strict_chunk
            or not pool.cfg.factored_prefix or pool.cfg.init_method != "k31"
            or pool.prefix_dense is not None or not pool.batch_prefill
            or not getattr(owner, "_exact_tail_installed", False)):
        raise ValueError("AGG contract requires the full-depth strict k31 P48 recipe")
    if owner.fullstack:
        from sglang.srt.models.flash_next_duet.pd_shallow_install import factor_only_contract

        factor_only_contract(owner)
    elif owner.emitters or getattr(owner, "twinstar", None) is not None:
        raise ValueError("AGG contract requires native DUET P48 or legacy factor-only P48")
    # The flag-off external model delegates here; wrapper depth is not an ABI.
    native_forward = owner.model.forward
    install_contracts(ForwardBatch, ScheduleBatch, GDNAttnBackend, FactorStateHandoff)
    legacy_forward = owner.forward

    @wraps(legacy_forward)
    def forward(input_ids, positions, forward_batch, *args, **kwargs):
        if not getattr(forward_batch, "_pfactor_agg_contract", False):
            return legacy_forward(input_ids, positions, forward_batch, *args, **kwargs)
        if not eligible(forward_batch):
            raise RuntimeError("AGG batch contract changed after checkpoint planning")
        linear = get_attn_backend().linear_attn_backend
        plan = linear.forward_metadata.factored_extend
        if plan is None or getattr(pool, "_exact_tail_transaction", None) is not None:
            raise RuntimeError("AGG prefill needs one native full-N plan without a tail transaction")
        with BatchCollector(pool, plan, graph=pool._agg_prefill_graph):
            output = native_forward(input_ids, positions, forward_batch, *args, **kwargs)
            if linear.forward_metadata.factored_extend is not plan:
                raise RuntimeError("native AGG forward replaced its full-N state plan")
            return output

    owner.forward = forward
    owner._pfactor_agg_installed = True
    pool._agg_prefill_enabled = True
    logger.info("GDN P48 AGG contract installed: full-N, deferred commit, phase=0 count=%d; "
                "B3.2 fallback for mixed/TBO/empty-prefix batches", pool.cfg.r)


def prewarm(pool):
    if not getattr(pool, "_agg_prefill_enabled", False):
        return
    from .gdn_factored_pool import (
        ORTH_METHOD, ORTH_WARPS_OVERRIDE, factorize_dense, factorize_layers,
    )
    graph = PrefillBatchGraph(include_tail=False, shared=pool._prefill_batch_graph.shared)
    graph.prewarm(pool, eager=factorize_layers,
                  policy=(ORTH_METHOD, ORTH_WARPS_OVERRIDE, factorize_dense))
    pool._agg_prefill_graph = graph
