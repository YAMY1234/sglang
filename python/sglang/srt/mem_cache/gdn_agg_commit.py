"""Default-off native AGG singleton commit graph; no P48 tail adapter.

Only B1 normal and B1 tracked graphs are admitted in this first port. Other
batch modes retain the native grouped commit. All model work and publication
stay on the caller's stream, with no extra host synchronization per request.
"""
from functools import wraps
import logging
import os

import torch

from .gdn_agg_commit_budget import FLAG, reservation_bytes
from .gdn_prefill_batch_graph import BatchCollector, PrefillBatchGraph

logger = logging.getLogger(__name__)
SHAPES = ((1, None), (1, 1))


def eligible(batch, metadata, pool):
    mode = batch.forward_mode
    if (not mode.is_extend() or mode.is_mixed() or batch.batch_size != 1
            or getattr(batch, "spec_info", None) is not None
            or getattr(batch, "can_run_tbo", False)
            or getattr(batch, "tbo_split_seq_index", None) is not None
            or getattr(batch, "_pfactor_legacy_mixed", False)):
        return False
    plan = metadata.factored_extend
    if (plan is None or plan.slots.numel() != 1 or plan.next_layer != 0
            or plan.last_layer != len(pool.layer_ids) - 1 or plan.pending
            or getattr(plan, "batch_collector", None) is not None
            or getattr(plan, "checkpoint_group", None) is not None
            or getattr(plan, "preserve_layer_sink", False)
            or getattr(pool, "_exact_tail_transaction", None) is not None):
        return False
    # No device-value reads: shape and native plan ownership are sufficient.
    for name in ("track_ssm_h_src", "track_ssm_h_dst", "track_ssm_final_src",
                 "track_ssm_final_dst", "track_ssm_recompute_dst"):
        value = getattr(metadata, name, None)
        if value is not None and value.numel() > (0 if "recompute" in name else 1):
            return False
    return True


def install_forward(owner, pool, backend_provider, graph):
    """Wrap one owner, preserving the native model signature and fallback."""
    if getattr(owner, "_agg_commit_installed", False):
        raise RuntimeError("AGG commit entry installed twice")
    original = owner.forward

    @wraps(original)
    def forward(input_ids, positions, forward_batch, *args, **kwargs):
        backend = backend_provider()
        metadata = backend.forward_metadata
        if not eligible(forward_batch, metadata, pool):
            pool._agg_commit_stats["fallback"] += 1
            return original(input_ids, positions, forward_batch, *args, **kwargs)
        if getattr(pool, "_agg_commit_active", False):
            raise RuntimeError("overlapping AGG commit transactions are not admitted")
        plan = metadata.factored_extend
        pool._agg_commit_active = True
        try:
            with BatchCollector(pool, plan, graph=graph):
                result = original(input_ids, positions, forward_batch, *args, **kwargs)
                if backend.forward_metadata is not metadata or metadata.factored_extend is not plan:
                    raise RuntimeError("AGG commit metadata changed within a model forward")
            pool._agg_commit_stats["committed"] += 1
            return result
        finally:
            pool._agg_commit_active = False
            plan.batch_collector = None
            plan.pending.clear()

    owner.forward = forward
    owner._agg_commit_installed = True


def install(runner):
    if os.environ.get(FLAG, "0") != "1":
        return
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from .gdn_factored_pool import (
        ORTH_METHOD, ORTH_WARPS_OVERRIDE, factorize_dense, factorize_layers,
    )
    from . import gdn_prefill_joint as joint

    role = runner.server_args.disaggregation_mode
    reserve = reservation_bytes(role)
    pool = runner.req_to_token_pool.factored_gdn_pool
    owner = runner.model
    if (pool is None or not pool.a.is_cuda or not pool.batch_prefill
            or pool.cfg.init_method != "k31" or not pool.cfg.strict_chunk
            or not pool.cfg.factored_prefix or pool.prefix_dense is not None
            or pool.prefix_layer_count() != len(pool.layer_ids)
            or getattr(owner, "twinstar", None) is not None
            or getattr(owner, "pd_shallow_role", None) == "prefill"
            or joint.configured()
            or any(os.environ.get(key) == "1" for key in (
                "SGLANG_GDN_PSIDE_GRAPH", "SGLANG_GDN_PREFILL_EXACT_TAIL_BATCH",
                "SGLANG_GDN_PREFILL_BATCH_GRAPH", "SGLANG_GDN_PREFILL_CHECKPOINT_GRAPH"))):
        raise ValueError("AGG commit graph requires native full-depth strict k31 without tail/joint adapters")
    # Full five-bucket normal+tracked prewarm owns 3.26953125 GiB/rank at
    # 36 layers, HV24, K=V128. Only two B1 graphs own 108 MiB before graph
    # temporaries. The explicit reservation includes those temporaries too.
    minimum = 2 * len(pool.layer_ids) * pool.hv * pool.v * pool.k * 4
    if minimum > reserve:
        raise ValueError("AGG commit state buffers exceed the reserved capacity")
    torch.cuda.synchronize(pool.a.device)
    torch.cuda.empty_cache()
    free_before = torch.cuda.mem_get_info(pool.a.device)[0]
    before_alloc = torch.cuda.memory_allocated(pool.a.device)
    before_reserved = torch.cuda.memory_reserved(pool.a.device)
    graph = PrefillBatchGraph(include_tail=False, shapes=SHAPES)
    graph.prewarm(pool, eager=factorize_layers,
                  policy=(ORTH_METHOD, ORTH_WARPS_OVERRIDE, factorize_dense))
    allocated = max(0, torch.cuda.memory_allocated(pool.a.device) - before_alloc)
    reserved = max(0, torch.cuda.memory_reserved(pool.a.device) - before_reserved)
    torch.cuda.empty_cache()
    charged = max(0, free_before - torch.cuda.mem_get_info(pool.a.device)[0])
    if max(charged, allocated, reserved) > reserve:
        raise RuntimeError("AGG commit prewarm exceeded its reserved workspace")
    pool._agg_commit_graph = graph
    pool._agg_commit_charged_bytes = charged
    pool._agg_commit_stats = dict(committed=0, fallback=0, state_bytes=minimum,
        allocated_growth_bytes=allocated, reserved_growth_bytes=reserved,
        charged_bytes=charged, reserved_budget_bytes=reserve)
    install_forward(owner, pool, lambda: get_attn_backend().linear_attn_backend, graph)
    logger.info("GDN AGG commit installed: shapes=%s state_bytes=%d allocated_growth=%d "
                "reserved_growth=%d charged_bytes=%d reserved_budget=%d",
                SHAPES, minimum, allocated, reserved, charged, reserve)
