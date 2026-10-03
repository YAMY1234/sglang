"""Explicit, default-off PD role admission for tracked 2b and final A."""

import logging
import os

import torch

logger = logging.getLogger(__name__)


def install(owner, runner):
    from sglang.srt.environ import envs
    from sglang.srt.disaggregation.state_handoff import HandoffKind
    from sglang.srt.mem_cache.gdn_pd_factor_deferred import (
        PDDeferredController,
        PDDeferredHandoff,
    )

    tracked = envs.SGLANG_GDN_TRACKED_FACTOR_SIDE_STREAM_PD_P.get()
    final = envs.SGLANG_GDN_FINAL_FACTOR_DEFERRED_PD.get()
    owner.pd_factor_deferred = None
    role = owner.pd_shallow_role
    pool = getattr(runner.req_to_token_pool, "factored_gdn_pool", None)
    if role not in ("prefill", "decode") or pool is None or not (tracked or final):
        logger.info(
            "pd_factor_deferred=0 (role=%s factored=%s requested=%s/%s)",
            role,
            pool is not None,
            tracked,
            final,
        )
        return
    args = runner.server_args
    if (
        args.pp_size != 1
        or (role == "prefill" and not args.disable_overlap_schedule)
        or not runner.spec_algorithm.is_none()
        or args.is_embedding
        or args.enable_two_batch_overlap
        or args.enable_torch_compile
        or args.enable_linear_replayssm
        or runner.is_draft_worker
        or runner.lora_manager is not None
        or pool.cfg.decode_method == "warm"
        or not pool.cfg.strict_chunk
        or not pool.cfg.factored_prefix
        or pool.cfg.init_method != "k31"
        or pool.prefix_dense is not None
        or pool.prefix_layer_count() != len(pool.layer_ids)
        or len(pool.layer_ids) != 36
        or owner.fullstack_v3_latent
    ):
        raise ValueError(
            "PD deferred factors require native strict factored k31 P31/D48 PP1"
        )
    if role == "prefill":
        if not tracked or not envs.SGLANG_GDN_PREFILL_FACTOR_GRAPH_K31.get():
            raise ValueError(
                "PD deferred factors require PD tracked 2b and k31 whole graph"
            )
        conflicts = (
            "SGLANG_GDN_PREFILL_JOIN_BRANCHES",
            "SGLANG_GDN_PREFILL_CHECKPOINT_GRAPH",
            "SGLANG_GDN_PREFILL_EXACT_TAIL_BATCH",
            "SGLANG_GDN_PREFILL_BATCH_GRAPH",
            "SGLANG_GDN_PSIDE_GRAPH",
            "SGLANG_GDN_PSIDE_COMPOSITE",
            "SGLANG_PFACTOR4_DEEP_BATCH",
            "TWINSTAR_PD_EMITTER_GRAPH",
            "SGLANG_FLASHNEXT_PD_DEFERRED_TAIL",
            "SGLANG_FLASHNEXT_ARRIVAL_OVERLAP",
        )
        if any(os.environ.get(name, "0") != "0" for name in conflicts):
            raise ValueError(
                "PD deferred factors require isolated 2b/A recipe; conflicting graph/tail mode"
            )
        if not pool._generic_prompt_only_state_cache:
            raise ValueError("PD deferred factors require prompt-only state cache")
        if not runner.attn_backend.linear_attn_backend.kernel_dispatcher.supports_packed_decode:
            raise ValueError("PD A requires the native packed dense recurrence")
    rp = runner.req_to_token_pool
    state = rp.pd_boundary_state
    original = rp.pd_state_handoffs[HandoffKind.STATE_FACTOR]
    if isinstance(original, PDDeferredHandoff):
        raise RuntimeError("PD deferred contract already installed")
    rp.pd_state_handoffs[HandoffKind.STATE_FACTOR] = PDDeferredHandoff(
        original, pool, state, final=final, role=role
    )
    if role == "prefill":
        controller = PDDeferredController(pool, rp, state, final=final)
        owner.pd_factor_deferred = controller
        if final:
            # Common pool and native eager/replay readers must also wait F.
            if pool._final_factor_deferred is not None:
                raise RuntimeError(
                    "AGG and PD final deferred controllers cannot share a pool"
                )
            pool._final_factor_deferred = controller
    logger.info(
        "pd_factor_deferred=%d tracked_2b=%d role=%s wire=%s default=0",
        final,
        tracked,
        role,
        "2:r/r; D completes deep tail" if final else "1:r+1/r",
    )


def transaction_for(owner, batch, prefix_batch, boundary, metadata, boundary_metadata):
    controller = getattr(owner, "pd_factor_deferred", None)
    if controller is None:
        return None
    if torch.cuda.is_current_stream_capturing():
        return controller.fallback("capture")
    if (
        boundary is None
        or metadata is None
        or not 1 <= batch.batch_size <= 16
        or prefix_batch.batch_size != batch.batch_size
        or boundary.batch_size != batch.batch_size
        or batch.twinstar_prompt_final != [True] * batch.batch_size
        or batch.spec_info is not None
        or batch.capture_hidden_mode.is_full()
        or owner.state_audit_dir is not None
        or batch.forward_mode.is_mixed()
        or batch.req_pool_indices_cpu is None
    ):
        return controller.fallback("shape_or_mode")
    if boundary.mamba_track_mask is not None and bool(
        boundary.mamba_track_mask.any().item()
    ):
        return controller.fallback("decode_checkpoint")
    pool, plan = controller.pool, metadata.factored_extend
    side = pool._tracked_factor_side
    if side is None or not side.deferred:
        return controller.fallback("graphs_not_prewarmed")
    if (
        plan is None
        or plan.next_layer != 0
        or plan.last_layer != len(pool.layer_ids) - 1
        or plan.pending
        or plan.checkpoint_group is not None
        or not metadata.has_mamba_track_mask
    ):
        return controller.fallback("plan")
    tracks = metadata.track_ssm_h_dst
    src, dst = metadata.track_ssm_final_src, metadata.track_ssm_final_dst
    if tracks is None or not 1 <= tracks.numel() <= 16:
        return controller.fallback("tracked_shape")
    if controller.final and (
        (src is not None and src.numel()) or (dst is not None and dst.numel())
    ):
        return controller.fallback("prefix_final_copy")
    from sglang.srt.mem_cache.gdn_tracked_factor_side import disjoint_destinations

    reason = disjoint_destinations(
        plan.slots.tolist(),
        tracks.tolist(),
        [] if src is None else src.tolist(),
        [] if dst is None else dst.tolist(),
    )
    if reason:
        return controller.fallback(reason)
    from sglang.srt.mem_cache.gdn_factored_pool import (
        factorize_layers,
        factorize_dense,
        ORTH_METHOD,
        ORTH_WARPS_OVERRIDE,
    )

    from sglang.srt.mem_cache.gdn_pd_factor_deferred import graph_shape

    normal, tracked = graph_shape(plan.slots.numel(), tracks.numel())
    key = side.whole_graph.key(
        normal,
        tracked,
        factorize_layers,
        (ORTH_METHOD, ORTH_WARPS_OVERRIDE, factorize_dense),
        False,
    )
    if key not in side.entries or key not in side.alt_entries:
        return controller.fallback("signature")
    from sglang.srt.mem_cache.gdn_pd_factor_deferred import PDDeferredTransaction

    return PDDeferredTransaction(controller, batch, plan, metadata)
