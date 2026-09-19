from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.overrides import (
    _deepseek_v4_kv_cache_dtype,
    declare_resolution,
    model_config_of,
    resolving_view,
    run_post_process_pass,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


# These are correctness envelopes, not a generic PP partition policy.  Every
# entry has passed the sparse-owner validator and has a final stage containing
# all bundled DSpark target layers.  Keep the list explicit so unsupported
# cuts remain fail-closed and startup errors can name the partitions users can
# actually select.
_DSV41_VALIDATED_PP_PARTITIONS = {
    (2, 2, 2): (20, 20),
    (1, 1, 4): (8, 6, 6, 20),
}


def _validate_deepseek_v41_pp_layout(
    hf_config, pp_size: int
) -> tuple[tuple[tuple[int, int], ...], tuple[int | None, ...]]:
    """Validate that V4.1 low-ratio sparse state stays inside each PP stage.

    Ratio-1/2 attention layers reuse the nearest preceding ``kv_source`` of the
    same ratio. Index sources reuse that owner's index keys, and index sources
    after ``candidate_source`` also reuse its candidate mask. None of those
    tensors are part of the PP wire contract, so every producer and consumer
    must remain on the same stage.
    """
    from sglang.srt.distributed.utils import get_pp_indices

    num_layers = int(hf_config.num_hidden_layers)
    partitions = tuple(
        get_pp_indices(num_layers, rank, pp_size) for rank in range(pp_size)
    )
    layer_stages = [0] * num_layers
    covered = []
    for stage, (start, end) in enumerate(partitions):
        if not 0 <= start < end <= num_layers:
            raise ValueError(
                f"DeepSeek-V4.1 PP stage {stage} has invalid layer range "
                f"[{start}, {end})."
            )
        for layer_id in range(start, end):
            layer_stages[layer_id] = stage
            covered.append(layer_id)
    if covered != list(range(num_layers)):
        raise ValueError(
            "DeepSeek-V4.1 PP partition must cover every transformer layer "
            f"exactly once, got ranges={partitions}."
        )

    all_ratios = tuple(int(ratio) for ratio in hf_config.compress_ratios)
    num_nextn_layers = int(getattr(hf_config, "num_nextn_predict_layers", 0))
    expected_ratio_count = num_layers + num_nextn_layers
    if len(all_ratios) != expected_ratio_count:
        raise ValueError(
            "DeepSeek-V4.1 compress_ratios must cover transformer and next-token "
            f"layers, got {len(all_ratios)} for {num_layers} transformer + "
            f"{num_nextn_layers} next-token layers."
        )
    # Sparse PP ownership in S3.1 is for the target transformer only. The
    # trailing next-token layers belong to speculative decoding, which remains
    # fail-closed in this stage.
    ratios = all_ratios[:num_layers]
    sources = tuple(int(layer_id) for layer_id in hf_config.kv_source_layer_ids)
    invalid_sources = [source for source in sources if not 0 <= source < num_layers]
    if invalid_sources:
        raise ValueError(
            f"DeepSeek-V4.1 kv_source layers are out of range: {invalid_sources}."
        )
    owners: list[int | None] = [None] * num_layers
    for layer_id, ratio in enumerate(ratios):
        if ratio not in (1, 2):
            continue
        candidates = [
            source
            for source in sources
            if source <= layer_id and ratios[source] == ratio
        ]
        if not candidates:
            raise ValueError(
                f"DeepSeek-V4.1 layer {layer_id} (ratio {ratio}) has no "
                "preceding kv_source owner."
            )
        owner = max(candidates)
        owners[layer_id] = owner
        if layer_stages[owner] != layer_stages[layer_id]:
            raise ValueError(
                "DeepSeek-V4.1 PP partition crosses a ratio-1/2 compressed-KV "
                f"owner boundary: layer {layer_id} on stage "
                f"{layer_stages[layer_id]} reads kv_source {owner} on stage "
                f"{layer_stages[owner]}; compressed KV/index relay is not "
                "implemented."
            )

    index_sources = tuple(
        int(layer_id) for layer_id in hf_config.index_source_layer_ids
    )
    for layer_id in index_sources:
        if not 0 <= layer_id < num_layers:
            raise ValueError(
                f"DeepSeek-V4.1 index_source layer {layer_id} is out of range."
            )
        owner = owners[layer_id]
        if owner is not None and layer_stages[owner] != layer_stages[layer_id]:
            raise ValueError(
                "DeepSeek-V4.1 PP partition crosses an index-key owner boundary: "
                f"index_source {layer_id} reads kv_source {owner}."
            )

    candidate_source = int(hf_config.candidate_source_layer_id)
    if candidate_source >= 0:
        if candidate_source not in index_sources:
            raise ValueError(
                "DeepSeek-V4.1 candidate_source_layer_id must also be an "
                f"index_source, got {candidate_source}."
            )
        for layer_id in index_sources:
            if (
                layer_id > candidate_source
                and layer_stages[layer_id] != layer_stages[candidate_source]
            ):
                raise ValueError(
                    "DeepSeek-V4.1 PP partition crosses a candidate-mask owner "
                    f"boundary: index_source {layer_id} on stage "
                    f"{layer_stages[layer_id]} consumes candidate_source "
                    f"{candidate_source} on stage "
                    f"{layer_stages[candidate_source]}; candidate relay is not "
                    "implemented."
                )

    return partitions, tuple(owners)


def _partition_ranges(partition: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    start = 0
    ranges = []
    for size in partition:
        ranges.append((start, start + size))
        start += size
    return tuple(ranges)


def _validated_pp_partition_summary() -> str:
    return ", ".join(
        f"TP{tp}/EP{ep}/PP{pp}:[{','.join(str(v) for v in partition)}]"
        for (tp, ep, pp), partition in _DSV41_VALIDATED_PP_PARTITIONS.items()
    )


def deepseek_v41_pp_layout_missing(cfg, hf_config) -> tuple[str, ...]:
    """Return unmet topology/partition requirements for validated DSV4.1 PP."""
    missing: list[str] = []
    topology = (cfg.tp_size, cfg.ep_size, cfg.pp_size)
    expected_partition = _DSV41_VALIDATED_PP_PARTITIONS.get(topology)
    if expected_partition is None:
        missing.append(
            "validated TP/EP/PP topology and partition from "
            f"[{_validated_pp_partition_summary()}] "
            f"(got TP{cfg.tp_size}/EP{cfg.ep_size}/PP{cfg.pp_size})"
        )
        return tuple(missing)

    try:
        partitions, _ = _validate_deepseek_v41_pp_layout(hf_config, cfg.pp_size)
    except ValueError as exc:
        missing.append(str(exc))
        missing.append(
            "validated feasible partitions: "
            f"[{','.join(str(v) for v in expected_partition)}]"
        )
        return tuple(missing)

    expected_ranges = _partition_ranges(expected_partition)
    if partitions != expected_ranges:
        missing.append(
            "SGLANG_PP_LAYER_PARTITION="
            f"{','.join(str(v) for v in expected_partition)} "
            f"(resolved ranges={partitions}); validated feasible partitions: "
            f"[{','.join(str(v) for v in expected_partition)}]"
        )
    return tuple(missing)


def deepseek_v41_pp_dspark_prefill_missing(cfg, hf_config) -> tuple[str, ...]:
    """Return unmet requirements for the validated PP-prefill DSpark envelope.

    Keep this predicate shared by the generic PP, DSpark, language-model-only,
    and model-family gates so a future relaxation cannot accidentally open only
    part of the startup path.  In particular, aggregate PP+DSpark stays closed:
    this envelope is only for the disaggregated prefill role.
    """
    missing: list[str] = []
    if getattr(hf_config, "model_type", None) != "deepseek_v41":
        missing.append("DeepSeek-V4.1")
    if cfg.disaggregation_mode != "prefill":
        missing.append("--disaggregation-mode prefill")
    if not cfg.language_model_only:
        missing.append("--language-model-only")
    if str(cfg.speculative_algorithm).upper() != "DSPARK":
        missing.append("--speculative-algorithm DSPARK")
    if getattr(hf_config, "model_type", None) == "deepseek_v41":
        missing.extend(deepseek_v41_pp_layout_missing(cfg, hf_config))
    if (
        cfg.dp_size != 1
        or cfg.enable_dp_attention
        or cfg.attn_cp_size != 1
        or cfg.dcp_size != 1
        or cfg.enable_prefill_context_parallel
    ):
        missing.append("DP1 and CP1")
    if cfg.enable_encoder_swa_bounded_replay or cfg.enable_decoder_swa_bounded_replay:
        missing.append("encoder/decoder SWA bounded replay disabled")

    if getattr(hf_config, "model_type", None) == "deepseek_v41" and cfg.pp_size > 1:
        try:
            partitions, _ = _validate_deepseek_v41_pp_layout(hf_config, cfg.pp_size)
        except ValueError:
            # deepseek_v41_pp_layout_missing already reports the precise
            # ownership error and the supported partition list.
            pass
        else:
            draft_layers = tuple(
                int(layer_id)
                for layer_id in getattr(hf_config, "dspark_target_layer_ids", ())
            )
            final_start, final_end = partitions[-1]
            if not draft_layers or any(
                not final_start <= layer_id < final_end for layer_id in draft_layers
            ):
                missing.append(
                    "all DSpark target layers owned by the final PP stage "
                    f"[{final_start}, {final_end}) (got {draft_layers})"
                )
    return tuple(missing)


def deepseek_v41_pp2_dspark_prefill_missing(cfg, hf_config) -> tuple[str, ...]:
    """Compatibility alias for callers from the original S3.2 PP2 patch."""
    return deepseek_v41_pp_dspark_prefill_missing(cfg, hf_config)


def validate_deepseek_v4_mega_moe_token_budget(
    server_args: ServerArgs,
) -> None:
    """Ensure the DSV4 prefill budget fits MegaMoE's per-rank buffer."""
    cfg = resolving_view(server_args)
    mega_moe_enabled = cfg.moe_a2a_backend == "megamoe"
    if not mega_moe_enabled or cfg.disaggregation_mode == "decode":
        # decode node will skip the check because decode bs is not relevant with --chunk-prefill-size
        return

    if cfg.pp_size > 1 and cfg.enable_dynamic_chunking:
        return

    if cfg.chunked_prefill_size is None or cfg.chunked_prefill_size <= 0:
        raise ValueError(
            "DeepSeekV4 with MegaMoE requires chunked prefill to be enabled. "
            "Set --chunked-prefill-size to a positive value; "
            "--chunked-prefill-size=-1 is unsafe because MegaMoE's per-rank "
            "token requirement would not have a strict prefill-forward bound."
        )

    if cfg.enable_prefill_cp:
        token_partition_size = cfg.attn_cp_size
        token_partition_name = "attn_cp_size"
        token_alignment = 1
        local_chunked_prefill_size = (
            cfg.chunked_prefill_size + token_partition_size - 1
        ) // token_partition_size
    elif cfg.enable_dp_attention:
        token_partition_size = cfg.dp_size
        token_partition_name = "dp_size"
        token_alignment = max(
            cfg.tp_size // cfg.dp_size // cfg.attn_cp_size,
            1,
        )
        local_chunked_prefill_size = cfg.chunked_prefill_size // token_partition_size
    else:
        # Pure TP and PP with static chunking are handled here.
        token_partition_size = 1
        token_partition_name = "none"
        # global_num_tokens will ceil_align to attn_tp_size so the validation needs to do alignment as well
        token_alignment = max(
            cfg.tp_size // token_partition_size // cfg.attn_cp_size,
            1,
        )
        local_chunked_prefill_size = cfg.chunked_prefill_size

    if local_chunked_prefill_size <= 0:
        raise ValueError(
            "DeepSeekV4 with MegaMoE requires a positive effective per-rank "
            "chunked prefill size. "
            f"Current values: chunked_prefill_size="
            f"{cfg.chunked_prefill_size}, "
            f"token_partition={token_partition_name}, "
            f"token_partition_size={token_partition_size}."
        )

    required_tokens_per_rank = (
        (local_chunked_prefill_size + token_alignment - 1)
        // token_alignment
        * token_alignment
    )
    max_tokens_per_rank = (
        envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    )
    if max_tokens_per_rank < required_tokens_per_rank:
        raise ValueError(
            "DeepSeekV4 with MegaMoE requires "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK to "
            "cover each rank's effective prefill token budget. "
            f"Current values: chunked_prefill_size="
            f"{cfg.chunked_prefill_size}, "
            f"token_partition={token_partition_name}, "
            f"token_partition_size={token_partition_size}, "
            f"token_alignment={token_alignment}, "
            f"required_per_rank={required_tokens_per_rank}, "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="
            f"{max_tokens_per_rank}. Set "
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK to at "
            f"least {required_tokens_per_rank}, or lower "
            "--chunked-prefill-size until the effective per-rank budget fits. "
            "Otherwise MegaMoE falls back to the fused MoE path at runtime."
        )


def apply_deepseek_v4_defaults(server_args: ServerArgs, model_arch: str) -> None:
    """Residual imperative arm of the DeepSeek V4 defaults.

    The attention/page/window/MoE-runner declarations moved to the override
    registry (arg_groups/overrides.py: _deepseek_v4_overrides) and the
    kv-cache dtype default to the resolution pipeline
    (_deepseek_v4_kv_cache_dtype, invoked below at its legacy slot). This
    keeps, at the legacy slot: the ROCm env fill (env-write policy), the
    max_running_requests fill (the speculative hook is a later writer of
    that field) and the validations.
    """
    cfg = resolving_view(server_args)

    # FlashMLA sparse prefill (SGLANG_OPT_FLASHMLA_SPARSE_PREFILL, default on)
    # currently returns incorrect output for DeepSeek-V4-Flash on ROCm/HIP
    # (MI355X), which breaks the disaggregation nightly. Keep the previous
    # (dense prefill) behavior on ROCm until the sparse kernel is validated
    # there;
    if get_platform().is_hip:
        logger.warning(
            "Disabling SGLANG_OPT_FLASHMLA_SPARSE_PREFILL by default on ROCm/HIP "
            f"for {model_arch}; set it explicitly to override."
        )
        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(False)

    # The kv-cache dtype default moved to the resolution pipeline
    # (arg_groups/overrides.py: _deepseek_v4_kv_cache_dtype), invoked here at
    # its legacy slot.

    run_post_process_pass(server_args, _deepseek_v4_kv_cache_dtype)

    if cfg.max_running_requests is None:
        declare_resolution(
            server_args,
            "apply_deepseek_v4_defaults",
            max_running_requests=256,
        )
        logger.warning(
            f"Setting max_running_requests to {cfg.max_running_requests} for {model_arch}."
        )

    if cfg.speculative_algorithm is not None:
        assert cfg.speculative_algorithm in (
            "EAGLE",
            "DSPARK",
        ), (
            f"Only EAGLE and DSPARK speculative algorithms are supported for {model_arch}"
        )
        if cfg.speculative_algorithm == "EAGLE":
            assert cfg.speculative_eagle_topk == 1, (
                f"Only EAGLE speculative algorithm with topk == 1 is supported for {model_arch}"
            )


def validate_deepseek_v4_cp(server_args: ServerArgs) -> None:
    """Validate DeepSeek V4 context-parallel configuration."""
    cfg = resolving_view(server_args)
    if not cfg.enable_prefill_cp:
        return

    if cfg.cp_strategy != "interleave":
        raise ValueError(
            f"DeepSeekV4 only supports interleave CP strategy, got {cfg.cp_strategy}"
        )

    if get_platform().is_hip or get_platform().is_npu:
        # Protected platform implementations still consume the legacy runtime
        # fields. Generic backends use enable_prefill_cp/cp_strategy directly.
        declare_resolution(
            server_args,
            "validate_deepseek_v4_cp",
            enable_dsa_prefill_context_parallel=True,
        )
        declare_resolution(
            server_args,
            "validate_deepseek_v4_cp",
            enable_prefill_context_parallel=False,
        )
        declare_resolution(
            server_args,
            "validate_deepseek_v4_cp",
            dsa_prefill_cp_mode="round-robin-split",
        )
    declare_resolution(
        server_args,
        "validate_deepseek_v4_cp",
        enable_dp_attention=True,
    )
    declare_resolution(
        server_args,
        "validate_deepseek_v4_cp",
        moe_dense_tp_size=1,
    )
    declare_resolution(
        server_args,
        "validate_deepseek_v4_cp",
        attn_cp_size=cfg.tp_size // cfg.dp_size,
    )
    assert cfg.dp_size == 1, (
        "For round-robin split mode, dp attention is not supported."
    )
    assert cfg.tp_size <= 8, (
        "Context parallel only supports single machine (tp_size <= 8). Cross-machine CP has precision issues."
    )
    supported_a2a_backends = ("none", "deepep", "megamoe", "mori")
    if cfg.moe_a2a_backend not in supported_a2a_backends:
        raise ValueError(
            f"DeepSeekV4 CP supports moe_a2a_backend in {supported_a2a_backends}, "
            f"got {cfg.moe_a2a_backend!r}."
        )
    logger.warning(
        f"Enable Context Parallel for DeepSeekV4, "
        f"dp_size={cfg.dp_size}, moe_dense_tp_size={cfg.moe_dense_tp_size}, "
        f"attn_cp_size={cfg.attn_cp_size}, ep_size={cfg.ep_size}, tp_size={cfg.tp_size}"
    )


def validate_deepseek_v41_features(server_args: ServerArgs) -> None:
    """Reject the server features DeepSeek-V4.1 cannot serve yet."""
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
        is_unified_kv_triton,
    )

    cfg = resolving_view(server_args)
    model_config = model_config_of(server_args)
    hf_config = model_config.hf_config
    if hf_config.model_type != "deepseek_v41":
        if cfg.enable_encoder_swa_bounded_replay:
            raise ValueError(
                "--enable-encoder-swa-bounded-replay requires DeepSeek-V4.1"
            )
        return
    if cfg.pp_size > 1:
        missing = []
        if not cfg.language_model_only:
            missing.append("--language-model-only")
        is_validated_pd_prefill = not deepseek_v41_pp_dspark_prefill_missing(
            cfg, hf_config
        )
        is_validated_aggregate = (
            cfg.speculative_algorithm is None and cfg.disaggregation_mode == "null"
        )
        if not (is_validated_aggregate or is_validated_pd_prefill):
            if cfg.speculative_algorithm is not None:
                missing.append(
                    "speculative decoding disabled, except DSpark on the validated "
                    "PD-prefill PP2/PP4 paths"
                )
            if cfg.disaggregation_mode != "null":
                missing.append(
                    "aggregate mode, except the validated PD-prefill PP2/PP4 paths"
                )
        missing.extend(deepseek_v41_pp_layout_missing(cfg, hf_config))
        if (
            cfg.dp_size != 1
            or cfg.enable_dp_attention
            or cfg.attn_cp_size != 1
            or cfg.dcp_size != 1
            or cfg.enable_prefill_context_parallel
        ):
            missing.append("DP1 and CP1")
        if (
            cfg.enable_encoder_swa_bounded_replay
            or cfg.enable_decoder_swa_bounded_replay
        ):
            missing.append("encoder/decoder SWA bounded replay disabled")

        if missing:
            raise ValueError(
                "DeepSeek-V4.1 pipeline parallelism is fail-closed outside the "
                "validated standalone language-only aggregate PP2/PP4 and "
                "language-model-only PD-prefill PP2/PP4+DSpark "
                "configuration; require "
                + ", ".join(missing)
                + ". Aggregate PP+DSpark, decode-role PP, vision, or "
                "cross-stage sparse/DSpark owners needs its dedicated "
                "implementation stage."
            )
    if cfg.enable_encoder_swa_bounded_replay:
        from sglang.srt.model_executor.cuda_graph_config import Backend

        incompatible = (
            ("non-CUDA hardware", not get_platform().is_cuda),
            (
                "prefill CUDA graphs",
                cfg.cuda_graph_config.prefill.backend != Backend.DISABLED,
            ),
            ("DP attention", cfg.enable_dp_attention),
            (
                "context parallelism",
                cfg.enable_prefill_context_parallel or cfg.attn_cp_size > 1,
            ),
            ("external cache linker", cfg.enable_unified_cache_external_linker),
            ("unified memory", cfg.enable_unified_memory),
            ("PD disaggregation", cfg.disaggregation_mode != "null"),
            ("mixed prefill/decode", cfg.enable_mixed_chunk),
            ("LoRA", cfg.enable_lora),
            ("radix sessions", cfg.enable_session_radix_cache),
        )
        for feature, enabled in incompatible:
            if enabled:
                raise ValueError(
                    f"--enable-encoder-swa-bounded-replay does not support {feature} yet"
                )
        if (
            cfg.max_running_requests is None
            or cfg.max_running_requests <= 0
            or not cfg.chunked_prefill_size
            or cfg.chunked_prefill_size < 128
        ):
            raise ValueError(
                "encoder SWA replay requires explicit --max-running-requests and --chunked-prefill-size >= 128"
            )

    unsupported = (
        (
            "speculative decoding other than DSpark",
            cfg.speculative_algorithm is not None
            and str(cfg.speculative_algorithm).upper() != "DSPARK",
        ),
        ("HiSparse", cfg.enable_hisparse),
        ("the unified KV layout", is_unified_kv_triton()),
        ("two-batch overlap", cfg.enable_two_batch_overlap),
    )
    for feature, enabled in unsupported:
        if enabled:
            raise ValueError(
                f"DeepSeek-V4.1 does not support {feature} yet; disable it to "
                "serve this model."
            )

    if cfg.disaggregation_mode != "null" and cfg.speculative_algorithm is not None:
        from sglang.srt.speculative.ragged_verify import (
            RaggedVerifyMode,
            read_ragged_verify_mode,
        )

        if (
            read_ragged_verify_mode() is not RaggedVerifyMode.STATIC
            or cfg.disaggregation_transfer_backend != "mooncake"
            or cfg.dp_size != 1
            or cfg.enable_dp_attention
            or cfg.attn_cp_size != 1
            or cfg.dcp_size != 1
            or cfg.enable_prefill_context_parallel
        ):
            raise ValueError(
                "DeepSeek-V4.1 DSpark PD requires static verify, Mooncake, "
                "DP=1 and CP=1. Both servers must enable DSpark with the same "
                "block size and TP size."
            )

    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase

    prefill_graph = cfg.cuda_graph_config.prefill
    if prefill_graph.backend != Backend.DISABLED and prefill_graph.max_seq_len is None:
        # The captured low-ratio indexer scores a static context width; 16k
        # keeps it inside the candidate window at under 1 ms per layer.
        declare_resolution(
            server_args,
            "validate_deepseek_v41_features",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, Phase.PREFILL, max_seq_len=16 * 1024
            ),
        )
        logger.warning(
            "Setting cuda_graph_config[prefill].max_seq_len to 16384 for "
            "DeepSeek-V4.1; longer contexts run eager prefill."
        )

    if cfg.enable_decoder_swa_bounded_replay:
        from sglang.srt.model_executor.cuda_graph_config import Backend

        # The late layers see a per-request tail slice, so their token count is
        # no longer the captured prefill shape.
        incompatible = (
            (
                "the prefill CUDA graph",
                cfg.cuda_graph_config.prefill.backend != Backend.DISABLED,
            ),
            # input_ids_global is a DP-wide gather, not a per-local-token tensor,
            # so the tail slice does not apply to it.
            ("DP attention", cfg.enable_dp_attention),
        )
        for feature, enabled in incompatible:
            if enabled:
                raise ValueError(
                    "--enable-decoder-swa-bounded-replay cannot be combined with "
                    f"{feature} yet; disable one of them."
                )
