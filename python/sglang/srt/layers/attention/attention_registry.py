import logging
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING

from sglang.srt.configs.linear_attn_model_registry import (
    get_linear_attn_config,
    import_backend_class,
)
from sglang.srt.utils import (
    check_pkg_version_at_least,
    get_device_capability,
    is_flashinfer_available,
    is_hip,
    is_musa,
    is_npu,
)

_is_musa = is_musa()
_is_npu = is_npu()
_is_hip = is_hip()

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    # evade circular imports
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.model_runner import ModelRunner

ATTENTION_BACKENDS = {}


@lru_cache(maxsize=1)
def has_flashinfer_sm100_gdn_decode_kernels() -> bool:
    """Probe the concrete SM100 GDN decode/MTP APIs."""
    if not is_flashinfer_available() or not check_pkg_version_at_least(
        "flashinfer-python", "0.6.12"
    ):
        return False
    try:
        from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
        from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
            gated_delta_rule_mtp as gated_delta_rule_mtp_bf16,
        )
    except (ImportError, RuntimeError):
        return False
    return callable(gated_delta_rule_decode_pretranspose) and callable(
        gated_delta_rule_mtp_bf16
    )


@lru_cache(maxsize=1)
def has_flashinfer_sm100_gdn_prefill_kernels() -> bool:
    """Probe the concrete SM100 GDN prefill APIs."""
    if not is_flashinfer_available() or not check_pkg_version_at_least(
        "flashinfer-python", "0.6.12"
    ):
        return False
    try:
        from flashinfer.gdn_kernels import chunk_gated_delta_rule_sm100
        from flashinfer.gdn_prefill import chunk_gated_delta_rule
    except (ImportError, RuntimeError):
        return False
    return callable(chunk_gated_delta_rule) and callable(
        chunk_gated_delta_rule_sm100
    )


def has_flashinfer_sm100_gdn_kernels() -> bool:
    """Return whether both SM100 decode/MTP and prefill APIs are present."""
    return has_flashinfer_sm100_gdn_decode_kernels() and (
        has_flashinfer_sm100_gdn_prefill_kernels()
    )


def _supports_flashinfer_gdn_auto_selection(
    runner: "ModelRunner",
    *,
    device_capability: tuple[int, int] | None = None,
) -> bool:
    """Common model, hardware, and state checks for GDN auto-selection."""
    server_args = runner.server_args
    config = runner.hybrid_gdn_config
    if config is None or server_args.linear_attn_backend != "triton":
        return False

    if device_capability is None:
        device_capability = get_device_capability()
    # PR #3742 changes the SM100/SM103 Blackwell path. Other Blackwell
    # generations and Hopper keep their existing default until separately
    # benchmark-validated.
    if device_capability[0] != 10:
        return False

    try:
        import torch

        state_is_bf16 = (
            runner.req_to_token_pool.mamba_pool.mamba_cache.temporal.dtype
            == torch.bfloat16
        )
    except (AttributeError, ImportError):
        # Unit-test runners and early initialization paths may not expose the
        # allocated pool. The server argument is the source of that dtype.
        state_is_bf16 = server_args.mamba_ssm_dtype == "bfloat16"
    if not state_is_bf16:
        return False

    return True


def should_auto_select_flashinfer_gdn_decode(
    runner: "ModelRunner",
    *,
    device_capability: tuple[int, int] | None = None,
    flashinfer_gdn_available: bool | None = None,
) -> bool:
    if runner.server_args.linear_attn_decode_backend is not None or not (
        _supports_flashinfer_gdn_auto_selection(
            runner,
            device_capability=device_capability,
        )
    ):
        return False
    if flashinfer_gdn_available is None:
        flashinfer_gdn_available = has_flashinfer_sm100_gdn_decode_kernels()
    return flashinfer_gdn_available


def should_auto_select_flashinfer_gdn_prefill(
    runner: "ModelRunner",
    *,
    device_capability: tuple[int, int] | None = None,
    cuda_version: str | None = None,
    flashinfer_gdn_available: bool | None = None,
) -> bool:
    if runner.server_args.linear_attn_prefill_backend is not None or not (
        _supports_flashinfer_gdn_auto_selection(
            runner,
            device_capability=device_capability,
        )
    ):
        return False

    config = runner.hybrid_gdn_config
    # The current SM100 CuTe DSL prefill kernel is specialized for K=V=128.
    if (
        getattr(config, "linear_key_head_dim", None) != 128
        or getattr(config, "linear_value_head_dim", None) != 128
    ):
        return False

    if cuda_version is None:
        import torch

        cuda_version = torch.version.cuda
    cuda_major = int(cuda_version.split(".")[0]) if cuda_version else 0
    if cuda_major < 13:
        return False
    if flashinfer_gdn_available is None:
        flashinfer_gdn_available = has_flashinfer_sm100_gdn_prefill_kernels()
    return flashinfer_gdn_available


def maybe_auto_select_flashinfer_gdn_backends(runner: "ModelRunner") -> bool:
    selected = []
    if should_auto_select_flashinfer_gdn_decode(runner):
        runner.server_args.linear_attn_decode_backend = "flashinfer"
        selected.append("decode")
    if should_auto_select_flashinfer_gdn_prefill(runner):
        runner.server_args.linear_attn_prefill_backend = "flashinfer"
        selected.append("prefill")
    if selected:
        logger.info(
            "SM100/SM103 GDN model with bf16 state, 128-dim heads, and "
            "FlashInfer GDN kernels detected; defaulting %s backend(s) to "
            "flashinfer.",
            "/".join(selected),
        )
    return bool(selected)


def register_attention_backend(name):
    def decorator(fn):
        ATTENTION_BACKENDS[name] = fn
        return fn

    return decorator


@register_attention_backend("flashinfer")
def create_flashinfer_backend(runner):
    import torch

    if not runner.use_mla_backend:
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        # Init streams
        if runner.server_args.speculative_algorithm == "EAGLE":
            if (
                not hasattr(runner, "plan_stream_for_flashinfer")
                or not runner.plan_stream_for_flashinfer
            ):
                runner.plan_stream_for_flashinfer = torch.cuda.Stream()
        return FlashInferAttnBackend(
            runner, init_new_workspace=runner.init_new_workspace
        )
    else:
        from sglang.srt.layers.attention.flashinfer_mla_backend import (
            FlashInferMLAAttnBackend,
        )

        return FlashInferMLAAttnBackend(runner)


@register_attention_backend("trtllm_mla")
def create_trtllm_mla_backend(runner):
    if not runner.use_mla_backend:
        raise ValueError("trtllm_mla backend can only be used with MLA models.")
    from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

    return TRTLLMMLABackend(runner)


@register_attention_backend("tokenspeed_mla")
def create_tokenspeed_mla_backend(runner):
    if not runner.use_mla_backend:
        raise ValueError("tokenspeed_mla backend can only be used with MLA models.")
    from sglang.srt.layers.attention.tokenspeed_mla_backend import (
        TokenspeedMLABackend,
    )

    return TokenspeedMLABackend(runner)


@register_attention_backend("cutedsl_mla")
def create_cutedsl_mla_backend(runner):
    if not runner.use_mla_backend:
        raise ValueError("cutedsl_mla backend can only be used with MLA models.")
    from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

    return TRTLLMMLABackend(runner, backend="cute-dsl")


@register_attention_backend("aiter")
def create_aiter_backend(runner):
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

    return AiterAttnBackend(runner)


@register_attention_backend("wave")
def create_wave_backend(runner):
    from sglang.srt.layers.attention.wave_backend import WaveAttnBackend

    return WaveAttnBackend(runner)


@register_attention_backend("ascend")
def create_ascend_backend(runner):
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        AscendAttnBackend,
    )

    return AscendAttnBackend(runner)


@register_attention_backend("dsa")
def create_dsa_backend(runner):
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

    return DeepseekSparseAttnBackend(runner)


@register_attention_backend("nsa")
def _create_nsa_compat(runner):
    warnings.warn(
        "attention-backend='nsa' is deprecated; use 'dsa' instead. "
        "The alias will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_dsa_backend(runner)


@register_attention_backend("dsv4")
def create_dsv4_backend(runner):
    if _is_npu:
        from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (
            DeepseekV4AscendAttnBackend,
        )

        return DeepseekV4AscendAttnBackend(runner)
    elif _is_hip:
        from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
            DeepseekV4HipRadixBackend,
        )

        logger.info(
            "Using DeepseekV4HipRadixBackend for compressed attention backend (HIP)."
        )
        return DeepseekV4HipRadixBackend(runner)
    else:
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        logger.info("Using DeepseekV4AttnBackend for dsv4 attention backend (CUDA).")
        return DeepseekV4AttnBackend(runner)


@register_attention_backend("triton")
def create_triton_backend(runner):
    assert not runner.model_config.is_encoder_decoder, (
        "Cross attention is not supported in the triton attention backend. "
        "Please use `--attention-backend flashinfer`."
    )
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    return TritonAttnBackend(runner)


@register_attention_backend("torch_native")
def create_torch_native_backend(runner):
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

    return TorchNativeAttnBackend(runner)


@register_attention_backend("flex_attention")
def create_flex_attention_backend(runner):
    from sglang.srt.layers.attention.torch_flex_backend import TorchFlexAttnBackend

    return TorchFlexAttnBackend(runner)


@register_attention_backend("flashmla")
def create_flashmla_backend(runner):
    from sglang.srt.layers.attention.flashmla_backend import FlashMLABackend

    return FlashMLABackend(runner)


@register_attention_backend("fa3")
def create_flashattention_v3_backend(runner):

    major, minor = get_device_capability()
    if not _is_musa:
        assert (major == 8 and not runner.use_mla_backend) or major == 9, (
            "FlashAttention v3 Backend requires SM>=80 and SM<=90. "
            "Please use `--attention-backend flashinfer`."
        )
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
        )

        return FlashAttentionBackend(runner)
    else:
        assert major == 3 and minor >= 1, (
            "FlashAttention v3 Backend requires MP>=31. "
            "Please use `--attention-backend triton`."
        )
        from sglang.srt.hardware_backend.musa.attention import (
            MusaFlashAttentionBackend,
        )

        return MusaFlashAttentionBackend(runner)


@register_attention_backend("fa4")
def create_flashattention_v4_backend(runner):
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend

    return FlashAttentionBackend(runner, fa_impl_ver=4)


@register_attention_backend("cutlass_mla")
def create_cutlass_mla_backend(runner):
    from sglang.srt.layers.attention.cutlass_mla_backend import CutlassMLABackend

    return CutlassMLABackend(runner)


@register_attention_backend("trtllm_mha")
def create_trtllm_mha_backend(runner):
    if runner.use_mla_backend:
        raise ValueError("trtllm_mha backend can only be used with non-MLA models.")
    from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

    return TRTLLMHAAttnBackend(runner)


@register_attention_backend("intel_amx")
def create_intel_amx_backend(runner):
    from sglang.srt.layers.attention.intel_amx_backend import IntelAMXAttnBackend

    return IntelAMXAttnBackend(runner)


@register_attention_backend("dual_chunk_flash_attn")
def create_dual_chunk_flash_attn_backend(runner):
    from sglang.srt.layers.attention.dual_chunk_flashattention_backend import (
        DualChunkFlashAttentionBackend,
    )

    return DualChunkFlashAttentionBackend(runner)


def attn_backend_wrapper(runner: "ModelRunner", full_attn_backend: "AttentionBackend"):
    """
    Wrapper for special models like hybrid GDN, so we don't
    need to change the code of the original attention backend.
    """
    assert not (
        runner.hybrid_gdn_config is not None and runner.use_mla_backend
    ), "hybrid_gdn can only be used with non-MLA models."

    if cfg := runner.mambaish_config:
        from sglang.srt.layers.attention.fla.utils import check_environments
        from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
        from sglang.srt.layers.attention.linear.lightning_backend import (
            LightningAttentionBackend,
        )
        from sglang.srt.layers.attention.linear.utils import (
            initialize_linear_attn_config,
        )
        from sglang.srt.utils import is_blackwell, is_npu

        if not is_npu():
            from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
                HybridLinearAttnBackend,
                Mamba2AttnBackend,
            )
            from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
        else:
            from sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend import (
                AscendGDNAttnBackend as GDNAttnBackend,
            )
            from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
                AscendHybridLinearAttnBackend as HybridLinearAttnBackend,
            )
            from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
                AscendMamba2AttnBackend as Mamba2AttnBackend,
            )

        check_environments()
        if runner.hybrid_gdn_config is not None:
            maybe_auto_select_flashinfer_gdn_backends(runner)
        initialize_linear_attn_config(runner.server_args)
        if runner.hybrid_gdn_config is not None:
            if is_blackwell():
                assert (
                    runner.server_args.attention_backend == "triton"
                    or runner.server_args.attention_backend == "trtllm_mha"
                    or runner.server_args.attention_backend == "fa4"
                    or runner.server_args.attention_backend == "flashinfer"
                ), "triton, trtllm_mha, fa4, or flashinfer backend are the only supported backends on Blackwell GPUs for hybrid GDN models, use --attention-backend to specify the backend."
            if is_npu():
                assert (
                    runner.server_args.attention_backend == "ascend"
                ), "ascend backend is the only supported backend on NPU for hybrid GDN models, use --attention-backend ascend to specify the backend."
            logger.info(f"Using hybrid linear attention backend for hybrid GDN models.")
            linear_attn_backend = GDNAttnBackend(runner)
        elif runner.mamba2_config is not None:
            linear_attn_backend = Mamba2AttnBackend(runner)
        elif runner.kimi_linear_config is not None:
            linear_attn_backend = KDAAttnBackend(runner)
        elif runner.hybrid_lightning_config is not None:
            linear_attn_backend = LightningAttentionBackend(runner)
        else:
            spec_result = get_linear_attn_config(runner.model_config.hf_config)
            if spec_result is not None:
                spec, _ = spec_result
                BackendClass = import_backend_class(spec.backend_class_name)
                linear_attn_backend = BackendClass(runner)
            else:
                raise ValueError(
                    "Expected hybrid GDN or NemotronH models, but got unknown model. "
                    "If this is a custom hybrid model, use register_linear_attn_model() "
                    "from sglang.srt.configs.linear_attn_model_registry."
                )
        if runner.is_draft_worker:
            # FIXME: we assume that MTP/NEXTN always use full-attention.
            full_attn_layers = [0]
        else:
            full_attn_layers = cfg.full_attention_layer_ids
        return HybridLinearAttnBackend(
            full_attn_backend, linear_attn_backend, full_attn_layers
        )

    return full_attn_backend


@register_attention_backend("intel_xpu")
def create_intel_xpu_backend(runner):
    from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend

    return XPUAttentionBackend(runner)
