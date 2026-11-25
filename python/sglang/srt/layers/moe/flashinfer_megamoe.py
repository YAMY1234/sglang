# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic FlashInfer MegaMOE backend (moe_ep.MoEEpMegaLayer).

Wraps FlashInfer's fused EP all-to-all + expert-compute mega kernel so it can
be selected as a model-agnostic MoE runner backend through the standard
FusedMoE dispatch -> run_moe_core -> combine flow. The mega kernel does its EP
communication internally via the deep_gemm symmetric buffer, so the dispatcher
and combine stay pure no-ops; this module owns the layer build + forward.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)

logger = logging.getLogger(__name__)


def _format_megakernel_config(config: Any) -> str:
    """Readable one-line repr of a mega kernel config.

    The config dataclasses carry per-expert tensor fields (e.g. fc1_alpha /
    fc2_alpha / fc1_norm_const); their default repr dumps every element, so
    abbreviate tensors to shape/dtype/device instead.
    """
    import dataclasses

    if not dataclasses.is_dataclass(config):
        return repr(config)

    parts = []
    for field, value in zip(dataclasses.fields(config), dataclasses.astuple(config)):
        if isinstance(value, torch.Tensor):
            value = (
                f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
                f"device={value.device})"
            )
        else:
            value = repr(value)
        parts.append(f"{field.name}={value}")
    return f"{type(config).__name__}({', '.join(parts)})"


if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher import (
        DispatchOutput,
        StandardCombineInput,
    )


@contextmanager
def _capture_safe_ue8m0_pack() -> Generator[None, None, None]:
    """Make deep_gemm's UE8M0 scale packing safe under CUDA graph capture.

    The deep_gemm mega staging path (block-FP8 models such as DeepSeek-V4-Flash)
    runs ``per_token_cast_to_fp8(..., use_packed_ue8m0=True)`` on every forward,
    which calls ``deep_gemm.utils.math.pack_ue8m0_to_int``. Its upstream
    implementation carries two debug assertions::

        assert (x_int >= 0).all() and (x_int & 0x7fffff == 0).all()

    ``.all()`` forces a device->host sync, which is illegal while a CUDA graph is
    capturing. Replace the helper only around the mega forward that runs during
    capture, then restore the exact upstream function.

    TODO(deepseek-ai/DeepGEMM#414): remove once upstream provides a capture-safe
    helper: https://github.com/deepseek-ai/DeepGEMM/issues/414
    """
    if not torch.cuda.is_available() or not torch.cuda.is_current_stream_capturing():
        yield
        return

    try:
        import deep_gemm.utils.math as _dgm
    except ImportError:
        # deep_gemm is only needed by the block-FP8 mega path; NVFP4/MXFP8 mega
        # runs on cutedsl and does not import it. Nothing to patch here.
        yield
        return

    def _pack_ue8m0_to_int(x: torch.Tensor) -> torch.Tensor:
        x_int = x.view(torch.int)
        return (x_int >> 23).to(torch.uint8).view(torch.int)

    original = _dgm.pack_ue8m0_to_int
    _dgm.pack_ue8m0_to_int = _pack_ue8m0_to_int
    try:
        yield
    finally:
        _dgm.pack_ue8m0_to_int = original


@dataclass
class FlashInferMegaMoeQuantInfo(MoeQuantInfo):
    mega: Any
    mega_forward: Callable[[Any, Any], torch.Tensor] | None = None
    fc1_alpha: torch.Tensor | None = None
    fc2_alpha: torch.Tensor | None = None
    fc1_norm_const: torch.Tensor | None = None
    apply_routed_scaling_factor: bool = False

    def __post_init__(self) -> None:
        if self.mega_forward is None:
            self.mega_forward = _select_megamoe_forward(self.mega)


def _forward_megamoe_with_workspace_view(mega: Any, tensors: Any) -> torch.Tensor:
    return mega.forward(tensors, return_workspace_view=True)


def _forward_megamoe_legacy(mega: Any, tensors: Any) -> torch.Tensor:
    return mega.forward(tensors)


def _select_megamoe_forward(mega: Any) -> Callable[[Any, Any], torch.Tensor]:
    if mega.supports_output_view:
        return _forward_megamoe_with_workspace_view
    return _forward_megamoe_legacy


def _resolve_max_tokens_per_rank() -> int:
    """Per-rank symmetric-buffer sizing for the mega kernel.

    Honors the explicit env override; otherwise derives the largest per-(DP)rank
    token count a single MoE forward can route (same bound the cutedsl A2A path
    uses), falling back to 1024 if it cannot be determined.
    """
    configured = envs.SGLANG_FLASHINFER_MEGAMOE_MAX_TOKENS_PER_RANK.get()
    if configured > 0:
        return configured

    from sglang.srt.runtime_context import cutedsl_moe_max_num_tokens

    derived = cutedsl_moe_max_num_tokens()
    return derived if derived > 0 else 1024


def resolve_flashinfer_megamoe_combine_dtype() -> str:
    combine_dtype = envs.SGLANG_FLASHINFER_MEGAMOE_COMBINE_DTYPE.get().strip().lower()
    if combine_dtype not in ("bf16", "mxfp8", "nvfp4"):
        raise ValueError(
            "SGLANG_FLASHINFER_MEGAMOE_COMBINE_DTYPE must be one of "
            f"'bf16', 'mxfp8', or 'nvfp4', got {combine_dtype!r}."
        )
    if (
        combine_dtype != "bf16"
        and envs.SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE.get()
    ):
        raise ValueError(
            "SGLANG_FLASHINFER_MEGAMOE_COMBINE_DTYPE="
            f"{combine_dtype!r} is incompatible with "
            "SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE=1."
        )
    return combine_dtype


_SM107_VARIANTS = ("inference", "genphase")
# FlashInfer's GenPhase variant is built for at most this many tokens per rank;
# the SM107 gate applies it to both variants.
_SM107_MAX_TOKENS_PER_RANK = 1024


def _resolve_sm107_variant() -> str | None:
    variant = envs.SGLANG_FLASHINFER_MEGAMOE_SM107_VARIANT.get().strip().lower()
    if not variant:
        return None
    if variant not in _SM107_VARIANTS:
        raise ValueError(
            "SGLANG_FLASHINFER_MEGAMOE_SM107_VARIANT must be empty, 'inference' "
            f"or 'genphase', got {variant!r}."
        )
    return variant


def _resolve_sm107_knobs() -> dict[str, Any] | str | None:
    knobs = envs.SGLANG_FLASHINFER_MEGAMOE_SM107_KNOBS.get().strip()
    if not knobs:
        return None
    if knobs in ("cache", "auto"):
        return knobs
    parsed = json.loads(knobs)
    if not isinstance(parsed, dict):
        raise ValueError(
            "SGLANG_FLASHINFER_MEGAMOE_SM107_KNOBS must be a JSON object, "
            f"'cache' or 'auto', got {knobs!r}."
        )
    # FlashInfer compares tile, cluster and schedule knobs as tuples.
    return {k: tuple(v) if isinstance(v, list) else v for k, v in parsed.items()}


def _sm107_unsupported_reason(layer: FusedMoE) -> str | None:
    if envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.get():
        return "W4A16 activations"
    capability = torch.cuda.get_device_capability()
    if capability != (10, 7):
        return f"compute capability {capability}"
    if layer.moe_ep_size == 1:
        return "expert parallel size 1"
    max_tokens_per_rank = _resolve_max_tokens_per_rank()
    if max_tokens_per_rank > _SM107_MAX_TOKENS_PER_RANK:
        return f"{max_tokens_per_rank} tokens per rank"
    if (
        resolve_flashinfer_megamoe_combine_dtype() != "bf16"
        or envs.SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE.get()
    ):
        return "a quantized combine or the in-kernel FC2 reduce"
    return None


def _flashinfer_megamoe_sm107_variant(layer: FusedMoE) -> str | None:
    variant = _resolve_sm107_variant()
    if variant is None:
        return None
    reason = _sm107_unsupported_reason(layer)
    if reason is not None:
        from sglang.srt.utils import print_warning_once

        print_warning_once(
            f"SGLANG_FLASHINFER_MEGAMOE_SM107_VARIANT={variant} is ignored "
            f"({reason}); FlashInfer MegaMOE keeps its default kernel."
        )
        return None
    return variant


def flashinfer_megamoe_hands_off_epilogue(layer: FusedMoE) -> bool:
    """Whether this MegaMOE layer runs the SM107 kernel with its fused epilogue
    (SGLANG_OPT_FLASHINFER_MEGAMOE_FUSED_EPILOGUE): the top-k reduce writes only
    the live rows, which may go with the shared-expert output to the next layer's
    input norm."""
    return (
        envs.SGLANG_OPT_FLASHINFER_MEGAMOE_FUSED_EPILOGUE.get()
        # The handoff applies no routed scale; it is folded into the top-k weights.
        and layer.should_fuse_routed_scaling_factor_in_topk
        and _flashinfer_megamoe_sm107_variant(layer) is not None
    )


def _layer_ep_world_rank(layer: FusedMoE) -> tuple[int, int]:
    world_size = int(layer.moe_ep_size)
    rank = int(layer.moe_ep_rank)
    if world_size <= 0:
        raise ValueError(f"moe_ep_size must be positive, got {world_size}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"moe_ep_rank must be in [0, {world_size}), got {rank}.")
    return world_size, rank


def _scalar_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().to(torch.float32).max())
    return float(value)


def _local_expert_vector(value: torch.Tensor, num_local_experts: int) -> torch.Tensor:
    value = value.detach().to(torch.float32)
    if value.dim() == 0:
        return value.expand(num_local_experts).contiguous()
    if value.shape != (num_local_experts,):
        raise ValueError(
            f"expected per-local-expert vector of shape ({num_local_experts},), "
            f"got {tuple(value.shape)}"
        )
    return value.contiguous()


def _validate_nvfp4_fc1_alpha(layer: FusedMoE) -> None:
    """MegaMOE reuses ``g1_alphas`` as fc1_alpha; the kernel takes one alpha per
    expert, so the gate and up FC1 alphas must agree."""
    if not layer.moe_runner_config.is_gated:
        return
    gate_alpha = _local_expert_vector(layer.g1_alphas, layer.num_local_experts)
    up_alpha = _local_expert_vector(layer.g1_alphas_up, layer.num_local_experts)
    if not torch.allclose(gate_alpha, up_alpha):
        raise ValueError(
            "FlashInfer NVFP4 MegaMOE requires matching gate/up FC1 alpha "
            "values because the kernel accepts one alpha per expert."
        )


def _bind_transformed_weights(
    layer: FusedMoE,
    transformed_weights: Any,
    *,
    w13_scale_name: str,
    w2_scale_name: str,
) -> None:
    from sglang.srt.layers.utils.common import copy_or_rebind_param

    (w13_weight, w13_scale), (w2_weight, w2_scale) = transformed_weights
    copy_or_rebind_param(layer, "w13_weight", w13_weight)
    copy_or_rebind_param(layer, w13_scale_name, w13_scale)
    copy_or_rebind_param(layer, "w2_weight", w2_weight)
    copy_or_rebind_param(layer, w2_scale_name, w2_scale)


def _init_flashinfer_megamoe_layer_state(layer: FusedMoE) -> None:
    layer._flashinfer_megamoe_layer = None
    layer._flashinfer_megamoe_forward = None
    layer._flashinfer_megamoe_input_norm_const = None
    layer._flashinfer_megamoe_sm107_variant = None


def _get_or_init_flashinfer_megamoe_layer_state(layer: FusedMoE) -> Any:
    if not hasattr(layer, "_flashinfer_megamoe_layer"):
        _init_flashinfer_megamoe_layer_state(layer)
    return layer._flashinfer_megamoe_layer


def _ensure_flashinfer_megamoe_layer(
    layer: FusedMoE,
    *,
    megakernel_config: Any,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> Any:
    mega = _get_or_init_flashinfer_megamoe_layer_state(layer)
    if mega is not None:
        return mega

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
    )

    transformed_weights = (
        (layer.w13_weight.data, w13_scale.data),
        (layer.w2_weight.data, w2_scale.data),
    )
    world_size, rank = _layer_ep_world_rank(layer)

    max_tokens_per_rank = _resolve_max_tokens_per_rank()
    logger.debug(
        "FlashInfer MegaMOE layer[%s] build: megakernel_config=%s "
        "(world_size=%d, num_experts=%d, max_tokens_per_rank=%d, hidden_size=%d)",
        layer.layer_id,
        _format_megakernel_config(megakernel_config),
        world_size,
        layer.num_experts,
        max_tokens_per_rank,
        layer.hidden_size,
    )

    process_group = None
    if world_size > 1:
        from sglang.srt.runtime_context import get_parallel

        process_group = get_parallel().moe_ep_group.device_group
    bootstrap = BootstrapConfig(
        world_size=world_size,
        rank=rank,
        device=torch.cuda.current_device(),
        process_group=process_group,
    )
    fleet_params = FleetParams(
        num_experts=layer.num_experts,
        max_tokens_per_rank=max_tokens_per_rank,
        token_hidden_size=layer.hidden_size,
    )
    backend = MegaConfig(
        megakernel=megakernel_config,
        preprocess_weights=False,
        transformed_weights=transformed_weights,
    )
    mega = MoEEpMegaLayer(
        bootstrap=bootstrap,
        fleet_params=fleet_params,
        # weights already preprocessed in prepare_*; with transformed_weights set
        # the kernel never reads `weights` (see MoEEpMegaLayer), so pass None.
        weights=None,
        backend=backend,
    )
    layer._flashinfer_megamoe_layer = mega
    # SM107 kernels take their tactic from FlashInfer's knobs; SGLang's startup profiles
    # do not apply to them.
    if (
        hasattr(megakernel_config, "knobs")
        and layer._flashinfer_megamoe_sm107_variant is None
    ):
        from sglang.srt.layers.moe.flashinfer_megamoe_autotune import (
            MegaMoeTunedForward,
        )

        layer._flashinfer_megamoe_forward = MegaMoeTunedForward(
            bootstrap, fleet_params, backend
        )
    elif (
        layer._flashinfer_megamoe_sm107_variant is not None
        and flashinfer_megamoe_hands_off_epilogue(layer)
    ):
        from sglang.srt.utils import print_info_once

        print_info_once(
            "FlashInfer MegaMOE fused epilogue: the SM107 top-k reduce writes only "
            "the live rows"
        )
        # An owned [T, H] output: the top-k reduce then covers the T live rows,
        # not the workspace's whole token capacity.
        layer._flashinfer_megamoe_forward = _forward_megamoe_legacy
    else:
        layer._flashinfer_megamoe_forward = _select_megamoe_forward(mega)
    return mega


def ensure_fp4_moe_layer_for_flashinfer_megamoe(layer: FusedMoE) -> Any:
    mega = _get_or_init_flashinfer_megamoe_layer_state(layer)
    if mega is not None:
        return mega

    from flashinfer.moe_ep import DeepGemmMegaMoeConfig

    return _ensure_flashinfer_megamoe_layer(
        layer,
        megakernel_config=DeepGemmMegaMoeConfig(
            intermediate_size=layer.intermediate_size_per_partition,
            top_k=layer.top_k,
            activation_clamp=layer.moe_runner_config.swiglu_limit,
        ),
        w13_scale=layer.w13_weight_scale_inv,
        w2_scale=layer.w2_weight_scale_inv,
    )


def _nvfp4_megamoe_activation_kwargs(config: MoeRunnerConfig) -> dict[str, Any]:
    if config.is_gated and config.activation == "silu":
        return {"activation": "swiglu", "gate_up_clamp": config.swiglu_limit}
    if config.is_gated and config.activation == "situ":
        # Kimi K3 stores SiTU's tanh constants in the existing GEMM1 fields.
        # The up-branch tanh scale is not a SwiGLU hard-clamp limit.
        return {
            "activation": "situ",
            "situ_beta": config.gemm1_alpha,
            "situ_linear_beta": config.gemm1_clamp_limit,
            "gate_up_clamp": None,
        }
    raise ValueError("FlashInfer NVFP4 MegaMOE only supports gated SiLU or SiTU.")


def _sm107_megakernel_config(
    layer: FusedMoE,
    *,
    variant: str,
    input_norm_const: float,
    activation_kwargs: dict[str, Any],
) -> Any:
    from flashinfer.moe_ep import Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig

    from sglang.srt.utils import print_info_once

    knobs = _resolve_sm107_knobs()
    print_info_once(
        f"FlashInfer MegaMOE runs the SM107 {variant} kernel (EP size "
        f"{layer.moe_ep_size}, {_resolve_max_tokens_per_rank()} tokens per rank, "
        f"knobs {knobs if knobs is not None else 'default'})."
    )
    # GenPhase accepts only this cluster shape and the bulk FC2 store.
    variant_fields = (
        {"cluster_shape_mn": (4, 1), "fc2_use_bulk": True}
        if variant == "genphase"
        else {}
    )
    return Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=layer.intermediate_size_per_partition,
        top_k=layer.top_k,
        kernel_variant=variant,
        knobs=knobs,
        input_norm_const=input_norm_const,
        **variant_fields,
        **activation_kwargs,
    )


# The SM107 tail option replaces a method of this release's CuTe DSL kernel sources.
_SM107_KERNEL_OPTIONS_FLASHINFER_VERSION = "0.7.3"


def _install_sm107_kernel_options(sm107_variant: str | None) -> None:
    if not envs.SGLANG_OPT_FLASHINFER_MEGAMOE_SM107_SINGLE_TAIL_BARRIER.get():
        return

    import flashinfer

    from sglang.srt.utils import print_info_once, print_warning_once

    if (
        sm107_variant != "inference"
        or flashinfer.__version__ != _SM107_KERNEL_OPTIONS_FLASHINFER_VERSION
    ):
        print_warning_once(
            "SGLANG_OPT_FLASHINFER_MEGAMOE_SM107_SINGLE_TAIL_BARRIER is ignored: it "
            "patches the SM107 inference kernel of FlashInfer "
            f"{_SM107_KERNEL_OPTIONS_FLASHINFER_VERSION}, found the "
            f"{sm107_variant or 'default'} kernel on FlashInfer {flashinfer.__version__}."
        )
        return

    from sglang.srt.layers.moe.flashinfer_megamoe_sm107_opts import (
        install_sm107_megamoe_single_tail_barrier,
    )

    install_sm107_megamoe_single_tail_barrier()
    print_info_once(
        "FlashInfer MegaMOE SM107 kernel options: single_tail_barrier=True "
        f"(FlashInfer {flashinfer.__version__})."
    )


# The SM107 dispatch options need one router data CTA: at most this many routes per rank.
_SM107_DISPATCH_OPTIONS_MAX_ROUTES_PER_RANK = 16384


def _sm107_dispatch_options_unsupported_reason(
    sm107_variant: str | None, top_k: int, flashinfer_version: str
) -> str | None:
    if (
        sm107_variant != "inference"
        or flashinfer_version != _SM107_KERNEL_OPTIONS_FLASHINFER_VERSION
    ):
        return (
            f"they patch the SM107 inference kernel of FlashInfer "
            f"{_SM107_KERNEL_OPTIONS_FLASHINFER_VERSION}, found the "
            f"{sm107_variant or 'default'} kernel on FlashInfer {flashinfer_version}"
        )
    routes = _resolve_max_tokens_per_rank() * top_k
    if routes > _SM107_DISPATCH_OPTIONS_MAX_ROUTES_PER_RANK:
        return (
            f"{routes} routes per rank exceed the one-data-CTA router's "
            f"{_SM107_DISPATCH_OPTIONS_MAX_ROUTES_PER_RANK}"
        )
    if resolve_flashinfer_megamoe_combine_dtype() != "bf16":
        return "they need the BF16 combine"
    knobs = _resolve_sm107_knobs()
    if isinstance(knobs, str):
        return f"they need a fixed token-back mode, found knobs {knobs!r}"
    if knobs is not None and knobs.get("token_back_mode", "epi_warps") != "epi_warps":
        return (
            f"they need token_back_mode 'epi_warps', found {knobs['token_back_mode']!r}"
        )
    return None


def _install_sm107_dispatch_options(sm107_variant: str | None, top_k: int) -> None:
    requested = [
        name
        for name, env in (
            (
                "tagged_dispatch",
                envs.SGLANG_OPT_FLASHINFER_MEGAMOE_SM107_TAGGED_DISPATCH,
            ),
            (
                "per_cta_push_layout",
                envs.SGLANG_OPT_FLASHINFER_MEGAMOE_SM107_PER_CTA_PUSH_LAYOUT,
            ),
        )
        if env.get()
    ]
    if not requested:
        return

    import flashinfer

    from sglang.srt.utils import print_info_once, print_warning_once

    reason = _sm107_dispatch_options_unsupported_reason(
        sm107_variant, top_k, flashinfer.__version__
    )
    if reason is not None:
        for name in requested:
            print_warning_once(
                f"SGLANG_OPT_FLASHINFER_MEGAMOE_SM107_{name.upper()} is ignored: "
                f"{reason}."
            )
        return

    from sglang.srt.layers.moe.flashinfer_megamoe_sm107_dispatch import (
        install_sm107_megamoe_per_cta_push_layout,
        install_sm107_megamoe_tagged_dispatch,
    )

    installers = {
        "tagged_dispatch": install_sm107_megamoe_tagged_dispatch,
        "per_cta_push_layout": install_sm107_megamoe_per_cta_push_layout,
    }
    # After the single tail barrier option: the tagged dispatch wraps that tail.
    for name in requested:
        installers[name]()
    print_info_once(
        "FlashInfer MegaMOE SM107 dispatch options: "
        + ", ".join(f"{name}=True" for name in requested)
        + f" (FlashInfer {flashinfer.__version__})."
    )


def _install_fused_route_stage(sm107_variant: str | None) -> None:
    if not envs.SGLANG_OPT_FLASHINFER_MEGAMOE_FUSED_ROUTE_QUANT.get():
        return

    import flashinfer

    from sglang.srt.utils import print_info_once, print_warning_once

    # The router reproduces this release's NVFP4 stager and writes its SM107 workspace.
    if (
        sm107_variant != "inference"
        or flashinfer.__version__ != _SM107_KERNEL_OPTIONS_FLASHINFER_VERSION
    ):
        print_warning_once(
            "SGLANG_OPT_FLASHINFER_MEGAMOE_FUSED_ROUTE_QUANT is ignored: it stages the "
            "SM107 inference kernel's inputs as FlashInfer "
            f"{_SM107_KERNEL_OPTIONS_FLASHINFER_VERSION} does, found the "
            f"{sm107_variant or 'default'} kernel on FlashInfer {flashinfer.__version__}."
        )
        return

    from sglang.srt.layers.moe.flashinfer_megamoe_route_stage import (
        install_fused_route_stage,
    )

    install_fused_route_stage()
    print_info_once(
        "FlashInfer MegaMOE: the router stages the SM107 NVFP4 inputs and routes "
        f"in its own launch (FlashInfer {flashinfer.__version__})."
    )


class FlashInferMegaMoeInputStage(msgspec.Struct, frozen=True):
    workspace: Any
    x_q: torch.Tensor  # [capacity, hidden // 2] uint8 view of the NVFP4 rows
    x_sf: torch.Tensor  # [capacity, hidden // 16] uint8 view of the E4M3 scales
    topk_idx: torch.Tensor  # [capacity, top_k] int32
    topk_weights: torch.Tensor  # [capacity, top_k] fp32
    norm_const: float
    scale_multiplier: float


def flashinfer_megamoe_input_stage(
    layer: FusedMoE, x: torch.Tensor
) -> FlashInferMegaMoeInputStage | None:
    """The SM107 workspace views a router may stage ``layer``'s next forward on ``x``
    into, as FlashInfer's fused NVFP4 stager would, or None where that stager would
    not run (SGLANG_OPT_FLASHINFER_MEGAMOE_FUSED_ROUTE_QUANT)."""
    from sglang.srt.layers.moe.flashinfer_megamoe_route_stage import (
        fused_route_stage_installed,
    )

    mega = _get_or_init_flashinfer_megamoe_layer_state(layer)
    if (
        mega is None
        or not fused_route_stage_installed()
        or layer._flashinfer_megamoe_sm107_variant != "inference"
        or not mega._mega_config.quantize_input
    ):
        return None
    # The workspace and backend MoEEpMegaLayer._prepare_inputs resolves (FlashInfer 0.7.3).
    handle = mega._default_workspace_handle
    workspace = mega._workspace if handle is None else handle._backend_workspace
    if workspace is None:
        return None
    kernel = mega._kernel if handle is None else handle._kernel
    num_tokens, hidden = x.shape
    # stage_mega_moe_inputs' conditions for its fused stager; otherwise it quantizes in torch.
    if not (
        os.environ.get("FLASHINFER_MEGA_FUSED_STAGE", "1") != "0"
        and 0 < num_tokens <= workspace.x.shape[0]
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
        and hidden % 64 == 0
        and workspace.x.shape[1] * 2 == hidden
        and workspace.x_sf.shape[1] == hidden // 16
        and workspace.topk_idx.dtype == torch.int32
        and workspace.topk_idx.shape[1] == layer.top_k
    ):
        return None
    input_norm_const = kernel._kernel_config.input_norm_const
    return FlashInferMegaMoeInputStage(
        workspace=workspace,
        x_q=workspace.x.view(torch.uint8),
        x_sf=workspace.x_sf.view(torch.uint8),
        topk_idx=workspace.topk_idx,
        topk_weights=workspace.topk_weights,
        norm_const=input_norm_const,
        # The stager's scale_multiplier, rounded to fp32 at launch as there.
        scale_multiplier=input_norm_const / 6.0,
    )


def ensure_nvfp4_moe_layer_for_flashinfer_megamoe(layer: FusedMoE) -> Any:
    mega = _get_or_init_flashinfer_megamoe_layer_state(layer)
    if mega is not None:
        return mega

    # Before the layer's first forward, which compiles the kernel.
    _install_sm107_kernel_options(layer._flashinfer_megamoe_sm107_variant)
    _install_sm107_dispatch_options(
        layer._flashinfer_megamoe_sm107_variant, layer.top_k
    )
    _install_fused_route_stage(layer._flashinfer_megamoe_sm107_variant)
    activation_kwargs = _nvfp4_megamoe_activation_kwargs(layer.moe_runner_config)
    if envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.get():
        from flashinfer.moe_ep import Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig

        megakernel_config = Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=layer.intermediate_size_per_partition,
            top_k=layer.top_k,
            enable_in_kernel_fc2_reduce=envs.SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE.get(),
            **activation_kwargs,
        )
    elif layer._flashinfer_megamoe_sm107_variant is not None:
        megakernel_config = _sm107_megakernel_config(
            layer,
            variant=layer._flashinfer_megamoe_sm107_variant,
            input_norm_const=layer._flashinfer_megamoe_input_norm_const,
            activation_kwargs=activation_kwargs,
        )
    else:
        from flashinfer.moe_ep import Nvfp4CutedslMegaMoeConfig

        input_norm_const = layer._flashinfer_megamoe_input_norm_const
        if input_norm_const is None:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "FlashInfer NVFP4 MegaMOE layer must be initialized before "
                    "CUDA graph capture."
                )
            logger.warning(
                "FlashInfer NVFP4 MegaMOE layer[%s]: input_norm_const was not "
                "precomputed at weight-load time; computing it lazily now via a "
                "blocking device sync. This should only happen once per layer, "
                "but if it happens during warmup it will look like a stall.",
                layer.layer_id,
            )
            input_norm_const = _scalar_float(layer.w13_input_scale_quant)
            layer._flashinfer_megamoe_input_norm_const = input_norm_const

        megakernel_config = Nvfp4CutedslMegaMoeConfig(
            intermediate_size=layer.intermediate_size_per_partition,
            top_k=layer.top_k,
            apply_topk_in_fc1=True,
            enable_in_kernel_fc2_reduce=envs.SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE.get(),
            combine_dtype=resolve_flashinfer_megamoe_combine_dtype(),
            input_norm_const=input_norm_const,
            **activation_kwargs,
        )

    return _ensure_flashinfer_megamoe_layer(
        layer,
        megakernel_config=megakernel_config,
        w13_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
    )


def ensure_mxfp8_moe_layer_for_flashinfer_megamoe(layer: FusedMoE) -> Any:
    mega = _get_or_init_flashinfer_megamoe_layer_state(layer)
    if mega is not None:
        return mega

    from flashinfer.moe_ep import Mxfp8CutedslMegaMoeConfig

    return _ensure_flashinfer_megamoe_layer(
        layer,
        megakernel_config=Mxfp8CutedslMegaMoeConfig(
            intermediate_size=layer.intermediate_size_per_partition,
            top_k=layer.top_k,
            kind="mxfp8_e4m3",
            gate_up_clamp=layer.moe_runner_config.swiglu_limit,
            enable_in_kernel_fc2_reduce=envs.SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE.get(),
        ),
        w13_scale=layer.w13_weight_scale_inv,
        w2_scale=layer.w2_weight_scale_inv,
    )


def prepare_fp4_moe_weights_for_flashinfer_megamoe(
    layer: FusedMoE,
) -> None:
    """Prepare loaded FP4 weights for MegaMOE.

    SGLang loads FP4-packed expert weights plus raw block scales. FlashInfer's
    current moe_ep API owns backend-specific weight preprocessing, including
    DeepGEMM scale layout transforms.
    """
    if (
        not layer.moe_runner_config.is_gated
        or layer.moe_runner_config.activation != "silu"
    ):
        raise ValueError("FlashInfer DeepGEMM MegaMOE only supports gated SiLU.")
    _init_flashinfer_megamoe_layer_state(layer)

    from flashinfer.moe_ep import (
        MoEWeightPack,
        preprocess_mega_weights,
    )

    weights = MoEWeightPack(
        w13=layer.w13_weight.data,
        w2=layer.w2_weight.data,
        w13_scale=layer.w13_weight_scale_inv.data,
        w2_scale=layer.w2_weight_scale_inv.data,
    )
    transformed_weights = preprocess_mega_weights(
        weights,
        intermediate_size=layer.intermediate_size_per_partition,
        hidden_size=layer.hidden_size,
    )
    _bind_transformed_weights(
        layer,
        transformed_weights,
        w13_scale_name="w13_weight_scale_inv",
        w2_scale_name="w2_weight_scale_inv",
    )


def make_nvfp4_megamoe_weight_loader(
    layer: FusedMoE, weight_loader: Callable
) -> Callable:
    def load_weights(*args, **kwargs):
        # Reload into canonical views of the existing prepared allocations.
        # The online loader also writes block scales through its inner loader,
        # so restore all four planes before handing it the first source tensor.
        for name, (load_view, _) in getattr(
            layer, "_flashinfer_megamoe_weight_views", {}
        ).items():
            getattr(layer, name).data = load_view
        return weight_loader(*args, **kwargs)

    return load_weights


def _prepare_nvfp4_weights_for_sm107(layer: FusedMoE, *, variant: str) -> None:
    # The SM107 layout interleaves gate/up rows and transposes the weights in new
    # storage, so the checkpoint loader cannot write into it again.
    if layer._flashinfer_megamoe_sm107_variant is not None:
        raise ValueError(
            "SGLANG_FLASHINFER_MEGAMOE_SM107_VARIANT does not support weight reload."
        )
    from flashinfer.moe_ep import MoEWeightPack, preprocess_sm107_nvfp4_mega_weights

    transformed_weights = preprocess_sm107_nvfp4_mega_weights(
        MoEWeightPack(
            w13=layer.w13_weight.data,
            w2=layer.w2_weight.data,
            w13_scale=layer.w13_weight_scale.data,
            w2_scale=layer.w2_weight_scale.data,
        ),
        intermediate_size=layer.intermediate_size_per_partition,
        hidden_size=layer.hidden_size,
    )
    _bind_transformed_weights(
        layer,
        transformed_weights,
        w13_scale_name="w13_weight_scale",
        w2_scale_name="w2_weight_scale",
    )
    layer._flashinfer_megamoe_input_norm_const = _scalar_float(
        layer.w13_input_scale_quant
    )
    layer._flashinfer_megamoe_sm107_variant = variant


def prepare_nvfp4_moe_weights_for_flashinfer_megamoe(
    layer: FusedMoE,
) -> None:
    use_w4a16 = envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.get()
    # Both NVFP4 modes reload into the same prepared allocations. Preserve the
    # live layer, tuned profiles and pooled workspace owner for CUDA graphs.
    mega = _get_or_init_flashinfer_megamoe_layer_state(layer)

    from flashinfer.moe_ep import MoEWeightPack

    activation_kwargs = _nvfp4_megamoe_activation_kwargs(layer.moe_runner_config)
    if layer.moe_runner_config.apply_router_weight_on_input:
        raise ValueError(
            "FlashInfer NVFP4 MegaMOE does not support routing weights on input."
        )

    hidden_alignment = 32 if use_w4a16 else 128
    intermediate_alignment = 64 if use_w4a16 else 128
    if layer.hidden_size % hidden_alignment != 0:
        raise ValueError(
            "FlashInfer NVFP4 MegaMOE requires hidden_size to be a multiple "
            f"of {hidden_alignment}, got {layer.hidden_size}."
        )
    if layer.quant_config.use_per_token_activation and not use_w4a16:
        raise ValueError(
            "FlashInfer NVFP4 MegaMOE does not support per-token activation "
            "scaling. Use flashinfer_trtllm/flashinfer_trtllm_routed for "
            "ModelOpt NVFP4 per-token activation."
        )
    if layer.intermediate_size_per_partition % intermediate_alignment != 0:
        raise ValueError(
            "FlashInfer NVFP4 MegaMOE requires intermediate_size_per_partition "
            f"to be a multiple of {intermediate_alignment}, "
            f"got {layer.intermediate_size_per_partition}."
        )
    if layer.num_experts % layer.moe_ep_size != 0:
        raise ValueError(
            "FlashInfer NVFP4 MegaMOE requires num_experts to be divisible by "
            f"ep_size, got {layer.num_experts=} and {layer.moe_ep_size=}."
        )

    _validate_nvfp4_fc1_alpha(layer)
    sm107_variant = (
        layer._flashinfer_megamoe_sm107_variant
        or _flashinfer_megamoe_sm107_variant(layer)
    )
    if sm107_variant is not None:
        _prepare_nvfp4_weights_for_sm107(layer, variant=sm107_variant)
        return
    weight_views = getattr(layer, "_flashinfer_megamoe_weight_views", None)
    if weight_views is None:
        load_layouts = {}
        for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
            param = getattr(layer, name)
            load_layouts[name] = (param.shape, param.stride(), param.dtype)
    weights = MoEWeightPack(
        w13=layer.w13_weight.data,
        w2=layer.w2_weight.data,
        w13_scale=layer.w13_weight_scale.data,
        w2_scale=layer.w2_weight_scale.data,
    )
    if use_w4a16:
        from flashinfer.moe_ep import preprocess_bf16_nvfp4_cutedsl_mega_weights

        transformed_weights = preprocess_bf16_nvfp4_cutedsl_mega_weights(
            weights,
            intermediate_size=layer.intermediate_size_per_partition,
            hidden_size=layer.hidden_size,
        )
    else:
        from flashinfer.moe_ep import preprocess_nvfp4_cutedsl_mega_weights

        input_norm_const = _scalar_float(layer.w13_input_scale_quant)
        if (
            mega is not None
            and input_norm_const != layer._flashinfer_megamoe_input_norm_const
        ):
            # FlashInfer captures this scalar by value; updating Python config
            # would leave existing CUDA graphs using the old normalization.
            raise ValueError(
                "FlashInfer NVFP4 W4A4 MegaMOE weight reload requires an unchanged "
                "input activation scale. Restart the server to change it."
            )
        layer._flashinfer_megamoe_input_norm_const = input_norm_const
        transformed_weights = preprocess_nvfp4_cutedsl_mega_weights(
            weights,
            intermediate_size=layer.intermediate_size_per_partition,
            hidden_size=layer.hidden_size,
            gate_up_clamp=activation_kwargs["gate_up_clamp"],
            activation_clamp=None,
        )
    if weight_views is not None:
        for name, (_, prepared_view) in weight_views.items():
            getattr(layer, name).data = prepared_view
    _bind_transformed_weights(
        layer,
        transformed_weights,
        w13_scale_name="w13_weight_scale",
        w2_scale_name="w2_weight_scale",
    )
    if weight_views is None:
        # Padded scale storage can exceed the canonical load shape. as_strided
        # exposes only the original extent without allocating another copy.
        layer._flashinfer_megamoe_weight_views = {}
        for name, (shape, stride, dtype) in load_layouts.items():
            prepared_view = getattr(layer, name).data
            load_view = torch.as_strided(prepared_view.view(dtype), shape, stride)
            layer._flashinfer_megamoe_weight_views[name] = (load_view, prepared_view)


def prepare_mxfp8_moe_weights_for_flashinfer_megamoe(
    layer: FusedMoE,
) -> None:
    if (
        not layer.moe_runner_config.is_gated
        or layer.moe_runner_config.activation != "silu"
    ):
        raise ValueError("FlashInfer MXFP8 MegaMOE only supports gated SiLU.")
    _init_flashinfer_megamoe_layer_state(layer)

    from flashinfer.moe_ep import (
        MoEWeightPack,
        preprocess_mxfp8_cutedsl_mega_weights,
    )

    if layer.hidden_size % 128 != 0:
        raise ValueError(
            "FlashInfer MXFP8 MegaMOE requires hidden_size to be a multiple "
            f"of 128, got {layer.hidden_size}."
        )
    if layer.intermediate_size_per_partition % 128 != 0:
        raise ValueError(
            "FlashInfer MXFP8 MegaMOE requires intermediate_size_per_partition "
            f"to be a multiple of 128, got {layer.intermediate_size_per_partition}."
        )
    if layer.num_experts % layer.moe_ep_size != 0:
        raise ValueError(
            "FlashInfer MXFP8 MegaMOE requires num_experts to be divisible by "
            f"ep_size, got {layer.num_experts=} and {layer.moe_ep_size=}."
        )

    weights = MoEWeightPack(
        w13=layer.w13_weight.data,
        w2=layer.w2_weight.data,
        w13_scale=layer.w13_weight_scale_inv.data,
        w2_scale=layer.w2_weight_scale_inv.data,
    )
    transformed_weights = preprocess_mxfp8_cutedsl_mega_weights(
        weights,
        intermediate_size=layer.intermediate_size_per_partition,
        hidden_size=layer.hidden_size,
        kind="mxfp8_e4m3",
        gate_up_clamp=layer.moe_runner_config.swiglu_limit,
        activation_clamp=None,
    )
    _bind_transformed_weights(
        layer,
        transformed_weights,
        w13_scale_name="w13_weight_scale_inv",
        w2_scale_name="w2_weight_scale_inv",
    )


@register_fused_func("flashinfer_megamoe", "flashinfer_megamoe")
def run_flashinfer_megamoe(
    dispatch_output: DispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """Run the fused mega kernel and return per-rank outputs (no combine)."""
    from flashinfer.moe_ep import MoEEpTensors

    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    assert isinstance(quant_info, FlashInferMegaMoeQuantInfo), (
        f"Unexpected quant_info type for flashinfer_megamoe: {type(quant_info)}"
    )

    x = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights
    topk_ids = topk_output.topk_ids
    mega = quant_info.mega

    t = MoEEpTensors(
        hidden_states=x.to(torch.bfloat16),
        # FlashInfer's fused staging accepts the int32 router output and widens
        # directly into its final int64 workspace buffer. Keep this path copy-free.
        topk_ids=topk_ids,
        topk_weights=topk_weights.to(torch.float32),
        # Stage per-layer scales on every call so FlashInfer can share its
        # reference-counted workspace across layers and observe weight reloads.
        fc1_alpha=quant_info.fc1_alpha,
        fc2_alpha=quant_info.fc2_alpha,
        fc1_norm_const=quant_info.fc1_norm_const,
    )
    with _capture_safe_ue8m0_pack():
        assert quant_info.mega_forward is not None
        y = quant_info.mega_forward(mega, t)

    if quant_info.apply_routed_scaling_factor:
        rsf = runner_config.routed_scaling_factor
        if rsf is not None and rsf != 1.0:
            y.mul_(rsf)

    return StandardCombineInput(hidden_states=y)


def warmup_all_flashinfer_megamoe_layers(model: torch.nn.Module) -> None:
    """Force every FlashInfer MegaMOE layer to build before CUDA graph capture.
    ``ensure_*_moe_layer_for_flashinfer_megamoe`` builds a layer's state
    lazily on its first forward; if that first forward instead happens inside
    a CUDA graph capture, the lazy build's blocking device sync isn't allowed
    and it raises rather than silently recovering. Call this once, explicitly,
    right before capture begins (see ``ModelRunner.init_cuda_graphs``) so
    every layer is built eagerly outside any graph.
    Only the nvfp4 path is wired up below; extend the dispatch if fp4/mxfp8
    MegaMOE hits the same gap.
    """
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    n_built = 0
    for module in model.modules():
        if not isinstance(module, FusedMoE):
            continue
        if not hasattr(module, "_flashinfer_megamoe_layer"):
            # This layer's quant method never went through one of the
            # prepare_*_moe_weights_for_flashinfer_megamoe hooks -- not a
            # MegaMOE layer (or not on the flashinfer_megamoe backend).
            continue
        if getattr(module, "_flashinfer_megamoe_layer", None) is not None:
            continue  # already built (e.g. warmup's dummy batch hit it)

        # Dispatch mirrors modelopt_quant.py's apply(): only the nvfp4
        # method is wired up here today.
        if type(module.quant_method).__name__ == "ModelOptNvFp4FusedMoEMethod":
            ensure_nvfp4_moe_layer_for_flashinfer_megamoe(module)
            n_built += 1
        else:
            logger.warning(
                "warmup_all_flashinfer_megamoe_layers: layer[%s] uses "
                "quant_method=%s, which this eager pre-capture warmup does "
                "not know how to build. If capture then fails with "
                "'must be initialized before CUDA graph capture', add a "
                "branch here for that quant method.",
                getattr(module, "layer_id", "?"),
                type(module.quant_method).__name__,
            )

    if n_built:
        logger.info(
            "warmup_all_flashinfer_megamoe_layers: eagerly built %d "
            "FlashInfer MegaMOE layer(s) before CUDA graph capture.",
            n_built,
        )
