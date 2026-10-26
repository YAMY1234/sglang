"""FlashInfer MNNVL CuTe DSL AllReduce fusion, shared across architectures.

Two patterns share one workspace, both consumed at the next layer's input
RMSNorm (and, with terminal_finalize, at the final norm): AR + residual +
RMSNorm, and the same with the MoE finalize and the shared-expert add folded in
when the runner hands back a MoeDeferredFinalize.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Callable, Optional, Sequence

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layer_boundary import (
    DeferredFinalize,
    FfnInputFusion,
    SumGroup,
    TokenAxis,
    UnreducedOutput,
    get_attn_tp_context,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import (
    cutedsl_moe_max_num_tokens,
    get_disagg,
    get_exec,
    get_flags,
    get_parallel,
)

_LayerPredicate = Callable[[torch.nn.Module], bool]

logger = logging.getLogger(__name__)


def _fused_norm_gamma(layernorm: torch.nn.Module) -> Optional[torch.Tensor]:
    """The multiplier as applied -- GemmaRMSNorm's pre-folded w + 1. None declines."""
    if isinstance(layernorm, GemmaRMSNorm):
        return layernorm.gemma_weight
    if isinstance(layernorm, RMSNorm) and layernorm.has_weight:
        return layernorm.weight
    return None


def _is_supported_forward_mode(forward_mode: ForwardMode) -> bool:
    return forward_mode in (
        ForwardMode.DECODE,
        ForwardMode.EXTEND,
        ForwardMode.TARGET_VERIFY,
    )


class _R13OverlapSite:
    """One registered attention/GDN output projection and its graph events."""

    __slots__ = ("events", "fusion", "linear", "plan", "service")

    def __init__(self, service, fusion, plan, linear) -> None:
        self.service = service
        self.fusion = fusion
        self.plan = plan
        self.linear = linear
        self.events = None


# The megakernel plugin reaches the boundary service from the exact RowParallelLinear
# object it is replacing. Keep a strong reference and verify identity on lookup so a
# recycled Python id can never select a stale site.
_R13_OVERLAP_SITES: dict[int, _R13OverlapSite] = {}


def try_r13_t4comm_overlap_output_projection(
    *,
    linear: torch.nn.Module,
    q_input: torch.Tensor,
    out_dtype: torch.dtype,
    forward_batch: ForwardBatch,
) -> Optional[torch.Tensor]:
    """Run the R13 split projection/AR experiment for a registered output.

    ``None`` declines without changing work. A returned tensor is the same TP
    partial the ordinary boundary expects; its fused AR outputs are retained by
    the service until that boundary immediately consumes the partial.
    """
    if not envs.SGLANG_R13_T4COMM_OVERLAP.get() or torch.compiler.is_compiling():
        return None
    site = _R13_OVERLAP_SITES.get(id(linear))
    if site is None or site.linear is not linear:
        return None
    return site.service.try_r13_overlap_output_projection(
        site=site,
        q_input=q_input,
        out_dtype=out_dtype,
        forward_batch=forward_batch,
    )


def _resolve_max_m(*, max_running_requests: int | None) -> int:
    decode_config = get_exec().graph.cuda_graph_config.decode
    prefill_config = get_exec().graph.cuda_graph_config.prefill
    candidates = [
        cutedsl_moe_max_num_tokens(),
        max_running_requests,
        decode_config.max_bs,
        prefill_config.max_bs,
        *(decode_config.bs or []),
        *(prefill_config.bs or []),
    ]
    positive = [
        int(value) for value in candidates if value is not None and int(value) > 0
    ]
    if not positive:
        raise RuntimeError("framework reported no positive fusion workspace M bound")
    return max(positive)


class MoeDeferredFinalize(DeferredFinalize, frozen=True):
    """Unfinalized routed output plus the separately gated shared contribution.
    The next layer's fused finalize + AR + add + norm takes it; ``finish`` is
    the MoE's own unfused tail, for any other reader."""

    routed_output: torch.Tensor
    expert_weights: torch.Tensor
    permuted_indices: torch.Tensor
    gated_shared_output: torch.Tensor | None
    m: int
    finish: Callable[[], torch.Tensor]

    def complete(self) -> torch.Tensor:
        return self.finish()

    @classmethod
    def from_flashinfer(
        cls,
        deferred_output,
        *,
        gated_shared_output: torch.Tensor | None,
        m: int,
        reduce: Callable[[torch.Tensor], torch.Tensor],
    ) -> MoeDeferredFinalize:
        """``reduce`` is the MoE's own sum of its finalized output."""
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            finalize_flashinfer_trtllm_deferred_output,
        )

        top_k = int(deferred_output.top_k)
        return cls(
            routed_output=deferred_output.gemm2_out.view(
                -1, deferred_output.gemm2_out.shape[-1]
            ),
            expert_weights=deferred_output.expert_weights.view(-1, top_k)[:m],
            permuted_indices=deferred_output.expanded_idx_to_permuted_idx.view(
                -1, top_k
            )[:m],
            gated_shared_output=gated_shared_output,
            m=int(m),
            finish=lambda: reduce(
                finalize_flashinfer_trtllm_deferred_output(
                    deferred_output, gated_shared_output
                )
            ),
        )


def _static_fp8_bmm_input_scale(linear: torch.nn.Module) -> Optional[torch.Tensor]:
    """The input scale of a ModelOpt static FP8 linear on FlashInfer's BMM (which
    quantizes with static_quant_fp8, no repeated scale), else None."""
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod

    method = linear.quant_method
    if not (
        isinstance(method, ModelOptFp8LinearMethod)
        and not method.use_marlin
        and not method.use_sm120_fp8
        and linear.use_flashinfer_bmm
    ):
        return None
    scale = linear.input_scale
    if scale.numel() != 1 or scale.dtype != torch.float32:
        return None
    return scale


class CuteDSLWorkspace:
    def __init__(
        self,
        *,
        hidden_size: int,
        top_k: int,
        rms_epsilon: float,
        include_shared_expert: bool = True,
        fp8_inputs: Sequence[tuple[torch.nn.Module, torch.nn.Module]] = (),
    ) -> None:
        self.include_shared_expert = include_shared_expert
        self.hidden_size = int(hidden_size)
        self.top_k = int(top_k)
        self.rms_epsilon = float(rms_epsilon)
        self.max_m: int | None = None
        self._workspace = None
        # (input norm, the projection reading it) a layer, for SGLANG_PLB_AR_FP8_OUT;
        # keyed by the norm's fused gamma at first use, once weights are loaded.
        self._fp8_inputs = tuple(fp8_inputs)
        self._fp8_input_by_gamma: dict[int, torch.nn.Module] | None = None
        self._r13_overlap_sites: list[_R13OverlapSite] = []
        self._r13_overlap_pending = None

    def prepare(self, *, max_m: int) -> None:
        if self._workspace is not None:
            assert self.max_m is not None
            if int(max_m) > self.max_m:
                raise RuntimeError(
                    f"fusion workspace is already prepared for M_max={self.max_m}; "
                    f"refusing M_max={max_m}"
                )
            return
        from sglang.srt.layers.flashinfer_mnnvl_cutedsl import (
            get_flashinfer_mnnvl_cutedsl_ar_fusion,
        )

        workspace = get_flashinfer_mnnvl_cutedsl_ar_fusion(
            hidden_size=self.hidden_size,
            top_k=self.top_k,
            max_m=int(max_m),
            rms_epsilon=self.rms_epsilon,
            # _fused_norm_gamma() already returns the multiplier as applied.
            weight_bias=0.0,
            include_shared_expert=self.include_shared_expert,
        )
        self._workspace = workspace
        self.max_m = workspace.max_m
        if envs.SGLANG_R13_T4COMM_OVERLAP.get():
            if not envs.SGLANG_PERFLP_AR_FP8_PAYLOAD.get():
                raise RuntimeError(
                    "R13-T4COMM overlap requires the unchanged R11 E4M3 payload"
                )
            if not self._r13_overlap_sites:
                raise RuntimeError(
                    "R13-T4COMM overlap requested without registered output projections"
                )
            for site in self._r13_overlap_sites:
                site.events = tuple(torch.cuda.Event() for _ in range(3))
            logger.info(
                "R13-T4COMM prepared %d registered output projections for "
                "split GEMM/AR overlap",
                len(self._r13_overlap_sites),
            )

    def register_r13_overlap_output(self, fusion, plan, linear) -> None:
        if not envs.SGLANG_R13_T4COMM_OVERLAP.get():
            return
        if self._workspace is not None:
            raise RuntimeError("register R13 overlap outputs before prepare()")
        site = _R13OverlapSite(self, fusion, plan, linear)
        old = _R13_OVERLAP_SITES.get(id(linear))
        if old is not None and old.linear is linear:
            raise RuntimeError("R13 overlap output projection registered twice")
        _R13_OVERLAP_SITES[id(linear)] = site
        self._r13_overlap_sites.append(site)

    def try_r13_overlap_output_projection(
        self,
        *,
        site: _R13OverlapSite,
        q_input: torch.Tensor,
        out_dtype: torch.dtype,
        forward_batch: ForwardBatch,
    ) -> Optional[torch.Tensor]:
        if not (
            forward_batch.forward_mode == ForwardMode.TARGET_VERIFY
            and q_input.ndim == 2
            and out_dtype == torch.bfloat16
            and site.events is not None
            and self._workspace is not None
            and site.fusion._eligible(site.plan, forward_batch, int(q_input.shape[0]))
            and not get_exec().comm.enable_quant_communications
        ):
            return None
        from sglang.srt.layers.layer_boundary.residual.batch import stream_of

        stream = stream_of(forward_batch)
        residual = stream.residual
        gamma = _fused_norm_gamma(site.plan.norm)
        # During attention/mixer compute the input boundary has already written
        # the residual and no producer contribution has yet been recorded.
        if stream.pending is not None or residual is None or gamma is None:
            return None
        if self._r13_overlap_pending is not None:
            raise RuntimeError(
                "R13 overlap output reached the next projection before its "
                "layer boundary consumed the previous result"
            )
        result = self._workspace.r13_overlap_fp8_projection(
            q_input=q_input,
            weight=site.linear.weight,
            input_scale=site.linear.input_scale,
            weight_scale=site.linear.weight_scale,
            out_dtype=out_dtype,
            residual=residual,
            gamma=gamma,
            events=site.events,
        )
        if result is None:
            return None
        partial, norm_output, residual_output = result
        self._r13_overlap_pending = (
            partial,
            residual,
            gamma,
            norm_output,
            residual_output,
        )
        return partial

    def take_r13_overlap(
        self,
        *,
        partial: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        pending = self._r13_overlap_pending
        if pending is None:
            return None
        if not (
            pending[0] is partial and pending[1] is residual and pending[2] is gamma
        ):
            raise RuntimeError(
                "R13 overlap result does not match the immediately following "
                "layer-boundary consumer"
            )
        self._r13_overlap_pending = None
        return pending[3], pending[4]

    def supports(self, m: int) -> bool:
        """False before prepare(), so it doubles as the readiness check."""
        return self._workspace is not None and self._workspace.supports(m)

    def _fp8_input_scale(self, gamma: torch.Tensor) -> Optional[torch.Tensor]:
        """The static input scale of the FP8 projection reading the norm with ``gamma``."""
        if not self._fp8_inputs or torch.compiler.is_compiling():
            return None
        if self._fp8_input_by_gamma is None:
            self._fp8_input_by_gamma = {
                _fused_norm_gamma(norm).data_ptr(): linear
                for norm, linear in self._fp8_inputs
                if _fused_norm_gamma(norm) is not None
            }
        linear = self._fp8_input_by_gamma.get(gamma.data_ptr())
        return None if linear is None else _static_fp8_bmm_input_scale(linear)

    def finalize(
        self,
        handoff: MoeDeferredFinalize,
        residual: torch.Tensor,
        gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._workspace is not None
        input_scale = self._fp8_input_scale(gamma)
        if input_scale is not None:
            from sglang.kernels.ops.quantization.fp8_kernel import (
                hand_over_static_quant_fp8,
            )

            norm, residual_out, norm_fp8 = (
                self._workspace.moe_finalize_all_reduce_rms_norm_fp8(
                    routed_output=handoff.routed_output,
                    expert_weights=handoff.expert_weights,
                    permuted_indices=handoff.permuted_indices,
                    gated_shared_output=handoff.gated_shared_output,
                    residual=residual,
                    gamma=gamma,
                    input_scale=input_scale,
                )
            )
            if norm_fp8 is not None:
                hand_over_static_quant_fp8(norm, input_scale, norm_fp8)
            return norm, residual_out
        return self._workspace.moe_finalize_all_reduce_rms_norm(
            routed_output=handoff.routed_output,
            expert_weights=handoff.expert_weights,
            permuted_indices=handoff.permuted_indices,
            gated_shared_output=handoff.gated_shared_output,
            residual=residual,
            gamma=gamma,
        )

    def all_reduce_residual_rms_norm(
        self,
        local_contribution: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._workspace is not None
        return self._workspace.all_reduce_residual_rms_norm(
            local_contribution=local_contribution,
            residual=residual,
            gamma=gamma,
        )


class CuteDSLFusion:
    """One layer's CuTe DSL kernels, given to its stage at
    construction. At the attention input: the MoE finalize + AR + add + norm of
    a handoff the previous layer left, and the AR + add + norm of a sum it left.
    At the FFN input: the attention output's AR + add + norm. At the FFN exit:
    handing off a deferred MoE finalize with its unfused completion.
    install_cutedsl_fusion supplies the service and producer capability;
    consumer eligibility is checked only where the output is consumed. Producer
    policy: can_defer_all_reduce keeps a LoRA/TP1 shared-expert sum deferrable
    while the workspace can take it; can_defer_finalize also covers a terminal
    FFN when terminal_finalize is installed."""

    def __init__(self) -> None:
        self.service: CuteDSLWorkspace | None = None
        # Producer capabilities; consumer kernel selection is independent.
        self.defers_finalize = False
        self.terminal_finalize = False

    def install(
        self,
        service: CuteDSLWorkspace,
        *,
        defers_finalize: bool,
        terminal_finalize: bool = False,
    ) -> None:
        """Install the shared workspace and this producer's capabilities."""
        self.service = service
        self.defers_finalize = defers_finalize
        self.terminal_finalize = terminal_finalize

    def attn_input_fusions(self, plan) -> tuple:
        return (
            partial(self._finalize_add_norm, plan),
            partial(self._all_reduce_add_norm, plan),
        )

    def ffn_input_fusions(self, plan) -> tuple:
        parallel = get_parallel()
        if (
            TokenAxis.ATTN_TP not in plan.incoming_residual_rows.sharded
            # The workspace sums over TP, which is then the attention-TP group.
            and parallel.attn_tp_size == parallel.tp_size
            and _fused_norm_gamma(plan.norm) is not None
        ):
            return (
                FfnInputFusion(
                    completes=SumGroup.ATTN_TP,
                    run=partial(
                        self._ffn_input_all_reduce_add_norm,
                        plan,
                    ),
                ),
            )
        return ()

    def _finalize_add_norm(
        self, plan, owed, residual, forward_batch, post_residual_addition
    ):
        """Finish a deferred MoE finalize with the all-reduce, residual add and
        input norm, when this layer's kernel takes the batch; otherwise the
        handoff is completed by its producer's own tail."""
        if not isinstance(owed, MoeDeferredFinalize):
            return None
        gamma = _fused_norm_gamma(plan.norm)
        if gamma is None or not self._finalize_eligible(plan, forward_batch, owed.m):
            return None
        if post_residual_addition is not None:
            residual = residual + post_residual_addition
        assert self.service is not None
        return self.service.finalize(handoff=owed, residual=residual, gamma=gamma)

    def _all_reduce_add_norm(
        self, plan, owed, residual, forward_batch, post_residual_addition
    ):
        """Complete the all-reduce the previous layer left with the residual add
        and input norm."""
        if not isinstance(owed, UnreducedOutput) or not (
            self._all_reduce_eligible(plan, forward_batch, int(owed.partial.shape[0]))
        ):
            return None
        if post_residual_addition is not None:
            residual = residual + post_residual_addition
        assert self.service is not None
        return self.service.all_reduce_residual_rms_norm(
            local_contribution=owed.partial,
            residual=residual,
            gamma=_fused_norm_gamma(plan.norm),
        )

    def _ffn_input_all_reduce_add_norm(
        self, plan, hidden_states, residual, forward_batch
    ):
        """The attention output's all-reduce with the residual add and the
        post-attention norm."""
        if not (
            self._eligible(plan, forward_batch, int(hidden_states.shape[0]))
            and residual is not None
            and not get_exec().comm.enable_quant_communications
        ):
            return None
        assert self.service is not None
        gamma = _fused_norm_gamma(plan.norm)
        overlapped = self.service.take_r13_overlap(
            partial=hidden_states,
            residual=residual,
            gamma=gamma,
        )
        if overlapped is not None:
            return overlapped
        return self.service.all_reduce_residual_rms_norm(
            local_contribution=hidden_states,
            residual=residual,
            gamma=gamma,
        )

    def _finalize_eligible(self, plan, forward_batch: ForwardBatch, m: int) -> bool:
        return self._eligible(plan, forward_batch, m) and (
            get_parallel().moe_ep_size == 1 or envs.SGLANG_TRTL3_EP_DEFER_FINALIZE.get()
        )

    def _all_reduce_eligible(self, plan, forward_batch: ForwardBatch, m: int) -> bool:
        """Incoming, and independent of this layer's own successor."""
        return (
            self._eligible(plan, forward_batch, m)
            and _fused_norm_gamma(plan.norm) is not None
            and not get_exec().comm.enable_quant_communications
        )

    def can_defer_all_reduce(self, plan, forward_batch: ForwardBatch) -> bool:
        """Preserve the producer's fused sum path when this workspace is usable.

        The next consumer still selects its kernel and completes the sum with
        an ordinary all-reduce if that kernel declines the actual input.
        """
        return (
            self._eligible(plan, forward_batch, int(forward_batch.input_ids.shape[0]))
            and not get_exec().comm.enable_quant_communications
        )

    def can_defer_finalize(self, plan, forward_batch: ForwardBatch) -> bool:
        """Whether this producer can emit a handoff with an unfused fallback."""
        return (
            self.defers_finalize
            and (not plan.terminal or self.terminal_finalize)
            and self._finalize_eligible(
                plan, forward_batch, int(forward_batch.input_ids.shape[0])
            )
        )

    def _eligible(self, plan, forward_batch: ForwardBatch, m: int) -> bool:
        parallel = get_parallel()
        return bool(
            self.service is not None
            and _is_supported_forward_mode(forward_batch.forward_mode)
            and self.service.supports(m)
            and not is_dp_attention_enabled()
            # Also forces moe_dp_size == 1, so a hybrid EP x MoE-TP reduction
            # always merges into the one TP reduction the workspace performs.
            and parallel.attn_cp_size == 1
            and not get_attn_tp_context().input_scattered
            and get_moe_a2a_backend().is_none()
            and parallel.tp_size > 1
            # The FFN runs on the full rows, not each rank's own slice.
            and TokenAxis.ATTN_TP not in plan.fused_input_rows(forward_batch).sharded
        )


def _fusion_of(layer: torch.nn.Module) -> CuteDSLFusion | None:
    fusions = getattr(layer.__dict__.get("ffn_boundary"), "fusions", None)
    return fusions if isinstance(fusions, CuteDSLFusion) else None


def install_cutedsl_fusion(
    layers: Sequence[torch.nn.Module],
    *,
    hidden_size: int,
    top_k: int,
    rms_epsilon: float,
    can_defer_finalize: _LayerPredicate,
    label: str,
    terminal_finalize: bool = False,
    include_shared_expert: bool = True,
    fp8_input_linear: Optional[Callable[[torch.nn.Module], torch.nn.Module]] = None,
    overlap_output_linear: Optional[
        Callable[[torch.nn.Module], torch.nn.Module]
    ] = None,
) -> CuteDSLWorkspace | None:
    """One shared workspace handle per fusion-enabled layer, or None.

    Every entry of ``layers`` carries ``attn_boundary`` and ``ffn_boundary``.
    terminal_finalize lets the last layer hand its finalize to the final norm;
    the model must then pass the returned service as
    residual_batch.final_norm(finalize_norm=...). fp8_input_linear(layer) is the
    projection reading the layer's input norm (SGLANG_PLB_AR_FP8_OUT).
    overlap_output_linear(layer) identifies the attention/GDN output projection
    whose target-verify GEMM may feed the R13 split overlap experiment.
    """
    if get_flags().moe.in_speculative_scope:
        # A draft shares the target's process, which holds one workspace.
        return None

    fusion_layers = [layer for layer in layers if _fusion_of(layer) is not None]
    if not fusion_layers:
        return None

    # The workspace compiles one epsilon and cannot see a per-layer one.
    for layer in fusion_layers:
        for norm in (
            layer.attn_boundary.norm,
            layer.ffn_boundary.norm,
        ):
            if _fused_norm_gamma(norm) is None:
                continue
            if float(norm.variance_epsilon) != float(rms_epsilon):
                raise RuntimeError(
                    f"{label} CuTe DSL fusion compiles one workspace for "
                    f"rms_epsilon={rms_epsilon}, but a fused norm uses "
                    f"{norm.variance_epsilon}"
                )

    service = CuteDSLWorkspace(
        hidden_size=hidden_size,
        top_k=top_k,
        rms_epsilon=rms_epsilon,
        include_shared_expert=include_shared_expert,
        fp8_inputs=(
            [
                (layer.attn_boundary.norm, fp8_input_linear(layer))
                for layer in fusion_layers
            ]
            if fp8_input_linear is not None and envs.SGLANG_PLB_AR_FP8_OUT.get()
            else ()
        ),
    )
    hands_off = 0
    for layer in layers:
        fusion = _fusion_of(layer)
        if fusion is None:
            continue
        defers_finalize = bool(can_defer_finalize(layer))
        hands_off += defers_finalize
        fusion.install(
            service,
            defers_finalize=defers_finalize,
            terminal_finalize=terminal_finalize,
        )
        if overlap_output_linear is not None and envs.SGLANG_R13_T4COMM_OVERLAP.get():
            service.register_r13_overlap_output(
                fusion,
                layer.ffn_boundary.plan,
                overlap_output_linear(layer),
            )
    logger.info(
        "Installed one %s FlashInfer MNNVL CuTe DSL fusion handle for %d of %d layers "
        "(%d can defer the MoE finalize)",
        label,
        len(fusion_layers),
        len(layers),
        hands_off,
    )
    return service


def prepare_cutedsl_fusion(
    model: torch.nn.Module, *, max_running_requests: int | None
) -> None:
    """Build the workspace of every service installed anywhere in ``model``.

    Scans the module tree, so a wrapper holding the model as a submodule (a VLM's
    language_model) needs no hook of its own.
    """
    fusions = [
        fusion
        for module in model.modules()
        # Only decoder modules carry an FFN stage.
        if isinstance(
            fusion := getattr(module.__dict__.get("ffn_boundary"), "fusions", None),
            CuteDSLFusion,
        )
    ]
    if not fusions or any(f.service is None for f in fusions):
        raise ValueError(
            "--flashinfer-allreduce-fusion-backend cutedsl is set, but "
            f"{type(model).__name__} installed no CuTe DSL fusion service, so no "
            "allreduce fusion would run at all. Drop the flag, or choose 'auto', "
            "'trtllm' or 'mnnvl'."
        )
    if get_disagg().enable_pdmux:
        raise RuntimeError(
            "FlashInfer MNNVL CuTe DSL fusion does not support concurrent PDMux "
            "streams sharing one mutable workspace"
        )
    services = {id(f.service): f.service for f in fusions}
    max_m = _resolve_max_m(max_running_requests=max_running_requests)
    for service in services.values():
        service.prepare(max_m=max_m)
    logger.info(
        "Prepared the FlashInfer MNNVL CuTe DSL fusion workspace for M_max=%d "
        "(%d fused layers)",
        max_m,
        len(fusions),
    )
