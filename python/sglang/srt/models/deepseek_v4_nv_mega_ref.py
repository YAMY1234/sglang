"""Test-only reference for the sgl-nvidia-mega DSV4.1 plugin (not for upstream).

With SGLANG_NV_MEGA_INTREE_REF=1 and no --monokernel-backend, the plugin's per-rank
runner and slots are built by the plugin's own build after load, and SGLang's own
``DeepseekV4CudaDecoderLayer.forward`` runs them at the steps and widths the plugin
routes, with the arguments of the plugin's ``run_layer``. Comparing this branch with
the plugin's routed forward checks the plugin's re-assembled step against SGLang's.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Optional

import msgspec

from sglang.srt.monokernels import runtime as mono_runtime
from sglang.srt.monokernels.guards import deadline_s, device_isa
from sglang.srt.monokernels.spec import BuildContext, ShapeKey
from sglang.srt.runtime_context import get_context, get_device, get_exec

logger = logging.getLogger(__name__)


class StepRef(msgspec.Struct, frozen=True):
    """One layer's slots for one routed graph key, as run_layer takes them."""

    runner: Any
    hop_a: bool
    moe: Any  # the plugin's MoeSlot, or None
    fast_gate: bool

    def attention(self, attn, *, x, positions, forward_batch, quantized):
        """run_layer's hop-A attention: wo_b publishes into the Lamport hop."""
        from sglang.srt.models import deepseek_v4_mhc as mhc

        o = attn.forward_before_wo_b(
            x=x, positions=positions, forward_batch=forward_batch, x_quant=quantized
        )
        partial = self.runner.publish_wo_b(attn.wo_b, o)
        if partial is None:
            return attn._project_wo_b(o, defer_all_reduce=True)
        return mhc.AttnOutput(partial, fused_post=self.runner.hop_a)

    def moe_forward(self, mlp, *, hidden_states, forward_batch, transport):
        """run_layer's MoE slot call of SGLang's dual-stream MoE."""
        moe = self.moe
        return mlp.forward_normal_dual_stream(
            hidden_states,
            None,
            transport.input_ids,
            input_ids_global=transport.input_ids_global,
            num_token_non_padded=forward_batch.moe_num_token_non_padded(),
            use_vision_topk=mlp.gate.e_score_correction_bias_vl is not None,
            return_moe_output=transport.return_moe_output,
            leave_output_sum=True,
            select_experts=moe.select if self.fast_gate and moe.gate else None,
            run_routed_experts=moe.run,
        )


class LayerRef(msgspec.Struct, frozen=True):
    """A layer's slots per routed graph key; other steps run SGLang's path."""

    steps: dict  # ShapeKey -> StepRef

    def at(self, forward_batch) -> Optional[StepRef]:
        # host-side: the batch size and the input's row count, as the framework routes
        return self.steps.get(mono_runtime.shape_key(forward_batch))


def _step_refs(*, served, slots, runner, moe, is_draft: bool) -> dict:
    steps = {}
    for key in served:
        s = slots[key.rows]
        # the plugin's draft runs no hop A and SGLang's top-k (run_layer arguments)
        steps[key] = StepRef(
            runner=runner,
            hop_a=s.hop_a and not is_draft,
            moe=moe if s.routed else None,
            fast_gate=s.fast_gate and not is_draft,
        )
    return steps


def _widths(spec, model, isa: str) -> tuple[tuple[int, ...], frozenset[ShapeKey]]:
    """The routed widths and graph keys, as install_monokernel_forward picks them."""
    mode, tokens = mono_runtime._server_step()
    if spec.tokens_per_request is not None:
        tokens = spec.tokens_per_request(model)
    capture_keys = mono_runtime._capture_keys(mode=mode, tokens=tokens)
    capture_rows = {k.rows for k in capture_keys}
    widths = tuple(
        w
        for w in spec.widths(isa)
        if w % tokens == 0 and (not capture_keys or w in capture_rows)
    )
    return widths, mono_runtime._served_keys(mode=mode, widths=widths, tokens=tokens)


def install(model) -> None:
    """Build the plugin's runner and slots for ``model`` and attach them per layer."""
    if not get_context().is_config_namespace_published("exec"):
        logger.info("SGLANG_NV_MEGA_INTREE_REF: no server configuration, not installed")
        return
    backend = get_exec().kernel.monokernel_backend
    if backend is not None:
        logger.warning(
            "SGLANG_NV_MEGA_INTREE_REF is ignored with --monokernel-backend %s", backend
        )
        return
    import sgl_nvidia_mega
    from sgl_nvidia_mega.deepseek_v41 import forward as mega_forward

    path = sgl_nvidia_mega.SPECS.get(type(model).__name__)
    if path is None:
        logger.info("SGLANG_NV_MEGA_INTREE_REF: no spec for %s", type(model).__name__)
        return
    module, _, attr = path.partition(":")
    spec = getattr(importlib.import_module(module), attr)
    hf_config = model.config
    reasons = spec.model_disable_reasons(model, hf_config, {})
    isa = device_isa()
    if isa not in spec.devices:
        reasons.append(f"{spec.name} has no kernels for {isa}")
    if reasons:
        raise ValueError("SGLANG_NV_MEGA_INTREE_REF: " + "; ".join(reasons))
    widths, served = _widths(spec, model, isa)
    context = BuildContext(
        isa=isa,
        widths=widths,
        deadline_s=deadline_s(get_device().watchdog_timeout),
        copies={},
    )
    built = spec.build(model, hf_config, context)
    is_draft = isinstance(built, mega_forward.Draft)
    if is_draft:
        layers = list(enumerate(model.stages))
    else:
        m = model.model
        layers = [
            (m.layers[i].layer_id, m.layers[i])
            for i in range(m.start_layer, m.end_layer)
        ]
    for key, layer in layers:
        layer._nv_mega_ref = LayerRef(
            steps=_step_refs(
                served=served,
                slots=built.slots,
                runner=built.runner,
                moe=built.moe.get(key),
                is_draft=is_draft,
            )
        )
    logger.info(
        "SGLANG_NV_MEGA_INTREE_REF: %s slots in SGLang's own layer loop for %s",
        spec.name,
        mono_runtime._format_steps(served),
    )
