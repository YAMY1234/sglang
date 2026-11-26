# Copyright 2023-2026 SGLang Team
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
"""Opt-in routing + NVFP4 input staging in one launch for FlashInfer's compute
capability 10.7 inference MegaMOE (SGLANG_OPT_FLASHINFER_MEGAMOE_FUSED_ROUTE_QUANT).

The router (kernels/ops/moe/moe_route_nvfp4_stage.py) writes this layer's NVFP4
activations, scales and routes straight into the MegaMOE workspace that
FlashInfer's own fused stager would fill, with the same values. FlashInfer's
forward then runs unchanged except that the backend's ``stage_inputs`` skips its
staging launch for that workspace, keeping the per-call scalar binding and the
staged-token bookkeeping. Every other call stages as before.
"""

from __future__ import annotations

_installed = False


def install_fused_route_stage() -> None:
    """Let the compute capability 10.7 NVFP4 backend skip staging that the router
    already wrote."""
    global _installed
    if _installed:
        return
    from flashinfer.moe_ep.backends.mega.kernel.sm107.nvfp4_nvfp4_bf16_cutedsl.backend import (
        Sm107Nvfp4BlockScaledMegaKernelBackend,
    )
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import note_staged_tokens

    from sglang.srt.layers.moe import route_quant_handoff

    stock_stage_inputs = Sm107Nvfp4BlockScaledMegaKernelBackend.stage_inputs

    def stage_inputs(self, t, workspace, *, quantize_input: bool) -> None:
        if quantize_input and route_quant_handoff.take_megamoe_staged(
            t.hidden_states, t.topk_ids, workspace
        ):
            # The stager's remaining work: scalars, then the live-count memos.
            self._stage_scalars(t, workspace)
            num_tokens = t.hidden_states.shape[0]
            note_staged_tokens(workspace.topk_idx, num_tokens)
            workspace.note_staged_tokens(num_tokens)
            return
        stock_stage_inputs(self, t, workspace, quantize_input=quantize_input)

    Sm107Nvfp4BlockScaledMegaKernelBackend.stage_inputs = stage_inputs
    _installed = True


def fused_route_stage_installed() -> bool:
    return _installed
