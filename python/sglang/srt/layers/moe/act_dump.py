"""Routed-expert inputs of a few large MoE calls, saved for offline numerics of the expert kernels.

Off unless SGLANG_K31_MOE_ACT_DUMP_DIR is set. For each layer in SGLANG_K31_MOE_ACT_DUMP_LAYERS, the first
SGLANG_K31_MOE_ACT_DUMP_CALLS calls with at least SGLANG_K31_MOE_EXPERT_COUNT_MIN_TOKENS rows save the activations
(bf16 [T, H]), top-k ids and weights to <dir>/moe-L<layer>-c<call>-r<pid>.pt. Each save syncs the host, so enable
it only for warm-up traffic, never inside a timed window.
"""

import logging
import os
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_saved: dict[int, int] = {}


def maybe_dump(
    layer_id: int,
    x: torch.Tensor,
    topk_ids: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
) -> None:
    out_dir = envs.SGLANG_K31_MOE_ACT_DUMP_DIR.get()
    if not out_dir or topk_ids is None or torch.cuda.is_current_stream_capturing():
        return
    layers = {int(v) for v in envs.SGLANG_K31_MOE_ACT_DUMP_LAYERS.get().split(",") if v}
    if (
        layer_id not in layers
        or x.shape[0] < envs.SGLANG_K31_MOE_EXPERT_COUNT_MIN_TOKENS.get()
    ):
        return
    call = _saved.get(layer_id, 0)
    if call >= envs.SGLANG_K31_MOE_ACT_DUMP_CALLS.get():
        return
    _saved[layer_id] = call + 1
    path = os.path.join(out_dir, f"moe-L{layer_id}-c{call}-r{os.getpid()}.pt")
    torch.save(
        {
            "x": x.to(torch.bfloat16).cpu(),
            "topk_ids": topk_ids.cpu(),
            "topk_weights": topk_weights.float().cpu(),
        },
        path,
    )
    logger.info("K31 MoE activation dump: %s tokens=%d", path, x.shape[0])
