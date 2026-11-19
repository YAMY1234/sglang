"""Per-expert route counts of large MoE calls, logged for sizing the routed-expert GEMM tiles.

Off unless SGLANG_K31_MOE_EXPERT_COUNT_LOG_INTERVAL > 0. Every Nth call with at least
SGLANG_K31_MOE_EXPERT_COUNT_MIN_TOKENS rows scatter-adds its top-k ids into a count vector,
copies it to pinned memory without blocking, and the line is written at a later call once
the copy has landed: no host sync on the forward path, nothing during CUDA graph capture.
"""

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_calls = 0
_host: Optional[torch.Tensor] = None
_done: Optional[torch.cuda.Event] = None
_pending: Optional[tuple] = None


def maybe_record(layer_id: int, topk_ids: Optional[torch.Tensor], num_experts: int) -> None:
    global _calls, _host, _done, _pending
    interval = envs.SGLANG_K31_MOE_EXPERT_COUNT_LOG_INTERVAL.get()
    if interval <= 0 or topk_ids is None or topk_ids.dim() != 2:
        return
    if (
        topk_ids.shape[0] < envs.SGLANG_K31_MOE_EXPERT_COUNT_MIN_TOKENS.get()
        or torch.cuda.is_current_stream_capturing()
    ):
        return
    _calls += 1
    if _pending is not None and _done.query():
        call, lid, tokens, top_k = _pending
        logger.info(
            "K31 MoE expert counts: call=%d layer=%d tokens=%d top_k=%d experts=%d counts=%s",
            call, lid, tokens, top_k, num_experts, ",".join(map(str, _host.tolist())),
        )
        _pending = None
    if _calls % interval or _pending is not None:
        return
    # ids < 0 (masked rows) land in slot 0 and are dropped
    ids = (topk_ids.reshape(-1).to(torch.int64) + 1).clamp_(min=0)
    counts = torch.zeros(num_experts + 1, dtype=torch.int64, device=ids.device)
    counts.scatter_add_(0, ids, torch.ones_like(ids))
    if _host is None or _host.numel() != num_experts:
        _host = torch.empty(num_experts, dtype=torch.int64, pin_memory=True)
    _host.copy_(counts[1:], non_blocking=True)
    _done = torch.cuda.Event()
    _done.record()
    _pending = (_calls, layer_id, topk_ids.shape[0], topk_ids.shape[1])
