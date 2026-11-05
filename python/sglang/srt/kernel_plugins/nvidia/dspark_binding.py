from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from sglang.srt.kernel_plugins.nvidia.binding import (
    megakernel_agree,
    megakernel_draft_sampler,
)
from sglang.srt.layers.dp_attention import get_dp_tp_group

if TYPE_CHECKING:
    from sglang.srt.speculative.dspark_components.dspark_draft import DraftNoise


DraftResult = tuple[torch.Tensor, torch.Tensor | None]
DraftStep = Callable[[torch.Tensor, torch.Tensor], DraftResult]
DraftSteps = Callable[[int], DraftStep | None]


def bind_dspark_draft_strategy(
    *,
    draft_model: torch.nn.Module,
    gamma: int,
    max_bs: int,
    greedy_draft: bool,
    noise: DraftNoise,
    draft_tokens: torch.Tensor,
    step_temperatures: torch.Tensor,
    allocate_step_logits: Callable[[], torch.Tensor],
) -> tuple[DraftSteps | None, torch.Tensor | None]:
    """Bind the optional DSpark steps and their shared logits buffer once."""

    factory = megakernel_draft_sampler(draft_model)
    if factory is None:
        return None, None

    step_logits = None if greedy_draft else allocate_step_logits()
    steps = factory(
        gamma=gamma,
        max_bs=max_bs,
        greedy_draft=greedy_draft,
        noise=noise,
        draft_tokens=draft_tokens,
        step_logits=step_logits,
        step_temperatures=step_temperatures,
    )
    # A rank on the stock block waits in its per-step token sync (this group),
    # which a specialized step skips.
    served = tuple(bs for bs in range(1, max_bs + 1) if steps(bs) is not None)
    megakernel_agree(served, what="draft sampler batch sizes", group=get_dp_tp_group())
    return steps, step_logits
