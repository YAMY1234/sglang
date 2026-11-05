"""Model-neutral, borrowed quantized activation passed with BF16 MoE input."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import torch


class Nvfp4RoutedInputPreQuant(NamedTuple):
    """Linear NVFP4 activations quantized with this MoE's input-scale domain.

    The producer uses the same w13_input_scale_quant as the consuming runner.
    x_sf is linear E4M3. BF16 hidden states stay available for routing/output
    dtype. ready is optional: same-stream producers need no event or wait.
    """

    x_fp4: torch.Tensor
    x_sf: torch.Tensor
    ready: torch.cuda.Event | None = None
