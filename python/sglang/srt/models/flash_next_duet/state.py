"""W=0 uses dense recurrence after one prompt-end state projection."""

import torch


class PromptEndProjection:
    def __init__(self, fullstack, tp_rank):
        self.rank = fullstack["gdn_rank"]
        self.tp_rank = tp_rank
        self.directions = torch.load(
            fullstack["state_sink_vbar"], map_location="cpu", weights_only=True
        )["vbar"]
        self._device_directions = {}

    @torch.no_grad()
    def apply(self, layer_id, states, slots, prompt_final):
        if prompt_final is None or not any(prompt_final):
            return
        from sglang.srt.layers.attention.linear.kernels.gdn_prefill_reference import (
            K31_OVERSAMPLE,
            K31_SEED,
            factorize_prefill_k31,
        )

        if self.rank >= min(states.shape[-2:]):
            return
        heads, v, k = states.shape[-3:]
        lo = self.tp_rank * heads
        if layer_id not in self._device_directions:
            full = self.directions[layer_id].to(
                device=states.device, dtype=torch.float32
            )
            generator = torch.Generator(device=states.device).manual_seed(K31_SEED)
            omega = torch.randn(
                1,
                full.shape[0],
                v,
                min(self.rank + K31_OVERSAMPLE, v, k),
                device=states.device,
                generator=generator,
            )
            self._device_directions[layer_id] = (
                full[lo : lo + heads],
                omega[:, lo : lo + heads],
            )
        direction, omega = self._device_directions[layer_id]
        # Each request receives the reference's batch-1 probe, independent of
        # batch order, padding and the number of requests finishing this chunk.
        for row, final in enumerate(prompt_final):
            if not final:
                continue
            index = slots[row : row + 1].long()
            a, u, w = factorize_prefill_k31(
                states[index], direction, self.rank, self.rank, torch.float32, omega
            )
            projected = (
                direction[None, :, :, None] * a[:, :, None, :] + w.transpose(-1, -2) @ u
            )
            states[index] = projected.to(states.dtype)


def prompt_projection(model_config, tp_rank):
    from sglang.srt.model_executor.fullstack_policy import (
        fullstack_config,
        fullstack_enabled,
    )

    if not fullstack_enabled(model_config):
        return None
    fs = fullstack_config(model_config)
    if fs["gdn_rank"] > 0 and fs["gdn_every"] == 0:
        return PromptEndProjection(fs, tp_rank)
    return None
