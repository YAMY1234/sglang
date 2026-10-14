"""Graph-owned GLM W32 cuts; all request cadence decisions stay on device.

The main stream owns decode. A serial side stream owns completed layer groups,
with a main-to-side edge per group and a side-to-main edge at the last layer.
That final join precedes phase publication and every slot lifecycle operation.
"""

import os

import torch
from sglang.srt.layers.attention.linear.kernels.glm_w32_project import (
    ProjectionWorkspace,
    phase_update,
    project,
)

from ._state_factor import OMEGA_SEED


class GraphGlmStatePolicy:
    device_scheduled = True

    def __init__(self, pool, directions, spec, layer_ids, *, omega_heads):
        self.pool = pool
        self.states = pool.mamba_pool.mamba_cache.temporal
        layers, slots, heads, values, keys = self.states.shape
        self.rank, self.every = spec["state_rank"], spec["state_every"]
        if (self.rank, self.every, values, keys) != (32, 32, 128, 128):
            raise ValueError("GLM graph projection requires R=W=32 and K=V=128")
        if self.states.dtype != torch.float32 or not self.states.is_contiguous():
            raise ValueError("GLM graph projection requires a contiguous FP32 pool")
        if spec.get("state_sink") != "explicit":
            raise ValueError("GLM graph projection requires the explicit right sink")
        if int(os.environ.get("SGLANG_DUET_STATE_OVERSAMPLE", "8")) != 8:
            raise ValueError("GLM graph projection fixes service oversample=8")
        self.layer_map = {l: pool.mamba2_layer_index(l) for l in layer_ids}
        if list(self.layer_map.values()) != list(range(layers)):
            raise ValueError(
                "GLM graph projection requires all resident layers in order"
            )
        self.layer_ids = tuple(layer_ids)
        device = self.states.device
        self.vbar = torch.stack(
            [directions[l].to(device=device, dtype=torch.float32) for l in layer_ids]
        ).contiguous()
        if (
            self.vbar.shape != (layers, heads, values)
            or not torch.isfinite(self.vbar).all()
        ):
            raise ValueError("invalid GLM sink directions")
        self.phase = torch.zeros(slots, dtype=torch.int32, device=device)
        self.pending = torch.zeros(slots, dtype=torch.bool, device=device)
        self.active = torch.zeros_like(self.pending)
        self.warm = torch.zeros(
            layers, slots, heads, values, self.rank, device=device, dtype=torch.float32
        )
        offset, total = omega_heads
        # The reference reseeds independently for cold (40) and warm (8) draws.
        self.omega, self.extra = [
            torch.randn(
                1,
                total,
                values,
                width,
                generator=torch.Generator(device=device).manual_seed(OMEGA_SEED),
                device=device,
                dtype=torch.float32,
            )[0, offset : offset + heads].contiguous()
            for width in (40, 8)
        ]
        self.group_size = min(8, layers)
        self.workspace = ProjectionWorkspace(slots, heads, self.group_size, device)
        self.side = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        pool.mamba_pool.register_slot_state(self)

    def before_graph(self, batch):
        pass

    def after_graph(self, batch):
        pass

    def slots(self, batch):
        return self.pool.get_mamba_indices(
            batch.req_pool_indices[: batch.batch_size]
        ).long()

    def mark_prompt_end(self, req_pool_indices, slots):
        self._join()
        self.phase[slots] = 0
        self.pending[slots] = True
        self.active[slots] = True

    def _project(self, indices, first, layers, *, cold):
        project(
            self.states,
            self.vbar,
            self.warm,
            self.omega,
            self.extra,
            indices,
            self.phase,
            self.pending,
            self.active,
            self.workspace,
            first=first,
            layers=layers,
            cold=cold,
            window=self.every,
        )

    def before_layer(self, layer_id, indices):
        if layer_id != self.layer_ids[0]:
            return
        for first in range(0, len(self.layer_ids), self.group_size):
            self._project(
                indices,
                first,
                min(self.group_size, len(self.layer_ids) - first),
                cold=True,
            )
        phase_update(
            indices, self.phase, self.pending, self.active, self.every, clear=True
        )

    def after_layer(self, layer_id, indices):
        index = self.layer_map[layer_id]
        last = index + 1 == len(self.layer_ids)
        if (index + 1) % self.group_size and not last:
            return
        first = index // self.group_size * self.group_size
        main = torch.cuda.current_stream(self.states.device)
        self.side.wait_stream(main)
        with torch.cuda.stream(self.side):
            self._project(indices, first, index + 1 - first, cold=False)
        if last:
            main.wait_stream(self.side)
            phase_update(indices, self.phase, self.pending, self.active, self.every)

    def _join(self):
        if self.side is not None:
            torch.cuda.current_stream(self.states.device).wait_stream(self.side)

    def validate_slot_lifecycle(self, operation, indices, **kwargs):
        # Called before the native pool mutates, including native dense copies.
        self._join()
        return False

    def reset_slots(self, indices):
        self._join()
        self.phase[indices] = 0
        self.pending[indices] = False
        self.active[indices] = False
        self.warm[:, indices] = 0

    def copy_slots(self, src, dst):
        self._join()
        for value in (self.phase, self.pending, self.active):
            value[dst] = value[src].clone()
        self.warm[:, dst] = self.warm[:, src].clone()

    def get_cpu_slots(self, indices):
        self._join()
        return tuple(
            value[indices].cpu().clone()
            for value in (self.phase, self.pending, self.active)
        ) + (self.warm[:, indices].cpu().clone(),)

    def load_cpu_slots(self, data, indices):
        self._join()
        if len(data) != 4:
            raise ValueError("incomplete GLM graph projection snapshot")
        for target, saved in zip((self.phase, self.pending, self.active), data[:3]):
            target[indices] = saved.to(target.device)
        self.warm[:, indices] = data[3].to(self.warm.device)
