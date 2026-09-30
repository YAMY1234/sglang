"""Slot-owned Mamba-2 sink, content factors and an exact spec-sized update window.

The truncation arithmetic follows twinstar.duet.state at dd9c7bdbd955.
Keeping the factors themselves avoids a second (different) factorization.
Dense tensors returned by materialize/step are temporary read-side scratch.
"""

from __future__ import annotations

import torch


# The truncation arithmetic (CholeskyQR2 in fp64, warm-started subspace iteration, explicit left-side sink) moved to
# sglang.srt.duet.state_factor (docs/162 §3.5, F5); the names below keep the Lightning line's import path.
from ._common import load as _load  # noqa: E402

_state_factor = _load("state_factor")
factorize = _state_factor.factorize_left
orthonormalize = _state_factor.orthonormalize


class LightningMambaStatePool:
    """A MambaPool SlotIndexedState sibling. The stock dense pool is scratch.

    Functional implementation processes each request's truncation separately,
    so warm-start RNG is invariant to scheduler batching (reference batch=1).
    """

    fields = ("coeff", "left", "right", "warm", "ring_x", "ring_b", "ring_decay", "count", "valid", "conv")

    def __init__(self, size, layer_ids, directions, *, n, rank, window, conv_dim, conv_width):
        self.size = size
        self.layer_ids = tuple(layer_ids)
        self.layer_map = {layer: i for i, layer in enumerate(layer_ids)}
        self.directions = directions[list(layer_ids)].float().contiguous()
        self.rank, self.window = rank, window
        layers, heads, p = self.directions.shape
        self.n = n
        self.fields = type(self).fields
        self.exact_mode = rank == 0 or window == 0
        self.device = self.directions.device
        self._allocate(layers, heads, p, self.n, size, conv_dim, conv_width)

    @classmethod
    def for_test(cls, size, directions, n, rank, window, conv_dim=4, conv_width=3):
        obj = cls.__new__(cls)
        obj.size = size
        obj.layer_ids = tuple(range(directions.shape[0]))
        obj.layer_map = {i: i for i in obj.layer_ids}
        obj.directions = directions.float()
        obj.rank, obj.window, obj.n = rank, window, n
        obj.device = directions.device
        obj.fields = cls.fields
        obj.exact_mode = rank == 0 or window == 0
        obj._allocate(*directions.shape, n, size, conv_dim, conv_width)
        return obj

    def _allocate(self, layers, heads, p, n, size, conv_dim, conv_width):
        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(*shape, dtype=dtype, device=self.device)
        self.coeff = zeros(layers, size, heads, n)
        self.left = zeros(layers, size, heads, p, self.rank)
        self.right = zeros(layers, size, heads, self.rank, n)
        self.warm = zeros(layers, size, heads, n, self.rank)
        self.ring_x = zeros(layers, size, self.window, heads, p)
        self.ring_b = zeros(layers, size, self.window, heads, n)
        self.ring_decay = zeros(layers, size, self.window, heads)
        self.count = zeros(layers, size, dtype=torch.int32)
        self.valid = zeros(layers, size, dtype=torch.bool)
        # FP32 is necessary for trained emitter history; shallow BF16 values
        # copy exactly. Reference _to moves device only, not dtype.
        self.conv = zeros(layers, size, conv_dim, conv_width)
        if self.exact_mode:
            self.exact = zeros(layers, size, heads, p, n)
            self.fields = (*self.fields, "exact")

    def prune(self, layer, slot, state, *, warm):
        i = self.layer_map[layer]
        previous = self.warm[i, slot:slot + 1] if warm and self.valid[i, slot].item() else None
        coeff, left, right, basis = factorize(
            state.unsqueeze(0) if state.ndim == 3 else state,
            self.directions[i], self.rank, previous,
        )
        self.coeff[i, slot].copy_(coeff[0])
        self.left[i, slot].copy_(left[0])
        self.right[i, slot].copy_(right[0])
        self.warm[i, slot].copy_(basis[0])
        self.valid[i, slot] = True
        self.count[i, slot] = 0

    def initialize(self, layer, slot, state, conv):
        if self.rank:
            self.prune(layer, slot, state, warm=False)
        else:
            self.valid[self.layer_map[layer], slot] = True
        if self.exact_mode:
            self.exact[self.layer_map[layer], slot].copy_(
                self._materialize_factors(layer, slot) if self.rank else state)
        self.conv[self.layer_map[layer], slot].copy_(conv)

    def materialize(self, layer, slot):
        i = self.layer_map[layer]
        if not self.valid[i, slot].item():
            raise RuntimeError(f"uninitialized Lightning Mamba slot {slot}, layer {layer}")
        if self.exact_mode:
            return self.exact[i, slot]
        return self._materialize_factors(layer, slot)

    def _materialize_factors(self, layer, slot):
        i = self.layer_map[layer]
        state = self.directions[i, :, :, None] * self.coeff[i, slot, :, None, :]
        state = state + self.left[i, slot] @ self.right[i, slot]
        for t in range(int(self.count[i, slot].item())):
            state = (state * self.ring_decay[i, slot, t, :, None, None]
                     + self.ring_x[i, slot, t, :, :, None] * self.ring_b[i, slot, t, :, None, :])
        return state

    def step(self, layer, slot, decay, scaled_x, b, conv):
        i = self.layer_map[layer]
        state = self.materialize(layer, slot)
        state = state * decay[:, None, None] + scaled_x[:, :, None] * b[:, None, :]
        if self.exact_mode:
            self.exact[i, slot].copy_(state)
            self.conv[i, slot].copy_(conv)
            return state
        t = int(self.count[i, slot].item())
        self.ring_decay[i, slot, t].copy_(decay)
        self.ring_x[i, slot, t].copy_(scaled_x)
        self.ring_b[i, slot, t].copy_(b)
        self.conv[i, slot].copy_(conv)
        self.count[i, slot] = t + 1
        if t + 1 == self.window:
            # Read output from the unpruned state, then keep a pruned state for
            # the NEXT token, exactly like reference post-step state_policy.
            self.prune(layer, slot, state, warm=True)
        return state

    def reset_slots(self, indices):
        for name in self.fields:
            getattr(self, name)[:, indices] = 0

    def copy_slots(self, src_index, dst_index):
        for name in self.fields:
            value = getattr(self, name)
            value[:, dst_index] = value[:, src_index].clone()

    def get_cpu_slots(self, indices):
        return {name: getattr(self, name)[:, indices].cpu().clone() for name in self.fields}

    def load_cpu_slots(self, data, indices):
        if set(data) != set(self.fields):
            raise ValueError("incomplete Lightning pool snapshot")
        for name in self.fields:
            getattr(self, name)[:, indices] = data[name].to(self.device)

    def iter_transfer_state_entries(self):
        for name in self.fields:
            value = getattr(self, name)
            for i, layer in enumerate(self.layer_ids):
                yield "lightning_" + name, value[i], None, layer
