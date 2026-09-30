"""DUET slot state: LeftSinkStatePool (Lightning) and generic SlotSidecar.

LeftSinkStatePool handles S <- decay * S + x b^T without an erase term. Adapters supply
layer IDs, dimensions, sink directions and decode r/W; this is not a KDA/GDN
update rule. ``conv`` is opaque adapter history carried with each slot. Warm
basis, sink, factors, ring, counter, validity and history move atomically under
reset/copy/snapshot/load, matching MambaPool's SlotIndexedState interface.
The stock dense pool remains temporary read-side scratch. Arithmetic is
unchanged from the accepted Lightning path; no reference re-factorization.

SlotSidecar is the design-line interface for counters, pending-prefix flags and
warm bases with state stored in a separate pool. Both public contracts coexist;
Lightning does not allocate a second counter/warm sidecar beside its factor pool.
"""

from __future__ import annotations

import torch

from .state_factor import factorize_left


class LeftSinkStatePool:
    """A MambaPool SlotIndexedState sibling. The stock dense pool is scratch.

    Functional implementation processes each request's truncation separately,
    so warm-start RNG is invariant to scheduler batching (reference batch=1).
    """

    fields = (
        "coeff",
        "left",
        "right",
        "warm",
        "ring_x",
        "ring_b",
        "ring_decay",
        "count",
        "valid",
        "conv",
    )

    def __init__(
        self,
        size,
        layer_ids,
        directions,
        *,
        n,
        rank,
        window,
        conv_dim,
        conv_width,
        transfer_prefix="duet_",
    ):
        self.transfer_prefix = transfer_prefix
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
        obj.transfer_prefix = "duet_"
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
        previous = (
            self.warm[i, slot : slot + 1]
            if warm and self.valid[i, slot].item()
            else None
        )
        coeff, left, right, basis = factorize_left(
            state.unsqueeze(0) if state.ndim == 3 else state,
            self.directions[i],
            self.rank,
            previous,
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
                self._materialize_factors(layer, slot) if self.rank else state
            )
        self.conv[self.layer_map[layer], slot].copy_(conv)

    def materialize(self, layer, slot):
        i = self.layer_map[layer]
        if not self.valid[i, slot].item():
            raise RuntimeError(
                f"uninitialized DUET left-sink slot {slot}, layer {layer}"
            )
        if self.exact_mode:
            return self.exact[i, slot]
        return self._materialize_factors(layer, slot)

    def _materialize_factors(self, layer, slot):
        i = self.layer_map[layer]
        state = self.directions[i, :, :, None] * self.coeff[i, slot, :, None, :]
        state = state + self.left[i, slot] @ self.right[i, slot]
        for t in range(int(self.count[i, slot].item())):
            state = (
                state * self.ring_decay[i, slot, t, :, None, None]
                + self.ring_x[i, slot, t, :, :, None]
                * self.ring_b[i, slot, t, :, None, :]
            )
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
        return {
            name: getattr(self, name)[:, indices].cpu().clone() for name in self.fields
        }

    def load_cpu_slots(self, data, indices):
        if set(data) != set(self.fields):
            raise ValueError("incomplete DUET pool snapshot")
        for name in self.fields:
            getattr(self, name)[:, indices] = data[name].to(self.device)

    def iter_transfer_state_entries(self):
        for name in self.fields:
            value = getattr(self, name)
            for i, layer in enumerate(self.layer_ids):
                yield self.transfer_prefix + name, value[i], None, layer


class SlotSidecar:
    """counters, pending-prefix flags and warm bases for `layers` recurrent layers x `size` slots."""

    fields = ("count", "pending_prefix", "warm", "warm_valid")

    def __init__(self, size, layer_ids, *, heads, warm_dim, rank, every, device="cpu"):
        self.size = int(size)
        self.layer_ids = tuple(int(l) for l in layer_ids)
        self.layer_map = {layer: i for i, layer in enumerate(self.layer_ids)}
        self.heads, self.warm_dim, self.rank, self.every = (
            int(heads),
            int(warm_dim),
            int(rank),
            int(every),
        )
        if self.rank < 0 or self.every < 0:
            raise ValueError(
                "rank and cadence must be nonnegative (0 = exact state / prune only at the prompt end)"
            )
        self.device = torch.device(device)
        layers = len(self.layer_ids)
        self.count = torch.zeros(
            layers, self.size, dtype=torch.int32, device=self.device
        )
        self.pending_prefix = torch.zeros(
            layers, self.size, dtype=torch.bool, device=self.device
        )
        self.warm_valid = torch.zeros(
            layers, self.size, dtype=torch.bool, device=self.device
        )
        # V (N x r) per head, fp32 -- the reference's warm start; empty when rank == 0.
        self.warm = torch.zeros(
            layers,
            self.size,
            self.heads,
            self.warm_dim,
            max(self.rank, 0),
            dtype=torch.float32,
            device=self.device,
        )

    # ------------------------------------------------------------------ per-step contract
    def note_prefill(self, layer, slots):
        """Prompt-final state written for `slots` (prefill or emitter): prune once with a cold start, counter 0."""
        i = self.layer_map[layer]
        idx = self._idx(slots)
        self.pending_prefix[i, idx] = True
        self.warm_valid[i, idx] = False
        self.count[i, idx] = 0

    def take_pending_prefix(self, layer, slots):
        """Slots whose prompt-final prune is still owed (clears the flag)."""
        i = self.layer_map[layer]
        idx = self._idx(slots)
        owed = self.pending_prefix[i, idx].clone()
        self.pending_prefix[i, idx] = False
        return idx[owed]

    def note_step(self, layer, slots):
        """One full-depth step done for `slots` (the boundary token's step is step 1); returns the slots whose
        counter reached a multiple of `every` (empty when every == 0 -- prune only at the prompt end)."""
        i = self.layer_map[layer]
        idx = self._idx(slots)
        self.count[i, idx] += 1
        if self.every <= 0 or self.rank <= 0:
            return idx[:0]
        due = (self.count[i, idx] % self.every) == 0
        return idx[due]

    def warm_basis(self, layer, slots):
        """(B, H, N, r) warm bases for `slots`, or None when any of them has no valid basis (cold start)."""
        i = self.layer_map[layer]
        idx = self._idx(slots)
        if self.rank <= 0 or not bool(self.warm_valid[i, idx].all()):
            return None
        return self.warm[i, idx]

    def store_warm(self, layer, slots, basis):
        i = self.layer_map[layer]
        idx = self._idx(slots)
        if basis is None:
            self.warm_valid[i, idx] = False
            return
        if tuple(basis.shape) != (len(idx), self.heads, self.warm_dim, self.rank):
            raise ValueError(
                f"warm basis shape {tuple(basis.shape)} differs from slot geometry "
                f"{(len(idx), self.heads, self.warm_dim, self.rank)}"
            )
        self.warm[i, idx] = basis.to(self.warm.dtype)
        self.warm_valid[i, idx] = True

    # ------------------------------------------------------------------ slot lifecycle (MambaPool sibling contract)
    def reset_slots(self, indices):
        idx = self._idx(indices)
        self.count[:, idx] = 0
        self.pending_prefix[:, idx] = False
        self.warm_valid[:, idx] = False
        self.warm[:, idx] = 0

    def copy_slots(self, src, dst):
        s, d = self._idx(src), self._idx(dst)
        if len(s) != len(d):
            raise ValueError("copy_slots needs equally many sources and destinations")
        self.count[:, d] = self.count[:, s]
        self.pending_prefix[:, d] = self.pending_prefix[:, s]
        self.warm_valid[:, d] = self.warm_valid[:, s]
        self.warm[:, d] = self.warm[:, s]

    def get_cpu_slots(self, indices):
        idx = self._idx(indices)
        return {
            name: getattr(self, name)[:, idx].to("cpu").clone() for name in self.fields
        }

    def load_cpu_slots(self, data, indices):
        idx = self._idx(indices)
        for name in self.fields:
            t = data[name]
            if tuple(t.shape[1:2]) != (len(idx),):
                raise ValueError(
                    f"{name}: host copy covers {t.shape[1]} slots, {len(idx)} requested"
                )
            getattr(self, name)[:, idx] = t.to(self.device)

    def iter_transfer_state_entries(self):
        """(name, tensor, slice_axis=None, layer id) for every per-layer field -- MambaPool._iter_transfer_state_entries contract."""
        for i, layer in enumerate(self.layer_ids):
            for name in self.fields:
                # MambaPool contract: (name, tensor, slice_axis, layer_id); no TP slice axis for the sidecar.
                yield f"duet_sidecar_{name}", getattr(self, name)[i], None, layer

    def nbytes(self):
        return sum(
            getattr(self, name).numel() * getattr(self, name).element_size()
            for name in self.fields
        )

    def _idx(self, slots):
        idx = torch.as_tensor(slots, device=self.device).reshape(-1).long()
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= self.size):
            raise IndexError("slot index out of range")
        return idx
