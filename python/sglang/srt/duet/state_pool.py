"""Per-slot sidecar of the DUET recurrent-state policy (docs/162 §3.5 `state_pool`), model independent.

What the reference keeps per request (twinstar/duet/gdnfactor.py, kimi/model.py KimiCache.step_index, state.py
StateFactor._warm) and every line re-implemented per slot: the decode step counter (the boundary token's full-depth
step counts as step 1), the "prompt-final prune pending" flag, and the warm-start right factor V (N x r per head)
of the last truncation.  The dense or factored state itself stays in the model's own pool; this sidecar only tells
the adapter WHICH slots to prune at a step and hands the warm basis in and out.  Lifecycle hooks mirror the
MambaPool slot-state contract (reset / copy / host round trip / transfer entries) so HiCache, radix and PD
transports move the sidecar with the state.  Moved / generalised from the Kimi line (twinstar_sgl/kimi_duet_runtime.py
DuetStatePruner) and the Lightning line (models/lightning_duet/state.py LightningMambaStatePool count/valid/warm).
"""
from __future__ import annotations

import torch


class SlotSidecar:
    """counters, pending-prefix flags and warm bases for `layers` recurrent layers x `size` slots."""

    fields = ("count", "pending_prefix", "warm", "warm_valid")

    def __init__(self, size, layer_ids, *, heads, warm_dim, rank, every, device="cpu"):
        self.size = int(size)
        self.layer_ids = tuple(int(l) for l in layer_ids)
        self.layer_map = {layer: i for i, layer in enumerate(self.layer_ids)}
        self.heads, self.warm_dim, self.rank, self.every = int(heads), int(warm_dim), int(rank), int(every)
        if self.rank < 0 or self.every < 0:
            raise ValueError("rank and cadence must be nonnegative (0 = exact state / prune only at the prompt end)")
        self.device = torch.device(device)
        layers = len(self.layer_ids)
        self.count = torch.zeros(layers, self.size, dtype=torch.int32, device=self.device)
        self.pending_prefix = torch.zeros(layers, self.size, dtype=torch.bool, device=self.device)
        self.warm_valid = torch.zeros(layers, self.size, dtype=torch.bool, device=self.device)
        # V (N x r) per head, fp32 -- the reference's warm start; empty when rank == 0.
        self.warm = torch.zeros(layers, self.size, self.heads, self.warm_dim, max(self.rank, 0),
                                dtype=torch.float32, device=self.device)

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
            raise ValueError(f"warm basis shape {tuple(basis.shape)} differs from slot geometry "
                             f"{(len(idx), self.heads, self.warm_dim, self.rank)}")
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
        return {name: getattr(self, name)[:, idx].to("cpu").clone() for name in self.fields}

    def load_cpu_slots(self, data, indices):
        idx = self._idx(indices)
        for name in self.fields:
            t = data[name]
            if tuple(t.shape[1:2]) != (len(idx),):
                raise ValueError(f"{name}: host copy covers {t.shape[1]} slots, {len(idx)} requested")
            getattr(self, name)[:, idx] = t.to(self.device)

    def iter_transfer_state_entries(self):
        """(name, tensor, layer index, layer id) for every per-layer field, for PD / HiCache transports."""
        for i, layer in enumerate(self.layer_ids):
            for name in self.fields:
                yield f"duet_sidecar_{name}", getattr(self, name)[i], i, layer

    def nbytes(self):
        return sum(getattr(self, name).numel() * getattr(self, name).element_size() for name in self.fields)

    def _idx(self, slots):
        idx = torch.as_tensor(slots, device=self.device).reshape(-1).long()
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= self.size):
            raise IndexError("slot index out of range")
        return idx
