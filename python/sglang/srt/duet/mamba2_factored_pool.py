"""Experimental Lightning factor authority with empty native dense storage.

The existing bridge supplies lifecycle validation; dense storage is empty.
Prompt-final factorization uses temporary one-layer state. Decode owns only
coeff/U/V/fcount plus native conv, with no warm or cold-factor duplicates.
"""

from dataclasses import dataclass, field

import os

import torch

from .dense_bridge import DenseBridgeStatePool


@dataclass
class Mamba2ExtendPlan:
    slots: torch.Tensor
    host_slots: torch.Tensor
    pending: list = field(default_factory=list)
    written: set = field(default_factory=set)
    pending_bytes: int = 0
    peak_pending_bytes: int = 0
    commits: int = 0


class Mamba2FactoredStatePool(DenseBridgeStatePool):
    fields = ("coeff", "u", "v", "fcount", "count", "valid", "conv", "dense")

    @property
    def prefill_commit_mode(self):
        mode = getattr(self, "_prefill_commit_mode", None)
        if mode is None:
            mode = os.environ.get("SGLANG_LTN_PREFILL_COMMIT", "reference")
        if mode not in ("reference", "qr", "device"):
            raise ValueError("SGLANG_LTN_PREFILL_COMMIT expects reference/qr/device")
        return mode

    def _idle(self):
        super()._idle()
        if getattr(self, "_extend_plan", None) is not None:
            raise RuntimeError("slot lifecycle operation during Mamba extend")

    def begin_extend(self, slots):
        """Own all rows until commit publication, without a resident dense pool."""
        self._idle()
        host = self._host_indices(slots)
        if host.numel() == 0 or host.unique().numel() != host.numel():
            raise ValueError("extend needs nonempty, unique physical slots")
        self.reset_slots(slots)
        plan = Mamba2ExtendPlan(slots.long(), host)
        self._extend_plan = plan
        return plan

    def commit_extend(self, layer_id, plan, state, *, rows=None):
        """GDN-style final-state collector; factorize independent rows in groups."""
        if getattr(self, "_extend_plan", None) is not plan:
            raise RuntimeError("stale Mamba extend plan")
        layer = self.layer_map[layer_id]
        rows = tuple(range(plan.slots.numel())) if rows is None else tuple(rows)
        if (not rows or len(set(rows)) != len(rows)
                or any(row < 0 or row >= plan.slots.numel() for row in rows)):
            raise ValueError("invalid Mamba extend rows")
        coordinates = {(layer, row) for row in rows}
        if coordinates & plan.written:
            raise RuntimeError("duplicate layer-row in Mamba extend")
        expected = (len(rows), *self.directions.shape[1:], self.n)
        if state.shape != expected or state.dtype != torch.float32 or state.device != self.device:
            raise ValueError(f"invalid Mamba final state: {state.shape}, expected {expected}")
        size = state.numel() * state.element_size()
        if self.prefill_commit_mode != "device" and plan.pending and plan.pending_bytes + size > 64 * 1024**2:
            self._flush_extend(plan)
        plan.pending.append((layer, rows, state))
        plan.written.update(coordinates)
        plan.pending_bytes += size
        plan.peak_pending_bytes = max(plan.peak_pending_bytes, plan.pending_bytes)

    def _flush_extend(self, plan):
        if not plan.pending:
            return
        if self.prefill_commit_mode == "device":
            from sglang.srt.layers.attention.linear.kernels.mamba2_prefill_commit import commit_device
            commit_device(self, plan)
            plan.pending.clear()
            plan.pending_bytes = 0
            plan.commits += 1
            return
        layers = torch.tensor([i for i, rows, _ in plan.pending for _ in rows], device=self.device)
        rows = torch.tensor([r for _, rows, _ in plan.pending for r in rows], device=self.device)
        slots = plan.slots[rows]
        states = torch.cat([s for _, _, s in plan.pending], dim=0)
        # _prune broadcasts the identical batch-1 cold probe to every row.
        self._prune(layers, slots, states, warm=False)
        plan.pending.clear()
        plan.pending_bytes = 0
        plan.commits += 1

    def finish_extend(self, plan):
        if getattr(self, "_extend_plan", None) is not plan:
            raise RuntimeError("stale Mamba extend finish")
        if plan.written != {(layer, row) for layer in range(len(self.layer_ids))
                            for row in range(plan.slots.numel())}:
            raise RuntimeError("Mamba extend missing layer writes")
        self._flush_extend(plan)
        self.valid[:, plan.slots] = True
        self._host_valid[:, plan.host_slots] = True
        self._extend_plan = None

    def abort_extend(self, plan):
        if getattr(self, "_extend_plan", None) is plan:
            plan.pending.clear()
            self.valid[:, plan.slots] = False
            self._host_valid[:, plan.host_slots] = False
            self._extend_plan = None

    def _allocate(self, layers, heads, p, n, size, conv_dim, conv_width):
        if self.exact_mode or self.rank + self.window > min(p, n):
            raise ValueError(
                "factored Mamba requires 0 < rank,window and rank+window <= min(P,N)"
            )
        dense = self._initial_dense
        if dense is not None and (
            dense.shape != (layers, size, 0, 0, 0)
            or dense.dtype != torch.float32
            or dense.device != self.device
        ):
            raise ValueError(
                "factored Mamba requires EMPTY native FP32 temporal storage"
            )
        conv = self._initial_conv
        if conv is not None and (
            conv.shape != (layers, size, conv_dim, conv_width)
            or conv.dtype != torch.float32
            or conv.device != self.device
        ):
            raise ValueError(
                "factored Mamba native conv geometry/dtype/device mismatch"
            )
        capacity = 1 << (self.rank + self.window - 1).bit_length()

        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(*shape, device=self.device, dtype=dtype)

        self.dense = dense if dense is not None else zeros(layers, size, 0, 0, 0)
        self.conv = (
            conv if conv is not None else zeros(layers, size, conv_dim, conv_width)
        )
        self.coeff = zeros(layers, size, heads, n)
        self.u = zeros(layers, size, heads, capacity, p)
        self.v = zeros(layers, size, heads, capacity, n)
        self.fcount = zeros(layers, size, heads, dtype=torch.int32)
        self.count = zeros(
            layers, size, dtype=torch.int32
        )  # lifecycle compatibility only
        self.valid = zeros(layers, size, dtype=torch.bool)

    def _prune(self, layers, slots, state, *, warm):
        from .state_factor import factorize_left

        if warm:
            raise ValueError("factored Mamba decode expiry is device-owned")
        coeff, left, right, _ = factorize_left(
            state,
            self.directions[layers],
            self.rank,
            probe=self._probes[False].expand(layers.numel(), -1, -1, -1),
        )
        if self.prefill_commit_mode == "qr":
            from sglang.srt.layers.attention.linear.kernels.mamba2_prefill_commit import qr_store
            qr_store(self, layers, slots, coeff, left, right)
            self.count[layers, slots] = 0
            return
        q, r = torch.linalg.qr(left, mode="reduced")
        self.coeff[layers, slots] = coeff
        self.u[layers, slots, :, : self.rank] = q.transpose(-1, -2)
        self.v[layers, slots, :, : self.rank] = r @ right
        self.fcount[layers, slots] = self.rank
        self.count[layers, slots] = 0

    def resident_bytes(self):
        # Native conv is shared, included once. No dense, warm, or cold-factor
        # duplicate. Probes/directions are shared fixed overhead, not per-slot.
        return sum(
            getattr(self, name).numel() * getattr(self, name).element_size()
            for name in self.fields
        )

    def materialize(self, layer, slot):
        i = self.layer_map[layer]
        if not self._host_valid[i, slot]:
            raise RuntimeError("uninitialized factored Mamba slot")
        mask = (
            torch.arange(self.u.shape[-2], device=self.device)[None, :]
            < self.fcount[i, slot, :, None]
        )
        return self.directions[i, :, :, None] * self.coeff[
            i, slot, :, None, :
        ] + self.u[i, slot].transpose(-1, -2) @ (self.v[i, slot] * mask[..., None])

    def validate_load_cpu_slots(self, data, indices, *, _native_owned=False):
        result = super().validate_load_cpu_slots(
            data, indices, _native_owned=_native_owned
        )
        count = data["fcount"].cpu()
        valid = data["valid"].cpu()[..., None].expand_as(count)
        if bool(
            (
                (count[valid] < self.rank) | (count[valid] >= self.rank + self.window)
            ).any()
        ):
            raise ValueError("invalid factored Mamba snapshot phase")
        return result


def enabled():
    import os

    return (
        os.environ.get("SGLANG_LTN_FACTORED_DECODE", "0") == "1"
        and os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE", "")
        == "sglang.srt.models.lightning_duet"
    )


def per_slot_bytes(shape, layers, rank, window, conv_itemsize=4):
    heads, p, n = shape.temporal
    capacity = 1 << (rank + window - 1).bit_length()
    conv = sum(a * b for a, b in shape.conv) * conv_itemsize
    return layers * (conv + heads * (n + capacity * (p + n)) * 4 + heads * 4 + 4 + 1)


def configured_bytes(cache_params, layers):
    import json
    import os
    from pathlib import Path

    from sglang.srt.runtime_context import get_server_args

    from .options import DuetOptions

    spec = json.loads((Path(os.environ["SGLANG_DUET_DIR"]) / "spec.json").read_text())
    opts = DuetOptions.resolve(spec, get_server_args())
    per_slot = per_slot_bytes(
        cache_params.shape,
        layers,
        opts.decode_ssm_r,
        opts.decode_ssm_w,
        cache_params.dtype.conv.itemsize,
    )
    heads, p, n = cache_params.shape.temporal
    width = min(opts.decode_ssm_r + 8, p, n)
    # Shared direction + the bridge constructor's cold/warm random probes.
    fixed = (layers * heads * p + heads * n * (2 * width - opts.decode_ssm_r)) * 4
    return per_slot, fixed
