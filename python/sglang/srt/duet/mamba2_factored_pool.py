"""Experimental Lightning factor authority; native dense remains prefill scratch.

The existing bridge supplies lifecycle validation; dense storage is empty.
Prompt-final factorization uses temporary one-layer state. Decode owns only
coeff/U/V/fcount plus native conv, with no warm or cold-factor duplicates.
"""

import torch

from .dense_bridge import DenseBridgeStatePool


class Mamba2FactoredStatePool(DenseBridgeStatePool):
    fields = ("coeff", "u", "v", "fcount", "count", "valid", "conv", "dense")

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
