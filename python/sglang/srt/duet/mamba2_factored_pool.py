"""Experimental Lightning factor authority; native dense remains prefill scratch.

The existing bridge owns lifecycle validation and prompt-final factorization.
Decode owns only coeff/U/V/fcount plus native conv. Cold factors and warm
workspace remain allocated for now; this is not a pool-capacity optimization.
"""
import torch

from .dense_bridge import DenseBridgeStatePool


class Mamba2FactoredStatePool(DenseBridgeStatePool):
    fields = (*DenseBridgeStatePool.fields, "u", "v", "fcount")

    def _allocate(self, layers, heads, p, n, size, conv_dim, conv_width):
        if self.exact_mode or self.rank + self.window > min(p, n):
            raise ValueError("factored Mamba requires 0 < rank,window and rank+window <= min(P,N)")
        super()._allocate(layers, heads, p, n, size, conv_dim, conv_width)
        capacity = 1 << (self.rank + self.window - 1).bit_length()
        self.u = torch.zeros(layers, size, heads, capacity, p, device=self.device)
        self.v = torch.zeros(layers, size, heads, capacity, n, device=self.device)
        self.fcount = torch.zeros(layers, size, heads, device=self.device, dtype=torch.int32)

    def _prune(self, layers, slots, state, *, warm):
        # Only prompt initialization uses the published randomized factorizer.
        # Reorthogonalize its left factor once without changing its product.
        super()._prune(layers, slots, state, warm=warm)
        q, r = torch.linalg.qr(self.left[layers, slots], mode="reduced")
        self.u[layers, slots, :, :self.rank] = q.transpose(-1, -2)
        self.v[layers, slots, :, :self.rank] = r @ self.right[layers, slots]
        self.fcount[layers, slots] = self.rank

    def materialize(self, layer, slot):
        i = self.layer_map[layer]
        if not self._host_valid[i, slot]:
            raise RuntimeError("uninitialized factored Mamba slot")
        mask = torch.arange(self.u.shape[-2], device=self.device)[None, :] < self.fcount[i, slot, :, None]
        return (self.directions[i, :, :, None] * self.coeff[i, slot, :, None, :]
                + self.u[i, slot].transpose(-1, -2) @ (self.v[i, slot] * mask[..., None]))

    def validate_load_cpu_slots(self, data, indices, *, _native_owned=False):
        result = super().validate_load_cpu_slots(data, indices, _native_owned=_native_owned)
        count = data["fcount"].cpu()
        valid = data["valid"].cpu()[..., None].expand_as(count)
        if bool(((count[valid] < self.rank) | (count[valid] >= self.rank + self.window)).any()):
            raise ValueError("invalid factored Mamba snapshot phase")
        return result
