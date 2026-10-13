"""Lightning dense authority, with host-planned/device-checked W16 buckets.

The host mirror is scheduling metadata, never a state reader. Lifecycle operations
are forward-stream boundaries and may read indices on the host. Ordinary decode
uses CPU request metadata and device slot indices; it never reads GPU counters.
Only due matrices enter the unchanged reference solver. No graph/overlap support
is claimed: a transaction owns all its slots until the final layer's prune joins.
"""

import torch

from .state_factor import OMEGA_SEED, OVERSAMPLE, factorize_left
from .state_pool import LeftSinkStatePool


class DenseBridgeStatePool(LeftSinkStatePool):
    fields = ("coeff", "left", "right", "warm", "count", "valid", "conv", "dense")

    def __init__(self, *args, native_dense=None, native_conv=None, **kwargs):
        self._initial_dense = native_dense
        self._initial_conv = native_conv
        super().__init__(*args, **kwargs)
        if native_dense is not None:
            if (
                native_dense.shape != self.dense.shape
                or native_dense.dtype != torch.float32
            ):
                raise ValueError("dense bridge requires the native FP32 SSM geometry")
            self.dense = native_dense
        if native_conv is not None:
            if (
                native_conv.shape != self.conv.shape
                or native_conv.dtype != torch.float32
            ):
                raise ValueError(
                    "dense bridge requires native FP32 convolution history"
                )
            self.conv = native_conv
        self._native_bound = native_dense is not None
        self._host_count = torch.zeros(self.count.shape, dtype=torch.int32)
        self._host_valid = torch.zeros(self.valid.shape, dtype=torch.bool)
        self._transaction = None
        # Separate batch-1 cold/warm draws, exactly the reference seed/shape.
        heads, p = self.directions.shape[1:]
        width = min(self.rank + OVERSAMPLE, p, self.n)
        self._probes = {}
        if self.rank:
            for warm in (False, True):
                generator = torch.Generator(device=self.device).manual_seed(OMEGA_SEED)
                self._probes[warm] = torch.randn(
                    1,
                    heads,
                    self.n,
                    width - (self.rank if warm else 0),
                    generator=generator,
                    device=self.device,
                    dtype=torch.float32,
                )
        self.prune_calls = 0
        self.prune_matrices = 0

    def _allocate(self, layers, heads, p, n, size, conv_dim, conv_width):
        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(*shape, dtype=dtype, device=self.device)

        self.coeff = zeros(layers, size, heads, n)
        self.left = zeros(layers, size, heads, p, self.rank)
        self.right = zeros(layers, size, heads, self.rank, n)
        self.warm = zeros(layers, size, heads, n, self.rank)
        self.count = zeros(layers, size, dtype=torch.int32)
        self.valid = zeros(layers, size, dtype=torch.bool)
        self.conv = (
            self._initial_conv
            if self._initial_conv is not None
            else zeros(layers, size, conv_dim, conv_width)
        )
        # Bound to native storage in production; standalone allocation for CPU gates.
        self.dense = (
            self._initial_dense
            if self._initial_dense is not None
            else zeros(layers, size, heads, p, n)
        )

    def _idle(self):
        if self._transaction is not None:
            raise RuntimeError(
                "slot lifecycle operation during a dense bridge transaction"
            )

    def _host_indices(self, indices):
        idx = torch.as_tensor(indices).detach().cpu().long().reshape(-1)
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= self.size):
            raise IndexError("slot index out of range")
        return idx

    def initialize(self, layer, slot, state, conv):
        self._idle()
        i = self.layer_map[layer]
        if not 0 <= slot < self.size:
            raise IndexError("slot index out of range")
        if self.rank:
            self._prune(
                torch.tensor([i], device=self.device),
                torch.tensor([slot], device=self.device),
                state[None],
                warm=False,
            )
        else:
            self.dense[i, slot].copy_(state)
        self.conv[i, slot].copy_(conv)
        self.count[i, slot] = 0
        self.valid[i, slot] = True
        self._host_count[i, slot] = 0
        self._host_valid[i, slot] = True

    def _prune(self, layers, slots, state, *, warm):
        count = layers.numel()
        if not count:
            return
        coeff, left, right, basis = factorize_left(
            state,
            self.directions[layers],
            self.rank,
            self.warm[layers, slots] if warm else None,
            probe=self._probes[warm].expand(count, -1, -1, -1),
        )
        self.coeff[layers, slots] = coeff
        self.left[layers, slots] = left
        self.right[layers, slots] = right
        self.warm[layers, slots] = basis
        # Rebuild exactly once after prompt-final / W16 prune. This dense is
        # authoritative until the next prune; factors need no innovation ring.
        self.dense[layers, slots] = (
            self.directions[layers, :, :, None] * coeff[:, :, None] + left @ right
        )
        self.count[layers, slots] = 0
        self.prune_calls += 1
        self.prune_matrices += count

    def materialize(self, layer, slot):
        i = self.layer_map[layer]
        if not self._host_valid[i, slot]:
            raise RuntimeError(f"uninitialized dense bridge slot {slot}, layer {layer}")
        return self.dense[i, slot]

    def begin_step(self, host_slots, device_slots):
        """Reserve one writer per slot. Host phases are validated on device.

        -1 is padding; every other invalid slot fails before recurrence. The
        runtime derives host slots from prompt mappings and CPU request IDs.
        The assertion also guards mapping/count/valid against stale restoration.
        """
        self._idle()
        host_slots = list(host_slots)
        real = [s for s in host_slots if s != -1]
        if len(real) != len(set(real)):
            raise ValueError("dense bridge requires one writer per slot")
        if any(s < 0 or s >= self.size for s in real):
            raise IndexError("slot index out of range")
        host = torch.tensor(host_slots, dtype=torch.long)
        if device_slots.shape != host.shape:
            raise ValueError("CPU/device batch sizes differ")
        rows = torch.tensor(
            [r for r, s in enumerate(host_slots) if s != -1], dtype=torch.long
        )
        selected = host[rows]
        if not bool(self._host_valid[:, selected].all()):
            raise RuntimeError("uninitialized dense bridge batch")
        expected_slots = host.to(self.device)
        expected_count = self._host_count[:, selected].to(self.device)
        idx = device_slots[rows.to(self.device)].long()
        # No GPU scalar reads or dynamic nonzero/boolean-index buckets.
        torch._assert_async(
            (device_slots == expected_slots).all(), "stale bridge slot mapping"
        )
        safe_idx = idx.clamp(0, self.size - 1)
        torch._assert_async(self.valid[:, safe_idx].all(), "invalid bridge slot")
        torch._assert_async(
            (self.count[:, safe_idx] == expected_count).all(), "stale bridge phase"
        )
        due_layers, due_rows = [], []
        if not self.exact_mode:
            # This nonzero is CPU-only scheduling metadata, never a dynamic
            # device bucket or a GPU->host counter read.
            li, ri = ((self._host_count[:, selected] + 1) == self.window).nonzero(
                as_tuple=True
            )
            due_layers, due_rows = li.tolist(), rows[ri].tolist()
        self._transaction = (
            selected,
            rows,
            device_slots,
            due_layers,
            due_rows,
            set(),
            idx,
        )
        return device_slots

    def note_layer_step(self, layer):
        if self._transaction is None:
            raise RuntimeError("no dense bridge transaction")
        _selected, _rows, _slots, _, _, written, _idx = self._transaction
        i = self.layer_map[layer]
        if i in written:
            raise RuntimeError("second writer for dense bridge layer")
        written.add(i)

    def finish_step(self):
        if self._transaction is None:
            raise RuntimeError("no dense bridge transaction")
        selected, _rows, slots, due_layers, due_rows, written, idx = self._transaction
        if written != set(range(len(self.layer_ids))):
            raise RuntimeError("dense bridge transaction missing layer writes")
        if not self.exact_mode:
            # One batched metadata commit after all layer writers have joined.
            self.count[:, idx] += 1
        if due_layers:
            layers = torch.tensor(due_layers, device=self.device, dtype=torch.long)
            due = slots[torch.tensor(due_rows, device=self.device)].long()
            self._prune(layers, due, self.dense[layers, due], warm=True)
        if not self.exact_mode:
            self._host_count[:, selected] = (
                self._host_count[:, selected] + 1
            ) % self.window
        self._transaction = None

    def step(self, layer, slot, decay, scaled_x, b, conv):
        """CPU/reference compatibility; production uses indexed native SSU."""
        i = self.layer_map[layer]
        if len(self.layer_ids) != 1:
            raise ValueError("use begin/note/finish for multi-layer recurrence")
        self.begin_step([slot], torch.tensor([slot], device=self.device))
        state = (
            self.materialize(layer, slot) * decay[:, None, None]
            + scaled_x[:, :, None] * b[:, None]
        )
        self.dense[i, slot].copy_(state)
        self.conv[i, slot].copy_(conv)
        self.note_layer_step(layer)
        self.finish_step()
        return state

    def native_sibling(self):
        return NativeBridgeSlotState(self)

    def _lifecycle_fields(self, native_owned):
        return tuple(
            n for n in self.fields if not native_owned or n not in ("dense", "conv")
        )

    def reset_slots(self, indices, *, _native_owned=False):
        self._idle()
        host = self._host_indices(indices)
        for name in self._lifecycle_fields(_native_owned):
            getattr(self, name)[:, indices] = 0
        self._host_count[:, host] = 0
        self._host_valid[:, host] = False

    def copy_slots(self, src_index, dst_index, *, _native_owned=False):
        self._idle()
        src, dst = self._host_indices(src_index), self._host_indices(dst_index)
        if src.numel() != dst.numel() or dst.unique().numel() != dst.numel():
            raise ValueError("copy requires matching sources and unique destinations")
        for name in self._lifecycle_fields(_native_owned):
            value = getattr(self, name)
            value[:, dst_index] = value[:, src_index].clone()
        self._host_count[:, dst] = self._host_count[:, src].clone()
        self._host_valid[:, dst] = self._host_valid[:, src].clone()

    def get_cpu_slots(self, indices, *, _native_owned=False):
        self._idle()
        return {
            name: getattr(self, name)[:, indices].cpu().clone()
            for name in self._lifecycle_fields(_native_owned)
        }

    def load_cpu_slots(self, data, indices, *, _native_owned=False):
        self._idle()
        host = self._host_indices(indices)
        names = self._lifecycle_fields(_native_owned)
        if set(data) != set(names):
            raise ValueError("incomplete dense bridge snapshot")
        for name in names:
            target = getattr(self, name)
            if (
                data[name].shape != target[:, indices].shape
                or data[name].dtype != target.dtype
            ):
                raise ValueError(f"invalid dense bridge snapshot field {name}")
        count, valid = data["count"].cpu(), data["valid"].cpu()
        if bool((count < 0).any()) or (
            self.window and bool((count >= self.window).any())
        ):
            raise ValueError("invalid dense bridge snapshot phase")
        for name in names:
            getattr(self, name)[:, indices] = data[name].to(self.device)
        self._host_count[:, host] = count
        self._host_valid[:, host] = valid

    def iter_transfer_state_entries(self):
        # Native already advertises dense/conv; don't duplicate their transport
        # pointers. The standalone test pool advertises the whole state.
        for name, value, axis, layer in super().iter_transfer_state_entries():
            if self._native_bound and name in (
                self.transfer_prefix + "dense",
                self.transfer_prefix + "conv",
            ):
                continue
            yield name, value, axis, layer


class NativeBridgeSlotState:
    """MambaPool owns dense/conv lifecycle exactly once; this owns the sidecar.

    Separating these writers also prevents an overlapping native copy followed
    by a second alias copy from accidentally undoing a slot permutation.
    """

    def __init__(self, pool):
        self.pool = pool

    def reset_slots(self, indices):
        self.pool.reset_slots(indices, _native_owned=True)

    def copy_slots(self, src, dst):
        self.pool.copy_slots(src, dst, _native_owned=True)

    def get_cpu_slots(self, indices):
        return self.pool.get_cpu_slots(indices, _native_owned=True)

    def load_cpu_slots(self, data, indices):
        self.pool.load_cpu_slots(data, indices, _native_owned=True)

    def iter_transfer_state_entries(self):
        yield from self.pool.iter_transfer_state_entries()
