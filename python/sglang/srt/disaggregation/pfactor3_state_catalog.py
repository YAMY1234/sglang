"""Factor handoff with one logical field per factor component.

Only the transport catalog changes. Each layer/slot remains an independent
physical row in the existing contiguous allocation; gather/scatter, payload
validation, state arithmetic and publication fences use their original paths.
"""
from dataclasses import dataclass
import os

import numpy as np

from .flashnext_staging import Catalog, Entry


def make_catalog(*, args, pool, mode=None):
    """Default to compact P48 factors; P31 requires its separate opt-in.

    Selection runs once at registration on both P and D. Explicit ``0`` keeps
    the layerwise format; ``1`` requires the compact layout and its strict
    validation. Unset/``auto`` selects only the supported factor layout.
    P31 opt-in coalesces a/U/W/count only; h31 and the 9/8 boundary stay native.
    """
    if mode == "0":
        return Catalog(args=args, pool=pool)
    if mode not in (None, "auto", "1"):
        raise ValueError("SGLANG_PFACTOR3_COMPACT_STATE_FIELDS must be 0, 1 or auto")
    shallow_mode = os.environ.get("SGLANG_PFACTOR4_COMPACT_SHALLOW", "0")
    if shallow_mode not in ("0", "1"):
        raise ValueError("SGLANG_PFACTOR4_COMPACT_SHALLOW must be 0 or 1")
    allow_shallow = shallow_mode == "1"
    if mode != "1":
        mamba = pool.mamba_pool
        records = list(mamba._iter_transfer_state_entries())
        names = {name for name, _, _, _ in records}
        factors = {f"gdn_factored_{kind}" for kind in ("a", "u", "w", "count")}
        layers = {layer for name, _, _, layer in records if name == "gdn_factored_a"}
        prefix_limit = getattr(mamba, "prefix_layer_limit", None)
        if (getattr(pool, "shared_arena", False)
                or (not allow_shallow and ((prefix_limit is not None and prefix_limit < 48)
                    or any("pd_h31" in name for name in names)))
                or not factors.issubset(names) or len(layers) < 2):
            return Catalog(args=args, pool=pool)
    return CompactFactorCatalog(args=args, pool=pool, allow_shallow=allow_shallow)


@dataclass
class LayerRows(Entry):
    slot_count: int = 0
    layers: tuple[int, ...] = ()


class CompactFactorCatalog(Catalog):
    def __init__(self, *, args, pool, allow_shallow=False):
        super().__init__(args=args, pool=pool)
        prefix_limit = getattr(pool.mamba_pool, "prefix_layer_limit", None)
        if allow_shallow and prefix_limit is not None and prefix_limit not in (31, 48):
            raise ValueError("compact shallow fields require the registered P31 boundary")
        if getattr(pool, "shared_arena", False) or (not allow_shallow and any(
            "pd_h31" in entry.name for entry in self.entries
        )):
            raise ValueError("pfactor3 compact fields require the P48 state-only handoff")
        names = tuple(f"mamba.gdn_factored_{kind}.0" for kind in ("a", "u", "w", "count"))
        groups = {name: [e for e in self.entries if e.name == name] for name in names}
        layers = tuple(e.layer for e in groups[names[0]])
        if len(layers) < 2 or len(set(layers)) != len(layers):
            raise ValueError("pfactor3 compact fields require multiple distinct factor layers")
        # Capacity accounts for ALL layer rows even though the merged tensor's
        # first dimension is now physical layer*slot. Keep this byte bound from
        # the validated original registration, before reducing header count.
        self.fixed_transfer_bytes = sum(
            e.tensor[0].nbytes for e in self.entries if not e.tokens_per_row
        )
        merged = {}
        for name, entries in groups.items():
            if tuple(e.layer for e in entries) != layers:
                raise ValueError("factor components disagree on layer order")
            first = entries[0]
            t = first.tensor
            slots = t.shape[0]
            if not t.is_contiguous() or first.tokens_per_row or first.conv_groups:
                raise ValueError("unsupported compact factor row layout")
            for i, entry in enumerate(entries):
                x = entry.tensor
                if (entry.component != first.component or entry.slice_axis != first.slice_axis
                        or x.shape != t.shape or x.stride() != t.stride()
                        or x.dtype != t.dtype or x.device != t.device
                        or x.untyped_storage().data_ptr() != t.untyped_storage().data_ptr()
                        or x.data_ptr() != t.data_ptr() + i * t.nbytes):
                    raise ValueError("factor layer rows must share one contiguous registered allocation")
            flat = t.as_strided((slots * len(layers), *t.shape[1:]), t.stride())
            # The logical layer sequence is part of the wire identity. Peers
            # with another order/subset fail lookup instead of mixing layers.
            wire_name = name + ".layers=" + ",".join(map(str, layers))
            merged[name] = LayerRows(
                layer=-1, name=wire_name, tensor=flat, component=first.component,
                index=first.index, tokens_per_row=0, slice_axis=first.slice_axis,
                slot_count=slots, layers=layers,
            )
        old = self.entries
        self.entries, self.by_key = [], {}
        emitted = set()
        for entry in old:
            if entry.name not in merged:
                self._add(entry)
            elif entry.name not in emitted:
                self._add(merged[entry.name])
                emitted.add(entry.name)

    @staticmethod
    def _indices(entry, kv_indices, kv_by_entry, state_indices):
        if not isinstance(entry, LayerRows):
            return Catalog._indices(entry, kv_indices, kv_by_entry, state_indices)
        slots = np.asarray(state_indices[entry.component], dtype=np.int64).reshape(-1)
        if np.any(slots < 0) or np.any(slots >= entry.slot_count):
            raise ValueError("factor slot outside registered layer storage")
        return (np.arange(len(entry.layers), dtype=np.int64)[:, None] * entry.slot_count
                + slots[None, :]).reshape(-1)
