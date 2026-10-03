"""Opt-in factored GDN state payload for the stock DRAM HiCache (TwinStar docs/139).

The factored arm keeps MambaPool.temporal empty; the committed state of a slot is
the FactoredGDNPool sibling: per GDN layer the sink coefficient a (fp32), the
factors U / W and the truncation count, plus the slot's P-checkpoint flag. Those
are the host payload, next to the stock conv window and PLE. Local authority
metadata (stale, dense_of, dense ring, dense_required) is not: a restored slot is
factored-only and authoritative, exactly like a copied or PD-received slot.

All factor layers and the flag restore with target layer zero. The extend plan
reads the flag and the authority metadata before any per-layer wait, so the GDN
backend waits for layer zero first (hicache_wait_restore).
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNPool
from sglang.srt.mem_cache.pool_host import flashnext_stock
from sglang.srt.mem_cache.pool_host.flashnext_stock import (
    FlashNextStockMambaHost,
    _allocate,
    ple_tensors,
)
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.utils import is_cuda

if is_cuda():
    from sglang.kernels.ops.mamba.transfer_mamba import (
        transfer_kv_mamba_lf_pf,
        transfer_kv_mamba_pf_lf,
    )

# Slot flag rows are padded to the transfer kernel's 16-byte item.
FLAG_WORDS = 4


def enabled() -> bool:
    return os.environ.get("SGLANG_FLASHNEXT_FACTOR_HICACHE") == "1"


def factor_sibling(pool):
    """The single FactoredGDNPool registered on `pool`, or None."""
    found = [s for s in pool._slot_siblings if isinstance(s, FactoredGDNPool)]
    if len(found) > 1:
        raise ValueError("factor HiCache expects one factored GDN pool")
    return found[0] if found else None


def payload_tensors(factor):
    return (factor.a, factor.U, factor.W, factor.count)


def payload_bytes(factor) -> int:
    """Device bytes of the transported factor state (same role as dense temporal)."""
    return sum(t.nbytes for t in payload_tensors(factor)) + factor.prefix_valid.nbytes


def qualify(pool):
    """Return the factor sibling when this opt-in path applies, else None.

    Everything the stock path rejects is still rejected; only the factor sibling
    and the empty dense temporal buffer it implies are accepted in addition.
    """
    factor = factor_sibling(pool)
    if factor is None or not enabled():
        return None
    cfg = factor.cfg
    if cfg.exact_prefix or not cfg.factored_prefix or factor.prefix_valid is None:
        raise ValueError("factor HiCache requires factored_prefix=1 and exact_prefix=0")
    if pool.mamba_cache.temporal.numel() != 0:
        raise ValueError("factor HiCache expects the dense temporal pool to be empty")
    ple_tensors(_without_factor(pool, factor))
    return factor


def _without_factor(pool, factor):
    # Stock qualification of the remaining siblings (PLE, PD boundary state)
    # and of the partial-layer limit, minus the two factor-specific facts.
    return SimpleNamespace(
        _slot_siblings=[s for s in pool._slot_siblings if s is not factor],
        mamba_cache=SimpleNamespace(temporal=torch.ones(1)),
        prefix_layer_limit=getattr(pool, "prefix_layer_limit", None),
    )


class StateBytes:
    """Device-pool bytes for the host budget split: stock formula, factor state added."""

    def __init__(self, pool, factor):
        self.pool, self.factor = pool, factor

    def get_kv_size_bytes(self):
        return self.pool.get_kv_size_bytes() + payload_bytes(self.factor)


def install_restore_wait(factor, req_to_token_pool) -> None:
    def wait():
        counter = getattr(req_to_token_pool, "layer_transfer_counter", None)
        if counter is not None:
            counter.wait_until(0)

    factor.hicache_wait_restore = wait


class FlashNextFactoredMambaHost(FlashNextStockMambaHost):
    def __init__(self, device_pool, *args, **kwargs):
        self.factor = qualify(device_pool)
        if self.factor is None:
            raise ValueError("factor HiCache host requires a qualified factored GDN pool")
        self.ple = ple_tensors(_without_factor(device_pool, self.factor))
        is_boundary = getattr(flashnext_stock, "is_pd_boundary_state", lambda state: False)
        self.pd_boundaries = tuple(s for s in device_pool._slot_siblings if is_boundary(s))
        self.ple_host = []
        self.ple_ptrs = []
        self.factor_host = []
        self.factor_ptrs = []
        self.flag_host = None
        MambaPoolHost.__init__(self, device_pool, *args, **kwargs)
        if self.layout != "page_first":
            raise ValueError("factor HiCache requires the page_first layout")
        device = self.device_pool.device
        rows = self.factor.prefix_valid.shape[0]
        # Separate staging per direction: backup and load run on different streams.
        self.flag_out = torch.zeros(rows, FLAG_WORDS, dtype=torch.int32, device=device)
        self.flag_in = torch.zeros(rows, FLAG_WORDS, dtype=torch.int32, device=device)
        self.flag_out_ptrs = torch.tensor([self.flag_out.data_ptr()], dtype=torch.uint64, device=device)

    def get_size_per_token(self):
        factor = sum(t[:, 0].numel() * t.element_size() for t in payload_tensors(self.factor))
        return super().get_size_per_token() + factor + FLAG_WORDS * 4

    def init_kv_buffer(self):
        buffers = super().init_kv_buffer()
        for tensor in payload_tensors(self.factor):
            item = tensor[0, 0].numel() * tensor.element_size()
            if item % 16:
                raise ValueError("factor HiCache items must be 16-byte multiples")
            host = _allocate(self, (self.size, tensor.shape[0], 1, *tensor.shape[2:]),
                             tensor.dtype, tensor[:, 0].numel() * tensor.element_size())
            self.factor_host.append(host)
            self.factor_ptrs.append(torch.tensor(
                [t.data_ptr() for t in tensor], dtype=torch.uint64,
                device=self.device_pool.device,
            ))
        self.flag_host = _allocate(self, (self.size, 1, 1, FLAG_WORDS), torch.int32, FLAG_WORDS * 4)
        return [*buffers, *self.factor_host, self.flag_host]

    def backup_from_device_all_layer(self, device_pool, host_indices,
                                     device_indices, io_backend="kernel"):
        if io_backend != "kernel":
            raise ValueError("factor HiCache requires kernel I/O")
        if getattr(self.factor, "_pside_deferred_commit", None) is not None:
            raise RuntimeError("factor HiCache backup before the deferred P commit was joined")
        self.factor.pside_join()
        super().backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend)
        if not device_indices.numel():
            return
        for tensor, host, ptrs in zip(payload_tensors(self.factor), self.factor_host, self.factor_ptrs):
            self._copy_tensor_all_layers_lf_pf(
                tensor, host, device_indices, host_indices, tensor.shape[0],
                io_backend, ptrs,
            )
        n = device_indices.numel()
        self.flag_out[:n, 0] = self.factor.prefix_valid[device_indices]
        dst = host_indices.to(device_indices.device, non_blocking=True)
        transfer_kv_mamba_lf_pf(
            src_ptrs=self.flag_out_ptrs, dst=self.flag_host,
            src_indices=torch.arange(n, device=device_indices.device), dst_indices=dst,
            item_size=FLAG_WORDS * 4, dst_layout_dim=FLAG_WORDS * 4, num_layers=1,
        )

    def load_to_device_per_layer(self, device_pool, host_indices, device_indices,
                                layer_id, io_backend="kernel", *, is_draft=False):
        if io_backend != "kernel":
            raise ValueError("factor HiCache requires kernel I/O")
        self.factor.pside_join()
        super().load_to_device_per_layer(
            device_pool, host_indices, device_indices, layer_id, io_backend,
            is_draft=is_draft,
        )
        if layer_id != 0 or not device_indices.numel():
            return
        # Every factor layer lands with layer zero: the extend plan reads the flag
        # before any per-layer wait, and densify graphs may read all layers at once.
        factor, dev = self.factor, device_indices.device
        src = host_indices.to(dev, non_blocking=True).long()
        dst = device_indices.long()
        n = dst.numel()
        for tensor, host in zip(payload_tensors(factor), self.factor_host):
            layers, rows = tensor.shape[:2]
            item = tensor[0, 0].numel() * tensor.element_size()
            layer = torch.arange(layers, device=dev)
            # Page-first host rows are (slot, layer) items; layer-first device rows
            # are (layer, slot) items: one launch moves every layer.
            transfer_kv_mamba_pf_lf(
                src=host, dst=tensor,
                src_indices=(src[:, None] * layers + layer[None, :]).reshape(-1),
                dst_indices=(layer[None, :] * rows + dst[:, None]).reshape(-1),
                layer_id=0, item_size=item, src_layout_dim=item,
            )
        transfer_kv_mamba_pf_lf(
            src=self.flag_host, dst=self.flag_in, src_indices=src,
            dst_indices=torch.arange(n, device=dev), layer_id=0,
            item_size=FLAG_WORDS * 4, src_layout_dim=FLAG_WORDS * 4,
        )
        factor.prefix_valid.index_copy_(0, dst, self.flag_in[:n, 0])
        # Same authority as load_cpu_slots: the restored factors are the state.
        factor.stale.index_fill_(0, dst, 1)
        factor.dense_of.index_fill_(0, dst, -1)
        if factor.dense_required is not None:
            factor.dense_required.index_fill_(0, dst, 0)
        spec = getattr(factor, "spec_state", None)
        if spec is not None:
            spec.invalidate_slots(dst)

    def _iter_page_tensors(self, index):
        yield from super()._iter_page_tensors(index)
        for tensor in (*self.factor_host, self.flag_host):
            yield tensor[index]

    def get_hybrid_pool_buffer(self):
        return [*super().get_hybrid_pool_buffer(), *self.factor_host, self.flag_host]

    def get_data_page(self, index, flat=True):
        raise NotImplementedError("factor HiCache is DRAM-only")

    def get_dummy_flat_data_page(self):
        raise NotImplementedError("factor HiCache is DRAM-only")

    def set_from_flat_data_page(self, index, data_page):
        raise NotImplementedError("factor HiCache is DRAM-only")
