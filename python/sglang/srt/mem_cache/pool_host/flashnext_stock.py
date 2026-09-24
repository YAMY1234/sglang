"""Opt-in stock QSA/PLE payloads for DRAM HiCache.

Only committed, page-aligned radix prefixes are transported. Candidate state,
pending QSA groups and ReplaySSM input windows are not prefix payloads. Factor
and latent pools deliberately fail qualification here instead of losing state.
"""
from __future__ import annotations

import os

import torch

from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost


def enabled() -> bool:
    return os.environ.get("SGLANG_FLASHNEXT_STOCK_HICACHE") == "1"


def ple_tensors(pool):
    """Layer-first views of committed PLE state; no speculative scratch."""
    result = []
    for state in pool._slot_siblings:
        if isinstance(state, ShortConvPool):
            if state.enabled:
                result.append(("short_conv", state.conv_state))
        elif isinstance(state, NGramPool):
            if state.enabled:
                result.append(("ngram", state.context.unsqueeze(0)))
        else:
            raise ValueError("stock HiCache does not support factor/latent slot state")
    if pool.mamba_cache.temporal.numel() == 0:
        raise ValueError("stock HiCache requires the dense temporal pool")
    if getattr(pool, "prefix_layer_limit", None) is not None:
        raise ValueError("stock HiCache does not support partial-layer pools")
    return result


def _allocate(host, shape, dtype, granularity):
    return ALLOC_MEMORY_FUNCS[host.device_pool.device](
        shape, dtype=dtype, device=host.device, pin_memory=host.pin_memory,
        allocator=host.allocator, registration_granularity_bytes=granularity,
    )


class FlashNextStockMambaHost(MambaPoolHost):
    def __init__(self, device_pool, *args, **kwargs):
        self.ple = ple_tensors(device_pool)
        self.ple_host = []
        self.ple_ptrs = []
        super().__init__(device_pool, *args, **kwargs)

    def get_size_per_token(self):
        extra = sum(t[:, 0].numel() * t.element_size() for _, t in self.ple)
        return super().get_size_per_token() + extra

    def init_kv_buffer(self):
        buffers = super().init_kv_buffer()
        for _, tensor in self.ple:
            shape = (self.size, tensor.shape[0], 1, *tensor.shape[2:])
            host = _allocate(self, shape, tensor.dtype,
                             tensor[:, 0].numel() * tensor.element_size())
            self.ple_host.append(host)
            self.ple_ptrs.append(torch.tensor(
                [t.data_ptr() for t in tensor], dtype=torch.uint64,
                device=self.device_pool.device,
            ))
        return [*buffers, *self.ple_host]

    def backup_from_device_all_layer(self, device_pool, host_indices,
                                     device_indices, io_backend="kernel"):
        super().backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend)
        for (_, tensor), host, ptrs in zip(self.ple, self.ple_host, self.ple_ptrs):
            self._copy_tensor_all_layers_lf_pf(
                tensor, host, device_indices, host_indices, tensor.shape[0],
                io_backend, ptrs,
            )

    def load_to_device_per_layer(self, device_pool, host_indices, device_indices,
                                layer_id, io_backend="kernel", *, is_draft=False):
        super().load_to_device_per_layer(
            device_pool, host_indices, device_indices, layer_id, io_backend,
            is_draft=is_draft,
        )
        if layer_id != 0:
            return
        # PLE ngram context is read before target layer zero. The caller waits
        # for this same layer-completion event before preparing PLE inputs.
        for (_, tensor), host in zip(self.ple, self.ple_host):
            for layer in range(tensor.shape[0]):
                self._copy_tensor_pf_lf(
                    host, tensor[layer], host_indices, device_indices, layer,
                    tensor.shape[0], io_backend,
                )
        if device_pool.replayssm_write_pos is not None:
            # Radix state is a folded checkpoint (the same contract as COW).
            device_pool.replayssm_write_pos[device_indices] = 0

    def _iter_page_tensors(self, index):
        yield from super()._iter_page_tensors(index)
        for tensor in self.ple_host:
            yield tensor[index]

    def get_hybrid_pool_buffer(self):
        return [*super().get_hybrid_pool_buffer(), *self.ple_host]


class FlashNextStockQSAHost(MHATokenToKVPoolHost):
    def __init__(self, device_pool, *args, **kwargs):
        pools = (device_pool, *kwargs.get("mtp_draft_device_pools", ()))
        self.qsa_owners = tuple(p._hicache_qsa_owner for p in pools)
        owner = self.qsa_owners[0]
        self.ratio = owner.qsa_compress_ratio
        self.index_heads = owner.qsa_index_kv_heads
        self.index_dim = owner.qsa_index_head_dim
        self.index_dtype = owner.index_state_dtype
        for pool, item in zip(pools, self.qsa_owners):
            if (item.qsa_compress_ratio, item.qsa_index_kv_heads,
                item.qsa_index_head_dim, item.index_state_dtype) != (
                    self.ratio, self.index_heads, self.index_dim, self.index_dtype):
                raise ValueError("target/draft QSA HiCache schemas differ")
            if len(item.qsa_compressed_k_buffer_pool) != pool.layer_num:
                raise ValueError("QSA index layer mapping does not match KV")
            if pool.head_dim != pool.v_head_dim:
                raise ValueError("stock QSA HiCache requires equal K/V dimensions")
        self.index_buffer = None
        super().__init__(device_pool, *args, **kwargs)
        self.index_device_ptrs = torch.tensor(
            [t.data_ptr() for item in self.qsa_owners
             for t in item.qsa_compressed_k_buffer_pool],
            dtype=torch.uint64, device=device_pool.device,
        )

    def get_size_per_token(self):
        dense = super().get_size_per_token()
        numerator = self.layer_num * self.index_heads * self.index_dim * self.index_dtype.itemsize
        if numerator % self.ratio:
            raise ValueError("QSA index bytes/token must be integral")
        return dense + numerator // self.ratio

    def init_kv_buffer(self):
        if self.layout != "page_first" or self.page_size % self.ratio:
            raise ValueError("stock QSA HiCache requires aligned page_first pages")
        buffer = super().init_kv_buffer()
        self.index_buffer = _allocate(
            self, (self.size // self.ratio, self.layer_num,
                   self.index_heads, self.index_dim), self.index_dtype,
            self.page_size // self.ratio * self.index_layout_bytes,
        )
        return buffer

    @property
    def index_item_bytes(self):
        return self.index_heads * self.index_dim * self.index_dtype.itemsize

    @property
    def index_layout_bytes(self):
        return self.layer_num * self.index_item_bytes

    def _group_indices(self, indices, device):
        if indices.numel() % self.page_size:
            raise ValueError("QSA HiCache accepts complete KV pages only")
        # Allocators supply complete pages in token order. A compressed group
        # never straddles a page, including noncontiguous physical page lists.
        return (indices[::self.ratio] // self.ratio).to(device, non_blocking=True)

    def backup_from_device_all_layer(self, device_pool, host_indices,
                                     device_indices, io_backend):
        if io_backend != "kernel":
            raise ValueError("stock QSA HiCache requires kernel I/O")
        from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla_lf_pf

        if not device_indices.numel():
            return
        super().backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend)
        transfer_kv_all_layer_mla_lf_pf(
            src_layers=self.index_device_ptrs, dst=self.index_buffer,
            src_indices=self._group_indices(device_indices, device_indices.device),
            dst_indices=self._group_indices(host_indices, device_indices.device),
            item_size=self.index_item_bytes, dst_layout_dim=self.index_layout_bytes,
            num_layers=self.layer_num,
        )

    def load_to_device_per_layer(self, device_pool, host_indices, device_indices,
                                layer_id, io_backend, *, is_draft=False):
        if io_backend != "kernel":
            raise ValueError("stock QSA HiCache requires kernel I/O")
        from sgl_kernel.kvcacheio import transfer_kv_per_layer_mla_pf_lf

        if not device_indices.numel():
            return
        super().load_to_device_per_layer(
            device_pool, host_indices, device_indices, layer_id, io_backend,
            is_draft=is_draft,
        )
        owner = device_pool._hicache_qsa_owner
        device_layer = 0 if is_draft else layer_id
        host_layer = layer_id if is_draft else self._host_layer_index(layer_id)
        transfer_kv_per_layer_mla_pf_lf(
            src=self.index_buffer, dst=owner.qsa_compressed_k_buffer_pool[device_layer],
            src_indices=self._group_indices(host_indices, device_indices.device),
            dst_indices=self._group_indices(device_indices, device_indices.device),
            layer_id=host_layer, item_size=self.index_item_bytes,
            src_layout_dim=self.index_layout_bytes,
        )

    def destroy(self):
        if getattr(self, "_destroyed", False):
            return
        # Preserve base allocation ownership: K/V are views, not independently
        # registered allocations. Unregister the original plus the index buffer.
        self.kv_buffer = [self.kv_buffer, self.index_buffer]
        super().destroy()
        self.index_buffer = None

    def get_data_page(self, index, flat=True):
        raise NotImplementedError("stock QSA HiCache is DRAM-only")

    def get_dummy_flat_data_page(self):
        raise NotImplementedError("stock QSA HiCache is DRAM-only")

    def set_from_flat_data_page(self, index, data_page):
        raise NotImplementedError("stock QSA HiCache is DRAM-only")
