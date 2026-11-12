# SPDX-License-Identifier: Apache-2.0
"""HiCache L3 storage in remote DRAM pools served over NIXL (``--hicache-storage-backend nixl_dram``).

The object layout, key names and batching are ``MooncakeStore``'s: this backend only swaps the Mooncake client for
``NixlDramClient``, so KV pages and side pools (DSA indexer, draft) are stored as the same objects with the same
keys, written from and read into the registered host pool without a copy.
"""

import logging
from typing import Any, List, Optional, Sequence

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    SidecarPoolSpec,
    disabled_l3_layout,
)
from sglang.srt.mem_cache.pool_host import HostKVCache
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    MooncakeBaseStore,
    MooncakeStore,
)
from sglang.srt.mem_cache.storage.nixl_dram.nixl_dram_client import (
    NixlDramClient,
    NixlDramConfig,
)

logger = logging.getLogger(__name__)


class HiCacheNixlDram(MooncakeStore):
    def __init__(
        self,
        storage_config: Optional[HiCacheStorageConfig] = None,
        mem_pool: Optional[HostKVCache] = None,
        client: Optional[NixlDramClient] = None,
    ):
        # Not MooncakeStore.__init__: that sets up a Mooncake client. The attributes below are the ones its
        # object-layout and batch methods read, set to the same values.
        MooncakeBaseStore.__init__(self)
        extra_config = (storage_config.extra_config if storage_config else None) or {}
        self.config = NixlDramConfig.load(extra_config)
        self.store = client if client is not None else NixlDramClient(self.config)

        self._replicate_config_cls = None
        self._supports_group_ids = False
        self._supports_session_ids = False
        self.enable_group_semantics = False
        self._use_group_semantics = False
        self.dcp_size = storage_config.dcp_size if storage_config is not None else 1

        prefix_parts = []
        if extra_config.get("extra_backend_tag") is not None:
            prefix_parts.append(str(extra_config["extra_backend_tag"]))
        if storage_config is not None and storage_config.model_name:
            prefix_parts.append("-".join(storage_config.model_name.split("/")))
        if self.dcp_size > 1:
            prefix_parts.append(
                f"tp{storage_config.tp_size}_dcp{self.dcp_size}"
                f"_page{storage_config.logical_page_size}"
                f"_pp{storage_config.pp_size}_cp{storage_config.attn_cp_size}"
            )
        self.config_prefix = "_".join(prefix_parts) if prefix_parts else None

        self.storage_config = storage_config
        if storage_config is not None:
            self.local_rank = storage_config.tp_rank
            self.is_mla_backend = storage_config.is_mla_model
            self.pp_rank = storage_config.pp_rank
            self.pp_size = storage_config.pp_size
            self.attn_cp_rank = storage_config.attn_cp_rank
            self.attn_cp_size = storage_config.attn_cp_size
            self.enable_storage_metrics = storage_config.enable_storage_metrics
            self.should_split_heads = storage_config.should_split_heads
        else:
            self.local_rank = 0
            self.is_mla_backend = False
            self.pp_rank, self.pp_size = 0, 1
            self.attn_cp_rank, self.attn_cp_size = 0, 1
            self.enable_storage_metrics = False
            self.should_split_heads = False

        self.enable_pp = self.pp_size > 1
        if self.enable_pp:
            self.mha_suffix = f"{self.local_rank}_{self.pp_rank}"
            self.mla_suffix = f"{self.pp_rank}"
        else:
            self.mha_suffix = f"{self.local_rank}"
            self.mla_suffix = ""
        if self.dcp_size > 1:
            self.mla_suffix = f"dcp{storage_config.dcp_rank}_cp{self.attn_cp_rank}" + (
                f"_{self.pp_rank}" if self.enable_pp else ""
            )
        self.split_factor = 0
        if self.should_split_heads:
            self.split_factor = storage_config.tp_lcm_size // storage_config.tp_size
            base_rank = self.local_rank * self.split_factor
            ranks = [base_rank + i for i in range(self.split_factor)]
            self.mha_suffix = [
                f"{r}_{self.pp_rank}" if self.enable_pp else f"{r}" for r in ranks
            ]

        self.registered_pools = {}
        self.gb_per_page = None
        self.prefetch_pgs, self.backup_pgs = [], []
        self.prefetch_bandwidth, self.backup_bandwidth = [], []
        logger.info(
            "HiCache NIXL DRAM L3 backend: tp_rank %d pp_rank %d/%d, key prefix %s, MLA %s",
            self.local_rank,
            self.pp_rank,
            self.pp_size,
            self.config_prefix,
            self.is_mla_backend,
        )

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        self._set_page_units(mem_pool_host)
        super().register_mem_pool_host(mem_pool_host)

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name):
        self._set_page_units(host_pool)
        super().register_mem_host_pool_v2(host_pool, host_pool_name)

    def _set_page_units(self, pool):
        # page_size rows of dim 0 hold whole pages (dim 0 = page or token) or whole layers (layer_first), so cutting
        # a buffer's registration there never splits a page part MooncakeStore transfers.
        page_size = getattr(pool, "page_size", 0)
        if not self.config.reg_chunk_mib or not page_size:
            return
        buffers = [getattr(pool, "kv_buffer", None), *self._iter_host_pool_buffers(pool)]
        self.store.set_page_units(
            {
                b.data_ptr(): page_size * b.stride(0) * b.element_size()
                for b in buffers
                if b is not None and b.dim() > 1 and b.is_contiguous()
            }
        )

    def describe_l3_object_layout(
        self,
        *,
        sidecar_pool_specs: Sequence[SidecarPoolSpec],
        swa_trailing_pages: Optional[int],
        page_size: int,
    ) -> dict:
        # The objects live in private pools behind this backend, not in a Mooncake master a router could query.
        return disabled_l3_layout(
            "NIXL DRAM L3 objects are not visible outside the engine",
            backend="nixl_dram",
            page_size=page_size,
        )

    def _put_batch_zero_copy_impl(
        self,
        key_strs: List[str],
        buffer_ptrs: List[Any],
        buffer_sizes: List[Any],
        group_ids: Optional[List[str]] = None,
        session_ids: Optional[List[str]] = None,
    ) -> List[int]:
        return self.store.batch_put_from(key_strs, buffer_ptrs, buffer_sizes)

    def _get_batch_zero_copy_impl(
        self, key_strs: List[str], buffer_ptrs: List[Any], buffer_sizes: List[Any]
    ) -> List[int]:
        return self.store.batch_get_into(key_strs, buffer_ptrs, buffer_sizes)

    def close(self):
        self.store.close()
