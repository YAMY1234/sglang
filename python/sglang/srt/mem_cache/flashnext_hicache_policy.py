"""Opt-in KV/QSA host cache with GPU-resident recurrent checkpoints.

The Mamba component remains the prefix-match validator. A KV-only host copy
is reusable only while its checkpoint still exists on device. This policy
does not reconstruct missing state or change any recurrent arithmetic.
"""

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_memory


def kv_only_enabled() -> bool:
    return bool(
        envs.SGLANG_FLASHNEXT_HICACHE_KV_ONLY.get()
        and get_memory().enable_hierarchical_cache
    )


def validate_kv_only_stack(cache, kv_pool, memory, storage_backend):
    if cache._tree_core_backend != "python":
        raise ValueError("Flash-Next KV-only HiCache requires the Python tree core")
    if not hasattr(kv_pool, "_hicache_qsa_owner"):
        raise ValueError("Flash-Next KV-only HiCache requires the QSA KV pool")
    if (
        storage_backend is not None
        or memory.hicache_io_backend != "kernel"
        or memory.hicache_mem_layout != "page_first"
        or memory.hicache_write_policy != "write_back"
        or memory.hicache_host_memory_mode != "cache"
    ):
        raise ValueError(
            "Flash-Next KV-only HiCache requires DRAM/kernel/page_first/write_back/cache"
        )
