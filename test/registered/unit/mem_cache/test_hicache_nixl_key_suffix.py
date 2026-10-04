"""Regression: the NIXL backend's storage-key suffix ignored pipeline-parallel
rank, so under PP>1 every stage mapped the same page hash to the same file path
and silently overwrote each other's KV slice. The suffix must separate PP (and
CP) ranks exactly like HiCacheFile does, and stay unchanged for PP=1/CP=1 so
existing on-disk keys remain valid.

CPU-only: imports nixl_utils (no `nixl` package needed).
"""

from __future__ import annotations

import unittest

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.nixl.nixl_utils import build_storage_key_suffix
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _cfg(**over) -> HiCacheStorageConfig:
    base = dict(
        tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1,
        is_mla_model=False, enable_storage_metrics=False, is_page_first_layout=True,
        model_name="nvidia/Qwen3.5-397B-A17B-NVFP4",
    )
    base.update(over)
    return HiCacheStorageConfig(**base)


class TestNixlStorageKeySuffix(unittest.TestCase):
    def test_pp_ranks_get_distinct_suffixes(self):
        suffixes = {build_storage_key_suffix(_cfg(pp_size=4, pp_rank=r)) for r in range(4)}
        self.assertEqual(len(suffixes), 4, suffixes)
        self.assertEqual(
            build_storage_key_suffix(_cfg(pp_size=4, pp_rank=2)),
            "_nvidia-Qwen3.5-397B-A17B-NVFP4_0_1_4_2",
        )

    def test_pp1_cp1_suffix_unchanged_from_pre_fix_layout(self):
        # Pre-fix keys were "_{model}_{tp_rank}_{tp_size}" (MHA) / "_{model}" (MLA);
        # single-stage deployments must keep reading their existing files.
        self.assertEqual(
            build_storage_key_suffix(_cfg(tp_rank=1, tp_size=2)),
            "_nvidia-Qwen3.5-397B-A17B-NVFP4_1_2",
        )
        self.assertEqual(
            build_storage_key_suffix(_cfg(is_mla_model=True, tp_rank=1, tp_size=2)),
            "_nvidia-Qwen3.5-397B-A17B-NVFP4",
        )

    def test_matches_file_backend_suffix_rules(self):
        # Same rule set as HiCacheFile.__init__ (hicache_storage.py): TP part only
        # for non-MLA, PP part when pp_size>1, CP part when attn_cp_size>1.
        self.assertEqual(
            build_storage_key_suffix(
                _cfg(is_mla_model=True, pp_size=2, pp_rank=1, attn_cp_size=2, attn_cp_rank=1)
            ),
            "_nvidia-Qwen3.5-397B-A17B-NVFP4_2_1_cp1_2",
        )


if __name__ == "__main__":
    unittest.main()
