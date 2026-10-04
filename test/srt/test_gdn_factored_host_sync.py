"""SGLANG_GDN_FACTORED_HOST_SYNC_FREE: reset_slots and plan_extend without per-tensor host syncs.

CPU cases check that both paths leave every pool tensor and plan field bitwise
identical. The CUDA case counts synchronizations with torch's sync debug mode.
"""

import copy
import importlib
import os
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

os.environ.setdefault("TRITON_INTERPRET", "1")
ROOT = Path(__file__).resolve().parents[2] / "python"
LAYERS, HV, K, V = 4, 2, 32, 32

try:
    import torch
    import triton  # noqa: F401

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def _envs(enabled):
    env = types.ModuleType("sglang.srt.environ")
    env.envs = types.SimpleNamespace(
        SGLANG_GDN_FACTORED_HOST_SYNC_FREE=types.SimpleNamespace(get=lambda: enabled),
        SGLANG_GDN_K31_CHOLQR_MIXED=types.SimpleNamespace(get=lambda: False))
    return mock.patch.dict(sys.modules, {"sglang.srt.environ": env})


def _module():
    try:
        return importlib.import_module("sglang.srt.mem_cache.gdn_factored_pool")
    except ImportError:
        for name in ("sglang", "sglang.srt", "sglang.srt.utils", "sglang.srt.mem_cache",
                     "sglang.srt.layers", "sglang.srt.layers.attention",
                     "sglang.srt.layers.attention.linear",
                     "sglang.srt.layers.attention.linear.kernels", "sglang.srt.configs",
                     "sglang.srt.model_executor", "sglang.srt.duet"):
            module = types.ModuleType(name)
            module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
            sys.modules[name] = module
        return importlib.import_module("sglang.srt.mem_cache.gdn_factored_pool")


def _pool(fp, enabled, device="cpu"):
    vbar = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
    fixed = torch.Generator().manual_seed(7)
    torch.save({"vbar": {l: torch.randn(HV, V, generator=fixed) for l in range(LAYERS)}}, vbar)
    cfg = fp.FactoredGDNConfig.parse(
        "r=8,m=8,dtype=fp16,ring=4,async=1,strict_chunk=1,init_method=k31,"
        f"decode_method=iter,vbar={vbar},factored_prefix=1")
    params = types.SimpleNamespace(shape=types.SimpleNamespace(temporal=(HV, V, K)))
    with _envs(enabled):
        pool = fp.FactoredGDNPool(size=8, cache_params=params, mamba_layer_ids=list(range(LAYERS)),
                                  device=device, cfg=cfg, tp_rank=0, max_running_requests=8)
    g = torch.Generator().manual_seed(3)
    for t in (pool.a, pool.U, pool.W):
        t.copy_(torch.randn(t.shape, generator=g).to(t.dtype).to(t.device))
    for t in (pool.stale, pool.dense_of, pool.prefix_valid, pool.count):
        t.copy_(torch.randint(0, 3, t.shape, generator=g).to(t.dtype).to(t.device))
    if pool.dense_required is not None:
        pool.dense_required.copy_(torch.randint(0, 2, pool.dense_required.shape, generator=g)
                                  .to(pool.dense_required.dtype).to(pool.dense_required.device))
    return pool


POOL_FIELDS = ("a", "U", "W", "count", "stale", "dense_of", "dense_required", "prefix_valid")


def _same(test, x, y, label):
    if x is None or y is None:
        test.assertIs(x, y, label)
        return
    test.assertEqual(x.dtype, y.dtype, label)
    test.assertTrue(torch.equal(x.cpu(), y.cpu()), label)


@unittest.skipUnless(HAVE_TORCH, "needs torch + triton")
class HostSyncFreeTest(unittest.TestCase):
    def setUp(self):
        self.fp = _module()

    def test_reset_slots_index_fill_matches_assignment(self):
        for indices in ([1], [2, 5, 2], [], [0, 7, 3, 4]):
            pools = [_pool(self.fp, enabled) for enabled in (False, True)]
            for pool in pools:
                pool.reset_slots(torch.tensor(indices, dtype=torch.long))
            for name in POOL_FIELDS:
                _same(self, getattr(pools[0], name), getattr(pools[1], name), (indices, name))

    def test_plan_extend_packed_transfers_match_per_tensor_path(self):
        cases = [
            # (slots, extend_lens, prefix_lens, prompt_final)
            ([1, 2], [64, 128], [0, 0], [True, True]),
            ([3, -1, 4], [256, 0, 64], [64, 0, 0], [False, True, True]),
            ([5], [32], [0], [False]),
        ]
        for slots, lens, prefix, final in cases:
            pools = [_pool(self.fp, enabled) for enabled in (False, True)]
            for pool in pools:
                # Prefix rows need a valid checkpoint unless they reuse the ring.
                pool.prefix_valid.fill_(1)
                pool.dense_required.zero_()
            plans = [pool.plan_extend(torch.tensor(slots), lens, prefix_lens=prefix,
                                      prompt_final=final) for pool in pools]
            for field in ("slots", "use_ring", "ring_src", "ring_dst", "ring_dst_rows",
                          "dense_required_after_commit", "use_prefix"):
                _same(self, getattr(plans[0], field), getattr(plans[1], field), (slots, field))
            for field in ("n_ring_src", "n_ring_miss", "all_fresh", "next_layer", "last_layer"):
                self.assertEqual(getattr(plans[0], field), getattr(plans[1], field), (slots, field))
            for name in POOL_FIELDS:
                _same(self, getattr(pools[0], name), getattr(pools[1], name), (slots, name))
            self.assertEqual(pools[0].ring_owner, pools[1].ring_owner)
            self.assertEqual(pools[0].ring_lru, pools[1].ring_lru)

    @unittest.skipUnless(HAVE_TORCH and torch.cuda.is_available(), "CUDA sync counting")
    def test_cuda_reset_has_no_sync_and_plan_has_one(self):
        pool = _pool(self.fp, True, device="cuda")
        pool.prefix_valid.fill_(1)
        pool.dense_required.zero_()
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            pool.reset_slots(torch.tensor([1, 2], device="cuda"))
        finally:
            torch.cuda.set_sync_debug_mode(0)
        torch.cuda.set_sync_debug_mode("warn")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pool.plan_extend(torch.tensor([3, 4], device="cuda"), [64, 64],
                                 prefix_lens=[0, 0], prompt_final=[True, True])
        finally:
            torch.cuda.set_sync_debug_mode(0)
        syncs = [w for w in caught if "synchroniz" in str(w.message)]
        self.assertEqual(len(syncs), 1, [str(w.message) for w in syncs])


if __name__ == "__main__":
    unittest.main()
