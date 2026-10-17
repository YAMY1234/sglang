"""SGLANG_GDN_FACTORED_COPY_SYNC_FREE (TwinStar docs/234 (A')): radix-COW copy_slots without host syncs.

CPU cases: both paths leave every pool tensor bitwise identical (incl. the prefix-layer-limit branch and repeated /
overlapping-free index sets), and the gate's check mode accepts the sync-free write and rejects its three negative
controls.  CUDA case: the sync-free copy_slots raises nothing under torch's sync debug mode "error", the default one
does synchronize (the premise of the change).
"""

import os
import unittest
import warnings
from unittest import mock

from test_gdn_factored_host_sync import HAVE_TORCH, POOL_FIELDS, _module, _pool, _same

if HAVE_TORCH:
    import torch

ON = {"SGLANG_GDN_FACTORED_COPY_SYNC_FREE": "1"}
CASES = (([1], [2]), ([3, 4], [5, 6]), ([7, 7], [1, 2]), ([2, 6, 4], [3, 1, 7]))


@unittest.skipUnless(HAVE_TORCH, "needs torch + triton")
class CopySyncFreeTest(unittest.TestCase):
    def setUp(self):
        self.fp = _module()
        from sglang.srt.mem_cache import gdn_cow_sync_free

        self.cow = gdn_cow_sync_free

    def _pair(self, src, dst, limit=None):
        pools = [_pool(self.fp, False) for _ in range(2)]
        for pool, env in zip(pools, ({}, ON)):
            if limit is not None:
                pool.prefix_layer_limit = limit
            with mock.patch.dict(os.environ, env):
                pool.copy_slots(torch.tensor(src), torch.tensor(dst))
        return pools

    def test_matches_assignment(self):
        for limit in (None, 2):
            for src, dst in CASES:
                pools = self._pair(src, dst, limit)
                for name in POOL_FIELDS:
                    _same(self, getattr(pools[0], name), getattr(pools[1], name), (limit, src, dst, name))

    def test_default_path_untouched_when_off(self):
        calls = self.cow.STATS["calls"]
        for value in (None, "0"):
            pool = _pool(self.fp, False)
            with mock.patch.dict(os.environ):
                os.environ.pop("SGLANG_GDN_FACTORED_COPY_SYNC_FREE", None)
                if value is not None:
                    os.environ["SGLANG_GDN_FACTORED_COPY_SYNC_FREE"] = value
                pool.copy_slots(torch.tensor([1]), torch.tensor([2]))
        self.assertEqual(self.cow.STATS["calls"], calls)

    def test_check_mode_accepts_and_rejects_negatives(self):
        s0 = dict(self.cow.STATS)
        env = dict(ON, SGLANG_GDN_FACTORED_COPY_SYNC_FREE_CHECK="1")
        for src, dst in CASES:
            pool = _pool(self.fp, False)
            with mock.patch.dict(os.environ, env):
                pool.copy_slots(torch.tensor(src), torch.tensor(dst))
        d = {k: self.cow.STATS[k] - s0[k] for k in s0}
        self.assertEqual(d["checks"], len(CASES))
        self.assertEqual(d["mismatch"], 0)
        self.assertEqual(d["neg_stale"], len(CASES))
        self.assertEqual(d["neg_required"], len(CASES))
        self.assertEqual(d["neg_valid"], len(CASES))
        self.assertEqual(d["neg_na"], 0)

    @unittest.skipUnless(HAVE_TORCH and torch.cuda.is_available(), "CUDA sync counting")
    def test_cuda_sync_free_copy_has_no_sync(self):
        pool = _pool(self.fp, False, device="cuda")
        src, dst = torch.tensor([1, 2], device="cuda"), torch.tensor([3, 4], device="cuda")
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            with mock.patch.dict(os.environ, ON):
                pool.copy_slots(src, dst)
        finally:
            torch.cuda.set_sync_debug_mode(0)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("warn")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pool.copy_slots(src, dst)
        finally:
            torch.cuda.set_sync_debug_mode(0)
        self.assertGreaterEqual(len([w for w in caught if "synchroniz" in str(w.message)]), 3)


if __name__ == "__main__":
    unittest.main()
