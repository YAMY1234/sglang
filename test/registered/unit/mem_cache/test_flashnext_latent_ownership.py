"""CPU admission/ownership invariants, independent of CUDA availability."""
import importlib.util
from pathlib import Path
import random
import sys
import unittest

source = Path(__file__).resolve().parents[4] / 'python/sglang/srt/mem_cache/flashnext_latent_layout.py'
spec = importlib.util.spec_from_file_location('latent_ownership_contract', source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class PrivateOwnershipTests(unittest.TestCase):
    def test_shared_prompt_never_shares_deep_pages(self):
        pool = module.PrivatePageOwners(4*64, 64)
        a = pool.bind('request-a-same-prompt', 65)
        b = pool.bind('request-b-same-prompt', 65)
        self.assertFalse(set(a) & set(b))
        self.assertFalse(pool.can_admit('request-c', 1))
        pool.release('request-a-same-prompt')
        self.assertEqual(len(pool.free), 2)
        self.assertEqual(pool.bind('request-b-same-prompt', 65), b)
        c = pool.bind('request-c', 65)
        self.assertEqual(set(a), set(c))
        self.assertFalse(set(b) & set(c))

    def test_batch_admission_counts_prefix_hits_and_pending_requests(self):
        pool = module.PrivatePageOwners(6*64, 64)
        pending = [('a', 128), ('a', 128), ('b', 128)]
        self.assertTrue(pool.can_admit('c', 128, pending))
        self.assertFalse(pool.can_admit('c', 129, pending))
        self.assertEqual(len(pool.free), 6)
        pool.bind('a', 128)
        self.assertTrue(pool.can_admit('c', 128, pending))
        with self.assertRaises(ValueError):
            pool.bind('a', 129)
        with self.assertRaises(ValueError):
            pool.can_admit('huge', 7*64)

    def test_cancel_retract_reuse_stress(self):
        rng = random.Random(272)
        pool = module.PrivatePageOwners(32*64, 64)
        for _ in range(2000):
            key = str(rng.randrange(24))
            if key in pool.owners:
                pool.release(key)
                pool.release(key)  # cancellation cleanup is idempotent
            else:
                tokens = rng.randrange(1, 5*64)
                if pool.can_admit(key, tokens):
                    pool.bind(key, tokens)
            used = [p for pages in pool.owners.values() for p in pages]
            self.assertEqual(len(set(used)), len(used))
            self.assertFalse(set(used) & set(pool.free))
            self.assertEqual(set(used) | set(pool.free), set(range(1, 33)))
        for key in list(pool.owners):
            pool.release(key)
        pool.clear()
        self.assertEqual(len(pool.free), 32)

    def test_wire_geometry_includes_scale_and_token_ids(self):
        self.assertEqual(module.FlashNextLatentLayout(1).token_bytes, 7180)
        self.assertEqual(module.FlashNextLatentLayout(2).token_bytes * 2, 7192)
        with self.assertRaises(ValueError):
            module.FlashNextLatentLayout(4)


if __name__ == '__main__':
    unittest.main()
