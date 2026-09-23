import ast
import importlib.util
from pathlib import Path
import random
import sys
import unittest

root = Path(__file__).resolve().parents[4] / 'python/sglang/srt/mem_cache'
spec = importlib.util.spec_from_file_location('unified_legacy_layout', root/'flashnext_latent_layout.py')
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)
source = root/'flashnext_unified_layout.py'
tree = ast.parse(source.read_text())
scope = {'PrivatePageOwners': legacy.PrivatePageOwners}
# Execute the real implementation without importing SGLang's CUDA package root.
exec(compile(ast.Module(body=[n for n in tree.body if not isinstance(n, ast.ImportFrom)],
                        type_ignores=[]), str(source), 'exec'), scope)
UnifiedPageOwners = scope['UnifiedPageOwners']
UnifiedPrivatePageOwners = scope['UnifiedPrivatePageOwners']


class UnifiedOwnershipTest(unittest.TestCase):
    def test_private_release_returns_prefix_capacity(self):
        arena = UnifiedPageOwners(90)
        private = UnifiedPrivatePageOwners(18*64, 64, arena)
        arena.reserve_shared(range(1, 6))
        private.bind('one', 9*64)
        self.assertEqual(arena.audit()['free_units'], 0)
        with self.assertRaises(MemoryError):
            arena.reserve_shared([6])
        private.release('one')
        arena.reserve_shared(range(6, 11))
        self.assertEqual(arena.audit()['shared_units'], 90)
        self.assertEqual(arena.deep, {})

    def test_shared_release_serves_private_and_preserves_other_readers(self):
        arena = UnifiedPageOwners(90)
        arena.reserve_shared(range(1, 11))
        protected = list(arena.shared[10])
        arena.release_shared(range(1, 6))
        private = UnifiedPrivatePageOwners(18*64, 64, arena)
        a = private.bind('same-prompt-a', 4*64)
        b = private.bind('same-prompt-b', 4*64)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(arena.shared[10], protected)
        self.assertTrue(private.can_admit('last', 64))
        self.assertFalse(private.can_admit('two', 128))
        arena.audit()

    def test_failed_reserve_is_atomic_and_double_free_rejected(self):
        arena = UnifiedPageOwners(20)
        before = list(arena.free)
        with self.assertRaises(MemoryError):arena.reserve_shared([1,2,3])
        self.assertEqual(arena.free, before)
        with self.assertRaises(ValueError):arena.reserve_deep([1,1])
        self.assertEqual(arena.free, before)
        arena.reserve_shared([1]);arena.release_shared([1])
        with self.assertRaises(ValueError):arena.release_shared([1])
        arena.audit()

    def test_mixed_reuse_partition(self):
        rng = random.Random(379);arena = UnifiedPageOwners(450)
        for i in range(1500):
            kind = rng.choice(('shared','deep'));table = getattr(arena,kind)
            width = getattr(arena,kind+'_units')
            if table and rng.random() < .5:
                arena._release(kind, [rng.choice(list(table))])
            elif len(arena.free) >= width:
                arena._reserve(kind, [i+1])
            arena.audit()


if __name__ == '__main__':unittest.main()
