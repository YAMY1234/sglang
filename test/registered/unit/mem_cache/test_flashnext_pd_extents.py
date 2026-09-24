import ast
from bisect import bisect_left
import heapq
import importlib.util
from pathlib import Path
import random
import unittest

from test_flashnext_unified_layout import UnifiedPageOwners, UnifiedPrivatePageOwners

root = Path(__file__).resolve().parents[4]/'python/sglang/srt/mem_cache'
scope = dict(UnifiedPageOwners=UnifiedPageOwners, bisect_left=bisect_left, heapq=heapq)
source = root/'flashnext_pd_extents.py'
tree = ast.parse(source.read_text())
exec(compile(ast.Module(body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))],
                        type_ignores=[]), str(source), 'exec'), scope)
Owners, Free = scope['ContiguousDeepOwners'], scope['FreeExtents']


class ExtentTest(unittest.TestCase):
    def test_whole_layer_is_continuous_and_shared_owners_unchanged(self):
        arena = Owners(1079*14)
        arena.reserve_shared(range(1,1080))
        shared = {k:list(v) for k,v in arena.shared.items()}
        private = UnifiedPrivatePageOwners(1079*64,64,arena)
        pages = private.bind('request',69000)
        self.assertEqual(len(pages),1079)
        for layer in range(5):
            ids = [arena.deep[page][layer] for page in pages]
            self.assertEqual(ids,list(range(ids[0],ids[0]+1079)))
        self.assertEqual(arena.shared, shared)
        self.assertEqual(arena.audit()['deep_layer_runs'],5)
        private.release('request')
        self.assertEqual(arena.shared, shared)
        self.assertEqual(len(arena.free),1079*5)
        arena.audit()

    def test_fragmented_fallback_has_exact_partition_without_aliasing(self):
        arena = Owners(180)
        arena.reserve_shared(range(1,21))
        arena.release_shared(range(1,21,2))
        arena.reserve_deep(range(1,19))
        result = arena.audit()
        self.assertGreater(result['fragmented_deep_layers'],0)
        self.assertEqual(result['free_units'],0)
        arena.release_deep(range(1,19))
        arena.release_shared(range(2,21,2))
        self.assertEqual(arena.free.starts,[1])
        self.assertEqual(arena.free.ends,{1:181})

    def test_failure_is_atomic_and_double_free_rejected(self):
        arena = Owners(20)
        before = list(arena.free)
        with self.assertRaises(MemoryError):arena.reserve_shared([1,2,3])
        with self.assertRaises(ValueError):arena.reserve_deep([1,1])
        self.assertEqual(list(arena.free), before)
        arena.reserve_shared([1]);arena.release_shared([1])
        with self.assertRaises(ValueError):arena.release_shared([1])
        with self.assertRaises(ValueError):arena.free.extend([1,1])
        with self.assertRaises(ValueError):arena.free.extend([1])
        arena.audit()

    def test_mixed_lifetimes_and_reuse_preserve_partition(self):
        rng=random.Random(442);arena=Owners(900)
        for i in range(3000):
            kind=rng.choice(('shared','deep'));table=getattr(arena,kind)
            if table and rng.random()<.53:
                arena._release(kind,rng.sample(list(table),min(len(table),rng.randint(1,7))))
            else:
                count=min(len(arena.free)//getattr(arena,kind+'_units'),rng.randint(1,7))
                arena._reserve(kind,range(i*7+1,i*7+1+count))
            arena.audit()

    def test_existing_wire_coalescer_uses_fifteen_deep_segments_and_same_bytes(self):
        import numpy as np
        spec=importlib.util.spec_from_file_location('pd_pages',root/'flashnext_pd_pages.py')
        pages=importlib.util.module_from_spec(spec);spec.loader.exec_module(pages)
        arena=Owners(140)
        arena.reserve_shared(range(1,11));arena.reserve_deep(range(1,11))
        table=pages.request_layer_pages(arena.shared,arena.deep,list(range(1,11)),list(range(1,11)),0)
        table=np.concatenate([table]*3)
        ptrs=[0]*36;dst=[1000+i*32 for i in range(36)]
        plans=pages.entry_transfer_blocks(ptrs,dst,[1]*36,table,np.arange(1,11),[(i,i) for i in range(36)])
        self.assertEqual(sum(len(plans[i]) for i in range(36) if i%12>=7),15)
        memory=bytearray(i%251 for i in range(3000))
        for i,plan in enumerate(plans):
            expected=bytes(memory[p] for p in table[i])
            for src,dest,size in plan:memory[dest:dest+size]=memory[src:src+size]
            self.assertEqual(memory[dst[i]+1:dst[i]+11],expected)


if __name__=='__main__':unittest.main()
