"""Real paged allocator methods: shared/private reuse and deferred radix frees."""
import abc
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from test_flashnext_unified_layout import UnifiedPageOwners, UnifiedPrivatePageOwners

root=Path(__file__).resolve().parents[4]/'python/sglang/srt/mem_cache/allocator'
scope=dict(torch=torch,abc=abc,get_bool_env_var=lambda _:False,_is_hip=False,
           math=math,EvictParams=SimpleNamespace)
for filename,classname in [('base.py','BaseTokenToKVPoolAllocator'),
                            ('paged.py','PagedTokenToKVPoolAllocator'),
                            ('flashnext_unified.py','FlashNextUnifiedAllocator')]:
    path=root/filename;tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==classname)
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future,cls],type_ignores=[]))
    exec(compile(module,str(path),'exec'),scope)
Allocator=scope['FlashNextUnifiedAllocator']


class UnifiedAllocatorTest(unittest.TestCase):
    def test_private_release_replenishes_prefix_without_second_backing(self):
        arena=UnifiedPageOwners(90)
        pool=SimpleNamespace(arena=arena)
        alloc=Allocator(640,page_size=64,dtype=torch.bfloat16,device='cpu',kvcache=pool,need_sort=True)
        private=UnifiedPrivatePageOwners(18*64,64,arena)
        first=alloc.alloc(5*64)
        private.bind('request',8*64)
        self.assertEqual(alloc.available_size(),0) # Five units cannot hold a 9-unit prefix page.
        self.assertIsNone(alloc.alloc(64))
        self.assertEqual(len(arena.shared),5)
        private.release('request')
        second=alloc.alloc(5*64)
        self.assertIsNotNone(second)
        self.assertEqual(len(arena.free),0)
        alloc.free_group_begin();alloc.free(first);alloc.free_page_ids(second[::64]//64)
        self.assertEqual(alloc.available_size(),0)
        alloc.free_group_end()
        self.assertEqual(alloc.available_size(),640)
        self.assertEqual(alloc.verify_byte_accounting(),[])
        arena.audit()
        # Allocation merges staged logical IDs before taking physical ownership.
        again=alloc.alloc(640)
        self.assertTrue(torch.equal(again,torch.arange(64,704)))
        self.assertEqual(alloc.verify_byte_accounting(),[])

    def test_admission_eviction_flushes_deferred_pages(self):
        arena=UnifiedPageOwners(90)
        pool=SimpleNamespace(arena=arena,admission_units=lambda req,pending:50,
                             can_admit=lambda req,pending:len(arena.free)>=50)
        alloc=Allocator(640,page_size=64,dtype=torch.bfloat16,device='cpu',kvcache=pool,need_sort=False)
        slots=alloc.alloc(640);alloc.free_group_begin()
        calls=[]
        class Cache:
            def evict(_,params):
                calls.append(params.num_tokens)
                alloc.free(slots[:params.num_tokens])
                return SimpleNamespace(num_tokens_evicted=params.num_tokens)
        self.assertTrue(alloc.prepare_private_admission(None,[],Cache()))
        self.assertEqual(calls,[6*64])
        self.assertEqual(len(arena.free),54)
        self.assertEqual(alloc.verify_byte_accounting(),[])
        arena.audit()


if __name__=='__main__':unittest.main()
