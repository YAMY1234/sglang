"""Real materialization plan under Triton CPU interpretation, page64 and PD256."""
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

os.environ['TRITON_INTERPRET'] = '1'
import torch
import triton
import triton.language as tl

root = Path(__file__).resolve().parents[4]
cache = root/'python/sglang/srt/mem_cache'

def load(name):
    spec = importlib.util.spec_from_file_location(name, cache/(name+'.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

policy = load('flashnext_pd_page_policy')
materialization = load('flashnext_materialization')
old_text = subprocess.check_output(['git','show','a182d648d7c:python/sglang/srt/mem_cache/flashnext_materialization.py'], cwd=root, text=True)
old_plan = next(x for x in ast.parse(old_text).body if isinstance(x,ast.FunctionDef) and x.name=='_plan')
old_scope=dict(triton=triton,tl=tl)
exec(compile(ast.Module(body=[old_plan],type_ignores=[]),'<legacy-materialization>','exec'),old_scope)


def plan(fn, page, locations, mapping, *, legacy=False):
    n=locations.numel();groups=n//4
    positions=torch.empty(n,dtype=torch.int64);rope=torch.empty((n,3),dtype=torch.int64)
    kv=torch.empty((5,n),dtype=torch.int64);compressed=torch.empty((5,groups),dtype=torch.int32)
    rows=torch.empty((groups,4),dtype=torch.int32)
    fn[(triton.cdiv(n,256),)](locations,mapping,positions,rope,kv,compressed,rows,
        n,groups,1024,mapping.stride(0),256,**({} if legacy else dict(PAGE=page)))
    return positions,rope,kv,compressed,rows


class PDPageTest(unittest.TestCase):
    def test_default_policy_and_pd_only_opt_in(self):
        for role in ('null','prefill','decode'):
            cfg=SimpleNamespace(disaggregation_mode=role,flashnext_pd_page256=False)
            self.assertEqual(policy.qsa_page_size(cfg,None),64)
            with self.assertRaises(ValueError):policy.validate_arena_page(256,cfg)
            cfg.flashnext_pd_page256=True
            if role=='null':
                with self.assertRaises(ValueError):policy.qsa_page_size(cfg,'triton')
            else:
                self.assertEqual(policy.qsa_page_size(cfg,'triton'),256)
                policy.validate_arena_page(256,cfg)
            with self.assertRaises(ValueError):policy.qsa_page_size(cfg,'trtllm_mha')

    def test_native_plan_default_is_bitwise_legacy(self):
        torch.manual_seed(450)
        for n in (1,3,4,63,64,257,1027):
            locations=torch.cat([torch.arange(p*64,(p+1)*64) for p in (3,1,8,7,2)*4])[:n]
            mapping=torch.randperm(80).reshape(16,5).to(torch.int32)
            old=plan(old_scope['_plan'],64,locations,mapping,legacy=True)
            new=plan(materialization._plan,64,locations,mapping)
            for a,b in zip(old,new):self.assertTrue(torch.equal(a,b))

    def test_page256_physical_plan_matches_independent_payload_lookup(self):
        # Fragmented virtual/physical pages, partial tail, group boundary.
        for page in (64,256):
            for n in (1,3,4,page-1,page,page+4,1027):
                locations=torch.cat([torch.arange(p*page,(p+1)*page) for p in (3,1,8,7,2)*4])[:n]
                mapping=torch.arange(80).reshape(16,5).flip(0).to(torch.int32)
                _,_,kv,compressed,rows=plan(materialization._plan,page,locations,mapping)
                payload=torch.arange(80*page)
                for layer in range(5):
                    expected=torch.tensor([int(mapping[int(i)//page,layer])*page+int(i)%page for i in locations])
                    self.assertTrue(torch.equal(payload[kv[layer]],payload[expected]))
                    self.assertTrue(torch.equal(compressed[layer],expected[:n//4*4:4]//4))
                self.assertTrue(torch.equal(rows.flatten(),torch.arange(n//4*4)))

    def test_native_latent_mapping_and_admission_use_selected_page(self):
        tree=ast.parse((cache/'flashnext_unified_pool.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FlashNextUnifiedLatentPool')
        functions=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_payload_locations','admission_units')]
        scope={};exec(compile(ast.Module(body=functions,type_ignores=[]),str(cache/'flashnext_unified_pool.py'),'exec'),scope)
        for page in (64,256):
            pool=SimpleNamespace(page_size=page,physical_page_map=torch.arange(90).reshape(10,9),
                request_bound=lambda r:r.length,private=SimpleNamespace(pages_needed=lambda rid,bound:2))
            ids=torch.tensor([page-1,page,page+3,3*page-1])
            actual=scope['_payload_locations'](pool,ids)
            expected=torch.tensor([[int(pool.physical_page_map[int(i)//page,j])*page+int(i)%page for j in (7,8)] for i in ids])
            self.assertTrue(torch.equal(actual,expected))
            req=SimpleNamespace(rid='r',length=page+1,prefix_indices=list(range(page)),kv=SimpleNamespace(kv_allocated_len=page))
            self.assertEqual(scope['admission_units'](pool,req),2*5+9)


if __name__=='__main__':unittest.main()
