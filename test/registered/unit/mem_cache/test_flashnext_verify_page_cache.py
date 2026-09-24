"""CPU address equivalence and per-forward invalidation of final verify cache."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import torch

ROOT=Path(__file__).resolve().parents[4]


class Base:
    def init_forward_metadata(self, fb): pass
    def init_forward_metadata_out_graph(self, fb, in_capture=False): pass
    def init_forward_metadata_in_graph(self, fb): pass
    def get_indexer_metadata(self, layer, fb): return fb.out_cache_loc
    def forward_extend(self, q,k,v,layer,fb,save=True,**kwargs): return fb.out_cache_loc
    def forward_decode(self, q,k,v,layer,fb,save=True,**kwargs): return fb.out_cache_loc


def load_backend():
    name='sglang.srt.layers.attention.qwen_sparse_attn_backend'
    original=sys.modules.get(name)
    stub=ModuleType(name);stub.QwenSparseAttnBackend=Base;sys.modules[name]=stub
    try:
        path=ROOT/'python/sglang/srt/layers/attention/flashnext_latent_backend.py'
        spec=importlib.util.spec_from_file_location('verify_page_cache_backend',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module.FlashNextLatentAttnBackend
    finally:
        if original is None:sys.modules.pop(name,None)
        else:sys.modules[name]=original


Backend=load_backend()
pool_path=ROOT/'python/sglang/srt/mem_cache/flashnext_latent_pool.py'
tree=ast.parse(pool_path.read_text())
pool_class=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='FlashNextLatentPool')
method=next(x for x in pool_class.body if isinstance(x,ast.FunctionDef) and x.name=='deep_batch')
scope={'copy':copy,'torch':torch};exec(compile(ast.Module(body=[method],type_ignores=[]),str(pool_path),'exec'),scope)


class Pool:
    def __init__(self):
        self.deep_req_to_token=torch.arange(8*64).reshape(8,64)
        self.calls=0
    def deep_batch(self,fb):
        self.calls+=1
        return scope['deep_batch'](self,fb)


class PageCacheTest(unittest.TestCase):
    def test_all_metadata_boundaries_recompute_after_positions_or_pages_change(self):
        for phase in ('init_forward_metadata','init_forward_metadata_out_graph','init_forward_metadata_in_graph'):
            pool=Pool();backend=Backend.__new__(Backend)
            backend.latent_pool=pool;backend.deep_backend=Base();backend.verify_page_cache=True
            backend._reset_private_batch()
            fb=SimpleNamespace(forward_mode=SimpleNamespace(is_decode_or_idle=lambda:False,is_target_verify=lambda:True),
                req_pool_indices=torch.tensor([2,5]),positions=torch.tensor([7,8,9,10,19,20,21,22]),
                spec_info=SimpleNamespace(draft_token_num=4,topk=1))
            for repeat in range(3):
                getattr(backend,phase)(fb)
                expected=scope['deep_batch'](pool,fb).out_cache_loc
                start=pool.calls
                for layer in (31,35,39,43,47):
                    self.assertTrue(torch.equal(backend.get_indexer_metadata(layer,fb),expected))
                    self.assertTrue(torch.equal(backend.forward_extend(None,None,None,SimpleNamespace(layer_id=layer),fb),expected))
                self.assertEqual(pool.calls,start)
                fb.positions.add_(1)
                pool.deep_req_to_token.add_(1000)
            self.assertEqual(pool.calls,3)

    def test_disabled_path_rebuilds_and_decode_is_not_cached(self):
        for enabled in (False,True):
            pool=Pool();backend=Backend.__new__(Backend)
            backend.latent_pool=pool;backend.deep_backend=Base();backend.verify_page_cache=enabled
            backend._reset_private_batch()
            fb=SimpleNamespace(forward_mode=SimpleNamespace(is_decode_or_idle=lambda:True,is_target_verify=lambda:False),
                req_pool_indices=torch.tensor([2,5]),seq_lens=torch.tensor([8,20]))
            for _ in range(3):
                actual=backend._deep_batch(fb).out_cache_loc
                self.assertTrue(torch.equal(actual,scope['deep_batch'](pool,fb).out_cache_loc))
                fb.seq_lens.add_(1)
            self.assertEqual(pool.calls,3)


if __name__=='__main__':unittest.main()
