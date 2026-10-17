"""P4 capture-time promise: skip norm only when the exact CUDA route will fuse it."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[2]
NAME = 'sglang.srt.layers.attention.linear.kernels.gdn_factored_norm'
SPEC = importlib.util.spec_from_file_location(NAME, ROOT / 'python' / Path(*NAME.split('.')).with_suffix('.py'))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)

class Tensor:
    def __init__(self, shape, dtype=torch.bfloat16, contiguous=True):
        self.shape, self.dtype, self.device, self.contiguous = shape, dtype, torch.device('cuda'), contiguous
    def is_contiguous(self):
        return self.contiguous
    def __getitem__(self, i):
        return Tensor(self.shape[1:], self.dtype, self.contiguous)

class FusionPromiseTest(unittest.TestCase):
    def setUp(self):
        self.gf = NS(STEP_IMPL_BY_RMAX={32:'cuda'}, STEP_CUDA_BATCHES=frozenset(range(1,9)),
            STEP_ROWS_BY_RMAX={}, STEP_VSPLIT_BY_RMAX={}, STEP_CUDA_MAX_BATCH=0, DEFAULT_KERNEL='split')
        self.pool = NS(U=Tensor((36,9,24,32,128),torch.float16),
                       W=Tensor((36,9,24,32,128),torch.float16), cfg=NS(kernel=None))
        self.backend = NS(factored=self.pool)
        self.batch = NS(forward_mode=NS(is_decode=lambda:True))
        self.x, self.z = Tensor((1,5120)), Tensor((1,24,128))
        self.norm = NS(weight=Tensor((128,)), bias=None, activation='sigmoid',
                       norm_before_gate=True, group_size=None, eps=1e-6)
        self.ext = NS(load_extension=lambda:object())
        package = NS(gdn_factored=self.gf)
        self.mods = {NAME.rsplit('.',1)[0]:package,
                     NAME.rsplit('.',1)[0]+'.gdn_factored_cuda':self.ext}

    def promise(self):
        with patch.dict(os.environ, {'SGLANG_GDN_FACTORED_STEP_FUSE_NORM':'1'}), patch.dict(sys.modules,self.mods):
            return m.can_fuse_norm(self.backend,self.batch,self.x,self.z,self.norm)

    def test_default_off_never_loads_extension(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertFalse(m.can_fuse_norm(None,None,None,None,None))
    def test_tp2_all_and_only_qualified_buckets(self):
        for b in range(1,13):
            self.x, self.z = Tensor((b,5120)), Tensor((b,24,128))
            self.assertEqual(self.promise(),b<=8)
    def test_tp1_all_and_only_qualified_buckets(self):
        self.pool.U.shape=self.pool.W.shape=(36,9,48,32,128)
        for b in range(1,9):
            self.x,self.z=Tensor((b,10240)),Tensor((b,48,128))
            self.assertEqual(self.promise(),b<=3)
    def test_runtime_toggle_captures_separate_arms(self):
        self.assertTrue(self.promise())
        with patch.dict(os.environ,{'SGLANG_GDN_FACTORED_STEP_FUSE_NORM':'0'}):
            self.assertFalse(m.can_fuse_norm(None,None,None,None,None))
        self.assertTrue(self.promise())
    def test_fallbacks_do_not_skip_norm(self):
        cases = [
            (self.backend,'factored',None),(self.batch.forward_mode,'is_decode',lambda:False),
            (self.pool,'_exact_tail_transaction',object()),(self.pool.cfg,'kernel','fused'),
            (self.norm,'activation','silu'),(self.norm,'norm_before_gate',False),
            (self.norm,'group_size',64),(self.norm,'bias',Tensor((128,))),
            (self.norm,'eps',float('nan')),(self.norm,'eps',0),
            (self.norm.weight,'dtype',torch.float32),(self.norm.weight,'contiguous',False),
            (self.z,'contiguous',False),(self.z,'shape',(1,25,128)),
            (self.ext,'load_extension',lambda:None),
            (self.gf,'STEP_IMPL_BY_RMAX',{}),(self.gf,'STEP_CUDA_BATCHES',frozenset()),
            (self.gf,'STEP_CUDA_MAX_BATCH',1),
            (self.gf,'STEP_ROWS_BY_RMAX',{32:4}),(self.gf,'STEP_VSPLIT_BY_RMAX',{32:2}),
        ]
        for obj,attr,value in cases:
            with self.subTest(attr=attr,value=value):
                if attr=='STEP_CUDA_MAX_BATCH':
                    with patch.object(self.x,'shape',(2,5120)),patch.object(self.z,'shape',(2,24,128)),patch.object(obj,attr,value):
                        self.assertFalse(self.promise())
                else:
                    with patch.object(obj,attr,value,create=True):
                        self.assertFalse(self.promise())
    def test_invalidation_requires_int32_prefix(self):
        for dtype in (torch.int32,torch.int64):
            self.assertEqual(m.cuda_step_eligible(self.gf,self.x,self.x,self.x,self.pool.U[0],self.pool.W[0],
                invalidate=True,prefix_valid=Tensor((9,),dtype)),dtype==torch.int32)

if __name__=='__main__':
    unittest.main()
