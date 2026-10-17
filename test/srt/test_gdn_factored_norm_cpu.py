"""P4 capture-time promise: skip norm only when the exact CUDA route will fuse it."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS, ModuleType
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
            self.assertEqual(self.promise(),b in (1,2,8))
    def test_unadmitted_norm_buckets_keep_the_original_cuda_step(self):
        for b in range(1,9):
            self.x,self.z=Tensor((b,5120)),Tensor((b,24,128))
            self.assertTrue(m.cuda_step_eligible(self.gf,self.x,self.x,self.x,self.pool.U[0],self.pool.W[0]))
            self.assertEqual(self.promise(),b in (1,2,8))

    def test_tp1_all_and_only_qualified_buckets(self):
        self.pool.U.shape=self.pool.W.shape=(36,9,48,32,128)
        for b in range(1,9):
            self.x,self.z=Tensor((b,10240)),Tensor((b,48,128))
            self.assertEqual(self.promise(),b<=3)
    def test_served_hybrid_backend_is_unwrapped(self):
        self.backend=NS(linear_attn_backend=self.backend)
        self.assertTrue(self.promise())

    def test_prefill_decode_wrapper_selects_decode_then_linear_leaf(self):
        self.backend=NS(prefill_backend=NS(factored=None),
                        decode_backend=NS(linear_attn_backend=self.backend))
        self.assertTrue(self.promise())

    def test_wrapper_cycle_rejects_fusion(self):
        loop=NS();loop.decode_backend=loop
        self.backend=loop
        self.assertFalse(self.promise())

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

class ModelWiringTest(unittest.TestCase):
    def test_both_served_model_forwards_apply_norm_exactly_once(self):
        # Execute the repository's real forward bodies with CPU projection /
        # attention doubles; imports/CUDA ops are outside this wiring test.
        for file,cls in (('qwen3_next.py','Qwen3GatedDeltaNet'),
                         ('qwen3_5.py','Qwen3_5GatedDeltaNet')):
            tree=ast.parse((ROOT/'python/sglang/srt/models'/file).read_text())
            node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
            fn=next(x for x in node.body if isinstance(x,ast.FunctionDef) and x.name=='forward')
            fn.returns=None
            for arg in fn.args.args: arg.annotation=None
            program=ast.Module(body=[fn],type_ignores=[])
            for fused in (False,True):
                with self.subTest(model=cls,fused=fused):
                    mixed=torch.ones(1,5120,dtype=torch.bfloat16)
                    z=torch.ones(1,24,128,dtype=torch.bfloat16)
                    a=b=torch.ones(1,24,dtype=torch.bfloat16)
                    class Attn:
                        def can_fuse_norm(self,*args): return fused
                        def __call__(self,batch,**kwargs):
                            self.kwargs=kwargs
                            return z[None]*(2 if fused else 1)
                    class Norm:
                        weight=torch.ones(128,dtype=torch.bfloat16)
                        eps=1e-6
                        calls=0
                        def __call__(self,x,z):
                            self.calls+=1
                            return x*2
                    obj=NS(_forward_input_proj=lambda h:(h,h),num_v_heads=24,num_k_heads=12,
                           attn_tp_size=1,head_k_dim=128,head_v_dim=128,attn=Attn(),norm=Norm(),
                           out_proj=lambda x:(x,None))
                    gl=dict(torch=torch,_is_cpu=False,_is_npu=False,_is_xpu=False,_is_cuda=True,
                            _is_amx_available=False,_gdn_decode_fused_proj_conv=False,
                            _GDN_FUSED_QKVZBA_RATIOS=(1,2,3,4),
                            triton=NS(cdiv=lambda a,b:(a+b-1)//b))
                    split=lambda *args:(mixed,z,b,a)
                    gl.update(fused_qkvzba_split_reshape_cat=split,
                              fused_qkvzba_split_reshape_cat_contiguous=split)
                    exec(compile(ast.fix_missing_locations(program),file,'exec'),gl)
                    batch=NS(forward_mode=NS(is_decode=lambda:True,is_extend_without_speculative=lambda:False))
                    y=gl['forward'](obj,torch.ones(1,4096),batch)
                    self.assertTrue(torch.equal(y,torch.full((1,3072),2,dtype=y.dtype)))
                    self.assertEqual(obj.norm.calls,int(not fused))
                    self.assertEqual('norm_weight' in obj.attn.kwargs,fused)
                    if fused:
                        self.assertIs(obj.attn.kwargs['z'],z)
                        self.assertIs(obj.attn.kwargs['norm_weight'],obj.norm.weight)

class CaptureTest(unittest.TestCase):
    def setUp(self):
        path=ROOT/'python/sglang/srt/layers/attention/linear/kernels/gdn_factored_norm_capture.py'
        spec=importlib.util.spec_from_file_location('_p4_capture_cpu',path)
        self.capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.capture)

    def test_missing_flag_and_graph_capture_never_read_tensors(self):
        with tempfile.TemporaryDirectory() as d:
            flag=Path(d)/'enable'
            with patch.dict(os.environ,{'SGLANG_GDN_P4_CAPTURE_DIR':d,
                                         'SGLANG_GDN_P4_CAPTURE_FLAGFILE':str(flag)}), \
                 patch.object(torch.cuda,'is_current_stream_capturing',return_value=True):
                # All inputs deliberately invalid: either guard must exit first.
                self.capture.capture_step(*([None]*18))
                flag.touch()
                self.capture.capture_step(*([None]*18))
            self.assertEqual(list(Path(d).glob('*.pt')),[])

    def test_compacts_slots_and_preserves_payload_and_padding(self):
        with tempfile.TemporaryDirectory() as d:
            flag=Path(d)/'enable';flag.touch()
            fa=torch.arange(6*2*128,dtype=torch.float32).reshape(6,2,128)
            fu=fa[:,:,None,:].expand(6,2,32,128).half().contiguous()
            mixed=torch.randn(3,768).bfloat16();ab=torch.randn(3,2).bfloat16()
            count=torch.arange(6,dtype=torch.int32)[:,None].expand(6,2).contiguous()
            stale=torch.arange(6,dtype=torch.int32);pv=stale+10
            indices=torch.tensor([-1,5,2],dtype=torch.int64)
            weight=torch.randn(128).bfloat16();z=torch.randn(3,2,128).bfloat16()
            args=(mixed,ab,ab,torch.zeros(2),torch.zeros(2).bfloat16(),torch.zeros(2,128),
                  fa,fu,fu,count,stale,indices,pv,True,z,weight,1e-6,1)
            with patch.dict(os.environ,{'SGLANG_GDN_P4_CAPTURE_DIR':d,
                                         'SGLANG_GDN_P4_CAPTURE_FLAGFILE':str(flag),
                                         'SGLANG_GDN_P4_CAPTURE_LIMIT':'2'}), \
                 patch.object(torch.cuda,'is_current_stream_capturing',return_value=False):
                for _ in range(3): self.capture.capture_step(*args)
            paths=sorted(Path(d).glob('*.pt'));self.assertEqual(len(paths),2)
            for path in paths:
                data=torch.load(path,weights_only=False)
                self.assertEqual(data['source_pool_slots'],[0,2,5])
                self.assertEqual(data['slots'].tolist(),[-1,2,1])
                self.assertTrue(torch.equal(data['p']['fa'][0],fa[[0,2,5]]))
                self.assertTrue(torch.equal(data['p']['prefix_valid'],pv[[0,2,5]]))
                self.assertTrue(torch.equal(data['inp']['mixed'],mixed))
                self.assertTrue(torch.equal(data['z'],z))
                self.assertTrue(torch.equal(data['weight'],weight))

if __name__=='__main__':
    unittest.main()
