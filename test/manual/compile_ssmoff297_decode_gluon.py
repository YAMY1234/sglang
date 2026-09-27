"""Production-image CPU sm103 compilation; does not execute CUDA mathematics."""
import importlib.util
import json
import os
from pathlib import Path
import sys
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
os.environ['TRITON_INTERPRET']='0'
import torch
assert not torch.cuda.is_available()
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource
p=Path(__file__).resolve().parents[2]/'python/sglang/srt/layers/attention/linear/kernels/gdn_decode_gluon.py'
spec=importlib.util.spec_from_file_location('decode_gluon_offline',p)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
rows=[]
for warps in (1,2,4):
    for index in ('*i32','*i64'):
        types=['*bf16']*3+['*fp32','*bf16','*fp32','*fp32']+['*fp16']*2+['*i32']*2+[index,'*bf16','*i32','fp32','fp32']
        constants=dict(MIXED_ROW=5120,A_ROW=24,B_ROW=24,INDEX_STRIDE=1,OUTPUT_ROW=24*128,H=8,HV=24,WARPS=warps,INVALIDATE=True)
        signature=dict(zip(m.packed_step.arg_names,types));signature.update({n:'constexpr' for n in constants})
        result=triton.compile(GluonASTSource(m.packed_step,signature=signature,constexprs=constants),
                             target=GPUTarget('cuda',103,32),options={'num_warps':warps})
        rows.append(dict(warps=warps,index=index,metadata=result.metadata._asdict()))
print(json.dumps(dict(passed=True,offline_compile=True,gpu_math_executed=False,cases=rows),default=str))
