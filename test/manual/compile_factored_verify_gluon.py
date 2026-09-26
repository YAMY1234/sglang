"""Offline sm103 compile in the production image, with CUDA hidden.

Gluon has no CPU interpreter. The companion 49-case test interprets the same
resident TL algebra; this check compiles the explicit layouts, not GPU math.
CUDA bitwise admission remains mandatory and separate.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
# Compile-only Gluon must load normal JIT helpers; the separate arithmetic
# process uses TRITON_INTERPRET=1. CUDA stays hidden here.
os.environ['TRITON_INTERPRET'] = '0'
import torch
assert not torch.cuda.is_available()
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource

p=Path(__file__).resolve().parents[2]/'python/sglang/srt/layers/attention/linear/kernels/gdn_verify_gluon.py'
spec=importlib.util.spec_from_file_location('verify_gluon_offline',p)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
types=['*bf16']*3+['*fp32']*4+['*fp16']*2+['*i32']*2+['*i64','*bf16','fp32','fp32']
constants=dict(MIXED_ROW=4*5120,MIXED_STEP=5120,A_ROW=4*24,A_STEP=24,
               B_ROW=4*24,B_STEP=24,INDEX_STRIDE=1,H=8,HV=24,K=128,V=128,RMAX=16,R=8,
               RFULL=16,ITERS=3,REL_TOL=1e-4,TOKENS=4,BATCH=1,GATHER=False,HEAD_MAJOR=False)
signature={n:t for n,t in zip(m._factored_verify_gluon_kernel.arg_names,types)}
for n in constants:signature[n]='constexpr'
source=GluonASTSource(m._factored_verify_gluon_kernel,signature=signature,constexprs=constants)
result=triton.compile(source,target=GPUTarget('cuda',103,32),options={'num_warps':1})
print(json.dumps(dict(passed=True,offline_compile=True,gpu_math_executed=False,
                     target='sm103',metadata=result.metadata._asdict()),default=str))
