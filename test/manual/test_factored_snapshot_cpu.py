"""CPU interpretation: multi-tile snapshots with noncontiguous request indices."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

if os.environ.get('TRITON_INTERPRET')!='1' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
    raise RuntimeError('CPU-only interpreter guard')
import torch

path=Path(__file__).resolve().parents[2]/'python/sglang/srt/layers/attention/linear/kernels/gdn_verify_io.py'
spec=importlib.util.spec_from_file_location('snapshot_multitile',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.manual_seed(443)
pool=SimpleNamespace(a=torch.randn(3,8,3,128),U=torch.randn(3,8,3,16,128,dtype=torch.float16),
                     W=torch.randn(3,8,3,16,64,dtype=torch.float16),count=torch.arange(3*8*3,dtype=torch.int32).reshape(3,8,3))
working={n:torch.full((3,4,*getattr(pool,n).shape[2:]),-1,dtype=getattr(pool,n).dtype)
         for n in ('a','U','W','count')}
slots=torch.tensor([5,99,2,99])[::2]
expected={n:t.clone() for n,t in working.items()}
for n,t in expected.items():t[:,:2]=getattr(pool,n)[:,slots]
module.snapshot_factors(pool,working,slots)
for n in working:
    if not torch.equal(working[n].view(torch.uint8),expected[n].view(torch.uint8)):
        raise AssertionError(n)
print(json.dumps(dict(passed=True,device='CPU',triton_interpret=True,layer_count=3,
    K=128,V=64,heads=3,slot_stride=2,tiles_per_request_layer=6,inactive_rows_untouched=True)))
