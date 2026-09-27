"""Replay private actual-model captures; diagnostic only, never an admission gate."""
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
import triton
import triton.language as tl
from sglang.srt.layers.attention.linear.kernels.gdn_factored import factored_packed_decode
from sglang.srt.layers.attention.linear.kernels.gdn_norm_step import store_normalized
from sglang.kernels.ops.attention.fla import layernorm_gated as norm


@triton.jit
def only_norm(X, Y, Z, W, V:tl.constexpr, EPS:tl.constexpr,
              ROWS:tl.constexpr, ACT:tl.constexpr):
    h=tl.program_id(0);v=tl.arange(0,V)
    x=tl.load(X+h*V+v).to(tl.float32)
    store_normalized(x,Y+h*V+v,Z,W,0,h,0,V,V,EPS,ROWS,ACT)


def different(a,b):
    return int((a.contiguous().view(torch.uint8)!=b.contiguous().view(torch.uint8)).sum().item())


def replay(path, gpu):
    # Captures are produced by this diagnostic, not downloaded/untrusted pickles.
    d=torch.load(path,map_location='cuda' if gpu else 'cpu',weights_only=False)
    def run(fused):
        kw=copy.deepcopy(d['shadow'])
        for name,value in d['captured'].items():kw[name]=value.clone()
        kw['norm_context']=(d['z'],d['weight'],d['eps'],d['rows'],d['activation']) if fused else None
        output=factored_packed_decode(d['mixed'],d['a'],d['b'],**kw)
        states={name:different(kw[name],d['shadow'][name]) for name in d['captured']}
        return output,states
    raw,raw_states=run(False);fused,fused_states=run(True)
    def reference(z):
        return norm.rms_norm_gated(x=raw.reshape(-1,raw.shape[-1]),weight=d['weight'],
            bias=None,z=z,eps=d['eps'],norm_before_gate=True,is_rms_norm=True,
            activation=d['activation']).reshape_as(raw)
    with (nullcontext() if gpu else patch.object(norm,'calc_rows_per_block',return_value=d['rows'])), \
         (nullcontext() if gpu else patch.object(norm,'device_context',return_value=nullcontext())), \
         (nullcontext() if gpu else patch.object(norm,'is_arch_support_pdl',return_value=False)):
        ref3=reference(d['z']);ref2=reference(d['z'].reshape(-1,d['z'].shape[-1]))
    single=torch.empty_like(raw)
    z=d['z'].contiguous()
    only_norm[(raw.shape[-2],)](raw,single,z,d['weight'],raw.shape[-1],d['eps'],
        d['rows'],d['activation'],num_warps=1)
    mask=(fused.view(torch.int16)!=ref2.view(torch.int16)).flatten()
    positions=mask.nonzero().flatten().tolist()
    points=[]
    for i in positions[:16]:
        head=i//raw.shape[-1];col=i%raw.shape[-1]
        points.append(dict(index=i,raw=float(raw.flatten()[i]),z=float(z.flatten()[i]),
            weight=float(d['weight'][col]),fused=float(fused.flatten()[i]),
            reference=float(ref2.flatten()[i]),norm_only=float(single.flatten()[i]),
            variance_fp64=float(raw.flatten()[head*raw.shape[-1]:(head+1)*raw.shape[-1]].double().square().mean())))
    return dict(path=str(path),layer=d['layer'],eps=d['eps'],rows=d['rows'],
        activation=d['activation'],gpu=gpu,output_dtype=str(raw.dtype),
        original_tree=os.environ.get('SGLANG_GDN_NORM_ORIGINAL_TREE','0'),
        raw_state_bytes=raw_states,fused_state_bytes=fused_states,
        raw_vs_saved=different(raw,d['raw']),fused_vs_saved=different(fused,d['actual']),
        reference3_vs_saved=different(ref3,d['reference']),reference2_vs_3=different(ref2,ref3),
        fused_vs_reference=different(fused,ref2),norm_only_vs_reference=different(single,ref2),
        points=points)


if __name__=='__main__':
    gpu=os.environ.get('REPLAY_TEST_DEVICE')=='cuda'
    root=Path(os.environ['SSMOFF_NORM_CAPTURE_ROOT'])
    paths=sorted(p for p in root.glob('*.pt') if not p.stem.endswith('-step0'))
    assert paths, 'no valid decode captures'
    records=[replay(p,gpu) for p in (paths if gpu else paths[:2])]
    print(json.dumps(dict(complete=True,diagnostic_only=True,cuda_math_executed=gpu,
        cases=records)),flush=True)
