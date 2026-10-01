"""Native factor backend/pool replay for the default-off decode metadata port.

CPU uses the Triton interpreter plus offline SM103 compilation. CUDA repeats
served dimensions and graph replay with changing live slot/mask controls.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNConfig, FactoredGDNPool
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.linear.kernels import gdn_factored as kernels

FIELDS=('a','U','W','count','stale','prefix_valid','dense_of','dense_required','dense_ring')


def pool(device,batch,layers,heads,width):
    p=FactoredGDNPool(size=2*batch+4,cache_params=NS(shape=NS(temporal=(heads,width,width))),
        mamba_layer_ids=list(range(layers)),device=device,
        cfg=FactoredGDNConfig(dtype=torch.float16,strict_chunk=1,factored_prefix=1,ring=1,kernel='split'))
    torch.manual_seed(0x504634)
    p.a.normal_();p.U.normal_(std=.05);p.W.normal_(std=.05)
    p.prefix_valid.fill_(1)
    return p


def equal(left,right):
    for name in FIELDS:
        a,b=getattr(left,name),getattr(right,name)
        assert (a is None and b is None) or torch.equal(a,b),name
    assert left.ring_owner==right.ring_owner
    assert left.ring_lru==right.ring_lru


def tracking_case(device,batch,index_dtype,kind,mask_dtype=torch.bool):
    p=pool(device,batch,3 if device=='cpu' else 36,2 if device=='cpu' else 24,16 if device=='cpu' else 128)
    src=torch.arange(1,batch+1,device=device,dtype=index_dtype)
    dst=src+batch
    mask=torch.ones(batch,device=device,dtype=mask_dtype)
    if kind=='alias':dst[0]=src[0]
    if kind=='negative_src':src[0]=-1
    if kind=='negative_dst':dst[0]=-1
    if kind=='masked':mask[::2]=0
    if kind=='padding':src[-1]=dst[-1]=-1;mask[-1]=0
    before,after=copy.deepcopy(p),copy.deepcopy(p)
    errors=[]
    for owner,enabled in ((before,False),(after,True)):
        owner.decode_metadata_fused=enabled
        try:owner.track_copy(src,mask,dst);errors.append(None)
        except (RuntimeError,TypeError) as exc:errors.append(type(exc).__name__)
    assert errors[0]==errors[1],errors
    equal(before,after)
    if mask_dtype==torch.bool:assert errors[0] is None
    else:assert errors[0] is not None,'native where requires boolean mask'
    return dict(stage='track',B=batch,kind=kind,index_dtype=str(index_dtype),mask_dtype=str(mask_dtype),
                passed=True,native_error=errors[0])


def decode_case(device,batch,index_dtype,count,padded,graph=False):
    heads,qheads,width,layers=(2,1,16,2) if device=='cpu' else (24,8,128,36)
    p=pool(device,batch,layers,heads,width);p.count.fill_(count)
    mixed=torch.randn(batch,2*qheads*width+heads*width,device=device).bfloat16()
    a=torch.randn(batch,heads,device=device).bfloat16();b=torch.randn_like(a)
    bias=torch.randn(heads,device=device);decay=torch.randn(heads,device=device)
    src=torch.arange(1,batch+1,device=device,dtype=index_dtype);dst=src+batch
    mask=torch.ones(batch,device=device,dtype=torch.bool)
    if padded:src[-1]=dst[-1]=-1;mask[-1]=False
    fb=NS(mamba_track_mask=mask)
    pp=[copy.deepcopy(p),copy.deepcopy(p)]
    backends=[]
    for owner,enabled in zip(pp,(False,True)):
        owner.decode_metadata_fused=enabled
        backends.append(NS(factored=owner,_factored_side_stream=None,_factored_batch_trunc=True,
            _track_mamba_state_decode=lambda *a,**k:None,forward_metadata=NS(mamba_track_indices=dst)))
    def run(backend):
        outputs=[]
        for lid in p.layer_ids:
            layer=NS(layer_id=lid,A_log=decay,dt_bias=bias,num_q_heads=qheads,num_v_heads=heads,
                     head_k_dim=width,head_v_dim=width)
            outputs.append(GDNAttnBackend._forward_decode_factored(backend,layer,fb,mixed,a,b,
                           torch.empty(0,device=device),torch.empty(0,device=device),src))
        return outputs
    outputs=[run(be) for be in backends]
    equal(*pp)
    assert all(torch.equal(a,b) for a,b in zip(*outputs))
    if graph:
        graphs=[];captured=[]
        for be in backends:
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=stream):out=run(be)
            graphs.append(g);captured.append(out)
        # Real decode graph bindings remain fixed while slot/mask values change.
        # No repeated active destination or cross-row source/destination hazards.
        for turn in range(3):
            src.copy_(torch.arange(1,batch+1,device=device,dtype=index_dtype).roll(turn))
            dst.copy_(torch.arange(batch+1,2*batch+1,device=device,dtype=index_dtype))
            mask.fill_(True)
            if turn==1:src[-1]=dst[-1]=-1;mask[-1]=False
            if turn==2:mask[::2]=False
            for g in graphs:g.replay()
            torch.cuda.synchronize()
            equal(*pp)
            assert all(torch.equal(a,b) for a,b in zip(*captured))
    return dict(stage='native_backend',B=batch,index_dtype=str(index_dtype),initial_count=count,
        padded=padded,layers=layers,heads=heads,width=width,graph_replays=3 if graph else 0,
        conv_tracking='unchanged path excluded from this component gate',passed=True)


def compile_gate():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    sig=dict(zip(('mixed_qkv','a_gate','b_gate','A_log','dt_bias','vbar','a_ptr','u_ptr','w_ptr',
        'cnt_ptr','stale_ptr','ssm_state_indices','o','scale','gs_eps'),
        ('*bf16','*bf16','*bf16','*fp32','*fp32','*fp32','*fp32','*fp16','*fp16',
         '*i32','*i32','*i64','*bf16','fp32','fp32')))
    constants=dict(stride_mixed_tok=5120,stride_a_tok=24,stride_b_tok=24,stride_idx=1,
        H=8,HV=24,K=128,V=128,RMAX=16,SOFTPLUS_THRESHOLD=20.,LATE_W_LOAD=False)
    for enabled in (False,True):
        signature=dict(sig);k=dict(constants,INVALIDATE_PREFIX=enabled)
        if enabled:signature['prefix_ptr']='*i32'
        else:k['prefix_ptr']=None
        out=triton.compile(ASTSource(fn=kernels._factored_packed_step_kernel,
            signature=signature,constexprs=k),target=GPUTarget('cuda',103,32),options={'num_warps':1})
        assert out.asm['ptx'] and out.asm['cubin']
    sig=dict(zip(('a_ptr','u_ptr','w_ptr','cnt_ptr','stale_ptr','src_idx','mask_ptr','dst_idx',
                 'stride_a_layer','stride_u_layer','stride_w_layer','stride_c_layer','prefix_ptr'),
                ('*fp32','*fp16','*fp16','*i32','*i32','*i64','*i1','*i64','i64','i64','i64','i64','*i32')))
    out=triton.compile(ASTSource(fn=kernels._factored_track_copy_kernel,signature=sig,
        constexprs=dict(A_ROW=3072,U_ROW=49152,W_ROW=49152,C_ROW=24,BLOCK=1024,INVALIDATE_PREFIX=True)),
        target=GPUTarget('cuda',103,32),options={'num_warps':4})
    assert out.asm['ptx'] and out.asm['cubin']
    print('PFACTOR4_DECODE_METADATA_COMPILE PASS sm103',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--compile-only',action='store_true');ap.add_argument('--output')
    args=ap.parse_args()
    if args.compile_only:compile_gate();return
    device='cpu' if os.environ.get('TRITON_INTERPRET')=='1' else 'cuda'
    if device=='cpu':
        subprocess.run([sys.executable,__file__,'--compile-only'],check=True,
            env=dict(os.environ,TRITON_INTERPRET='0'),timeout=180)
    rows=[]
    for enabled in ('0','1'):
        with patch.dict(os.environ,SGLANG_PFACTOR4_DECODE_METADATA=enabled):
            probe=pool(device,1,2,2,16)
            assert probe.decode_metadata_fused==(enabled=='1')
    del probe
    batches=(1,8) if device=='cpu' else (1,8,32,96)
    for dtype in (torch.int32,torch.int64):
        for batch in batches:
            for count in (8,15):
                for padded in ((False,) if batch==1 else (False,True)):
                    r=decode_case(device,batch,dtype,count,padded,graph=device=='cuda' and count==15)
                    rows.append(r);print('DECODE_METADATA_CASE',json.dumps(r),flush=True)
        for kind in ('copy','alias','negative_src','negative_dst','masked','padding'):
            rows.append(tracking_case(device,8,dtype,kind))
        # Invalid integer masks retain the native exception and mutations.
        rows.append(tracking_case(device,1,dtype,'copy',torch.int32))
    result=dict(passed=all(r['passed'] for r in rows),complete=True,device=device,
        model_dtype='bfloat16',factor_dtype='float16',rows=rows,production_default=False,
        gate='native backend and pool; all state/output bytes exact; GPU live-control graph replay',
        scope='CPU reduced tensor geometry plus true SM103 signature; GPU full served dimensions')
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('PFACTOR4_DECODE_METADATA_GATE',json.dumps(result),flush=True)


if __name__=='__main__':main()
