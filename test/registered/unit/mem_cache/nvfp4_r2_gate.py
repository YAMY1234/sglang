"""Actual-kernel byte gates for R2-c/r16 MGS grouping and R2-d/fp64 Jacobi.

Run in the pinned image with TRITON_INTERPRET=1 for CPU; repeat on CUDA
before serving. Original admitted eigensolver is loaded from a frozen file.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch


def same_bytes(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def expiry(device):
    from sglang.srt.layers.attention.linear.kernels import gdn_factored as k
    from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNConfig
    from sglang.srt.model_executor.fullstack_policy import factored_batch_layers_enabled
    cfg=FactoredGDNConfig(r=16,m=16,init_method='k31',strict_chunk=1)
    with patch.dict(os.environ,SGLANG_GDN_FACTORED_BATCH_MGS16='0'):
        assert not factored_batch_layers_enabled(cfg)
    with patch.dict(os.environ,SGLANG_GDN_FACTORED_BATCH_MGS16='1'):
        assert factored_batch_layers_enabled(cfg)
    rows=[]
    for rank in (8,16):
        layers,heads=(3,2) if device=='cpu' else (36,24)
        rmax=2*rank;slots=5
        for phase in ('none','due','mixed'):
            torch.manual_seed(0x1544)
            u=torch.randn(layers,slots,heads,rmax,128,device=device)
            w=torch.randn_like(u)
            count=torch.full((layers,slots,heads),rank,dtype=torch.int32,device=device)
            if phase=='due':count.fill_(rmax)
            if phase=='mixed':
                count.flatten()[::2]=rmax
                count.flatten()[1::3]=rmax-1
                count[-1,3,0]=rmax+1
            index=torch.tensor([3,99,-1,99,1,99],device=device,dtype=torch.int32)[::2]
            original=[x.clone() for x in (u,w,count)]
            expected=[x.clone() for x in original]
            for layer in range(layers):
                k.factored_expiry_truncate(*[x[layer] for x in expected],index,rank,rmax)
            with patch.dict(os.environ,SGLANG_GDN_FACTORED_BATCH_MGS16=str(int(rank==16))):
                k.factored_expiry_truncate_layers(u,w,count,index,rank,rmax)
            actual=(u,w,count)
            assert all(same_bytes(a,b) for a,b in zip(expected,actual)), (rank,phase)
            assert all(same_bytes(a[:,[0,2,4]],b[:,[0,2,4]]) for a,b in zip(original,actual))
            rows.append(dict(rank=rank,phase=phase,layers=layers,heads=heads,
                full_pool_bytes_equal=True,untouched_slots_equal=True,index_stride=2))
    return rows


def eigensolver(device, original_file):
    from sglang.srt.layers.attention.linear.kernels import gdn_k31_eigh as k
    from sglang.srt.layers.attention.linear.kernels import gdn_prefill_reference as ref
    from sglang.srt.duet.state_factor import small_eigh, pad_below_spectrum
    spec=importlib.util.spec_from_file_location('nvfp4_original_eigh',original_file)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    rows=[]
    batch=2 if device=='cpu' else 864
    for kind in ('full','rank4','zero','repeated','indefinite'):
        torch.manual_seed(0x1545)
        x=torch.randn(batch,24,24,device=device,dtype=torch.float64)
        if kind=='rank4':x[...,4:]=0
        g=x@x.transpose(-1,-2)
        if kind=='zero':g.zero_()
        if kind=='repeated':g=torch.eye(24,device=device,dtype=torch.float64).expand(batch,-1,-1).contiguous()
        if kind=='indefinite':g-=12*torch.eye(24,device=device,dtype=torch.float64)
        padded=pad_below_spectrum(g)
        lower=(g.diagonal(dim1=-2,dim2=-1)-(g.abs().sum(-1)-g.diagonal(dim1=-2,dim2=-1).abs())).amin(-1)
        assert bool((padded[...,24:,24:].diagonal(dim1=-2,dim2=-1)<lower[...,None]).all())
        baseline=old.eigh(g)
        for enabled in (False,True):
            with patch.dict(os.environ,SGLANG_GDN_K31_EIGH_EARLY_EXIT=str(int(enabled))):
                result=small_eigh(g,override='jacobi')
            equal=[same_bytes(a,b) for a,b in zip(baseline,result)]
            scale=g.abs().amax().clamp_min(1.)
            residual=float(((g@result[1]-result[1]*result[0][...,None,:]).abs().amax()/scale).item())
            orth=float((result[1].transpose(-1,-2)@result[1]-torch.eye(24,device=device)).abs().amax())
            assert all(equal) and residual<1e-10 and orth<1e-10,(kind,enabled,equal,residual,orth)
            rows.append(dict(kind=kind,early_exit=enabled,matrices=batch,
                d_z_bytes_equal=equal,relative_residual=residual,orth_error=orth,padding_below_spectrum=True))
    # Verify the retained rank-16 consumer, not only eigenvalues/projectors.
    for kind in ('full','rank4','zero'):
        heads=2 if device=='cpu' else 24
        torch.manual_seed(0x1545)
        s=torch.randn(1,heads,128,128,device=device)
        if kind=='rank4':s=s[...,:4]@s[...,:4,:]
        if kind=='zero':s.zero_()
        v=torch.randn(heads,128,device=device)
        omega=torch.randn(1,heads,128,24,device=device)
        with patch.object(k,'eigh',old.eigh),patch.object(ref,'K31_EIGH','jacobi'):
            baseline=ref.factorize_prefill_k31(s,v,16,32,torch.float32,omega)
        with patch.dict(os.environ,SGLANG_GDN_K31_EIGH_EARLY_EXIT='1'),patch.object(ref,'K31_EIGH','jacobi'):
            result=ref.factorize_prefill_k31(s,v,16,32,torch.float32,omega)
        equal=[same_bytes(a,b) for a,b in zip(baseline,result)]
        assert all(equal),(kind,equal)
        rows.append(dict(kind=kind,full_factor_bytes_equal=equal,heads=heads,rank=16))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',choices=('c','d'),required=True)
    ap.add_argument('--original-eigh',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();device='cpu' if os.environ.get('TRITON_INTERPRET')=='1' else 'cuda'
    rows=expiry(device) if args.candidate=='c' else eigensolver(device,args.original_eigh)
    result=dict(passed=True,device=device,candidate=args.candidate,rows=rows)
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
