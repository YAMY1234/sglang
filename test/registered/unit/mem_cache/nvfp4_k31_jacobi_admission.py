"""Acceptance gate: same-threshold early exit, padding and retained rank-16 state.

CPU: TRITON_INTERPRET=1 PYTHONPATH=python python -B test/registered/unit/mem_cache/nvfp4_k31_jacobi_admission.py
CUDA: PYTHONPATH=python python -B test/registered/unit/mem_cache/nvfp4_k31_jacobi_admission.py
The CUDA test checks 864 matrices per spectrum and 24-head factor consumers.
"""
import json
import os
from unittest.mock import patch
import torch

def same_bytes(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def eigensolver(device):
    from sglang.srt.layers.attention.linear.kernels import gdn_k31_eigh as k
    from sglang.srt.layers.attention.linear.kernels import gdn_prefill_reference as ref
    from sglang.srt.duet.state_factor import small_eigh, pad_below_spectrum
    # The fixed twelve-sweep kernel is retained unchanged as the control.
    fixed_eigh=k.eigh
    old=type('FixedControl',(),{'eigh':staticmethod(lambda matrix: fixed_eigh(matrix,early_exit=False))})
    # Actual adapter dispatch: default enabled, explicit zero remains control.
    for value, expected in ((None, True), ('0', False), ('1', True)):
        env=dict(os.environ)
        env.pop('SGLANG_GDN_K31_EIGH_EARLY_EXIT',None)
        if value is not None:env['SGLANG_GDN_K31_EIGH_EARLY_EXIT']=value
        seen=[]
        def solver(matrix, **kwargs):
            seen.append(kwargs['early_exit']);return torch.linalg.eigh(matrix)
        with patch.dict(os.environ,env,clear=True),patch.object(k,'eigh',solver),patch.object(ref,'K31_EIGH','jacobi'):
            ref._small_eigh_fp64(torch.eye(24,device=device)[None])
        assert seen==[expected],(value,seen)
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
        with patch.object(k,'eigh',lambda matrix, **kwargs: old.eigh(matrix)),patch.object(ref,'K31_EIGH','jacobi'):
            baseline=ref.factorize_prefill_k31(s,v,16,32,torch.float32,omega)
        with patch.dict(os.environ,SGLANG_GDN_K31_EIGH_EARLY_EXIT='1'),patch.object(ref,'K31_EIGH','jacobi'):
            result=ref.factorize_prefill_k31(s,v,16,32,torch.float32,omega)
        equal=[same_bytes(a,b) for a,b in zip(baseline,result)]
        assert all(equal),(kind,equal)
        rows.append(dict(kind=kind,full_factor_bytes_equal=equal,heads=heads,rank=16))
    return rows


if __name__=='__main__':
    device='cpu' if os.environ.get('TRITON_INTERPRET')=='1' else 'cuda'
    print(json.dumps(dict(passed=True,device=device,rows=eigensolver(device)),indent=2))
