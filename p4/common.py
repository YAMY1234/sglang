"""WS-1b fixture distribution and the production Triton sigmoid reference."""
import hashlib
import json
from pathlib import Path
import torch
from sglang.srt.layers.attention.linear.kernels import gdn_factored as gf
from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated

FIELDS = ('fa', 'fu', 'fw', 'fcount', 'stale', 'prefix_valid')

def bitwise(x,y):
    return torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))

def clone(p):
    return {k:v.clone() for k,v in p.items()}

def configure():
    gf.STEP_IMPL_BY_RMAX[32]='cuda'
    gf.STEP_CUDA_BATCHES=frozenset(range(1,9))
    gf.STEP_ROWS_BY_RMAX[32]=0
    gf.STEP_VSPLIT_BY_RMAX[32]=1
    gf.STEP_CUDA_MAX_BATCH=0

def fixture(tp,b,gen,layers=1):
    # Same layouts / QR basis / distributions as bench_decode.factored_pool and
    # make_inputs used by validate_step_b78 / WS-1b. RNG offsets are task-owned.
    h,hv,s=16//tp,48//tp,max(17,b+2)
    q=torch.linalg.qr(torch.randn(hv,128,32,device='cuda',generator=gen))[0]
    p=dict(fa=torch.randn(layers,s,hv,128,device='cuda',generator=gen)*.05,
           fu=q.transpose(-1,-2)[None,None].expand(layers,s,hv,32,128).half().contiguous(),
           fw=(torch.randn(layers,s,hv,32,128,device='cuda',generator=gen)*.1).half(),
           fcount=(16+torch.arange(s,device='cuda',dtype=torch.int32)%16)[None,:,None].expand(layers,s,hv).contiguous(),
           stale=torch.zeros(s,device='cuda',dtype=torch.int32),
           prefix_valid=torch.ones(s,device='cuda',dtype=torch.int32))
    inp=dict(mixed=(torch.randn(b,2*h*128+hv*128,device='cuda',generator=gen)*.5).bfloat16(),
             a=torch.randn(b,hv,device='cuda',generator=gen).bfloat16(),
             b=torch.randn(b,hv,device='cuda',generator=gen).bfloat16(),
             alog=(torch.rand(layers,hv,device='cuda',generator=gen)*2-4).float(),
             bias=torch.randn(layers,hv,device='cuda',generator=gen).bfloat16(),
             vbar=(torch.randn(layers,hv,128,device='cuda',generator=gen)*.1).float())
    slots=torch.arange(1,b+1,device='cuda',dtype=torch.int32)
    z=torch.randn(b,hv,128,device='cuda',generator=gen).bfloat16()
    return p,inp,slots,z

def step(ext,p,i,slots,out,l,tp,z=None,weight=None,eps=1e-6,invalidate=False):
    ext.factored_step_r32(i['mixed'],i['a'],i['b'],i['alog'][l],i['bias'][l],i['vbar'][l],
        p['fa'][l],p['fu'][l],p['fw'][l],p['fcount'][l],p['stale'],slots,out,16//tp,128**-.5,
        gf.GS_EPS,p['prefix_valid'],invalidate,z,weight,eps)

def norm(x,z,weight,eps=1e-6):
    return rms_norm_gated(x=x.reshape(-1,128),weight=weight,bias=None,z=z.reshape(-1,128),
        eps=eps,norm_before_gate=True,is_rms_norm=True,activation='sigmoid').reshape(x.shape)

def weights(model):
    """Read only actual per-layer norm vectors from safetensors (no model load)."""
    from safetensors import safe_open
    model=Path(model)
    index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    names=sorted(n for n in index if n.endswith('linear_attn.norm.weight'))
    if not names:
        raise RuntimeError(f'no linear_attn.norm.weight in {model}')
    result,meta=[],[]
    for name in names:
        with safe_open(str(model/index[name]),framework='pt',device='cpu') as f:
            w=f.get_tensor(name).to(torch.bfloat16).contiguous()
        if w.shape!=(128,):
            raise RuntimeError((name,w.shape))
        meta.append(dict(name=name,sha256=hashlib.sha256(w.view(torch.uint8).numpy().tobytes()).hexdigest()))
        result.append(w.cuda())
    return result,meta

def diff(x,y):
    return dict(bytes_differ=int((x.contiguous().view(torch.uint8)!=y.contiguous().view(torch.uint8)).sum()),
                max_abs=float((x.float()-y.float()).abs().max()),
                indices=torch.nonzero(x!=y)[:8].tolist())
