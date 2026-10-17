"""WS-1b fixture distribution and the production Triton sigmoid reference."""
import hashlib
import json
from pathlib import Path
import torch
from sglang.srt.layers.attention.linear.kernels import gdn_factored as gf
from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated

from sglang.srt.layers.attention.linear.kernels.gdn_factored_norm_capture import restore_z, z_metadata

FIELDS = ('fa', 'fu', 'fw', 'fcount', 'stale', 'prefix_valid')


def layouts(path=None):
    # These are stress fixtures, NOT observed service metadata. A measured
    # catalog produced from v2 snapshots replaces the primary TP2 layout.
    result = {}
    for tp in (2, 1):
        hv, h = 48 // tp, 16 // tp
        width = (2 * h + 2 * hv) * 128 + 2 * hv
        result[tp] = [
            dict(name='contiguous', stride=[hv*128,128,1], storage_offset=0,
                 source='synthetic contiguous control', measured=False, primary=False),
            dict(name='projection_view', stride=[width,128,1], storage_offset=(2*h+hv)*128,
                 source='source-derived merged projection stress; not measured service layout',
                 measured=False, primary=True),
            dict(name='head_gaps', stride=[hv*384+256,384,1], storage_offset=256,
                 source='synthetic head-stride stress', measured=False, primary=False)]
    if path:
        catalog = json.loads(Path(path).read_text())
        for key, entries in catalog.items():
            tp = int(key.removeprefix('tp'))
            assert tp in result and entries
            for entry in result[tp]: entry['primary'] = False
            assert any(x['primary'] for x in entries)
            for entry in entries:
                assert len(entry['stride']) == 3 and entry['stride'][-1] == 1
                assert entry.get('source') and 'measured' in entry
            result[tp].extend(entries)
    return result


def layout_metadata(layout, tp, b):
    return dict(shape=[b,48//tp,128], stride=layout['stride'],
                storage_offset=layout['storage_offset'])

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

def fixture(tp,b,gen,layers=1,layout=None):
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
    if layout is not None:
        z=restore_z(z,layout_metadata(layout,tp,b))
    return p,inp,slots,z

def step(ext,p,i,slots,out,l,tp,z=None,weight=None,eps=1e-6,invalidate=False):
    args=(i['mixed'],i['a'],i['b'],i['alog'][l],i['bias'][l],i['vbar'][l],
          p['fa'][l],p['fu'][l],p['fw'][l],p['fcount'][l],p['stale'],slots,out,16//tp,128**-.5,
          gf.GS_EPS,p['prefix_valid'],invalidate)
    ext.factored_step_r32(*args, *((z,weight,eps) if weight is not None else ()))

def norm(x,z,weight,eps=1e-6):
    return rms_norm_gated(x=x.reshape(-1,128),weight=weight,bias=None,z=z,
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
