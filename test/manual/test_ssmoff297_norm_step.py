"""Exact decode plus RMS/output gate, using the served normalization kernel."""
import copy
import json
import sys
import torch
import triton
from test_ssmoff297_kernels import GPU,pool,kernels,same
from sglang.kernels.ops.attention.fla.layernorm_gated import _layer_norm_fwd_1pass_kernel


def reference(x,z,weight,rows,activation):
    value=x.reshape(-1,x.shape[-1]);gate=z.reshape_as(value)
    output=torch.empty_like(value);m,n=value.shape
    rstd=torch.empty(m,dtype=torch.float32)
    _layer_norm_fwd_1pass_kernel[(triton.cdiv(m,rows),1)](
        value,output,weight,None,gate,None,rstd,n,n,n,0,0,m,n,1e-6,
        BLOCK_N=n,ROWS_PER_BLOCK=rows,HAS_BIAS=False,HAS_Z=True,Z_IS_3D=False,
        Z_HEADS=1,NORM_BEFORE_GATE=True,IS_RMS_NORM=True,ACTIVATION=activation,
        USE_GDC=GPU,**({'launch_pdl':True} if GPU else {}),num_warps=1)
    return output.reshape_as(x)


def fixture(layers,count,dtype,slot):
    hv=24 if GPU else 6;k=128 if GPU else 16;h=hv//3
    p=pool(layers,hv,k);p.count.fill_(count);p.decode_metadata_fused=True
    x=torch.randn(layers,1,(2*h+hv)*k,dtype=torch.bfloat16)
    gates=torch.randn(layers,2,1,hv,dtype=torch.bfloat16)
    z=torch.randn(layers,1,hv,k,dtype=torch.bfloat16)
    weight=torch.randn(layers,k,dtype=torch.bfloat16)
    log=torch.randn(layers,hv);bias=torch.randn(layers,hv,dtype=torch.bfloat16)
    slots=torch.tensor([slot],dtype=dtype)
    return dict(p=p,x=x,gates=gates,z=z,weight=weight,log=log,bias=bias,slots=slots,
                output=[torch.empty(1,1,hv,k,dtype=torch.bfloat16) for _ in range(layers)],
                h=h,hv=hv,k=k,layers=layers)


def step(f,fused,rows,activation):
    p=f['p'];result=[]
    for li in range(f['layers']):
        out=kernels.factored_packed_decode(f['x'][li],f['gates'][li,0],f['gates'][li,1],
            A_log=f['log'][li],dt_bias=f['bias'][li],scale=f['k']**-.5,vbar=p.vbar[li],
            fa=p.a[li],fu=p.U[li],fw=p.W[li],fcount=p.count[li],stale=p.stale,
            ssm_state_indices=f['slots'],num_q_heads=f['h'],num_v_heads=f['hv'],
            head_k_dim=f['k'],head_v_dim=f['k'],r=8,rfull=16,truncate=False,kernel='split',
            out=f['output'][li],prefix_valid=p.prefix_valid if li==0 else None,
            norm_context=(f['z'][li],f['weight'][li],1e-6,rows,activation) if fused else None)
        result.append(out if fused else reference(out,f['z'][li],f['weight'][li],rows,activation))
    kernels.factored_expiry_truncate_layers(p.U,p.W,p.count,f['slots'],8,16)
    return result


def check(count,dtype,rows,activation,slot=2):
    a=fixture(2,count,dtype,slot);b=copy.deepcopy(a)
    for _ in range(2):
        x=step(a,False,rows,activation);y=step(b,True,rows,activation)
        for xx,yy in zip(x,y):same(xx,yy,'normalized output')
        for name in ('a','U','W','count','stale','prefix_factored_valid'):
            same(getattr(a['p'],name),getattr(b['p'],name),'norm fusion '+name)
    return dict(count=count,dtype=str(dtype),rows=rows,activation=activation,slot=slot,bitwise=True)


def timing(fused):
    torch.manual_seed(297);f=fixture(36,8,torch.int64,2)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(16):step(f,fused,1,'sigmoid')
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):step(f,fused,1,'sigmoid')
    for _ in range(32):graph.replay()
    pairs=[]
    for _ in range(10):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(256):graph.replay()
        b.record();pairs.append((a,b))
    torch.cuda.synchronize();values=[a.elapsed_time(b)/256 for a,b in pairs]
    return dict(fused=fused,mean_ms=sum(values)/len(values),samples_ms=values,
                scope='36-layer B1 recurrence, actual RMS/sigmoid output gate and expiry; not model C1')


torch.manual_seed(297)
cases=[(count,dtype,rows,activation,2) for count in range(8,16)
       for dtype in (torch.int32,torch.int64) for rows in (1,4)
       for activation in ('sigmoid','silu')]
cases += [(15,torch.int64,rows,activation,-1) for rows in (1,4) for activation in ('sigmoid','silu')]
records=[]
for args in cases:
    try:record=check(*args)
    except Exception as error:record=dict(arguments=str(args),bitwise=False,error=str(error))
    records.append(record)
passed=all(row['bitwise'] for row in records)
result=dict(complete=True,passed=passed,device='CUDA' if GPU else 'CPU',cases=records)
if passed and GPU:
    result['timings']=[timing(fused) for fused in (False,True,True,False)]
    base=(result['timings'][0]['mean_ms']+result['timings'][3]['mean_ms'])/2
    candidate=(result['timings'][1]['mean_ms']+result['timings'][2]['mean_ms'])/2
    result['timing_comparison']=dict(base_ms=base,candidate_ms=candidate,delta_ms=candidate-base)
print(json.dumps(result),flush=True)
sys.exit(0 if passed else 1)
