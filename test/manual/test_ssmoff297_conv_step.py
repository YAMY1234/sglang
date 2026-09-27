"""Compare fused convolution/recurrence against the actual stock convolution."""
import copy
import json
import sys
import torch
from test_ssmoff297_kernels import GPU,pool,kernels,same
from prefill_kernels.gdn_conv_step import publish
from sglang.kernels.ops.mamba import causal_conv1d_triton as conv

if not GPU:
    conv.is_arch_support_pdl=lambda:False


def fixture(layers,count,dtype,has_bias,src,dst,active):
    hv=24 if GPU else 6;k=128 if GPU else 16;h=hv//3;dim=(2*h+hv)*k
    p=pool(layers,hv,k);p.count.fill_(count);p.decode_metadata_fused=True
    state=torch.randn(layers,10,3,dim,dtype=torch.bfloat16).transpose(-1,-2)
    weight=torch.randn(layers,dim,4,dtype=torch.bfloat16)
    bias=torch.randn(layers,dim,dtype=torch.bfloat16) if has_bias else None
    mixed=torch.randn(layers,1,dim,dtype=torch.bfloat16)
    gates=torch.randn(layers,2,1,hv,dtype=torch.bfloat16)
    log=torch.randn(layers,hv);dt=torch.randn(layers,hv,dtype=torch.bfloat16)
    indices=torch.tensor([src],dtype=dtype);destinations=torch.tensor([dst],dtype=dtype)
    mask=torch.tensor([active]);pending=torch.empty(layers,1,dim,3,dtype=state.dtype)
    outputs=[torch.empty(1,1,hv,k,dtype=torch.bfloat16) for _ in range(layers)]
    return dict(p=p,state=state,weight=weight,bias=bias,mixed=mixed,gates=gates,log=log,dt=dt,
                indices=indices,destinations=destinations,mask=mask,pending=pending,outputs=outputs,
                h=h,hv=hv,k=k,layers=layers,src=src,dst=dst,active=active)


def step(f,fused):
    p=f['p']
    for li in range(f['layers']):
        weight=f['weight'][li];bias=None if f['bias'] is None else f['bias'][li]
        x=f['mixed'][li]
        if not fused:
            x=conv.causal_conv1d_update(x,f['state'][li],weight,bias,'silu',
                                       conv_state_indices=f['indices'])
        kernels.factored_packed_decode(x,f['gates'][li,0],f['gates'][li,1],
            A_log=f['log'][li],dt_bias=f['dt'][li],scale=f['k']**-.5,
            vbar=p.vbar[li],fa=p.a[li],fu=p.U[li],fw=p.W[li],fcount=p.count[li],stale=p.stale,
            ssm_state_indices=f['indices'],num_q_heads=f['h'],num_v_heads=f['hv'],
            head_k_dim=f['k'],head_v_dim=f['k'],r=8,rfull=16,truncate=False,kernel='split',
            prefix_valid=p.prefix_valid if li==0 else None,out=f['outputs'][li],
            conv_context=(f['state'][li],weight,bias,f['pending'][li]) if fused else None)
    kernels.factored_expiry_truncate_layers(p.U,p.W,p.count,f['indices'],8,16)
    if fused:
        publish(f['pending'],f['state'],f['indices'],f['mask'],f['destinations'])
    elif f['active'] and f['src']>=0 and f['dst']>=0 and f['src']!=f['dst']:
        f['state'][:,f['dst']].copy_(f['state'][:,f['src']])
    p.track_copy(f['indices'],f['mask'],f['destinations'])


def check(count,dtype,has_bias,src=2,dst=5,active=True,layers=2):
    base=fixture(layers,count,dtype,has_bias,src,dst,active)
    candidate=copy.deepcopy(base)
    # Replays must retain exactly the same window and state trajectory.
    for _ in range(2):
        step(base,False);step(candidate,True)
        for a,b in zip(base['outputs'],candidate['outputs']):same(a,b,'convolved decode output')
        same(base['state'],candidate['state'],'conv window and tracked window')
        for name in ('a','U','W','count','stale','prefix_factored_valid'):
            same(getattr(base['p'],name),getattr(candidate['p'],name),'factored '+name)
    return dict(count=count,dtype=str(dtype),bias=has_bias,src=src,dst=dst,
                active=active,layers=layers,bitwise=True)


def timing(fused):
    torch.manual_seed(297)
    f=fixture(36,8,torch.int64,False,2,5,False)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(16):step(f,fused)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):step(f,fused)
    for _ in range(32):graph.replay()
    pairs=[]
    for _ in range(10):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(256):graph.replay()
        b.record();pairs.append((a,b))
    torch.cuda.synchronize()
    values=[a.elapsed_time(b)/256 for a,b in pairs]
    return dict(fused=fused,mean_ms=sum(values)/len(values),samples_ms=values,
                scope='36-layer B1 convolution and recurrence, expiry and tracking; not model C1')


torch.manual_seed(297)
rows=[]
cases=[(c,d,b,2,5,True,2) for c in range(8,16)
       for d in (torch.int32,torch.int64) for b in (False,True)]
cases += [(15,torch.int64,False,s,d,m,2) for s,d,m in
          [(2,2,True),(-1,5,True),(2,-1,True),(2,5,False)]]
if GPU:cases.append((15,torch.int64,False,2,5,True,36))
for args in cases:
    try:row=check(*args)
    except Exception as error:
        row=dict(arguments=str(args),bitwise=False,error=str(error))
    rows.append(row)
passed=all(row['bitwise'] for row in rows)
result=dict(complete=True,passed=passed,device='CUDA' if GPU else 'CPU',cases=rows)
if passed and GPU:
    result['timings']=[timing(fused) for fused in (False,True,True,False)]
    base=(result['timings'][0]['mean_ms']+result['timings'][3]['mean_ms'])/2
    candidate=(result['timings'][1]['mean_ms']+result['timings'][2]['mean_ms'])/2
    result['timing_comparison']=dict(base_ms=base,candidate_ms=candidate,delta_ms=candidate-base)
print(json.dumps(result),flush=True)
sys.exit(0 if passed else 1)
