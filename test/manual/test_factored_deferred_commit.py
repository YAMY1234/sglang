"""Real-kernel deferred-cut transaction safety and accepted-token cadence.

This compares against a sequential oracle with the NEW commit-cut policy.
It does not claim equality to the frozen W8 candidate-output trajectory.
"""
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace

GPU = os.environ.get('REPLAY_TEST_DEVICE') == 'cuda'
if not GPU and (os.environ.get('TRITON_INTERPRET') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
    raise RuntimeError('CPU interpreter with CUDA hidden required')
os.environ['SGLANG_GDN_VERIFY_DEFER_CUT'] = '1'
os.environ['SGLANG_GDN_VERIFY_DIAGNOSTICS'] = '1'
import torch
import triton
import triton.language as tl

@triton.jit
def _bf16_cast_probe(x,y,B:tl.constexpr):
    i=tl.arange(0,B)
    v=tl.load(x+i)
    tl.store(y+i,v.to(tl.bfloat16).to(tl.float32))

if GPU:
    torch.set_default_device('cuda')


def main():
    root=Path(__file__).resolve().parents[2]
    for name,path in [('deferred_kernels','python/sglang/srt/layers/attention/linear/kernels'),
                      ('deferred_owners','python/sglang/srt/mem_cache')]:
        module=ModuleType(name);module.__path__=[str(root/path)];sys.modules[name]=module
    kernel=importlib.import_module('deferred_kernels.gdn_factored')
    io=importlib.import_module('deferred_kernels.gdn_verify_io')
    commit_kernel=importlib.import_module('deferred_kernels.gdn_commit_window')
    meta_kernel=importlib.import_module('deferred_kernels.gdn_verify_meta')
    Owner=importlib.import_module('deferred_owners.gdn_factored_replay').FactoredGDNReplayState
    sys.modules['sglang.srt.layers.attention.linear.kernels.gdn_factored']=kernel
    sys.modules['sglang.srt.layers.attention.linear.kernels.gdn_verify_io']=io
    sys.modules['sglang.srt.layers.attention.linear.kernels.gdn_commit_window']=commit_kernel
    sys.modules['sglang.srt.layers.attention.linear.kernels.gdn_verify_meta']=meta_kernel
    torch.manual_seed(746)
    layers,capacity,heads,key=2,2,24 if GPU else 2,128 if GPU else 16
    qheads=4 if GPU else 1
    width=2*qheads*key+heads*key
    pool=SimpleNamespace(cfg=SimpleNamespace(r=8,m=8,rfull=16,
            kernel_kwargs=lambda:dict(kernel='split',post_order=True)),
        a=torch.randn(layers,8,heads,key)*.01,
        U=torch.randn(layers,8,heads,32,key,dtype=torch.float16)*.01,
        W=torch.randn(layers,8,heads,32,key,dtype=torch.float16)*.01,
        count=torch.full((layers,8,heads),8,dtype=torch.int32),
        stale=torch.zeros(8,dtype=torch.int32),dense_of=torch.arange(8,dtype=torch.int32),
        dense_required=torch.ones(8,dtype=torch.int32),prefix_valid=torch.ones(8,dtype=torch.int32),
        vbar=torch.randn(layers,heads,key)*.01,layer_index=lambda x:x)
    desc=[SimpleNamespace(layer_id=i,num_q_heads=qheads,num_v_heads=heads,head_k_dim=key,head_v_dim=key,
        A_log=torch.randn(heads),dt_bias=torch.randn(heads)) for i in range(layers)]
    slots=torch.tensor([2,5]); cases=[]
    audit=tempfile.TemporaryDirectory(prefix='ssmon-cadence-')
    os.environ['SGLANG_GDN_VERIFY_CADENCE_AUDIT']=audit.name
    graph=os.environ.get('REPLAY_TEST_GRAPH')=='1'
    raw_append=os.environ.get('SGLANG_GDN_VERIFY_APPEND_RAW')=='1'
    dense_errors=[]
    bf16_cast_mode='not-probed'
    if raw_append:
        probe_x=torch.tensor([.3944,.537,.42,.4455],dtype=torch.float32)
        probe_y=torch.empty_like(probe_x)
        _bf16_cast_probe[(1,)](probe_x,probe_y,4)
        trunc=(probe_x.view(torch.int32)&-65536).view(torch.float32)
        rounded=probe_x.to(torch.bfloat16).float()
        if torch.equal(probe_y,rounded):bf16_cast_mode='round-to-nearest'
        elif torch.equal(probe_y,trunc):bf16_cast_mode='truncate-low-16-bits'
        else:raise AssertionError('unrecognized BF16 cast semantics')
        if GPU and bf16_cast_mode!='round-to-nearest':
            raise AssertionError('CUDA BF16 cast changed')
    def same(a,b,label):
        if not torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)):
            raise AssertionError(label+' '+json.dumps(dict(
                different=int((a!=b).sum().item()),max_abs=float((a.float()-b.float()).abs().max().item()),
                resources=getattr(kernel,'VERIFY_LAST_RESOURCES',{}))))
    meta_negative_cases = 0
    if not GPU and os.environ.get('SGLANG_GDN_VERIFY_META_FUSED') == '1':
        probe=Owner(copy.deepcopy(pool),capacity,4,qkv_width=width,batched_commit=True,
                    verify_window_fused=True,snapshot_kernel=True,graph_commit=graph)
        def rejects(call):
            nonlocal meta_negative_cases
            try: call()
            except RuntimeError: meta_negative_cases+=1
            else: raise AssertionError('invalid transaction metadata was accepted')
        for bad_slots in (torch.tensor([2,2]),torch.tensor([-1,5]),torch.tensor([2,8])):
            rejects(lambda:probe.snapshot_commit(bad_slots))
        ticket=probe.snapshot_commit(slots)
        probe.written.fill_(True)
        rejects(lambda:probe._validate(ticket,torch.tensor([0,4])))
        probe._validate(ticket,torch.tensor([0,3]))
        probe.written.zero_()
        rejects(lambda:probe._validate(ticket,torch.tensor([0,3])))
        probe.written.fill_(True)
        probe.invalidate_slots(slots[:1])
        rejects(lambda:probe._validate(ticket,torch.tensor([0,3])))
        probe.rollback(ticket)
        for name in probe.names: same(getattr(probe.pool,name),getattr(pool,name),'metadata rejection must not publish')
    for phase in range(8):
        current=copy.deepcopy(pool);current.count[:,slots]=8+phase
        owner=Owner(current,capacity,4,qkv_width=width,batched_commit=True,
                    verify_window_fused=True,snapshot_kernel=True,graph_commit=graph)
        accepted_total=phase; cuts=0
        # Repeated mixed prefix lengths cross several accepted-token periods.
        for turn,consumed in enumerate([1,4,2,3,1,1,4,4]):
            tracked=not owner.commit_fused or turn%2==1
            before={name:getattr(current,name).clone() for name in owner.names}
            ticket=owner.snapshot_commit(slots)
            mixed=torch.randn(layers,capacity,4,width,dtype=torch.bfloat16)
            ga=torch.randn(layers,capacity,4,heads,dtype=torch.bfloat16);gb=torch.randn_like(ga)
            oracle=copy.deepcopy(current)
            for li,layer in enumerate(desc):
                output=owner.forward_layer(layer,mixed[li].flatten(0,1),ga[li].flatten(0,1),gb[li].flatten(0,1))
                for name, value in (('mixed',mixed),('a',ga),('b',gb)):
                    same(owner.inputs[name][li],value[li],'recorded raw input '+name)
                args=dict(owner.layer_arguments[li]);args['truncate']=False
                verify={name:before[name][li].clone() for name in owner.names}
                expected=[]
                for step in range(4):
                    expected.append(kernel.factored_packed_decode(mixed[li,:,step],ga[li,:,step],gb[li,:,step],
                        fa=verify['a'],fu=verify['U'],fw=verify['W'],fcount=verify['count'],stale=oracle.stale,
                        ssm_state_indices=slots,raw_append=raw_append,**args)[:,0])
                same(output.reshape(capacity,4,heads,key),torch.stack(expected,dim=1),'append verify output')
                if raw_append and turn == 0:
                    # Independent dense FP64 recurrence checks the actual
                    # first raw-append output, including BF16 gate rounding.
                    U=before['U'][li,slots].double();W=before['W'][li,slots].double()
                    mask=torch.arange(32)[None,None,:] < before['count'][li,slots,:,None]
                    state=(U*mask[...,None]).transpose(-1,-2) @ W
                    m=mixed[li,:,0].double()
                    q=m[:,:qheads*key].reshape(capacity,qheads,key).repeat_interleave(heads//qheads,dim=1)
                    k=m[:,qheads*key:2*qheads*key].reshape(capacity,qheads,key).repeat_interleave(heads//qheads,dim=1)
                    v=m[:,2*qheads*key:].reshape(capacity,heads,key)
                    q=q/torch.sqrt((q*q).sum(-1,keepdim=True)+1e-6)*args['scale']
                    k=k/torch.sqrt((k*k).sum(-1,keepdim=True)+1e-6)
                    x=ga[li,:,0].double()+layer.dt_bias.double()
                    soft=torch.where(x<=20,torch.log1p(torch.exp(x)),x)
                    decay=torch.exp(-torch.exp(layer.A_log.double())*soft)
                    beta32=torch.sigmoid(gb[li,:,0].double()).float()
                    if bf16_cast_mode=='truncate-low-16-bits':
                        # This image's interpreter masks low bits; the CUDA
                        # probe must instead confirm round-to-nearest. Keep
                        # the independent oracle on the measured cast rule.
                        beta=(beta32.view(torch.int32)&-65536).view(torch.float32).double()
                    else:
                        beta=beta32.to(gb.dtype).double()
                    sink=before['a'][li,slots].double();vb=current.vbar[li].double()
                    sink=decay[...,None]*(sink-beta[...,None]*k*(k*sink).sum(-1,keepdim=True))+beta[...,None]*k
                    residual=beta[...,None]*((v-vb)-decay[...,None]*(k[...,None,:]@state).squeeze(-2))
                    dense=decay[...,None,None]*state+k[..., :,None]*residual[...,None,:]
                    ref=(q[...,None,:]@dense).squeeze(-2)+(sink*q).sum(-1,keepdim=True)*vb
                    actual=output.reshape(capacity,4,heads,key)[:,0].double()
                    error=float((actual-ref).abs().max().item());dense_errors.append(error)
                    torch.testing.assert_close(actual,ref,rtol=.012,atol=3e-6)
                for step in range(consumed):
                    kernel.factored_packed_decode(mixed[li,:,step],ga[li,:,step],gb[li,:,step],
                        fa=oracle.a[li],fu=oracle.U[li],fw=oracle.W[li],fcount=oracle.count[li],
                        stale=oracle.stale,ssm_state_indices=slots,**args)
                    if tracked and step == consumed//2:
                        for name in owner.names:
                            getattr(oracle,name)[li,7].copy_(getattr(oracle,name)[li,5])
            kernel.factored_expiry_truncate_layers(oracle.U,oracle.W,oracle.count,slots,8,16,deferred_cut=True)
            if tracked:
                kernel.factored_expiry_truncate_layers(oracle.U,oracle.W,oracle.count,torch.tensor([7]),8,16,deferred_cut=True)
            for name in owner.names: same(before[name],getattr(current,name),'verify must not publish '+name)
            tracking=dict(track_slots=torch.tensor([-1,7]),track_steps=torch.tensor([-1,consumed//2])) if tracked else {}
            owner.commit(ticket,torch.full((capacity,),consumed-1,dtype=torch.int64),**tracking)
            for name in owner.names: same(getattr(oracle,name),getattr(current,name),'committed '+name)
            cuts+=(accepted_total%8+consumed)//8
            accepted_total+=consumed
            assert torch.all(current.count[:,slots]==8+accepted_total%8)
            assert cuts==accepted_total//8
            cases.append(dict(start_phase=phase,turn=turn,consumed=consumed,total=accepted_total,
                              cuts=cuts,count=8+accepted_total%8))
        ticket=owner.snapshot_commit(slots)
        before={name:getattr(current,name).clone() for name in owner.names}
        owner.forward_layer(desc[0],mixed[0].flatten(0,1),ga[0].flatten(0,1),gb[0].flatten(0,1))
        owner.rollback(ticket)
        for name in owner.names: same(before[name],getattr(current,name),'zero-consumption rollback')
    records=[json.loads(row) for row in (Path(audit.name)/'rank0.jsonl').read_text().splitlines()]
    assert len(records)==len(cases)
    assert all(sum(row['cuts'])==2*(((case['total']-case['consumed'])%8+case['consumed'])//8)
               for row,case in zip(records,cases))
    print(json.dumps(dict(complete=True,device='CUDA' if GPU else 'CPU',cases=cases,
        graph_commit=graph,cadence_records=len(records),
        bf16_cast_mode=bf16_cast_mode,meta_fused=owner.meta_fused,meta_negative_cases=meta_negative_cases,raw_append=raw_append,dense_oracle_max_abs=max(dense_errors,default=None),
        record_fused=owner.record_fused,
        commit_fused=owner.commit_fused,query_heads=qheads,value_heads=heads,
        resources=getattr(kernel,'VERIFY_LAST_RESOURCES',{}),
        frozen_equivalence=False,scope='new-policy sequential/transaction equality and accepted-token W8 cadence; not model quality')))
    audit.cleanup()


if __name__=='__main__': main()
