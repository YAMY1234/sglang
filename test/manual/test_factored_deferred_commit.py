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
if GPU:
    torch.set_default_device('cuda')


def main():
    root=Path(__file__).resolve().parents[2]
    for name,path in [('deferred_kernels','python/sglang/srt/layers/attention/linear/kernels'),
                      ('deferred_owners','python/sglang/srt/mem_cache')]:
        module=ModuleType(name);module.__path__=[str(root/path)];sys.modules[name]=module
    kernel=importlib.import_module('deferred_kernels.gdn_factored')
    io=importlib.import_module('deferred_kernels.gdn_verify_io')
    Owner=importlib.import_module('deferred_owners.gdn_factored_replay').FactoredGDNReplayState
    sys.modules['sglang.srt.layers.attention.linear.kernels.gdn_factored']=kernel
    sys.modules['sglang.srt.layers.attention.linear.kernels.gdn_verify_io']=io
    torch.manual_seed(746)
    layers,capacity,heads,key=2,2,2,128 if GPU else 16
    width=2*key+heads*key
    pool=SimpleNamespace(cfg=SimpleNamespace(r=8,m=8,rfull=16,
            kernel_kwargs=lambda:dict(kernel='split',post_order=True)),
        a=torch.randn(layers,8,heads,key)*.01,
        U=torch.randn(layers,8,heads,32,key,dtype=torch.float16)*.01,
        W=torch.randn(layers,8,heads,32,key,dtype=torch.float16)*.01,
        count=torch.full((layers,8,heads),8,dtype=torch.int32),
        stale=torch.zeros(8,dtype=torch.int32),dense_of=torch.arange(8,dtype=torch.int32),
        dense_required=torch.ones(8,dtype=torch.int32),prefix_valid=torch.ones(8,dtype=torch.int32),
        vbar=torch.randn(layers,heads,key)*.01,layer_index=lambda x:x)
    desc=[SimpleNamespace(layer_id=i,num_q_heads=1,num_v_heads=heads,head_k_dim=key,head_v_dim=key,
        A_log=torch.randn(heads),dt_bias=torch.randn(heads)) for i in range(layers)]
    slots=torch.tensor([2,5]); cases=[]
    audit=tempfile.TemporaryDirectory(prefix='ssmon-cadence-')
    os.environ['SGLANG_GDN_VERIFY_CADENCE_AUDIT']=audit.name
    graph=os.environ.get('REPLAY_TEST_GRAPH')=='1'
    def same(a,b,label):
        if not torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)):
            raise AssertionError(label)
    for phase in range(8):
        current=copy.deepcopy(pool);current.count[:,slots]=8+phase
        owner=Owner(current,capacity,4,qkv_width=width,batched_commit=True,
                    verify_window_fused=True,snapshot_kernel=True,graph_commit=graph)
        accepted_total=phase; cuts=0
        # Repeated mixed prefix lengths cross several accepted-token periods.
        for turn,consumed in enumerate([1,4,2,3,1,1,4,4]):
            before={name:getattr(current,name).clone() for name in owner.names}
            ticket=owner.snapshot_commit(slots)
            mixed=torch.randn(layers,capacity,4,width,dtype=torch.bfloat16)
            ga=torch.randn(layers,capacity,4,heads,dtype=torch.bfloat16);gb=torch.randn_like(ga)
            oracle=copy.deepcopy(current)
            for li,layer in enumerate(desc):
                output=owner.forward_layer(layer,mixed[li].flatten(0,1),ga[li].flatten(0,1),gb[li].flatten(0,1))
                args=dict(owner.layer_arguments[li]);args['truncate']=False
                verify={name:before[name][li].clone() for name in owner.names}
                expected=[]
                for step in range(4):
                    expected.append(kernel.factored_packed_decode(mixed[li,:,step],ga[li,:,step],gb[li,:,step],
                        fa=verify['a'],fu=verify['U'],fw=verify['W'],fcount=verify['count'],stale=oracle.stale,
                        ssm_state_indices=slots,**args)[:,0])
                same(output.reshape(capacity,4,heads,key),torch.stack(expected,dim=1),'append verify output')
                for step in range(consumed):
                    kernel.factored_packed_decode(mixed[li,:,step],ga[li,:,step],gb[li,:,step],
                        fa=oracle.a[li],fu=oracle.U[li],fw=oracle.W[li],fcount=oracle.count[li],
                        stale=oracle.stale,ssm_state_indices=slots,**args)
                    if step == consumed//2:
                        for name in owner.names:
                            getattr(oracle,name)[li,7].copy_(getattr(oracle,name)[li,5])
            kernel.factored_expiry_truncate_layers(oracle.U,oracle.W,oracle.count,slots,8,16,deferred_cut=True)
            kernel.factored_expiry_truncate_layers(oracle.U,oracle.W,oracle.count,torch.tensor([7]),8,16,deferred_cut=True)
            for name in owner.names: same(before[name],getattr(current,name),'verify must not publish '+name)
            owner.commit(ticket,torch.full((capacity,),consumed-1,dtype=torch.int64),
                         track_slots=torch.tensor([-1,7]),track_steps=torch.tensor([-1,consumed//2]))
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
        resources=getattr(kernel,'VERIFY_LAST_RESOURCES',{}),
        frozen_equivalence=False,scope='new-policy sequential/transaction equality and accepted-token W8 cadence; not model quality')))
    audit.cleanup()


if __name__=='__main__': main()
