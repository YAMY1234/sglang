"""Directive 453: raw-input replay semantics using real interpreted r8/W8 kernels.

This is a CPU design experiment, not an integrated serving implementation.
TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES='' python this_file
"""
import ast
import copy
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

if os.environ.get('TRITON_INTERPRET') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('requires CPU Triton interpretation with CUDA hidden')
import torch

ROOT=Path(__file__).resolve().parents[2]
KERNELS=ROOT/'python/sglang/srt/layers/attention/linear/kernels'
spec=importlib.util.spec_from_file_location('copy_helpers',Path(__file__).with_name('test_factored_spec_copy_cpu.py'))
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
same=helper.same
package=ModuleType('replay_kernels');package.__path__=[str(KERNELS)];sys.modules[package.__name__]=package
kernels=importlib.import_module('replay_kernels.gdn_factored')
sys.modules['sglang.srt.layers.attention.linear.kernels.gdn_factored']=kernels
Transaction=helper.load('replay_transaction',ROOT/'python/sglang/srt/mem_cache/gdn_factored_spec.py').FactoredGDNVerifyState
backend_path=ROOT/'python/sglang/srt/layers/attention/linear/gdn_backend.py'
tree=ast.parse(backend_path.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='GDNAttnBackend')
method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_forward_verify_factored')
scope={};exec(compile(ast.Module(body=[method],type_ignores=[]),str(backend_path),'exec'),scope);verify=scope[method.name]


def main():
    torch.manual_seed(453)
    layers,capacity,batch,heads,key,value=2,3,2,2,16,16
    base=SimpleNamespace(cfg=SimpleNamespace(r=8,m=8,rfull=16,kernel_kwargs=lambda:dict(kernel='split',post_order=True)),
        a=torch.randn(layers,8,heads,key)*.01,
        U=torch.randn(layers,8,heads,16,key,dtype=torch.float16)*.01,
        W=torch.randn(layers,8,heads,16,value,dtype=torch.float16)*.01,
        count=torch.full((layers,8,heads),8,dtype=torch.int32),
        stale=torch.zeros(8,dtype=torch.int32),dense_of=torch.arange(8,dtype=torch.int32),
        dense_required=torch.ones(8,dtype=torch.int32),prefix_valid=torch.ones(8,dtype=torch.int32),
        vbar=torch.randn(layers,heads,value)*.01,layer_index=lambda layer:layer)
    base.count[:,2].fill_(14);base.count[:,5].fill_(15)
    direct=copy.deepcopy(base);replay=copy.deepcopy(base)
    direct.spec_state=Transaction(direct,capacity,4,direct_checkpoints=True)
    desc=[SimpleNamespace(layer_id=i,num_q_heads=1,num_v_heads=heads,head_k_dim=key,head_v_dim=value,
                         A_log=torch.randn(heads),dt_bias=torch.randn(heads)) for i in range(layers)]
    slots=torch.tensor([2,5]);head=torch.randn(heads*value,31);cases=[]
    def step(pool,li,mixed,a,b,indices):
        d=desc[li]
        return kernels.factored_packed_decode(mixed,a,b,A_log=d.A_log,dt_bias=d.dt_bias,
            scale=key**-.5,vbar=pool.vbar[li],fa=pool.a[li],fu=pool.U[li],fw=pool.W[li],
            fcount=pool.count[li],stale=pool.stale,ssm_state_indices=indices,
            num_q_heads=1,num_v_heads=heads,head_k_dim=key,head_v_dim=value,r=8,rfull=16,
            truncate=True,kernel='split',post_order=True)
    for iteration in range(21):
        mixed=torch.randn(layers,capacity,4,2*key+heads*value,dtype=torch.bfloat16)
        a=torch.randn(layers,capacity,4,heads,dtype=torch.bfloat16);b=torch.randn_like(a)
        # Own the raw window; modifying/reusing the forward's input cannot alter commit.
        raw=(mixed.clone(),a.clone(),b.clone())
        before={n:getattr(replay,n).clone() for n in Transaction.names}
        working=copy.deepcopy(replay);ticket=direct.spec_state.snapshot_commit(slots)
        for li,d in enumerate(desc):
            out=verify(SimpleNamespace(factored=direct,topk=1),d,mixed[li].flatten(0,1),a[li].flatten(0,1),b[li].flatten(0,1)).view(capacity,4,heads,value)
            indices=torch.tensor([2,5,-1])
            for k in range(4):
                candidate=step(working,li,mixed[li,:,k],a[li,:,k],b[li,:,k],indices)
                same(out[:batch,k],candidate[:batch,0],f'verify output {iteration}/{li}/{k}')
                same(out[:batch,k].float().flatten(1)@head,candidate[:batch,0].float().flatten(1)@head,'CPU readout logits')
                for n in Transaction.names:
                    same(getattr(working,n)[li,slots],direct.spec_state.checkpoints[n][li,:batch,k],f'candidate {n}')
        for n in Transaction.names:same(getattr(replay,n),before[n],'persistent untouched before replay')
        selected=torch.tensor([iteration%4,(iteration+2)%4] if iteration<4 else [0,0])
        # Replay only consumed target inputs. selected=0 still consumes one real
        # input, while a pure abort below consumes none and leaves state untouched.
        for k in range(int(selected.max())+1):
            indices=torch.where(selected>=k,slots,-1)
            for li in range(layers):
                step(replay,li,raw[0][li,:batch,k],raw[1][li,:batch,k],raw[2][li,:batch,k],indices)
                # Track an accepted interior checkpoint before subsequent replay.
                if k==0:
                    for n in Transaction.names:getattr(replay,n)[li,7].copy_(getattr(replay,n)[li,5])
        for dst in (slots,torch.tensor([7])):
            replay.stale[dst]=1;replay.dense_of[dst]=-1;replay.dense_required[dst]=0;replay.prefix_valid[dst]=0
        direct.spec_state.commit(ticket,selected,track_slots=torch.tensor([-1,7]),track_steps=torch.tensor([-1,0]))
        for n in (*Transaction.names,'stale','dense_of','dense_required','prefix_valid'):
            same(getattr(direct,n),getattr(replay,n),f'commit {n} round {iteration}')
        cases.append(dict(round=iteration,last_consumed_indices=selected.tolist(),bitwise=True))
    ticket=direct.spec_state.snapshot_commit(slots)
    before={n:getattr(replay,n).clone() for n in Transaction.names}
    direct.spec_state.rollback(ticket)
    for n in Transaction.names:same(getattr(direct,n),before[n],'abort')
    print(json.dumps(dict(passed=True,device='CPU',triton_interpret=True,factor_dtype='float16',
        layers=layers,head_dim=key,cases=cases,w8_inside_window=True,consecutive_zero_drafts=17,
        heterogeneous_prefix_lengths=True,interior_track_state=True,padded_rows=True,persistent_untouched_until_commit=True,
        input_window_owned=True,candidate_outputs_and_cpu_readout_logits_bitwise=True,
        production_replay_integrated=False,full_model_gpu_logits_validated=False)))

if __name__=='__main__':main()
