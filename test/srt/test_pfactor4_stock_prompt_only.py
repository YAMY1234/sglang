"""Stock fairness gate: production host paths, verify live commits and dense math."""
import argparse
import ast
import hashlib
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from test_pfactor4_prompt_only import SRT, FLAG, equal, source_methods, host_case, policy
from test_pfactor4_prompt_only_policy import run_default_policy_checks
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.kernels.ops.attention.fla.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode


def verify(env):
    path=SRT/'speculative/spec_utils.py'; tree=ast.parse(path.read_text())
    for name in ('prepare_mamba_track_for_verify','_verify_commit_step_indices'):
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
        exec('from __future__ import annotations\n'+ast.unparse(node),env)
    rows=[]
    for fullstack in (False,True):
        model=NS(hf_config=NS(twinstar={'fullstack':{'version':3}}))
        b=NS(model_config=model,req_to_token_pool=NS(),mamba_track_indices=torch.tensor([7]),
             mamba_track_mask=True,mamba_track_seqlens=True,mamba_track_buffer_indices=[0])
        with patch.dict(os.environ,**{FLAG:'0' if fullstack else 'all'},
                        TWINSTAR_FULLSTACK='1' if fullstack else '0',
                        SGLANG_EXTERNAL_MODEL_PACKAGE='twinstar_sgl'):
            env['prepare_mamba_track_for_verify'](b)
        assert b.mamba_track_indices is None and b.mamba_track_buffer_indices is None
        for accepted in (1,2,3,4):
            live,track=env['_verify_commit_step_indices'](batch=b,accept_index=torch.arange(4).reshape(1,4),
                        accept_lens=torch.tensor([accepted]),draft_token_num=4)
            assert live.tolist()==[accepted-1] and track is None
        rows.append(dict(fullstack=fullstack,passed=True,live_commit_unchanged=True))
    return dict(passed=True,rows=rows,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def allocator_warning():
    records=[]
    class Sink(logging.Handler):
        def emit(self,record):records.append(record.getMessage())
    log=logging.getLogger('sglang.srt.mem_cache.allocator.mamba');sink=Sink();log.addHandler(sink)
    try:
        p=MambaSlotAllocator(4,'cpu');p.alloc(3);assert not records
        p.alloc(1);assert len(records)==1 and 'free_slots=0 state_slots=4' in records[0]
        assert p.alloc(1) is None and len(records)==1
        p.clear();p.alloc_group_begin(4);assert len(records)==2
        p.alloc(1);p.alloc_group_end();assert p.available_size()==3
    finally:log.removeHandler(sink)
    return dict(passed=True,warning_count=2,direct_and_group_alloc=True,no_tensor_value_read=True)


def dense_numeric(batch,dtype):
    torch.manual_seed(1602+batch)
    slots=torch.arange(1,batch+1);dst=slots+batch
    base=torch.randn(2*batch+1,2,16,16,dtype=torch.float32)*.01
    base[dst]=base[slots]
    inputs=[(torch.randn(batch,64,dtype=dtype),torch.randn(batch,2,dtype=dtype),
             torch.randn(batch,2,dtype=dtype)) for _ in range(8)]
    states=[];outputs=[]
    for value in ('0','all'):
        state=base.clone();outs=[]
        with patch.dict(os.environ,**{FLAG:value},TWINSTAR_FULLSTACK='0'):
            p_only=policy.prompt_only_state_cache(NS(hf_config=NS()),NS())
            for i,(qkv,a,b) in enumerate(inputs):
                out=torch.empty(batch,1,2,16,dtype=dtype)
                fused_recurrent_gated_delta_rule_packed_decode(mixed_qkv=qkv,a=a,b=b,
                    A_log=torch.zeros(2),dt_bias=torch.zeros(2),scale=.25,
                    initial_state=state,out=out,ssm_state_indices=slots,use_qk_l2norm_in_kernel=True)
                outs.append(out.clone())
                if i in (1,5) and not p_only:state[dst]=state[slots]
        states.append(state);outputs.append(outs)
    assert equal(states[0][slots],states[1][slots])
    assert all(equal(x,y) for x,y in zip(*outputs))
    assert equal(states[1][dst],base[dst])
    restored=states[1][dst].clone();assert equal(restored,base[slots])
    return dict(passed=True,B=batch,dtype=str(dtype),steps=8,native_dense_recurrence=True,
                bitwise_live_state=True,bitwise_outputs=True,prefill_restore_bitwise=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    default=run_default_policy_checks();env,sources=source_methods()
    # Same hot-reuse regression as 56cdc, with an ordinary dense stock pool.
    # 8192 detects an accidental P-1 prefill edit; 8193 covers the old fixture.
    host=[host_case(env,b,o,e,stock=True,prompt=p) for b in (1,8) for o in (False,True)
          for e in ('0','all') for p in (8192,8193)]
    fullstack=[host_case(env,1,o,'0',stock=True,fullstack=True) for o in (False,True)]
    numeric=[dense_numeric(b,d) for b in (1,8) for d in (torch.bfloat16,torch.float16)]
    result=dict(passed=True,complete=True,device='cpu',host=host,fullstack=fullstack,
                verify=verify(env),numeric=numeric,allocator=allocator_warning(),
                default_policy=default,sources=sources,
                scope='Identical live-state/input trajectories are bitwise; cache destinations intentionally differ. Full-model hot-hit tokens require the remeasure gate.')
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('PFACTOR4_STOCK_PONLY_GATE',json.dumps(result),flush=True)
if __name__=='__main__':main()
