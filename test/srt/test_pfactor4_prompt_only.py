"""M11: real scheduler-source replay and native factor live-state bitwise gate.

CPU uses the Triton interpreter; only CPU pin-memory transport is substituted.
Checkpoint destinations intentionally differ. Live factors/output must not.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from test_pfactor4_prompt_only_policy import run_default_policy_checks
from sglang.srt.model_executor import fullstack_policy as policy
from sglang.srt.mem_cache.gdn_factored_pool import FactoredGDNConfig, FactoredGDNPool
from sglang.srt.layers.attention.linear.kernels.gdn_factored import factored_packed_decode

SRT = Path(policy.__file__).parents[1]
FLAG = 'SGLANG_GDN_PROMPT_ONLY_STATE_CACHE'


def equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def source_methods():
    env = dict(torch=torch, mamba_track_grid=lambda page: 256,
               mamba_cache_chunk_size=lambda: 64, mamba_checkpoint_grid=lambda page: 256,
               _MambaRadixCacheV2TrackEntry=lambda **kw: NS(**kw),
               get_exec=lambda: NS(mamba=NS(enable_mamba_extra_buffer=True,
                                           enable_mamba_extra_buffer_lazy=False)))
    sources = {}
    for path, names in {
        'managers/schedule_batch.py': ['set_mamba_track_indices_from_reqs', '_mamba_radix_cache_v2_req_prepare_for_extend'],
        'managers/scheduler_components/batch_result_processor.py': ['_mamba_prefix_cache_update', '_mamba_check_track_boundary'],
        'mem_cache/unified_cache/components/mamba_component.py': ['prepare_for_caching_req'],
    }.items():
        text = (SRT/path).read_text();tree = ast.parse(text)
        sources[path] = hashlib.sha256(text.encode()).hexdigest()
        for name in names:
            node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
            exec('from __future__ import annotations\n'+ast.unparse(node), env)
        if path.endswith('schedule_batch.py'):
            node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'prepare_for_decode')
            block = next(n for n in node.body if isinstance(n, ast.If)
                         and ast.unparse(n.test) == 'get_exec().mamba.enable_mamba_extra_buffer')
            fn = ast.FunctionDef(name='decode_tracking', args=ast.arguments(posonlyargs=[],
                args=[ast.arg(arg='self')], kwonlyargs=[], kw_defaults=[], defaults=[]),
                body=[block], decorator_list=[])
            exec('from __future__ import annotations\n'+ast.unparse(ast.fix_missing_locations(fn)), env)
    return env, sources


def guards():
    cfg = NS(hf_config=NS())
    rows = []
    with patch.dict(os.environ, TWINSTAR_FULLSTACK='0', SGLANG_EXTERNAL_MODEL_PACKAGE=''):
        for enabled in ('0', '1'):
            for strict, factor, exact in ((1,1,0),(1,0,1),(1,0,0),(0,1,0)):
                p = NS(factored_gdn_pool=NS(cfg=NS(strict_chunk=strict,factored_prefix=factor,exact_prefix=exact)))
                with patch.dict(os.environ, **{FLAG:enabled}):
                    policy.initialize_checkpoint_policy(cfg,p)
                    result = policy.prompt_only_state_cache(cfg,p)
                assert result == (enabled=='1' and strict==1 and bool(factor or exact))
                rows.append(dict(enabled=enabled,strict=strict,factor=factor,exact=exact,result=result,passed=True))
            with patch.dict(os.environ, **{FLAG:enabled}):
                p=NS();policy.initialize_checkpoint_policy(cfg,p)
                assert not policy.prompt_only_state_cache(cfg,p)
        with patch.dict(os.environ, **{FLAG:'bad'}):
            try:policy.initialize_checkpoint_policy(cfg,NS())
            except ValueError:pass
            else:raise AssertionError('bad opt-in accepted')
    with patch.dict(os.environ,TWINSTAR_FULLSTACK='1',SGLANG_EXTERNAL_MODEL_PACKAGE='twinstar_sgl',**{FLAG:'0'}):
        model=NS(hf_config=NS(twinstar={'fullstack':{'version':3}}));p=NS()
        policy.initialize_checkpoint_policy(model,p)
        assert policy.prompt_only_state_cache(model,p)
    return dict(passed=True,rows=rows,stock_unchanged=True,fullstack_flag_off_unchanged=True)


def host_case(env, batch_size, overlap, enabled, *, stock=False, fullstack=False, prompt=8193):
    p = NS(factored_gdn_pool=NS(cfg=NS(strict_chunk=1,factored_prefix=1,exact_prefix=0)),
           get_mamba_ping_pong_other_idx=lambda i:1-i if overlap else i,
           get_mamba_ping_pong_keep_idx=lambda req:req.kv.mamba_last_track_idx)
    if stock: p.factored_gdn_pool = None
    reqs = [NS(origin_input_ids=range(prompt),prefix_indices=[],mamba_branching_seqlen=None,
                extend_range=NS(length=prompt,end=prompt),decode_batch_idx=0,
                kv=NS(mamba_ping_pong_track_buffer=torch.tensor([10+2*i,11+2*i]),
                      mamba_next_track_idx=0,mamba_last_track_idx=1,
                      mamba_last_track_seqlen=None,kv_committed_len=prompt)) for i in range(batch_size)]
    p.req_index_to_mamba_ping_pong_track_buffer_mapping=torch.stack([r.kv.mamba_ping_pong_track_buffer for r in reqs])
    b=NS(req_to_token_pool=p,reqs=reqs,model_config=NS(hf_config=NS(),hf_text_config=NS(mamba_chunk_size=64)),
         tree_cache=NS(page_size=64),device='cpu',req_pool_indices=torch.arange(batch_size),
         enable_overlap=overlap,spec_algorithm=NS(is_none=lambda:True))
    if fullstack: b.model_config.hf_config.twinstar = {'fullstack': {'version':3}}
    scheduler=NS(tree_cache=b.tree_cache)
    scheduler._mamba_check_track_boundary=lambda *args:env['_mamba_check_track_boundary'](scheduler,*args)
    tensor=torch.tensor
    def unpinned(*args,**kw):kw.pop('pin_memory',None);return tensor(*args,**kw)
    masks=[]
    with patch.dict(os.environ,TWINSTAR_FULLSTACK='1' if fullstack else '0',SGLANG_EXTERNAL_MODEL_PACKAGE='twinstar_sgl' if fullstack else '',**{FLAG:str(enabled)}), \
         patch.object(torch,'tensor',unpinned),patch.object(torch.Tensor,'pin_memory',lambda self,*a,**k:self):
        policy.initialize_checkpoint_policy(b.model_config, p)
        p_only = policy.prompt_only_state_cache(b.model_config, p)
        prefill_depth = ((prompt - int(policy.prefill_prompt_only_state_cache(b.model_config,p)))//256)*256
        for req in reqs:
            entry=env['_mamba_radix_cache_v2_req_prepare_for_extend'](b,req)
            assert entry.track_mask and req.kv.mamba_last_track_seqlen==prefill_depth
        for position in range(prompt+1,9217):
            b.seq_lens_cpu=torch.full((batch_size,),position,dtype=torch.long)
            for req in reqs:req.decode_batch_idx+=1;req.kv.kv_committed_len=position
            env['decode_tracking'](b)
            masks.append(sum(b.mamba_track_mask_cpu))
            for i,req in enumerate(reqs):env['_mamba_prefix_cache_update'](scheduler,req,b,NS(),i)
        component=NS(cache=NS(enable_mamba_extra_buffer=True,req_to_token_pool=p),int8_ckpt_pool=None)
        for req in reqs:
            params=NS();length=env['prepare_for_caching_req'](component,req,params,9217,True)
            assert length==(prefill_depth if p_only else 9216)
            assert params.mamba_value.item()==req.kv.mamba_ping_pong_track_buffer[req.kv.mamba_last_track_idx].item()
        assert sum(masks)==(0 if p_only else batch_size*4)
    return dict(passed=True,B=batch_size,overlap=overlap,enabled=enabled,
                stock=stock,fullstack=fullstack,prompt=prompt,prefill_depth=prefill_depth,decode_track_copies=sum(masks),checkpoint_depth=length,native_source_replay=True,
                output_arithmetic_called=False)


def numerical(batch, dtype):
    device='cpu' if os.environ.get('TRITON_INTERPRET')=='1' else 'cuda'
    torch.manual_seed(1597+batch)
    base=FactoredGDNPool(size=64,cache_params=NS(shape=NS(temporal=(2,16,16))),
        mamba_layer_ids=[0,1],device=device,
        cfg=FactoredGDNConfig(dtype=torch.float16,strict_chunk=1,factored_prefix=1,ring=16))
    src=torch.arange(1,batch+1,device=device);dst=src+batch;restore=src+2*batch
    base.a.normal_(0,.01);base.U.normal_(0,.01);base.W.normal_(0,.01)
    base.count.fill_(8);base.prefix_factored_valid[src]=1;base.copy_slots(src,dst)
    qkv=[torch.randn(batch,64,device=device,dtype=dtype) for _ in range(8)]
    aa=[torch.randn(batch,2,device=device,dtype=dtype) for _ in range(8)]
    bb=[torch.randn(batch,2,device=device,dtype=dtype) for _ in range(8)]
    alog=torch.zeros(2,device=device);bias=torch.zeros(2,device=device)
    outputs=[];states=[]
    for enabled in (0,1):
        out=[]
        with patch.dict(os.environ,**{FLAG:str(enabled)}):
            p=FactoredGDNPool(size=64,cache_params=NS(shape=NS(temporal=(2,16,16))),
                mamba_layer_ids=[0,1],device=device,cfg=base.cfg)
            # Match the live/checkpoint starting tensors while retaining each
            # freshly constructed pool's immutable flag selection.
            for field, value in vars(base).items():
                if isinstance(value, torch.Tensor):
                    getattr(p, field).copy_(value)
            mask=torch.full((batch,),not policy.generic_prompt_only_state_cache(NS(factored_gdn_pool=p)),device=device,dtype=torch.bool)
            for step in range(8):
                for layer in range(2):
                    out.append(factored_packed_decode(qkv[step],aa[step],bb[step],A_log=alog,dt_bias=bias,
                        scale=.25,vbar=p.vbar[layer],fa=p.a[layer],fu=p.U[layer],fw=p.W[layer],
                        fcount=p.count[layer],stale=p.stale,ssm_state_indices=src,num_q_heads=1,num_v_heads=2,
                        head_k_dim=16,head_v_dim=16,r=8,rfull=16,kernel='split',post_order=True).clone())
                if step in (1,5):p.track_copy(src,mask,dst)
        outputs.append(out);states.append(p)
    assert all(equal(a,b) for a,b in zip(*outputs))
    for field in ('a','U','W','count'):
        assert equal(getattr(states[0],field)[:,src],getattr(states[1],field)[:,src]),field
        assert equal(getattr(base,field)[:,dst],getattr(states[1],field)[:,dst]),field
    assert torch.all(states[0].prefix_valid[dst]==0) and torch.all(states[1].prefix_valid[dst]==1)
    states[1].copy_slots(dst,restore)
    for field in ('a','U','W','count'):
        assert equal(getattr(base,field)[:,dst],getattr(states[1],field)[:,restore]),field
    return dict(passed=True,B=batch,dtype=str(dtype),factor_dtype='float16',steps=8,layers=2,
                bitwise_live_state=True,bitwise_outputs=True,prefill_checkpoint_preserved=True,
                checkpoint_restore_bitwise=True,original_decode_checkpoint_invalidated=True,
                native_recurrence=True,native_track_copy=True,device=device)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    default_policy = run_default_policy_checks()
    env,sources=source_methods();guard=guards()
    host=[host_case(env,b,o,e) for b in (1,8) for o in (False,True) for e in (0,1)]
    numeric=[numerical(b,d) for b in (1,8) for d in (torch.bfloat16,torch.float16)]
    result=dict(passed=True,complete=True,factor_dtype='float16',guards=guard,host=host,numeric=numeric,
                default_policy=default_policy,
                sources=sources,scope='Real scheduler source and native recurrence/track-copy. Output bitwise for identical live-state/input trajectory; changed cache hit paths are separately validated, not claimed to produce identical token sequences across arbitrary requests.')
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('PFACTOR4_M11_GATE',json.dumps(result),flush=True)


if __name__=='__main__':main()
