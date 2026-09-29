"""CPU-only regression for PP DSA transfers, using exact production AST functions.

Run: python python/sglang/test/test_pp129.py -v
No serving-stack/CUDA imports, local paths, or output files are required.
Synthetic DMA buffers use the captured 78 target layers plus the MTP slot.
"""
from pathlib import Path
from collections import deque
import ast
import types
import json
import datetime
import unittest


def run_regression_checks():
    B = Path(__file__).resolve().parents[1]
    ns={'deque':deque,'group_concurrent_contiguous':lambda a,b:([a],[b])}
    for f,names in [(B/'srt/disaggregation/mooncake/conn.py',['_globalize_dsa_layout','_validate_elided_dsa_layout','_send_kvcache_generic']),(B/'srt/disaggregation/utils.py',['build_transfer_entry_pairs'])]:
     tree=ast.parse(f.read_text());functions=[]
     for name in names:
      fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name);fn.decorator_list=[];functions.append(fn)
     module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+functions,type_ignores=[]);exec(compile(ast.fix_missing_locations(module),str(f),'exec'),ns)
    globalize=ns['_globalize_dsa_layout'];validate=ns['_validate_elided_dsa_layout'];types_=['full', 'full', 'full', 'shared', 'shared', 'shared', 'full', 'shared', 'shared', 'shared', 'full', 'shared', 'shared', 'shared', 'full', 'shared', 'shared', 'shared', 'full', 'shared', 'shared', 'shared', 'full', 'shared', 'shared', 'shared', 'full', 'shared', 'shared', 'shared', 'full', 'shared', 'shared', 'shared', 'full', 'shared', 'shared', 'shared', 'full', 'shared', 'shared', 'shared', 'full', 'shared', 'shared', 'shared', 'full', 'shared', 'shared', 'shared', 'full', 'shared', 'shared', 'shared', 'full', 'shared', 'shared', 'shared', 'full', 'shared', 'shared', 'shared', 'full', 'shared', 'shared', 'shared', 'full', 'shared', 'shared', 'shared', 'full', 'shared', 'shared', 'shared', 'full', 'shared', 'shared', 'shared']+['full'];width=len(types_)
    class Future:
     def __init__(self,v):self.v=v
     def result(self):return self.v
    class ImmediateExecutor:
     def submit(self,f,*args):return Future(f(*args))
    rows=[];checks=0
    for part in [[20,20,20,18],[22,20,20,16]]:
     for elided_dst in [True,False]:
      dest_lens=[2112 if t!='shared' or not elided_dst else 0 for t in types_];dest_ptrs=[1000000*(i+1) if n else 0 for i,n in enumerate(dest_lens)];all_owned=[];start=0
      for rank,layers in enumerate(part):
       end=start+layers+(rank==3);local_lens=[2112 if t!='shared' else 0 for t in types_[start:end]];local_ptrs=[100000000+1000000*i if n else 0 for i,n in enumerate(local_lens)];sp,sl,dp,dl=globalize(local_ptrs,local_lens,dest_ptrs,dest_lens,start);validate(sp,sl,dp,dl);checks+=1
       assert len(sp)==len(sl)==len(dp)==len(dl)==79
       assert all(sl[i]==0 and sp[i]==0 for i in range(width) if i<start or i>=end)
       owned=[i for i,n in enumerate(sl) if n];all_owned.extend(owned)
       for custom in [False,True]:
        blocks=[]
        def transfer(session,entries):blocks.extend(entries);return 0
        ctx=types.SimpleNamespace(is_mla_backend=True,is_hybrid_mla_backend=False,pp_size=4,enable_custom_mem_pool=custom,max_transfer_batch_indices=0,_transfer_data=transfer,_await_transfer_futures=lambda fs:max((f.result() for f in fs),default=0))
        rc=ns['_send_kvcache_generic'](ctx,'test-session',sp,dp,sl,[2,3],[5,6],ImmediateExecutor(),src_layer_ids=list(range(width)),dst_layer_ids=list(range(width)))
        expected=[(sp[i]+2*sl[i],dp[i]+5*sl[i],2*sl[i]) for i in owned];assert rc==0 and sorted(blocks)==sorted(expected),(part,rank,blocks,expected);checks+=1
       bad=list(dl);bad[owned[0]]+=1
       try:validate(sp,sl,dp,bad)
       except ValueError:checks+=1
       else:raise AssertionError('same-layer unequal nonzero lengths accepted')
       bad=list(dl);bad[owned[0]]=0
       try:validate(sp,sl,dp,bad)
       except ValueError:checks+=1
       else:raise AssertionError('nonzero source / zero destination accepted')
       bad=list(sp);bad[owned[0]]=0
       try:validate(bad,sl,dp,dl)
       except ValueError:checks+=1
       else:raise AssertionError('positive source length with null pointer accepted')
       rows.append({'partition':part,'rank':rank,'global_start':start,'global_end_with_MTP':end,'peer_elided':elided_dst,'owned_nonzero_global_ids':owned,'global_descriptor_entries':len(sp),'custom_and_batch_DMA_address_checks':'PASS','same_layer_length_mismatch':'REJECT','nonzero_src_zero_dst':'REJECT'})
       start+=layers
      expected=[i for i,t in enumerate(types_) if t!='shared'];assert sorted(all_owned)==expected and len(set(all_owned))==len(all_owned);checks+=1
    # Preserve PP1 and reject malformed description lengths and out-of-range stage spans.
    full_lens=[2112]*width;full_ptrs=[1000000*(i+1) for i in range(width)];validate(full_ptrs,full_lens,full_ptrs,full_lens);checks+=1
    for args in [([1],[1],[],[],0),([1,2],[1],full_ptrs,full_lens,0),([1],[1],full_ptrs,full_lens,79),([1],[1],full_ptrs,full_lens,-1)]:
     try:globalize(*args)
     except ValueError:checks+=1
     else:raise AssertionError('invalid stage description accepted')
    report={'collected':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks_passed':checks,'method':'Exact patched AST functions and existing build_transfer_entry_pairs; real GLM indexer map, synthetic addresses/pages, immediate fake executor and DMA collector. No GPU or full serving-library imports. Both per-layer custom pool and batched DMA paths checked.','case_b_direction':'source elided / destination full passes; source nonzero / destination elided rejects per lead #129 point 2.','rows':rows}
    print(f'PASS {checks} checks: 16 stage cases, 32 exact generic DMA plans, disjoint global coverage including MTP, required negative cases.')
    # Exercise the actual caller as well, so the global view and IDs must reach DMA.
    tree=ast.parse((B/'srt/disaggregation/mooncake/conn.py').read_text());fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='maybe_send_extra');fn.decorator_list=[]
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[]);exec(compile(ast.fix_missing_locations(mod),'actual maybe_send_extra','exec'),ns)
    ns['StateType']=types.SimpleNamespace(**{k:k for k in ['DSA','DSA_TAIL','MAMBA','MINIMAX_INDEX_K','SWA','SWA_RING','DSV4_REQUEST_STATE','BLOCK_SCALE','BLOCK_SCALE_SWA']});ns['np']=types.SimpleNamespace(array=lambda x,dtype=None:x,int32=int)
    integration=0
    for part in [[20,20,20,18],[22,20,20,16]]:
     for elided_dst in [True,False]:
      dest_lens=[2112 if t!='shared' or not elided_dst else 0 for t in types_];dest_ptrs=[1000000*(i+1) if n else 0 for i,n in enumerate(dest_lens)];start=0
      for rank,n in enumerate(part):
       end=start+n+(rank==3);local_lens=[2112 if t!='shared' else 0 for t in types_[start:end]];local_ptrs=[100000000+1000000*i if n else 0 for i,n in enumerate(local_lens)];blocks=[]
       def transfer(session,entries):blocks.extend(entries);return 0
       args=types.SimpleNamespace(state_types=['DSA'],state_data_ptrs=[local_ptrs],state_item_lens=[local_lens],state_dim_per_tensor=[[]],state_layer_ids=[[]],prefill_start_layer=start)
       ctx=types.SimpleNamespace(kv_args=args,pp_size=4,is_mla_backend=True,is_hybrid_mla_backend=False,enable_custom_mem_pool=False,max_transfer_batch_indices=0,_transfer_data=transfer,_is_generic_kvcache_state_type=lambda st:True,_requires_exact_state_index_match=lambda st:False,_globalize_dsa_layout=globalize,_validate_elided_dsa_layout=validate)
       ctx._send_kvcache_generic=types.MethodType(ns['_send_kvcache_generic'],ctx)
       peer=types.SimpleNamespace(dst_state_data_ptrs=[dest_ptrs],dst_state_item_lens=[dest_lens],dst_state_dim_per_tensor=[[]],dst_state_layer_ids=[[]]);req=types.SimpleNamespace(dst_state_indices=[[5,6]],mooncake_session_id='test')
       rc=ns['maybe_send_extra'](ctx,req,[[2,3]],ImmediateExecutor(),peer)
       expected=[(local_ptrs[i]+2*size,dest_ptrs[start+i]+5*size,2*size) for i,size in enumerate(local_lens) if size];assert rc==0 and sorted(blocks)==sorted(expected),(rank,blocks,expected);integration+=1;start+=n
    report['integration_maybe_send_extra_to_DMA']=integration;report['checks_passed']+=integration;return report


class TestPPDSAGlobalLayerTransfer(unittest.TestCase):
    def test_exact_production_transfer_functions(self):
        report = run_regression_checks()
        self.assertEqual(report['checks_passed'], 121)
        self.assertEqual(report['integration_maybe_send_extra_to_DMA'], 16)


if __name__ == '__main__':
    unittest.main()
