"""Diagnostic only: real 36-layer GDN verify, original vs fused four-input chain.

No model throughput claim. CUDA graphs remove Python launch spacing in BOTH
paths, matching production target verify. Common reset cost is measured alone.
"""
import argparse
import copy
import importlib
import json
from pathlib import Path
import statistics
import sys
from types import ModuleType, SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]


def package(name, path):
    m = ModuleType(name); m.__path__ = [str(path)]; sys.modules[name] = m


def run(batch):
    package('bench639_kernels', ROOT/'python/sglang/srt/layers/attention/linear/kernels')
    package('bench639_owners', ROOT/'python/sglang/srt/mem_cache')
    kernels = importlib.import_module('bench639_kernels.gdn_factored')
    Owner = importlib.import_module('bench639_owners.gdn_factored_replay').FactoredGDNReplayState
    sys.modules['sglang.srt.layers.attention.linear.kernels.gdn_factored'] = kernels
    torch.manual_seed(639)
    layers, heads, qheads, key, value, tokens = 36, 24, 8, 128, 128, 4
    width = 2*qheads*key + heads*value
    p = SimpleNamespace(cfg=SimpleNamespace(r=8,m=8,rfull=16,
        kernel_kwargs=lambda:dict(kernel='split',post_order=True)),
        a=torch.randn(layers,batch,heads,key,device='cuda')*.01,
        U=torch.randn(layers,batch,heads,16,key,device='cuda',dtype=torch.float16)*.01,
        W=torch.randn(layers,batch,heads,16,value,device='cuda',dtype=torch.float16)*.01,
        count=(8+torch.arange(batch,device='cuda')%8)[None,:,None].expand(layers,batch,heads).to(torch.int32).clone(),
        stale=torch.ones(batch,device='cuda',dtype=torch.int32),dense_of=None,dense_required=None,prefix_valid=None,
        vbar=torch.randn(layers,heads,value,device='cuda')*.01,layer_index=lambda x:x)
    desc=[SimpleNamespace(layer_id=i,num_q_heads=qheads,num_v_heads=heads,head_k_dim=key,
        head_v_dim=value,A_log=torch.randn(heads,device='cuda'),dt_bias=torch.randn(heads,device='cuda')) for i in range(layers)]
    mixed=torch.randn(layers,batch*tokens,width,device='cuda',dtype=torch.bfloat16)
    ga=torch.randn(layers,batch*tokens,heads,device='cuda',dtype=torch.bfloat16);gb=torch.randn_like(ga)
    results={}; final_states={}; final_outputs={}
    for fused in (False,True):
        tx=Owner(copy.deepcopy(p),batch,4,qkv_width=width,batched_commit=True,
                 verify_window_fused=fused)
        tx.work_indices.copy_(tx.row_ids)
        def reset():
            for name in tx.names:tx.working[name].copy_(getattr(p,name))
        def body():
            reset()
            return [tx.forward_layer(d,mixed[i],ga[i],gb[i]) for i,d in enumerate(desc)]
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):body()
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):outputs=body()
        reset_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(reset_graph,stream=stream):reset()
        torch.cuda.current_stream().wait_stream(stream)
        def measure(g):
            values=[]
            for _ in range(7):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(30):g.replay()
                end.record();end.synchronize();values.append(begin.elapsed_time(end)/30)
            return values
        reset_times=measure(reset_graph);times=measure(graph)
        graph.replay();torch.cuda.synchronize()
        name='fused' if fused else 'original'
        final_states[name]={k:v.clone() for k,v in tx.working.items()}
        final_outputs[name]=[x.clone() for x in outputs]
        results[name]=dict(graph_ms=statistics.median(times),reset_ms=statistics.median(reset_times),
            verify_ms=statistics.median(times)-statistics.median(reset_times),samples_ms=times,
            scope='36 real forward_layer calls incl raw-input storage/output handling; CUDA graph on both; common reset separately measured')
        print(json.dumps(dict(batch=batch,phase=name,result=results[name])),flush=True)
    for name in final_states['original']:
        if not torch.equal(final_states['original'][name].contiguous().view(torch.uint8),
                           final_states['fused'][name].contiguous().view(torch.uint8)):
            raise AssertionError('runtime-shaped state differs: '+name)
    for old,new in zip(final_outputs['original'],final_outputs['fused']):
        if not torch.equal(old.view(torch.uint8),new.view(torch.uint8)):
            raise AssertionError('runtime-shaped verify output differs')
    return dict(batch=batch,layers=layers,heads=heads,key=key,value=value,tokens=tokens,
        count_phase='8 + request_row % 8, same across heads/layers; controlled microbenchmark, not observed production phase distribution',
        bitwise=True,results=results,speedup=results['original']['verify_ms']/results['fused']['verify_ms'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--batches',type=int,nargs='+',default=[16,32,96]);a=parser.parse_args()
    rows=[]
    for b in a.batches:
        rows.append(run(b));a.out.write_text(json.dumps(dict(complete=False,rows=rows),indent=2)+'\n')
    a.out.write_text(json.dumps(dict(complete=True,rows=rows,diagnostic_only=True),indent=2)+'\n')
