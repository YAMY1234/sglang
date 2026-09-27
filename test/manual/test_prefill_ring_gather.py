"""Bitwise memory-copy and rebinding probe; isolated timings do not admit a model."""
import argparse, importlib, json, os, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--device',choices=('cpu','cuda'),required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
os.environ['REPLAY_TEST_DEVICE']=args.device
import torch
import test_factored_prefill_graph as tx
gather=importlib.import_module('prefill_owners.gdn_prefill_ring_gather').gather_owned_ring_layers
rows=[];torch.manual_seed(298832)
for shape in [(2,3,2,7,5)]+([(36,3,24,128,128)] if tx.GPU else []):
 for dtype in (torch.float32,torch.float16):
  source=torch.randn(*shape,dtype=dtype)
  if dtype==torch.float32:
   bits=source.view(torch.int32).flatten();bits[:5]=torch.tensor([0,-2147483648,2139095040,-8388608,2143289345],dtype=torch.int32)
  before=source.clone()
  for index_dtype in (torch.int32,torch.int64):
   for ids in ([2],[2,0,2],[]):
    index=torch.tensor(ids,dtype=index_dtype)
    for block in (1024,4096):
     result=gather(source,index,block=block);expected=torch.index_select(source,1,index)
     tx.same(result,expected,'raw ring gather bits');tx.same(source,before,'source immutable')
     assert result.is_contiguous()
     rows.append(dict(shape=shape,dtype=str(dtype),index_dtype=str(index_dtype),indices=ids,block=block,bitwise=True))
  transposed=source.transpose(-1,-2);index=torch.tensor([2,0])
  tx.same(gather(transposed,index),torch.index_select(transposed,1,index),'strided fallback')
  try:gather(source,torch.arange(3),out=source)
  except ValueError:pass
  else:raise AssertionError('alias output accepted')
result=dict(complete=True,passed=True,device=args.device,rows=rows,scope='Primitive memory copies only; no model, numerical or TTFT admission.')
if tx.GPU:
 source=torch.randn(36,3,24,128,128);index=torch.tensor([2]);output=torch.empty(36,1,24,128,128)
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):gather(source,index,out=output)
 torch.cuda.current_stream().wait_stream(stream)
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph,stream=stream):gather(source,index,out=output)
 for slot in (0,2,1):
  index.fill_(slot);graph.replay();torch.cuda.synchronize();tx.same(output,source[:,slot:slot+1],'captured live index')
 result['capture_rebindings']=3
 timings={}
 for name,read in [('torch',lambda:torch.index_select(source,1,index,out=output)),('flat1024',lambda:gather(source,index,out=output,block=1024)),('flat4096',lambda:gather(source,index,out=output,block=4096))]:
  for _ in range(5):read()
  samples=[]
  for _ in range(3):
   torch.cuda.synchronize();start=time.perf_counter();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
   for _ in range(50):read()
   end.record();end.synchronize();samples.append(dict(event_ms=begin.elapsed_time(end)/50,wall_ms=(time.perf_counter()-start)*1000/50))
  timings[name]=samples
 result['fixed_output_timings']=timings
args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
