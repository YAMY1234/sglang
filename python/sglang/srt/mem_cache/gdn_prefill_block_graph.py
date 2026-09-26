"""Bounded shared GDN prefill graphs; each call owns its returned state/output.

Only singleton 256/8192-token shapes are admitted. Model prefill capture remains
unchanged. Layer weights and input values are rebound before every replay;
intermediate checkpoint h is consumed by the caller before the next call.
"""
from collections import OrderedDict
import os
import torch
import triton
import triton.language as tl


@triton.jit
def _bind_inputs(sources, destinations, COUNTS:tl.constexpr, WIDTHS:tl.constexpr,
                 STRIDES:tl.constexpr, COLS:tl.constexpr, BLOCK:tl.constexpr):
    pid=tl.program_id(0)
    begin=0
    for i in tl.static_range(len(COUNTS)):
        blocks=tl.cdiv(COUNTS[i],BLOCK)
        if pid>=begin and pid<begin+blocks:
            offsets=(pid-begin)*BLOCK+tl.arange(0,BLOCK)
            address=(offsets//WIDTHS[i])*STRIDES[i]+(offsets%WIDTHS[i])*COLS[i]
            values=tl.load(sources[i]+address,mask=offsets<COUNTS[i],other=0)
            tl.store(destinations[i]+offsets,values,mask=offsets<COUNTS[i])
        begin+=blocks


def bind_inputs(buffers,tensors):
    values=tuple(tensors.values());destinations=tuple(buffers.values())
    if values[0].is_cuda or os.environ.get('TRITON_INTERPRET')=='1':
        counts=tuple(x.numel() for x in values)
        widths=tuple(x.shape[-2]*x.shape[-1] if x.ndim==4 else x.shape[-1] for x in values)
        strides=tuple(x.stride(1) if x.ndim==4 else x.stride(0) if x.ndim==2 else x.numel() for x in values)
        cols=tuple(x.stride(-1) for x in values)
        _bind_inputs[(sum(triton.cdiv(n,1024) for n in counts),)](
            values,destinations,counts,widths,strides,cols,1024,num_warps=4)
    else:
        for name,x in tensors.items():buffers[name].copy_(x)


class PrefillBlockGraph:
    def __init__(self):
        self.entries=OrderedDict()
        self.stats=dict(captured=0,replayed=0,fallback=0)

    def run(self, tensors, evaluate):
        key=tuple((name,tuple(x.shape),x.dtype,x.device) for name,x in tensors.items())
        key+=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        if key not in self.entries:
            buffers={name:torch.empty_like(x,memory_format=torch.contiguous_format) for name,x in tensors.items()}
            def bind():
                bind_inputs(buffers,tensors)
            def call():return evaluate(buffers)
            graph=None
            pinned_indices=()
            if tensors['state'].is_cuda:
                # The FLA helpers keep only four cu_seqlens identities. Other
                # requests/layers can evict these allocations while this graph
                # still refers to their addresses. Own them for the graph's
                # lifetime, independently of that helper cache.
                from sglang.kernels.ops.attention.fla.index import (
                    prepare_lens, prepare_chunk_indices, prepare_chunk_offsets,
                )
                cu=buffers['cu']
                bind()
                pinned_indices=(prepare_lens(cu), prepare_chunk_indices(cu,64),
                                prepare_chunk_offsets(cu,64))
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        bind();outputs=call()
                torch.cuda.current_stream().wait_stream(stream)
                bind()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):outputs=call()
                self.stats['captured']+=1
            else:
                bind();outputs=call()
            # evaluate contains no layer-specific tensor references; these
            # are all supplied through buffers on every call.
            self.entries[key]=(buffers,graph,outputs,evaluate,pinned_indices)
            while len(self.entries)>2:self.entries.popitem(last=False)
        buffers,graph,outputs,evaluator,_=self.entries[key]
        self.entries.move_to_end(key)
        bind_inputs(buffers,tensors)
        if graph is None:
            outputs=evaluator(buffers)
        else:
            graph.replay();self.stats['replayed']+=1
        output,last,h=outputs
        # A later layer reuses this graph. Returned output and final state
        # therefore own their storage, including in-place chunk updates.
        state=buffers['state'] if last is None else last
        return output.clone(),state.clone(),h
