"""Bounded shared GDN prefill graphs; each call owns its returned state/output.

Default production admits singleton 256/8192-token shapes. Model prefill capture remains
unchanged. Layer weights and input values are rebound before every replay;
intermediate checkpoint h is consumed by the caller before the next call.
The opt-in PREFILL_BLOCK_BUCKETS backend path admits singleton lengths up to
32768. Its singleton cu_seqlens must be [0, actual_tokens].
"""
from collections import OrderedDict
import os
import torch
import triton
import triton.language as tl


def check_result(actual, reference, reference_state):
    """Admission only: compare all bytes, including signed zeros and NaNs."""
    final = reference_state if reference[1] is None else reference[1]
    checked = {}
    for name, x, y in zip(('output', 'state', 'checkpoint'), actual,
                          (reference[0], final, reference[2])):
        checked[name] = (x is None and y is None) if x is None or y is None else (
            x.shape == y.shape and x.dtype == y.dtype and
            torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)))
    if not all(checked.values()):
        raise RuntimeError('prefill block graph differs from eager: ' + str(checked))
    return checked


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


@triton.jit
def _bind_inputs_dynamic(sources, destinations, actual_counts,
                         LIMITS:tl.constexpr, WIDTHS:tl.constexpr,
                         STRIDES:tl.constexpr, COLS:tl.constexpr, BLOCK:tl.constexpr):
    pid=tl.program_id(0)
    begin=0
    for i in tl.static_range(len(LIMITS)):
        blocks=tl.cdiv(LIMITS[i],BLOCK)
        if pid>=begin and pid<begin+blocks:
            offsets=(pid-begin)*BLOCK+tl.arange(0,BLOCK)
            address=(offsets//WIDTHS[i])*STRIDES[i]+(offsets%WIDTHS[i])*COLS[i]
            values=tl.load(sources[i]+address,mask=offsets<actual_counts[i],other=0)
            tl.store(destinations[i]+offsets,values,mask=offsets<actual_counts[i])
        begin+=blocks


def bind_inputs(buffers,tensors):
    values=tuple(tensors.values());destinations=tuple(buffers.values())
    if values[0].is_cuda or os.environ.get('TRITON_INTERPRET')=='1':
        counts=tuple(x.numel() for x in values)
        widths=tuple(x.shape[-2]*x.shape[-1] if x.ndim==4 else x.shape[-1] for x in values)
        strides=tuple(x.stride(1) if x.ndim==4 else x.stride(0) if x.ndim==2 else x.numel() for x in values)
        cols=tuple(x.stride(-1) for x in values)
        if os.environ.get('SGLANG_GDN_PREFILL_DYNAMIC_BIND','0') == '1':
            limits=tuple(x.numel() for x in destinations)
            return _bind_inputs_dynamic[(sum(triton.cdiv(n,1024) for n in limits),)](
                values,destinations,counts,limits,widths,strides,cols,1024,num_warps=4)
        return _bind_inputs[(sum(triton.cdiv(n,1024) for n in counts),)](
            values,destinations,counts,widths,strides,cols,1024,num_warps=4)
    else:
        for name,x in tensors.items():buffers[name].copy_(x)


def pin_chunk_indices(cu, tokens):
    """Own every cached allocation referenced by the captured FLA kernels."""
    from sglang.kernels.ops.attention.fla.index import (
        prepare_lens, prepare_chunk_indices, prepare_chunk_offsets,
    )
    sizes = sorted({64, min(64, max(16, triton.next_power_of_2(tokens)))})
    return (prepare_lens(cu), prepare_chunk_offsets(cu, 64),
            *(prepare_chunk_indices(cu, size) for size in sizes))


class PrefillBlockGraph:
    def __init__(self, *, bucketed=False):
        self.entries=OrderedDict()
        self.stats=dict(captured=0,replayed=0,fallback=0)
        self.bucketed=bucketed

    def _bind(self,buffers,tensors):
        targets=buffers
        if self.bucketed and not tensors['state'].is_cuda:
            n=tensors['q'].shape[1]
            targets={name:(x[:,:n] if name in ('q','k','v') else
                           x[:n] if name in ('a','b') else x)
                     for name,x in buffers.items()}
        bind_inputs(targets,tensors)

    def run(self, tensors, evaluate):
        tokens=capacity=None
        if self.bucketed:
            if tensors['q'].ndim!=4 or tensors['q'].shape[0]!=1 or tensors['cu'].numel()!=2:
                raise ValueError('bucket graph requires one flattened sequence')
            tokens=int(tensors['q'].shape[1])
            if not 1<=tokens<=32768:
                raise ValueError('bucket graph token count outside 1..32768')
            # Match chunk_o BT=min(64,max(16,next_power_of_2(T))).
            # Padding T<=32 up to 64 changes its dot tile and rounding.
            capacity=max(16,1<<(tokens-1).bit_length())
        def shape(name,x):
            sizes=list(x.shape)
            if self.bucketed and name in ('q','k','v'):sizes[1]=capacity
            if self.bucketed and name in ('a','b'):sizes[0]=capacity
            return tuple(sizes)
        key=tuple((name,shape(name,x),x.dtype,x.device) for name,x in tensors.items())
        key+=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        if key not in self.entries:
            buffers={name:(torch.zeros(shape(name,x),dtype=x.dtype,device=x.device)
                          if self.bucketed else torch.empty_like(x,memory_format=torch.contiguous_format))
                     for name,x in tensors.items()}
            def bind(*,capture=False):
                self._bind(buffers,tensors)
                if capture and self.bucketed:
                    # Capture a superset of chunk indices. Kernels use the
                    # rebound device cu_seqlens to mask/iterate actual tokens.
                    # B=1 means the only starting chunk offset is always zero.
                    buffers['cu'][1].fill_(capacity)
            def call():return evaluate(buffers)
            graph=None
            pinned_indices=()
            if tensors['state'].is_cuda:
                # The FLA helpers keep only four cu_seqlens identities. Other
                # requests/layers can evict these allocations while this graph
                # still refers to their addresses. Own them for the graph's
                # lifetime, independently of that helper cache.
                cu=buffers['cu']
                bind(capture=True)
                pinned_indices=pin_chunk_indices(cu, buffers['q'].shape[1])
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        bind(capture=True);outputs=call()
                torch.cuda.current_stream().wait_stream(stream)
                bind(capture=True)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):outputs=call()
                self.stats['captured']+=1
            else:
                bind();outputs=call()
            # evaluate contains no layer-specific tensor references; these
            # are all supplied through buffers on every call.
            self.entries[key]=(buffers,graph,outputs,evaluate,pinned_indices)
            while len(self.entries)>(12 if self.bucketed else 2):self.entries.popitem(last=False)
        buffers,graph,outputs,evaluator,_=self.entries[key]
        self.entries.move_to_end(key)
        self._bind(buffers,tensors)
        if graph is None:
            outputs=evaluator(buffers)
        else:
            graph.replay();self.stats['replayed']+=1
        output,last,h=outputs
        # A later layer reuses this graph. Returned output and final state
        # therefore own their storage, including in-place chunk updates.
        state=buffers['state'] if last is None else last
        if self.bucketed:
            output=output[:,:tokens]
            if h is not None:h=h[:,:triton.cdiv(tokens,64)]
        return output.clone(),state.clone(),h
