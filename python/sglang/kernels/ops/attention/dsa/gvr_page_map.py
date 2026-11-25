"""Measurement-only selected logical index -> physical slot map."""
import triton
import triton.language as tl


@triton.jit
def _page_map(raw, table, lengths, output, K:tl.constexpr, TABLE_STRIDE:tl.constexpr,
              PAGE:tl.constexpr, BLOCK:tl.constexpr):
    row=tl.program_id(0)
    col=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    value=tl.load(raw+row*K+col,col<K,other=-1)
    n=tl.load(lengths+row)
    valid=(col<K)&(value>=0)&(value<n)
    page=tl.load(table+row*TABLE_STRIDE+value//PAGE,valid,other=0)
    slot=tl.where(valid,page*PAGE+value%PAGE,-1)
    tl.store(output+row*K+col,slot,col<K)


def page_map(raw,table,lengths,output,block=256,page=64):
    assert raw.ndim==2 and raw.shape==output.shape
    assert raw.is_contiguous() and output.is_contiguous()
    assert table.stride(1)==1 and lengths.is_contiguous()
    _page_map[(raw.shape[0],triton.cdiv(raw.shape[1],block))](
        raw,table,lengths,output,raw.shape[1],table.stride(0),page,block,num_warps=4)
    return output
