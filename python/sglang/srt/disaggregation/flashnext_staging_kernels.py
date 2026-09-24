"""Byte-preserving gather/scatter for per-field physical row maps.

One launch handles KV, compressed indices, factors, conv and boundary metadata.
Local descriptors contain addresses; the logical wire manifest never does.
"""
from dataclasses import dataclass
import math

import torch
import triton
import triton.language as tl

from .flashnext_staging_manifest import Manifest, align


@triton.jit
def _copy(Descriptors, Starts, Staging, N: tl.constexpr, GATHER: tl.constexpr,
          BLOCK: tl.constexpr):
    tile=tl.program_id(0)
    lo=0
    hi=N
    while lo+1 < hi:
        mid=(lo+hi)//2
        before=tl.load(Starts+mid)<=tile
        lo=tl.where(before,mid,lo)
        hi=tl.where(before,hi,mid)
    base=Descriptors+lo*10
    ptr=tl.load(base+0).to(tl.pointer_type(tl.uint8))
    indices=tl.load(base+1).to(tl.pointer_type(tl.int64))
    stride=tl.load(base+2)
    slice_offset=tl.load(base+3)
    group_stride=tl.load(base+4)
    slice_bytes=tl.load(base+5)
    row_bytes=tl.load(base+6)
    offset=tl.load(base+7)
    size=tl.load(base+8)
    rounded=tl.load(base+9)
    x=(tile-tl.load(Starts+lo))*BLOCK+tl.arange(0,BLOCK)
    live=x<size
    row=tl.load(indices+x//row_bytes,mask=live,other=0)
    within=x%row_bytes
    address=row*stride+slice_offset+(within//slice_bytes)*group_stride+within%slice_bytes
    if GATHER:
        value=tl.load(ptr+address,mask=live,other=0)
        # Canonical zero alignment bytes avoid leaking a previous slot payload.
        tl.store(Staging+offset+x,value,mask=x<rounded)
    else:
        value=tl.load(Staging+offset+x,mask=live,other=0)
        tl.store(ptr+address,value,mask=live)


@dataclass
class LocalRows:
    tensor: torch.Tensor
    rows: torch.Tensor
    # One row may contain several independent head-sharded sub-blocks.
    slice_offset: int = 0
    group_stride: int = 0
    slice_bytes: int = 0


def descriptors(*, manifest: Manifest, local: dict, device):
    """Validate the actual local views before a kernel can touch any address."""
    manifest.validate()
    data=[]
    starts=[0]
    retained=[]
    for field in manifest.fields:
        view=local.get(field.key)
        if view is None:
            if field.handoff_only:
                continue
            raise ValueError(f"missing local field {field.key}")
        t,rows=view.tensor,view.rows
        if (not t.is_contiguous() or t.ndim < 1 or not rows.is_contiguous()
                or rows.dtype != torch.int64 or rows.ndim != 1
                or len(rows) != field.shape[0] or t.device != rows.device
                or t.device != torch.device(device)):
            raise ValueError("incompatible local row view")
        # Indices are constructed/validated on CPU by the transfer adapter.
        # Do not synchronize GPU row maps in the background hot path.
        row_bytes=field.nbytes//field.shape[0]
        stride=t[0].numel()*t.element_size()
        width=view.slice_bytes or row_bytes
        group_stride=view.group_stride or width
        groups=row_bytes//width
        if (width <= 0 or row_bytes%width or view.slice_offset < 0
                or group_stride < width
                or view.slice_offset+(groups-1)*group_stride+width > stride):
            raise ValueError("local shard exceeds a physical row")
        rounded=align(field.nbytes)
        data.append([t.data_ptr(),rows.data_ptr(),stride,view.slice_offset,
                     group_stride,width,row_bytes,field.offset,field.nbytes,rounded])
        starts.append(starts[-1]+triton.cdiv(rounded,4096))
        retained.extend((t,rows))
    if not data:
        raise ValueError("empty local staging copy")
    return (torch.tensor(data,dtype=torch.int64,device=device),
            torch.tensor(starts,dtype=torch.int64,device=device),starts[-1],retained)


def copy_payload(*, manifest: Manifest, local: dict, staging: torch.Tensor, gather: bool):
    """Real transport entry point, also used by Triton CPU interpretation tests.

    The caller retains the returned descriptor tensors until its CUDA event
    completes, and orders the stream after all source writes or bulk DMA.
    """
    if staging.dtype != torch.uint8 or staging.ndim != 1 or not staging.is_contiguous() or staging.numel()<manifest.nbytes:
        raise ValueError("staging buffer is smaller than the manifest")
    if gather and any(f.key not in local for f in manifest.fields):
        raise ValueError("gather must include every manifest field, including latent")
    desc,starts,tiles,retained=descriptors(manifest=manifest,local=local,device=staging.device)
    _copy[(tiles,)](desc,starts,staging,N=desc.shape[0],GATHER=gather,BLOCK=4096)
    return desc,starts,retained
