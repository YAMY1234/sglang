"""Flash-Next staging storage and local payload catalog (PD only).

Storage is reserved after model loading and BEFORE KV budget profiling. The
Mooncake adapter registers these allocations later, when its engine exists.
Neither the cache allocator nor its 64-token page geometry is changed.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math

import numpy as np
import torch

from .flashnext_staging_kernels import LocalRows, copy_payload
from .flashnext_staging_manifest import Field, Manifest, LeasePool

logger=logging.getLogger(__name__)
_RESERVES={}


def reserve_before_kv_profile(*, config=None):
    if config is None:
        from sglang.srt.runtime_context import get_disagg
        config=get_disagg()
    if not getattr(config,'flashnext_pd_staging',False):
        return None
    if config.disaggregation_mode not in ('prefill','decode'):
        raise ValueError('Flash-Next staging is a PD-only service flag')
    device=torch.cuda.current_device()
    if device in _RESERVES:
        return _RESERVES[device]
    from .mooncake.utils import init_mooncake_custom_mem_pool
    enabled,pool,kind=init_mooncake_custom_mem_pool(f'cuda:{device}')
    if not enabled or pool is None or kind != 'NVLINK':
        raise RuntimeError('Flash-Next staging requires the dedicated NVLink allocator')
    count=int(config.flashnext_pd_staging_slots)
    size=int(config.flashnext_pd_staging_slot_mib)*(1<<20)
    leases=LeasePool(slots=count,slot_bytes=size)
    before=torch.cuda.memory_allocated(device)
    with torch.cuda.use_mem_pool(pool):
        buffers=[torch.empty(size,dtype=torch.uint8,device=device) for _ in range(count)]
    result=Storage(buffers,leases,pool,torch.cuda.memory_allocated(device)-before)
    _RESERVES[device]=result
    logger.info('Flash-Next staging reserved before KV sizing: role=%s slots=%d slot_bytes=%d reserved_bytes=%d torch_delta_bytes=%d allocator=NVLINK',
                config.disaggregation_mode,count,size,leases.reserved_bytes,result.allocated_delta)
    return result


@dataclass
class Storage:
    buffers: list
    leases: LeasePool
    custom_pool: object
    allocated_delta: int
    registered: bool = False

    def register(self, engine):
        if not self.registered:
            rc=engine.batch_register([x.data_ptr() for x in self.buffers],
                                     [x.numel() for x in self.buffers])
            if rc not in (None,0):
                raise RuntimeError(f'NVLink staging registration failed: {rc}')
            self.registered=True


@dataclass
class Entry:
    layer: int
    name: str
    tensor: torch.Tensor
    component: int  # -1=KV; otherwise state_types index
    index: int
    tokens_per_row: int
    compression: int = 1
    slice_axis: int | None = None
    conv_groups: tuple | None = None

    @property
    def key(self):
        return self.layer,self.name


class Catalog:
    """Bind every registered byte range to a typed local tensor at startup."""
    def __init__(self, *, args, pool):
        from .base.conn import StateType
        if args.page_size != 64 or args.num_draft_entries:
            raise ValueError('Flash-Next staging requires page64 without draft entries')
        self.pool=pool
        self.args=args
        self.entries=[]
        self.by_key={}
        # The final source aliases backing pointers; entry identity is layer
        # and field, never pointer equality or an assumed common source page.
        counts={}
        for i,(layer,ptr,size) in enumerate(zip(args.kv_layer_ids,args.kv_data_ptrs,args.kv_item_lens,strict=True)):
            part=pool.deep if getattr(pool,'shared_arena',False) and layer>=31 else pool
            ordinal=counts.get(layer,0);counts[layer]=ordinal+1
            if ordinal not in (0,1):raise ValueError('unexpected QSA KV entry multiplicity')
            name='K' if ordinal==0 else 'V'
            t=(part.get_key_buffer if ordinal==0 else part.get_value_buffer)(layer)
            view=t.view(-1,64,*t.shape[1:])
            if view.data_ptr()!=ptr or view[0].nbytes!=size:
                raise ValueError('KV tensor disagrees with Mooncake registration')
            self._add(Entry(layer,name,view,-1,i,64,slice_axis=2))
        mamba=list(pool.mamba_pool._iter_transfer_state_entries())
        for ci,kind in enumerate(args.state_types):
            layers=args.state_layer_ids[ci]
            ptrs=args.state_data_ptrs[ci]
            sizes=args.state_item_lens[ci]
            if kind==StateType.MAMBA:
                records=mamba
                if len(records)!=len(ptrs):raise ValueError('Mamba catalog differs from wire registration')
                counts={}
                for i,(name,t,axis,layer) in enumerate(records):
                    ordinal=counts.get((layer,name),0);counts[layer,name]=ordinal+1
                    entry=Entry(layer,f'mamba.{name}.{ordinal}',t,ci,i,0,
                                slice_axis=None if axis is None else axis+1,
                                conv_groups=(tuple(args.state_conv_shard_groups[ci][i])
                                    if args.state_conv_shard_groups[ci][i] else None))
                    if t.data_ptr()!=ptrs[i] or t[0].nbytes!=sizes[i] or layer!=layers[i]:
                        raise ValueError('Mamba tensor disagrees with Mooncake registration')
                    self._add(entry)
            elif kind==StateType.QSA_COMPRESSED:
                for i,layer in enumerate(layers):
                    part=pool.deep if getattr(pool,'shared_arena',False) and layer>=31 else pool
                    t=part.get_qsa_compressed_k_buffer(layer)
                    view=t.view(-1,16,*t.shape[1:])
                    if view.data_ptr()!=ptrs[i] or view[0].nbytes!=sizes[i]:
                        raise ValueError('compressed-index registration mismatch')
                    # Compressed index vectors follow QSA's local head shard.
                    self._add(Entry(layer,'compressed_index',view,ci,i,64,4,slice_axis=2 if view.ndim>=4 else None))
            elif kind==StateType.QSA_PENDING:
                tensors=[*pool.qsa_key_state_buffer_pool]
                if getattr(pool,'shared_arena',False):tensors.extend(pool.deep.qsa_key_state_buffer_pool)
                tensors.append(pool.qsa_rope_position_buffer)
                for i,(layer,t) in enumerate(zip(layers,tensors,strict=True)):
                    view=t.view(-1,4,*t.shape[1:])
                    if view.data_ptr()!=ptrs[i] or view[0].nbytes!=sizes[i]:
                        raise ValueError('pending-ring registration mismatch')
                    self._add(Entry(layer,'pending',view,ci,i,0,
                                    slice_axis=2 if i<len(tensors)-1 and view.ndim>=4 else None))
            else:
                raise ValueError(f'unsupported staging state component {kind}')

    def _add(self,entry):
        if entry.key in self.by_key:raise ValueError('duplicate local staging field')
        self.entries.append(entry);self.by_key[entry.key]=entry

    @staticmethod
    def _indices(entry,kv_indices,kv_by_entry,state_indices):
        if entry.component==-1:
            return np.asarray(kv_indices if kv_by_entry is None else kv_by_entry[entry.index],dtype=np.int64)
        raw=np.asarray(state_indices[entry.component],dtype=np.int64)
        if entry.tokens_per_row and raw.ndim==2:
            return raw[entry.index]
        return raw.reshape(-1)

    def source_payload(self, *, room, generation, source_rank, source_tp,
                       prompt_tokens, token_start, token_end, kv_indices,
                       kv_by_entry, state_indices, chunk_index, last_chunk,
                       shallow_boundary):
        fields=[];local={}
        cache={}
        for entry in self.entries:
            if entry.component>=0 and not last_chunk and not entry.tokens_per_row:continue
            if entry.component>=0 and entry.tokens_per_row:
                source_index=next(e.index for e in self.entries if e.component==-1 and e.layer==entry.layer)
                rows=np.asarray(kv_indices if kv_by_entry is None else kv_by_entry[source_index],dtype=np.int64)
            else:
                rows=self._indices(entry,kv_indices,kv_by_entry,state_indices)
            if not len(rows):continue
            if rows.ndim!=1 or np.any(rows<0) or np.any(rows>=len(entry.tensor)):
                raise ValueError('source page/slot outside registered buffer')
            begin,end=(token_start,token_end) if entry.tokens_per_row else (0,prompt_tokens)
            shape=(len(rows),*entry.tensor.shape[1:])
            axis=entry.slice_axis
            shard={}
            if axis is not None:
                heads=shape[axis]
                shard=dict(shard_axis=axis,head_start=heads*source_rank,
                           head_end=heads*(source_rank+1),total_heads=heads*source_tp)
            field=Field(entry.layer,entry.name,str(entry.tensor.dtype).removeprefix('torch.'),
                        shape,begin,end,entry.tokens_per_row,entry.compression,**shard)
            # Reuse identical page/slot vectors within this one payload.
            identity=rows.tobytes()
            if identity not in cache:
                cache[identity]=torch.tensor(rows,device=entry.tensor.device,dtype=torch.int64)
            fields.append(field);local[field.key]=LocalRows(entry.tensor,cache[identity])
        if getattr(self.pool,'shared_arena',False):
            # Native final wire payload: two arena units spread over K/V. No
            # factor/latent arithmetic is performed here; copy its 1,972 bytes.
            # The emitter excludes the boundary token under shallow handoff.
            end=min(token_end,prompt_tokens-int(shallow_boundary))
            n=max(0,end-token_start)
            pages=np.asarray(kv_indices,dtype=np.int64)
            owner=np.asarray([self.pool.arena.shared[int(p)][7:9] for p in pages],dtype=np.int64)
            physical=(owner[:,:,None]*64+np.arange(64)[None,None,:]).transpose(1,0,2).reshape(2,-1)
            rows=[torch.tensor(x[:n],device=self.pool.unified_k.device,dtype=torch.int64) for x in physical]
            for i,(t,col,width) in enumerate(((self.pool.unified_k,0,512),(self.pool.unified_k,1,512),
                                            (self.pool.unified_v,0,512),(self.pool.unified_v,1,436))):
                if n<=0:continue
                f=Field(-1,f'latent.payload{i}','uint8',(n,width),token_start,end,1,handoff_only=True)
                fields.append(f);local[f.key]=LocalRows(t,rows[col],slice_bytes=width)
        manifest=Manifest.build(room=room,generation=generation,source_rank=source_rank,
            source_tp=source_tp,prompt_tokens=prompt_tokens,chunk_index=chunk_index,
            last_chunk=last_chunk,shallow_count=9 if shallow_boundary else 0,
            deep_count=8 if shallow_boundary else 0,fields=fields)
        return manifest,local

    def destination_payload(self, *, manifest, kv_indices, state_indices,
                            decode_prefix_tokens, destination_rank, destination_tp):
        local={};cache={}
        for field in manifest.fields:
            if field.handoff_only:continue
            entry=self.by_key.get(field.key)
            if entry is None:raise ValueError(f'unknown destination staging field {field.key}')
            if entry.tokens_per_row:
                begin=(field.token_start-decode_prefix_tokens)//64
                rows=np.asarray(kv_indices[begin:begin+field.shape[0]],dtype=np.int64)
            else:
                rows=self._indices(entry,kv_indices,None,state_indices)
            if len(rows)!=field.shape[0] or np.any(rows<0) or np.any(rows>=len(entry.tensor)):
                raise ValueError('destination page/slot outside registered buffer')
            tensor=entry.tensor
            if str(tensor.dtype).removeprefix('torch.')!=field.dtype:
                raise ValueError('source/destination staging dtype differs')
            args={}
            if field.shard_axis!=-1:
                axis=field.shard_axis
                local_heads=tensor.shape[axis]
                first=destination_rank*local_heads
                if (local_heads*destination_tp!=field.total_heads
                        or not first<=field.head_start<field.head_end<=first+local_heads):
                    raise ValueError('payload head interval needs source subdivision for this destination')
                if entry.conv_groups and destination_tp!=manifest.source_tp:
                    raise ValueError('heterogeneous conv requires independent q/k/v field subdivision')
                inner=math.prod(tensor.shape[axis+1:])*tensor.element_size()
                args=dict(slice_offset=(field.head_start-first)*inner,
                          group_stride=local_heads*inner,
                          slice_bytes=(field.head_end-field.head_start)*inner)
            elif tuple(tensor.shape[1:])!=tuple(field.shape[1:]):
                raise ValueError('replicated staging field shape differs')
            identity=rows.tobytes()
            if identity not in cache:
                cache[identity]=torch.tensor(rows,device=tensor.device,dtype=torch.int64)
            local[field.key]=LocalRows(tensor,cache[identity],**args)
        return local
