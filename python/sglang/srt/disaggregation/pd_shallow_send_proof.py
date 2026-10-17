"""Opt-in live D byte gate; never enable on performance measurements.

One immutable P request source is transported twice before the worker's one
aux/Success. Both passes use the real Endpoint/backend and real D scatter.
"""
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def save_receipt(endpoint, manifest, rid, phase, **data):
    if not endpoint.proof_directory:
        raise ValueError('P12 negative gate requires staging proof directory')
    directory=Path(endpoint.proof_directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/(f'p12-negative-rank{endpoint.manager.attn_tp_rank}'
        f'-room{manifest.room}-g{manifest.generation}-c{manifest.chunk_index}-{phase}.pt')
    temporary=path.with_suffix(f'.{os.getpid()}.tmp')
    data.setdefault('source_device',str(endpoint.storage.buffers[0].device))
    torch.save(dict(rid=rid,rank=endpoint.manager.attn_tp_rank,room=manifest.room,
        generation=manifest.generation,chunk_index=manifest.chunk_index,phase=phase,
        prompt_tokens=manifest.prompt_tokens,
        timing_eligible=False,**data),temporary)
    temporary.replace(path)
    logger.info('P12_WIRE_NEGATIVE_RECEIPT phase=%s room=%s rank=%s',phase,manifest.room,endpoint.manager.attn_tp_rank)


def negative_destination(endpoint, request, *, phase):
    """Gate-only D snapshots before remote writes and after native abort drain.

    Compare actual destination rows independently of the staging copy kernel.
    Handoff-only fields have no destination pool; absence of READY is required
    for those. The protocol still owns/reclaims its leases as usual.
    """
    if os.environ.get('SGLANG_FLASHNEXT_PD_STAGING_PROOF_TENSORS')!='1':
        raise ValueError('P12 negative gate requires tensor snapshots on D')
    m=request['manifest'];room=endpoint.rooms[m.room]
    fields={}
    with torch.cuda.stream(endpoint.stream):
        endpoint.stream.wait_event(room['ready'])
        local=endpoint.catalog.destination_payload(manifest=m,kv_indices=room['kv'],
            state_indices=room['state'],decode_prefix_tokens=room['prefix'],
            destination_rank=endpoint.manager.attn_tp_rank,destination_tp=endpoint.manager.attn_tp_size)
        for field in m.fields:
            view=local.get(field.key)
            if view is None:
                if not field.handoff_only:raise ValueError('missing negative destination field')
                continue
            raw=view.tensor.index_select(0,view.rows).contiguous().view(torch.uint8)
            raw=raw.reshape(field.shape[0],-1).cpu()
            width=field.nbytes//field.shape[0]
            span=view.slice_bytes or width;stride=view.group_stride or span
            fields[field.key]=torch.cat([raw[row,view.slice_offset+g*stride:
                view.slice_offset+g*stride+span] for row in range(field.shape[0])
                for g in range(width//span)])
    save_receipt(endpoint,m,request['proof'],phase,fields=fields,
        handoff_only=[f.key for f in m.fields if f.handoff_only],
        source_device=str(endpoint.storage.buffers[0].device))
    logger.info('P12_WIRE_NEGATIVE_D phase=%s room=%s rank=%s',phase,m.room,endpoint.manager.attn_tp_rank)


def inject_negative(endpoint, chunk, rid, manifest, early_ranges):
    """Inject only after the real backend completed independent KV DMA."""
    mode=os.environ.get('SGLANG_GDN_PD_P12_WIRE_NEGATIVE')
    if mode not in ('event','abort') or not rid.startswith(f'p12-negative-{mode}-'):
        raise ValueError('P12 negative rid must match explicit event/abort gate mode')
    if not endpoint.proof_directory:
        raise ValueError('P12 negative gate requires staging proof directory')
    fence=chunk.state_handoff_fence
    if fence.validated:raise ValueError('P12 negative gate requires an unvalidated factor fence')
    base=dict(mode=mode,kv_bytes=sum(n for _,n in early_ranges),backend_kv_completed=True)
    if mode=='abort':
        endpoint.manager.conclude_failure(bootstrap_room=chunk.room,
            failure_reason='P12 gate: native abort after independent KV DMA')
        save_receipt(endpoint,manifest,rid,'injected',**base)
    else:
        original=fence.publication_done
        class NotPublished:
            def synchronize(self):
                if original is not None:original.synchronize()
            def query(self):return False
        fence.publication_done=NotPublished()
        try:
            try:
                fence.wait_aux(chunk.wait_event)
            except Exception as error:
                from .state_handoff import FactorPublicationError
                if not isinstance(error,FactorPublicationError):raise
                save_receipt(endpoint,manifest,rid,'injected',event_rejected=True,**base)
                raise
            raise RuntimeError('P12 negative: incomplete publication was accepted')
        finally:
            fence.publication_done=original
    logger.info('P12_WIRE_NEGATIVE_P mode=%s room=%s rank=%s',mode,chunk.room,endpoint.manager.attn_tp_rank)


def install_replay(endpoint):
    if not endpoint.proof_directory or os.environ.get('SGLANG_FLASHNEXT_PD_STAGING_PROOF_TENSORS') != '1':
        raise ValueError('P12 wire replay requires staging proof directory and tensor snapshots')
    original = endpoint.transfer

    def replay(*, chunk, request, target):
        rid = endpoint.proof_rooms.get(chunk.room)
        fence = chunk.state_handoff_fence
        if not rid or not getattr(fence, 'kv_first', False):
            raise ValueError('P12 wire replay requires a selected shallow P12 handoff')
        # Finish precisely this request's producer first. Pool/page ownership
        # remains with the request until both transfers and the one aux finish.
        fence.wait_aux(chunk.wait_event)
        try:
            fence.kv_first = False
            endpoint.proof_rooms[chunk.room] = str(rid) + '-p12-before'
            rc = original(chunk=chunk, request=request, target=target)
            if rc != 0:
                return rc
            fence.kv_first = True
            endpoint.proof_rooms[chunk.room] = str(rid) + '-p12-after'
            return original(chunk=chunk, request=request, target=target)
        finally:
            fence.kv_first = True

    endpoint.transfer = replay
    logger.info('P12_WIRE_REPLAY_ACTIVE frozen_source=True aux_success=once timing_eligible=False')
