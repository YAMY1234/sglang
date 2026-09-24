"""Registered bulk transfer protocol using existing Mooncake control threads.

P's background transfer worker waits for D's scatter acknowledgement. This
orders the existing Success/abort acknowledgement after all destination writes
without blocking either scheduler or changing request/cache semantics.
"""
from __future__ import annotations

from collections import deque
import itertools
import json
import logging
import os
import threading
import time

import torch

from .flashnext_staging import _RESERVES
from .flashnext_staging_kernels import copy_payload
from .flashnext_staging_manifest import Manifest


HEADER=b'FLASHNEXT_STAGE_V1'
logger=logging.getLogger(__name__)


class Endpoint:
    def __init__(self, manager):
        self.manager=manager
        self.device=manager.kv_args.gpu_id
        self.catalog=manager.kv_args.flashnext_staging_catalog
        self.storage=_RESERVES.get(self.device)
        if self.storage is None:
            raise RuntimeError('staging must be reserved before KV memory profiling')
        if manager.enable_staging or manager.dcp_size!=1 or manager.pp_size!=1:
            raise ValueError('Flash-Next staging is separate from generic staging and requires DCP1/PP1')
        if not manager.enable_deferred_decode_kv_release:
            raise ValueError('staging requires deferred decode release for abort drain')
        self.storage.register(manager.engine)
        self.cv=threading.Condition(threading.RLock())
        self.serial=itertools.count(1)
        self.responses={}
        self.rooms={}
        self.waiting=deque()
        self.active={}
        self.aborted=set()
        self.proof_rooms={}
        self.proof_directory=os.environ.get('SGLANG_FLASHNEXT_PD_STAGING_PROOF_DIR')
        self.stream=torch.cuda.Stream(device=self.device)

    def select_proof(self, *, room, rid):
        if self.proof_directory and str(rid).startswith('pdtune-isolated-'):
            self.proof_rooms[room]=str(rid)

    def _send(self, endpoint, port, parts):
        from sglang.srt.utils.network import NetworkAddress
        na=NetworkAddress(endpoint,port)
        self.manager._send_multipart_locked(na.to_tcp(),[HEADER,*parts],is_ipv6=na.is_ipv6)

    def _reply(self, request, kind, payload):
        self._send(request['ip'],request['port'],[kind,request['key'].encode(),json.dumps(payload).encode()])

    def register_room(self, *, room, kv_indices, state_indices, prefix):
        # Allocation/reset writes on the scheduler's current stream must finish
        # before the separate arrival thread overwrites these destination slots.
        ready=torch.cuda.Event();ready.record()
        with self.cv:
            if room in self.rooms:raise RuntimeError('duplicate staging room registration')
            self.aborted.discard(room)
            self.rooms[room]=dict(kv=kv_indices.copy(),state=state_indices,
                                  prefix=prefix or 0,ready=ready,scatter_inflight=0)

    def on_message(self, msg):
        if msg[0]!=HEADER:return False
        torch.cuda.set_device(self.device)
        kind=msg[1]
        if kind in (b'SLOT',b'DONE',b'ERROR'):
            with self.cv:
                self.responses[msg[2].decode(),kind]=json.loads(msg[3])
                self.cv.notify_all()
        elif kind==b'RESERVE':
            m=Manifest.from_bytes(msg[5])
            request=dict(key=msg[2].decode(),ip=msg[3].decode(),port=int(msg[4]),manifest=m,
                         proof=msg[6].decode() if len(msg)>6 else '')
            with self.cv:
                self.waiting.append(request)
                self._grant_waiting()
        elif kind==b'READY':
            self._scatter(msg[2].decode(),int(msg[3]))
        else:
            raise ValueError('unknown Flash-Next staging message')
        return True

    def _grant_waiting(self):
        # Called under cv. Never wait here: this same control thread must still
        # receive READY for the slots that will make capacity available.
        while self.waiting:
            request=self.waiting[0];m=request['manifest']
            if m.room not in self.rooms or m.room in self.aborted:
                self.waiting.popleft();self._reply(request,b'ERROR',dict(reason='room not live'));continue
            if request['key'] in self.active:
                raise ValueError('duplicate staging reservation')
            lease=self.storage.leases.acquire(room=m.room,nbytes=m.nbytes)
            if lease is None:return
            self.waiting.popleft()
            self.storage.leases.begin(lease,operation='remote-writer')
            self.active[request['key']]=(request,lease)
            self._reply(request,b'SLOT',dict(ptr=self.storage.buffers[lease.slot].data_ptr(),
                        nbytes=lease.nbytes,generation=lease.generation))

    def _scatter(self, key, generation):
        with self.cv:
            record=self.active.get(key)
            if record is None:return  # already drained abort, or stale notification
            request,lease=record;m=request['manifest']
            if lease.generation!=generation:raise ValueError('stale staging READY generation')
            self.storage.leases.finish(lease,operation='remote-writer')
            if m.room in self.aborted or m.room not in self.rooms:
                self.storage.leases.abort(lease);self.storage.leases.release(lease)
                del self.active[key];self._reply(request,b'ERROR',dict(reason='room aborted'))
                self._grant_waiting();return
            room=self.rooms[m.room]
            room['scatter_inflight']+=1
            self.storage.leases.begin(lease,operation='scatter')
        begun=time.perf_counter_ns()
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        try:
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(room['ready'])
                local=self.catalog.destination_payload(manifest=m,kv_indices=room['kv'],
                    state_indices=room['state'],decode_prefix_tokens=room['prefix'],
                    destination_rank=self.manager.attn_tp_rank,destination_tp=self.manager.attn_tp_size)
                start.record()
                retained=copy_payload(manifest=m,local=local,staging=self.storage.buffers[lease.slot],gather=False)
                end.record()
            end.synchronize()  # background control thread only; request not exposed yet
            elapsed=start.elapsed_time(end)
            if request['proof']:
                if not self.proof_directory:raise RuntimeError('D staging byte audit was not enabled')
                from .flashnext_staging_audit import audit_payload
                audit_payload(manifest=m,local=local,staging=self.storage.buffers[lease.slot],
                    directory=self.proof_directory,role='decode',rank=self.manager.attn_tp_rank,rid=request['proof'])
            del retained,local
            response=dict(scatter_ms=elapsed,scatter_wall_ms=(time.perf_counter_ns()-begun)/1e6)
            kind=b'DONE'
        except Exception as exc:
            # Drain even a partially launched copy before abort can release KV.
            self.stream.synchronize()
            kind=b'ERROR';response=dict(reason=repr(exc))
        with self.cv:
            room['scatter_inflight']-=1
            self.storage.leases.finish(lease,operation='scatter')
            self.storage.leases.release(lease)
            del self.active[key]
            self._reply(request,kind,response)
            self._grant_waiting()

    def _wait(self, key, kind):
        deadline=time.monotonic()+60
        with self.cv:
            while (key,kind) not in self.responses:
                error=self.responses.pop((key,b'ERROR'),None)
                if error is not None:raise RuntimeError(f'staging peer rejected payload: {error}')
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('staging control response deadline')
                self.cv.wait(min(remaining,.1))
            return self.responses.pop((key,kind))

    def transfer(self, *, chunk, request, target):
        """One background worker dispatch, replacing all per-page payload calls."""
        if not chunk.is_last_chunk or (chunk.index_slice.start or 0)!=0:
            raise ValueError('Flash-Next staging gathers only the complete handoff')
        if request.decode_prefix_len:
            raise ValueError('staging decode radix prefix reuse is not enabled')
        if target.dst_attn_tp_size!=self.manager.attn_tp_size:
            raise ValueError('heterogeneous TP transport requires q/k/v manifest subdivision')
        torch.cuda.set_device(self.device)
        prompt=int(chunk.num_kv_tokens)
        shallow=any('pd_h31' in entry.name for entry in self.catalog.entries)
        # Include aligned field headers and all fixed state in the byte bound.
        per_page=sum(e.tensor[0].nbytes for e in self.catalog.entries if e.tokens_per_row)
        if getattr(self.catalog.pool,'shared_arena',False):per_page+=1972*64
        fixed=sum(e.tensor[0].nbytes for e in self.catalog.entries if not e.tokens_per_row)
        max_pages=(self.storage.leases.slot_bytes-fixed-(len(self.catalog.entries)+4)*256)//per_page
        if max_pages<1:raise ValueError('staging slot cannot hold one page plus boundary state')
        pages=chunk.prefill_kv_indices
        generation=next(self.serial)
        proof=self.proof_rooms.pop(chunk.room,'')
        for ci,begin in enumerate(range(0,len(pages),max_pages)):
            end=min(len(pages),begin+max_pages);last=end==len(pages)
            maps=chunk.prefill_kv_indices_by_entry
            m,local=self.catalog.source_payload(room=chunk.room,generation=generation,
                source_rank=self.manager.attn_tp_rank,source_tp=self.manager.attn_tp_size,
                prompt_tokens=prompt,token_start=begin*64,token_end=min(prompt,end*64),
                kv_indices=pages[begin:end],kv_by_entry=maps[:,begin:end] if maps is not None else None,
                state_indices=chunk.state_indices,chunk_index=ci,last_chunk=last,shallow_boundary=shallow)
            deadline=time.monotonic()+60
            lease=None
            while lease is None:
                lease=self.storage.leases.acquire(room=chunk.room,nbytes=m.nbytes)
                if lease is None:
                    if time.monotonic()>deadline:raise TimeoutError('P staging backpressure deadline')
                    with self.cv:self.cv.wait(.001)
            key=f'{self.manager.get_session_id()}:{chunk.room}:{generation}:{ci}'
            started=time.perf_counter_ns()
            start,end_event=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            gathering=False
            bulk_outstanding=False
            try:
                self._send(request.endpoint,request.dst_port,[b'RESERVE',key.encode(),
                    self.manager.local_ip.encode(),str(self.manager.rank_port).encode(),m.to_bytes(),proof.encode()])
                self.storage.leases.begin(lease,operation='gather')
                gathering=True
                with torch.cuda.stream(self.stream):
                    if chunk.wait_event is not None:self.stream.wait_event(chunk.wait_event)
                    start.record()
                    retained=copy_payload(manifest=m,local=local,staging=self.storage.buffers[lease.slot],gather=True)
                    end_event.record()
                slot=self._wait(key,b'SLOT')
                end_event.synchronize()
                self.storage.leases.finish(lease,operation='gather')
                gathering=False
                if proof:
                    from .flashnext_staging_audit import audit_payload
                    audit_payload(manifest=m,local=local,staging=self.storage.buffers[lease.slot],
                        directory=self.proof_directory,role='prefill',rank=self.manager.attn_tp_rank,rid=proof)
                if slot['nbytes']!=m.nbytes:raise ValueError('D staging capacity differs from manifest')
                self.storage.leases.begin(lease,operation='bulk')
                bulk_outstanding=True
                before=time.perf_counter_ns()
                rc=self.manager._transfer_data(request.mooncake_session_id,
                    [(self.storage.buffers[lease.slot].data_ptr(),slot['ptr'],m.nbytes)])
                bulk_ms=(time.perf_counter_ns()-before)/1e6
                self.storage.leases.finish(lease,operation='bulk')
                bulk_outstanding=False
                if rc!=0:return rc
                self._send(request.endpoint,request.dst_port,[b'READY',key.encode(),str(slot['generation']).encode()])
                done=self._wait(key,b'DONE')
                metrics=dict(room=chunk.room,chunk=ci,rank=self.manager.attn_tp_rank,
                    bytes=m.nbytes,fields=len(m.fields),bulk_segments=1,gather_ms=start.elapsed_time(end_event),
                    bulk_ms=bulk_ms,**done,wall_ms=(time.perf_counter_ns()-started)/1e6,
                    proof=bool(proof),reserved_bytes=self.storage.leases.reserved_bytes,
                    peak_slots=self.storage.leases.peak_slots,peak_payload_bytes=self.storage.leases.peak_bytes)
                self.manager.flashnext_staging_metrics.append(metrics)
                logger.info('Flash-Next staging transfer: %s',json.dumps(metrics,sort_keys=True))
                del retained,local
            finally:
                self.stream.synchronize()
                if gathering:self.storage.leases.finish(lease,operation='gather')
                # An exception from the engine leaves DMA drain unknown. Keep
                # that slot quarantined; the native transfer worker fails the
                # process rather than making its backing available again.
                if not bulk_outstanding:self.storage.leases.release(lease)
                with self.cv:self.cv.notify_all()
        return 0

    def abort_drained(self, room):
        """Call only AFTER native prefill abort acknowledgements have arrived."""
        with self.cv:
            self.aborted.add(room)
            state=self.rooms.get(room)
            if state and state['scatter_inflight']:return False
            for key,(request,lease) in list(self.active.items()):
                if lease.room!=room:continue
                # Native ACK proves no P DMA remains, and scatter_inflight==0
                # proves no local writer remains. Future READY is ignored.
                self.storage.leases.finish(lease,operation='remote-writer')
                self.storage.leases.abort(lease);self.storage.leases.release(lease)
                del self.active[key]
                self._reply(request,b'ERROR',dict(reason='aborted after drain'))
            self._grant_waiting()
            return True

    def clear_room(self, room):
        with self.cv:
            if any(lease.room==room for _,lease in self.active.values()):
                raise RuntimeError('cannot clear a staging room with active destination writes')
            self.rooms.pop(room,None)
            self.aborted.discard(room)
