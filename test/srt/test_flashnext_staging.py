"""TRITON_INTERPRET=1 executes the real keyword transport copy entry on CPU."""
import json
import os
os.environ.setdefault("TRITON_INTERPRET", "1")
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.flashnext_staging_manifest import Field, Manifest, LeasePool
from sglang.srt.disaggregation.flashnext_staging_kernels import LocalRows, copy_payload


def manifest(fields, tokens=257):
    return Manifest.build(room=71,generation=1,source_rank=0,source_tp=2,
        prompt_tokens=tokens,chunk_index=0,last_chunk=True,shallow_count=9,
        deep_count=8,fields=fields)


class StagingTest(unittest.TestCase):
    def make_catalog(self, seed):
        from sglang.srt.disaggregation.base.conn import StateType
        from sglang.srt.disaggregation.flashnext_staging import Catalog
        import numpy as np
        def fixture(seed):
            generator=torch.Generator().manual_seed(seed)
            def tensor(shape,dtype=torch.bfloat16):
                return torch.randint(0,120,shape,generator=generator,dtype=torch.int64).to(dtype)
            layers=[3,47]
            keys={l:tensor((13*64,1,8)) for l in layers}
            values={l:tensor((13*64,1,8)) for l in layers}
            compact={l:tensor((13*16,1,8)) for l in layers}
            pending=[tensor((8*4,1,8)) for l in layers]
            rope=tensor((8*4,2),torch.int64)
            states=[('count',tensor((8,2),torch.int32),0,0),
                    ('pd_h31',tensor((8,10)),None,4294967290)]
            pool=SimpleNamespace(mamba_pool=SimpleNamespace(_iter_transfer_state_entries=lambda:iter(states)),
                get_key_buffer=keys.__getitem__,get_value_buffer=values.__getitem__,
                get_qsa_compressed_k_buffer=compact.__getitem__,
                qsa_key_state_buffer_pool=pending,qsa_rope_position_buffer=rope)
            kv=[*keys.values(),*values.values()]
            components=[[s[1] for s in states],[*pending,rope],list(compact.values())]
            widths=[1,4,16]
            args=SimpleNamespace(page_size=64,num_draft_entries=0,kv_layer_ids=layers*2,
                kv_data_ptrs=[x.data_ptr() for x in kv],kv_item_lens=[x[0].nbytes*64 for x in kv],
                state_types=[StateType.MAMBA,StateType.QSA_PENDING,StateType.QSA_COMPRESSED],
                state_layer_ids=[[x[3] for x in states],[*layers,4294967295],layers],
                state_data_ptrs=[[x.data_ptr() for x in group] for group in components],
                state_item_lens=[[x[0].nbytes*width for x in group] for width,group in zip(widths,components)],
                state_conv_shard_groups=[[None]*len(group) for group in components])
            return Catalog(args=args,pool=pool)
        return fixture(seed)

    def test_catalog_real_source_destination_keyword_dispatch(self):
        import numpy as np
        p,d=self.make_catalog(4),self.make_catalog(5)
        src=np.asarray([11,4,8,2,9]);dst=np.asarray([3,10,1,7,5])
        maps=np.stack((src,src[::-1],src,src[::-1]))
        m,source=p.source_payload(room=19,generation=1,source_rank=0,source_tp=2,
            prompt_tokens=257,token_start=0,token_end=257,kv_indices=src,
            kv_by_entry=maps,state_indices=[[[2]],[3],maps[:2]],chunk_index=0,
            last_chunk=True,shallow_boundary=True)
        received=Manifest.from_bytes(m.to_bytes())
        target=d.destination_payload(manifest=received,kv_indices=dst,
            state_indices=[[[5]],[6],dst],decode_prefix_tokens=0,
            destination_rank=0,destination_tp=2)
        buffer=torch.empty(m.nbytes,dtype=torch.uint8)
        copy_payload(manifest=m,local=source,staging=buffer,gather=True)
        copy_payload(manifest=received,local=target,staging=buffer.clone(),gather=False)
        for field in m.fields:
            a,b=source[field.key],target[field.key]
            self.assertTrue(torch.equal(a.tensor[a.rows].view(torch.uint8),b.tensor[b.rows].view(torch.uint8)),field.key)

    def test_wire_has_no_physical_map_and_rejects_corruption(self):
        m=manifest([Field(3,"K","bfloat16",(5,64,3),0,257,64),
                    Field(0,"count","int32",(1,2),0,257,0)])
        self.assertEqual(Manifest.from_bytes(m.to_bytes()),m)
        for word in (b"ptr",b"address",b"page_id",b"rows"):
            self.assertNotIn(word,m.to_bytes())
        x=json.loads(m.to_bytes());x['fields'][1]['offset']=0
        with self.assertRaises(ValueError):Manifest.from_bytes(json.dumps(x).encode())
        with self.assertRaises(ValueError):manifest([m.fields[0],m.fields[0]])

    def test_final_latent_catalog_copies_exact_native_payload(self):
        import numpy as np
        p=self.make_catalog(8)
        p.pool.shared_arena=True
        p.pool.unified_k=torch.arange(16*64*512,dtype=torch.int64).to(torch.uint8).view(16*64,512)
        p.pool.unified_v=(p.pool.unified_k.to(torch.int16)+17).to(torch.uint8)
        p.pool.arena=SimpleNamespace(shared={11:[0]*7+[5,9],4:[0]*7+[2,13]})
        src=np.asarray([11,4]);maps=np.stack((src,src[::-1],src,src[::-1]))
        # The last token is a boundary and is absent from the emitter's latent.
        m,source=p.source_payload(room=19,generation=1,source_rank=0,source_tp=2,
            prompt_tokens=65,token_start=0,token_end=65,kv_indices=src,kv_by_entry=maps,
            state_indices=[[[2]],[3],maps[:2]],chunk_index=0,last_chunk=True,shallow_boundary=True)
        buf=torch.empty(m.nbytes,dtype=torch.uint8)
        copy_payload(manifest=m,local=source,staging=buf,gather=True)
        total=0
        for i,(t,page,width) in enumerate(((p.pool.unified_k,5,512),(p.pool.unified_k,9,512),
                                          (p.pool.unified_v,5,512),(p.pool.unified_v,9,436))):
            f=next(f for f in m.fields if f.name==f'latent.payload{i}')
            self.assertEqual((f.token_start,f.token_end,f.shape,f.handoff_only),(0,64,(64,width),True))
            expected=t[page*64:(page+1)*64,:width].contiguous().reshape(-1)
            self.assertTrue(torch.equal(buf[f.offset:f.offset+f.nbytes],expected))
            total+=f.nbytes
        self.assertEqual(total,64*1972)

    def test_real_copy_entry_fragmented_pages_all_bytes_and_tail(self):
        gen=torch.Generator().manual_seed(91)
        for tokens in (1,63,64,65,257):
            n=(tokens+63)//64
            specs=[Field(3,"K","bfloat16",(n,64,3),0,tokens,64),
                   Field(47,"V","bfloat16",(n,64,2),0,tokens,64),
                   Field(47,"compressed_index","bfloat16",(n,16,4),0,tokens,64,compression=4),
                   Field(-1,"latent","uint8",(n,64,5),0,tokens,64,handoff_only=True),
                   Field(0,"count","int32",(1,2),0,tokens,0),
                   Field(0,"conv","float32",(1,3,4),0,tokens,0),
                   Field(31,"h31","bfloat16",(1,17),0,tokens,0)]
            m=manifest(specs,tokens)
            source={};target={};expected={}
            for f in m.fields:
                rowbytes=f.nbytes//f.shape[0]
                src=torch.randint(0,256,(13,rowbytes),dtype=torch.uint8,generator=gen)
                dst=torch.full_like(src,237)
                a=torch.tensor([11,4,8,2,9][:f.shape[0]],dtype=torch.int64)
                b=torch.tensor([3,10,1,7,5][:f.shape[0]],dtype=torch.int64)
                source[f.key]=LocalRows(src,a);target[f.key]=LocalRows(dst,b)
                expected[f.key]=src[a].clone()
            staging=torch.full((m.nbytes,),129,dtype=torch.uint8)
            # Same real entry and keyword ABI the background transport will use.
            copy_payload(manifest=m,local=source,staging=staging,gather=True)
            received=staging.clone()  # exact bulk byte-copy model
            copy_payload(manifest=m,local=target,staging=received,gather=False)
            for f in m.fields:
                view=target[f.key]
                self.assertTrue(torch.equal(view.tensor[view.rows],expected[f.key]))
                self.assertTrue(torch.all(view.tensor[0]==237))
                self.assertTrue(torch.all(staging[f.offset+f.nbytes:(f.offset+f.nbytes+255)//256*256]==0))

    def test_head_slice_scatter_uses_destination_layout(self):
        # Two outer groups, P owns heads 0..7, D owns 0..3, this payload 2..3.
        f=Field(3,"K","uint8",(3,2,2,4),0,3,1,shard_axis=2,
                head_start=2,head_end=4,total_heads=8)
        m=manifest([f],3)
        src=torch.arange(7*2*8*4,dtype=torch.int64).to(torch.uint8).view(7,2,8,4)
        dst=torch.full((8,2,4,4),221,dtype=torch.uint8)
        a=torch.tensor([6,2,4]);b=torch.tensor([1,5,3])
        staging=torch.empty(m.nbytes,dtype=torch.uint8)
        copy_payload(manifest=m,local={f.key:LocalRows(src,a,8,32,8)},staging=staging,gather=True)
        copy_payload(manifest=m,local={f.key:LocalRows(dst,b,8,16,8)},staging=staging,gather=False)
        self.assertTrue(torch.equal(dst[b,:,2:4],src[a,:,2:4]))
        self.assertTrue(torch.all(dst[b,:,:2]==221))

    def test_pool_backpressure_and_abort_drain_generation(self):
        p=LeasePool(slots=2,slot_bytes=1024)
        a=p.acquire(room=4,nbytes=1000);b=p.acquire(room=5,nbytes=500)
        self.assertIsNone(p.acquire(room=6,nbytes=100))
        p.begin(a,operation='bulk');p.abort(a)
        with self.assertRaises(RuntimeError):p.release(a)
        with self.assertRaises(ValueError):p.begin(a,operation='late-scatter')
        p.finish(a,operation='bulk');p.release(a)
        c=p.acquire(room=6,nbytes=200)
        self.assertEqual(a.slot,c.slot);self.assertGreater(c.generation,a.generation)
        with self.assertRaises(ValueError):p.finish(a,operation='late-bulk')
        p.begin(c,operation='scatter')
        with self.assertRaises(RuntimeError):p.release(c)
        p.finish(c,operation='scatter');p.release(c);p.release(b)
        self.assertEqual((p.peak_slots,p.peak_bytes,p.reserved_bytes),(2,1500,2048))

    def test_real_copy_multiple_tiles_per_field(self):
        f=Field(3,'K','bfloat16',(5,64,128),0,257,64)
        m=manifest([f])
        a=torch.arange(9*64*128,dtype=torch.int64).to(torch.bfloat16).view(9,64,128)
        b=torch.zeros_like(a)
        src=torch.tensor([8,1,5,3,2]);dst=torch.tensor([2,7,3,4,6])
        buf=torch.empty(m.nbytes,dtype=torch.uint8)
        copy_payload(manifest=m,local={f.key:LocalRows(a,src)},staging=buf,gather=True)
        copy_payload(manifest=m,local={f.key:LocalRows(b,dst)},staging=buf,gather=False)
        self.assertTrue(torch.equal(a[src].view(torch.uint8),b[dst].view(torch.uint8)))

    def test_native_mooncake_worker_bulk_chunks_and_scatter_ack(self):
        # Only device scheduling/DMA are modeled. Execute the actual native
        # worker body -> Endpoint -> Catalog -> Triton byte-copy keyword ABI.
        import ast
        import contextlib
        import ctypes
        import logging
        import threading
        import time
        import tempfile
        from collections import defaultdict,deque
        from pathlib import Path
        from unittest.mock import patch
        import numpy as np
        from sglang.srt.disaggregation.base.conn import KVPoll
        from sglang.srt.disaggregation.common.utils import TransferKVChunk
        from sglang.srt.disaggregation.flashnext_staging import _RESERVES,Storage
        from sglang.srt.disaggregation.flashnext_staging_transport import Endpoint,HEADER
        class Event:
            def __init__(self,**kwargs):self.t=0
            def record(self):self.t=time.perf_counter()
            def synchronize(self):pass
            def elapsed_time(self,end):return (end.t-self.t)*1000
        class Stream:
            def __init__(self,**kwargs):pass
            def wait_event(self,event):pass
            def synchronize(self):pass
        p,d=self.make_catalog(21),self.make_catalog(22)
        src=np.asarray([11,4,8,2,9]);dst=np.asarray([3,10,1,7,5])
        maps=np.stack((src,src[::-1],src,src[::-1]))
        states=[[[2]],[3],maps[:2]];dst_states=[[[5]],[6],dst]
        def manager(catalog):
            return SimpleNamespace(kv_args=SimpleNamespace(gpu_id=0,flashnext_staging_catalog=catalog),
                enable_staging=False,dcp_size=1,pp_size=1,enable_deferred_decode_kv_release=True,
                engine=SimpleNamespace(batch_register=lambda ptrs,lens:0),attn_tp_rank=0,attn_tp_size=2,
                local_ip='127.0.0.1',rank_port=19,get_session_id=lambda:'source',
                flashnext_staging_metrics=deque())
        pm,dm=manager(p),manager(d)
        def storage():
            return Storage([torch.empty(16384,dtype=torch.uint8) for _ in range(2)],
                LeasePool(slots=2,slot_bytes=16384),None,32768)
        ps,ds=storage(),storage();bulk=[]
        with contextlib.ExitStack() as stack:
            proof_dir=stack.enter_context(tempfile.TemporaryDirectory())
            stack.enter_context(patch.dict(os.environ,{'SGLANG_FLASHNEXT_PD_STAGING_PROOF_DIR':proof_dir}))
            for name,value in dict(Event=Event,Stream=Stream,set_device=lambda device:None,
                stream=lambda stream:contextlib.nullcontext()).items():
                stack.enter_context(patch.object(torch.cuda,name,value))
            stack.enter_context(patch.dict(_RESERVES,{0:ps}))
            pe=Endpoint(pm);_RESERVES[0]=ds;de=Endpoint(dm)
            pe._send=lambda ip,port,parts:de.on_message([HEADER,*parts])
            de._send=lambda ip,port,parts:pe.on_message([HEADER,*parts])
            de.register_room(room=19,kv_indices=dst,state_indices=dst_states,prefix=0)
            pe.select_proof(room=19,rid='pdtune-isolated-native-entry')
            def transfer(session,blocks):
                self.assertEqual(len(blocks),1)
                key=next(iter(de.active));_,lease=de.active[key]
                with self.assertRaises(ValueError):de._scatter(key,lease.generation+1)
                with self.assertRaises(RuntimeError):de.clear_room(19)
                for a,b,n in blocks:ctypes.memmove(b,a,n)
                bulk.append(blocks)
                return 0
            pm._transfer_data=transfer
            target=SimpleNamespace(flashnext_staging=True,dst_attn_tp_size=2,
                requires_dcp_relayout=False,dst_aux_ptrs=[])
            request=SimpleNamespace(room=19,is_dummy=False,mooncake_session_id='destination',
                endpoint='127.0.0.1',dst_port=23,dst_kv_indices=dst,dst_device_kv_indices=None,
                decode_prefix_len=0,required_dst_info_num=1)
            chunk=TransferKVChunk(room=19,prefill_kv_indices=src,index_slice=slice(0,5),
                is_last_chunk=True,prefill_aux_index=0,state_indices=states,num_kv_tokens=257,
                prefill_kv_indices_by_entry=maps,wait_event=Event())
            pm.enable_trace=False;pm._staging_outstanding=defaultdict(int)
            pm.request_status={19:KVPoll.Transferring};pm.check_status=pm.request_status.__getitem__
            pm.transfer_infos={19:{'destination':request}};pm.session_lock=threading.Lock()
            pm.failed_sessions=set();pm._prefill_unique_rank=lambda:0
            pm.decode_kv_args_table={'destination':target};pm.flashnext_staging=pe
            pm._get_dsa_cache_transfer_skip_flags=lambda target:(False,False)
            pm.send_aux=lambda *args:0;pm.req_to_decode_prefix_len={};pm.bootstrap_port=19
            pm._maybe_ack_drained_abort=lambda room:self.assertFalse(de.active)
            def conclude(*,bootstrap_room,status,targets,failure_reason):
                self.assertFalse(de.active,'Success cannot precede scatter completion')
                self.assertEqual(status,KVPoll.Success)
                pm.request_status[bootstrap_room]=status
            pm.conclude_transfer=conclude
            path=Path(__file__).parents[2]/'python/sglang/srt/disaggregation/mooncake/conn.py'
            tree=ast.parse(path.read_text())
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MooncakeKVManager')
            native=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='transfer_worker')
            module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),native],type_ignores=[])
            scope=dict(time=time,KVPoll=KVPoll,logger=logging.getLogger(__name__))
            exec(compile(ast.fix_missing_locations(module),str(path),'exec'),scope)
            class Queue:
                def __init__(self):self.used=False
                def get(self):
                    if self.used:raise SystemExit
                    self.used=True;return chunk
            with self.assertRaises(SystemExit):
                scope['transfer_worker'](pm,queue=Queue(),executor=None)
            self.assertEqual(pm.request_status[19],KVPoll.Success)
            self.assertGreaterEqual(len(bulk),3)
            self.assertEqual(len(bulk),len(pm.flashnext_staging_metrics))
            proofs=[json.loads(p.read_text()) for p in Path(proof_dir).glob('*.json')]
            self.assertEqual(len(proofs),2*len(bulk))
            for chunk_index in range(len(bulk)):
                pair=[r for r in proofs if r['manifest']['chunk_index']==chunk_index]
                self.assertEqual(pair[0]['fields'],pair[1]['fields'])
                self.assertEqual(pair[0]['manifest_sha256'],pair[1]['manifest_sha256'])
            self.assertFalse(de.active)
            # Build full logical views independently of the transport's chunks.
            m,source=p.source_payload(room=19,generation=7,source_rank=0,source_tp=2,
                prompt_tokens=257,token_start=0,token_end=257,kv_indices=src,
                kv_by_entry=maps,state_indices=states,chunk_index=0,last_chunk=True,shallow_boundary=True)
            target=d.destination_payload(manifest=m,kv_indices=dst,state_indices=dst_states,
                decode_prefix_tokens=0,destination_rank=0,destination_tp=2)
            for f in m.fields:
                a,b=source[f.key],target[f.key]
                self.assertTrue(torch.equal(a.tensor[a.rows].view(torch.uint8),b.tensor[b.rows].view(torch.uint8)),f.key)
            for s in (ps,ds):
                a=s.leases.acquire(room=999,nbytes=16384);b=s.leases.acquire(room=998,nbytes=16384)
                self.assertIsNotNone(a);self.assertIsNotNone(b)
                s.leases.release(a);s.leases.release(b)
            # Abort may free D's registered target only after native P ACK.
            de.clear_room(19)
            de.register_room(room=19,kv_indices=dst,state_indices=dst_states,prefix=0)
            abort_manifest=Manifest.build(room=19,generation=8,source_rank=0,source_tp=2,
                prompt_tokens=257,chunk_index=0,last_chunk=True,shallow_count=9,deep_count=8,
                fields=[next(f for f in m.fields if not f.tokens_per_row)])
            de.on_message([HEADER,b'RESERVE',b'aborted',b'127.0.0.1',b'19',abort_manifest.to_bytes()])
            _,lease=de.active['aborted']
            with self.assertRaises(RuntimeError):ds.leases.release(lease)
            self.assertTrue(de.abort_drained(19))
            self.assertFalse(de.active)
            de.on_message([HEADER,b'READY',b'aborted',str(lease.generation).encode()])
            de.clear_room(19)


if __name__=='__main__':unittest.main()
