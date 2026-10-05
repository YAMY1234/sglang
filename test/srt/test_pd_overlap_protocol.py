"""Real-import install/warmup/FIFO/fence CPU gates; no CUDA/RDMA admission."""
import copy
import os
import sys
import unittest
from contextlib import ExitStack
from types import MethodType, ModuleType, SimpleNamespace as NS
from unittest.mock import Mock, patch

import pd_shallow_publication_cpu as cpu
from sglang.srt.mem_cache import gdn_pd_overlap as protocol
from sglang.srt.mem_cache import gdn_pd_publication as publication
from sglang.srt.mem_cache import gdn_pd_shallow_publication as shallow
from sglang.srt.mem_cache import gdn_prefill_exact_tail as exact
from sglang.srt.disaggregation.state_handoff import FactorStateHandoff
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from test_pd_publication_records import forward, versions, EnqueueRuntime

FLAG='SGLANG_GDN_PD_PUBLISH_OVERLAP_OK'
torch=cpu.torch


def bind(w, fb, requests):
    record=getattr(fb,'pd_publication_record',None)
    if record is not None:
        protocol.bind_result_record(NS(forward_iter=record.batch_id,reqs=requests,req_to_token_pool=w.rp),
                                    NS(pd_publication_record=record))
    return record


class OverlapProtocol(unittest.TestCase):
    def test_real_install_matrix_and_default_warmup_ready(self):
        for flag,overlap in (('0',False),('0',True),('1',False),('1',True)):
            with self.subTest(flag=flag,overlap=overlap):
                if flag=='0' and overlap:
                    with self.assertRaisesRegex(ValueError,'isolated strict'):
                        with cpu.worker(recipe={FLAG:flag},overlap=overlap): pass
                    continue
                with cpu.worker(recipe={FLAG:flag},overlap=overlap) as w:
                    self.assertEqual(protocol.protocol_ready(w.runner),flag=='1')
                    receipt=cpu.warmup(w)
                    receipt.killed.assert_not_called();receipt.ready.assert_called_once_with()
                    self.assertEqual(receipt.status,cpu.http_server.ServerStatus.Up)
                    if flag=='1':
                        self.assertIs(w.pub.records,w.rp._pd_publication_records)
                        self.assertFalse(w.pub.records.states)

    def test_wire_outputs_and_counts_identical_rows_and_finality(self):
        for rows,finality in ((1,[True]),(2,[True,True]),(4,[True]*4),
                              (8,[True]*8),(16,[True]*16),(2,[False,True]),(1,[False])):
            snapshots=[]
            for flag,overlap in (('0',False),('1',False),('1',True)):
                with self.subTest(rows=rows,finality=finality,flag=flag,overlap=overlap), \
                     cpu.worker(recipe={FLAG:flag},overlap=overlap) as w:
                    fb=cpu.batch(rows);fb.twinstar_prompt_final=finality
                    output=w.runner.forward(fb)
                    reqs=[cpu.request(cpu.fake_sender(),i+1,i) for i in range(rows)]
                    bind(w,fb,reqs)
                    w.pub.ticket.complete()
                    for req,final in zip(reqs,finality):
                        if final:
                            w.handler().before_send(req)
                            req.disagg_kv_sender.send(cpu.np.array([1]),state_indices=[[int(req.kv.mamba_pool_idx)]])
                            self.assertEqual(req.disagg_kv_sender.poll(),cpu.KVPoll.Success)
                    snapshots.append((output.numpy().tobytes(),cpu.field_bytes(w)))
                    for i,final in enumerate(finality):
                        self.assertTrue(torch.all(w.pool.count[:24,i+1]==w.pool.cfg.r+int(final)))
                        self.assertTrue(torch.all(w.pool.count[24:,i+1]==w.pool.cfg.r))
            self.assertEqual(snapshots[0],snapshots[1]);self.assertEqual(snapshots[0],snapshots[2])

    def test_prepare_gate_rejects_unaudited_writes_before_install(self):
        mutations=[lambda w:setattr(w.rp,'enable_mamba_extra_buffer',False),
                   lambda w:setattr(w.rp,'mamba_ckpt_pool',object()),
                   lambda w:setattr(w.runner.server_args,'enable_hierarchical_cache',True),
                   lambda w:setattr(w.rp,'mamba_v2p_table',object()),
                   lambda w:setattr(w.pool,'host_sync_free',False),
                   lambda w:setattr(w.runner.server_args,'pp_size',2),
                   lambda w:setattr(w.runner.server_args,'speculative_algorithm','EAGLE'),
                   lambda w:setattr(w.runner.server_args,'is_embedding',True)]
        for mutate in mutations:
            with self.assertRaisesRegex(ValueError,'complete deferred P recipe'):
                with cpu.worker(recipe={FLAG:'1'},overlap=True,setup=mutate):pass
        for name in ('SGLANG_GDN_PREFILL_EXACT_TAIL_BATCH','SGLANG_GDN_PREFILL_COMMIT_GRAPH',
                     'SGLANG_GDN_PD_BATCH_PUBLISH_DEFERRED','SGLANG_GDN_PD_PUBLISH_JOIN_OFFLOAD',
                     'SGLANG_GDN_PD_SHALLOW_PUBLISH_DEFERRED'):
            with self.subTest(missing=name),self.assertRaisesRegex(ValueError,'complete deferred P recipe'):
                with cpu.worker(recipe={FLAG:'1',name:'0'},overlap=True):pass
        with self.assertRaisesRegex(ValueError,'complete deferred P recipe'):
            with cpu.worker(recipe={FLAG:'1','SGLANG_FLASHNEXT_FACTOR_GUARD_ABORT':'1'},overlap=True):pass
        with cpu.worker(recipe={FLAG:'1'},overlap=True) as w:
            old=versions(w)
            with self.assertRaisesRegex(RuntimeError,'retraction state restore'):
                w.pool.load_cpu_slots((torch.zeros(1),),torch.tensor([1]))
            self.assertEqual(versions(w),old)
            w.pool.load_cpu_slots(None,torch.tensor([1]))

    def test_all_three_gates_require_preinstalled_protocol_reverse(self):
        # Revert just preparation: key=1 is insufficient for the three gates.
        with patch.object(protocol,'prepare_protocol',return_value=None):
            with self.assertRaisesRegex(ValueError,'isolated strict'):
                with cpu.worker(recipe={FLAG:'1'},overlap=True):pass
        with cpu.worker(recipe={FLAG:'1'}) as w, \
             patch('sglang.srt.runtime_context.get_schedule',return_value=NS(disable_overlap_schedule=False)):
            manager=w.rp._pd_publication_records
            manager.prepared=False
            with self.assertRaisesRegex(ValueError,'disabled overlap'):
                publication.install(w.pool,w.runner)
            saved=w.pool._pd_shallow_publication
            del w.pool._pd_shallow_publication
            try:
                with self.assertRaisesRegex(ValueError,'isolated native PC'):
                    shallow.install(w.runner)
            finally:w.pool._pd_shallow_publication=saved;manager.prepared=True
            with patch.object(w.pool,'_pd_batch_publication',None):
                with self.assertRaisesRegex(RuntimeError,'no record-aware publisher'):
                    protocol.verify_protocol(w.runner)

    def test_non_P_and_dense_arms_do_not_install_or_wrap(self):
        for arm,role in (('S','prefill'),('P','prefill'),('PC','decode'),('PC','null')):
            with self.subTest(arm=arm,role=role),cpu.worker(arm=arm,role=role,recipe={FLAG:'1'}) as w:
                old=w.rp.mamba_allocator.free.__func__
                protocol.prepare_protocol(w.runner);protocol.verify_protocol(w.runner)
                self.assertFalse(protocol.protocol_ready(w.runner))
                self.assertNotIn('_pd_publication_records',vars(w.rp))
                self.assertIs(w.rp.mamba_allocator.free.__func__,old)

    def test_record_capture_bypass_does_not_add_event_or_scope(self):
        with cpu.worker(recipe={FLAG:'1'},overlap=True) as w:
            fb=cpu.batch();fb.forward_mode=cpu.ForwardMode.DECODE
            w.capture_mode.return_value=True
            w.owner.forward=lambda ids,pos,b:ids
            with patch.object(shallow,'select_batch',side_effect=AssertionError('capture select')):
                self.assertTrue(torch.equal(w.runner.forward(fb),fb.input_ids))
            self.assertEqual(w.pub.records.stats['publications'],0)
            # Same capture rule for the independent P48 join wrapper.
            runner=NS(forward=Mock(return_value=fb.input_ids))
            publication.install_forward_join(runner,w.pool)
            self.assertTrue(torch.equal(runner.forward(fb),fb.input_ids))
            self.assertEqual(w.pub.records.stats['publications'],0)

    def test_chunk_replacement_cancel_and_epoch_failure(self):
        with cpu.worker(recipe={FLAG:'1'},overlap=True,runtime=EnqueueRuntime()) as w:
            cpu.warmup(w)
            req,a,ra,slot,track=forward(w,1)
            protocol.bind_result_record(a,ra)
            fb=cpu.batch(offset=req.kv.req_pool_idx);fb.mamba_track_indices=track
            fb.pd_publication_batch_id=2
            w.runner.forward(fb)
            rb=NS(pd_publication_record=fb.pd_publication_record)
            b=NS(forward_iter=2,reqs=[req],req_to_token_pool=w.rp)
            protocol.bind_result_record(b,rb)
            self.assertNotIn((req.kv.req_pool_idx,int(w.rp.req_generation[req.kv.req_pool_idx])),
                             w.pub.records.states[1].remaining)
            req.disagg_kv_sender=cpu.fake_sender();w.handler().before_send(req)
            fence=req.disagg_kv_sender._state_handoff_fence
            w.rp.req_generation[req.kv.req_pool_idx]+=1
            with self.assertRaisesRegex(RuntimeError,'generation changed'):
                fence.wait(cpu.Event(ready=True))

    def test_mooncake_and_ascend_worker_memmove_use_own_record(self):
        from sglang.srt.disaggregation.ascend.conn import AscendKVSender
        for ascend in (False,True):
            with self.subTest(ascend=ascend),cpu.worker(recipe={FLAG:'1'},overlap=True) as w:
                wire=cpu.Transport(w.pool)
                state=w.rp.pd_boundary_state
                extra=[state.hidden,state.position,state.valid,
                       torch.arange(100*8,dtype=torch.float32).reshape(100,8),
                       torch.arange(100*4,dtype=torch.bfloat16).reshape(100,4)]
                wire.source.extend(extra);wire.dest.extend(torch.full_like(t,-7) for t in extra)
                if ascend:
                    sender=AscendKVSender.__new__(AscendKVSender)
                    vars(sender).update(vars(wire.sender));wire.sender=sender
                    del sender.set_state_handoff_fence
                fb=cpu.batch();w.runner.forward(fb)
                req=cpu.request(wire.sender);record=bind(w,fb,[req])
                w.handler().before_send(req)
                self.assertIs(wire.sender._state_handoff_fence.record,record)
                wire.enqueue();wire.start()
                try:
                    record.publication_done.complete()
                    self.assertTrue(wire.finished.wait(5))
                    self.assertEqual(wire.manager.request_status[7],cpu.KVPoll.Success)
                    expected=[t[1].contiguous().view(torch.uint8).numpy().tobytes() for t in wire.source]
                    self.assertEqual(wire.received(),expected)
                finally:wire.stop()

    def test_nonfenced_senders_use_record_local_synchronous_fallback(self):
        from sglang.srt.disaggregation.nixl.conn import NixlKVSender
        try:import mori.io
        except ImportError:
            sdk=ModuleType('mori');sdk.__path__=[];cpp=ModuleType('mori.cpp');io=ModuleType('mori.io')
            cpp.TransferStatus=type('TransferStatus',(),{})
            for name in ('BackendType','EngineDesc','IOEngine','IOEngineConfig','MemoryDesc',
                         'MemoryLocationType','PollCqMode','RdmaBackendConfig','StatusCode'):
                setattr(io,name,type(name,(),{}))
            sdk.cpp,sdk.io=cpp,io;sys.modules.update({'mori':sdk,'mori.cpp':cpp,'mori.io':io})
        from sglang.srt.disaggregation.mori.conn import MoriKVSender
        for cls in (NixlKVSender,MoriKVSender):
            with self.subTest(sender=cls.__name__),cpu.worker(recipe={FLAG:'1'},overlap=True) as w:
                sender=cls.__new__(cls)
                sender._send_failed=False;sender._transfer_start_time=None
                sender.bootstrap_room=7;sender.chunk_id=0;sender.aux_index=0;sender.has_sent=False
                sender._prepare_send_indices=lambda ids,states:(ids,slice(0,len(ids)),True,False)
                sender._record_transfer_indices=Mock();sent=[]
                sender.kv_mgr=NS(add_transfer_request=lambda *a,**kw:sent.append(cpu.field_bytes(w)),
                                _should_skip_cp_replicated_state_transfer=lambda:False)
                fb=cpu.batch();w.runner.forward(fb);req=cpu.request(sender);record=bind(w,fb,[req])
                record.publication_done.complete()
                with patch.object(w.pub,'join',side_effect=AssertionError('global pool join')):
                    w.handler().before_send(req)
                sender.send(cpu.np.array([1]),state_indices=[[1]])
                self.assertEqual(len(sent),1)

    def test_enqueue_error_blocks_raw_free_even_without_result_record(self):
        with cpu.worker(recipe={FLAG:'1'},overlap=True) as w:
            from sglang.srt.managers.schedule_batch import Req
            from sglang.srt.sampling.sampling_params import SamplingParams
            req=Req('enqueue-error','cpu',[1]*8,SamplingParams(max_new_tokens=1))
            w.rp.alloc([req]);index=req.kv.req_pool_idx
            slot=w.rp.mamba_allocator.alloc(1)
            w.rp.req_index_to_mamba_index_mapping[index]=slot[0]
            with patch.object(w.runtime,'launch_after_forward',side_effect=RuntimeError('enqueue failed')):
                with self.assertRaisesRegex(RuntimeError,'enqueue failed'):
                    w.runner.forward(cpu.batch(offset=index))
            self.assertFalse(w.pub.records.states)  # after_forward was not reached
            self.assertIsNotNone(w.pub.pending)
            operations=[lambda:w.rp.free(req),lambda:w.rp.free_rows([index]),
                        lambda:w.rp.mamba_allocator.free(slot),lambda:w.rp.alloc_rows(1),
                        lambda:w.rp.mamba_allocator.alloc(1),lambda:w.rp.clear(),
                        lambda:w.rp.mamba_allocator.clear()]
            for operation in operations:
                with self.assertRaisesRegex(RuntimeError,'allocator leases retained'):operation()
            self.assertEqual(req.kv.req_pool_idx,index)
            self.assertNotIn(index,w.rp.free_slots)
            self.assertNotIn(int(slot[0]),w.rp.mamba_allocator.free_slots.tolist())

    def test_failed_fake_send_retires_only_after_own_publication(self):
        with cpu.worker(recipe={FLAG:'1'},overlap=True) as w:
            req,batch,result,slot,_=forward(w,1)
            protocol.bind_result_record(batch,result)
            req.disagg_kv_sender=cpu.fake_sender()
            w.handler().before_send(req)
            result.pd_publication_record.publication_done.complete()
            w.pool.count[0,slot]=w.pool.cfg.r  # violate the unchanged r+1 wire contract
            with self.assertRaises(RuntimeError):
                req.disagg_kv_sender.send(cpu.np.array([1]),state_indices=[[int(slot[0])]])
            self.assertFalse(req.disagg_kv_sender.has_sent)
            self.assertEqual(req.disagg_kv_sender.poll(),cpu.KVPoll.Failed)
            w.rp.mamba_allocator.free(slot)
            w.rp.free(req)
            self.assertFalse(w.pub.records.states)
            self.assertFalse(w.pub.records.deferred_frees)
            self.assertIn(int(slot[0]),w.rp.mamba_allocator.free_slots.tolist())

    def test_P48_record_fence_and_capture_real_install(self):
        # Use the genuine modules/installers with a full-depth CPU model leaf.
        recipe={FLAG:'1','TWINSTAR_PD_FACTOR_ONLY_TAIL':'1','SGLANG_GDN_PREFILL_AGG_CONTRACT':'1',
                shallow.FLAG:'0'}
        with cpu.worker(arm='C',recipe=recipe,overlap=True) as w:
            from twinstar_sgl import pd_factor_only
            from sglang.srt.model_executor.forward_batch_info import ForwardBatch
            from sglang.srt.managers.schedule_batch import ScheduleBatch
            from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
            for cls,name in ((ForwardBatch,'init_new'),
                             (ScheduleBatch,'_mamba_radix_cache_v2_req_prepare_for_extend'),
                             (GDNAttnBackend,'init_forward_metadata'),(FactorStateHandoff,'before_send')):
                w.stack.enter_context(patch.object(cls,name,cls.__dict__[name]))
            w.stack.enter_context(patch.object(ForwardBatch,'_pfactor_agg_contract_installed',False,create=True))
            # Actual frozen helper contracts, not synthetic __wrapped__ markers.
            pd_factor_only.install_batch_contract(ForwardBatch,ScheduleBatch)
            pd_factor_only.install_metadata_contract(GDNAttnBackend)
            pd_factor_only.install_handoff_contract(FactorStateHandoff)
            owner=w.owner
            owner.p_layer_ids=list(range(48));owner.fullstack_v3_latent=False
            owner.fullstack.update(prefill_saving_policy='kv-and-ssm',qsa_code='off')
            owner.emitters={str(i):object() for i in range(31,48)};owner._emit_ids=lambda:[]
            owner.config=NS(layers_block_type=['attention' if i%4==3 else 'linear_attention' for i in range(48)])
            owner.model.model.layers=[object() for _ in range(48)]
            layers=[]
            for lid in cpu.IDS:
                layer=cpu.RadixLinearAttention.__new__(cpu.RadixLinearAttention)
                torch.nn.Module.__init__(layer);vars(layer).update(vars(cpu.layer(lid)));layers.append(layer)
            owner.model.model.modules=lambda:iter(layers)
            plan=cpu.make_plan(w.pool)
            backend=NS(forward_metadata=NS(factored_extend=plan))
            with forward_context(ForwardContext(attn_backend=NS(linear_attn_backend=backend))):
                def core(ids,pos,fb,**kw):
                    for lid in cpu.IDS:
                        dense=torch.full((1,2,16,16),(lid+1)*0.015625)+torch.eye(16)[None,None]*0.5
                        cpu.native.FactoredGDNPool.commit_extend_batched(w.pool,lid,plan,dense)
                    return ids.float()*2+pos.float()
                owner.model.forward=core
                w.init()  # actual imported ModelRunner -> all three installs
                pub=w.pool._pd_batch_publication;pub.runtime=cpu.Runtime()
                self.assertTrue(protocol.protocol_ready(w.runner))
                for buffers,graph in w.pool._agg_prefill_graph.entries.values():
                    graph.body=lambda buffers=buffers:buffers.evaluate(cpu.native.factorize_layers)
                fb=cpu.batch();fb._pfactor_agg_contract=True
                w.runner.forward(fb);req=cpu.request(cpu.fake_sender());record=bind(w,fb,[req])
                self.assertIsNotNone(record.publication_done)
                record.publication_done.complete()
                req._pfactor_agg_contract=True
                FactorStateHandoff(w.pool).before_send(req)
                fence=req.disagg_kv_sender._state_handoff_fence
                self.assertIs(fence.record,record)
                fence.wait(cpu.Event(ready=True));self.assertTrue(fence.validated)
                self.assertTrue(torch.all(w.pool.count[:,1]==w.pool.cfg.r))
                req.disagg_kv_sender=NS()
                with patch.object(pub,'join',side_effect=AssertionError('global pool join')):
                    FactorStateHandoff(w.pool).before_send(req)


if __name__=='__main__':unittest.main()
