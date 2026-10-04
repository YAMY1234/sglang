"""Real CPU execution and ownership/wire gates for opt-in shallow publication."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
from types import MethodType, ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from pd_shallow_publication_cpu import (
    worker,batch,request,fake_sender,field_bytes,warmup,RECIPE,IDS,ROOT,
    candidate,exact,graph_module,native,model_runner,http_server,
    HandoffKind,ForwardMode,KVPoll,Event,Transport,np,torch,
)

RECEIPT=[]
CALLS={}


def execution_profile(frame,event,arg):
    if event != 'call':return
    name=frame.f_code.co_filename
    if name.endswith(('gdn_pd_shallow_publication.py','gdn_prefill_exact_tail.py',
                      'gdn_prefill_batch_graph.py','model_runner.py')):
        flag=os.environ.get(candidate.FLAG,'unset')
        key=Path(name).name+':'+frame.f_code.co_qualname
        counts=CALLS.setdefault(flag,{})
        counts[key]=counts.get(key,0)+1


class ShallowPublicationTest(unittest.TestCase):
    def test_default_warmup_real_import_off_on_to_ready(self):
        for flag in ('0','1'):
            with self.subTest(flag=flag),worker(flag) as w:
                result=warmup(w)
                result.killed.assert_not_called();result.ready.assert_called_once_with()
                self.assertEqual(result.status,http_server.ServerStatus.Up)
                self.assertEqual(result.payloads[0]['sampling_params']['max_new_tokens'],8)
                self.assertIsNotNone(getattr(w.owner,'_exact_tail_installed',None))
                self.assertEqual(w.pub is not None,flag=='1')
                RECEIPT.append(dict(flag=flag,warmup_ready=True,
                    publication_submitted=0 if w.pub is None else w.pub.stats['submitted'],
                    captures=w.pool._prefill_batch_graph.stats['captured']))

    def test_same_first_token_all_wire_and_rank_r_checkpoints_rows_1_2_4_8_16(self):
        for rows in (1,2,4,8,16):
            states=[];outputs=[]
            for flag in ('0','1'):
                with self.subTest(rows=rows,flag=flag),worker(flag) as w:
                    out=w.runner.forward(batch(rows))
                    if w.pub is not None:
                        self.assertEqual(w.pub.stats['submitted'],1)
                        self.assertTrue(torch.all(w.pool.count[:,1]==3))
                        self.assertIsNone(w.plans[-1].exact_tail_inputs)
                        self.assertEqual(len(w.pub.pending[1].exact_tail_inputs),24)
                        w.pub.ticket.complete()
                    for i in range(rows):
                        sender=fake_sender();w.handler().before_send(request(sender,i+1,i))
                        sender.send(np.array([1]),state_indices=[[i+1]])
                        self.assertEqual(sender.poll(),KVPoll.Success)
                    states.append(field_bytes(w));outputs.append(out)
                    self.assertTrue(torch.all(w.pool.count[:24,1:rows+1]==w.pool.cfg.r+1))
                    self.assertTrue(torch.all(w.pool.count[24:,1:rows+1]==w.pool.cfg.r))
                    self.assertTrue(torch.all(w.pool.count[:,50:50+rows]==w.pool.cfg.r))
            self.assertEqual(states[0],states[1],rows)
            self.assertTrue(torch.equal(outputs[0],outputs[1]))

    def test_submit_once_and_owned_inputs_survive_exit_and_source_mutation(self):
        with worker('1') as w:
            w.runner.forward(batch())
            plan=w.pub.pending[1]
            self.assertIsNot(plan,w.plans[-1])
            self.assertIsNone(w.plans[-1].exact_tail_inputs)
            self.assertEqual(w.pub.stats['submitted'],1)
            self.assertEqual(w.pub.stats['launched'],1)
            self.assertEqual(len(plan.exact_tail_inputs),24)
            # Reusing the old caller's plan must not alter queued controls.
            w.plans[-1].slots.fill_(80);w.plans[-1].ring_dst.fill_(12)
            w.pub.ticket.complete()
            self.assertTrue(torch.all(w.pool.count[:24,1]==w.pool.cfg.r+1))
            self.assertTrue(torch.all(w.pool.count[:,80]==3))

    def test_disjoint_next_forward_runs_before_prior_bank_join(self):
        with worker('1') as w:
            w.runner.forward(batch());old=w.pub.ticket
            self.assertFalse(old.query())
            # The second trunk runs first, then its submission joins old bank.
            w.runner.forward(batch(offset=1))
            self.assertEqual(w.pub.stats['disjoint_forwards'],1)
            self.assertEqual(w.pub.stats['dependent_forwards'],0)
            self.assertTrue(old.query())
            self.assertFalse(w.pub.ticket.query())
            w.pub.ticket.complete()
            self.assertTrue(torch.all(w.pool.count[:24,1:3]==w.pool.cfg.r+1))

    def test_intersecting_reader_and_slot_reuse_wait_for_queued_publication(self):
        with worker('1') as w:
            w.runner.forward(batch());ticket=w.pub.ticket
            w.pool.pside_join({90})
            self.assertFalse(ticket.query())
            w.pool.pside_join({50})  # tracked prefix reader
            self.assertTrue(ticket.query());self.assertIsNone(w.pub.pending)
            # Recycle only after pending write, not merely after queueing.
            w.pool._reset_slots_fill=MethodType(native.FactoredGDNPool._reset_slots_fill,w.pool)
            native.FactoredGDNPool.reset_slots(w.pool,torch.tensor([1,50]))
            self.assertTrue(torch.all(w.pool.count[:,1]==w.pool.cfg.r))
            w.rp.req_generation[0]+=1
            w.runner.forward(batch());w.pub.ticket.complete()
            self.assertTrue(torch.all(w.pool.count[:24,1]==w.pool.cfg.r+1))

    def test_unknown_scope_empty_singleton_mixed_tbo_and_large_batches_fallback(self):
        with worker('1') as w:
            good=batch()
            self.assertIsNotNone(candidate.select_batch(w.runner,good))
            for change in (dict(batch_size=32),dict(extend_seq_lens_cpu=[1]),
                    dict(extend_seq_lens_cpu=None),dict(twinstar_prompt_final=None),
                    dict(forward_mode=ForwardMode.MIXED),dict(can_run_tbo=True),
                    dict(tbo_split_seq_index=0),dict(tbo_parent_token_range=(0,1)),
                    dict(spec_info=object()),dict(_pfactor_agg_contract=True),
                    dict(_pfactor_legacy_mixed=True)):
                altered=copy.copy(good);vars(altered).update(change)
                self.assertIsNone(candidate.select_batch(w.runner,altered),change)
            w.rp.mamba_v2p_table=torch.arange(32)
            self.assertIsNone(candidate.select_batch(w.runner,good))

    def test_disabled_and_non_pc_roles_do_not_install_or_wrap(self):
        for flag in ('0','1'):
            for arm,role in [('S','prefill'),('P','prefill'),('C','prefill'),('PC','decode'),('PC','null')]:
                with self.subTest(flag=flag,arm=arm,role=role),worker(flag,arm=arm,role=role) as w:
                    old=w.runner.forward;handlers=dict(w.rp.pd_state_handoffs)
                    self.assertFalse(candidate.install(w.runner))
                    self.assertIs(w.runner.forward,old)
                    self.assertEqual(w.rp.pd_state_handoffs,handlers)
                    self.assertIsNone(getattr(w.pool,'_pd_shallow_publication',None))

    def test_reject_missing_dependencies_without_changing_baseline_off(self):
        keys=['SGLANG_GDN_PD_BATCH_PUBLISH_DEFERRED','SGLANG_GDN_PD_PUBLISH_JOIN_OFFLOAD']
        for key in keys:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'shallow deferred'):
                with worker('1',recipe={key:'0'}):pass
            with worker('0',recipe={key:'0'}) as w:self.assertIsNone(w.pub)

    def test_default_off_exact_publish_reversion_is_caught_by_order_gate(self):
        # Load the entire frozen old module, not an AST-extracted function.
        path=Path(os.environ['PFACTOR072_BASE'])/'python/sglang/srt/mem_cache/gdn_prefill_exact_tail.py'
        spec=importlib.util.spec_from_file_location('sglang.srt.mem_cache._pfactor072_base_exact',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with worker('1') as w,patch.object(exact.ExactTailTransaction,'publish',module.ExactTailTransaction.publish):
            w.runner.forward(batch())
            with self.assertRaises(AssertionError):self.assertEqual(w.pub.stats['submitted'],1)
            self.assertTrue(torch.all(w.pool.count[:24,1]==w.pool.cfg.r+1))
            self.assertIsNone(w.pub.pending)

    def test_missing_install_reverse_gate_never_claims_enabled(self):
        with patch.object(candidate,'install',return_value=False),worker('1') as w:
            with self.assertRaises(AssertionError):self.assertIsNotNone(w.pub)
            self.assertIsNone(w.pub)

    def test_legacy_sender_wait_and_real_validator_are_not_bypassed(self):
        with worker('1') as w:
            w.runner.forward(batch());ticket=w.pub.ticket
            w.handler().before_send(request(NS()))
            self.assertTrue(ticket.query())
            w.pool.count[0,1]=w.pool.cfg.r
            with self.assertRaisesRegex(RuntimeError,'phases differ'):
                w.handler().before_send(request(NS()))

    def test_bad_count_valid_or_generation_blocks_fake_success(self):
        for bad in ('count','valid','generation','event'):
            with self.subTest(bad=bad),worker('1') as w:
                w.runner.forward(batch());sender=fake_sender()
                w.handler().before_send(request(sender))
                w.pub.ticket.complete()
                if bad=='count':w.pool.count[0,1]=w.pool.cfg.r
                elif bad=='valid':w.rp.pd_boundary_state.valid[1]=0
                elif bad=='generation':w.rp.req_generation[0]+=1
                else:w.pub.ticket.query=lambda:False
                with self.assertRaises(RuntimeError):sender.send(np.array([1]),state_indices=[[1]])
                self.assertFalse(sender.has_sent)
                self.assertEqual(sender.poll(),KVPoll.Failed)

    def test_mooncake_real_queue_worker_and_memmove_wire_off_on(self):
        receipts=[]
        for flag in ('0','1'):
            with worker(flag) as w:
                wire=Transport(w.pool)
                state=w.rp.pd_boundary_state
                # The real state-copy worker carries boundary bytes alongside
                # all 144 factor entries; unchanged conv/KV sentinels are also
                # registered and checked so publication cannot replace them.
                extra=[state.hidden,state.position,state.valid,
                       torch.arange(100*8,dtype=torch.float32).reshape(100,8),
                       torch.arange(100*4,dtype=torch.bfloat16).reshape(100,4)]
                wire.source.extend(extra);wire.dest.extend(torch.full_like(t,-7) for t in extra)
                w.runner.forward(batch())
                w.handler().before_send(request(wire.sender))
                wire.enqueue();wire.start()
                try:
                    if w.pub is not None:
                        self.assertTrue(w.pub.ticket.entered.wait(2))
                        self.assertEqual(wire.sent,[])
                        w.pub.ticket.complete()
                    self.assertTrue(wire.finished.wait(5))
                    self.assertEqual(wire.manager.request_status[7],KVPoll.Success)
                    expected=[t[1].contiguous().view(torch.uint8).numpy().tobytes() for t in wire.source]
                    self.assertEqual(wire.received(),expected)
                    receipts.append(wire.received())
                finally:wire.stop()
        self.assertEqual(receipts[0],receipts[1])
        RECEIPT.append(dict(wire_fields=len(receipts[0]),wire_byte_failures=0,
                            real_mooncake_sender_worker=True,transport='CPU memmove'))

    def test_wait_is_idempotent_and_receive_delegates_unchanged(self):
        with worker('1') as w:
            w.runner.forward(batch());sender=fake_sender();req=request(sender)
            w.handler().before_send(req);fence=sender._state_handoff_fence
            w.pub.ticket.complete();producer=Event(ready=True)
            fence.wait(producer);n=len(w.pub.ticket.wait_threads)
            fence.wait(producer)
            self.assertEqual(len(w.pub.ticket.wait_threads),n)
            original=w.handler().original
            with patch.object(original,'prepare_receive',return_value='prepare') as prepare,patch.object(original,'commit_receive',return_value='commit') as commit:
                self.assertEqual(w.handler().prepare_receive(req),'prepare')
                self.assertEqual(w.handler().commit_receive(req),'commit')
                prepare.assert_called_once_with(req);commit.assert_called_once_with(req)

    def test_all_sender_classes_execute_same_handoff_contract_off_on(self):
        # Import actual sender classes. Only the unavailable vendor SDK gets
        # inert type declarations; no sender method is extracted or replaced.
        from sglang.srt.disaggregation.nixl.conn import NixlKVSender
        try:
            import mori.io
        except ImportError:
            sdk=ModuleType('mori');sdk.__path__=[]
            cpp=ModuleType('mori.cpp');io=ModuleType('mori.io')
            cpp.TransferStatus=type('TransferStatus',(),{})
            for name in ('BackendType','EngineDesc','IOEngine','IOEngineConfig','MemoryDesc',
                         'MemoryLocationType','PollCqMode','RdmaBackendConfig','StatusCode'):
                setattr(io,name,type(name,(),{}))
            sdk.cpp,sdk.io=cpp,io
            sys.modules.update({'mori':sdk,'mori.cpp':cpp,'mori.io':io})
        from sglang.srt.disaggregation.mori.conn import MoriKVSender
        from sglang.srt.disaggregation.ascend.conn import AscendKVSender
        for flag in ('0','1'):
            for cls in (NixlKVSender,MoriKVSender):
                with self.subTest(flag=flag,sender=cls.__name__),worker(flag) as w:
                    sender=cls.__new__(cls)
                    sender._send_failed=False;sender._transfer_start_time=None
                    sender.bootstrap_room=7;sender.chunk_id=0;sender.aux_index=0;sender.has_sent=False
                    sender._prepare_send_indices=lambda ids,states:(ids,slice(0,len(ids)),True,False)
                    sender._record_transfer_indices=Mock()
                    observed=[]
                    def transfer(*args,**kwargs):
                        if w.pub is not None:self.assertIsNone(w.pub.pending)
                        self.assertTrue(torch.all(w.pool.count[:24,1]==w.pool.cfg.r+1))
                        observed.append(field_bytes(w))
                    sender.kv_mgr=NS(add_transfer_request=transfer,
                        _should_skip_cp_replicated_state_transfer=lambda:False)
                    w.runner.forward(batch());w.handler().before_send(request(sender))
                    sender.send(np.array([1]),state_indices=[[1]])
                    self.assertEqual(len(observed),1)
                    RECEIPT.append(dict(sender=cls.__name__,flag=flag,real_send=True,
                                        synchronous_fallback=True))
            with self.subTest(flag=flag,sender='AscendKVSender'),worker(flag) as w:
                wire=Transport(w.pool)
                sender=AscendKVSender.__new__(AscendKVSender)
                vars(sender).update(vars(wire.sender));wire.sender=sender
                # Do not retain the fixture's method bound to its old NS.
                del sender.set_state_handoff_fence
                w.runner.forward(batch());w.handler().before_send(request(sender))
                wire.enqueue();wire.start()
                try:
                    if w.pub is not None:
                        self.assertTrue(w.pub.ticket.entered.wait(2));self.assertEqual(wire.sent,[])
                        w.pub.ticket.complete()
                    self.assertTrue(wire.finished.wait(5))
                    self.assertEqual(wire.manager.request_status[7],KVPoll.Success)
                finally:wire.stop()
                RECEIPT.append(dict(sender='AscendKVSender',flag=flag,real_send=True,
                                    transport='inherited Mooncake CPU memmove'))

    def test_nonfinal_checkpoint_dense_ring_is_byte_identical(self):
        results=[]
        for flag in ('0','1'):
            with worker(flag) as w:
                w.runner.forward(batch(final=False))
                w.pool.pside_join({1})
                self.assertTrue(torch.all(w.pool.count[:,1]==w.pool.cfg.r))
                results.append(field_bytes(w))
        self.assertEqual(*results)

    def test_mixed_finality_keeps_nonfinal_dense_continuation_and_tail_subset(self):
        results=[]
        for flag in ('0','1'):
            with worker(flag) as w:
                fb=batch(2);fb.twinstar_prompt_final=[False,True]
                w.runner.forward(fb);w.pool.pside_join({1,2})
                self.assertTrue(torch.all(w.pool.count[:,1]==w.pool.cfg.r))
                self.assertTrue(torch.all(w.pool.count[:24,2]==w.pool.cfg.r+1))
                self.assertTrue(torch.all(w.pool.count[24:,2]==w.pool.cfg.r))
                w.handler().before_send(request(NS(),2,1))
                results.append(field_bytes(w))
        self.assertEqual(*results)

    def test_capture_bypasses_wrapper_without_selection_or_publication(self):
        with worker('1') as w:
            # A decode capture invokes the existing model route. It must not
            # select or mutate publication scope; exact-tail skips decode.
            fb=batch();fb.forward_mode=ForwardMode.DECODE
            w.capture_mode.return_value=True
            with patch.object(candidate,'select_batch',side_effect=AssertionError('capture selected')):
                # Original model body is a hardware backend leaf for decode.
                w.owner.forward=lambda ids,pos,b:ids
                self.assertTrue(torch.equal(w.runner.forward(fb),fb.input_ids))
            self.assertEqual(w.pub.stats['submitted'],0)

    def test_entire_frozen_transaction_off_outputs_and_wire_equal(self):
        path=Path(os.environ['PFACTOR072_BASE'])/'python/sglang/srt/mem_cache/gdn_prefill_exact_tail.py'
        spec=importlib.util.spec_from_file_location('sglang.srt.mem_cache._pfactor072_frozen',path)
        frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
        current=exact.ExactTailTransaction
        for rows in (1,8):
            observed=[]
            for cls in (frozen.ExactTailTransaction,current):
                with patch.object(exact,'ExactTailTransaction',cls),worker('0') as w:
                    output=w.runner.forward(batch(rows))
                    observed.append((output.numpy().tobytes(),field_bytes(w)))
            self.assertEqual(*observed)

    def test_enqueue_failure_retains_inputs_and_blocks_every_reader(self):
        with worker('1') as w:
            with patch.object(w.runtime,'launch_after_forward',side_effect=RuntimeError('enqueue failed')):
                with self.assertRaisesRegex(RuntimeError,'enqueue failed'):w.runner.forward(batch())
            self.assertIsNotNone(w.pub.pending);self.assertIsNotNone(w.pub.failed)
            for call in (lambda:w.pool.pside_join({1}),
                         lambda:w.handler().before_send(request(fake_sender())),
                         lambda:w.runner.forward(batch(offset=1))):
                with self.assertRaisesRegex(RuntimeError,'not publishable'):call()

    def test_request_cancel_or_reuse_during_worker_wait_never_sends(self):
        with worker('1') as w:
            wire=Transport(w.pool)
            w.runner.forward(batch());w.handler().before_send(request(wire.sender))
            wire.enqueue();wire.start()
            try:
                self.assertTrue(w.pub.ticket.entered.wait(2))
                w.rp.req_generation[0]+=1
                w.pub.ticket.complete()
                self.assertTrue(wire.finished.wait(5))
                self.assertEqual(wire.manager.request_status[7],KVPoll.Failed)
                self.assertEqual(wire.sent,[])
            finally:wire.stop()


if __name__=='__main__':
    sys.setprofile(execution_profile)
    result=unittest.main(verbosity=2,exit=False).result
    sys.setprofile(None)
    required=['gdn_pd_shallow_publication.py:'+name for name in (
        'select_batch','ShallowPublication.selected','ShallowPublication.submit_exact',
        'ShallowPublication.start_after_forward','ShallowTransferFence.__init__',
        'ShallowTransferFence.wait','ShallowHandoff.__init__','ShallowHandoff.before_send',
        'ShallowHandoff.prepare_receive','ShallowHandoff.commit_receive',
        'install','install.<locals>.forward')]
    for flag in ('0','1'):
        for key in ['model_runner.py:ModelRunner.init_cuda_graphs']+[
                'gdn_prefill_exact_tail.py:ExactTailTransaction.'+n for n in ('__enter__','add','publish')]:
            assert CALLS[flag].get(key,0)>0,(flag,key)
    for key in required:assert CALLS['1'].get(key,0)>0,key
    assert CALLS['0'].get('gdn_pd_shallow_publication.py:install',0)>0
    out=Path(os.environ['LOWC_OUT'])/'shallow072-unit.json'
    out.write_text(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,
        skipped=len(result.skipped),rows=RECEIPT,gpu_needle='NOT_RUN; dattr #055',
        cpu_backends=True,executed_functions=CALLS),indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
