"""Startup restoration and both real split routes must reach prewarmed tails."""

import ast
from collections import namedtuple
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor import gdn_prefill_model_split as split
from sglang.srt.model_executor import gdn_prefill_tail_graph as tail
import test_gdn_prefill_model_split as fixtures

module = fixtures.module


def decode(batch):
    batch.forward_mode = NS(is_decode=lambda: True)
    batch.spec_info = batch.input_embeds = batch.replace_embeds = None
    return batch


class TailGraphTest(unittest.TestCase):
    def test_full_depth_actual_split_keeps_native_ids_hc_and_single_recurrent_step(self):
        c = fixtures.ModelSplitTest().setup_case()
        decode(c.tail)
        body = NS(last_hc_hidden_states=None)

        def forward(ids, positions, batch):
            if batch is c.prefix:
                c.pool.collected.extend(range(36))
            else:
                self.assertEqual(c.pool.count.tolist(), [8]*36)
                c.pool.count += 1
            body.last_hc_hidden_states = ids[:, None]*2
            return ids[:, None]*10

        body.forward = forward
        c.owner.model = NS(model=body)
        adapter = tail.TailBody(c.owner)
        graph = NS(communication=NS(check=Mock()), execute_tail=Mock(side_effect=
            lambda batch: adapter.forward(batch.input_ids,batch.positions,batch)))
        c.owner._gdn_prefill_tail_graph = graph
        with patch.dict('os.environ',{tail.FLAG:'1'}), patch.object(split,'_input_scope',return_value=nullcontext()):
            with split._split_body(c.owner,c.batch,c.prefix,c.prefix_indices,c.tail,c.tail_indices,c.backend):
                result = body.forward(c.ids,c.positions,c.batch)
        torch.testing.assert_close(result,c.ids[:,None]*10)
        torch.testing.assert_close(body.last_hc_hidden_states,c.ids[:,None]*2)
        self.assertEqual(c.pool.count.tolist(),[9]*36)
        graph.execute_tail.assert_called_once_with(c.tail)
        self.assertIs(body.forward,forward)

    def test_shallow_actual_split_graph_keeps_h31_and_does_not_run_omitted_layers(self):
        c = fixtures.ModelSplitTest().setup_case()
        decode(c.tail)
        c.owner.pd_shallow_role = 'prefill'
        c.owner.model = NS(model=NS(forward=Mock(side_effect=AssertionError('full48 forbidden'))))
        adapter = tail.TailBody(c.owner)

        def shallow(owner,batch):
            if batch is c.prefix:c.pool.collected.extend(range(24))
            else:
                self.assertEqual(c.pool.count.tolist(),[8]*36)
                c.pool.count[:24] += 1
            return batch.input_ids[:,None].float(),batch.input_ids[:,None].float()

        graph=NS(communication=NS(check=Mock()),execute_tail=Mock(side_effect=
            lambda batch:adapter.forward(batch.input_ids,batch.positions,batch)))
        c.owner._gdn_prefill_tail_graph=graph
        pd=module('twinstar_sgl.pd_shallow',boundary_inputs=lambda *a:(c.tail,c.tail_indices),capture_extend_boundary=Mock())
        modules={'twinstar_sgl':module('twinstar_sgl',pd_shallow=pd),pd.__name__:pd,
            'twinstar_sgl.pd_final_metadata':module('metadata',initialize_shallow=lambda b,p:b.init_forward_metadata(p)),
            'twinstar_sgl.pd_shallow_audit':module('audit',snapshot=Mock()),
            'sglang.srt.layers.logits_processor':module('logits',LogitsProcessorOutput=NS)}
        with patch.dict(sys.modules,modules),patch.dict('os.environ',{tail.FLAG:'1'}), \
             patch.object(split,'_backend',return_value=c.backend),patch.object(split,'_input_scope',return_value=nullcontext()), \
             patch.object(split,'_shallow_hidden',side_effect=shallow), \
             patch.object(split,'_emit_prefix',side_effect=lambda *a:c.pool.collected.extend(range(24,36))):
            split._shallow_prefill(c.owner,c.ids,c.positions,c.batch)
        graph.execute_tail.assert_called_once_with(c.tail)
        self.assertEqual(adapter.layers,31)
        self.assertEqual(c.pool.count.tolist(),[9]*24+[8]*12)
        c.owner.model.model.forward.assert_not_called()

    def test_capture_reserved_gdn_ple_qsa_and_ring_restored(self):
        tensors=[torch.arange(16,dtype=torch.float32).reshape(8,2)+i for i in range(8)]
        factor=NS(stale=tensors[0],dense_of=tensors[1],dense_required=None,prefix_valid=tensors[2],
            dense_ring=tensors[3],ring_owner=[0],ring_lru=[0],ring_generation=3,stats={'calls':7})
        mamba=NS(_iter_transfer_state_entries=lambda:[('a',tensors[4],None,0),('ple',tensors[5],None,0)])
        rp=NS(translate_mamba_indices=lambda x:x,get_mamba_indices=lambda x:x,mamba_pool=mamba,
            factored_gdn_pool=factor,ple_window_cache=object())
        kv=NS(_transfer_full_attention_id=lambda n:n,get_key_buffer=lambda n:tensors[6],
            get_value_buffer=lambda n:tensors[6],get_qsa_compressed_k_buffer=lambda n:tensors[6],
            qsa_key_state_buffer_pool={n:tensors[7] for n in range(3,48,4)},qsa_rope_position_buffer=tensors[7])
        ple=NS(_prefetch_state=None)
        runner=NS(req_to_token_pool=rp,token_to_kv_pool=kv,device='cpu',
            model=NS(model=NS(model=NS(layers=[NS(ple=ple)]))))
        before=[t.clone() for t in tensors]
        state=tail.CaptureState(runner)
        for tensor in tensors:tensor[0].fill_(99)
        tensors[3].fill_(99);tensors[7][:4].fill_(99)
        factor.ring_generation=4;factor.stats['calls']=9;rp.ple_window_cache=None
        state.restore();state.check()
        for wanted,actual in zip(before,tensors):torch.testing.assert_close(wanted,actual)
        self.assertEqual(factor.ring_generation,3)
        self.assertIs(rp.ple_window_cache,state.ple_cache)
        factor.dense_ring=factor.dense_ring.clone()
        with self.assertRaisesRegex(RuntimeError,'graph-owned state'):
            state.check()

    def test_real_model_runner_startup_entry_installs_tail_after_regular_capture(self):
        path=Path(split.__file__).with_name('model_runner.py')
        tree=ast.parse(path.read_text());owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ModelRunner')
        method=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='init_cuda_graphs')
        order=[];captured=NS(eager_runner=None,prefill=NS(runner=None),decode=NS(runner=None),memory_usage=0,time_usage=0)
        namespace={'os':__import__('os'),'capture_cuda_graphs':lambda **kw:(order.append('ordinary'),captured)[1]}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
        runner=NS(req_to_token_pool=NS(factored_gdn_pool=None),server_args=NS(disaggregation_mode='prefill'))
        for model_flag, legacy_flag, expected in (
                ('1', '0', ['ordinary', 'tail']), ('0', '1', ['ordinary'])):
            with self.subTest(model_flag=model_flag, legacy_flag=legacy_flag):
                order.clear()
                with patch.dict('os.environ',{tail.FLAG:model_flag,tail.LEGACY_FLAG:legacy_flag}), \
                     patch.object(tail,'install',side_effect=lambda r:order.append('tail')):
                    namespace['init_cuda_graphs'](runner)
                self.assertEqual(order,expected)

    def test_model_tail_rejects_legacy_capture_without_mutating_flags(self):
        with patch.dict('os.environ',{tail.FLAG:'1',tail.LEGACY_FLAG:'1'}), \
             patch.object(tail.native,'make_runner') as capture:
            with self.assertRaisesRegex(ValueError,'per-layer tail graph disabled'):
                tail.install(NS())
            capture.assert_not_called()
            self.assertEqual(__import__('os').environ[tail.LEGACY_FLAG],'1')

    def test_actual_split_fallback_keeps_per_layer_graph_disabled_in_both_arms(self):
        mode=NS(is_extend=lambda:True,is_mixed=lambda:False)
        batch=NS(forward_mode=mode,spec_info=None,extend_seq_lens_cpu=[1],
                 twinstar_prompt_final=[True],batch_size=1)
        for shallow in (False,True):
            with self.subTest(shallow=shallow):
                def legacy(*args):
                    self.assertEqual(__import__('os').environ[tail.LEGACY_FLAG],'0')
                    return 'legacy'
                original=Mock(side_effect=legacy)
                external=module('legacy',prefill_extend=original,forward=original)
                owner=NS(n_layers=48,p_layer_ids=list(range(31)),
                    pd_shallow_role='prefill' if shallow else None,fullstack={},emitters={})
                runtime=module('runtime',get_disagg=lambda:NS(disaggregation_mode='prefill'))
                env={split.FLAG:'1',tail.FLAG:'1',tail.LEGACY_FLAG:'0',
                     'TWINSTAR_PD_FACTOR_ONLY_TAIL':'0' if shallow else '1'}
                with patch.dict('os.environ',env), \
                     patch.dict(sys.modules,{'sglang.srt.runtime_context':runtime}), \
                     patch.object(split.importlib,'import_module',return_value=external), \
                     patch.object(tail,'execute') as replay:
                    self.assertTrue(split.install_prefill_model_split(owner))
                    args=(owner,torch.tensor([1]),torch.tensor([0]),batch)
                    result=(external.prefill_extend(*args) if shallow else
                            external.forward(owner,object(),*args[1:]))
                    self.assertEqual(result,'legacy')
                    original.assert_called_once()
                    replay.assert_not_called()
                    self.assertEqual(owner._gdn_prefill_model_split_stats['fallback_empty-prefix'],1)

    def test_enabled_missing_graph_fails_without_formal_capture_or_eager(self):
        with patch.dict('os.environ',{tail.FLAG:'1'}):
            with self.assertRaisesRegex(RuntimeError,'startup'):
                tail.execute(NS(),decode(NS(batch_size=1,input_ids=torch.tensor([1]))))

    def test_startup_restores_on_failure_and_checks_actual_captured_bucket_inventory(self):
        key=namedtuple('Key','size')
        body=NS(forward=Mock(),last_hc_hidden_states=None)
        owner=NS(model=NS(model=body),_gdn_prefill_model_split_installed=True)
        runner=NS(model=owner,server_args=NS(disaggregation_mode='prefill'))
        graph=NS(capture_bs=list(tail.BUCKETS),_make_graph_key=key,
                 backend=NS(_graphs={key(n):object() for n in tail.BUCKETS}),memory_receipt={})
        for failure in ('capture','missing_bucket',None):
            owner.__dict__.pop('_gdn_prefill_tail_graph',None)
            graph.backend._graphs={key(n):object() for n in tail.BUCKETS if failure!='missing_bucket' or n!=16}
            state=NS(restore=Mock(),check=Mock())
            with patch.dict('os.environ',{tail.FLAG:'1'}),patch.object(tail,'CaptureState',return_value=state), \
                 patch.object(tail.native,'make_runner',side_effect=RuntimeError('capture failed') if failure=='capture' else None,return_value=graph) as make:
                if failure:
                    with self.assertRaises(RuntimeError):tail.install(runner)
                else:
                    self.assertTrue(tail.install(runner))
                    self.assertEqual(graph.memory_receipt['captured_buckets'],[1,2,4,8,16])
                    self.assertEqual(make.call_args.kwargs['layers'],48)
                    self.assertIs(make.call_args.kwargs['body'].original,body.forward)
            state.restore.assert_called_once()

    def test_real_native_runner_type_captures_all_buckets_and_rebinds_each_decode(self):
        from sglang.srt.model_executor import forward_context as context
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

        body=NS(forward=Mock(),last_hc_hidden_states=None)
        cls=tail.native.tail_runner_type(body,tail.BUCKETS)
        graph=object.__new__(cls)
        graph.capture_one_shape=Mock()
        graph._capture_one_stream()
        self.assertEqual([c.args[0] for c in graph.capture_one_shape.call_args_list],[16,8,4,2,1])
        graph.can_run_graph=Mock(return_value=True);graph.attn_backend=object();graph.replays=0
        received=[]

        def execute(current):
            self.assertFalse(current.forward_metadata_ready)
            received.append(current)
            return PPProxyTensors({'hidden':current.input_ids[:,None]*10,'hc':current.req_pool_indices[:,None]})

        graph.execute=execute
        for token,slot in ((17,4),(29,7)):
            batch=decode(NS(batch_size=1,input_ids=torch.tensor([token]),req_pool_indices=torch.tensor([slot]),forward_metadata_ready=True))
            with patch.object(context,'forward_context',return_value=nullcontext()),patch.object(context,'ForwardContext',NS):
                result=graph.execute_tail(batch)
            torch.testing.assert_close(result,torch.tensor([[token*10]]))
            torch.testing.assert_close(body.last_hc_hidden_states,torch.tensor([[slot]]))
            self.assertTrue(batch.forward_metadata_ready)
            self.assertIsNot(received[-1],batch)
        self.assertEqual(graph.replays,2)

    def test_real_native_runner_crops_nonbucket_hidden_and_hc_rows(self):
        from sglang.srt.model_executor import forward_context as context
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

        body=NS(forward=Mock(),last_hc_hidden_states=None)
        cls=tail.native.tail_runner_type(body,tail.BUCKETS)
        graph=object.__new__(cls)
        graph.can_run_graph=Mock(return_value=True);graph.attn_backend=object();graph.replays=0
        for size in (3,5,15):
            bucket=next(n for n in tail.BUCKETS if n>=size)
            for with_hc in (False,True):
                with self.subTest(size=size,with_hc=with_hc):
                    hidden=torch.arange(bucket*4).reshape(bucket,4)
                    hc=torch.arange(bucket*8).reshape(bucket,2,4)
                    tensors={'hidden':hidden}
                    if with_hc:tensors['hc']=hc
                    graph.execute=Mock(return_value=PPProxyTensors(tensors))
                    body.last_hc_hidden_states=torch.tensor([-1])
                    batch=decode(NS(batch_size=size,input_ids=torch.arange(size),forward_metadata_ready=True))
                    with patch.object(context,'forward_context',return_value=nullcontext()),patch.object(context,'ForwardContext',NS):
                        result=graph.execute_tail(batch)
                    self.assertEqual(result.shape,(size,4))
                    torch.testing.assert_close(result,hidden[:size])
                    if with_hc:
                        self.assertEqual(body.last_hc_hidden_states.shape,(size,2,4))
                        torch.testing.assert_close(body.last_hc_hidden_states,hc[:size])
                    else:self.assertIsNone(body.last_hc_hidden_states)
                    self.assertTrue(batch.forward_metadata_ready)
        self.assertEqual(graph.replays,6)


if __name__=='__main__':unittest.main()
