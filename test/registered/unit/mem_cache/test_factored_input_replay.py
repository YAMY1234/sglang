"""CPU storage/lifecycle checks for the real replay owner; kernels tested separately."""
import importlib
import ast
import logging
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[4]
name = 'input_replay_unit_owners'
package = ModuleType(name)
package.__path__ = [str(ROOT/'python/sglang/srt/mem_cache')]
sys.modules[name] = package
Old = importlib.import_module(name+'.gdn_factored_spec').FactoredGDNVerifyState
New = importlib.import_module(name+'.gdn_factored_replay').FactoredGDNReplayState
Config = importlib.import_module(name+'.gdn_factored_pool').FactoredGDNConfig


def pool(layers=2):
    return SimpleNamespace(cfg=Config(dtype=torch.float16),
        a=torch.zeros(layers, 8, 2, 16),
        U=torch.zeros(layers, 8, 2, 16, 16, dtype=torch.float16),
        W=torch.zeros(layers, 8, 2, 16, 32, dtype=torch.float16),
        count=torch.full((layers, 8, 2), 8, dtype=torch.int32),
        stale=torch.ones(8, dtype=torch.int32), dense_of=torch.full((8,), -1, dtype=torch.int32),
        dense_required=None, prefix_valid=None)


class ReplayStorageTest(unittest.TestCase):
    def test_real_pool_selects_only_opted_in_owner_and_rejects_two_implementations(self):
        module=importlib.import_module(name+'.gdn_factored_pool')
        params=SimpleNamespace(shape=SimpleNamespace(temporal=(2,32,16),conv=[(96,3)]),
                               dtype=SimpleNamespace(conv=torch.bfloat16))
        aliases={'sglang.srt.mem_cache.gdn_factored_spec':importlib.import_module(name+'.gdn_factored_spec'),
                 'sglang.srt.mem_cache.gdn_factored_replay':importlib.import_module(name+'.gdn_factored_replay')}
        def construct():
            return module.FactoredGDNPool(size=8,cache_params=params,mamba_layer_ids=[0,1],
                device='cpu',cfg=Config(dtype=torch.float16),spec_max_batch_size=3,
                speculative_num_draft_tokens=4)
        for enabled in (False,True):
            with patch.dict(sys.modules,aliases), patch.dict(os.environ,{
                    'SGLANG_GDN_VERIFY_REPLAY_INPUTS':str(int(enabled)),
                    'SGLANG_GDN_VERIFY_DIRECT_CHECKPOINT':'0'}):
                actual=construct()
                self.assertIs(type(actual.spec_state),New if enabled else Old)
        with patch.dict(sys.modules,aliases), patch.dict(os.environ,{
                'SGLANG_GDN_VERIFY_REPLAY_INPUTS':'1','SGLANG_GDN_VERIFY_DIRECT_CHECKPOINT':'1'}):
            with self.assertRaisesRegex(ValueError,'mutually exclusive'):
                construct()

    def test_reservation_saving_equals_measured_tensors_for_batch_and_pp(self):
        params = SimpleNamespace(shape=SimpleNamespace(temporal=(2,32,16), conv=[(96,3)]),
                                 dtype=SimpleNamespace(conv=torch.bfloat16))
        for layers in (1,2):
            for capacity in (1,3,97):
                p = pool(layers)
                old, new = Old(p, capacity, 4), New(p, capacity, 4, qkv_width=96)
                saved = p.cfg.verify_replay_saved_bytes_per_row(params, layers, 4)*capacity
                self.assertEqual(old.bytes()-new.bytes(), saved)
                self.assertEqual(new.checkpoints, {})

    def test_owned_window_and_precision_rejection(self):
        tx=New(pool(),3,4,qkv_width=96)
        x=torch.randn(2,4,96,dtype=torch.bfloat16)
        a=torch.randn(2,4,2,dtype=torch.bfloat16)
        original=x.clone()
        tx.record_inputs(0,x,a,a,{'scale':.25})
        x.zero_()
        self.assertTrue(torch.equal(tx.inputs['mixed'][0,:2],original))
        with self.assertRaisesRegex(ValueError,'precision'):
            tx.record_inputs(1,x.float(),a,a,{'scale':.25})

    def test_generation_failure_precedes_any_restore_or_publish(self):
        p=pool();tx=New(p,3,4,qkv_width=96)
        slots=torch.tensor([2,5]);ticket=tx.snapshot_commit(slots)
        tx.written.fill_(True);tx.layer_arguments=[{},{}]
        tx.invalidate_slots(slots[:1]);p.a.add_(7)
        with self.assertRaises(RuntimeError):
            tx.commit(ticket,torch.tensor([0,3]),_decode=lambda *a,**kw:self.fail('decoded stale transaction'))
        self.assertTrue(torch.all(p.a==7))

    def test_actual_planner_flagoff_matches_frozen_source_and_replay_returns_saved_bytes(self):
        path=ROOT/'python/sglang/srt/mem_cache/kv_cache_configurator.py'
        baseline=subprocess.check_output(['git','show','88ac72d4525:'+str(path.relative_to(ROOT))],
                                         cwd=ROOT,text=True)
        def call(source, *, replay=False, mode='explicit', factored=True):
            method=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef)
                        and n.name=='_handle_max_mamba_cache')
            params=SimpleNamespace(shape=SimpleNamespace(temporal=(24,128,128),conv=[(5120,3)]),
                dtype=SimpleNamespace(conv=torch.bfloat16),layers=list(range(36)),
                mamba_cache_per_req=36*(24*128*128*4+5120*3*2))
            schedule=SimpleNamespace(max_running_requests=96,
                max_mamba_cache_size=480 if mode=='explicit' else None,mamba_full_memory_ratio=.9)
            context=SimpleNamespace(override=lambda reason,**kw:schedule.__dict__.update(kw))
            config=SimpleNamespace(mamba2_cache_params=params)
            runner=SimpleNamespace(mambaish_config=config,ps=SimpleNamespace(pp_size=1,attn_dp_size=1),
                hybrid_gdn_config=True,model_config=None,spec_algorithm=SimpleNamespace(is_none=lambda:False),
                _calculate_mamba_ratio=lambda:5)
            scope=dict(os=os,logger=logging.getLogger(__name__),
                get_schedule=lambda:schedule,get_context=lambda:context,
                get_exec=lambda:SimpleNamespace(mamba=SimpleNamespace(
                    linear_attn_factored_state='r=8,m=8,dtype=fp16' if factored else '',
                    enable_linear_replayssm_spec=False)),
                get_spec=lambda:SimpleNamespace(speculative_num_draft_tokens=4),
                get_memory=lambda:SimpleNamespace(disable_radix_cache=mode=='no_radix'))
            exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
            module=importlib.import_module(name+'.gdn_factored_pool')
            with patch.dict(sys.modules,{'sglang.srt.mem_cache.gdn_factored_pool':module}), \
                 patch.dict(os.environ,{'SGLANG_GDN_VERIFY_REPLAY_INPUTS':str(int(replay))}):
                rest=scope[method.name](runner,100.)
            return rest,schedule.max_mamba_cache_size,params
        for mode in ('explicit','no_radix','auto'):
            for factored in (False,True):
                old=call(baseline,mode=mode,factored=factored)
                current=call(path.read_text(),mode=mode,factored=factored)
                self.assertEqual(old[:2],current[:2])
        old=call(path.read_text());new=call(path.read_text(),replay=True)
        saved=Config(dtype=torch.float16).verify_replay_saved_bytes_per_row(old[2],36,4)*97
        self.assertEqual(new[1],480)
        self.assertAlmostEqual((new[0]-old[0])*(1<<30),saved,places=3)


if __name__=='__main__':
    unittest.main()
