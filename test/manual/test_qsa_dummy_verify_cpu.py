"""Autotune placeholders must not leave request transactions open."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

ROOT=Path(__file__).resolve().parents[2]/'python/sglang/srt'


def method(path,name,scope=None):
    scope={} if scope is None else scope
    node=next(n for n in ast.walk(ast.parse((ROOT/path).read_text()))
              if isinstance(n,ast.FunctionDef) and n.name==name)
    exec('from __future__ import annotations\n'+ast.unparse(node),scope)
    return scope[name]


prepare=method(Path('model_executor/model_runner.py'),'prepare_dummy_forward_batch')
begin_qsa=method(Path('layers/attention/qwen_sparse_attn_backend.py'),'_begin_qsa_verify')
begin_factor=method(Path('layers/attention/linear/gdn_backend.py'),'_begin_factored_verify')


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'mem_cache'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def batch(verify=True):
    return NS(batch_size=4,req_pool_indices=torch.zeros(4,dtype=torch.int64),
        forward_mode=NS(is_target_verify=lambda:verify),_forward_num_tokens=lambda:16)


class DummyVerify(unittest.TestCase):
    def test_repeated_dummy_rows_do_not_open_qsa_then_real_request_can_begin(self):
        pool=NS(qsa_compress_ratio=4,qsa_key_state_buffer_pool=[torch.zeros(20,1,2)],
                qsa_rope_position_buffer=torch.zeros(20,3,dtype=torch.int64))
        txn=load('qsa_verify_state').QSAVerifyState(pool,4,4)
        runner=NS(token_to_kv_pool=pool,attn_tp_sequence_sharded=lambda n:False)
        backend=NS(qsa_verify=txn)
        before=[t.clone() for t in pool.qsa_key_state_buffer_pool]
        for _ in range(5):
            fb=batch();self.assertIs(prepare(runner,fb),fb)
            self.assertEqual(fb.num_padding,4)
            begin_qsa(backend,fb);self.assertTrue(txn.closed)
        for a,b in zip(before,pool.qsa_key_state_buffer_pool):self.assertTrue(torch.equal(a,b))
        fb=batch();fb.req_pool_indices=torch.arange(1,5)
        begin_qsa(backend,fb)
        self.assertFalse(txn.closed)
        self.assertEqual(txn.requests.tolist(),[1,2,3,4])
        with self.assertRaisesRegex(RuntimeError,'uncommitted'):begin_qsa(backend,fb)

    def test_factor_snapshot_sees_no_duplicate_dummy_slots_but_real_slots_remain_checked(self):
        calls=[]
        factor=NS(spec_state=object(),snapshot_commit=lambda slots:calls.append(slots.clone()))
        backend=NS(factored=factor,forward_metadata=NS(mamba_cache_indices=torch.zeros(4,dtype=torch.int64)))
        runner=NS(token_to_kv_pool=NS(qsa_compress_ratio=16),attn_tp_sequence_sharded=lambda n:False)
        for _ in range(5):begin_factor(backend,prepare(runner,batch()))
        self.assertEqual(calls,[])
        fb=batch();backend.forward_metadata.mamba_cache_indices=torch.tensor([8,1,7,3])
        begin_factor(backend,fb)
        self.assertEqual(calls[0].tolist(),[8,1,7,3])

    def test_sequential_flagoff_and_non_qsa_dummy_modes_are_unchanged(self):
        for qsa,verify in [(True,False),(False,True),(False,False)]:
            pool=NS(qsa_compress_ratio=16) if qsa else NS()
            runner=NS(token_to_kv_pool=pool,attn_tp_sequence_sharded=lambda n:True)
            fb=batch(verify);prepare(runner,fb)
            self.assertTrue(fb.attn_tp_sequence_sharded)
            self.assertFalse(hasattr(fb,'num_padding'))


if __name__=='__main__':unittest.main()
