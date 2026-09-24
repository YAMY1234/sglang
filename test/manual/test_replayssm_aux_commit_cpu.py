"""ReplaySSM must publish QSA/PLE exactly once, including zero-draft rounds."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS,ModuleType
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[2]

def methods(path,names,scope):
    tree=ast.parse(path.read_text())
    for name in names:
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
        exec('from __future__ import annotations\n'+ast.unparse(node),scope)


def load(path):
    spec=importlib.util.spec_from_file_location(path.stem,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class AuxiliaryCommitTest(unittest.TestCase):
    def test_fold_and_circular_close_real_qsa_transaction_and_commit_only_accepted_prefix(self):
        qsa_cls=load(ROOT/'python/sglang/srt/mem_cache/qsa_verify_state.py').QSAVerifyState
        for fold in (False,True):
            pool=NS(qsa_compress_ratio=4,qsa_key_state_buffer_pool=[torch.zeros(16,1,2)],
                qsa_rope_position_buffer=torch.zeros(16,3,dtype=torch.int64),_transfer_full_attention_id=lambda i:i)
            qsa=qsa_cls(pool,2,4);order=[]
            hybrid=NS(linear_attn_backend=NS(req_to_token_pool=NS()),
                full_attn_backend=NS(commit_qsa_verify=lambda steps:(order.append('qsa'),qsa.commit(steps))),
                _update_ple_state_after_mtp_verify=lambda *a:order.append(('ple',a)))
            scope={};methods(ROOT/'python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py',
                ['update_auxiliary_state_after_mtp_verify'],scope)
            hybrid.update_auxiliary_state_after_mtp_verify=lambda *a:scope['update_auxiliary_state_after_mtp_verify'](hybrid,*a)
            mamba=NS(replayssm_spec_fold=fold,replayssm_is_kda=False,replayssm_cache_base=torch.zeros(8),
                replayssm_spec_write_pos=torch.zeros(8),replayssm_is_flush=torch.zeros(8))
            spec=NS(replayssm_d=torch.zeros(2,8,4,2),replayssm_k=None,replayssm_g=None,
                replayssm_rawv=None,replayssm_rawk=None,temporal=torch.zeros(2,8,2,2),conv=[None],intermediate_conv_window=[None])
            states=torch.tensor([5,2]);track=torch.tensor([-1,6]);tracking=torch.tensor([-1,1])
            req=NS(mamba_pool=mamba,get_speculative_mamba2_params_all_layers=lambda:spec,get_mamba_indices=lambda r:states)
            runner=NS(model_config=NS(),req_to_token_pool=req,attn_backend=hybrid)
            batch=NS(forward_mode=NS(is_idle=lambda:False),req_pool_indices=torch.tensor([1,3]),mamba_track_indices=track)
            calls={'torch':torch,'mambaish_config':lambda config:True,
                '_verify_commit_step_indices':lambda **kw:(kw['accept_lens']-1,tracking)}
            methods(ROOT/'python/sglang/srt/speculative/spec_utils.py',
                ['_commit_replayssm_auxiliary_state','commit_mamba_states_after_verify'],calls)
            def kernel(**kw):order.append('ssm')
            def scatter(dst,src,slots,steps):
                valid=slots>=0;dst[:,slots[valid]]=src[:,torch.arange(len(slots))[valid],steps[valid]]
            modules={
                'sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold':NS(commit_gdn_replayssm_fold_after_verify=kernel),
                'sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode':NS(commit_gdn_replayssm_spec=kernel,commit_gdn_replayssm_circular=kernel),
                'sglang.kernels.ops.mamba.mamba_state_scatter_triton':NS(fused_conv_window_scatter_with_mask=lambda *a:order.append('conv')),
                'sglang.srt.mem_cache.gdn_factored_spec':NS(FactoredGDNVerifyState=NS(_scatter=scatter)),
            }
            with patch.dict(sys.modules,modules):
                for round_ in range(18):
                    order.clear();qsa.begin(batch.req_pool_indices)
                    positions=torch.tensor([6,7,8,9,10,11,12,13])+round_*4
                    values=torch.arange(16).reshape(8,1,2).float()+round_*100
                    ropes=torch.arange(24).view(8,3)+round_*100
                    qsa.record(0,values,ropes,positions)
                    before=pool.qsa_key_state_buffer_pool[0].clone();steps=torch.tensor([0,3] if round_==0 else [0,0])
                    calls['commit_mamba_states_after_verify'](NS(model_runner=runner),batch,steps+1,torch.arange(8).view(2,4),4)
                    expected=before.clone()
                    for i,request in enumerate(batch.req_pool_indices):
                        for j in range(int(steps[i])+1):expected[int(request)*4+int(positions[i*4+j])%4]=values[i*4+j]
                    self.assertTrue(qsa.closed)
                    self.assertTrue(torch.equal(expected,pool.qsa_key_state_buffer_pool[0]))
                    self.assertEqual(order.count('qsa'),1)
                    ple=next(x for x in order if isinstance(x,tuple));self.assertTrue(torch.equal(ple[1][1],steps))
                    self.assertEqual(order[-1],'qsa')


if __name__=='__main__':unittest.main()
