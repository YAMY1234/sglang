"""Execute real verify/host publication functions without a GPU scheduler import."""
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[2]
SRT = ROOT / 'python/sglang/srt'

def methods(path, names, scope):
    baseline = os.environ.get('STAGE2_TEST_BASELINE_SHA')
    source = subprocess.check_output(['git', 'show', baseline + ':' + str(path.relative_to(ROOT))], cwd=ROOT, text=True) if baseline else path.read_text()
    tree = ast.parse(source)
    for name in names:
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        exec('from __future__ import annotations\n' + ast.unparse(node), scope)

spec = importlib.util.spec_from_file_location('fullstack_policy_cpu', SRT/'model_executor/fullstack_policy.py')
POLICY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(POLICY)

class PrefixPublicationTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'TWINSTAR_FULLSTACK': '1', 'SGLANG_EXTERNAL_MODEL_PACKAGE': 'twinstar_sgl'})
        self.env.start(); self.addCleanup(self.env.stop)
        self.mods = patch.dict(sys.modules, {'sglang.srt.model_executor.fullstack_policy': POLICY})
        self.mods.start(); self.addCleanup(self.mods.stop)
        self.config = NS(hf_config=NS(twinstar={'fullstack': {'version': 3}}))
        self.calls = []
        def set_indices(batch, positions):
            self.calls.append('set_indices')
            batch.mamba_track_indices = torch.tensor([7])
        self.scope = {'torch': torch, 'mamba_track_grid': lambda page: 256,
                      'get_exec': lambda: NS(mamba=NS(enable_mamba_extra_buffer=True, enable_mamba_extra_buffer_lazy=False)),
                      'set_mamba_track_indices_from_reqs': set_indices}
        methods(SRT/'speculative/spec_utils.py', ['prepare_mamba_track_for_verify', '_verify_commit_step_indices'], self.scope)
        methods(SRT/'managers/scheduler_components/batch_result_processor.py', ['_mamba_prefix_cache_update', '_mamba_check_track_boundary'], self.scope)
        methods(SRT/'mem_cache/unified_cache/components/mamba_component.py', ['prepare_for_caching_req'], self.scope)

    def batch(self):
        return NS(model_config=self.config, reqs=[NS()], seq_lens=torch.tensor([8446]),
                  tree_cache=NS(page_size=64), spec_algorithm=NS(is_none=lambda: False),
                  mamba_track_indices=torch.tensor([7]), mamba_track_mask=torch.tensor([True]),
                  mamba_track_seqlens=torch.tensor([8192]), mamba_track_buffer_indices=[0])

    def test_verify_never_overwrites_prefill_checkpoint(self):
        b = self.batch(); self.scope['prepare_mamba_track_for_verify'](b)
        self.assertIsNone(b.mamba_track_indices)
        self.assertIsNone(b.mamba_track_mask)
        self.assertIsNone(b.mamba_track_seqlens)
        self.assertEqual(self.calls, [])

    def test_consumed_target_inputs_still_commit_live_state(self):
        for accepted_drafts in range(4):
            b = self.batch(); self.scope['prepare_mamba_track_for_verify'](b)
            live, track = self.scope['_verify_commit_step_indices'](batch=b, accept_index=torch.arange(4).reshape(1, 4), accept_lens=torch.tensor([1 + accepted_drafts]), draft_token_num=4)
            self.assertEqual(live.tolist(), [accepted_drafts])
            self.assertIsNone(track)

    def test_repeated_zero_acceptance_and_crossings_preserve_donated_depth(self):
        req = NS(origin_input_ids=range(8193), seqlen=8450, kv=NS(mamba_ping_pong_track_buffer=torch.tensor([7, 8]), mamba_last_track_seqlen=8192, mamba_next_track_idx=0, mamba_last_track_idx=0))
        b = self.batch()
        scheduler = NS(tree_cache=b.tree_cache)
        scheduler._mamba_check_track_boundary = lambda *a: self.scope['_mamba_check_track_boundary'](scheduler, *a)
        b.req_to_token_pool = NS(get_mamba_ping_pong_other_idx=lambda i: 1-i)
        for consumed in [3] + [1]*520:
            result = NS(num_correct_drafts_per_req_cpu=[consumed-1])
            self.scope['_mamba_prefix_cache_update'](scheduler, req, b, result, 0)
            self.assertEqual(req.kv.mamba_last_track_seqlen, 8192)
            self.assertEqual(req.kv.mamba_next_track_idx, 0)
            req.seqlen += consumed
        component = NS(cache=NS(enable_mamba_extra_buffer=True, req_to_token_pool=NS(get_mamba_ping_pong_keep_idx=lambda req: 0)), int8_ckpt_pool=None)
        insert = NS()
        cache_len = self.scope['prepare_for_caching_req'](component, req, insert, token_ids_len=9000, is_finished=True)
        self.assertEqual(cache_len, 8192)
        self.assertEqual(insert.mamba_value.tolist(), [7])

    def test_stock_and_flag_off_keep_original_tracking(self):
        for enabled, package in [('0', 'twinstar_sgl'), ('1', '')]:
            with self.subTest(enabled=enabled, package=package), patch.dict(os.environ, {'TWINSTAR_FULLSTACK': enabled, 'SGLANG_EXTERNAL_MODEL_PACKAGE': package}):
                b = self.batch(); self.scope['prepare_mamba_track_for_verify'](b)
                live, track = self.scope['_verify_commit_step_indices'](batch=b, accept_index=torch.arange(4).reshape(1, 4), accept_lens=torch.tensor([3]), draft_token_num=4)
                self.assertEqual(live.tolist(), [2]); self.assertEqual(track.tolist(), [1])
                req = NS(seqlen=8450, kv=NS(mamba_ping_pong_track_buffer=torch.tensor([7,8]), mamba_last_track_seqlen=8192, mamba_next_track_idx=0))
                b.req_to_token_pool = NS(get_mamba_ping_pong_other_idx=lambda i: 1-i)
                scheduler = NS(tree_cache=b.tree_cache)
                scheduler._mamba_check_track_boundary = lambda *a: self.scope['_mamba_check_track_boundary'](scheduler, *a)
                self.scope['_mamba_prefix_cache_update'](scheduler, req, b, NS(num_correct_drafts_per_req_cpu=[2]), 0)
                self.assertEqual(req.kv.mamba_last_track_seqlen, 8448)
                self.assertEqual(req.kv.mamba_next_track_idx, 1)

if __name__ == '__main__': unittest.main()
