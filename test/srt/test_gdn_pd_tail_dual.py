"""The exact-tail baseline must capture A, rather than startup's P5 policy."""
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from sglang.srt.mem_cache import gdn_factored_pool as gp


class TailDualTest(unittest.TestCase):
    def run_prewarm(self, dual):
        captures = []
        class Graph:
            def __init__(self, *, shared=None):
                self.shared = {} if shared is None else shared
            def prewarm(self, *a, **k):
                captures.append((self, tuple(os.environ[n] for n in
                    ('SGLANG_GDN_K31_FUSED', 'SGLANG_GDN_K31_CHOLQR_MIXED'))))
        pool = NS(cfg=NS(init_method='k31', factored_prefix=True),
                  prefix_dense=None, a=NS(is_cuda=True), device='cuda',
                  _pdfix_commit_graph=NS(prewarm=Mock()))
        env = dict(SGLANG_GDN_PREFILL_COMMIT_GRAPH='1',
                   SGLANG_GDN_PREFILL_BATCH_GRAPH='1',
                   SGLANG_GDN_PREFILL_CHECKPOINT_GRAPH='0',
                   SGLANG_GDN_K31_AB_DUAL=str(dual),
                   SGLANG_GDN_K31_AB_ENV='SGLANG_GDN_K31_FUSED,SGLANG_GDN_K31_CHOLQR_MIXED',
                   SGLANG_GDN_K31_FUSED='1', SGLANG_GDN_K31_CHOLQR_MIXED='1')
        with patch.dict(os.environ, env), patch.object(gp, 'k31_graph_safe', return_value=True), \
             patch('sglang.srt.mem_cache.gdn_prefill_batch_graph.PrefillBatchGraph', Graph):
            gp.FactoredGDNPool.prewarm_commit_graph(pool)
            self.assertEqual(os.environ['SGLANG_GDN_K31_FUSED'], '1')
            self.assertEqual(os.environ['SGLANG_GDN_K31_CHOLQR_MIXED'], '1')
        return pool, captures

    def test_debug_baseline_and_candidate_capture_distinct_policies_shared_inputs(self):
        pool, captures = self.run_prewarm(1)
        self.assertEqual([flags for _, flags in captures], [('0', '0'), ('1', '1')])
        self.assertIsNot(captures[0][0], captures[1][0])
        self.assertIs(captures[0][0].shared, captures[1][0].shared)
        self.assertIs(pool._prefill_batch_graph, pool._prefill_batch_graph_ab['A'])

    def test_production_flag_off_keeps_single_startup_policy(self):
        pool, captures = self.run_prewarm(0)
        self.assertEqual([flags for _, flags in captures], [('1', '1')])
        self.assertFalse(hasattr(pool, '_prefill_batch_graph_ab'))


if __name__ == '__main__':
    unittest.main()
