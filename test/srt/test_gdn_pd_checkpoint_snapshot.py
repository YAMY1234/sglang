"""P3 alone replaces the aligned legacy final state with an fp32 snapshot."""
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from sglang.srt.layers.attention.linear import gdn_backend as gdn
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel


class SnapshotTest(unittest.TestCase):
    def run_case(self, mask):
        stage = torch.zeros(2, 1, 2, 2)
        h = torch.full((1, 4, 1, 2, 2), .12345, dtype=torch.bfloat16)
        snapshot = torch.full_like(stage, .1234567)
        commit = Mock()
        calls = []

        def extend(**kwargs):
            calls.append(kwargs)
            if kwargs.get('track_state') is not None:
                kwargs['track_state'].copy_(snapshot)
            return torch.zeros(1, 130, 1, 2), None, h

        backend = NS(factored=NS(initial_dense=lambda *a: stage,
                                commit_extend_batched=commit, batch_prefill=True),
                     _stepwise_active=lambda _: False,
                     kernel_dispatcher=NS(extend_kernel=object.__new__(TritonGDNKernel),
                                          extend=extend))
        metadata = NS(factored_extend=NS(slots=torch.tensor([1, 2])),
                      track_ssm_h_src=torch.tensor([1, 3]),
                      track_ssm_h_batch_src=torch.tensor([0, 1]),
                      track_ssm_h_dst=torch.tensor([10, 11]),
                      track_chunk_idx=torch.tensor([1, 1]),
                      track_ssm_final_src=None, track_ssm_final_dst=None,
                      has_mamba_track_mask=True, track_ssm_recompute_dst=None,
                      state_checkpoint_cu_starts=None, num_state_checkpoints=None,
                      state_checkpoint_every_n_tokens=None)
        env = {name: '0' for name in ('SGLANG_GDN_PSIDE_GRAPH', 'SGLANG_GDN_PREFILL_DENSE_GRAPH')}
        with patch.dict(os.environ, env), patch.object(gdn, '_FACTORED_DUMP_DIR', None), \
             patch.object(gdn, '_gating', return_value=(None, None)):
            gdn.GDNAttnBackend._forward_extend_factored(
                backend, layer=NS(layer_id=0),
                forward_batch=NS(_pd_fulln_fp32_track_mask=mask),
                query=torch.zeros(1, 130, 1, 2, dtype=torch.bfloat16), key=None,
                value=None, a=None, b=None, query_start_loc=torch.tensor([0, 65, 130]),
                forward_metadata=metadata, output=None)
        return calls[0], commit.call_args.args[3], h, snapshot

    def test_only_legacy_final_rows_use_the_fp32_snapshot(self):
        call, tracked, h, snapshot = self.run_case(torch.tensor([True, False]))
        self.assertIsNotNone(call['track_state'])
        self.assertEqual(tracked.dtype, torch.float32)
        self.assertTrue(torch.equal(tracked[0], snapshot[0]))
        self.assertTrue(torch.equal(tracked[1], h[0, 3].float()))
        self.assertFalse(torch.equal(tracked[0], h[0, 1].float()))

    def test_absent_p3_marker_keeps_the_original_bf16_h(self):
        call, tracked, h, _ = self.run_case(None)
        self.assertNotIn('track_state', call)
        self.assertEqual(tracked.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(tracked, h[0, [1, 3]]))


if __name__ == '__main__':
    unittest.main()
