"""Track checkpoints retain their values without a per-layer device-to-host copy."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Optional, Tuple
import unittest
from unittest.mock import patch

import msgspec
import torch
import torch.nn.functional as F


SOURCE = Path(__file__).parents[4] / 'python/sglang/srt/models/qwen4_exp.py'
TREE = ast.parse(SOURCE.read_text())
NAMES = {'_PLEBatch', '_ple_track_targets', '_cache_ple_track_offsets',
         '_ple_short_conv_per_request'}
NODES = [n for n in TREE.body if getattr(n, 'name', None) in NAMES]
PLE = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen4ExpPLELayer')
NODES += [n for n in PLE.body if isinstance(n, ast.FunctionDef) and n.name == '_short_conv']
NSPACE = dict(msgspec=msgspec, torch=torch, F=F, Optional=Optional, Tuple=Tuple,
              ForwardMode=object, ForwardBatch=object, _PLE_PREFILL_LOWMEM=True,
              get_is_capture_mode=lambda: False)
exec(compile(ast.Module(body=NODES, type_ignores=[]), str(SOURCE), 'exec'), NSPACE)


class PleTrackOffsetsTest(unittest.TestCase):
    def case(self, lengths, aligned):
        mode = NS(is_decode=lambda: False, is_target_verify=lambda: False)
        batch = NSPACE['_PLEBatch'](mode, False, sum(lengths), sum(lengths),
            torch.tensor(lengths), max(lengths), torch.zeros(sum(lengths), dtype=torch.long),
            torch.zeros(sum(lengths), dtype=torch.long), torch.ones(sum(lengths), dtype=torch.bool),
            torch.arange(1, len(lengths) + 1), None, None)
        forward = NS(extend_seq_lens_cpu=lengths, mamba_track_indices=torch.arange(3, 3 + len(lengths)),
                     mamba_track_mask=torch.tensor([True] * len(lengths)),
                     mamba_track_aligned_lens=lambda: torch.tensor(aligned))
        return batch, forward

    def test_offsets_clamp_and_refresh_for_each_forward(self):
        first, forward = self.case([4, 7], [-2, 99])
        cached = NSPACE['_cache_ple_track_offsets'](first, forward)
        self.assertEqual(cached.track_offsets_cpu, (0, 7))
        self.assertIsNone(first.track_offsets_cpu)
        second, forward = self.case([8, 1], [5, 0])
        self.assertEqual(NSPACE['_cache_ple_track_offsets'](second, forward).track_offsets_cpu, (5, 0))
        self.assertEqual(cached.track_offsets_cpu, (0, 7))

    def test_repeated_layers_preserve_conv_and_checkpoint_without_host_copies(self):
        batch, forward = self.case([4, 7], [2, 6])
        cached = NSPACE['_cache_ple_track_offsets'](batch, forward)
        torch.manual_seed(4)
        x = torch.randn(11, 2)
        initial = torch.randn(6, 2, 2)
        owner = NS(layer_id=0, conv_channels=2, short_conv_dilation=1, short_conv_state_len=2,
                   conv1d=NS(weight=torch.randn(2, 1, 3)))
        short_conv = NSPACE['_short_conv']
        original = torch.Tensor.tolist
        copies = []
        def counted(tensor):
            copies.append(1)
            return original(tensor)
        for iteration in range(36):
            ref_state = initial.clone()
            NSPACE['get_req_to_token_pool'] = lambda: NS(short_conv_layer_cache=lambda _: ref_state)
            reference = short_conv(owner, x, forward, batch)
            actual_state = initial.clone()
            NSPACE['get_req_to_token_pool'] = lambda: NS(short_conv_layer_cache=lambda _: actual_state)
            with patch.object(torch.Tensor, 'tolist', counted):
                actual = short_conv(owner, x, forward, cached)
            self.assertTrue(torch.equal(reference, actual))
            self.assertTrue(torch.equal(ref_state, actual_state))
        self.assertEqual(copies, [])

    def test_capture_and_absent_tracking_do_not_copy(self):
        batch, forward = self.case([4], [2])
        with patch.dict(NSPACE, get_is_capture_mode=lambda: True), \
             patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('host copy during capture')):
            self.assertIs(NSPACE['_cache_ple_track_offsets'](batch, forward), batch)
        forward.mamba_track_mask = None
        self.assertIs(NSPACE['_cache_ple_track_offsets'](batch, forward), batch)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
