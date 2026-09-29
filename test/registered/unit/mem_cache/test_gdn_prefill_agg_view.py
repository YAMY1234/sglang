"""Keep the P48 contract across the actual eager runner's batch replacement."""
from dataclasses import MISSING, fields
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.gdn_prefill_agg_contract import eligible
from sglang.srt.model_executor.cuda_graph_buffer_registry import CudaGraphBufferRegistry
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner.eager_runner import EagerRunner


class AggViewTest(unittest.TestCase):
    def test_real_eager_views_preserve_contract_and_fallback_shapes(self):
        cases = [(1, 8192, False, None, True), (8, 8192, False, None, True),
                 (16, 8192, False, None, True), (32, 8192, False, None, False),
                 (2, 8192, True, None, False), (8, 8192, False, 0, False),
                 (1, 1, False, None, False)]
        for no_copy in ('0', '1'):
            for rows, tokens, mixed, split, selected in cases:
                with self.subTest(no_copy=no_copy, rows=rows, tokens=tokens, mixed=mixed, split=split):
                    required = {f.name: None for f in fields(ForwardBatch)
                                if f.default is MISSING and f.default_factory is MISSING}
                    required.update(forward_mode=ForwardMode.MIXED if mixed else ForwardMode.EXTEND,
                        batch_size=rows, input_ids=torch.arange(rows*tokens), seq_lens_sum=rows*tokens)
                    batch = ForwardBatch(**required)
                    batch.extend_seq_lens_cpu = [tokens]*rows
                    batch.tbo_split_seq_index = split
                    batch._pfactor_agg_contract = selected
                    batch.pd_factor_only_full_batch = True
                    batch._pfactor_legacy_mixed = mixed
                    batch.twinstar_prompt_final = [True] * rows
                    batch._unregistered_contract = True
                    registry = CudaGraphBufferRegistry(device=torch.device('cpu'),
                        max_bs=rows, max_num_tokens=rows*tokens)
                    runner = SimpleNamespace(_eager_registry=registry)
                    with patch.dict(os.environ, {'SGLANG_EAGER_INPUT_NO_COPY': no_copy}):
                        view = EagerRunner.load_batch(runner, batch)
                    self.assertIsNot(view, batch)
                    self.assertFalse(hasattr(view, '_unregistered_contract'))
                    self.assertEqual(view._pfactor_agg_contract, selected)
                    self.assertTrue(view.pd_factor_only_full_batch)
                    self.assertEqual(view._pfactor_legacy_mixed, mixed)
                    self.assertEqual(view.twinstar_prompt_final, [True] * rows)
                    self.assertEqual(eligible(view), selected)
                    if mixed:
                        view.forward_mode = ForwardMode.EXTEND
                        self.assertFalse(eligible(view))
                    self.assertTrue(torch.equal(view.input_ids, batch.input_ids))


if __name__ == '__main__':
    unittest.main()
