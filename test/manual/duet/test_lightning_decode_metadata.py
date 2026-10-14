"""CPU regression: prompt token slices must enter native Mamba decode."""
import ast
import copy
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]


def load_cpu_metadata():
    path = ROOT / 'python/sglang/srt/layers/attention/mamba/mamba2_metadata.py'
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.ImportFrom) and n.module.startswith('sglang.'))]
    module = types.ModuleType('ltnf_cpu_metadata')
    module.ForwardBatch = object
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


class DecodeMetadataTest(unittest.TestCase):
    def test_sliced_prompt_uses_native_decode(self):
        path = ROOT / 'python/sglang/srt/models/lightning_duet/engine.py'
        owner = next(n for n in ast.parse(path.read_text()).body
                     if isinstance(n, ast.ClassDef) and n.name == 'Runtime')
        method = next(n for n in owner.body if getattr(n, 'name', '') == 'slice_batch')
        mode = types.SimpleNamespace(is_target_verify=lambda: False)
        scope = dict(torch=torch, copy=copy,
                     ForwardMode=types.SimpleNamespace(DECODE=mode, EXTEND=mode))
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
        fb = types.SimpleNamespace(
            input_ids=torch.tensor([4, 5, 6]), positions=torch.arange(3),
            req_pool_indices=torch.tensor([7]), seq_lens=torch.tensor([3]),
            orig_seq_lens=None, out_cache_loc=torch.arange(3),
            out_cache_loc_virtual=None, extend_num_tokens=3,
        )
        md = load_cpu_metadata()
        base = md.ForwardMetadata(query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
                                  mamba_cache_indices=torch.tensor([9], dtype=torch.int32))
        for decode in (False, True):
            with self.subTest(decode=decode):
                sliced = scope['slice_batch'](None, fb, 0, 0, 1, 1, decode=decode)
                actual = md.Mamba2Metadata.prepare_mixed(base, 64, sliced)
                self.assertEqual(actual.num_decodes, int(decode))
                self.assertEqual(actual.num_prefills, int(not decode))
                torch.testing.assert_close(actual.mamba_cache_indices, base.mamba_cache_indices)
        self.assertEqual(fb.extend_num_tokens, 3)


if __name__ == '__main__':
    unittest.main()
