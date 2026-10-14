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

    def test_factor_capture_indices_are_refreshed_outside_graph(self):
        md = load_cpu_metadata()
        path = ROOT / 'python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py'
        owner = next(n for n in ast.parse(path.read_text()).body
                     if isinstance(n, ast.ClassDef) and n.name == 'Mamba2AttnBackend')
        methods = [n for n in owner.body if getattr(n, 'name', '') in
                   ('init_forward_metadata_out_graph', 'init_forward_metadata')]
        node = ast.ClassDef(name='Backend', bases=[], keywords=[], body=methods, decorator_list=[])
        tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        scope = dict(Mamba2Metadata=md.Mamba2Metadata, ForwardBatch=object)
        exec(compile(tree, str(path), 'exec'), scope)
        backend = scope['Backend']()
        backend.ltn_factored_decode = True
        backend.mamba_chunk_size = 64
        slots = torch.tensor([3, 8], dtype=torch.int32)
        base = md.ForwardMetadata(query_start_loc=torch.arange(3, dtype=torch.int32),
                                  mamba_cache_indices=slots)
        backend._replay_metadata = lambda *a, **kw: base
        backend._forward_metadata = lambda *a: base
        fb = types.SimpleNamespace(batch_size=2, req_pool_indices=torch.tensor([1, 2]),
            forward_mode=types.SimpleNamespace(is_target_verify=lambda: False),
            spec_info=None, seq_lens_cpu=torch.tensor([5, 6]), seq_lens=torch.tensor([5, 6]),
            extend_num_tokens=None)
        backend.init_forward_metadata_out_graph(fb, in_capture=True)
        self.assertEqual(slots.tolist(), [-1, -1])
        captured = backend.forward_metadata.mamba_cache_indices
        # Backend replay preparation owns refresh; no captured poison operation.
        slots.copy_(torch.tensor([8, -1], dtype=torch.int32))
        backend.init_forward_metadata_out_graph(fb, in_capture=False)
        self.assertEqual(captured.data_ptr(), backend.forward_metadata.mamba_cache_indices.data_ptr())
        self.assertEqual(captured.tolist(), [8, -1])
        fb._lightning_dummy_batch = True
        backend.init_forward_metadata(fb)
        self.assertEqual(slots.tolist(), [-1, -1])

    def test_factor_dummy_forward_uses_factor_decode(self):
        path = ROOT / 'python/sglang/srt/models/lightning_duet/engine.py'
        owner = next(n for n in ast.parse(path.read_text()).body
                     if isinstance(n, ast.ClassDef) and n.name == 'Runtime')
        method = next(n for n in owner.body if getattr(n, 'name', '') == 'forward')
        scope = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
        runtime = types.SimpleNamespace(factored_decode=True,
            _decode_forward=lambda *a: 'factor', original_forward=lambda *a: 'native')
        fb = types.SimpleNamespace(_lightning_dummy_batch=True,
            forward_mode=types.SimpleNamespace(is_decode=lambda: True))
        self.assertEqual(scope['forward'](runtime, None, None, fb), 'factor')
        runtime.factored_decode = False
        self.assertEqual(scope['forward'](runtime, None, None, fb), 'native')


if __name__ == '__main__':
    unittest.main()
