"""Descriptor integrity for private P pages to ordinary D KV buffers."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[4] / 'python/sglang/srt/disaggregation/flashnext_idle_pd.py'
spec = importlib.util.spec_from_file_location('idle_pd', path)
idle_pd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(idle_pd)


class Pool:
    def __init__(self, layers, offset):
        self.layers = layers
        self.ids = layers * 2
        # Distinct K/V addresses make swapped groups visible.
        self.kv = ([offset + i * 100 for i in range(len(self.ids))],
                   [64000] * len(self.ids), [640] * len(self.ids))
        self.compressed = ([offset + 9000 + i * 100 for i in range(len(layers))],
                           [16000] * len(layers), [160] * len(layers))

    def get_contiguous_buf_infos(self): return self.kv
    def get_kv_layer_ids(self): return self.ids
    def get_qsa_compressed_state_buf_infos(self): return self.compressed
    def get_qsa_compressed_state_layer_ids(self): return self.layers


class MaterializedTransfer(unittest.TestCase):
    def test_private_source_and_full_destination_have_matching_wire_order(self):
        prefill = SimpleNamespace(idle_only=True, deep=Pool([31,35,39,43,47], 10000))
        decode = Pool(list(range(3,48,4)), 30000)
        source = idle_pd.deep_buffers(prefill)
        destination = idle_pd.deep_buffers(decode)
        self.assertEqual(source[0][1], destination[0][1])
        self.assertEqual(source[1][1], destination[1][1])
        self.assertEqual(destination[0][0][0], [30700,30800,30900,31000,31100,
                                               31900,32000,32100,32200,32300])
        self.assertEqual(len(source[0][0][0]), 10)
        self.assertEqual(len(source[1][0][0]), 5)
        # Different allocation sizes and physical page ids are intentional.
        # Only the per-page wire length and the global layer occurrence match.
        self.assertEqual(source[0][0][2], destination[0][0][2])

    def test_incomplete_or_reordered_layers_fail_closed(self):
        for layers in ([31,35,39,43], [35,31,39,43,47]):
            with self.assertRaises(ValueError): idle_pd.deep_buffers(Pool(layers,10000))
        pool=Pool([31,35,39,43,47],10000);pool.kv[0].pop()
        with self.assertRaises(ValueError):idle_pd.deep_buffers(pool)

    def test_flagoff_and_normal_decode_do_not_enable_extra_state(self):
        with patch.dict('os.environ', {}, clear=True):
            self.assertFalse(idle_pd.enabled(SimpleNamespace()))
        with patch.dict('os.environ', {'TWINSTAR_IDLE_PD_FULL_KV':'1','TWINSTAR_FULLSTACK':'0'},clear=True):
            self.assertFalse(idle_pd.enabled(SimpleNamespace()))
        with patch.dict('os.environ', {'TWINSTAR_IDLE_PD_FULL_KV':'1','TWINSTAR_FULLSTACK':'1'},clear=True):
            self.assertTrue(idle_pd.enabled(SimpleNamespace()))


if __name__ == '__main__': unittest.main()
