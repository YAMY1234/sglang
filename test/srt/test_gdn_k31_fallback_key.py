"""mixed_cholqr_fallbacks device key (docs/223 §9, R1 prerequisite): a bare "cuda" resolves to cuda:<current>.

Before the fix the counter was stored under the tensor's device (cuda:0) and looked up under torch.device("cuda"),
which never matched, so the K31_CHOLQR_MIXED_FALLBACK log always read 0. CPU only (current_device patched).
"""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]


def _load():
    name = "sglang.srt.layers.attention.linear.kernels.gdn_prefill_reference"
    for pkg in ("sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.attention",
                "sglang.srt.layers.attention.linear", "sglang.srt.layers.attention.linear.kernels"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [str(ROOT / "python" / pkg.replace(".", "/"))]
            sys.modules[pkg] = m
    spec = importlib.util.spec_from_file_location(name, ROOT / "python" / (name.replace(".", "/") + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load()


class FallbackKeyTest(unittest.TestCase):
    def test_bare_cuda_resolves_to_current_index(self):
        with mock.patch.object(torch.cuda, "current_device", return_value=0):
            self.assertEqual(ref._counter_key("cuda"), torch.device("cuda", 0))
            self.assertEqual(ref._counter_key(torch.device("cuda")), torch.device("cuda", 0))
        self.assertEqual(ref._counter_key("cuda:3"), torch.device("cuda", 3))
        self.assertEqual(ref._counter_key("cpu"), torch.device("cpu"))

    def test_lookup_finds_counter_stored_under_indexed_device(self):
        ref._MIXED_FALLBACKS.clear()
        ref._MIXED_FALLBACKS[torch.device("cuda", 0)] = torch.tensor(5)
        with mock.patch.object(torch.cuda, "current_device", return_value=0):
            self.assertEqual(ref.mixed_cholqr_fallbacks("cuda"), 5)
        ref._MIXED_FALLBACKS.clear()


if __name__ == "__main__":
    unittest.main()
