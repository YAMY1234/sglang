"""E/D modes cannot change component identity or leak TF32 into the base."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

path=Path(__file__).resolve().parents[4]/"python/sglang/srt/mem_cache/flashnext_scheme_c.py"
spec=importlib.util.spec_from_file_location("precision_codec",path)
codec=importlib.util.module_from_spec(spec);sys.modules[spec.name]=codec;spec.loader.exec_module(codec)


class Tiny(codec.FlashNextSchemeCCodec):
    WIDTH=32
    RANK=16
    SPIKES=4


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(398)
        self.codec=Tiny(device="cpu")
        for name in ("E","D","mean"):
            self.codec.load(name,torch.randn_like(getattr(self.codec,name)))
        self.codec.finalize()

    def test_default_matches_original_fp32_even_when_tf32_global_is_enabled(self):
        x=torch.randn(7,32)
        old=torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=True
            actual=self.codec.project(x,"E")
            self.assertTrue(torch.equal(actual,x@self.codec.E.T))
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32=old

    def test_backend_restored_when_projection_fails(self):
        old=torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=False
            self.codec.set_compute_precision("tf32")
            with self.assertRaises(RuntimeError):
                self.codec.project(torch.randn(1,31),"E")
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32=old

    def test_modes_preserve_components_and_bf16_uses_rounded_inputs(self):
        original={name:getattr(self.codec,name).clone() for name in ("E","D","mean")}
        x=torch.randn(7,32)
        self.codec.set_compute_precision("bf16")
        expected=x.bfloat16().float()@self.codec.E.bfloat16().float().T
        self.assertTrue(torch.equal(self.codec.project(x,"E"),expected))
        self.codec.set_compute_precision("fp32")
        self.assertIsNone(self.codec.E_bf16)
        for name,value in original.items():
            self.assertTrue(torch.equal(value,getattr(self.codec,name)))
        with self.assertRaises(ValueError):
            self.codec.set_compute_precision("fp16")


if __name__=="__main__":
    unittest.main()
