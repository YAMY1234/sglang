import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.engram import ENGRAM_WKV_PREFETCH_MAX_TOKENS, Engram
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend
from sglang.srt.models import deepseek_v4
from sglang.srt.models.deepseek_v4 import DeepseekV4Model
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _bare_engram() -> Engram:
    engram = Engram.__new__(Engram)
    nn.Module.__init__(engram)
    engram._wkv_out = None
    engram._wkv_ready = None
    engram._wkv_prefetched = False
    return engram


def _wkv(backend, use_mxfp8=True, block_fp8_as_mxfp8=False, ready=True):
    method = Fp8LinearMethod.__new__(Fp8LinearMethod)
    method.use_mxfp8 = use_mxfp8
    method.block_fp8_as_mxfp8 = block_fp8_as_mxfp8
    method.mxfp8_dense_backend = backend
    return SimpleNamespace(quant_method=method, block_fp8_mxfp8_ready=ready)


class TestEngramWkvPrefetchSupport(CustomTestCase):
    """Only a collective-free lookup with a workspace-free WKV GEMM qualifies."""

    def _supports(self, *, buffer=True, shared=True, wkv=None):
        engram = _bare_engram()
        engram._wkv_out = object() if buffer else None
        object.__setattr__(engram, "embed", SimpleNamespace(_shared=shared))
        cutedsl = Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL
        object.__setattr__(engram, "wkv", wkv or _wkv(cutedsl))
        return engram.supports_wkv_prefetch()

    def test_matrix(self):
        cutedsl = Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL
        cutlass = Mxfp8DenseGemmBackend.FLASHINFER_CUTLASS
        self.assertTrue(self._supports())
        self.assertFalse(self._supports(buffer=False))
        self.assertFalse(self._supports(shared=False))
        self.assertFalse(self._supports(wkv=_wkv(cutlass)))
        self.assertFalse(self._supports(wkv=_wkv(None)))
        self.assertFalse(self._supports(wkv=SimpleNamespace(quant_method=object())))
        block = dict(use_mxfp8=False, block_fp8_as_mxfp8=True)
        self.assertTrue(self._supports(wkv=_wkv(cutedsl, **block)))
        self.assertFalse(self._supports(wkv=_wkv(cutedsl, **block, ready=False)))
        self.assertFalse(
            self._supports(wkv=_wkv(cutedsl, use_mxfp8=False, block_fp8_as_mxfp8=False))
        )


def _bare_model(engrams, supported=None):
    model = DeepseekV4Model.__new__(DeepseekV4Model)
    nn.Module.__init__(model)
    layers = [SimpleNamespace(engram=None, hc_cfg="cfg")]
    layers += [SimpleNamespace(engram=e, hc_cfg="cfg") for e in engrams]
    object.__setattr__(model, "layers", layers)
    model.start_layer, model.end_layer = 0, len(layers)
    model.engram_wkv_stream = object()
    model._engram_wkv_prefetch_supported = supported
    return model


class TestEngramWkvPrefetchGate(CustomTestCase):
    def test_init_all_or_none(self):
        yes = SimpleNamespace(supports_wkv_prefetch=lambda: True)
        no = SimpleNamespace(supports_wkv_prefetch=lambda: False)
        self.assertTrue(_bare_model([yes, yes])._init_engram_wkv_prefetch())
        self.assertFalse(_bare_model([yes, no])._init_engram_wkv_prefetch())
        self.assertFalse(_bare_model([])._init_engram_wkv_prefetch())

    def _gate(
        self,
        num_tokens,
        *,
        capturing=True,
        stats_stream=True,
        cp_extend=False,
        tail=None,
        supported=True,
        stream=True,
        attn_dp_size=1,
    ):
        model = _bare_model([], supported=supported)
        if not stream:
            model.engram_wkv_stream = None
        hidden = torch.empty(num_tokens, 4, 8)
        with patch.object(
            deepseek_v4.mhc, "use_stats_stream", lambda *a: stats_stream
        ), patch.object(
            torch.cuda, "is_current_stream_capturing", lambda: capturing
        ), patch.object(
            deepseek_v4,
            "get_parallel",
            lambda: SimpleNamespace(attn_dp_size=attn_dp_size),
        ):
            return model._prefetches_engram_wkv(
                SimpleNamespace(), hidden, cp_extend=cp_extend, tail=tail
            )

    def test_range_and_capture(self):
        self.assertTrue(self._gate(1))
        self.assertTrue(self._gate(ENGRAM_WKV_PREFETCH_MAX_TOKENS))
        self.assertFalse(self._gate(0))
        self.assertFalse(self._gate(ENGRAM_WKV_PREFETCH_MAX_TOKENS + 1))
        self.assertFalse(self._gate(6, capturing=False))
        self.assertFalse(self._gate(6, stats_stream=False))
        self.assertFalse(self._gate(6, cp_extend=True))
        self.assertFalse(self._gate(6, tail=object()))
        self.assertFalse(self._gate(6, supported=False))
        self.assertFalse(self._gate(6, stream=False))
        self.assertFalse(self._gate(6, attn_dp_size=2))


class _DelayedTable(nn.Module):
    """Stands in for the shared host table: a local gather, lagged on its stream."""

    _shared = True

    def __init__(self, rows: int, dim: int):
        super().__init__()
        self.table = torch.randn(rows, dim, device="cuda").to(torch.bfloat16)

    def forward(self, hash_ids, forward_batch=None, *, cp_all_tokens=False):
        # Lag the prefetch stream so a missing wait reads stale values.
        torch.cuda._sleep(1_000_000)
        return self.table[hash_ids]


class _Wkv(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = (torch.randn(out_features, in_features, device="cuda") * 0.01).to(
            torch.bfloat16
        )

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight), None


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestEngramWkvPrefetchParity(CustomTestCase):
    """WKV on the prefetch stream is bit-identical eagerly and in CUDA graphs."""

    def test_matches_inline(self):
        hc_mult, dim, cols, head_dim, rows = 4, 256, 6, 64, 1000
        torch.manual_seed(0)
        engram = _bare_engram()
        engram.embed = _DelayedTable(rows, head_dim)
        engram.wkv = _Wkv(cols * head_dim, (hc_mult + 1) * dim)
        engram.q_weight = torch.randn(hc_mult, dim, device="cuda").to(torch.bfloat16)
        engram.k_weight = torch.randn(hc_mult, dim, device="cuda").to(torch.bfloat16)
        engram.eps, engram.clamp_value = 1e-6, 1e-6
        engram._wkv_out = torch.empty(
            ENGRAM_WKV_PREFETCH_MAX_TOKENS,
            (hc_mult + 1) * dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        engram._wkv_ready = torch.cuda.Event()
        stream = torch.cuda.Stream()

        def run(hashes, hidden, prefetch):
            # The per-layer slice of the model's hash tensor is non-contiguous.
            if prefetch:
                engram.prefetch_wkv(hashes[:, 1], stream=stream)
            return engram(hidden, hashes[:, 1])

        for num_tokens in range(1, ENGRAM_WKV_PREFETCH_MAX_TOKENS + 1):
            hashes = torch.randint(
                0, rows, (num_tokens, 2, cols), dtype=torch.int64, device="cuda"
            )
            hidden = torch.randn(
                num_tokens, hc_mult, dim, device="cuda", dtype=torch.bfloat16
            )
            expected = run(hashes, hidden, False)
            self.assertTrue(torch.isfinite(expected).all())
            torch.testing.assert_close(
                run(hashes, hidden, True), expected, rtol=0, atol=0
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = run(hashes, hidden, True)
            for _ in range(2):
                hashes.random_(0, rows)
                hidden.normal_()
                graph.replay()
                torch.testing.assert_close(
                    out, run(hashes, hidden, False), rtol=0, atol=0
                )
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
