"""Dispatch and numerics of the SM107 MXFP8 decode table (mxfp8_sm107_dense.DECODE_TABLE)."""

from __future__ import annotations

import sys

import pytest
import torch
from flashinfer import mxfp8_quantize
from sglang.kernels.ops.gemm import mxfp8_sm107_dense as dense
from sglang.srt.layers.quantization import fp8_utils
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

if not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)):
    pytest.skip(
        "the SM107 MXFP8 decode table requires CUDA SM107", allow_module_level=True
    )

# The decode q_a | kv projection, the one tabled shape a TP4 decode step runs.
N, K = 1792, 5120


def _operands(m: int):
    torch.manual_seed(m)
    x = torch.randn((m, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda") / K**0.5
    w_q, w_sf = mxfp8_quantize(w, is_sf_swizzled_layout=True, alignment=32)
    return x, w_q, w_sf


def _linear(x, w_q, w_sf, monkeypatch, table: bool):
    monkeypatch.setenv("SGLANG_OPT_SM107_MXFP8_DENSE_TABLE", "1" if table else "0")
    return fp8_utils.flashinfer_mxfp8_blockscaled_linear(
        x, w_q, w_sf, output_dtype=torch.bfloat16, backend="cute-dsl"
    )


@pytest.mark.parametrize("m", [1, 6, 12, dense.DECODE_TABLE_MIN_M - 1])
def test_small_m_skips_table(m):
    x, w_q, w_sf = _operands(m)
    q, sf = mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
    assert dense.decode_linear(q, sf, w_q, w_sf, torch.bfloat16) is None


@pytest.mark.parametrize("m", [6, 12])
def test_small_m_matches_untabled_path_bitwise(m, monkeypatch):
    x, w_q, w_sf = _operands(m)
    ref = _linear(x, w_q, w_sf, monkeypatch, table=False)
    out = _linear(x, w_q, w_sf, monkeypatch, table=True)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("m", [dense.DECODE_TABLE_MIN_M, 36])
def test_table_m_runs_table_and_matches_untabled(m, monkeypatch):
    x, w_q, w_sf = _operands(m)
    q, sf = mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
    assert dense.decode_linear(q, sf, w_q, w_sf, torch.bfloat16) is not None
    ref = _linear(x, w_q, w_sf, monkeypatch, table=False).float()
    out = _linear(x, w_q, w_sf, monkeypatch, table=True).float()
    # Different kernels, same MXFP8 operands: only the fp32 reduction order differs.
    assert (out - ref).abs().max() <= 1e-2 * ref.abs().max()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
