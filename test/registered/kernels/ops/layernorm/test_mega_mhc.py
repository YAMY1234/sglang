"""DeepGEMM mega mHC against the shifted-boundary equations."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


def test_carried_coefficients_follow_rows_and_are_dropped_on_residual_rewrite():
    residual = torch.arange(48).view(4, 4, 3)
    coefficients = (torch.arange(16).view(4, 4),) * 3
    state = mhc.HcState(residual, coeffs=coefficients)
    rows = torch.tensor([3, 1])
    sliced = state.take_rows(lambda t: t.index_select(0, rows))
    for actual, original in zip(sliced.coeffs, coefficients):
        torch.testing.assert_close(actual, original[rows])
    assert state.with_residual(residual + 1).coeffs is None
    state.release()
    assert state.coeffs is None


@pytest.mark.parametrize("rows", [1, 32, 1152, 8192])
def test_mega_shifted_boundary_and_capture(monkeypatch, rows):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("DeepGEMM mega mHC requires SM10x")
    deep_gemm = pytest.importorskip("deep_gemm")
    if not callable(getattr(deep_gemm, "mega_mhc", None)):
        pytest.skip("DeepGEMM does not export mega_mhc")
    h, mult = 5120, 4
    torch.manual_seed(rows)
    cfg = mhc.HcConfig(mult, 20, 1e-6, 1e-20, h, True, False)
    norm = RMSNorm(h, eps=cfg.rms_eps).to(device="cuda", dtype=torch.bfloat16)
    nxt = mhc.HcSubLayer(
        cfg, torch.randn(24, mult * h, device="cuda") * 0.01,
        torch.full((3,), 0.02, device="cuda"), torch.zeros(24, device="cuda"), norm,
    )
    y = torch.randn(rows, h, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(rows, mult, h, device="cuda", dtype=torch.bfloat16)
    pre = torch.rand(rows, mult, device="cuda")
    post = torch.rand_like(pre) * 2
    comb = torch.softmax(torch.randn(rows, mult, mult, device="cuda"), dim=-1)
    triplet = pre, post, comb
    stream = torch.cuda.Stream()
    mhc.warm_mega_mhc_streams(nxt, [None, stream])
    with torch.cuda.stream(stream):
        mhc._mega_post(cfg, nxt, norm, y, residual, triplet)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        state = mhc._mega_post(cfg, nxt, norm, y, residual, triplet)
    # Replay must use the refreshed tensors rather than warmup contents.
    y.mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    r = post.double()[..., None] * y.double()[:, None]
    r += torch.einsum("nij,nih->njh", comb.double(), residual.double())
    torch.testing.assert_close(state.residual.double(), r, atol=0.04, rtol=0.008)
    stored = state.residual.double()
    flat = stored.flatten(1)
    mix = (flat @ nxt.fn.double().T) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + cfg.rms_eps)
    scale, base = nxt.scale.double(), nxt.base.double()
    expected_pre = torch.sigmoid(mix[:, :mult] * scale[0] + base[:mult]) + cfg.eps
    expected_post = 2 * torch.sigmoid(mix[:, mult:2*mult] * scale[1] + base[mult:2*mult])
    expected_comb = torch.softmax((mix[:, 2*mult:] * scale[2] + base[2*mult:]).view(-1, mult, mult), -1) + cfg.eps
    expected_comb /= expected_comb.sum(-2, keepdim=True) + cfg.eps
    for _ in range(cfg.sinkhorn_iters - 1):
        expected_comb /= expected_comb.sum(-1, keepdim=True) + cfg.eps
        expected_comb /= expected_comb.sum(-2, keepdim=True) + cfg.eps
    for actual, expected in zip(state.coeffs, (expected_pre, expected_post, expected_comb)):
        torch.testing.assert_close(actual.double(), expected, atol=2e-4, rtol=2e-4)
    collapsed = (pre.double()[..., None] * stored).sum(1)
    expected = collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + norm.variance_epsilon)
    expected *= norm.weight.double()
    torch.testing.assert_close(state.input.rows.double(), expected, atol=0.02, rtol=0.008)

    cold = torch.cuda.Stream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=cold.cuda_stream))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert not mhc._mega_mhc_stream_ready()
