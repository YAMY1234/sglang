"""DSV4.1 routed-expert GEMMs past 8 rows (SGLANG_OPT_DSV41_ROUTED_EXPERT_GEMM_MAX_ROWS).

CPU: the tile bound the record is sized by holds for any routing, and the flag reaches the
layer (8 under admission). GPU (SM100 family): at 9-48 rows, including experts chosen by
more than 8 rows (several tiles), the route + GEMM pair + finalize match FlashInfer's routed
MoE, eager and replayed from a CUDA graph.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import dsv41_routed_expert_gemm as reg
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("rows,top_k,experts", [(12, 6, 384), (48, 6, 384), (64, 8, 32), (40, 3, 128)])
def test_max_ctas_bounds_any_routing(rows, top_k, experts):
    generator = torch.Generator().manual_seed(rows * top_k)
    for distinct in (top_k, top_k + 1, 2 * top_k, 64, experts):
        distinct = min(distinct, experts)
        pool = torch.randperm(experts, generator=generator)[:distinct]
        ids = torch.stack([pool[torch.randperm(distinct, generator=generator)[:top_k]] for _ in range(rows)])
        tiles = sum(math.ceil(c / reg.TILE) for c in torch.bincount(ids.flatten()).tolist() if c)
        assert tiles <= reg.max_ctas(rows, top_k, experts)


def _install(monkeypatch, *, max_rows, admission):
    built = []
    monkeypatch.setattr(reg, "RoutedExpertGemm", lambda *a, **kw: built.append(kw) or kw)
    layer = SimpleNamespace(num_local_experts=384, num_experts=384)
    with envs.SGLANG_OPT_DSV41_ROUTED_EXPERT_GEMM.override(
        True
    ), envs.SGLANG_OPT_DSV41_ROUTED_EXPERT_GEMM_ADMIT.override(
        admission
    ), envs.SGLANG_OPT_DSV41_ROUTED_EXPERT_GEMM_MAX_ROWS.override(max_rows):
        assert reg.maybe_install(layer, None, None, None, None, None, top_k=6)
    return built[0]


def test_flag_sets_the_row_bound(monkeypatch):
    assert _install(monkeypatch, max_rows=8, admission=False)["max_rows"] == 8
    assert _install(monkeypatch, max_rows=48, admission=False)["max_rows"] == 48
    # The admission GEMMs keep muLLM's 8-row bound.
    assert _install(monkeypatch, max_rows=48, admission=True)["max_rows"] == 8


# ----------------------------------------------------------------------------- GPU


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer  # noqa: F401
        from cuda.bindings import nvrtc  # noqa: F401
    except ImportError:
        return False
    return True


gpu = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")
TARGET_EXPERTS, TARGET_TOP_K = 384, 6


@pytest.fixture(scope="module")
def target_layer():
    if not _gpu_ready():
        pytest.skip("needs an SM100-family GPU")
    from test_dsv41_draft_routed_expert_gemm import _shuffled_layer

    return _shuffled_layer(torch.Generator().manual_seed(2), TARGET_EXPERTS)


@gpu
@pytest.mark.parametrize("rows", [9, 12, 16, 24, 40, 48])
@pytest.mark.parametrize("distinct", [6, 20, 384])  # 6: every expert takes all rows
@pytest.mark.parametrize("graph", [False, True])
def test_wide_routing_matches_flashinfer(target_layer, rows, distinct, graph):
    from test_dsv41_draft_routed_expert_gemm import _flashinfer, _gathered, _routing

    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    generator = torch.Generator().manual_seed(900 + rows + distinct)
    gemm = reg.RoutedExpertGemm(
        *target_layer, admission=False, top_k=TARGET_TOP_K, finalize=True, max_rows=48
    )
    x = torch.randn(rows, reg.HIDDEN, generator=generator).to(torch.bfloat16).cuda()
    x_q, x_s = flashinfer_mxfp8_quantize(x, False, alignment=reg.HIDDEN)
    x_s = x_s.view(torch.float8_e4m3fn).reshape(rows, -1)
    ids = _routing(rows, distinct, generator, TARGET_EXPERTS, TARGET_TOP_K).cuda()
    weights = torch.softmax(torch.randn(rows, TARGET_TOP_K, generator=generator), -1).cuda()
    output = torch.empty(rows, reg.HIDDEN, dtype=torch.bfloat16, device="cuda")
    assert gemm.supports(x_q, x_s, ids, weights)

    def run():
        routed = gemm(x_q, x_s, ids, weights)
        gemm.finalize(*routed, output)
        return routed

    if graph:
        run()
        torch.cuda.synchronize()
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            routed = run()
        output.zero_()
        for _ in range(2):
            captured.replay()
    else:
        routed = run()
    torch.cuda.synchronize()

    rows_out, _, permuted = routed
    want_rows, want_permuted = _flashinfer(x_q, x_s, ids, weights, target_layer)
    want = _flashinfer(x_q, x_s, ids, weights, target_layer, output=torch.empty_like(output))
    torch.cuda.synchronize()
    torch.testing.assert_close(
        _gathered(rows_out, permuted), _gathered(want_rows, want_permuted), rtol=2**-7, atol=1e-6
    )
    torch.testing.assert_close(output.float(), want.float(), rtol=2**-7, atol=1e-3)


@gpu
def test_eight_rows_keep_the_original_route(target_layer):
    gemm = reg.RoutedExpertGemm(
        *target_layer, admission=False, top_k=TARGET_TOP_K, finalize=True, max_rows=8
    )
    assert gemm.route_wide is None
    rows = 9
    x_q = torch.zeros(rows, reg.HIDDEN, dtype=torch.uint8, device="cuda")
    x_s = torch.zeros(rows, reg.HIDDEN // 32, dtype=torch.uint8, device="cuda")
    ids = torch.zeros(rows, TARGET_TOP_K, dtype=torch.int32, device="cuda")
    weights = torch.zeros(rows, TARGET_TOP_K, dtype=torch.float32, device="cuda")
    assert not gemm.supports(x_q, x_s, ids, weights)
