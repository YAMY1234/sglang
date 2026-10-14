"""DSpark draft MoE on the DSV4.1 low-batch routed-expert kernels (SGLANG_OPT_DSV41_DRAFT_FAST_PATHS).

CPU: the draft flag installs the pair and the finalize on draft layers only, at their own
top-k and independent of the target flags; the graph-rows check. GPU (SM100 family): for
128 experts / top-3 the route + GEMM pair + finalize match FlashInfer's routed MoE for 1, 5
and 8 rows, eager and replayed from a CUDA graph (muLLM df8786f6,
test_draft_routing_matches_flashinfer).
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import dsv41_routed_expert_gemm as reg
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DRAFT_EXPERTS = 128
DRAFT_TOP_K = 3


def _layer(draft: bool):
    layer = SimpleNamespace(num_local_experts=DRAFT_EXPERTS, num_experts=DRAFT_EXPERTS)
    if draft:
        layer.dsv41_draft_layer = True
    return layer


def _install(monkeypatch, layer, *, draft_flag, target_flag, finalize_flag=False):
    built = []
    monkeypatch.setattr(reg, "RoutedExpertGemm", lambda *a, **kw: built.append(kw) or kw)
    with envs.SGLANG_OPT_DSV41_DRAFT_FAST_PATHS.override(
        draft_flag
    ), envs.SGLANG_OPT_DSV41_ROUTED_EXPERT_GEMM.override(
        target_flag
    ), envs.SGLANG_OPT_DSV41_ROUTED_EXPERT_GEMM_FINALIZE.override(finalize_flag):
        installed = reg.maybe_install(
            layer, None, None, None, None, None, top_k=DRAFT_TOP_K
        )
    return installed, built


def test_draft_flag_off_installs_nothing_on_draft_layers(monkeypatch):
    installed, built = _install(monkeypatch, _layer(True), draft_flag=False, target_flag=False)
    assert not installed and not built


def test_draft_layers_take_the_pair_and_the_finalize(monkeypatch):
    installed, built = _install(monkeypatch, _layer(True), draft_flag=True, target_flag=False)
    assert installed
    assert built[0]["top_k"] == DRAFT_TOP_K and built[0]["finalize"] is True
    assert built[0]["admission"] is False


def test_draft_flag_leaves_target_layers_alone(monkeypatch):
    installed, built = _install(monkeypatch, _layer(False), draft_flag=True, target_flag=False)
    assert not installed and not built
    # The target flag keeps its own finalize choice.
    installed, built = _install(monkeypatch, _layer(False), draft_flag=True, target_flag=True)
    assert installed and built[0]["finalize"] is False


def test_draft_layer_under_expert_parallelism_keeps_flashinfer(monkeypatch):
    layer = _layer(True)
    layer.num_local_experts = DRAFT_EXPERTS // 4
    installed, built = _install(monkeypatch, layer, draft_flag=True, target_flag=False)
    assert not installed and not built


@pytest.mark.parametrize(
    "capture_bs,rows_per_request,ok",
    [
        ([1, 2, 4, 8], 5, True),  # one request: a 5-row graph
        ([2, 4, 8], 5, False),  # smallest graph has 10 rows
        ([1, 2], 6, True),
        ([], 5, True),  # no CUDA graphs: the draft runs its rows eagerly
    ],
)
def test_draft_graph_rows_check(capture_bs, rows_per_request, ok):
    assert reg.check_draft_graph_rows(capture_bs, rows_per_request) is ok


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


def _shuffled_layer(generator: torch.Generator, experts: int):
    """Random MXFP4 expert weights, shuffled as Mxfp4FlashinferTrtllmMoEMethod does."""
    from flashinfer.fp4_quantization import block_scale_interleave
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    e, h, p, s = experts, reg.HIDDEN, reg.PADDED, reg.SLICE
    w13 = torch.randint(0, 256, (e, 2 * p, h // 2), dtype=torch.uint8, generator=generator)
    w13_s = torch.randint(118, 126, (e, 2 * p, h // 32), dtype=torch.uint8, generator=generator)
    w13.view(e, 2, p, -1)[:, :, s:] = 0
    w2 = torch.randint(0, 256, (e, h, p // 2), dtype=torch.uint8, generator=generator)
    w2_s = torch.randint(118, 126, (e, h, p // 32), dtype=torch.uint8, generator=generator)
    w2[:, :, s // 2 :] = 0
    w13, w13_s, w2, w2_s = (t.cuda() for t in (w13, w13_s, w2, w2_s))
    cache: dict = {}
    g1, g1s, g2, g2s = [], [], [], []
    for i in range(e):
        perm = _maybe_get_cached_w3_w1_permute_indices(cache, w13[i], 128)
        g1.append(w13[i][perm.cuda()].contiguous())
        perm = _maybe_get_cached_w3_w1_permute_indices(cache, w13_s[i], 128, num_elts_per_sf=16)
        g1s.append(block_scale_interleave(w13_s[i][perm.cuda()].contiguous()))
        perm = get_w2_permute_indices_with_cache(cache, w2[i], 128)
        g2.append(w2[i][perm.cuda()].contiguous())
        perm = get_w2_permute_indices_with_cache(cache, w2_s[i], 128, num_elts_per_sf=16)
        g2s.append(block_scale_interleave(w2_s[i][perm.cuda()].contiguous()))
    w13 = torch.stack(g1)
    w13_s = torch.stack(g1s).view(torch.float8_e4m3fn).reshape(e, 2 * p, -1)
    w2 = torch.stack(g2)
    w2_s = torch.stack(g2s).view(torch.float8_e4m3fn).reshape(e, h, -1)
    clamp = torch.full((e,), 10.0, dtype=torch.float32, device="cuda")
    return w13, w13_s, w2, w2_s, clamp


def _routing(rows, distinct, generator, experts, top_k):
    distinct = min(max(distinct, top_k), experts)
    pool = torch.randperm(experts, generator=generator)[:distinct]
    ids = [pool[torch.randperm(distinct, generator=generator)[:top_k]] for _ in range(rows)]
    return torch.stack(ids).to(torch.int32)


def _flashinfer(x_q, x_s, ids, weights, layer, output=None):
    """FlashInfer's routed MoE as SGLang serves it: GEMM2 rows and the slot -> row map,
    or (``output`` given) the finalized output."""
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe

    from sglang.srt.layers.moe.utils import RoutingMethodType

    w13, s13, w2, s2, clamp = layer
    experts = w13.size(0)
    result = trtllm_fp4_block_scale_routed_moe(
        topk_ids=(ids, weights),
        routing_bias=None,
        routing_method_type=int(RoutingMethodType.TopK),
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        hidden_states=x_q,
        hidden_states_scale=x_s,
        gemm1_weights=w13,
        gemm1_weights_scale=s13,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=clamp,
        gemm2_weights=w2,
        gemm2_weights_scale=s2,
        gemm2_bias=None,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        num_experts=experts,
        top_k=ids.size(1),
        intermediate_size=reg.PADDED,
        local_expert_offset=0,
        local_num_experts=experts,
        do_finalize=output is not None,
        tune_max_num_tokens=8,
        output=output,
        enable_pdl=True,
    )
    if output is not None:
        return output
    return result[0], result[2].view(ids.shape)


def _gathered(rows_out, permuted):
    return rows_out[permuted.long()].float()


@pytest.fixture(scope="module")
def draft_layer():
    if not _gpu_ready():
        pytest.skip("needs an SM100-family GPU")
    return _shuffled_layer(torch.Generator().manual_seed(1), DRAFT_EXPERTS)


@gpu
@pytest.mark.parametrize("rows", [1, 5, 8])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("admission", [False, True])
def test_draft_routing_matches_flashinfer(draft_layer, rows, graph, admission):
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    generator = torch.Generator().manual_seed(500 + rows)
    gemm = reg.RoutedExpertGemm(
        *draft_layer, admission=admission, top_k=DRAFT_TOP_K, finalize=True
    )
    x = torch.randn(rows, reg.HIDDEN, generator=generator).to(torch.bfloat16).cuda()
    x_q, x_s = flashinfer_mxfp8_quantize(x, False, alignment=reg.HIDDEN)
    x_s = x_s.view(torch.float8_e4m3fn).reshape(rows, -1)
    ids = _routing(rows, 12, generator, DRAFT_EXPERTS, DRAFT_TOP_K).cuda()
    weights = torch.softmax(torch.randn(rows, DRAFT_TOP_K, generator=generator), -1).cuda()
    output = torch.empty(rows, reg.HIDDEN, dtype=torch.bfloat16, device="cuda")
    assert gemm.supports(x_q, x_s, ids, weights)
    # Top-6 ids are not this layer's routing.
    assert not gemm.supports(x_q, x_s, torch.zeros(rows, 6, dtype=torch.int32, device="cuda"), weights)

    def run():
        routed = gemm(x_q, x_s, ids, weights)
        gemm.finalize(*routed, output)
        return routed

    if graph:
        run()  # warm-up outside capture
        torch.cuda.synchronize()
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            routed = run()
        output.zero_()
        # Several replays: the admission counters alternate halves per launch.
        for _ in range(3):
            captured.replay()
    else:
        routed = run()
    torch.cuda.synchronize()

    rows_out, _, permuted = routed
    want_rows, want_permuted = _flashinfer(x_q, x_s, ids, weights, draft_layer)
    want = _flashinfer(x_q, x_s, ids, weights, draft_layer, output=torch.empty_like(output))
    torch.cuda.synchronize()
    # Same MMA order, SwiGLU and quantization as the TRT-LLM-gen tactics; one BF16 ulp.
    torch.testing.assert_close(
        _gathered(rows_out, permuted), _gathered(want_rows, want_permuted), rtol=2**-7, atol=1e-6
    )
    torch.testing.assert_close(output.float(), want.float(), rtol=2**-7, atol=1e-3)
