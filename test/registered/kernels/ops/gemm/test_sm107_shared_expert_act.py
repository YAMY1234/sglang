"""Shared-expert act_and_mul in the SM107 gate_up epilogue: bitwise to the stock GEMM + act_and_mul pair.

The act_and_mul reference is the JIT kernel's SiLU as it was before #48942, x / (1 + expf(-x)) * up built with
--use_fast_math, compiled here from that source (the current kernel differs on about 2% of all finite pairs).
The exhaustive case reaches the epilogue with every finite (gate, up) BF16 pair exactly: launch c's row m is one-hot
at K column 16c + m, so gate_up[m, r] = W[r, 16c + m]. A copy built with div.full in place of div.approx is the
control that must differ (near gate -88, 1 + exp(-gate) is past 2^126 and the two divides disagree).
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
        reason="the fused epilogue rides the SM107 routes",
    ),
]

_INTER, _K = 640, 2560
_ROWS = 16
_LAUNCHES = _K // _ROWS  # 160 one-hot launches cover every K column
_PAIRS = _K * _INTER


_FAST_MATH_ACT_CUDA = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

__global__ void silu_and_mul_kernel(const __nv_bfloat16* gate_up, __nv_bfloat16* out, int64_t rows, int64_t inter) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= rows * inter) return;
  const int64_t r = i / inter, c = i - r * inter;
  const float gate = __bfloat162float(gate_up[r * 2 * inter + c]);
  const float up = __bfloat162float(gate_up[r * 2 * inter + inter + c]);
  out[i] = __float2bfloat16(gate / (1.0f + expf(-gate)) * up);
}

void silu_and_mul(torch::Tensor gate_up, torch::Tensor out) {
  const int64_t rows = gate_up.size(0), inter = gate_up.size(1) / 2;
  const int64_t threads = 256, blocks = (rows * inter + threads - 1) / threads;
  silu_and_mul_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(gate_up.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), rows, inter);
}
"""


def _fast_math_silu_and_mul():
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="sm107_fused_act_reference_silu_and_mul",
        cpp_sources="void silu_and_mul(torch::Tensor gate_up, torch::Tensor out);",
        cuda_sources=_FAST_MATH_ACT_CUDA,
        functions=["silu_and_mul"],
        extra_cuda_cflags=["--use_fast_math"],
    ).silu_and_mul


def _full_divide_ptx(module) -> str:
    return module.SILU_MUL_PTX.replace("div.approx.ftz.f32", "div.full.ftz.f32")


def _bits(t):
    return t.contiguous().view(torch.int16)


@pytest.fixture(scope="module")
def env():
    from sglang.srt.layers.quantization import sm107_shared_expert_act, unquant

    exec_ctx = SimpleNamespace(
        kernel=SimpleNamespace(bf16_gemm_backend="cutedsl"),
        deterministic=SimpleNamespace(enable_deterministic_inference=False),
    )
    platform = SimpleNamespace(is_sm100=True, device_sm=107)
    with (
        mock.patch.object(unquant, "get_exec", lambda: exec_ctx),
        mock.patch.object(unquant, "get_platform", lambda: platform),
        # The routes are a runtime child: read only with the runtime master on.
        envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
        envs.SGLANG_OPT_SM107_BF16_SMALLM_ROUTES.override(True),
        envs.SGLANG_ENABLE_SM107_SHARED_EXPERT_FUSED_ACT.override(True),
    ):
        unquant.initialize_bf16_gemm_config()
    unquant.precompile_splitk_tactics()  # the load-time checks: routes and fused act serve only after them
    assert unquant._sm107_smallm_on and sm107_shared_expert_act.is_enabled()
    yield SimpleNamespace(
        unquant=unquant, fused=sm107_shared_expert_act, act=_fast_math_silu_and_mul()
    )
    unquant._sm107_smallm_on = False
    sm107_shared_expert_act._enabled = False


def _stock_pair(env, x, w, out):
    with mock.patch.object(env.unquant, "_sm107_smallm_on", False):
        gate_up = env.unquant._bf16_gemm_dispatch_impl(x, w, None)
    env.act(gate_up, out)
    return out


def _capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


class _OneHot:
    """One weight build = 2,560 x 640 (gate, up) pairs; pair q lands at flat output index q."""

    def __init__(self, env):
        self.W = torch.zeros(2 * _INTER, _K, dtype=torch.bfloat16, device="cuda")
        self.Wp = torch.zeros_like(self.W)
        rows = torch.arange(_ROWS, device="cuda")
        self.x = []
        for c in range(_LAUNCHES):
            x = torch.zeros(_ROWS, _K, dtype=torch.bfloat16, device="cuda")
            x[rows, _ROWS * c + rows] = 1.0
            self.x.append(x)
        anchor = torch.empty(_ROWS, 2 * _INTER, dtype=torch.bfloat16, device="cuda")
        self.out = {
            name: torch.empty(
                _LAUNCHES, _ROWS, _INTER, dtype=torch.bfloat16, device="cuda"
            )
            for name in ("ref", "approx", "full")
        }
        ptx = {"approx": env.fused.SILU_MUL_PTX, "full": _full_divide_ptx(env.fused)}
        self.graphs = [
            _capture(
                lambda: [
                    _stock_pair(env, self.x[c], self.W, self.out["ref"][c])
                    for c in range(_LAUNCHES)
                ]
            )
        ] + [
            _capture(
                lambda name=name: [
                    env.fused.run(
                        self.x[c],
                        self.Wp,
                        self.out[name][c],
                        anchor,
                        silu_mul_ptx=ptx[name],
                    )
                    for c in range(_LAUNCHES)
                ]
            )
            for name in ("approx", "full")
        ]

    def load(self, gate, up):
        self.W[:_INTER].copy_(gate.view(_K, _INTER).t())
        self.W[_INTER:].copy_(up.view(_K, _INTER).t())
        self.Wp.view(20, 2, 32, _K).copy_(
            self.W.view(2, 20, 32, _K).transpose(0, 1)
        )  # = permute_gate_up

    def run(self):
        for out in self.out.values():
            out.fill_(float("nan"))
        for graph in self.graphs:
            graph.replay()

    def differ(self, name):
        return _bits(self.out["ref"]).view(-1) != _bits(self.out[name]).view(-1)


def test_every_finite_pair_bitwise(env):
    h = _OneHot(env)
    assert torch.equal(h.Wp, env.fused.permute_gate_up(h.W))
    v = (
        torch.arange(-32768, 32768, dtype=torch.int32, device="cuda")
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    finite = v[torch.isfinite(v.float())]
    n = finite.numel()
    total = n * n
    q0 = torch.arange(_PAIRS, dtype=torch.int64, device="cuda")
    counts = torch.zeros(2, dtype=torch.int64, device="cuda")
    for b in range(-(-total // _PAIRS)):
        q = q0 + b * _PAIRS
        valid = q < total
        q = torch.where(valid, q, torch.zeros_like(q))
        h.load(finite[q // n], finite[q % n])
        h.run()
        counts[0] += (h.differ("approx") & valid).sum()
        counts[1] += (h.differ("full") & valid).sum()
    approx, full = counts.tolist()
    print("pairs", total, "approx differ", approx, "full differ", full)
    assert approx == 0
    assert (
        full > 0
    )  # the control fires: the comparison sees a divide that is not the stock one


def test_seeded_edge_and_rows_1_to_16(env):
    g = torch.Generator(device="cuda").manual_seed(1234)
    w = (torch.randn(2 * _INTER, _K, generator=g, device="cuda") * 0.02).to(
        torch.bfloat16
    )
    w[600, 5], w[_INTER + 600, 5] = (
        -87.5,
        -80.0,
    )  # stock gives +0 there; div.full does not
    wp = env.fused.permute_gate_up(w)
    anchor = torch.empty(_ROWS, 2 * _INTER, dtype=torch.bfloat16, device="cuda")
    x = torch.zeros(_ROWS, _K, dtype=torch.bfloat16, device="cuda")
    x[torch.arange(_ROWS), torch.arange(_ROWS)] = 1.0
    ref = _stock_pair(
        env, x, w, torch.empty(_ROWS, _INTER, dtype=torch.bfloat16, device="cuda")
    )
    assert int(_bits(ref)[5, 600]) == 0
    for name, ptx, want in (
        ("approx", env.fused.SILU_MUL_PTX, 0),
        ("full", _full_divide_ptx(env.fused), None),
    ):
        got = torch.full_like(ref, float("nan"))
        env.fused.run(x, wp, got, anchor, silu_mul_ptx=ptx)
        differing = int((_bits(ref) != _bits(got)).sum())
        assert differing == 0 if want == 0 else differing >= 1, (name, differing)

    for kind in ("randn", "outliers", "hot"):
        for rows in range(1, _ROWS + 1):
            x = torch.randn(rows, _K, generator=g, device="cuda")
            if kind == "outliers":
                x[:, torch.randint(0, _K, (8,), generator=g, device="cuda")] *= 30.0
            elif kind == "hot":
                x = x * 8.0
            x = x.to(torch.bfloat16)
            ref = _stock_pair(
                env,
                x,
                w,
                torch.empty(rows, _INTER, dtype=torch.bfloat16, device="cuda"),
            )
            got = torch.full_like(ref, float("nan"))
            env.fused.run(x, wp, got, anchor[:rows])
            assert torch.equal(_bits(ref), _bits(got)), (kind, rows, "eager")
            graph = _capture(lambda: env.fused.run(x, wp, got, anchor[:rows]))
            got.fill_(float("nan"))
            graph.replay()
            assert torch.equal(_bits(ref), _bits(got)), (kind, rows, "graph")
            del graph

    flipped = got.clone()
    flipped.view(-1).view(torch.int16)[3] ^= 1
    assert int((_bits(got) != _bits(flipped)).sum()) == 1


def test_dispatch_through_qwen2_moe_mlp(env):
    """Qwen2MoeMLP.forward with the switch on equals it with the switch off, eagerly and under graph replay; the fused
    kernel runs at 1-16 rows and not at 17."""
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
    from sglang.test.layer_ut_utils import init_single_process_dist

    init_single_process_dist()
    with set_default_torch_dtype(torch.bfloat16):
        mlp = Qwen2MoeMLP(_K, _INTER, "silu", quant_config=None, reduce_results=False)
    mlp = mlp.cuda()
    g = torch.Generator(device="cuda").manual_seed(7)
    with torch.no_grad():
        for layer in (mlp.gate_up_proj, mlp.down_proj):
            layer.weight.copy_(
                torch.randn(layer.weight.shape, generator=g, device="cuda") * 0.02
            )
    fused_rows = []
    real_run = env.fused.run

    def counting_run(x, *args, **kwargs):
        fused_rows.append(x.shape[0])
        return real_run(x, *args, **kwargs)

    def stock(x):
        with mock.patch.object(env.fused, "_enabled", False):
            return mlp(x)

    with mock.patch.object(env.fused, "run", counting_run), torch.inference_mode():
        for rows in range(1, _ROWS + 2):
            x = torch.randn(rows, _K, generator=g, device="cuda").to(torch.bfloat16)
            ref = stock(x)
            fused_rows.clear()
            got = mlp(x)
            assert torch.equal(_bits(ref), _bits(got)), (rows, "eager")
            assert (rows in fused_rows) == (rows <= _ROWS), (rows, fused_rows)
            state = {}
            graph = _capture(lambda: state.update(out=mlp(x)))
            x.copy_(
                torch.randn(rows, _K, generator=g, device="cuda").to(torch.bfloat16)
            )
            state["out"].fill_(float("nan"))
            graph.replay()
            assert torch.equal(_bits(stock(x)), _bits(state["out"])), (rows, "graph")
            del graph


def test_shared_down_tgv_rows(env):
    """With down_tgv set (SGLANG_ENABLE_SHARED_EXPERT_DOWN_TGV), the shared expert's down_proj runs the TGV kernel at
    4, 8 and 16 rows and the stock dispatch at every other row."""
    from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import cutedsl_bf16_gemm
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
    from sglang.test.layer_ut_utils import init_single_process_dist

    init_single_process_dist()
    with set_default_torch_dtype(torch.bfloat16):
        mlp = Qwen2MoeMLP(_K, _INTER, "silu", quant_config=None, reduce_results=False)
    mlp = mlp.cuda()
    g = torch.Generator(device="cuda").manual_seed(11)
    with torch.no_grad():
        mlp.down_proj.weight.copy_(
            torch.randn(mlp.down_proj.weight.shape, generator=g, device="cuda") * 0.02
        )
    with torch.inference_mode():
        for rows in range(1, _ROWS + 2):
            act = torch.randn(rows, _INTER, generator=g, device="cuda").to(
                torch.bfloat16
            )
            mlp.down_tgv = False
            stock = mlp._down(act)
            mlp.down_tgv = True
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                got = mlp._down(act)
                torch.cuda.synchronize()
            kernels = [e.name for e in prof.events() if "Tgv" in e.name]
            if rows in (4, 8, 16):
                want = cutedsl_bf16_gemm(act, mlp.down_proj.weight)
                assert kernels, (rows, "no TGV kernel")
            else:
                want = stock
            assert torch.equal(_bits(want), _bits(got)), rows


def test_load_check_catches_a_non_stock_divide(env):
    """The load-time self-check passes on the shipped epilogue and fails on a div.full copy (only the edge pair shows
    the difference)."""
    real_run = env.fused.run
    full = _full_divide_ptx(env.fused)

    def stock_gemm(x, w):
        with mock.patch.object(env.unquant, "_sm107_smallm_on", False):
            return env.unquant._bf16_gemm_dispatch_impl(x, w, None)

    def full_divide_run(*args, **kwargs):
        return real_run(*args, **{**kwargs, "silu_mul_ptx": full})

    assert env.fused.precompile(stock_gemm=stock_gemm)
    try:
        with mock.patch.object(env.fused, "run", full_divide_run):
            assert not env.fused.precompile(stock_gemm=stock_gemm)
        assert not env.fused.is_enabled()
    finally:
        env.fused._enabled = True
