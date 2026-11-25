"""T64: the parallel AIR top-p renorm ("AIR Tier1 v7") at the mid-curve decode's
EAGLE target-verify shapes, bit for bit against flashinfer's deterministic AIR renorm.

Mid decode (his 19881 / 22735 recipe, our W2 / X400 / X500-A points): TP4, no DP attention, EAGLE
steps 5 / top-k 1 / 6 draft tokens, 1-4 running requests per worker -> 6-24 verify rows per TP rank,
vocab 154,880, sampling defaults from the model's generation_config (temperature 1.0, top_p 0.95).

Besides the kernel alone (rows 6..24, many distributions), the dispatch is exercised as the verify
calls it: target probs from the same softmax op eagle_sample uses, top_ps repeated per draft token,
through sampling_renorm under a TP4 runtime context, flag on vs flag off.

T64_FORCE_COMPILE_TARGET=10.0f (separate process) builds the module for the target a Rubin (SM 10.7)
decode resolves to in the 043cfc4 image (get_compile_arch: 10.7 -> 10.0f when CUTLASS lacks
sm107) and sets SGLANG_AIR_TOP_P_REQUIRED_TARGET=10.0f, so the same build and guard Rubin will use
run here on an sm_100f-family GPU (GB300 = SM 10.3).
"""

import os
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-small")

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

_FORCE = os.environ.get("T64_FORCE_COMPILE_TARGET", "")
if _FORCE:
    # Before anything builds: every reader of the compile target sees the forced one.
    from sglang.kernels.jit.utils import arch as _arch
    from sglang.kernels.jit.utils.compile import cache as _cache
    from sglang.kernels.jit.utils.compile import toolchain as _toolchain

    assert _FORCE == "10.0f", _FORCE
    _forced = _arch.CompileArch(10, 0, "f")
    _arch.get_compile_arch = lambda: _forced
    _cache.get_compile_arch = _arch.get_compile_arch
    _toolchain.get_compile_arch = _arch.get_compile_arch
    os.environ["SGLANG_AIR_TOP_P_REQUIRED_TARGET"] = _FORCE

import flashinfer.sampling  # noqa: E402

from sglang.kernels.ops.sampling import air_top_p_parallel as air  # noqa: E402

if _FORCE:
    air.get_compile_arch = _arch.get_compile_arch

from sglang.kernels.ops.sampling import softmax as sampling_softmax  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers import sampling_renorm  # noqa: E402
from sglang.srt.runtime_context import get_context, get_parallel, reset_context  # noqa: E402
from sglang.srt.speculative import eagle_utils  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_air_top_p_parallel import CASES  # noqa: E402  (his distributions, vocab 154880)

VOCAB = 154880
DRAFT = 6  # speculative-num-draft-tokens 6 (steps 5, top-k 1)
MID_ROWS = [6, 12, 18, 24]  # 1..4 running requests x 6 draft tokens
_TP4 = dict(tp_size=4)  # TP4 decode, no DP attention


def _assert_bitwise(got, want):
    assert got.dtype == want.dtype and got.shape == want.shape
    diff = (got.view(torch.int32) != want.view(torch.int32)).nonzero()
    assert diff.numel() == 0, f"{diff.shape[0]} elements differ, first at {diff[:4].tolist()}"


def _reference(probs, top_p):
    return flashinfer.sampling.top_p_renorm_probs(probs, top_p, is_deterministic=True)


def _verify_probs(bs, gen, scale, temperature=1.0):
    """Target probs as eagle_sample builds them: softmax(logits / T), T per request, 6 rows each."""
    logits = torch.randn(bs * DRAFT, VOCAB, device="cuda", generator=gen) * scale
    # a few strong candidates per row, as a trained LM head gives
    hot = torch.randint(0, VOCAB, (bs * DRAFT, 8), device="cuda", generator=gen)
    logits.scatter_add_(1, hot, torch.rand(bs * DRAFT, 8, device="cuda", generator=gen) * 4 * scale)
    if _FORCE:  # keep other JIT modules out of the forced-target process
        return torch.softmax(logits / temperature, dim=-1)
    temperatures = torch.full((bs, 1), temperature, device="cuda")
    return sampling_softmax(logits, temperatures=temperatures, rows_per_request=DRAFT)


def _verify_top_ps(bs, value=0.95):
    top_ps = torch.full((bs,), value, device="cuda")
    return torch.repeat_interleave(top_ps, DRAFT, dim=0)


@pytest.mark.parametrize("rows", list(range(6, 25)))
@pytest.mark.parametrize("case", ["peaked", "heavy", "ties_dyadic", "exact_boundary", "denormal_tail", "flat", "onehot"])
def test_mid_rows_bitwise(rows, case):
    gen = torch.Generator(device="cuda").manual_seed(1000 + rows * 17 + len(case))
    make, _ = CASES[case]
    probs = make(rows, gen).contiguous()
    top_p = torch.full((rows,), 0.95, device="cuda")
    assert air.can_air_top_p_parallel(probs, top_p)
    _assert_bitwise(air.air_top_p_renorm_parallel(probs, top_p), _reference(probs, top_p))


@pytest.mark.parametrize("bs", [1, 2, 3, 4])
@pytest.mark.parametrize("scale", [1.0, 2.0, 3.0, 5.0, 8.0])
def test_verify_shaped_probs_bitwise(bs, scale):
    gen = torch.Generator(device="cuda").manual_seed(bs * 101 + int(scale * 10))
    for temperature in (1.0, 0.7):
        probs = _verify_probs(bs, gen, scale, temperature)
        top_p = _verify_top_ps(bs)
        fallback = torch.zeros(bs * DRAFT, dtype=torch.uint8, device="cuda")
        got = air.air_top_p_renorm_parallel(probs, top_p, fallback_rows=fallback)
        _assert_bitwise(got, _reference(probs, top_p))


def test_scalar_top_p_mid():
    gen = torch.Generator(device="cuda").manual_seed(5)
    probs = _verify_probs(4, gen, 3.0)
    _assert_bitwise(air.air_top_p_renorm_parallel(probs, 0.95), _reference(probs, 0.95))


@pytest.mark.parametrize("rows", [6, 24])
def test_mid_graph_replay(rows):
    gen = torch.Generator(device="cuda").manual_seed(31 + rows)
    bs = rows // DRAFT
    static_p, static_t = _verify_probs(bs, gen, 3.0), _verify_top_ps(bs)
    air.air_top_p_renorm_parallel(static_p, static_t)  # reserves the workspace before capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = air.air_top_p_renorm_parallel(static_p, static_t)
    for seed in range(16):
        probs = _verify_probs(bs, gen, (1.0, 3.0, 5.0, 8.0)[seed % 4])
        static_p.copy_(probs)
        graph.replay()
        _assert_bitwise(static_out, _reference(probs, static_t))


def test_mid_row_sequence_back_to_back():
    # Verify batches grow and shrink between steps (1..4 requests), unsynchronised.
    gen = torch.Generator(device="cuda").manual_seed(41)
    calls = []
    for bs in (1, 4, 2, 3, 1, 1, 4, 2, 4, 3):
        probs, top_p = _verify_probs(bs, gen, 3.0), _verify_top_ps(bs)
        calls.append((probs, top_p, air.air_top_p_renorm_parallel(probs, top_p)))
    for probs, top_p, got in calls:
        _assert_bitwise(got, _reference(probs, top_p))


class TestTP4Dispatch:
    """The verify's renorm under a TP4 (no DP attention) runtime context, flag off vs on."""

    def setup_method(self):
        reset_context()

    def teardown_method(self):
        reset_context()

    def _joint(self, probs, bs, flag):
        info = SimpleNamespace(
            need_top_p_sampling=True,
            need_top_k_sampling=False,
            top_ps=torch.full((bs,), 0.95, device="cuda"),
        )
        with envs.SGLANG_OPT_AIR_TOP_P_PARALLEL.override(flag):
            with get_context().override_server_args(**_TP4):
                assert get_parallel().tp_size == 4 and sampling_renorm.renorm_deterministic()
                return eagle_utils._apply_joint_top_k_top_p(
                    probs.clone(),
                    info,
                    DRAFT,
                    eagle_utils.top_k_renorm_prob,
                    eagle_utils._verify_top_p_renorm_prob,
                )

    @pytest.mark.parametrize("bs", [1, 2, 3, 4])
    def test_flag_on_equals_flag_off_and_engages(self, bs):
        gen = torch.Generator(device="cuda").manual_seed(77 + bs)
        probs = _verify_probs(bs, gen, 3.0)
        device = torch.cuda.current_device()
        before = air.air_top_p_worker_record(device)
        off = self._joint(probs, bs, False)
        mid = air.air_top_p_worker_record(device)
        on = self._joint(probs, bs, True)
        after = air.air_top_p_worker_record(device)
        _assert_bitwise(on, off)
        _assert_bitwise(off, _reference(probs, _verify_top_ps(bs)))
        # flag off never reaches the parallel kernel; flag on runs it once on bs * 6 rows
        assert (mid["calls"], mid["total_rows"]) == (before["calls"], before["total_rows"])
        assert (after["calls"] - mid["calls"], after["total_rows"] - mid["total_rows"]) == (1, bs * DRAFT)
        assert after["bypass"] == before["bypass"]

    def test_plain_sampler_keeps_flashinfer_with_flag_on(self, monkeypatch):
        gen = torch.Generator(device="cuda").manual_seed(91)
        probs = _verify_probs(2, gen, 3.0)
        calls = []
        monkeypatch.setattr(air, "air_top_p_renorm_parallel", lambda *a, **k: calls.append(1))
        monkeypatch.setattr(sampling_renorm, "air_top_p_renorm_parallel", lambda *a, **k: calls.append(1))
        with envs.SGLANG_OPT_AIR_TOP_P_PARALLEL.override(True):
            with get_context().override_server_args(**_TP4):
                got = sampling_renorm.top_p_renorm_prob(probs, _verify_top_ps(2))
        assert calls == []
        _assert_bitwise(got, _reference(probs, _verify_top_ps(2)))


def test_build_target_record():
    """Which target this process built; with T64_FORCE_COMPILE_TARGET the Rubin one (10.0f)."""
    gen = torch.Generator(device="cuda").manual_seed(3)
    probs = _verify_probs(1, gen, 3.0)
    air.air_top_p_renorm_parallel(probs, _verify_top_ps(1))
    record = air.air_top_p_worker_record(torch.cuda.current_device())
    module = record["module"]
    print("T64_BUILD", {k: module[k] for k in ("compile_target", "compile_cuda_flags", "so_path", "so_sha256")})
    print("T64_SOURCES", {k: record[k] for k in ("py_sha256", "cuh_sha256")})
    assert record["module_status"] == "LOADED"
    if _FORCE:
        assert module["compile_target"] == "10.0f"
        assert "-DSGL_CUDA_ARCH=1000" in module["compile_cuda_flags"]
        assert "/sm100f/" in module["so_path"] or "sm100f" in module["so_path"]
    assert "-use_fast_math" in module["compile_cuda_flags"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
