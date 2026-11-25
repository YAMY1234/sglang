"""The parallel AIR top-p renorm must equal flashinfer's deterministic AIR kernel bit for bit."""

import hashlib
import pathlib
import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-small")

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

import flashinfer.sampling  # noqa: E402

from sglang.kernels.ops.sampling.air_top_p_parallel import (  # noqa: E402
    air_top_p_fallback_count,
    air_top_p_renorm_parallel,
    air_top_p_worker_record,
    can_air_top_p_parallel,
)

VOCAB = 154880


def _peaked(rows, gen):
    return torch.softmax(torch.randn(rows, VOCAB, device="cuda", generator=gen) * 5, -1)


def _heavy(rows, gen):
    rank = torch.stack([torch.randperm(VOCAB, device="cuda", generator=gen) for _ in range(rows)]) + 1
    a = 1.25 + 0.1 * torch.rand(rows, 1, device="cuda", generator=gen)
    noise = 0.1 * torch.randn(rows, VOCAB, device="cuda", generator=gen)
    return torch.softmax(-a * rank.float().log() + noise, -1)


def _ties_dyadic(rows, gen):
    layer = torch.full((VOCAB,), 0.0, device="cuda")
    layer[:128], layer[128:8320], layer[8320:139392] = 2.0**-9, 2.0**-14, 2.0**-19
    return torch.stack([layer[torch.randperm(VOCAB, device="cuda", generator=gen)] for _ in range(rows)])


def _exact_boundary(rows, gen):
    # 5 x 3187671 * 2^-24 sums to exactly fp32(0.95); the remaining mass sums to 838861 * 2^-24
    row = torch.zeros(VOCAB, device="cuda", dtype=torch.float64)
    row[:5] = 3187671 * 2.0**-24
    row[5:104862] = 2.0**-21
    row[104862] = 5 * 2.0**-24
    row = row.float()
    return torch.stack([row[torch.randperm(VOCAB, device="cuda", generator=gen)] for _ in range(rows)])


def _denormal_tail(rows, gen):
    p = _peaked(rows, gen)
    bits = torch.randint(1, 1 << 23, p.shape, device="cuda", generator=gen, dtype=torch.int32)
    pick = torch.rand(p.shape, device="cuda", generator=gen)
    p = torch.where(pick < 0.3, bits.view(torch.float32), p)
    return torch.where(pick > 0.95, torch.zeros_like(p), p)


def _flat(rows, gen):
    return torch.full((rows, VOCAB), 1.0 / VOCAB, device="cuda")


def _onehot(rows, gen):
    logits = torch.zeros(rows, VOCAB, device="cuda")
    logits[torch.arange(rows), torch.randint(0, VOCAB, (rows,), device="cuda", generator=gen)] = 1e4
    return torch.softmax(logits, -1)


CASES = {
    "peaked": (_peaked, [0.95]),
    "heavy": (_heavy, [0.95]),
    "ties_dyadic": (_ties_dyadic, [0.95, 0.5, 0.2]),
    "exact_boundary": (_exact_boundary, [0.95]),
    "denormal_tail": (_denormal_tail, [0.95, 1.0, 0.999999]),
    "flat": (_flat, [0.95]),
    "onehot": (_onehot, [0.95]),
    "mixed_top_p": (_heavy, [0.95, 1.0, 0.5, 0.01, 0.999]),
}


def _top_p(rows, values):
    return torch.tensor([values[r % len(values)] for r in range(rows)], device="cuda")


def _assert_bitwise(got, want):
    assert got.dtype == want.dtype and got.shape == want.shape and got.stride() == want.stride()
    diff = (got.view(torch.int32) != want.view(torch.int32)).nonzero()
    assert diff.numel() == 0, f"{diff.shape[0]} elements differ, first at {diff[:4].tolist()}"


def _reference(probs, top_p):
    return flashinfer.sampling.top_p_renorm_probs(probs, top_p, is_deterministic=True)


@pytest.mark.parametrize("rows", [8, 56, 112, 128])
@pytest.mark.parametrize("case", list(CASES))
def test_bitwise_equal_to_flashinfer(rows, case):
    gen = torch.Generator(device="cuda").manual_seed(rows * 31 + len(case))
    make, values = CASES[case]
    probs, top_p = make(rows, gen).contiguous(), _top_p(rows, values)
    assert can_air_top_p_parallel(probs, top_p)
    probs_before = probs.clone()
    fallback = torch.empty(rows, dtype=torch.uint8, device="cuda")
    got = air_top_p_renorm_parallel(probs, top_p, fallback_rows=fallback)
    _assert_bitwise(got, _reference(probs, top_p))
    assert torch.equal(probs, probs_before)
    if case == "flat":  # K = vocab exceeds the kept-list cap: dense per-row fallback
        assert bool(fallback.bool().all())


def test_scalar_top_p():
    gen = torch.Generator(device="cuda").manual_seed(7)
    probs = _peaked(112, gen)
    _assert_bitwise(air_top_p_renorm_parallel(probs, 0.95), _reference(probs, 0.95))


def test_graph_replay_with_fresh_inputs():
    gen = torch.Generator(device="cuda").manual_seed(11)
    a, b = _peaked(112, gen), _heavy(112, gen)
    top_a, top_b = _top_p(112, [0.95]), _top_p(112, [0.5, 0.95, 0.999])
    static_p, static_t = a.clone(), top_a.clone()
    air_top_p_renorm_parallel(static_p, static_t)  # reserves the workspace before capture
    rng_before = torch.cuda.get_rng_state()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = air_top_p_renorm_parallel(static_p, static_t)
    for probs, top_p in ((b, top_b), (a, top_a), (b, top_b)):
        static_p.copy_(probs)
        static_t.copy_(top_p)
        graph.replay()
        _assert_bitwise(static_out, _reference(probs, top_p))
    assert torch.equal(torch.cuda.get_rng_state(), rng_before)


def test_row_count_sequences():
    # Shrinking and growing row counts on one workspace (and past its capacity) must not see
    # state a previous call left in other rows.
    gen = torch.Generator(device="cuda").manual_seed(13)
    for rows in (112, 8, 128, 56, 112, 8, 300, 8, 112, 300):
        for make, values in ((_peaked, [0.95]), (_heavy, [0.95, 0.5]), (_flat, [0.95])):
            probs, top_p = make(rows, gen), _top_p(rows, values)
            _assert_bitwise(air_top_p_renorm_parallel(probs, top_p), _reference(probs, top_p))


def _capture(rows, gen):
    static_p, static_t = _heavy(rows, gen), _top_p(rows, [0.95])
    static_f = torch.zeros(rows, dtype=torch.uint8, device="cuda")
    air_top_p_renorm_parallel(static_p, static_t)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = air_top_p_renorm_parallel(static_p, static_t, fallback_rows=static_f)
    return graph, static_p, static_t, static_f, static_out


def test_old_graph_after_growth_and_fallback_count():
    gen = torch.Generator(device="cuda").manual_seed(17)
    count0 = air_top_p_fallback_count()
    masked = 0
    graph, static_p, static_t, static_f, static_out = _capture(112, gen)
    for rows in (8, 2048, 56):  # 2048 grows the workspace past the graph's
        probs, top_p = _flat(rows, gen), _top_p(rows, [0.95])
        fallback = torch.zeros(rows, dtype=torch.uint8, device="cuda")
        _assert_bitwise(
            air_top_p_renorm_parallel(probs, top_p, fallback_rows=fallback), _reference(probs, top_p)
        )
        masked += int(fallback.sum())
        for make in (_peaked, _flat):  # the old graph, on its old workspace, after each shape
            probs, top_p = make(112, gen), _top_p(112, [0.95, 0.5])
            static_p.copy_(probs)
            static_t.copy_(top_p)
            graph.replay()
            _assert_bitwise(static_out, _reference(probs, top_p))
            masked += int(static_f.sum())
    assert masked > 0
    assert air_top_p_fallback_count() - count0 == masked


def _make(kind, rows, vocab, gen):
    if kind == "flat":  # one pass-0 bucket holds the whole row: the widest 32-bit field values
        return torch.full((rows, vocab), 1.0 / vocab, device="cuda")
    if kind == "heavy":
        rank = torch.stack([torch.randperm(vocab, device="cuda", generator=gen) for _ in range(rows)]) + 1
        return torch.softmax(-1.3 * rank.float().log() + 0.1 * torch.randn(rows, vocab, device="cuda", generator=gen), -1)
    return torch.softmax(torch.randn(rows, vocab, device="cuda", generator=gen) * 5, -1)


@pytest.mark.parametrize("grid_limit", [1, 3, 7, 0, 1 << 20])
@pytest.mark.parametrize("vocab", [2048, 6148, 12288, VOCAB])  # 6148: a 4-element last segment
@pytest.mark.parametrize("rows", [1, 5, 113])
def test_persistent_loop_edges(grid_limit, vocab, rows):
    # Small grids make every CTA walk many units across row boundaries (flush at a row change, after
    # two units and at the CTA end); a huge limit is capped at one unit per CTA.
    gen = torch.Generator(device="cuda").manual_seed(grid_limit * 7 + vocab + rows)
    for kind in ("peaked", "heavy", "flat"):
        probs, top_p = _make(kind, rows, vocab, gen), _top_p(rows, [0.95, 0.5, 0.999])
        got = air_top_p_renorm_parallel(probs, top_p, grid_limit=grid_limit)
        _assert_bitwise(got, _reference(probs, top_p))


def test_back_to_back_calls_without_sync():
    # Every call clears what it merged; a missing clear or a read before the merges land
    # shows up in a later call of this unsynchronised sequence.
    gen = torch.Generator(device="cuda").manual_seed(23)
    calls = []
    for i in range(12):
        rows = (112, 37, 1, 112)[i % 4]
        kind = ("peaked", "heavy", "flat")[i % 3]
        probs, top_p = _make(kind, rows, VOCAB, gen), _top_p(rows, [0.95, 0.5, 0.2, 1.0][i % 4 :])
        calls.append((probs, top_p, air_top_p_renorm_parallel(probs, top_p, grid_limit=(0, 5)[i % 2])))
    for probs, top_p, got in calls:
        _assert_bitwise(got, _reference(probs, top_p))


@pytest.mark.parametrize("rows,grid_limit", [(112, 0), (37, 0), (112, 5)])
def test_graph_replay_many_seeds_visibility(rows, grid_limit):
    # Select0 and Norm read histograms other CTAs merged in the previous kernel; only the kernel
    # boundary orders them (no PDL). Many replays with fresh inputs catch a stale or partial read.
    gen = torch.Generator(device="cuda").manual_seed(29 + rows + grid_limit)
    static_p, static_t = _make("heavy", rows, VOCAB, gen), _top_p(rows, [0.95])
    air_top_p_renorm_parallel(static_p, static_t, grid_limit=grid_limit)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = air_top_p_renorm_parallel(static_p, static_t, grid_limit=grid_limit)
    for seed in range(24):
        kind = ("peaked", "heavy", "flat", "peaked")[seed % 4]
        probs, top_p = _make(kind, rows, VOCAB, gen), _top_p(rows, [(0.95, 0.5, 0.999, 0.2)[seed % 4], 0.95])
        static_p.copy_(probs)
        static_t.copy_(top_p)
        graph.replay()
        _assert_bitwise(static_out, _reference(probs, top_p))


def test_worker_record_binds_the_loaded_library():
    # Host counters count Python-side calls (not replays); the fallback count is the device's.
    gen = torch.Generator(device="cuda").manual_seed(19)
    device = torch.cuda.current_device()
    air_top_p_renorm_parallel(_peaked(8, gen), 0.95)
    before = air_top_p_worker_record(device)
    probs = _flat(56, gen)
    fallback = torch.zeros(56, dtype=torch.uint8, device="cuda")
    air_top_p_renorm_parallel(probs, _top_p(56, [0.95]), fallback_rows=fallback)
    after = air_top_p_worker_record(device)
    assert before["module_status"] == after["module_status"] == "LOADED"
    assert before["counter_generation"] == after["counter_generation"] is not None
    assert (after["calls"] - before["calls"], after["total_rows"] - before["total_rows"]) == (1, 56)
    assert after["kernel_fallback_rows"] - before["kernel_fallback_rows"] == int(fallback.sum())
    assert after["gpu_uuid"] == str(torch.cuda.get_device_properties(device).uuid)
    module = after["module"]
    assert module["so_path"].endswith("/sgl_kernel_jit_sampling_air_top_p_parallel.so")
    assert module["so_sha256"] == hashlib.sha256(pathlib.Path(module["so_path"]).read_bytes()).hexdigest()
    assert module["cuh_sha256_at_load"] == after["cuh_sha256"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
