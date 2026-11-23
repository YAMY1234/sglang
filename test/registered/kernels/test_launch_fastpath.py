"""SGLANG_LAUNCH_FASTPATH must launch exactly what the plain Triton / CuTe DSL call launches,
including a separate kernel for every argument specialization."""

import pytest
import torch
import triton
import triton.language as tl

from sglang.kernels import launch_fastpath
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@triton.jit
def _scale_add_kernel(x_ptr, y_ptr, out_ptr, n, alpha, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * alpha + y, mask=mask)


@pytest.fixture
def fastpath_switch():
    before = launch_fastpath.enabled
    yield launch_fastpath.set_enabled
    launch_fastpath.set_enabled(before)


def _scale_add(n, offset, callable_grid):
    x = torch.randn(n + offset, device="cuda")[offset:]
    y = torch.randn(n + offset, device="cuda")[offset:]
    outs = []
    for on in (False, True, True):  # plain, fast-path fill, fast-path cached launch
        launch_fastpath.set_enabled(on)
        out = torch.full_like(x, float("nan"))
        grid = (
            (lambda meta: (triton.cdiv(n, meta["BLOCK"]),))
            if callable_grid
            else (triton.cdiv(n, 128),)
        )
        launch_fastpath.triton_launch(
            _scale_add_kernel, grid, x, y, out, n, 0.5, BLOCK=128, num_warps=4
        )
        outs.append(out)
    return outs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("callable_grid", [False, True])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("n", [1, 15, 16, 17, 1000, 4096])
def test_triton_launch_matches_plain_launch(fastpath_switch, n, offset, callable_grid):
    plain, filled, cached = _scale_add(n, offset, callable_grid)
    torch.testing.assert_close(filled, plain, rtol=0, atol=0)
    torch.testing.assert_close(cached, plain, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_triton_launch_caches_per_specialization(fastpath_switch):
    for n, offset in ((16, 0), (17, 0), (16, 1)):
        _scale_add(n, offset, callable_grid=False)
    keys = {
        key
        for key in launch_fastpath._triton_kernels
        if key[0] == id(_scale_add_kernel)
    }
    # n % 16 and the pointers' 16-byte alignment are both specialized on.
    assert len(keys) >= 3


class _FakeCompiled:
    """A CuTe DSL compiled function's call surface: marshal args, then run the program."""

    def __init__(self):
        self.execution_args = self
        self._default_executor = None
        self.marshalled = 0
        self.runs = []

    def generate_execution_args(self, args, kwargs):
        self.marshalled += 1
        return [id(a) for a in args], []

    def run_compiled_program(self, exe_args):
        self.runs.append(("fast", tuple(exe_args)))

    def __call__(self, *args):
        self.runs.append(("plain", args))


def test_cute_static_call_marshals_once(fastpath_switch):
    compiled, made = _FakeCompiled(), []

    def make_args():
        made.append(1)
        return ("a", "b")

    fastpath_switch(True)
    for _ in range(3):
        launch_fastpath.cute_static_call(compiled, make_args, cache_key=("test", 0))
    assert compiled.marshalled == 1 and len(made) == 1
    assert [kind for kind, _ in compiled.runs] == ["fast"] * 3
    assert len({exe for _, exe in compiled.runs}) == 1

    fastpath_switch(False)
    launch_fastpath.cute_static_call(compiled, make_args, cache_key=("test", 0))
    assert compiled.runs[-1] == ("plain", ("a", "b"))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
