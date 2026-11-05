"""CPU dispatch controls; CUDA compilation and arithmetic are tested separately."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/kernels/ops/attention/fast_topk.py"
)


@pytest.mark.parametrize("master", [False, True])
@pytest.mark.parametrize("explicit_starts", [False, True])
@pytest.mark.parametrize("zero_start", [False, True])
def test_master_selects_main_or_candidate_dispatch(master, explicit_starts, zero_start):
    with patch.dict(
        "os.environ", {"SGLANG_Q38_TOPK_ZERO_START": str(int(zero_start))}
    ):
        spec = importlib.util.spec_from_file_location("topk_master_case", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    scores = torch.empty((2, 520), dtype=torch.float32)
    lengths = torch.tensor([519, 520], dtype=torch.int32)
    starts = torch.tensor([0, 0], dtype=torch.int32) if explicit_starts else None
    compiled, launched = [], []

    def launch(actual_scores, actual_starts, output, actual_lengths):
        launched.append((actual_scores, actual_starts, output, actual_lengths))
        output.fill_(7)

    def load(name, *args, **kwargs):
        compiled.append((name, args, kwargs))
        return SimpleNamespace(fast_topk=launch)

    with (
        envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(master),
        patch.object(module, "is_arch_support_pdl", return_value=True),
        patch.object(module, "make_cpp_args", side_effect=lambda *args: args),
        patch.object(module, "load_jit", side_effect=load),
    ):
        output = module.fast_topk(scores, lengths, 512, starts)

    assert len(compiled) == len(launched) == 1
    name, args, kwargs = compiled[0]
    assert name == "fast_topk"
    use_zero_start = master and zero_start and not explicit_starts
    assert args == ((512, True, use_zero_start) if master else (512, True))
    assert kwargs["cuda_files"] == [
        "elementwise/deterministic_fast_topk.cuh"
        if master
        else "elementwise/fast_topk.cuh"
    ]
    assert kwargs["cuda_wrappers"] == [("fast_topk", f"FastTopKKernel<{args}>::run")]
    actual_scores, actual_starts, actual_output, actual_lengths = launched[0]
    assert (
        actual_scores is scores
        and actual_lengths is lengths
        and actual_output is output
    )
    if explicit_starts:
        assert actual_starts is starts
    elif use_zero_start:
        assert actual_starts is None
    else:
        assert actual_starts.dtype == torch.int32
        assert torch.equal(actual_starts, torch.zeros(2, dtype=torch.int32))
    assert output.dtype == torch.int32 and output.shape == (2, 512)
    assert torch.equal(output, torch.full((2, 512), 7, dtype=torch.int32))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
