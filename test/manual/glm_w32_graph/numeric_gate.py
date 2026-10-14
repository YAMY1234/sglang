"""CPU-interpreter / CUDA numerical gate against unmodified service FP64 eigh.

Run CPU: TRITON_INTERPRET=1 python numeric_gate.py --device cpu
Run GPU: python numeric_gate.py --device cuda
No model weights, runtime import, or GPU are needed for the CPU path.
"""

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3] / "python" / "sglang" / "srt"
# Standalone packages avoid SGLang's unrelated GPU-facing top-level API.
for name, path in [
    ("glm_gate_models", ROOT / "models"),
    ("glm_gate_models.glm_duet", ROOT / "models" / "glm_duet"),
]:
    spec = importlib.util.spec_from_file_location(
        name, path / "__init__.py", submodule_search_locations=[str(path)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
reference = importlib.import_module("glm_gate_models.glm_duet._state_factor")
spec = importlib.util.spec_from_file_location(
    "glm_gate_kernel", ROOT / "layers/attention/linear/kernels/glm_w32_project.py"
)
kernel = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = kernel
spec.loader.exec_module(kernel)


def relf(actual, expected):
    return float(
        (actual.double() - expected.double()).norm()
        / expected.double().norm().clamp_min(1e-30)
    )


def run(device):
    torch.manual_seed(205)
    torch.set_num_threads(4)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    for case in ("random", "low_rank", "near_rank_deficient", "zero"):
        state = torch.randn(1, 3, 1, 128, 128, device=device)
        if case != "random":
            state = torch.randn(1, 3, 1, 128, 12, device=device) @ torch.randn(
                1, 3, 1, 12, 128, device=device
            )
            if case == "near_rank_deficient":
                state += 1e-5 * torch.randn_like(state)
            if case == "zero":
                state.zero_()
        directions = torch.randn(1, 1, 128, device=device)
        phase = torch.tensor([0, 13, 31], device=device, dtype=torch.int32)
        pending = torch.tensor([True, False, False], device=device)
        active = torch.ones(3, device=device, dtype=torch.bool)
        indices = torch.tensor([0, 1, -1], device=device, dtype=torch.int32)
        warm = torch.zeros(1, 3, 1, 128, 32, device=device)
        workspace = kernel.ProjectionWorkspace(3, 1, 1, device)
        omega, extra = [
            torch.randn(
                1,
                1,
                128,
                n,
                device=device,
                generator=torch.Generator(device=device).manual_seed(
                    reference.OMEGA_SEED
                ),
            )[0]
            for n in (40, 8)
        ]
        untouched = state[:, 1:].clone()
        previous = None
        for cold in (True, False, False):
            if not cold:
                state[:, 0] += 0.02 * torch.randn_like(state[:, 0])
                phase[0] = 31
                pending[0] = False
            expected, _bases = reference.project_state_batched(
                state[:, 0].transpose(-1, -2),
                directions,
                32,
                [previous],
                omega_heads=(0, 1),
            )
            kernel.project(
                state,
                directions,
                warm,
                omega,
                extra,
                indices,
                phase,
                pending,
                active,
                workspace,
                cold=cold,
            )
            actual = state[:, 0].transpose(-1, -2)
            error = relf(actual, expected)
            assert torch.isfinite(actual).all() and torch.isfinite(warm).all()
            assert torch.equal(state[:, 1:], untouched), "non-due or padding modified"
            row = {"case": case, "cold": cold, "rel_f": error}
            records.append(row)
            print(json.dumps(row), flush=True)
            # Fixed before GPU: 1e-5 single-projection relative Frobenius ceiling.
            assert error <= 1e-5, row
            # Compare next projection starting with the actual previous basis:
            # this isolates one-cut arithmetic, not eigenvector sign conventions.
            previous = warm[:, 0].clone()
        kernel.phase_update(indices, phase, pending, active, 32)
        assert phase.cpu().tolist() == [0, 14, 31]
        assert torch.equal(state[:, 1:], untouched)
    result = {
        "passed": True,
        "device": device,
        "torch": torch.__version__,
        "max_rel_f": max(r["rel_f"] for r in records),
        "records": records,
    }
    print("GLM_W32_NUMERIC " + json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    run(args.device)
