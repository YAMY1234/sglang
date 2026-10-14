"""No-weight qualification of device IF + unmodified Torch eigensolver capture.

Run dry-run in the same CUDA image on CPU first. Each risky capture runs in its
own process so an invalidated capture cannot contaminate another measurement.
This is an experimental mechanism test; it is not wired into serving.
"""

import argparse
import contextlib
import ctypes
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path

import torch


def build(out):
    source = Path(__file__).with_suffix(".cu")
    # Keep CUDA bridge next to this script under its descriptive source name.
    source = source.with_name("conditional.cu")
    library = out / "conditional.so"
    command = [
        "nvcc",
        "-std=c++17",
        "-O2",
        "-arch=sm_103",
        "-shared",
        "-Xcompiler=-fPIC",
        str(source),
        "-o",
        str(library),
    ]
    subprocess.run(command, check=True)
    return library


def lib(out):
    bridge = ctypes.CDLL(str(out / "conditional.so"))
    bridge.attach_if.argtypes = [ctypes.c_size_t] * 3
    bridge.attach_if.restype = ctypes.c_int
    return bridge


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        fn()
    return graph


def cpu_check_dispatch():
    from numeric_gate import reference

    assert callable(reference._project_shared_omega)
    x = torch.arange(128, dtype=torch.float64).reshape(2, 8, 8)
    x = x @ x.mT + torch.eye(8)
    expected = torch.linalg.eigh(x)
    calls = []

    def check(info, api_name, *, is_matrix):
        calls.append(api_name)
        torch._assert_async((info == 0).all(), "solver info != 0")

    with torch.library._scoped_library("aten", "IMPL", "CPU") as scope:
        scope.impl("_linalg_check_errors", check)
        actual = torch.linalg.eigh(x)
        failed = False
        try:
            torch.ops.aten._linalg_check_errors(
                torch.ones(1, dtype=torch.int32), "qualification", is_matrix=False
            )
        except RuntimeError:
            failed = True
        assert failed
    assert "linalg.eigh" in calls and all(
        torch.equal(a, b) for a, b in zip(actual, expected)
    )
    return {"eigh_bitwise_equal": True, "nonzero_info_rejected": True}


def eig_capture(out, *, async_check=False):
    from numeric_gate import reference

    torch.manual_seed(206)
    state = torch.randn(32, 16, 128, 128, device="cuda")
    direction = torch.randn(32, 16, 128, device="cuda")
    # Generate once outside capture; retain identical shape/seed/FP32 boundaries.
    omega = torch.randn(
        1,
        64,
        128,
        40,
        generator=torch.Generator(device="cuda").manual_seed(reference.OMEGA_SEED),
        device="cuda",
    )[:, :16].expand(32, -1, -1, -1)
    dest = torch.empty_like(state)

    def fixed_math():
        n2 = (direction * direction).sum(-1).clamp_min(1e-12)
        a = torch.einsum("bhkv,bhv->bhk", state, direction) / n2[:, :, None]
        sink = a[..., :, None] * direction[:, :, None, :]
        c = state - sink
        orth = reference._batched_orthonormalize
        y = c @ omega
        y = c @ orth(c.transpose(-1, -2) @ orth(y))
        q = orth(y)
        bm = q.transpose(-1, -2) @ c
        gram = (bm @ bm.transpose(-1, -2)).double()
        jitter = (
            1e-7 * gram.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] + 1e-30
        )
        gram = gram + jitter * torch.eye(40, device=state.device, dtype=gram.dtype)
        z = torch.linalg.eigh(gram)[1].float()
        u = q @ z[..., -32:]
        uts = u.transpose(-1, -2) @ c
        dest.copy_(sink + u @ uts)

    receipt = {
        "kind": "reference_arithmetic_capture",
        "rows": 32,
        "heads": 16,
        "device_error_check": async_check,
    }
    checks = []

    def check(info, api_name, *, is_matrix):
        checks.append(api_name)
        torch._assert_async((info == 0).all(), "captured linalg solver info != 0")

    try:
        expected, _ = reference._project_shared_omega(
            state, direction, 32, None, True, "right", omega_heads=(0, 64)
        )
        context = (
            torch.library._scoped_library("aten", "IMPL", "CUDA")
            if async_check
            else contextlib.nullcontext()
        )
        with context as scope:
            if async_check:
                scope.impl("_linalg_check_errors", check)
            fixed_math()
            receipt["eager_bitwise_equal"] = torch.equal(dest, expected)
            graph = capture(fixed_math)
        if async_check:
            assert "linalg.eigh" in checks, checks
        graph.replay()
        torch.cuda.synchronize()
        receipt.update(
            capture_supported=True,
            replay_bitwise_equal=torch.equal(dest, expected),
            rel_f=float(
                (dest.double() - expected.double()).norm() / expected.double().norm()
            ),
        )
        bridge = lib(out)
        phase = torch.zeros(1, dtype=torch.int32, device="cuda")
        flag = torch.zeros_like(phase)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        parent = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(parent, stream=stream):
            flag.copy_((phase.remainder(32) == 31).int())
            code = bridge.attach_if(
                torch.cuda.current_stream().cuda_stream,
                graph.raw_cuda_graph(),
                flag.data_ptr(),
            )
            if code:
                raise RuntimeError(f"attach reference child CUDA error {code}")
            phase.add_(1)
        exact = []
        for step in range(1, 65):
            state.mul_(0.999)
            if step % 32 == 0:
                expected, _ = reference._project_shared_omega(
                    state, direction, 32, None, True, "right", omega_heads=(0, 64)
                )
            dest.fill_(-777)
            parent.replay()
            torch.cuda.synchronize()
            exact.append(
                torch.equal(dest, expected)
                if step % 32 == 0
                else bool((dest == -777).all())
            )
        receipt.update(
            conditional_replay_steps=64,
            conditional_replay_bitwise_equal=all(exact),
            conditional_steps_equal=exact,
        )
    except RuntimeError as e:
        receipt.setdefault("capture_supported", False)
        receipt.update(error=f"{type(e).__name__}: {e}")
    receipt["captured_check_calls"] = checks
    filename = (
        "arithmetic-capture-async.json" if async_check else "arithmetic-capture.json"
    )
    (out / filename).write_text(json.dumps(receipt, indent=2))
    print("GLMW32_REFERENCE_CAPTURE " + json.dumps(receipt), flush=True)


def conditional(out):
    bridge = lib(out)
    value = torch.zeros(1, device="cuda")
    phase = torch.zeros(1, dtype=torch.int32, device="cuda")
    flag = torch.zeros_like(phase)
    child = capture(lambda: value.add_(1))
    value.zero_()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    parent = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(parent, stream=stream):
        flag.copy_((phase.remainder(32) == 31).int())
        code = bridge.attach_if(
            torch.cuda.current_stream().cuda_stream,
            child.raw_cuda_graph(),
            flag.data_ptr(),
        )
        if code:
            raise RuntimeError(f"attach_if CUDA error {code}")
        phase.add_(1)
    observed = []
    for step in range(1, 65):
        parent.replay()
        observed.append(int(value.item()))
        assert observed[-1] == step // 32, (step, observed[-1])
    receipt = {
        "kind": "device_conditional",
        "replay_steps": 64,
        "observed": observed,
        "passed": True,
    }
    (out / "conditional-complete.json").write_text(json.dumps(receipt, indent=2))
    print("GLMW32_CONDITIONAL " + json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["dry-run", "eigh", "eigh-async", "conditional", "report"],
        required=True,
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "dry-run":
        library = build(args.out)
        lib(args.out)
        assert "keep_graph" in inspect.signature(torch.cuda.CUDAGraph).parameters
        assert hasattr(torch.cuda.CUDAGraph, "raw_cuda_graph")
        dispatch = cpu_check_dispatch()
        assert not torch.cuda.is_initialized()
        receipt = {
            "cpu_dispatch": dispatch,
            "passed": True,
            "library": str(library),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_initialized": False,
            "source": os.environ["GLMW32_SOURCE_COMMIT"],
            "protocol": os.environ["GLMW32_PROTOCOL_COMMIT"],
            "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        }
        (args.out / "conditional-dry-run.json").write_text(
            json.dumps(receipt, indent=2)
        )
        print("GLMW32_CONDITIONAL_DRY_RUN " + json.dumps(receipt), flush=True)
    elif args.mode == "report":
        native = json.loads((args.out / "arithmetic-capture.json").read_text())
        arithmetic = json.loads(
            (args.out / "arithmetic-capture-async.json").read_text()
        )
        condition = json.loads((args.out / "conditional-complete.json").read_text())
        receipt = {
            "source": os.environ["GLMW32_SOURCE_COMMIT"],
            "protocol": os.environ["GLMW32_PROTOCOL_COMMIT"],
            "native_arithmetic": native,
            "arithmetic": arithmetic,
            "condition": condition,
            "passed": bool(
                arithmetic.get("eager_bitwise_equal")
                and arithmetic.get("replay_bitwise_equal")
                and arithmetic.get("conditional_replay_bitwise_equal")
                and condition.get("passed")
            ),
        }
        (args.out / "mechanism-complete.json").write_text(json.dumps(receipt, indent=2))
        print("GLMW32_MECHANISM_RESULT " + json.dumps(receipt), flush=True)
        assert receipt["passed"], receipt
    elif args.mode in ("eigh", "eigh-async"):
        eig_capture(args.out, async_check=args.mode == "eigh-async")
    else:
        conditional(args.out)


if __name__ == "__main__":
    main()
