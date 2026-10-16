"""CPU qualification of docs/207's BF16 deferred-decay representation.

Run with TRITON_INTERPRET=1. This loads the unchanged production kernels in
an isolated package, without importing the serving stack or touching CUDA.
NumPy's FP64 delta rule is the independent oracle. A FAIL is a local
representation failure, never a model KL/GSM measurement.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import types
from pathlib import Path

os.environ["TRITON_INTERPRET"] = "1"

import numpy as np
import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[3]
KERNELS = ROOT / "python/sglang/srt/layers/attention/linear/kernels"
POOL = ROOT / "python/sglang/srt/mem_cache/kda_factored_pool.py"


@triton.jit
def _round_probe(src, dst):
    i = tl.arange(0, 2)
    value = tl.load(src + i).to(tl.bfloat16)
    tl.store(dst + i, value)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def modules():
    package = types.ModuleType("kda207_cpu_kernels")
    package.__path__ = [str(KERNELS)]
    sys.modules[package.__name__] = package
    kernel = load(package.__name__ + ".kda_factored", KERNELS / "kda_factored.py")
    pool = load("kda207_cpu_pool", POOL)
    return kernel, pool


def relative(actual, expected):
    return float(
        np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30)
    )


def numpy_rne_deferred(family, regime):
    """Independent FP32 deferred algebra with torch's exact BF16 RNE stores.

    The installed Triton interpreter truncates BF16 casts. Keep that evidence,
    but decide the representation precondition using correctly rounded stores.
    """
    generator = torch.Generator().manual_seed(617)
    inputs = (torch.randn(31, 2, 384, generator=generator) * 0.1).bfloat16()
    z = np.full(128, -6, dtype=np.float32)
    if regime != "uniform":
        z[::2] = (1 if regime == "moderate" else 36) if family == "kimi" else 6
    ell = (
        -np.exp(np.float32(1)) * np.where(z <= 20, np.logaddexp(0, z), z)
        if family == "kimi"
        else -5 / (1 + np.exp(-np.exp(np.float32(1)) * z))
    ).astype(np.float32)
    ell64 = (
        -math.e * np.logaddexp(0, z.astype(np.float64))
        if family == "kimi"
        else -5 / (1 + np.exp(-math.e * z.astype(np.float64)))
    )
    raw, values = np.zeros((64, 128), np.float32), np.zeros((64, 128), np.float32)
    exponents, logd = np.zeros(64, np.float32), np.zeros(128, np.float32)
    ref = np.zeros((128, 128), np.float64)
    rows = []
    for step, mixed in enumerate(inputs, 1):
        q0, k0, v = mixed[0].float().numpy().reshape(3, 128)
        k = k0 / np.sqrt(np.sum(k0 * k0) + np.float32(1e-6))
        q = q0 / np.sqrt(np.sum(q0 * q0) + np.float32(1e-6)) / np.float32(np.sqrt(128))
        logd += ell
        with np.errstate(divide="ignore", under="ignore"):
            physical = np.sign(raw) * np.exp(
                np.log(np.abs(raw)) + logd + exponents[:, None]
            )
        residual = v - np.sum(values * np.sum(physical * k, axis=1)[:, None], axis=0)
        largest, smallest = np.max(-logd), np.min(-logd)
        exponent = largest - np.minimum(
            np.float32(0.5) * (largest - smallest), np.float32(80)
        )
        with np.errstate(under="ignore"):
            before_store = np.float32(0.5) * k * np.exp(-logd - exponent)
        index = 31 + step
        raw[index] = torch.from_numpy(before_store.copy()).bfloat16().float().numpy()
        values[index] = torch.from_numpy(residual.copy()).bfloat16().float().numpy()
        exponents[index] = exponent
        with np.errstate(divide="ignore", under="ignore"):
            physical = np.sign(raw) * np.exp(
                np.log(np.abs(raw)) + logd + exponents[:, None]
            )
        k64 = k0.astype(np.float64)
        q64 = q0.astype(np.float64)
        k64 /= np.sqrt(k64 @ k64 + 1e-6)
        q64 /= np.sqrt(q64 @ q64 + 1e-6) * np.sqrt(128)
        ref *= np.exp(ell64)[:, None]
        ref += np.outer(0.5 * k64, v.astype(np.float64) - k64 @ ref)
        state = (physical.T @ values).astype(np.float64)
        output = np.sum(values * np.sum(physical * q, axis=1)[:, None], axis=0)
        rows.append(
            {
                "step": step,
                "new_nonzero_components_encoded_as_zero": int(
                    np.count_nonzero((raw[index] == 0) & (k != 0))
                ),
                "state_relative_error": relative(state, ref),
                "output_relative_error_before_api_cast": relative(output, q64 @ ref),
            }
        )
    return {
        "family": family,
        "regime": regime,
        "rounding": "torch BF16 RN ties-to-even, independent NumPy FP32 algebra",
        "first_new_column_loss_step": next(
            (x["step"] for x in rows if x["new_nonzero_components_encoded_as_zero"]),
            None,
        ),
        "max_state_relative_error": max(x["state_relative_error"] for x in rows),
        "max_output_relative_error_before_api_cast": max(
            x["output_relative_error_before_api_cast"] for x in rows
        ),
        "by_step": rows,
    }


def one_case(kernel, pool_module, family, regime, mode):
    cfg = pool_module.FactoredKDAConfig(
        32,
        32,
        "bf16" if mode == "bf16_deferred" else "fp32",
        deferred=mode != "fp32_immediate",
    )
    cache = types.SimpleNamespace(shape=types.SimpleNamespace(temporal=(1, 128, 128)))
    pool = pool_module.FactoredKDAPool(2, cache, [0], "cpu", cfg)
    pool.bind_directions({0: torch.zeros(1, 128)})
    slots = torch.tensor([0, -1], dtype=torch.int64)
    gate = torch.full((2, 128), -6.0, dtype=torch.bfloat16)
    if regime != "uniform":
        gate[:, ::2] = (
            (1.0 if regime == "moderate" else 36.0) if family == "kimi" else 6.0
        )
    beta_raw = torch.zeros(2, 1, dtype=torch.bfloat16)
    alog, bias = torch.ones(1), torch.zeros(128)
    lower = None if family == "kimi" else -5.0
    z = gate[0].float().numpy().astype(np.float64)
    ell = (
        -np.exp(1.0) * np.logaddexp(0, z)
        if lower is None
        else -5 / (1 + np.exp(-np.exp(1.0) * z))
    )
    ref = np.zeros((128, 128), np.float64)  # K,V; independent of pool.dense
    generator = torch.Generator().manual_seed(617)
    inputs = (torch.randn(31, 2, 384, generator=generator) * 0.1).bfloat16()
    rows = []
    immutable = True
    padding = True
    for step, mixed in enumerate(inputs, 1):
        before = [x.clone() for x in pool.tensors()]
        out = kernel.packed_decode(
            mixed,
            gate,
            beta_raw,
            alog,
            bias,
            pool.vbar[0],
            pool.a[0],
            pool.basis[0],
            pool.coeff[0],
            pool.count[0],
            slots,
            32,
            32,
            lower_bound=lower,
            immediate=cfg.use_immediate,
            log_decay=pool.log_decay[0] if pool.log_decay is not None else None,
            column_exponent=pool.column_exponent[0]
            if pool.column_exponent is not None
            else None,
        )
        if mode != "fp32_immediate":
            immutable &= torch.equal(
                pool.coeff[0, 0, :, : 31 + step], before[2][0, 0, :, : 31 + step]
            )
        immutable &= torch.equal(
            pool.basis[0, 0, :, : 31 + step], before[1][0, 0, :, : 31 + step]
        )
        padding &= bool(out[0, 1].eq(0).all())
        padding &= all(
            torch.equal(x[:, 2], old[:, 2]) for x, old in zip(pool.tensors(), before)
        )
        q, k, v = mixed[0].float().numpy().astype(np.float64).reshape(3, 128)
        q = q / np.sqrt(q @ q + 1e-6) / np.sqrt(128)
        k = k / np.sqrt(k @ k + 1e-6)
        ref *= np.exp(ell)[:, None]
        residual = v - k @ ref
        ref += np.outer(0.5 * k, residual)
        expected_out = q @ ref
        # No model/API output rounding in the state oracle. Report API rounding separately.
        actual_state = (
            pool.dense(0, slots[:1])[0, 0].float().numpy().T.astype(np.float64)
        )
        raw_new = pool.coeff[0, 0, 0, 31 + step].float().numpy()
        lost = int(np.count_nonzero((raw_new == 0) & (0.5 * k != 0)))
        finite = all(bool(torch.isfinite(x).all()) for x in pool.tensors()) and bool(
            torch.isfinite(out).all()
        )
        rows.append(
            {
                "step": step,
                "log_decay_span": float(np.ptp(step * ell)),
                "new_nonzero_components_encoded_as_zero": lost,
                "state_relative_error": relative(actual_state, ref),
                "output_relative_error_including_bf16_api": relative(
                    out[0, 0, 0].float().numpy(), expected_out
                ),
                "finite": finite,
            }
        )
    saved = pool.get_cpu_slots(slots[:1])
    pool.copy_slots(slots[:1], torch.tensor([1]))
    copied = all(torch.equal(x[:, 0], x[:, 1]) for x in pool.tensors())
    pool.reset_slots(slots[:1])
    reset = all(bool(x[:, 0].eq(0).all()) for x in pool.tensors()[:-1]) and bool(
        pool.count[:, 0].eq(32).all()
    )
    pool.load_cpu_slots(saved, slots[:1])
    restored = all(torch.equal(x[:, 0], x[:, 1]) for x in pool.tensors())
    actual_bytes = sum(x[0, 0, 0].numel() * x.element_size() for x in pool.tensors())
    shape_bytes = cfg.state_bytes_per_layer(cache.shape)
    expected_bytes = (
        34052
        if mode == "bf16_deferred"
        else 66820
        if mode == "fp32_deferred"
        else 66052
    )
    assert actual_bytes == shape_bytes == expected_bytes, (
        mode,
        actual_bytes,
        shape_bytes,
        expected_bytes,
    )
    lifecycle = copied and reset and restored and immutable and padding
    assert lifecycle
    return {
        "family": family,
        "regime": regime,
        "mode": mode,
        "steps": 31,
        "expiry_executed": False,
        "actual_bytes_per_head": actual_bytes,
        "lifecycle_and_old_columns": lifecycle,
        "all_finite": all(row["finite"] for row in rows),
        "first_new_column_loss_step": next(
            (r["step"] for r in rows if r["new_nonzero_components_encoded_as_zero"]),
            None,
        ),
        "max_state_relative_error": max(r["state_relative_error"] for r in rows),
        "max_output_relative_error_including_bf16_api": max(
            r["output_relative_error_including_bf16_api"] for r in rows
        ),
        "by_step": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    # Explicit ties-to-even check at ordinary BF16 magnitudes.
    tie_input = torch.tensor([1 + 1 / 256, 1 + 3 / 256])
    tie = tie_input.bfloat16().float()
    assert torch.equal(tie, torch.tensor([1.0, 1 + 1 / 64]))
    interpreted = torch.empty(2, dtype=torch.bfloat16)
    _round_probe[(1,)](tie_input, interpreted)
    rounding = {
        "input": tie_input.tolist(),
        "expected_rne": tie.tolist(),
        "interpreter": interpreted.float().tolist(),
        "interpreter_matches_rne": torch.equal(tie, interpreted.float()),
        "scope": "interpreter BF16 errors are diagnostic; independent RNE mirror decides representation failure",
    }
    kernel, pool_module = modules()
    results, rne_results = [], []
    for family, regimes in (
        ("kimi", ("uniform", "moderate", "extreme")),
        ("glm", ("uniform", "extreme")),
    ):
        for regime in regimes:
            rne_results.append(numpy_rne_deferred(family, regime))
            for mode in ("fp32_immediate", "fp32_deferred", "bf16_deferred"):
                row = one_case(kernel, pool_module, family, regime, mode)
                results.append(row)
                print(
                    json.dumps({k: v for k, v in row.items() if k != "by_step"}),
                    flush=True,
                )
    controls = [x for x in results if x["mode"] == "fp32_immediate"]
    assert all(
        x["max_state_relative_error"] < 2e-5 and x["first_new_column_loss_step"] is None
        for x in controls
    )
    failures = [x for x in rne_results if x["first_new_column_loss_step"] is not None]
    assert all(
        x["first_new_column_loss_step"] is None
        for x in rne_results
        if x["family"] == "glm" or x["regime"] != "extreme"
    )
    max_bf16 = (2 - 2**-7) * 2**127
    limits = {"bf16_with_subnormals_log_span": math.log(max_bf16) - math.log(2**-133)}
    receipt = {
        "scope": "CPU production Triton interpreter vs independent NumPy FP64 recurrence, synthetic inputs; not model KL/GSM",
        "decision": "STOP_CPU_REPRESENTATION_FAIL"
        if failures
        else "CPU_REPRESENTATION_PASS",
        "cuda_initialized": torch.cuda.is_initialized(),
        "runtime": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "triton": triton.__version__,
            "python": sys.version,
        },
        "rounding_probe": rounding,
        "independent_rne_results": rne_results,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                POOL,
                KERNELS / "kda_factored.py",
                KERNELS / "gdn_factored.py",
                KERNELS / "gdn_truncate.py",
            )
        },
        "limits": limits,
        "results": results,
        "elapsed_seconds": time.monotonic() - start,
        "gpu_hours": 0,
        "model_1024_kl_executed": False,
        "gsm200_executed": False,
    }
    assert not receipt["cuda_initialized"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in receipt.items()
                if k not in ("results", "independent_rne_results")
            }
        ),
        flush=True,
    )
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
