# ruff: noqa: I001
# Bootstrap lightweight serving modules before importing the GLM adapter.
"""Zero-GPU GLM R32 gate against old projection and a full-FP64 oracle.

Run: python test/manual/duet/glm_fast_state_cpu_gate.py --output /tmp/glm.json
Optional --dump-dir accepts captured records with state (1,H,K,V), direction,
previous, rank and explicit, in the same record schema as the Kimi dump gate.
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).parent))
from kda_prune_cpu_gate import fp64_factorize, metric
from test_kimi_duet_fastpath import load_shared_mgs
from glm_duet import _fast_state, _state_factor
from kimi_linear_duet.small_eigh import jacobi_cpu


def recurrence(state, direction, gen, steps, value_spectrum="flat"):
    # Structured values model a feature spectrum; flat values deliberately
    # stress near-degenerate singular values at the rank-32 truncation boundary.
    scale = (
        torch.logspace(0, -3, state.shape[-1])
        if value_spectrum == "decaying"
        else torch.ones(state.shape[-1])
    )
    for _ in range(steps):
        k = torch.randn(*state.shape[:-1], generator=gen)
        k *= torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        v = torch.randn(*direction.shape, generator=gen) * scale + 0.3 * direction
        decay = (
            -0.003
            * torch.nn.functional.softplus(
                torch.randn(*k.shape, generator=gen) * 0.15
                + torch.linspace(-1, 1, k.shape[-1])
            )
        ).exp()
        beta = torch.randn(*state.shape[:-2], generator=gen).sigmoid()
        state = state * decay[..., :, None]
        error = v - (k[..., :, None] * state).sum(-2)
        state += k[..., :, None] * (beta[..., None] * error)[..., None, :]
    return state


def evaluate(label, state, direction, previous, omega_heads, explicit, jacobi):
    rank = 32
    omega = _fast_state.probes(state, rank, omega_heads)
    zero = state.new_zeros(1, state.shape[1], state.shape[-1], rank)
    packed = torch.cat([zero if p is None else p for p in previous])
    use = torch.tensor([p is not None for p in previous])
    with patch.object(
        _fast_state.fast_state,
        "small_eigh",
        jacobi_cpu if jacobi else _fast_state.fast_state.small_eigh,
    ):
        dense, _bases = _fast_state.project_state_batched(
            state, direction, rank, previous, omega, explicit=explicit
        )
        coeff, u, v, _ = _fast_state.fast_state.factorize(
            state, direction, rank, packed, use, omega, explicit=explicit
        )
    old, old_bases = _state_factor.project_state_batched(
        state, direction, rank, previous, explicit=explicit, omega_heads=omega_heads
    )
    coeff64, u64, v64, _ = fp64_factorize(
        state, direction, rank, packed, use, omega, explicit
    )
    content = u @ v.transpose(-1, -2)
    content64 = u64 @ v64.transpose(-1, -2)
    sink = coeff[..., :, None] * direction[..., None, :]
    old_content = old - sink
    errors = [metric(a, b)["relative_frobenius"] for a, b in zip(content, content64)]
    old_errors = [
        metric(a, b)["relative_frobenius"] for a, b in zip(old_content, content64)
    ]
    coeff_old = torch.einsum("bhkv,bhv->bhk", state, direction)
    coeff_old /= direction.square().sum(-1).clamp_min(1e-12)[..., None]
    if not explicit:
        coeff_old.zero_()
    # Compare every coefficient, including zero entries, without an unstable
    # elementwise relative division. FP64 error uses a per-head coefficient scale.
    coeff_error = (coeff.double() - coeff64).abs()
    coeff_scaled = coeff_error / coeff64.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    row = {
        "label": label,
        "shape": list(state.shape),
        "explicit": explicit,
        "omega_heads": omega_heads,
        "jacobi_cpu_mirror": jacobi,
        "UVt_vs_fp64_max": max(errors),
        "UVt_vs_fp64_median": statistics.median(errors),
        "old_UVt_vs_fp64_max": max(old_errors),
        "old_UVt_vs_fp64_median": statistics.median(old_errors),
        "UVt_per_layer_slot": errors,
        "old_UVt_per_layer_slot": old_errors,
        "new_vs_old": metric(dense, old),
        "sink_coeff_bit_equal_old": torch.equal(coeff, coeff_old),
        "sink_coeff_fp64_max_abs": coeff_error.max().item(),
        "sink_coeff_fp64_scaled_max": coeff_scaled.max().item(),
        "adapter_vs_factors": metric(dense, sink + content),
    }
    row["passed"] = (
        max(errors) <= 1e-6
        and max(old_errors) <= 1e-6
        and row["sink_coeff_bit_equal_old"]
        and coeff_scaled.max().item() <= 1e-6
        and torch.isfinite(dense).all().item()
        and row["adapter_vs_factors"]["max_abs"] == 0
    )
    print(
        json.dumps(
            {
                k: row[k]
                for k in ("label", "passed", "UVt_vs_fp64_max", "old_UVt_vs_fp64_max")
            }
        ),
        flush=True,
    )
    return row, old, old_bases


def run(args):
    torch.set_num_threads(args.threads)
    load_shared_mgs()
    rows, dumps = [], []
    if not args.only_dumps:
        gen = torch.Generator().manual_seed(194)
        state = torch.zeros(34, 16, 128, 128)
        direction = torch.randn(34, 16, 128, generator=gen)
        direction /= direction.norm(dim=-1, keepdim=True)
        state = recurrence(state, direction, gen, 384, args.value_spectrum)
        for explicit in (True, False):
            s = state.clone()
            prev = [None] * 34
            for cut in range(4):
                row, old, prev = evaluate(
                    f"GLM-synthetic-{explicit}-cut{cut}",
                    s,
                    direction,
                    prev,
                    (16, 64),
                    explicit,
                    args.jacobi_cpu,
                )
                rows.append(row)
                s = recurrence(old, direction, gen, 32, args.value_spectrum)
    if args.dump_dir:
        for path in sorted(Path(args.dump_dir).rglob("*.pt")):
            data = torch.load(path, map_location="cpu", weights_only=True)
            records = data["records"]
            assert records and all(r["rank"] == 32 for r in records)
            state = torch.cat([r["state"] for r in records])
            direction = torch.stack([r["direction"] for r in records])
            heads = data.get("omega_heads", (0, state.shape[1]))
            row, *_ = evaluate(
                path.name,
                state,
                direction,
                [r["previous"] for r in records],
                heads,
                records[0]["explicit"],
                args.jacobi_cpu,
            )
            rows.append(row)
            dumps.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        assert dumps, "no real dump files"
    assert rows
    errors = [e for r in rows for e in r["UVt_per_layer_slot"]]
    assert not torch.cuda.is_initialized()
    return {
        "passed": all(r["passed"] for r in rows),
        "rows": rows,
        "dump_files": dumps,
        "UVt_vs_fp64_max": max(errors),
        "UVt_vs_fp64_median": statistics.median(errors),
        "max_dense_difference_old": max(r["new_vs_old"]["max_abs"] for r in rows),
        "torch_version": torch.__version__,
        "cuda_initialized": False,
        "threshold": 1e-6,
        "threads": args.threads,
        "synthetic_seed": 194,
        "value_spectrum": args.value_spectrum,
        "synthetic_geometry": "34 resident KDA layers × 16 heads/rank × 128×128; TP slice16:32/64; 384 initial + 32 updates/cut",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--jacobi-cpu", action="store_true")
    parser.add_argument("--dump-dir")
    parser.add_argument("--only-dumps", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--value-spectrum", choices=("flat", "decaying"), default="flat"
    )
    args = parser.parse_args()
    start = time.monotonic()
    result = run(args)
    result["seconds"] = time.monotonic() - start
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("rows", "dump_files")})
    )
    raise SystemExit(0 if result["passed"] else 1)
