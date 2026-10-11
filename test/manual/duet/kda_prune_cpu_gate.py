"""CPU gate for Kimi production cuts; synthetic-input recurrence + real dumps.

Usage: python test/manual/duet/kda_prune_cpu_gate.py --output /tmp/gate.json
       ... --dump-dir /path/to/state-dump-JOB
No CUDA initialization, model loading, or GPU submission.
"""

import argparse
import hashlib
import json
import sys
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
from test_kimi_duet_fastpath import fast_state, load, load_shared_mgs

reference = load("state_factor")


def fp64_factorize(state, direction, rank, previous, use_previous, omega, explicit):
    """Independent full-FP64 power=1 range finder + Householder QR/Ritz oracle."""
    state, direction = state.double(), direction.double()
    coeff = (state * direction[..., None, :]).sum(-1)
    coeff /= direction.square().sum(-1).clamp_min(1e-12)[..., None]
    if not explicit:
        coeff.zero_()
    content = state - coeff[..., :, None] * direction[..., None, :]
    cold, extra = [value.double() for value in omega]
    warm = torch.cat((previous.double(), extra.expand(*previous.shape[:-1], -1)), -1)
    probe = torch.where(use_previous[:, None, None, None], warm, cold)

    def qr(y):
        return torch.linalg.qr(y, mode="reduced")[0]

    y = content @ probe
    y = content @ qr(content.transpose(-1, -2) @ qr(y))
    q = qr(y)
    reduced = q.transpose(-1, -2) @ content
    gram = reduced @ reduced.transpose(-1, -2)
    eps = 1e-7 * gram.diagonal(dim1=-2, dim2=-1).mean(-1) + 1e-30
    gram += eps[..., None, None] * torch.eye(gram.shape[-1], dtype=torch.float64)
    u = q @ torch.linalg.eigh(gram)[1][..., -rank:]
    v = (u.transpose(-1, -2) @ content).transpose(-1, -2)
    return coeff, u, v, qr(v)


def metric(actual, expected):
    delta = actual.double() - expected.double()
    norm = expected.double().norm().item()
    scale = expected.double().abs().max().item()
    return dict(
        max_abs=delta.abs().max().item(),
        relative_frobenius=delta.norm().item() / max(norm, 1e-30),
        relative_max=delta.abs().max().item() / max(scale, 1e-30),
    )


def evaluate(
    label,
    state,
    direction,
    rank,
    previous=None,
    use_previous=None,
    explicit=True,
    captured=None,
):
    n, h, k, v = state.shape
    if previous is None:
        previous = torch.zeros(n, h, v, rank)
    if use_previous is None:
        use_previous = torch.zeros(n, dtype=torch.bool)
    omega = fast_state.probes(h, v, k, rank, "cpu")
    coeff, u, right, basis = fast_state.factorize(
        state, direction, rank, previous, use_previous, omega, explicit=explicit
    )
    sink = coeff[..., :, None] * direction[..., None, :]
    content = u @ right.transpose(-1, -2)
    actual = sink + content
    expected, bases = [], []
    for i in range(n):
        out, warm = reference.project_state(
            state[i : i + 1],
            direction[i],
            rank,
            previous[i : i + 1] if use_previous[i] else None,
            explicit=explicit,
        )
        expected.append(out)
        bases.append(warm)
    expected = torch.cat(expected)
    old_basis = torch.cat(bases)
    coeff64, u64, v64, basis64 = fp64_factorize(
        state, direction, rank, previous, use_previous, omega, explicit
    )
    sink64 = coeff64[..., :, None] * direction.double()[..., None, :]
    content64 = u64 @ v64.transpose(-1, -2)
    dense64 = sink64 + content64
    per_state = [metric(a, b) for a, b in zip(content, content64)]
    relative = [m["relative_frobenius"] for m in per_state]
    source_content = state - sink
    residual_new = (source_content.double() - content.double()).norm().item()
    residual_old = (state.double() - expected.double()).norm().item()
    residual64 = (state.double() - dense64).norm().item()
    row = dict(
        label=label,
        shape=list(state.shape),
        rank=rank,
        explicit=explicit,
        reconstruction_vs_old=metric(actual, expected),
        UVt_vs_old=metric(content, expected - sink),
        sink_vs_fp64=metric(sink, sink64),
        sink_coefficient_vs_fp64=metric(coeff, coeff64),
        UVt_vs_fp64=metric(content, content64),
        UVt_per_state_vs_fp64=per_state,
        UVt_relative_max=max(relative),
        UVt_relative_median=statistics.median(relative),
        reconstruction_vs_fp64=metric(actual, dense64),
        old_vs_fp64=metric(expected, dense64),
        warm_projector_vs_old=metric(
            basis @ basis.transpose(-1, -2), old_basis @ old_basis.transpose(-1, -2)
        ),
        residual=dict(
            new=residual_new,
            old=residual_old,
            fp64=residual64,
            absolute_delta=abs(residual_new - residual_old),
            relative_delta=abs(residual_new - residual_old)
            / max(state.double().norm().item(), 1e-30),
        ),
    )
    if captured is not None:
        row["strict_real_gate"] = row["UVt_relative_max"] <= 1e-6
        row["captured_old_vs_cpu_old"] = metric(expected, captured)
    # Scale-aware gates avoid elementwise relative errors at zero and preserve
    # sign/rotation freedom. Thresholds fixed before inspecting real dumps.
    comparison = row["reconstruction_vs_old"]
    row["passed"] = (
        torch.isfinite(actual).all().item()
        and comparison["relative_frobenius"] <= 1e-4
        and comparison["relative_max"] <= 1e-3
        and row["residual"]["relative_delta"] <= 5e-5
        and row.get("strict_real_gate", True)
    )
    return row, actual, basis, expected, old_basis


def synthetic_states(layers=20, slots=2, heads=32, dim=128, steps=384):
    """Frozen KDA geometry; true channel-decay delta rule, synthetic k/v/g/beta.

    Stream inputs to keep peak CPU memory bounded. Values include a directional
    sink component. No claim that these are pretrained model activations.
    """
    gen = torch.Generator().manual_seed(62)
    direction = torch.randn(layers, slots, heads, dim, generator=gen)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    state = torch.zeros(layers, slots, heads, dim, dim)
    state64 = state.double()
    log_a = torch.linspace(-6, -3, layers)[:, None, None, None]
    dt = torch.linspace(-1, 1, dim)[None, None, None, :]
    for t in range(steps):
        k = torch.randn(layers, slots, heads, dim, generator=gen)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        v = torch.randn(layers, slots, heads, dim, generator=gen) + 0.3 * direction
        forget = torch.randn(layers, slots, heads, dim, generator=gen) * 0.15
        decay = (-log_a.exp() * torch.nn.functional.softplus(forget + dt)).exp()
        beta = torch.randn(layers, slots, heads, generator=gen).sigmoid()
        state = state * decay[..., :, None]
        error = v - (k[..., :, None] * state).sum(-2)
        state = state + k[..., :, None] * (beta[..., None] * error)[..., None, :]
        state64 = state64 * decay.double()[..., :, None]
        error64 = v.double() - (k.double()[..., :, None] * state64).sum(-2)
        state64 = (
            state64
            + k.double()[..., :, None]
            * (beta.double()[..., None] * error64)[..., None, :]
        )
    return state.flatten(0, 1), direction.flatten(0, 1), state64.flatten(0, 1)


def run(args):
    torch.set_num_threads(args.threads)
    load_shared_mgs()
    rows = []
    metadata = {}
    if not args.only_dumps:
        gen = torch.Generator().manual_seed(62)
        random = torch.randn(40, 32, 128, 128, generator=gen)
        directions = torch.randn(40, 32, 128, generator=gen)
        for explicit in [True, False]:
            state = random
            previous = None
            use = None
            for cut in range(4):
                row, actual, basis, old, old_basis = evaluate(
                    f"random-{explicit}-cut{cut}",
                    state,
                    directions,
                    32,
                    previous,
                    use,
                    explicit,
                )
                rows.append(row)
                state = old + 0.01 * torch.randn(old.shape, generator=gen)
                previous = old_basis
                use = torch.ones(40, dtype=torch.bool)
        state, direction, state64 = synthetic_states()
        metadata["synthetic_input_recurrence"] = dict(
            layers=20,
            slots=2,
            heads=32,
            dim=128,
            steps=384,
            state_fp32_vs_fp64=metric(state, state64),
            inputs="synthetic; seed62; no frozen activation values",
        )
        for explicit in [True, False]:
            previous = None
            use = None
            s = state
            for cut in range(4):
                row, actual, basis, old, old_basis = evaluate(
                    f"synthetic-input-KDA-{explicit}-cut{cut}",
                    s,
                    direction,
                    32,
                    previous,
                    use,
                    explicit,
                )
                rows.append(row)
                s = old + 0.001 * torch.randn(old.shape, generator=gen)
                previous = old_basis
                use = torch.ones(40, dtype=torch.bool)
        for label, s in [
            ("zero", torch.zeros(2, 4, 128, 128)),
            (
                "rank1",
                torch.randn(2, 4, 128, 1, generator=gen)
                @ torch.randn(2, 4, 1, 128, generator=gen),
            ),
            (
                "ill-conditioned",
                torch.diag(torch.logspace(0, -8, 128))[None, None].expand(2, 4, -1, -1),
            ),
        ]:
            rows.append(
                evaluate(label, s, torch.randn(2, 4, 128, generator=gen), 32)[0]
            )
    if args.dump_dir:
        files = sorted(Path(args.dump_dir).rglob("*.pt"))
        metadata["dump_files"] = [
            dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for path in files
        ]
        assert files, "no dump files"
        grouped = {}
        for path in files:
            phase = "prefill" if "-prefill-" in path.name else "warm32"
            grouped.setdefault((path.parent, phase), []).append(path)
        for (parent, phase), group in grouped.items():
            records = []
            for path in group:
                data = torch.load(path, map_location="cpu", weights_only=True)
                assert data["build_sha"] == "7a841e4189445d108085423c3d6054bd9a7b486c"
                assert data["records"], f"{path}: empty records"
                for record, before, after in zip(
                    data["records"], data["dense_before"], data["dense_after"]
                ):
                    assert torch.equal(record["state"], before.transpose(-1, -2))
                    assert torch.equal(record["projected"], after.transpose(-1, -2))
                records.extend(data["records"])
            assert len(records) == 20, (parent, phase, len(records))
            assert len({r["layer_index"] for r in records}) == 20
            state = torch.cat([r["state"] for r in records])
            direction = torch.stack([r["direction"] for r in records])
            rank = records[0]["rank"]
            n, h, k, v = state.shape
            previous = torch.cat(
                [
                    torch.zeros(1, h, v, rank)
                    if r["previous"] is None
                    else r["previous"]
                    for r in records
                ]
            )
            use = torch.tensor([r["previous"] is not None for r in records])
            captured = torch.cat([r["projected"] for r in records])
            row, actual, *_ = evaluate(
                f"{data['metadata']['label']}-{phase}",
                state,
                direction,
                rank,
                previous,
                use,
                records[0]["explicit"],
                captured,
            )
            old_product = torch.cat(
                [r["U"] @ r["V"].transpose(-1, -2) + r["sink"] for r in records]
            )
            row["captured_UVt_sink_vs_after"] = metric(old_product, captured)
            new_sink = torch.einsum("mhkv,mhv->mhk", state, direction)
            new_sink /= direction.square().sum(-1).clamp_min(1e-12)[..., None]
            new_sink = new_sink[..., :, None] * direction[..., None, :]
            row["sink_vs_captured_gpu_old"] = metric(
                new_sink, torch.cat([r["sink"] for r in records])
            )
            row["new_cpu_vs_captured_gpu_old"] = metric(actual, captured)
            row["num_layers"] = len(records)
            print(row["label"], row["passed"], row["reconstruction_vs_old"], flush=True)
            rows.append(row)
    result = dict(
        passed=all(row["passed"] for row in rows),
        metadata=metadata,
        rows=rows,
        thresholds=dict(
            relative_frobenius=1e-4,
            relative_max=1e-3,
            residual_relative_delta=5e-5,
            real_UVt_per_state_vs_fp64=1e-6,
        ),
        torch_version=torch.__version__,
        device="cpu",
        threads=args.threads,
        max_absolute_difference=max(
            row["reconstruction_vs_old"]["max_abs"] for row in rows
        ),
        max_relative_difference=max(
            row["reconstruction_vs_old"]["relative_frobenius"] for row in rows
        ),
    )
    real_errors = [
        m["relative_frobenius"]
        for row in rows
        if "strict_real_gate" in row
        for m in row["UVt_per_state_vs_fp64"]
    ]
    if real_errors:
        result["real_state_count"] = len(real_errors)
        result["real_UVt_relative_max"] = max(real_errors)
        result["real_UVt_relative_median"] = statistics.median(real_errors)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dump-dir")
    parser.add_argument("--only-dumps", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    start = time.monotonic()
    result = run(args)
    result["seconds"] = time.monotonic() - start
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "passed",
                    "max_absolute_difference",
                    "max_relative_difference",
                    "seconds",
                ]
            }
        ),
        flush=True,
    )
    raise SystemExit(0 if result["passed"] else 1)
