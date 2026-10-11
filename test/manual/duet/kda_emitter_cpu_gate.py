"""CPU gate: synthetic-input, real-weight, real initial-state emitter chunks.

The release lacks q weights, so the fixture also contains byte-extracted base
q projections/convolutions. No fixture represents captured hidden activations.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "test/registered/unit/duet"))
from test_kimi_duet_emitter_chunk import cache, fixture

SEED = 20261011
RELATIVE_LIMIT = 1e-6


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def difference(actual, expected):
    delta = actual.double() - expected.double()
    return {
        "absolute_max": delta.abs().max().item() if delta.numel() else 0.0,
        "relative_frobenius": (
            delta.norm() / expected.double().norm().clamp_min(1e-30)
        ).item(),
        "bitwise_equal": torch.equal(actual, expected),
    }


def fp64_scan(k, v, g, beta, initial):
    state = torch.zeros(k.shape[0], k.shape[2], k.shape[3], v.shape[3]).double()
    if initial is not None:
        state = initial.double().clone()
    for t in range(k.shape[1]):
        state = state * g[:, t].double().exp()[..., :, None]
        error = v[:, t].double() - torch.einsum(
            "bhk,bhkv->bhv", k[:, t].double(), state
        )
        state += (
            k[:, t].double()[..., :, None]
            * (beta[:, t, :, None].double() * error)[..., None, :]
        )
    return state


def compare_emitter(emitter, namespace, hidden, *, initial=None, fp64=False):
    """Compare unchanged full emitter to chunked emitter, including scan inputs."""
    length = hidden.shape[0]
    batch = SimpleNamespace(extend_seq_lens_cpu=[length])
    slots = torch.tensor([0])
    expected, actual = cache(emitter, 1), cache(emitter, 1)
    original_scan = namespace["kda_state"]
    reference_inputs = []

    def full_scan(k, v, g, beta, initial_state=None):
        reference_inputs.extend(tensor.clone() for tensor in (k, v, g, beta))
        return original_scan(k, v, g, beta, initial)

    with patch.dict(namespace, kda_state=full_scan):
        emitter._emit_reference(hidden, batch, expected, slots)
    offset = 0
    stage = {
        name: {"squared_difference": 0.0, "squared_reference": 0.0, "absolute_max": 0.0}
        for name in ("k", "v", "g", "beta")
    }
    volumes, sizes = [], []

    def chunk_scan(k, v, g, beta, initial_state=None):
        nonlocal offset
        size = k.shape[1]
        volumes.append(k.shape[0] * size)
        sizes.append(size)
        for name, tensor, reference in zip(stage, (k, v, g, beta), reference_inputs):
            expected_chunk = reference[:, offset : offset + size]
            delta = tensor.double() - expected_chunk.double()
            stage[name]["squared_difference"] += delta.square().sum().item()
            stage[name]["squared_reference"] += (
                expected_chunk.double().square().sum().item()
            )
            stage[name]["absolute_max"] = max(
                stage[name]["absolute_max"], delta.abs().max().item()
            )
        if offset == 0:
            assert initial_state is None
            initial_state = initial
        else:
            assert initial_state is not None and initial_state.dtype == torch.float32
        offset += size
        return original_scan(k, v, g, beta, initial_state)

    with patch.dict(namespace, kda_state=chunk_scan):
        emitter._emit_batched(hidden, batch, actual, slots)
    assert offset == length and max(volumes) <= emitter.max_emitter_tokens
    for values in stage.values():
        values["relative_frobenius"] = (
            values["squared_difference"] / max(values["squared_reference"], 1e-60)
        ) ** 0.5
    row = {
        "length": length,
        "heads": emitter.Hl,
        "head_dim": emitter.Dh,
        "hidden_size": hidden.shape[-1],
        "block_budget": emitter.max_emitter_tokens,
        "blocks": sizes,
        "state": difference(actual.temporal, expected.temporal),
        "conv_tail": difference(actual.conv[0], expected.conv[0]),
        "native_bf16_conv_tail": difference(
            actual.conv[0].bfloat16(), expected.conv[0].bfloat16()
        ),
        "native_bf16_different_elements": (
            actual.conv[0].bfloat16() != expected.conv[0].bfloat16()
        )
        .sum()
        .item(),
        "scan_inputs": stage,
        "initial_state": "real captured dense S" if initial is not None else "zero",
    }
    row["passed"] = (
        max(
            row["state"]["relative_frobenius"],
            row["conv_tail"]["relative_frobenius"],
            *(values["relative_frobenius"] for values in stage.values()),
        )
        <= RELATIVE_LIMIT
    )
    if fp64:
        reference64 = fp64_scan(*reference_inputs, initial)
        row["state_vs_fp64_scan"] = difference(
            actual.temporal.transpose(-1, -2), reference64
        )
    return row


def real_weights(directory, layer):
    emitter, namespace = fixture(heads=32, dim=128, hidden=2304, budget=4096)
    with safe_open(
        str(directory / "duet_components.safetensors"), framework="pt", device="cpu"
    ) as release:

        def component(name):
            return release.get_tensor(f"emitters.{layer}.{name}").float()

        with safe_open(
            str(directory / f"layer{layer}-base-q.safetensors"),
            framework="pt",
            device="cpu",
        ) as base:
            q = base.get_tensor(f"model.layers.{layer}.self_attn.q_proj.weight").float()
            qconv = base.get_tensor(
                f"model.layers.{layer}.self_attn.q_conv1d.weight"
            ).float()
        emitter.norm_w = component("norm.weight")
        emitter.qkv_w = torch.cat(
            (q, component("mixer.k_proj.weight"), component("mixer.v_proj.weight"))
        )
        emitter.conv_w = torch.cat(
            (
                qconv,
                component("mixer.k_conv1d.weight"),
                component("mixer.v_conv1d.weight"),
            )
        ).reshape(3 * 4096, 4)
        for target, source in {
            "f_a_w": "f_a_proj.weight",
            "f_b_w": "f_b_proj.weight",
            "b_w": "b_proj.weight",
            "A_log": "A_log",
            "dt_bias": "dt_bias",
        }.items():
            setattr(emitter, target, component("mixer." + source))
    config = json.loads((directory / "config.json").read_text())
    emitter.eps = config["rms_norm_eps"]
    return emitter, namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quick", action="store_true", help="Smoke only; not the final gate"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.monotonic()
    runtime_path = ROOT / "python/sglang/srt/models/kimi_linear_duet/runtime.py"
    runtime_sha256 = digest(runtime_path)
    generator = torch.Generator().manual_seed(SEED)
    manifest = json.loads((args.weights_dir / "manifest.json").read_text())
    assert (
        digest(args.weights_dir / "duet_components.safetensors") == manifest["sha256"]
    )
    metadata = json.loads((args.weights_dir / "weight-provenance.json").read_text())
    from safetensors.torch import load_file

    for layer in (20, 21, 22, 24, 25):
        tensors = load_file(str(args.weights_dir / f"layer{layer}-base-q.safetensors"))
        for entry in metadata["base_q_tensors"]:
            if entry["source_key"] in tensors:
                raw = tensors[entry["source_key"]].view(torch.uint8).numpy().tobytes()
                assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    rows = []
    # Exact long-prefix partitioning and recurrence, scaled channel/hidden geometry.
    for length in [65] if args.quick else [1, 63, 64, 65, 127, 129, 170880]:
        emitter, namespace = fixture(
            heads=1, dim=1, hidden=2, budget=4096 if length == 170880 else 64
        )
        row = compare_emitter(
            emitter, namespace, torch.randn(length, 2, generator=generator)
        )
        row["kind"] = "synthetic-input, synthetic-weight, scaled long-prefix geometry"
        rows.append(row)
    # Real emitter weights and real dense states from every available length/stage.
    groups = []
    for gpu in range(4):
        for phase in ("prefill", "warm32"):
            selected = []
            for path in sorted((args.dump_dir / f"gpu{gpu}").glob(f"*-{phase}-*.pt")):
                payload = torch.load(path, map_location="cpu", weights_only=False)
                assert (
                    payload["build_sha"] == "7a841e4189445d108085423c3d6054bd9a7b486c"
                )
                selected.extend((path, record) for record in payload["records"])
            groups.append(selected)
    layers = (20,) if args.quick else (20, 21, 22, 24, 25)
    for layer in layers:
        emitter, namespace = real_weights(args.weights_dir, layer)
        for group_index, group in enumerate(groups[:1] if args.quick else groups):
            # Indices15..19 are actual model KDA layers20,21,22,24,25.
            path, record = next(
                (path, record)
                for path, record in group
                if record["layer_index"]
                == {20: 15, 21: 16, 22: 17, 24: 18, 25: 19}[layer]
            )
            lengths = [1, 63, 64, 65, 128, 4095, 4096, 4097]
            length = 65 if args.quick else lengths[group_index]
            row = compare_emitter(
                emitter,
                namespace,
                torch.randn(length, 2304, generator=generator),
                initial=record["state"],
                fp64=length <= 128,
            )
            row.update(
                kind="synthetic-input, real-weight",
                layer=layer,
                initial_source=str(path),
                initial_layer_index=record["layer_index"],
            )
            rows.append(row)
            print(
                f"layer={layer} T={length} relative={row['state']['relative_frobenius']:.9g} pass={row['passed']}",
                flush=True,
            )
        if not args.quick:
            # Real-weight multiple full blocks plus a final one-token tail.
            path, record = next(
                (path, record)
                for path, record in groups[0]
                if record["layer_index"]
                == {20: 15, 21: 16, 22: 17, 24: 18, 25: 19}[layer]
            )
            row = compare_emitter(
                emitter,
                namespace,
                torch.randn(8193, 2304, generator=generator),
                initial=record["state"],
            )
            row.update(
                kind="synthetic-input, real-weight",
                layer=layer,
                initial_source=str(path),
                initial_layer_index=record["layer_index"],
            )
            rows.append(row)
    assert digest(runtime_path) == runtime_sha256, "runtime changed during CPU gate"
    result = {
        "runtime_sha256": runtime_sha256,
        "passed": all(row["passed"] for row in rows),
        "rows": rows,
        "gate": "synthetic-input, real-weight; full vs chunked elementwise and relative Frobenius <=1e-6",
        "seed": SEED,
        "relative_limit": RELATIVE_LIMIT,
        "quick": args.quick,
        "torch": torch.__version__,
        "device": "cpu",
        "threads": 1,
        "weight_provenance": metadata,
        "weight_fixture_sha256": {
            path.name: digest(path)
            for path in sorted(args.weights_dir.iterdir())
            if path.is_file()
        },
        "seconds": time.monotonic() - started,
        "max_state_relative": max(row["state"]["relative_frobenius"] for row in rows),
        "max_state_absolute": max(row["state"]["absolute_max"] for row in rows),
        "max_tail_relative": max(
            row["conv_tail"]["relative_frobenius"] for row in rows
        ),
        "max_tail_absolute": max(row["conv_tail"]["absolute_max"] for row in rows),
        "GPU_results_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("rows", "weight_provenance", "weight_fixture_sha256")
            }
        )
    )
    raise SystemExit(not result["passed"])


if __name__ == "__main__":
    main()
