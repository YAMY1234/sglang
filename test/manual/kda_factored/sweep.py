"""Matched four-cell timing and CUDA-kernel attribution, no weights required.

The frozen first implementation is measured separately with its own source
path. Profiler kernel sums and uninstrumented graph times are distinct metrics.
"""

import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

import torch
from bench import run as dense_bench
from sglang.srt.layers.attention.linear.kernels.kda_factored import packed_decode
from sglang.srt.mem_cache.kda_factored_pool import FactoredKDAConfig


def measure(family, batch, window, dtype, launch=packed_decode, profile=False):
    torch.manual_seed(206)
    heads, dim, rank = (32 if family == "kimi" else 16), 128, 32
    cap = rank + window
    storage = torch.float32 if dtype == "fp32" else torch.bfloat16
    q = torch.zeros(batch, heads, cap, dim, device="cuda", dtype=storage)
    q[..., :rank, :] = (
        torch.linalg.qr(torch.randn(batch, heads, dim, rank, device="cuda"))[0]
        .transpose(-1, -2)
        .to(storage)
    )
    c = torch.zeros_like(q)
    c[..., :rank, :].normal_(std=0.02)
    a = (torch.randn(batch, heads, dim, device="cuda") * 0.02).to(storage)
    vb = torch.randn(heads, dim, device="cuda") * 0.1
    count = torch.full((batch, heads), rank, dtype=torch.int32, device="cuda")
    idx = torch.arange(batch, device="cuda", dtype=torch.int64)
    mixed = (torch.randn(batch, 3 * heads * dim, device="cuda") * 0.1).bfloat16()
    gate = (
        torch.randn(batch, heads * dim, device="cuda") - (0 if family == "kimi" else 3)
    ).bfloat16()
    beta = torch.randn(batch, heads, device="cuda").bfloat16()
    alog = torch.full((heads,), -4.0 if family == "kimi" else 1.0, device="cuda")
    bias = torch.zeros(heads * dim, device="cuda")
    initial = [x.clone() for x in (a, q, c, count)]

    def step():
        return launch(
            mixed,
            gate,
            beta,
            alog,
            bias,
            vb,
            a,
            q,
            c,
            count,
            idx,
            rank,
            window,
            lower_bound=None if family == "kimi" else -5.0,
        )

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _ in range(64):
        graph.replay()
    samples = []
    for _ in range(6):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(64):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 64)
    result = {
        "family": family,
        "batch": batch,
        "rank": rank,
        "window": window,
        "storage": dtype,
        "heads": heads,
        "capacity": cap,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
    }
    actual = sum(x.numel() * x.element_size() for x in (a, q, c, count)) // batch
    cfg = FactoredKDAConfig(rank, window, dtype)
    from types import SimpleNamespace

    assert actual == cfg.state_bytes_per_layer(
        SimpleNamespace(temporal=(heads, dim, dim))
    )
    result.update(
        actual_layer_slot_bytes=actual,
        actual_head_slot_bytes=actual // heads,
        dense_head_bytes=65536,
        byte_ratio=(actual / heads) / 65536,
    )
    if profile:
        for x, saved in zip((a, q, c, count), initial):
            x.copy_(saved)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for _ in range(64):
                step()
            torch.cuda.synchronize()
        events = [
            e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        kernels = [{"name": e.name, "us": e.device_time_total} for e in events]
        groups = {}
        for name in sorted({x["name"] for x in kernels}):
            values = [x["us"] for x in kernels if x["name"] == name]
            groups[name] = {
                "count": len(values),
                "total_us": sum(values),
                "samples_us": values,
            }
        result["profiler_kernel_groups"] = groups
        result["profiler_note"] = (
            "Eager CUDA kernel durations; no subtraction from separately measured graph time. For 64 calls, each Wth step/expiry is due; other expiry launches return immediately."
        )
    return result


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(os.environ["OUT"])
    out.mkdir(parents=True, exist_ok=True)
    old_path = os.environ.get("KDA_OLD_KERNEL")
    if old_path:
        spec = importlib.util.spec_from_file_location(
            "sglang.srt.layers.attention.linear.kernels.kdaf_old", old_path
        )
        old = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = old
        spec.loader.exec_module(old)
        for family in ("kimi", "glm"):
            for batch in (1, 8, 32):
                row = measure(family, batch, 32, "fp32", old.packed_decode, True)
                (out / f"profile-old-{family}-B{batch}.json").write_text(
                    json.dumps(row, indent=2)
                )
                print(
                    json.dumps(
                        {k: v for k, v in row.items() if k != "profiler_kernel_groups"}
                    ),
                    flush=True,
                )
    for family in ("kimi", "glm"):
        for batch in (1, 8, 32):
            baseline = dense_bench(family, batch)
            (out / f"three-arm-{family}-B{batch}.json").write_text(
                json.dumps(baseline, indent=2)
            )
            # Forward/reverse cell order: keep every sample, profiler only last.
            cells = [(32, "fp32"), (16, "fp32"), (32, "bf16"), (16, "bf16")]
            for order, sequence in enumerate((cells, list(reversed(cells)))):
                for w, dtype in sequence:
                    row = measure(family, batch, w, dtype, profile=order == 1)
                    (
                        out / f"sweep-{family}-B{batch}-W{w}-{dtype}-order{order}.json"
                    ).write_text(json.dumps(row, indent=2))
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in row.items()
                                if k != "profiler_kernel_groups"
                            }
                        ),
                        flush=True,
                    )
