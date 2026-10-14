"""Matched dense vs immutable BF16 columns, same inputs and GPU, B1/8/32."""

import hashlib
import json
import os
import statistics
from pathlib import Path

import torch
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_kda_packed_decode,
)
from sglang.srt.layers.attention.linear.kernels.kda_factored import packed_decode


def run(family, batch):
    torch.manual_seed(207)
    h, d, r, w = (32 if family == "kimi" else 16), 128, 32, 32
    lower = None if family == "kimi" else -5.0
    vb = torch.randn(h, d, device="cuda") * 0.1
    a = torch.randn(batch, h, d, device="cuda") * 0.02
    V = torch.zeros(batch, h, r + w, d, device="cuda", dtype=torch.bfloat16)
    V[..., :r, :] = torch.linalg.qr(torch.randn(batch, h, d, r, device="cuda"))[
        0
    ].transpose(-1, -2)
    U = torch.zeros_like(V)
    U[..., :r, :] = torch.randn(batch, h, r, d, device="cuda") * 0.02
    L = torch.zeros_like(a)
    E = torch.zeros(batch, h, r + w, device="cuda")
    count = torch.full((batch, h), r, device="cuda", dtype=torch.int32)
    idx = torch.arange(batch, device="cuda", dtype=torch.int64)
    dense = (
        V.float().transpose(-1, -2) @ U.float()
        + vb[None, :, :, None] * a[:, :, None, :]
    ).contiguous()
    mixed = (torch.randn(batch, 3 * h * d, device="cuda") * 0.1).bfloat16()
    gate = (
        torch.randn(batch, h * d, device="cuda") - (0 if lower is None else 3)
    ).bfloat16()
    beta = torch.randn(batch, h, device="cuda").bfloat16()
    alog = torch.full((h,), -4.0 if lower is None else 1.0, device="cuda")
    bias = torch.zeros(h * d, device="cuda")
    output = torch.empty(batch, 1, h, d, device="cuda", dtype=torch.bfloat16)
    tensors = (a, V, U, L, E, count, dense)
    initial = [x.clone() for x in tensors]

    def reset():
        for x, y in zip(tensors, initial):
            x.copy_(y)

    def native():
        fused_recurrent_kda_packed_decode(
            mixed, gate, beta, alog, bias, d**-0.5, dense, output, idx, True, lower
        )

    def factored():
        return packed_decode(
            mixed,
            gate,
            beta,
            alog,
            bias,
            vb,
            a,
            V,
            U,
            count,
            idx,
            r,
            w,
            lower_bound=lower,
            log_decay=L,
            column_exponent=E,
        )

    graphs = {}
    for name, func in [("dense", native), ("factored", factored)]:
        func()
        torch.cuda.synchronize()
        graphs[name] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[name]):
            func()
    samples = {x: [] for x in graphs}
    for order in [list(graphs), list(reversed(graphs))]:
        for name in order:
            reset()
            for _ in range(64):
                graphs[name].replay()
            for _ in range(3):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(64):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) / 64)
    reset()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for _ in range(64):
            factored()
        torch.cuda.synchronize()
    durations = [
        x.device_time_total
        for x in prof.events()
        if x.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert len(durations) == 64
    due = sum(x for i, x in enumerate(durations, 1) if i % w == 0) / sum(durations)
    D, F = [statistics.median(samples[k]) for k in ["dense", "factored"]]
    normal = F * (1 - due) * w / (w - 1)
    sha = hashlib.sha256()
    for x in [*initial, mixed, gate, beta, alog, bias, vb]:
        sha.update(x.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return {
        "family": family,
        "batch": batch,
        "rank": r,
        "window": w,
        "storage": "bf16",
        "computation": "fp32",
        "input_sha256": sha.hexdigest(),
        "samples_ms": samples,
        "dense_ms": D,
        "factored_ms": F,
        "factor_over_dense": F / D,
        "kernel_durations_us": durations,
        "due_step_share": due,
        "estimated_normal_ms": normal,
        "estimated_extra_cleanup_budget_ms": w * (D - normal),
        "budget_assumption": "profiler shares proportionally attribute graph step time",
        "bytes_per_head": sum(x.numel() * x.element_size() for x in tensors[:-1])
        // batch
        // h,
    }


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(os.environ["OUT"])
    root.mkdir(parents=True, exist_ok=True)
    for f in ["kimi", "glm"]:
        for b in [1, 8, 32]:
            result = run(f, b)
            (root / f"delayed-{f}-B{b}.json").write_text(json.dumps(result, indent=2))
            print(
                json.dumps(
                    {k: v for k, v in result.items() if k != "kernel_durations_us"}
                ),
                flush=True,
            )
