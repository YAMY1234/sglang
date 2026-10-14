"""Three arms, same GPU/inputs, 64-step windows including two expiries.

Dense and factored steps replay one-token graphs; the existing dense+W32 arm
replays the dense graph and runs its existing projection outside the graph.
CUDA events include gaps introduced by the host projection path. Report both
orders; no clock or cache controls are changed between arms.
"""

import json
import os
from pathlib import Path

import torch
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_kda_packed_decode,
)
from sglang.srt.layers.attention.linear.kernels.kda_factored import packed_decode
from sglang.srt.models.glm_duet._state_factor import project_state_batched
from sglang.srt.models.kimi_linear_duet import fast_state


def run(family, batch):
    torch.manual_seed(206)
    h, k, r, w = (32 if family == "kimi" else 16), 128, 32, 32
    lower = None if family == "kimi" else -5.0
    vb = torch.randn(h, k, device="cuda") * 0.1
    a = torch.randn(batch, h, k, device="cuda") * 0.02
    q = torch.zeros(batch, h, 64, k, device="cuda")
    q[..., :r, :] = torch.linalg.qr(torch.randn(batch, h, k, r, device="cuda"))[
        0
    ].transpose(-1, -2)
    c = torch.zeros_like(q)
    c[..., :r, :].normal_(std=0.02)
    count = torch.full((batch, h), r, dtype=torch.int32, device="cuda")
    dense = (
        vb[None, :, :, None] * a[:, :, None, :] + q.transpose(-1, -2) @ c
    ).contiguous()
    initial = [x.clone() for x in (a, q, c, count, dense)]
    mixed = (torch.randn(batch, 3 * h * k, device="cuda") * 0.1).bfloat16()
    gate = (
        torch.randn(batch, h * k, device="cuda") - (0 if lower is None else 3)
    ).bfloat16()
    beta = torch.randn(batch, h, device="cuda").bfloat16()
    alog = torch.full((h,), -4.0 if lower is None else 1.0, device="cuda")
    bias = torch.zeros(h * k, device="cuda")
    idx = torch.arange(batch, device="cuda", dtype=torch.int64)
    output = torch.empty(batch, 1, h, k, device="cuda", dtype=torch.bfloat16)

    def ds():
        fused_recurrent_kda_packed_decode(
            mixed, gate, beta, alog, bias, k**-0.5, dense, output, idx, True, lower
        )

    def fs():
        packed_decode(
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
            r,
            w,
            lower_bound=lower,
        )

    ds()
    fs()
    torch.cuda.synchronize()
    dg, fg = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(dg):
        ds()
    with torch.cuda.graph(fg):
        fs()
    omega = fast_state.probes(h, k, k, r, "cuda")
    previous = None

    def project():
        nonlocal previous
        if family == "kimi":
            warm = previous is not None
            old = previous if warm else torch.zeros(batch, h, k, r, device="cuda")
            result, previous = fast_state.project(
                dense.transpose(-1, -2),
                vb[None].expand(batch, -1, -1),
                r,
                old,
                torch.full((batch,), warm, device="cuda"),
                omega,
            )
        else:
            result, previous = project_state_batched(
                dense.transpose(-1, -2),
                vb,
                r,
                previous if previous is not None else [None] * batch,
                omega_heads=(0, 64),
            )
        dense.copy_(result.transpose(-1, -2))

    project()
    torch.cuda.synchronize()

    def cycle(arm):
        for step in range(64):
            (fg if arm == "factored" else dg).replay()
            if arm == "dense_w32" and (step + 1) % 32 == 0:
                project()

    samples = {x: [] for x in ("dense", "dense_w32", "factored")}
    for order in (list(samples), list(reversed(samples))):
        for arm in order:
            for x, saved in zip((a, q, c, count, dense), initial):
                x.copy_(saved)
            previous = None
            cycle(arm)
            for _ in range(3):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                cycle(arm)
                end.record()
                end.synchronize()
                samples[arm].append(start.elapsed_time(end) / 64)
    return {
        "family": family,
        "batch": batch,
        "heads_per_rank": h,
        "ms_per_step": samples,
        "factor_bytes_per_head": 66052,
        "dense_bytes_per_head": 65536,
    }


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(os.environ["OUT"])
    out.mkdir(parents=True, exist_ok=True)
    for family in ("kimi", "glm"):
        for batch in (1, 8, 32):
            result = run(family, batch)
            (out / f"bench-{family}-B{batch}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(json.dumps(result), flush=True)
