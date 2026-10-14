"""FP64 dense recurrence + best rank-R content SVD at the same W boundaries.

Also retains the independent same-MGS arithmetic oracle and the actual dense
KDA + existing model projection baseline. Full uncut recurrence is diagnostic.
No GPU implementation imports SVD/eigh; they are reference-only here.
"""

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from probe import cut, step_ref
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_kda_packed_decode,
)
from sglang.srt.layers.attention.linear.kernels.kda_factored import packed_decode
from sglang.srt.models.glm_duet._state_factor import project_state_batched
from sglang.srt.models.kimi_linear_duet import fast_state


def best(content, rank):
    # Chunk the test-only FP64 SVD to bound solver workspace.
    shape = content.shape
    flat = content.reshape(-1, shape[-2], shape[-1])
    result = torch.empty_like(flat)
    for lo in range(0, flat.shape[0], 32):
        u, s, vh = torch.linalg.svd(flat[lo : lo + 32], full_matrices=False)
        result[lo : lo + 32] = (u[..., :rank] * s[..., None, :rank]) @ vh[..., :rank, :]
    return result.reshape(shape)


def run(family, batch, window=32, dtype="fp32"):
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
    x = (torch.randn(64, batch, 3 * heads * dim, device="cuda") * 0.1).bfloat16()
    g = (
        torch.randn(64, batch, heads * dim, device="cuda")
        - (0 if family == "kimi" else 3)
    ).bfloat16()
    b = torch.randn(64, batch, heads, device="cuda").bfloat16()
    alog = torch.full((heads,), -4.0 if family == "kimi" else 1.0, device="cuda")
    bias = torch.zeros(heads * dim, device="cuda")
    a64, q64, c64 = a.double(), q.double(), c.double()
    sink = a64.clone()
    ref = sink[..., :, None] * vb.double()[..., None, :] + c64.transpose(-1, -2) @ q64
    full = ref.clone()
    low = ref.bfloat16().double()
    service = ref.transpose(-1, -2).float().contiguous()
    service_out = torch.empty(batch, 1, heads, dim, dtype=torch.bfloat16, device="cuda")
    omega, previous = fast_state.probes(heads, dim, dim, rank, "cuda"), None
    xm, gm, bm = x[0].clone(), g[0].clone(), b[0].clone()
    initial = [t.clone() for t in (a, q, c, count)]
    lower = None if family == "kimi" else -5.0

    def launch():
        return packed_decode(
            xm,
            gm,
            bm,
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
            lower_bound=lower,
        )

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = launch()
    for tensor, saved in zip((a, q, c, count), initial):
        tensor.copy_(saved)
    rows, cuts = [], []
    for t in range(64):
        xm.copy_(x[t])
        gm.copy_(g[t])
        bm.copy_(b[t])
        graph.replay()
        fused_recurrent_kda_packed_decode(
            xm, gm, bm, alog, bias, dim**-0.5, service, service_out, idx, True, lower
        )
        query, key, val = x[t].double().reshape(batch, 3, heads, dim).unbind(1)
        query /= (query.square().sum(-1, keepdim=True) + 1e-6).sqrt() * dim**0.5
        key /= (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        gate = g[t].double().reshape(batch, heads, dim)
        decay = (
            -alog.double().exp()[None, :, None] * F.softplus(gate)
            if lower is None
            else lower * torch.sigmoid(alog.double().exp()[None, :, None] * gate)
        ).exp()
        beta = b[t].double().sigmoid()

        def advance(state, decay=decay, val=val, key=key, beta=beta):
            state = state * decay[..., :, None]
            residual = val - (state * key[..., :, None]).sum(-2)
            return (
                state + (beta[..., None] * key)[..., :, None] * residual[..., None, :]
            )

        ref, full, low = advance(ref), advance(full), advance(low).bfloat16().double()
        sink = sink * decay
        sink += beta[..., None] * key * (1 - (sink * key).sum(-1, keepdim=True))
        a64, q64, c64 = step_ref(
            a64, q64, c64, rank + t % window, key, val, decay, beta, vb.double()
        )
        expected = (ref * query[..., :, None]).sum(-2)
        mgs_state = (
            a64[..., :, None] * vb.double()[..., None, :] + c64.transpose(-1, -2) @ q64
        )
        actual = output[0].double()
        norm = expected.norm(dim=-1).clamp_min(1e-30)
        error = (actual - expected).norm(dim=-1) / norm
        envelope = (expected.bfloat16().double() - expected).norm(dim=-1) / norm
        low_output = (low * query[..., :, None]).sum(-2).bfloat16().double()
        mgs_output = (mgs_state * query[..., :, None]).sum(-2)
        rows.append(
            {
                "step": t + 1,
                "best_rank_worst_head_relative": error.max().item(),
                "output_bf16_envelope_worst_head": envelope.max().item(),
                "heads_above_output_bf16_envelope": (error > envelope).sum().item(),
                "best_rank_rmse": (actual - expected).square().mean().sqrt().item(),
                "state_rounded_bf16_envelope_rmse": (low_output - expected)
                .square()
                .mean()
                .sqrt()
                .item(),
                "same_mgs_rmse": (actual - mgs_output).square().mean().sqrt().item(),
                "service_dense_project_rmse": (actual - service_out[:, 0].double())
                .square()
                .mean()
                .sqrt()
                .item(),
                "uncut_rmse_diagnostic": (actual - (full * query[..., :, None]).sum(-2))
                .square()
                .mean()
                .sqrt()
                .item(),
            }
        )
        if (t + 1) % window == 0:
            content = ref - sink[..., :, None] * vb.double()[..., None, :]
            optimal = best(content, rank)
            ref = sink[..., :, None] * vb.double()[..., None, :] + optimal
            low = ref.bfloat16().double()
            cut(q64, c64, rank)
            mgs_content = c64.transpose(-1, -2) @ q64
            cuts.append(
                {
                    "step": t + 1,
                    "best_residual_norm": (content - optimal).norm().item(),
                    "mgs_vs_best_trajectory_content_relative": (
                        (mgs_content - optimal).norm() / optimal.norm()
                    ).item(),
                }
            )
            # Existing service baseline always W32, even in W16 experiment.
        if (t + 1) % 32 == 0:
            if family == "kimi":
                warm = previous is not None
                old = (
                    previous
                    if warm
                    else torch.zeros(batch, heads, dim, rank, device="cuda")
                )
                projected, previous = fast_state.project(
                    service.transpose(-1, -2),
                    vb[None].expand(batch, -1, -1),
                    rank,
                    old,
                    torch.full((batch,), warm, device="cuda"),
                    omega,
                )
            else:
                projected, previous = project_state_batched(
                    service.transpose(-1, -2),
                    vb,
                    rank,
                    previous if previous is not None else [None] * batch,
                    omega_heads=(0, 64),
                )
            service.copy_(projected.transpose(-1, -2))
    return {
        "family": family,
        "batch": batch,
        "rank": rank,
        "window": window,
        "storage": dtype,
        "steps": rows,
        "cuts": cuts,
        "main_reference": "independent FP64 dense recurrence; preserve explicit sink recurrence and best rank-R SVD of content after every Wth output",
        "main_output_rounding_gate": all(
            r["heads_above_output_bf16_envelope"] == 0 for r in rows
        ),
        "state_rounded_envelope_gate": all(
            r["best_rank_rmse"] <= r["state_rounded_bf16_envelope_rmse"] for r in rows
        ),
    }


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(os.environ["OUT"])
    out.mkdir(parents=True, exist_ok=True)
    for family in ("kimi", "glm"):
        for batch in (1, 8, 32):
            result = run(family, batch)
            (out / f"oracle-{family}-B{batch}.json").write_text(
                json.dumps(result, indent=2)
            )
            print(
                json.dumps(
                    {k: v for k, v in result.items() if k not in ("steps", "cuts")}
                ),
                flush=True,
            )
