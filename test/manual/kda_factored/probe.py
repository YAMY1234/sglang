"""GPU numerical/graph gate, with independent FP64 dense recurrence.

No weights required. Reports projection distortion separately from arithmetic
error. The BF16 envelope is measured using the same projected FP64 states,
rounded to BF16 after each dense update; no absolute BF16 epsilon is used.
"""

import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_kda_packed_decode,
)
from sglang.srt.layers.attention.linear.kernels.kda_factored import packed_decode


def mgs(y, rank):
    q = y.clone()
    n0 = y.norm(dim=-2)
    for p in range(2):
        for j in range(rank):
            col = q[..., j].clone()
            prev = q[..., :j]
            col -= (prev * (prev * col[..., None]).sum(-2, keepdim=True)).sum(-1)
            n = col.norm(dim=-1)
            ok = n > 1e-12
            if p == 0:
                ok &= n > 1e-4 * n0[..., j]
            q[..., j] = torch.where(
                ok[..., None], col / n.clamp_min(1e-30)[..., None], 0.0
            )
    return q


def step_ref(a, q, c, count, key, value, decay, beta, vb):
    a = a * decay
    a += beta[..., None] * key * (1 - (a * key).sum(-1, keepdim=True))
    c = c * decay[..., None, :]
    residual = value - vb - ((c * key[..., None, :]).sum(-1)[..., None] * q).sum(-2)
    proj = (q * residual[..., None, :]).sum(-1)
    orth = residual - (q * proj[..., None]).sum(-2)
    corr = (q * orth[..., None, :]).sum(-1)
    orth -= (q * corr[..., None]).sum(-2)
    proj += corr
    norm = orth.norm(dim=-1)
    keep = norm > 1e-7 * residual.norm(dim=-1)
    q[..., count, :] = torch.where(
        keep[..., None], orth / norm.clamp_min(1e-30)[..., None], 0.0
    )
    proj[..., count] = torch.where(keep, norm, 0.0)
    c += proj[..., None] * (beta[..., None] * key)[..., None, :]
    return a, q, c


def cut(q, c, rank):
    gram = c @ c.transpose(-1, -2)
    d = gram.diagonal(dim1=-2, dim2=-1)
    order = d.argsort(dim=-1, descending=True, stable=True)[..., :rank]
    z = torch.zeros(*d.shape, rank, device=d.device, dtype=d.dtype)
    z.scatter_(-2, order[..., None, :], 1.0)
    for _ in range(3):
        z = mgs(gram @ z, rank)
    qn, cn = z.transpose(-1, -2) @ q, z.transpose(-1, -2) @ c
    q.zero_()
    c.zero_()
    q[..., :rank, :] = qn
    c[..., :rank, :] = cn


def run(family, batch):
    torch.manual_seed(206)
    h, k, v, rank, window = (32 if family == "kimi" else 16), 128, 128, 32, 32
    cap, steps = 64, 64
    dev = "cuda"
    basis = torch.zeros(batch, h, cap, v, device=dev)
    basis[..., :rank, :] = torch.linalg.qr(torch.randn(batch, h, v, rank, device=dev))[
        0
    ].transpose(-1, -2)
    coeff = torch.zeros(batch, h, cap, k, device=dev)
    coeff[..., :rank, :].normal_(std=0.02)
    sink = torch.randn(batch, h, k, device=dev) * 0.02
    vb = torch.randn(h, v, device=dev) * 0.1
    count = torch.full((batch, h), rank, dtype=torch.int32, device=dev)
    idx = torch.arange(batch, dtype=torch.int64, device=dev)
    lower = None if family == "kimi" else -5.0
    alog = torch.full((h,), -4.0 if lower is None else 1.0, device=dev)
    bias = torch.zeros(h, k, device=dev)
    mixed = (torch.randn(steps, batch, 3 * h * k, device=dev) * 0.1).bfloat16()
    gates = (
        torch.randn(steps, batch, h * k, device=dev) - (0 if lower is None else 3.0)
    ).bfloat16()
    betas = torch.randn(steps, batch, h, device=dev).bfloat16()
    a64, q64, c64 = sink.double(), basis.double(), coeff.double()

    def dense(a, q, c):
        return a[..., :, None] * vb.double()[..., None, :] + c.transpose(-1, -2) @ q

    ref = dense(a64, q64, c64)
    low = ref.bfloat16().double()
    uncut = ref.clone()
    uncut_low = ref.bfloat16().double()
    # One-token graph with changing inputs exercises captured live pointers.
    xm, gm, bm = mixed[0].clone(), gates[0].clone(), betas[0].clone()
    initial = [x.clone() for x in (sink, basis, coeff, count)]

    def launch():
        return packed_decode(
            xm,
            gm,
            bm,
            alog,
            bias,
            vb,
            sink,
            basis,
            coeff,
            count,
            idx,
            rank,
            window,
            lower_bound=lower,
        )

    for _ in range(2):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = launch()
    for x, saved in zip((sink, basis, coeff, count), initial):
        x.copy_(saved)
    errors, envelopes, full_errors, state_errors = [], [], [], []
    full_envelopes = []
    for t in range(steps):
        xm.copy_(mixed[t])
        gm.copy_(gates[t])
        bm.copy_(betas[t])
        graph.replay()
        query, key, val = mixed[t].double().reshape(batch, 3, h, k).unbind(1)
        query = query / (query.square().sum(-1, keepdim=True) + 1e-6).sqrt() / k**0.5
        key = key / (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        g = gates[t].double().reshape(batch, h, k) + bias.double()
        logd = (
            -alog.double().exp()[None, :, None] * F.softplus(g)
            if lower is None
            else lower * torch.sigmoid(alog.double().exp()[None, :, None] * g)
        )
        decay = logd.exp()
        beta = betas[t].double().sigmoid()

        def advance(s, decay=decay, beta=beta, val=val, key=key):
            s = s * decay[..., :, None]
            delta = beta[..., None] * (val - (s * key[..., :, None]).sum(-2))
            return s + key[..., :, None] * delta[..., None, :]

        ref = advance(ref)
        uncut = advance(uncut)
        uncut_low = advance(uncut_low).bfloat16().double()
        low = advance(low).bfloat16().double()
        expected = (ref * query[..., :, None]).sum(-2)
        envelope = (
            ((low * query[..., :, None]).sum(-2).bfloat16().double() - expected)
            .square()
            .mean()
            .sqrt()
        )
        errors.append((out[0].double() - expected).square().mean().sqrt().item())
        envelopes.append(envelope.item())
        full_envelopes.append(
            (
                (uncut_low * query[..., :, None]).sum(-2).bfloat16().double()
                - (uncut * query[..., :, None]).sum(-2)
            )
            .square()
            .mean()
            .sqrt()
            .item()
        )
        full_errors.append(
            (out[0].double() - (uncut * query[..., :, None]).sum(-2))
            .square()
            .mean()
            .sqrt()
            .item()
        )
        a64, q64, c64 = step_ref(
            a64, q64, c64, rank + t % window, key, val, decay, beta, vb.double()
        )
        if (t + 1) % window == 0:
            cut(q64, c64, rank)
            ref = dense(a64, q64, c64)
            low = ref.bfloat16().double()
        state_errors.append(
            (
                dense(
                    sink.double(),
                    basis.double()
                    * (torch.arange(cap, device=dev)[None, None, :] < count[..., None])[
                        ..., None
                    ],
                    coeff.double(),
                )
                - ref
            )
            .norm()
            .item()
            / ref.norm().item()
        )

    # Timing includes both captured launches and every Wth expiry.
    def measure(fn):
        for _ in range(64):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(64):
                fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 64)
        return samples

    factor_ms = measure(graph.replay)
    dense_state = (
        dense(*[x.double() for x in initial[:3]]).transpose(-1, -2).float().contiguous()
    )
    dense_out = xm.new_empty(batch, 1, h, v)

    def stock():
        fused_recurrent_kda_packed_decode(
            xm,
            gm,
            bm,
            alog,
            bias.reshape(-1),
            k**-0.5,
            dense_state,
            dense_out,
            ssm_state_indices=idx,
            use_qk_l2norm_in_kernel=True,
            lower_bound=lower,
        )

    stock()
    torch.cuda.synchronize()
    dg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(dg):
        stock()
    dense_ms = measure(dg.replay)
    return {
        "family": family,
        "batch": batch,
        "heads_per_rank": h,
        "rank": rank,
        "window": window,
        "steps": steps,
        "output_rmse": errors,
        "bf16_envelope_rmse": envelopes,
        "numerical_pass": all(e <= b for e, b in zip(errors, envelopes)),
        "unprojected_output_rmse": full_errors,
        "unprojected_bf16_envelope_rmse": full_envelopes,
        "unprojected_gate_pass": all(
            e <= b for e, b in zip(full_errors, full_envelopes)
        ),
        "state_relative_error": state_errors,
        "factor_ms": factor_ms,
        "dense_ms": dense_ms,
        "factor_bytes_per_head": 4 * (k + cap * (k + v) + 1),
        "dense_bytes_per_head": 4 * k * v,
    }


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    outdir = Path(os.environ["OUT"])
    outdir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "capability": torch.cuda.get_device_capability(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            }
        ),
        flush=True,
    )
    for family in ("kimi", "glm"):
        for batch in (1, 8, 32):
            start = time.monotonic()
            result = run(family, batch)
            (outdir / f"{family}-B{batch}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {
                        "family": family,
                        "batch": batch,
                        "pass_gate": result["numerical_pass"],
                        "max_error": max(result["output_rmse"]),
                        "max_state_error": max(result["state_relative_error"]),
                        "factor_ms": result["factor_ms"],
                        "dense_ms": result["dense_ms"],
                        "wall_s": time.monotonic() - start,
                    }
                ),
                flush=True,
            )
