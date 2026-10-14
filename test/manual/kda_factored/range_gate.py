"""Heterogeneous key decay must not erase newly appended slow-channel values."""

import json
from types import SimpleNamespace

import torch
from sglang.srt.layers.attention.linear.kernels.kda_factored import packed_decode
from sglang.srt.mem_cache.kda_factored_pool import FactoredKDAConfig, FactoredKDAPool

for storage in ("fp32", "bf16"):
    for family in ("kimi", "glm"):
        torch.manual_seed(617)
        cfg = FactoredKDAConfig(32, 32, storage, deferred=True)
        pool = FactoredKDAPool(
            1,
            SimpleNamespace(shape=SimpleNamespace(temporal=(2, 128, 128), conv=[])),
            [0],
            "cuda",
            cfg,
        )
        pool.bind_directions({0: torch.zeros(2, 128, device="cuda")})
        slots = torch.tensor([0], device="cuda")
        ref = torch.zeros(1, 2, 128, 128, device="cuda", dtype=torch.float64)
        gate = torch.full((1, 256), -6.0, device="cuda", dtype=torch.bfloat16)
        gate[:, ::2] = 1.0 if family == "kimi" else 6.0
        beta = torch.zeros(1, 2, device="cuda", dtype=torch.bfloat16)
        alog = torch.ones(2, device="cuda")
        bias = torch.zeros(256, device="cuda")
        lower = None if family == "kimi" else -5.0
        z = gate.double().reshape(1, 2, 128)
        logg = (
            -alog.double().exp()[None, :, None] * torch.nn.functional.softplus(z)
            if lower is None
            else lower * (alog.double().exp()[None, :, None] * z).sigmoid()
        )
        errors = []
        for step in range(31):
            mixed = (torch.randn(1, 768, device="cuda") * 0.1).bfloat16()
            old_u, old_v = pool.coeff.clone(), pool.basis.clone()
            output = packed_decode(
                mixed,
                gate,
                beta,
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
                log_decay=pool.log_decay[0],
                column_exponent=pool.column_exponent[0],
            )
            torch.testing.assert_close(
                pool.coeff[..., : 32 + step, :],
                old_u[..., : 32 + step, :],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                pool.basis[..., : 32 + step, :],
                old_v[..., : 32 + step, :],
                rtol=0,
                atol=0,
            )
            q, k, v = mixed.double().reshape(1, 3, 2, 128).unbind(1)
            k *= torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
            q *= torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) / 128**0.5
            ref *= logg.exp()[:, :, None, :]
            residual = v - (ref * k[:, :, None, :]).sum(-1)
            ref += 0.5 * residual[..., None] * k[:, :, None, :]
            expected = (ref * q[:, :, None, :]).sum(-1)
            # BF16 output cast is part of the API, even for FP32 factors.
            relative = (output[0].double() - expected).norm() / expected.norm()
            state_relative = (pool.dense(0, slots).double() - ref).norm() / ref.norm()
            errors.append((relative.item(), state_relative.item()))
            assert torch.isfinite(output).all() and torch.isfinite(pool.coeff).all()
        limits = (0.01, 0.0002) if storage == "fp32" else (0.03, 0.01)
        assert max(x[0] for x in errors) < limits[0], (storage, family, errors)
        assert max(x[1] for x in errors) < limits[1], (storage, family, errors)
        print(
            json.dumps(
                {
                    "range_gate": "PASS",
                    "family": family,
                    "storage": storage,
                    "steps": 31,
                    "log_decay_span": float(
                        (logg.max(-1).values - logg.min(-1).values).max() * 31
                    ),
                    "max_output_relative_error": max(x[0] for x in errors),
                    "max_state_relative_error": max(x[1] for x in errors),
                }
            ),
            flush=True,
        )
