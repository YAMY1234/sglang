"""Once-rounded columns, delayed decay and graph lifecycle: no model weights."""

import json
from types import SimpleNamespace

import torch
from sglang.srt.layers.attention.linear.kernels.kda_factored import packed_decode
from sglang.srt.mem_cache.kda_factored_pool import FactoredKDAConfig, FactoredKDAPool


def main():
    torch.manual_seed(207)
    cfg = FactoredKDAConfig(32, 32, "bf16")
    shape = SimpleNamespace(temporal=(2, 128, 128), conv=[])
    pool = FactoredKDAPool(2, SimpleNamespace(shape=shape), [0], "cuda", cfg)
    pool.bind_directions({0: torch.randn(2, 128, device="cuda") * 0.02})
    slots = torch.tensor([0, 1], device="cuda")
    initial = torch.randn(2, 2, 128, 8, device="cuda") * 0.02
    initial = initial @ initial.transpose(-1, -2)
    pool.commit(0, slots, initial)
    pool.dense(0, slots).clone()
    mixed = torch.randn(2, 768, device="cuda").bfloat16() * 0.1
    gate = torch.zeros(2, 256, device="cuda", dtype=torch.bfloat16)
    beta = torch.zeros(2, 2, device="cuda", dtype=torch.bfloat16)
    alog = torch.ones(2, device="cuda")
    bias = torch.zeros(256, device="cuda")
    rows = []
    for family, lower, gate_value in [("kimi", None, 0.0), ("glm", -5.0, 6.0)]:
        pool.commit(0, slots, initial)
        ref = pool.dense(0, slots).clone()
        gate.fill_(gate_value)
        first_window = []
        for step in range(1024):
            mixed.normal_(std=0.1)
            before_u = pool.coeff.clone() if step < 32 else None
            before_v = pool.basis.clone() if step < 32 else None
            count = 32 + step % 32
            result = packed_decode(
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
            q, k, v = mixed.float().reshape(2, 3, 2, 128).unbind(1)
            k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1.0e-6)
            q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1.0e-6) / 128**0.5
            z = gate.float().reshape(2, 2, 128)
            lg = (
                -alog.exp()[None, :, None] * torch.nn.functional.softplus(z)
                if lower is None
                else lower * z.sigmoid()
            )
            ref *= lg.exp()[:, :, None, :]
            residual = v - (ref * k[:, :, None, :]).sum(-1)
            ref += 0.5 * residual[..., None] * k[:, :, None, :]
            expected = (ref * q[:, :, None, :]).sum(-1)
            if step < 31:
                torch.testing.assert_close(
                    pool.coeff[..., :count, :], before_u[..., :count, :], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    pool.basis[..., :count, :], before_v[..., :count, :], rtol=0, atol=0
                )
                rel = (result[0].float() - expected).norm() / expected.norm().clamp_min(
                    1.0e-12
                )
                first_window.append(rel.item())
                assert rel < 0.03, (family, step, rel)
            assert torch.isfinite(result).all(), (family, step)
            if step in (31, 63, 255, 1023):
                dense = pool.dense(0, slots)
                assert torch.isfinite(dense).all()
                rows.append(
                    {
                        "family": family,
                        "step": step + 1,
                        "state_relative_error": (
                            (dense - ref).norm() / ref.norm()
                        ).item(),
                    }
                )
        saved = pool.get_cpu_slots(slots)
        recovered = pool.dense(0, slots).clone()
        pool.copy_slots(slots, slots.flip(0))
        torch.testing.assert_close(
            pool.dense(0, slots.flip(0)), recovered, rtol=0, atol=0
        )
        pool.reset_slots(slots)
        pool.load_cpu_slots(saved, slots)
        torch.testing.assert_close(pool.dense(0, slots), recovered, rtol=0, atol=0)
        print(
            json.dumps(
                {"family": family, "first_window_max_relative": max(first_window)}
            ),
            flush=True,
        )
    actual = sum(t.numel() * t.element_size() for t in pool.tensors())
    assert actual == 3 * cfg.state_bytes_per_layer(shape)
    print(
        json.dumps(
            {
                "gate": "PASS",
                "bytes_per_head": cfg.state_bytes_per_layer(shape) // 2,
                "diagnostic_random_trajectory": rows,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    main()
