"""CPU interpreter: diagnostic counts reflect due matrix work, including padding."""

import torch
from numeric_gate import kernel


def run():
    torch.manual_seed(206)
    l, slots, h, d = 2, 4, 1, 128
    s = torch.randn(l, slots, h, d, d)
    vb = torch.randn(l, h, d)
    warm = torch.zeros(l, slots, h, d, 32)
    omega = torch.randn(h, d, 40)
    extra = torch.randn(h, d, 8)
    indices = torch.tensor([0, 2, -1], dtype=torch.int32)
    phase = torch.tensor([0, 31, 31, 31], dtype=torch.int32)
    pending = torch.tensor([True, False, False, False])
    active = torch.tensor([True, True, True, False])
    audit = torch.zeros(l, slots, h, 2, dtype=torch.int32)
    workspace = kernel.ProjectionWorkspace(3, h, l, "cpu")
    for cold, slot in [(True, 0), (False, 2)]:
        for audited in (False, True):
            workspace.content.fill_(float("nan"))
            workspace.q.fill_(float("nan"))
            kernel._prepare[(l * len(indices) * h,)](
                s,
                vb,
                warm,
                omega,
                extra,
                workspace.content,
                workspace.coeff,
                workspace.q,
                indices,
                phase,
                pending,
                active,
                len(indices),
                h,
                slots,
                0,
                cold,
                32,
                d,
                32,
                40,
                64,
                Audit=audit if audited else None,
                num_warps=8,
                enable_fp_fusion=False,
            )
            if not audited:
                baseline = workspace.content.clone(), workspace.q.clone()
            else:
                for x, y in zip(baseline, [workspace.content, workspace.q]):
                    assert torch.allclose(x, y, rtol=0, atol=0, equal_nan=True)
        expected = torch.zeros(l, slots, h, dtype=torch.int32)
        expected[:, slot] = 1
        assert torch.equal(audit[..., int(cold)], expected)
    assert not torch.cuda.is_initialized()
    print(
        "GLMW32_AUDIT_CPU PASS exact counted/uninstrumented matrix work; due slots, inactive slots, padding"
    )


if __name__ == "__main__":
    run()
