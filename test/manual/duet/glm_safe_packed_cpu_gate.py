"""Run the real GLM safe-gate Triton kernels in the CPU interpreter.

Only GPU import/PDL/decorator plumbing is replaced by CPU plumbing; the kernel
functions and their wrappers are compiled from the repository source unchanged.
This is CPU N/S evidence, not GPU compilation/capture/performance qualification.
"""

import argparse
import ast
import json
import os
import sys
import time
import types
from pathlib import Path

os.environ["TRITON_INTERPRET"] = "1"
import torch
import triton.language as tl

ROOT = Path(__file__).resolve().parents[3]
ATOL, RTOL = 2e-2, 1e-2  # existing TestKDAPackedDecode gate; never widened


def kernels():
    modules = []
    for name in ("fused_recurrent", "fused_sigmoid_gating_recurrent"):
        path = ROOT / "python/sglang/kernels/ops/attention/fla" / (name + ".py")
        tree = ast.parse(path.read_text(), filename=str(path))
        tree.body = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").startswith("sglang.")
            )
        ]
        module = types.ModuleType("glm_cpu_" + name)
        module.__file__ = str(path)
        # The interpreter patches tl's builtins at launch. Resolve the same
        # tl.exp through that patched namespace (an import-time alias is stale).
        module.exp = lambda value: tl.exp(value)
        module.input_guard = lambda fn: fn
        module.is_arch_support_pdl = lambda: False
        sys.modules[module.__name__] = module
        exec(compile(tree, str(path), "exec"), module.__dict__)  # noqa: S102 - trusted repo source
        modules.append(module)
    return modules[0].fused_recurrent_kda_packed_decode, modules[
        1
    ].fused_sigmoid_gating_delta_rule_update


def inputs(batch=3, heads=2, dim=16, dtype=torch.float32, device="cpu", seed=31):
    gen = torch.Generator(device=device).manual_seed(seed)
    rand = lambda *shape: torch.randn(*shape, generator=gen, device=device)
    # Strided slot pitch: each logical slot shares an envelope with two other layers.
    envelope = rand(batch + 2, 3, heads, dim, dim) * 0.05
    state = envelope[:, 1]
    qkv = (rand(batch, 3 * heads * dim) * 0.2).to(dtype)
    a = (rand(batch, heads * dim) * 0.5).to(dtype)
    b = (rand(batch, 3 * heads) * 0.8).to(dtype)[:, :heads].unsqueeze(0)
    A_log = rand(heads) * 0.2
    dt = rand(heads * dim) * 0.1
    indices = torch.arange(batch, dtype=torch.int32, device=device)
    indices[-1] = -1
    return qkv, a, b, A_log, dt, state, indices


def pair(packed, generic, tensors, lower_bound=-5.0, norm=True, reference_state=None):
    qkv, a, b, A_log, dt, state, indices = tensors
    B, H, V, K = qkv.shape[0], *state.shape[1:]
    # Preserve the envelope pitch in independent source/target states.
    storage = torch.empty(B + 2, 3, H, V, K, device=state.device)
    s1 = storage[:, 1]
    s1.copy_(state)
    storage2 = storage.clone()
    s2 = storage2[:, 1]
    if reference_state is not None:
        s2.copy_(reference_state)
    before = s1.clone()
    output = torch.empty(B, 1, H, V, dtype=qkv.dtype, device=state.device)
    packed(
        qkv,
        a,
        b.reshape(B, H),
        A_log,
        dt,
        K**-0.5,
        s1,
        output,
        indices,
        use_qk_l2norm_in_kernel=norm,
        lower_bound=lower_bound,
        allow_cuda=False,
    )
    q, k, v = qkv.split([H * K, H * K, H * V], -1)
    reference = generic(
        A_log=A_log,
        a=a,
        dt_bias=dt,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q.view(1, B, H, K),
        k=k.view(1, B, H, K),
        v=v.view(1, B, H, V),
        b=b,
        initial_state_source=s2,
        initial_state_indices=indices,
        cu_seqlens=torch.arange(B + 1, dtype=torch.int32, device=state.device),
        use_qk_l2norm_in_kernel=norm,
        is_kda=True,
        lower_bound=lower_bound,
    )
    torch.testing.assert_close(
        output.transpose(0, 1).float(), reference.float(), atol=ATOL, rtol=RTOL
    )
    torch.testing.assert_close(s1, s2, atol=ATOL, rtol=RTOL)
    assert torch.isfinite(output).all() and torch.isfinite(reference).all()
    assert torch.isfinite(s1).all() and torch.isfinite(s2).all()
    untouched = set(range(B + 2)) - set(indices[indices >= 0].tolist())
    for s in untouched:
        assert torch.equal(s1[s], before[s]), "padding wrote a state slot"
    return (
        {
            "output_max_abs": float(
                (output.transpose(0, 1).float() - reference.float()).abs().max()
            ),
            "state_max_abs": float((s1 - s2).abs().max()),
        },
        s1,
        s2,
    )


def run():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    packed, generic = kernels()
    rows = []
    for B, H, D, dtype in (
        (1, 2, 16, torch.float32),
        (3, 2, 16, torch.float32),
        (2, 16, 128, torch.float32),
        (1, 16, 128, torch.bfloat16),
        (2, 16, 128, torch.bfloat16),
    ):
        for norm in (False, True):
            for extreme in (False, True):
                tensors = inputs(B, H, D, dtype=dtype)
                if B == 1 and not extreme:
                    tensors[-1][0] = 0
                if extreme:
                    tensors[1][:, ::2] = -100
                    tensors[1][:, 1::2] = 100
                    beta_rows = tensors[2].reshape(B, H)
                    beta_rows[:, ::2] = -80
                    beta_rows[:, 1::2] = 80
                row, s1, s2 = pair(packed, generic, tensors, norm=norm)
                row.update(batch=B, heads=H, dim=D, norm=norm, extreme_gate=extreme)
                # Successor output/state, changing input values at the same address.
                qkv, a, b, A, dt, _, idx = tensors
                qkv.mul_(0.7).add_(0.03)
                a.neg_()
                b.add_(0.17)
                qkv2 = qkv.clone()
                row2, _, _ = pair(
                    packed,
                    generic,
                    (qkv2, a, b, A, dt, s1, idx),
                    norm=norm,
                    reference_state=s2,
                )
                # Initial states from both routes already passed the state gate.
                torch.testing.assert_close(s1, s2, atol=ATOL, rtol=RTOL)
                rows.extend(
                    (row, {**row2, "successor": True, "batch": B, "heads": H, "dim": D})
                )
    assert not torch.cuda.is_initialized()
    return {
        "passed": True,
        "cases": len(rows),
        "rows": rows,
        "atol": ATOL,
        "rtol": RTOL,
        "backend": "Triton CPU interpreter, real source kernels",
        "cuda_initialized": False,
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    start = time.monotonic()
    result = run()
    result["seconds"] = time.monotonic() - start
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)
