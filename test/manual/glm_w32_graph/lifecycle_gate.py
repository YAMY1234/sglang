"""Slot ownership and graph cadence checks, independent of model weights."""

import argparse
import importlib
import json
import sys
from types import SimpleNamespace

import torch
from numeric_gate import kernel, reference, relf

sys.modules["sglang.srt.layers.attention.linear.kernels.glm_w32_project"] = kernel
policy_module = importlib.import_module("glm_gate_models.glm_duet.graph_state_policy")


def fixture(device="cpu"):
    states = torch.randn(2, 5, 1, 128, 128, device=device)
    siblings = []
    pool = SimpleNamespace(
        mamba_pool=SimpleNamespace(
            mamba_cache=SimpleNamespace(temporal=states),
            register_slot_state=siblings.append,
        ),
        mamba2_layer_index=lambda l: {0: 0, 2: 1}[l],
    )
    policy = policy_module.GraphGlmStatePolicy(
        pool,
        {l: torch.randn(1, 128, device=device) for l in (0, 2)},
        {"state_rank": 32, "state_every": 32, "state_sink": "explicit"},
        [0, 2],
        omega_heads=(0, 1),
    )
    assert siblings == [policy]
    return policy


def lifecycle():
    policy = fixture()
    policy.mark_prompt_end(None, [1, 2])
    policy.phase[1:3] = torch.tensor([3, 31])
    policy.warm[:, 1].fill_(1)
    policy.warm[:, 2].fill_(2)
    policy.copy_slots([1, 2], [2, 1])
    assert policy.phase.tolist() == [0, 31, 3, 0, 0]
    assert torch.equal(policy.warm[:, 1], torch.full_like(policy.warm[:, 1], 2))
    data = policy.get_cpu_slots([1, 2])
    policy.reset_slots([1, 2])
    assert not policy.active.any() and not policy.warm.any()
    policy.load_cpu_slots(data, [3, 4])
    assert policy.phase.tolist() == [0, 0, 0, 31, 3]
    assert policy.active.tolist() == [False, False, False, True, True]
    assert policy.pending.tolist() == [False, False, False, True, True]
    policy.mark_prompt_end(None, [3])
    assert policy.phase[3] == 0
    print(
        "GLMW32_LIFECYCLE "
        + json.dumps(
            {
                "passed": True,
                "cuda_initialized": torch.cuda.is_initialized(),
                "workspace_bytes": policy.workspace.bytes(),
            }
        ),
        flush=True,
    )


def graph():
    torch.manual_seed(207)
    policy = fixture("cuda")
    index = torch.tensor([1, 3, -1], device="cuda", dtype=torch.int32)

    def step():
        for l in policy.layer_ids:
            policy.before_layer(l, index)
            # Changing native-like state update; all slots here are fixture owned.
            policy.states[policy.layer_map[l]].mul_(0.99).add_(0.001)
            policy.after_layer(l, index)

    for _ in range(2):
        step()
    torch.cuda.synchronize()
    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured):
        step()
    # Reset after capture: capture is not a logical decode step.
    policy.reset_slots(torch.arange(5, device="cuda"))
    policy.mark_prompt_end(None, [1, 3])
    expected = policy.states.clone()
    warm = {}
    phases = {1: 0, 3: 0}
    pending = {1, 3}
    maximum = 0
    for iteration in range(65):
        if iteration == 17:
            # Dynamic admitted request, previous request absent; independent cadence.
            index.copy_(torch.tensor([1, 4, -1], device="cuda", dtype=torch.int32))
            policy.mark_prompt_end(None, [4])
            phases[4] = 0
            pending.add(4)
        selected = [1, 3] if iteration < 17 else [1, 4]
        for slot in selected:
            if slot in pending:
                for layer in range(2):
                    out, bases = reference.project_state_batched(
                        expected[layer, slot : slot + 1].transpose(-1, -2),
                        policy.vbar[layer],
                        32,
                        [None],
                        omega_heads=(0, 1),
                    )
                    expected[layer, slot] = out[0].transpose(-1, -2)
                    warm[layer, slot] = bases[0]
                pending.remove(slot)
        expected.mul_(0.99).add_(0.001)
        for slot in selected:
            phases[slot] = (phases[slot] + 1) % 32
            if phases[slot] == 0:
                for layer in range(2):
                    out, bases = reference.project_state_batched(
                        expected[layer, slot : slot + 1].transpose(-1, -2),
                        policy.vbar[layer],
                        32,
                        [warm[layer, slot]],
                        omega_heads=(0, 1),
                    )
                    expected[layer, slot] = out[0].transpose(-1, -2)
                    warm[layer, slot] = bases[0]
        captured.replay()
        torch.cuda.synchronize()
        error = relf(policy.states, expected)
        maximum = max(maximum, error)
        assert error <= 1e-5, (iteration, error)
        assert all(int(policy.phase[s]) == phases[s] for s in phases)
    print(
        "GLMW32_GRAPH "
        + json.dumps(
            {
                "passed": True,
                "steps": 65,
                "max_rel_f": maximum,
                "phase": policy.phase.cpu().tolist(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    lifecycle()
    if args.graph:
        graph()
