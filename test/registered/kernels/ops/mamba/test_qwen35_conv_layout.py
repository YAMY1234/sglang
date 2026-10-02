"""Dense and deduplicated verify windows must have identical conv/accept outputs."""

import argparse
import json
from pathlib import Path

import torch
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    fused_conv_window_scatter_with_mask,
)


def run_case(batch, draft, rounds=1024, dim=12288, strided_state=False):
    torch.manual_seed(234 + batch + draft)
    slots, layers, win = batch + 2, 2, 3
    kw = dict(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(layers, batch, draft, dim, **kw).transpose(2, 3)
    weights = torch.randn(layers, dim, win + 1, **kw)
    bias = torch.randn(layers, dim, **kw)
    initial = torch.randn(layers, slots, dim, win, **kw)
    indices = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
    intermediate_indices = torch.arange(batch, dtype=torch.int32, device="cuda")
    steps = torch.zeros(batch, dtype=torch.int32, device="cuda")
    paths = []
    for dense in (False, True):
        if dense:
            physical = torch.empty(layers, batch + 1, draft, dim, win, **kw)
            view = physical
        else:
            physical = torch.empty(layers, batch + 1, dim, draft + win - 1, **kw)
            st = physical.stride()
            view = physical.as_strided(
                (layers, batch + 1, draft, dim, win), (st[0], st[1], 1, st[2], 1)
            )
        if strided_state:
            state = torch.empty(layers, slots, win, dim, **kw).transpose(2, 3)
        else:
            state = torch.empty_like(initial)
        state.copy_(initial)
        # Production scatter targets a contiguous checkpoint. The conv input can be strided.
        accepted = torch.empty_like(initial)
        result = [None] * layers

        def step(state=state, view=view, accepted=accepted, result=result):
            for layer in range(layers):
                result[layer] = causal_conv1d_update(
                    x[layer],
                    state[layer],
                    weights[layer],
                    bias[layer],
                    activation="silu",
                    conv_state_indices=indices,
                    intermediate_conv_window=view[layer],
                    intermediate_state_indices=intermediate_indices,
                    pad_slot_id=-1,
                )
            accepted.copy_(state)
            fused_conv_window_scatter_with_mask(accepted, view, indices, steps)
            state.copy_(accepted)

        step()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        state.copy_(initial)
        paths.append(
            dict(
                state=state,
                accepted=accepted,
                physical=physical,
                view=view,
                result=result,
                graph=graph,
            )
        )
    comparisons = 0
    for iteration in range(rounds):
        x.normal_()
        indices.copy_(
            torch.arange(1, batch + 1, device="cuda", dtype=torch.int32).roll(
                iteration % batch
            )
        )
        steps.copy_(
            (torch.arange(batch, device="cuda", dtype=torch.int32) + iteration) % draft
        )
        if iteration % 19 == 0:
            indices[-1] = -1
            steps[-1] = -1
        for p in paths:
            if iteration % 13 == 0:
                p["physical"].fill_(float("nan"))
            p["graph"].replay()
        valid_rows = indices >= 0
        for layer in range(layers):
            assert torch.equal(
                paths[0]["result"][layer][valid_rows],
                paths[1]["result"][layer][valid_rows],
            ), (batch, draft, iteration, "output")
        assert torch.equal(
            paths[0]["view"][:, :batch][:, valid_rows],
            paths[1]["view"][:, :batch][:, valid_rows],
        ), (batch, draft, iteration, "all accept windows")
        assert torch.equal(paths[0]["state"], paths[1]["state"]), (
            batch,
            draft,
            iteration,
            "accepted state",
        )
        comparisons += 1
        if iteration % 64 == 63:
            # Reuse a freed slot with a new prefill state, then keep decoding.
            fresh = torch.randn(layers, dim, win, **kw)
            for p in paths:
                p["state"][:, 1].copy_(fresh)
    empty_indices = indices[:0]
    before = paths[0]["accepted"].clone()
    fused_conv_window_scatter_with_mask(
        paths[0]["accepted"], paths[0]["view"], empty_indices, steps[:0]
    )
    assert torch.equal(before, paths[0]["accepted"])
    return dict(
        batch=batch,
        draft=draft,
        dim=dim,
        rounds=rounds,
        comparisons=comparisons,
        strided_state=strided_state,
        bitwise=True,
        padding=True,
        empty_scatter=True,
        accept_boundary=True,
        poisoned_graph_reuse=True,
        slot_reuse=True,
        physical_bytes=[p["physical"].untyped_storage().nbytes() for p in paths],
    )


def test_qwen35_conv_layout():
    assert run_case(3, 7, rounds=64, dim=512)["bitwise"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for batch, draft, strided in (
        (1, 1, False),
        (1, 2, True),
        (8, 7, False),
        (16, 7, True),
        (16, 8, False),
    ):
        rows.append(run_case(batch, draft, strided_state=strided))
        args.output.write_text(
            json.dumps({"pass": True, "cases": rows}, indent=2) + "\n"
        )
        print(json.dumps(rows[-1]), flush=True)
