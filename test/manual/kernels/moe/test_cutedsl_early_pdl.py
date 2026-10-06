# SPDX-License-Identifier: Apache-2.0
"""Opt-in FlashInfer early-FC1 regression, requiring its experimental API.

The changed contract is scheduling: FC1 may trigger FC2 before publishing its
activations, while FC2 must still wait before consuming them. Capture the real
sort/FC1/async-memset/FC2 path, mutate inputs and routing between replays, and
compare FC1's valid bytes/scales exactly. Top-1 final outputs have no inter-expert
atomic ordering ambiguity. Top-10 keeps fused finalize and measures its own
A0/A1 noise; no deterministic reduction switch is introduced.

Manual until the external FlashInfer API is upstream; no production monkeypatch.
"""

import importlib
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from flashinfer import CuteDslMoEWrapper
from sglang.test.test_utils import CustomTestCase


def _fixture():
    path = Path(__file__).parents[3] / "registered/moe/test_cutedsl_moe.py"
    spec = importlib.util.spec_from_file_location("cutedsl_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ulp(a, b):
    x, y = a.view(torch.int16).to(torch.int32), b.view(torch.int16).to(torch.int32)
    x = torch.where(x < 0, -32768 - x, x)
    y = torch.where(y < 0, -32768 - y, y)
    return int((x - y).abs().max().item())


@torch.inference_mode()
def run_case(
    tokens, experts, top_k, tile=128, rounds=256, dump_dir=None, *, feature_enabled=True
):
    """Keep three banks alive and use one fixed tactic in every arm."""
    core = importlib.import_module("flashinfer.fused_moe.cute_dsl.fused_moe")
    fc1_module = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion"
    )

    fixture = _fixture()
    hidden, intermediate = 4096, 128  # Qwen3.5 TP8 expert projection dimensions.
    tensors = fixture._create_cutedsl_wrapper_tensors(
        tokens, hidden, intermediate, experts, top_k
    )
    tactic = (
        tile,
        ((tile, 128), (tile // 128, 1), False),
        ((tile, 128), (tile // 128, 1), False),
    )
    wrappers = [
        CuteDslMoEWrapper(
            num_experts=experts,
            top_k=top_k,
            hidden_size=hidden,
            intermediate_size=intermediate,
            use_cuda_graph=True,
            max_num_tokens=tokens,
            use_fused_finalize=True,
            enable_pdl=True,
            enable_fc1_early_pdl=flag,
        )
        for flag in (False, False, feature_enabled)
    ]
    real_fc1 = core.blockscaled_contiguous_gather_grouped_gemm_act_fusion
    captures = []

    def capture_fc1(*args, **kwargs):
        value = real_fc1(*args, **kwargs)
        captures.append((value, kwargs))
        return value

    graphs, outputs, intermediates = [], [], []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for wrapper in wrappers:
            for _ in range(3):
                fixture._run_wrapper(wrapper, tensors, tactic=tactic)
        torch.cuda.synchronize()
        for i, wrapper in enumerate(wrappers):
            graph = torch.cuda.CUDAGraph()
            if dump_dir is not None:
                try:
                    graph.enable_debug_mode()
                except BaseException as exc:
                    print("GRAPH_OBSERVER_ERROR", repr(exc), flush=True)
            with patch.object(
                core,
                "blockscaled_contiguous_gather_grouped_gemm_act_fusion",
                capture_fc1,
            ):
                with torch.cuda.graph(graph, stream=stream):
                    out = fixture._run_wrapper(wrapper, tensors, tactic=tactic)
            assert len(captures) == i + 1
            graphs.append(graph)
            outputs.append(out)
            intermediates.append(captures[-1])
            if dump_dir is not None:
                try:
                    graph.debug_dump(
                        str(
                            Path(dump_dir)
                            / f"m{tokens}-e{experts}-k{top_k}-t{tile}-bank{i}.dot"
                        )
                    )
                except BaseException as exc:
                    print("GRAPH_OBSERVER_ERROR", repr(exc), flush=True)

    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    row = dict(
        tokens=tokens,
        experts=experts,
        top_k=top_k,
        tile=tile,
        hidden=hidden,
        intermediate=intermediate,
        rounds=rounds,
        fc1_byte_mismatch=0,
        scale_byte_mismatch=0,
        nonfinite=0,
        top1_max_ulp=0,
        off_self_max_ulp=0,
        on_off_max_ulp=0,
        off_self_square_sum=0.0,
        on_off_square_sum=0.0,
        output_elements=0,
        round_mean_deltas=[],
        off_off_round_mean_deltas=[],
        wrapper_flags=[False, False, feature_enabled],
        use_fused_finalize=True,
        distinct_graphs=len({id(graph) for graph in graphs}) == 3,
        distinct_wrappers=len({id(wrapper) for wrapper in wrappers}) == 3,
        distinct_outputs=len({out.data_ptr() for out in outputs}) == 3,
        output_dtype=str(outputs[0].dtype),
        off_off_physical_row_differences=0,
        on_off_physical_row_differences=0,
        off_off_fc1_byte_mismatch=0,
        on_off_fc1_byte_mismatch=0,
        off_off_scale_byte_mismatch=0,
        on_off_scale_byte_mismatch=0,
    )
    initial_x = tensors["x"].clone()
    initial_ids = tensors["token_selected_experts"].clone()
    for step in range(rounds):
        tensors["x"].copy_(torch.bitwise_xor(initial_x, 0x88 if step % 2 else 0))
        tensors["token_selected_experts"].copy_((initial_ids + step) % experts)
        # The previous replay has completed before changing the next round's inputs.
        for index in (0, 1, 2) if step % 2 == 0 else (2, 1, 0):
            graphs[index].replay()
        torch.cuda.synchronize()
        canonical = []
        mappings = []
        for value, meta in intermediates:
            valid_tiles = int(meta["num_non_exiting_tiles"].item())
            rows = torch.arange(valid_tiles * tile, device="cuda")
            rows = rows[rows < meta["tile_idx_to_mn_limit"][rows // tile]]
            route_ids = meta["token_id_mapping"][rows].to(torch.int64)
            order = torch.argsort(route_ids)
            sorted_ids = route_ids[order]
            # Compare the same logical (token, top-k choice), not physical rows.
            # Expert sorting can permute rows without changing the MoE output.
            assert torch.equal(sorted_ids, torch.arange(tokens * top_k, device="cuda"))
            r = rows[order]
            routed_experts = meta["tile_idx_to_expert_idx"][r // tile]
            assert torch.equal(
                routed_experts, tensors["token_selected_experts"].flatten()
            )
            mappings.append(r)
            # FC1 allocates a contiguous buffer with a six-dimensional shape,
            # but writes the packed MMA layout via its raw pointer. Reconstruct
            # the documented strided logical view before selecting route rows.
            scale_view = fixture.convert_sf_to_mma_layout(
                value[1], m=value[0].shape[0], k=intermediate
            )
            canonical.append((value[0][r], scale_view[r % 32, (r // 32) % 4, r // 128]))
        for pair, bank in (("off_off", 1), ("on_off", 2)):
            row[pair + "_physical_row_differences"] += int(
                (mappings[0] != mappings[bank]).sum().item()
            )
            byte_diff = int((canonical[0][0] != canonical[bank][0]).sum().item())
            scale_diff = int((canonical[0][1] != canonical[bank][1]).sum().item())
            row[pair + "_fc1_byte_mismatch"] += byte_diff
            row[pair + "_scale_byte_mismatch"] += scale_diff
            row["fc1_byte_mismatch"] += byte_diff
            row["scale_byte_mismatch"] += scale_diff
        a0, a1, b = [out.float() for out in outputs]
        row["nonfinite"] += sum(
            int((~torch.isfinite(out)).sum().item()) for out in (a0, a1, b)
        )
        row["off_self_square_sum"] += float((a0 - a1).square().sum().item())
        row["on_off_square_sum"] += float((b - a0).square().sum().item())
        row["output_elements"] += a0.numel()
        row["round_mean_deltas"].append(float((b - (a0 + a1) / 2).mean().item()))
        row["off_off_round_mean_deltas"].append(float((a1 - a0).mean().item()))
        row["off_self_max_ulp"] = max(
            row["off_self_max_ulp"], _ulp(outputs[0], outputs[1])
        )
        row["on_off_max_ulp"] = max(row["on_off_max_ulp"], _ulp(outputs[0], outputs[2]))
        if top_k == 1:
            row["top1_max_ulp"] = max(
                row["top1_max_ulp"], row["on_off_max_ulp"], row["off_self_max_ulp"]
            )
    import math
    import statistics

    off_rms = math.sqrt(row["off_self_square_sum"] / row["output_elements"])
    on_rms = math.sqrt(row["on_off_square_sum"] / row["output_elements"])
    row.update(
        off_self_rms=off_rms,
        on_off_rms=on_rms,
        noise_ratio=on_rms / off_rms if off_rms else (0.0 if on_rms == 0 else None),
    )
    deltas = row["round_mean_deltas"]
    row["signed_mean_delta"] = statistics.mean(deltas)
    row["signed_mean_2se"] = 2 * statistics.stdev(deltas) / math.sqrt(rounds)
    null_deltas = row["off_off_round_mean_deltas"]
    row["off_off_signed_mean"] = statistics.mean(null_deltas)
    row["off_off_signed_mean_2se"] = (
        2 * statistics.stdev(null_deltas) / math.sqrt(rounds)
    )
    row["off_off_envelope"] = (
        abs(row["off_off_signed_mean"]) + row["off_off_signed_mean_2se"]
    )
    row["hard_checks_passed"] = (
        row["fc1_byte_mismatch"] == row["scale_byte_mismatch"] == row["nonfinite"] == 0
        and row["distinct_graphs"]
        and row["distinct_wrappers"]
        and row["distinct_outputs"]
    ) and (
        row["top1_max_ulp"] <= 1
        if top_k == 1
        else row["noise_ratio"] is not None and row["noise_ratio"] <= 3
    )
    row["legacy_passed"] = row["hard_checks_passed"] and (
        top_k == 1 or abs(row["signed_mean_delta"]) <= row["signed_mean_2se"]
    )
    # Order279: calibrate the signed-mean criterion to the stock pair's null.
    # Retain both raw series and the previous predicate; never erase a failure.
    row["passed"] = row["hard_checks_passed"] and (
        abs(row["signed_mean_delta"]) <= row["off_off_envelope"]
    )
    # Passive identity only; inability to inspect never changes execution.
    try:
        row["compiled_cache_keys"] = [repr(k) for k in fc1_module._gather_kernel_cache]
    except BaseException as exc:
        row["cache_observer_error"] = repr(exc)
    return row


class TestCuteDslEarlyPdl(CustomTestCase):
    def test_capture_dependency_and_numerical_parity(self):
        for tokens, experts, top_k, tile in (
            (1, 16, 1, 128),
            (7, 512, 10, 128),
            (56, 16, 10, 128),
            (8, 16, 10, 256),
        ):
            with self.subTest(tokens=tokens, experts=experts, top_k=top_k, tile=tile):
                result = run_case(tokens, experts, top_k, tile)
                self.assertTrue(result["passed"], json.dumps(result))


if __name__ == "__main__":
    unittest.main()
