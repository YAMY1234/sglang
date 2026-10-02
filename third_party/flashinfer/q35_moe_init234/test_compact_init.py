"""Tail tiles, empty/sentinel routing, and poisoned graph reuse on pinned SM100."""

import argparse
import json
from pathlib import Path

import flashinfer
import torch
from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
from flashinfer.fused_moe.cute_dsl.compact_init import (
    fill_tile_metadata,
    sparse_output_zero,
)
from flashinfer.fused_moe.cute_dsl.fused_moe import _moe_core_impl
from flashinfer.fused_moe.cute_dsl.moe_utils import moe_sort


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def routed(ids, first=0, local=32):
    return ((ids >= first) & (ids < first + local)).any(1)


def make_ids(n, phase, first=0):
    ids = torch.full((n, 10), 512, dtype=torch.int32, device="cuda")
    if phase == 0 and n:
        ids[: max(1, n // 3)] = first + torch.arange(10, device="cuda")
    elif phase == 1 and n:
        ids[::2, 0] = first
    elif phase == 3 and n:
        ids[:] = first + torch.arange(10, device="cuda")
    return ids


def init_gate():
    rows = []
    for n in (0, 1, 127, 128, 129, 896, 1792):
        for first in (0, 480):
            ids = make_ids(n, 0, first)
            output = torch.empty((n, 7168), device="cuda", dtype=torch.bfloat16)
            a = torch.empty(n + 1, device="cuda", dtype=torch.int32)
            b = torch.empty(n + 3, device="cuda", dtype=torch.int32)
            graph = capture(
                lambda: (
                    fill_tile_metadata(a, b),
                    sparse_output_zero(output, ids, first, 32),
                )
            )
            for phase in (0, 1, 2, 3, 0, 2):
                ids.copy_(make_ids(n, phase, first))
                a.fill_(1234567)
                b.fill_(-98765)
                output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                mask = routed(ids, first)
                assert torch.count_nonzero(a) == torch.count_nonzero(b) == 0
                assert torch.count_nonzero(output[mask]) == 0
                assert torch.isnan(output[~mask]).all()
            rows.append(
                dict(
                    tokens=n, local_offset=first, graph_reuse=True, exact_write_set=True
                )
            )
    return rows


def sort_gate():
    rows = []
    keys = (
        "out_tile_idx_to_expert_idx",
        "out_tile_idx_to_mn_limit",
        "out_expanded_idx_to_permuted_idx",
        "out_permuted_idx_to_expanded_idx",
        "out_total_num_padded_tokens",
        "out_num_non_exiting_tiles",
    )
    for n in (0, 1, 127, 129, 896, 1792):
        ids = make_ids(n, 0)
        weights = torch.ones((n, 10), dtype=torch.float32, device="cuda")
        args = dict(
            token_selected_experts=ids,
            token_final_scales=weights,
            num_experts=512,
            top_k=10,
            num_local_experts=32,
            tile_tokens_dim=128,
        )
        template = moe_sort(**args)
        outputs = [[torch.empty_like(t) for t in template] for _ in range(2)]
        graphs = [
            capture(
                lambda j=j: moe_sort(
                    **args, **dict(zip(keys, outputs[j])), fuse_tile_init=bool(j)
                )
            )
            for j in range(2)
        ]
        counts = []
        for phase in (0, 1, 2, 3, 0):
            ids.copy_(make_ids(n, phase))
            for j in range(2):
                for t in outputs[j]:
                    t.fill_(1234567 + j)
                graphs[j].replay()
            torch.cuda.synchronize()
            aa, bb = outputs
            for k in (0, 1, 2, 4, 5):
                assert torch.equal(aa[k], bb[k]), (n, phase, k)
            live = aa[2].flatten()
            live = live[live >= 0].long()
            assert torch.equal(aa[3][live], bb[3][live])
            assert int(aa[5]) >= 0
            counts.append(int(aa[5]))
        rows.append(
            dict(tokens=n, active_tile_counts=counts, full_metadata_bitwise=True)
        )
    return rows


def weights():
    e, h, i = 32, 7168, 512
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    w1 = torch.randn((e, 2 * i, h), device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn((e, h, i), device="cuda", dtype=torch.bfloat16) * 0.02
    q1, s1 = flashinfer.fp4_quantize(
        w1.flatten(0, 1), scale, sf_vec_size=16, is_sf_swizzled_layout=True
    )
    q2, s2 = flashinfer.fp4_quantize(
        w2.flatten(0, 1), scale, sf_vec_size=16, is_sf_swizzled_layout=True
    )
    return dict(
        w1_weight=q1.view(e, 2 * i, h // 2),
        w1_weight_sf=convert_sf_to_mma_layout(s1, 2 * i, h, e),
        w2_weight=q2.view(e, h, i // 2),
        w2_weight_sf=convert_sf_to_mma_layout(s2, h, i, e),
        w1_alpha=torch.ones(e, device="cuda"),
        w2_alpha=torch.ones(e, device="cuda"),
        fc2_input_scale=scale,
    )


def moe_gate():
    w = weights()
    rows = []
    for n in (1, 129, 896, 1792):
        ids = make_ids(n, 0)
        scales = torch.ones((n, 10), device="cuda", dtype=torch.float32) / 10
        x = torch.randn((n, 7168), device="cuda", dtype=torch.bfloat16) * 0.1
        q, sf = flashinfer.fp4_quantize(
            x, w["fc2_input_scale"], sf_vec_size=16, is_sf_swizzled_layout=False
        )
        args = dict(
            x=q,
            x_sf=sf,
            token_selected_experts=ids,
            token_final_scales=scales,
            num_experts=512,
            top_k=10,
            num_local_experts=32,
            local_expert_offset=0,
            enable_pdl=True,
            use_fused_finalize=True,
            use_async_memset=True,
            **w,
        )
        outs = [torch.empty_like(x) for _ in range(3)]
        graphs = [
            capture(
                lambda j=j: _moe_core_impl(
                    **args, moe_output=outs[j], compact_init=j > 0, sparse_output=j == 1
                )
            )
            for j in range(3)
        ]
        comparisons = []
        for phase in (0, 1, 2, 3, 0):
            ids.copy_(make_ids(n, phase))
            valid = routed(ids)
            references = []
            for _ in range(3):
                outs[0].fill_(float("nan"))
                graphs[0].replay()
                torch.cuda.synchronize()
                references.append(outs[0].float().clone())
            stack = torch.stack(references)
            low, high = stack.amin(0), stack.amax(0)
            magnitude = stack.abs().amax(0).to(torch.bfloat16)
            ulp = (
                torch.nextafter(
                    magnitude, torch.full_like(magnitude, float("inf"))
                ).float()
                - magnitude.float()
            )
            for j in (1, 2):
                outs[j].fill_(float("nan"))
                graphs[j].replay()
                torch.cuda.synchronize()
                actual = outs[j].float()
                mask = (
                    valid if j == 1 else torch.ones(n, dtype=torch.bool, device="cuda")
                )
                assert torch.isfinite(actual[mask]).all(), (n, phase, j)
                # Atomic BF16 reductions may reorder: retain the stock repeat envelope plus one ULP.
                bad = (actual < low - ulp) | (actual > high + ulp)
                assert not bad[mask].any(), (n, phase, j, int(bad[mask].sum()))
                if j == 2:
                    assert torch.count_nonzero(outs[j][~valid]) == 0
                comparisons.append(
                    dict(
                        phase=phase,
                        sparse=j == 1,
                        valid_rows=int(valid.sum()),
                        max_abs=float((actual[mask] - references[0][mask]).abs().max())
                        if mask.any()
                        else 0,
                        stock_repeat_max=float((high - low).max()),
                    )
                )
        rows.append(dict(tokens=n, graph_poison_reuse=True, comparisons=comparisons))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert torch.cuda.get_device_capability() == (10, 0)
    torch.manual_seed(234)
    result = {
        "pass": False,
        "scope": "Pinned actual FlashInfer kernels, synthetic same-shape weights; not GSM8K.",
    }
    try:
        for name, fn in (
            ("initialization", init_gate),
            ("routing", sort_gate),
            ("full_moe", moe_gate),
        ):
            result[name] = fn()
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(name, "passed", flush=True)
        result["pass"] = True
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
