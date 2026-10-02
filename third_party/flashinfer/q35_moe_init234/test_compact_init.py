"""Tail tiles, empty/sentinel routing, and poisoned graph reuse on pinned SM100."""

import argparse
import hashlib
import json
from pathlib import Path

import flashinfer
import numpy as np
import torch
from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
from flashinfer.fused_moe.cute_dsl.compact_init import (
    fill_tile_metadata,
    sparse_output_zero,
)
from flashinfer.fused_moe.cute_dsl.fused_moe import CuteDslMoEWrapper, _moe_core_impl
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


def time_graphs(graphs):
    samples = [[] for _ in graphs]
    for graph in graphs:
        for _ in range(10):
            graph.replay()
    torch.cuda.synchronize()
    for repeat in range(3):
        order = range(len(graphs)) if repeat % 2 == 0 else reversed(range(len(graphs)))
        for index in order:
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(128):
                graphs[index].replay()
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end) / 128)
    return {
        "stock_sparse_dense_ms": samples,
        "method": "3 alternating-order repeats, 128 graph replays per CUDA-event interval",
        "scope": "Synthetic module timing; not a service round or a promotion gate",
    }


def make_ids(n, phase, first=0):
    if phase == 4:
        assert n == 896 and first == 0
        source = json.loads(
            Path(__file__).with_name("qualification-routing.json").read_text()
        )
        return torch.tensor(source["routing"], dtype=torch.int32, device="cuda")
    ids = torch.full((max(n, 1), 10), 512, dtype=torch.int32, device="cuda")[:n]
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
            output = torch.empty((n, 4096), device="cuda", dtype=torch.bfloat16)
            a = torch.empty(n + 1, device="cuda", dtype=torch.int32)
            b = torch.empty(n + 3, device="cuda", dtype=torch.int32)
            graph = capture(
                lambda a=a, b=b, output=output, ids=ids, first=first: (
                    fill_tile_metadata(a, b),
                    sparse_output_zero(output, ids, first, 32),
                )
            )
            for phase in (
                (0, 1, 2, 3, 4, 0, 2) if n == 896 and first == 0 else (0, 1, 2, 3, 0, 2)
            ):
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
                {
                    "tokens": n,
                    "local_offset": first,
                    "graph_reuse": True,
                    "exact_write_set": True,
                }
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
        weights = torch.ones((max(n, 1), 10), dtype=torch.float32, device="cuda")[:n]
        args = {
            "token_selected_experts": ids,
            "token_final_scales": weights,
            "num_experts": 512,
            "top_k": 10,
            "num_local_experts": 32,
            "tile_tokens_dim": 128,
        }
        # The pinned stock API rejects empty TopK pointers; preserve that contract.
        if n == 0:
            errors = []
            for fused in (False, True):
                try:
                    moe_sort(**args, fuse_tile_init=fused)
                    torch.cuda.synchronize()
                except RuntimeError as exc:
                    assert (
                        "Routing kernel requires at least one input parameter"
                        in str(exc)
                    )
                    errors.append(type(exc).__name__)
                else:
                    raise AssertionError(
                        "Pinned stock empty-token contract unexpectedly changed"
                    )
            assert len(errors) == 2 and errors[0] == errors[1]
            rows.append(
                {
                    "tokens": 0,
                    "stock_and_fork": "same explicit input rejection",
                    "empty_local_rank": "covered by all-sentinel nonempty buffers",
                }
            )
            continue
        template = moe_sort(**args)
        outputs = [[torch.empty_like(t) for t in template] for _ in range(2)]
        graphs = [
            capture(
                lambda j=j, args=args, outputs=outputs: moe_sort(
                    **args, **dict(zip(keys, outputs[j])), fuse_tile_init=bool(j)
                )
            )
            for j in range(2)
        ]
        counts, stock_order_changes = [], []
        for phase in (0, 1, 2, 3, 4, 0) if n == 896 else (0, 1, 2, 3, 0):
            ids.copy_(make_ids(n, phase))
            for j in range(2):
                for t in outputs[j]:
                    t.fill_(1234567 + j)
                graphs[j].replay()
            torch.cuda.synchronize()
            aa, bb = outputs
            for k in (0, 1, 4, 5):
                assert torch.equal(aa[k], bb[k]), (n, phase, k)
            # Stock atomic offsets allow different within-expert permutations.
            expected = (ids.flatten() >= 0) & (ids.flatten() < 32)
            expanded = torch.arange(ids.numel(), device="cuda")[expected]
            for result in outputs:
                mapping = result[2].flatten()
                assert torch.equal(mapping >= 0, expected)
                permuted = mapping[expected].long()
                assert torch.unique(permuted).numel() == expanded.numel()
                assert ((permuted >= 0) & (permuted < int(result[4]))).all()
                assert torch.equal(result[3][permuted].long(), expanded)
                tiles = permuted // 128
                assert (tiles < int(result[5])).all()
                assert torch.equal(result[0][tiles], ids.flatten()[expected])
                assert (permuted < result[1][tiles]).all()
            first_order = aa[2].clone()
            graphs[0].replay()
            torch.cuda.synchronize()
            stock_order_changes.append(int((first_order != aa[2]).sum()))
            assert int(aa[5]) >= 0
            counts.append(int(aa[5]))
        rows.append(
            {
                "tokens": n,
                "active_tile_counts": counts,
                "tile_metadata_bitwise": True,
                "routing_bijection_and_expert_membership": True,
                "stock_repeat_permutation_changes": stock_order_changes,
                "metadata_elements": [template[k].numel() for k in (0, 1)],
                "metadata_bytes_per_replay": sum(
                    template[k].numel() * template[k].element_size() for k in (0, 1)
                ),
                "initialization_launches_stock_compact": [2, 1],
                "counts_scope": "Source-defined operations per module replay, not service runtime hook counts",
            }
        )
    return rows


def weights():
    e, h, i = 32, 4096, 1024
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    w1 = torch.randn((e, 2 * i, h), device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn((e, h, i), device="cuda", dtype=torch.bfloat16) * 0.02
    q1, s1 = flashinfer.fp4_quantize(
        w1.flatten(0, 1), scale, sf_vec_size=16, is_sf_swizzled_layout=True
    )
    q2, s2 = flashinfer.fp4_quantize(
        w2.flatten(0, 1), scale, sf_vec_size=16, is_sf_swizzled_layout=True
    )
    return {
        "w1_weight": q1.view(e, 2 * i, h // 2),
        "w1_weight_sf": convert_sf_to_mma_layout(s1, 2 * i, h, e),
        "w2_weight": q2.view(e, h, i // 2),
        "w2_weight_sf": convert_sf_to_mma_layout(s2, h, i, e),
        "w1_alpha": torch.ones(e, device="cuda"),
        "w2_alpha": torch.ones(e, device="cuda"),
        "fc2_input_scale": scale,
    }


def reduction_reference(graph, output, scales, ids):
    original = scales.clone()
    terms = []
    for slot in range(scales.shape[1]):
        scales.zero_()
        scales[:, slot].copy_(original[:, slot])
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        term = output.clone()
        assert torch.isfinite(term).all()
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(term, output), ("isolated route is not stable", slot)
        terms.append(term.float())
    scales.copy_(original)
    terms = torch.stack(terms).double()
    center = terms.sum(0).float()
    absolute_sum = terms.abs().sum(0).float()
    count = ((ids >= 0) & (ids < 32)).sum(1)
    n = (count - 1).clamp_min(0).float()[:, None]
    gamma = n / 256 / (1 - n / 256)
    bound = (gamma + torch.finfo(torch.float32).eps) * absolute_sum
    bound += count[:, None] * (2.0**-133)
    return center, bound, count


def check_reduction(actual, reference, mask, label):
    center, bound, count = reference
    assert torch.isfinite(actual[mask]).all(), (label, "nonfinite")
    difference = (actual.float() - center).abs()
    bad = difference > bound
    assert not bad[mask].any(), (label, "BF16 summation bound", int(bad[mask].sum()))
    exact = mask & (count <= 1)
    assert torch.equal(actual[exact], center[exact].to(actual.dtype)), (
        label,
        "zero/single-route bitwise mismatch",
    )
    return {
        "bad_elements": int(bad[mask].sum()),
        "max_abs": float(difference[mask].max()) if mask.any() else 0,
        "max_bound": float(bound[mask].max()) if mask.any() else 0,
        "zero_or_single_route_rows_bitwise": int(exact.sum()),
        "isolated_terms_repeat_bitwise": True,
    }


def moe_gate():
    w = weights()
    rows = []
    for n in (1, 129, 896, 1792):
        ids = make_ids(n, 0)
        scales = (
            torch.from_numpy(
                np.random.default_rng(23174).standard_normal((n, 10), dtype=np.float32)
                * 0.1
            )
            .cuda()
            .softmax(-1)
        )
        if n == 896:
            source = json.loads(
                Path(__file__).with_name("qualification-routing.json").read_text()
            )
            assert (
                hashlib.sha256(scales.cpu().numpy().tobytes()).hexdigest()
                == source["scales_sha256"]
            )
        x = torch.randn((n, 4096), device="cuda", dtype=torch.bfloat16) * 0.1
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
        resources = [
            {
                "aux_stream": torch.cuda.Stream(),
                "main_event": torch.cuda.Event(),
                "memset_event": torch.cuda.Event(),
            }
            for _ in outs
        ]
        graphs = [
            capture(
                lambda j=j, args=args, outs=outs, resources=resources: _moe_core_impl(
                    **args,
                    **resources[j],
                    moe_output=outs[j],
                    compact_init=j > 0,
                    sparse_output=j == 1,
                )
            )
            for j in range(3)
        ]
        comparisons = []
        for phase in (0, 1, 2, 3, 4, 0) if n == 896 else (0, 1, 2, 3, 0):
            ids.copy_(make_ids(n, phase))
            valid = routed(ids)
            reference = reduction_reference(graphs[0], outs[0], scales, ids)
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
            outs[0].fill_(float("nan"))
            graphs[0].replay()
            torch.cuda.synchronize()
            holdout = outs[0].float()
            stock_bad = (holdout < low - ulp) | (holdout > high + ulp)
            stock_bad_count = int(stock_bad.sum())
            stock_bound = check_reduction(
                holdout, reference, torch.ones_like(valid), "stock"
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
                # Retain the original envelope as a diagnostic; gate on the derived sum bound.
                bad = (actual < low - ulp) | (actual > high + ulp)
                candidate_bound = check_reduction(
                    outs[j], reference, mask, (n, phase, j)
                )
                if j == 2:
                    assert torch.count_nonzero(outs[j][~valid]) == 0
                comparisons.append(
                    {
                        "phase": phase,
                        "sparse": j == 1,
                        "valid_rows": int(valid.sum()),
                        "max_abs": float(
                            (actual[mask] - references[0][mask]).abs().max()
                        )
                        if mask.any()
                        else 0,
                        "stock_repeat_max": float((high - low).max()),
                        "stock_holdout_bad_elements": stock_bad_count,
                        "candidate_old_envelope_bad_elements": int(bad[mask].sum()),
                        "stock_summation_bound": stock_bound,
                        "candidate_summation_bound": candidate_bound,
                    }
                )
        ids.copy_(make_ids(n, 4 if n == 896 else 0))
        timing = time_graphs(graphs)
        timing["async_memset_resources"] = (
            "Persistent aux stream/main and memset events, matching production overlap"
        )
        rows.append(
            {
                "tokens": n,
                "graph_poison_reuse": True,
                "comparisons": comparisons,
                "timing": timing,
                "timing_routing": "histogram reconstruction"
                if n == 896
                else "constructed one-third local rows",
                "clear_bytes_stock_sparse": [
                    n * 4096 * 2,
                    int(routed(ids).sum()) * 4096 * 2,
                ],
            }
        )
    return rows


def wrapper_gate():
    import flashinfer.fused_moe.cute_dsl.compact_init as helpers

    n = 129
    w = weights()
    ids = make_ids(n, 0)
    x = torch.randn((n, 4096), device="cuda", dtype=torch.bfloat16) * 0.1
    q, sf = flashinfer.fp4_quantize(
        x, w["fc2_input_scale"], sf_vec_size=16, is_sf_swizzled_layout=False
    )
    scales = torch.ones_like(ids, dtype=torch.float32) / 10
    wrapper = CuteDslMoEWrapper(
        num_experts=512,
        top_k=10,
        hidden_size=4096,
        intermediate_size=1024,
        num_local_experts=32,
        use_cuda_graph=True,
        use_fused_finalize=True,
    )
    counts = {"fill": 0, "sparse": 0}
    original_fill, original_sparse = (
        helpers.fill_tile_metadata,
        helpers.sparse_output_zero,
    )

    def fill(*args):
        counts["fill"] += 1
        return original_fill(*args)

    def sparse(*args):
        counts["sparse"] += 1
        return original_sparse(*args)

    helpers.fill_tile_metadata, helpers.sparse_output_zero = fill, sparse
    outputs, graphs, receipts, numeric_receipts = [None] * 3, [], [], []
    try:
        for j in range(3):
            before = dict(counts)

            def forward(j=j):
                outputs[j] = wrapper.run(
                    x=q,
                    x_sf=sf,
                    token_selected_experts=ids,
                    token_final_scales=scales,
                    compact_init=j > 0,
                    sparse_output=j == 1,
                    **w,
                )

            graphs.append(capture(forward))
            delta = {key: counts[key] - before[key] for key in counts}
            assert (delta["fill"] > 0) == (j > 0), delta
            assert (delta["sparse"] > 0) == (j == 1), delta
            receipts.append(
                {"compact": j > 0, "sparse": j == 1, "warm_capture_calls": delta}
            )
        for phase in (0, 1, 2, 3, 0):
            ids.copy_(make_ids(n, phase))
            reference = reduction_reference(graphs[0], outputs[0], scales, ids)
            refs = []
            for _ in range(3):
                outputs[0].fill_(float("nan"))
                graphs[0].replay()
                torch.cuda.synchronize()
                refs.append(outputs[0].float().clone())
            stack = torch.stack(refs)
            low, high = stack.amin(0), stack.amax(0)
            magnitude = stack.abs().amax(0).to(torch.bfloat16)
            ulp = (
                torch.nextafter(
                    magnitude, torch.full_like(magnitude, float("inf"))
                ).float()
                - magnitude.float()
            )
            outputs[0].fill_(float("nan"))
            graphs[0].replay()
            torch.cuda.synchronize()
            holdout = outputs[0].float()
            stock_bad_count = int(
                ((holdout < low - ulp) | (holdout > high + ulp)).sum()
            )
            stock_bound = check_reduction(
                outputs[0],
                reference,
                torch.ones(n, dtype=torch.bool, device="cuda"),
                "wrapper-stock",
            )
            for j in (1, 2):
                outputs[j].fill_(float("nan"))
                graphs[j].replay()
                torch.cuda.synchronize()
                mask = (
                    routed(ids)
                    if j == 1
                    else torch.ones(n, device="cuda", dtype=torch.bool)
                )
                actual = outputs[j].float()
                assert torch.isfinite(actual[mask]).all()
                bad_count = int(
                    ((actual < low - ulp) | (actual > high + ulp))[mask].sum()
                )
                candidate_bound = check_reduction(
                    outputs[j], reference, mask, ("wrapper", phase, j)
                )
                numeric_receipts.append(
                    {
                        "phase": phase,
                        "sparse": j == 1,
                        "old_envelope_bad_candidate_stock": [
                            bad_count,
                            stock_bad_count,
                        ],
                        "stock_summation_bound": stock_bound,
                        "candidate_summation_bound": candidate_bound,
                    }
                )
                if j == 2:
                    assert torch.count_nonzero(outputs[j][~routed(ids)]) == 0
    finally:
        helpers.fill_tile_metadata, helpers.sparse_output_zero = (
            original_fill,
            original_sparse,
        )
    return {
        "tokens": n,
        "production_wrapper": True,
        "capture_receipts": receipts,
        "numeric_receipts": numeric_receipts,
        "timing": time_graphs(graphs),
        "note": "Warm/capture Python calls prove flag propagation; not runtime replay counts.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    capability = torch.cuda.get_device_capability()
    print("device", torch.cuda.get_device_name(), capability, flush=True)
    assert capability[0] == 10, capability
    torch.manual_seed(234)
    result = {
        "pass": False,
        "device": torch.cuda.get_device_name(),
        "capability": capability,
        "numeric_gate": "Isolated BF16 route terms repeat bitwise; stock and candidate within gamma_(k-1)*sumabs plus reference conversion/subnormal bound; k<=1 bitwise. Old envelope diagnostic only.",
        "scope": "Pinned actual FlashInfer kernels, synthetic same-shape weights; not GSM8K.",
    }
    try:
        for name, fn in (
            ("initialization", init_gate),
            ("routing", sort_gate),
            ("full_moe", moe_gate),
            ("production_wrapper", wrapper_gate),
        ):
            result[name] = fn()
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(name, "passed", flush=True)
        result["pass"] = True
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
