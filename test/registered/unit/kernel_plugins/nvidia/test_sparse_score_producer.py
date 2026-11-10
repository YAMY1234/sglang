"""Installed paired producer with CPU-only CUDA/provider metadata doubles."""

from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia import numeric_binding
from sglang.srt.layers.attention.qsa import qsa_indexer as impl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class DeviceMetadata(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def indexer():
    obj = object.__new__(impl.QSAIndexer)
    torch.nn.Module.__init__(obj)
    obj.index_n_heads, obj.index_kv_heads, obj.index_head_dim = 4, 1, 128
    obj.block_topk, obj.compress_ratio, obj.token_topk = 512, 4, 2048
    from capability_fixtures import contract_spec

    obj._numeric_capability_spec = contract_spec()
    obj.numeric_signatures((4,), 4)
    return obj


def tensors(capacity):
    q = torch.zeros((4, 8, 128), dtype=torch.bfloat16).as_subclass(DeviceMetadata)
    return (
        q,
        torch.zeros((1, 16, 1, 128), dtype=torch.bfloat16),
        torch.zeros((4, capacity // 16), dtype=torch.int64),
        torch.full((4,), capacity, dtype=torch.int64),
        capacity,
        torch.ones(4, dtype=torch.int64),
    )


def flags():
    from contextlib import ExitStack

    stack = ExitStack()
    from sglang.srt.kernel_plugins.nvidia import binding

    stack.enter_context(patch.object(binding, "megakernel_enabled", return_value=True))
    stack.enter_context(envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True))
    stack.enter_context(envs.SGLANG_QSA_MK_SCORE.override(True))
    stack.enter_context(envs.SGLANG_QSA_SPLIT_TOPK.override(False))
    stack.enter_context(patch.object(impl, "get_parallel", return_value=NS(tp_size=4)))
    stack.enter_context(patch.object(impl, "is_sm100_supported", return_value=True))
    stack.enter_context(
        patch.object(impl, "is_batch_invariant_mode_enabled", return_value=False)
    )
    return stack


@pytest.mark.parametrize("length", [499, 512, 513, 4096, 4097, 8192, 8193])
def test_actual_pool_capacity_and_paired_outputs_preserve_order(length):
    obj = indexer()
    with flags():
        signatures = obj.numeric_preparation_signatures(length * 4, 16)
        capacity = ((length + 15) // 16) * 16
        assert len(signatures) == 2
        assert all(s.context_bound == capacity for s in signatures)
        args = list(tensors(capacity))
        args[3].fill_(length)
        calls, outputs = [], []
        # A deliberately non-sorted order, including a tied cutoff, must pass
        # through this orchestration unchanged. Raw GPU ordering is tested separately.
        expected = torch.arange(511, -1, -1, dtype=torch.int32).expand(4, -1).clone()

        def choose(sig, *, prepared):
            assert prepared and sig in signatures

            def run(io):
                calls.append(sig.operation)
                outputs.append(io)
                if sig.operation == NumericOperation.SPARSE_SCORE:
                    assert io.query.dtype == torch.bfloat16 and io.scale == 128**0.5
                    assert (
                        io.compressed_slot_map.dtype
                        == io.compressed_lengths.dtype
                        == torch.int32
                    )
                    assert (
                        io.row_requests.dtype == torch.int64
                        and io.query_weights is None
                    )
                    io.scores_out.zero_()
                else:
                    assert io.scores is outputs[0].scores_out
                    assert io.counts_out is None
                    io.selected_indices_out.copy_(expected)

            return run

        with patch.object(numeric_binding, "numeric_operator", side_effect=choose):
            actual = obj._numeric_score_select(*args)
        assert calls == [NumericOperation.SPARSE_SCORE, NumericOperation.SPARSE_SELECT]
        assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "missing", [NumericOperation.SPARSE_SCORE, NumericOperation.SPARSE_SELECT]
)
def test_missing_either_offer_does_not_execute_either(missing):
    obj = indexer()
    with flags():
        obj.numeric_preparation_signatures(32768, 16)
        runs = []

        def choose(sig, *, prepared):
            return None if sig.operation == missing else lambda io: runs.append(io)

        with patch.object(numeric_binding, "numeric_operator", side_effect=choose):
            assert obj._numeric_score_select(*tensors(8192)) is None
        assert runs == []


@pytest.mark.parametrize(
    "case",
    [
        "off",
        "unprepared",
        "wrongq",
        "wrongcache",
        "wrongrequest",
        "batch_invariant",
        "split_precedence",
    ],
)
def test_ineligible_never_queries_registry(case):
    obj = indexer()
    with flags():
        obj.numeric_preparation_signatures(32768, 16)
        args = list(tensors(8192))
        with patch.object(numeric_binding, "numeric_operator") as lookup:
            if case == "off":
                with envs.SGLANG_QSA_MK_SCORE.override(False):
                    assert obj._numeric_score_select(*args) is None
            elif case == "batch_invariant":
                with patch.object(
                    impl, "is_batch_invariant_mode_enabled", return_value=True
                ):
                    assert obj._numeric_score_select(*args) is None
            elif case == "split_precedence":
                with (
                    patch.object(impl, "get_parallel", return_value=NS(tp_size=1)),
                    envs.SGLANG_QSA_SPLIT_TOPK.override(True),
                ):
                    assert obj._numeric_score_select(*args) is None
            else:
                if case == "unprepared":
                    obj._numeric_score_capacity = 0
                if case == "wrongq":
                    args[0] = args[0].float()
                if case == "wrongcache":
                    args[1] = args[1].float()
                if case == "wrongrequest":
                    args[5] = args[5].float()
                assert obj._numeric_score_select(*args) is None
            lookup.assert_not_called()


def test_finite_preparation_and_short_launch_variant():
    obj = indexer()
    with (
        envs.SGLANG_QSA_MK_SCORE.override(False),
        envs.SGLANG_QSA_NUMERIC_SCORE.override(False),
        envs.SGLANG_QSA_SPLIT_TOPK.override(False),
    ):
        assert obj.numeric_preparation_signatures(32768, 16) == ()
    with flags():
        for context, page in [(0, 16), (32768, 32), (524289, 16)]:
            assert obj.numeric_preparation_signatures(context, page) == ()
        with envs.SGLANG_QSA_MK_SCORE_SHORT32.override(True):
            score, select = obj.numeric_preparation_signatures(32769, 16)
            assert score.variant == 3 and select.variant == 0
            assert score.context_bound == select.context_bound == 8208


@pytest.mark.parametrize(
    "target,draft,tp,grouped,graph,request_count,expected",
    [
        (True, False, 4, True, True, 1, True),
        (True, False, 1, True, True, 1, False),
        (True, False, 1, False, True, 1, True),
        (False, True, 1, True, True, 1, True),
        (False, False, 4, True, True, 1, False),
        (True, False, 4, True, False, 1, False),
        (True, False, 4, True, True, 2, False),
    ],
)
def test_source_single_request_graph_admission(
    target, draft, tp, grouped, graph, request_count, expected
):
    obj = indexer()
    q, cache, *_ = tensors(8192)
    with flags():
        obj.numeric_preparation_signatures(32768, 16)
    requests = torch.ones(4, dtype=torch.int64)
    with (
        flags(),
        envs.SGLANG_QSA_GROUPED_SCORE.override(grouped),
        patch.object(impl, "get_parallel", return_value=NS(tp_size=tp)),
    ):
        result = obj._score_requests(
            q,
            cache,
            NS(is_cuda_graph=graph, req_pool_indices=requests),
            NS(req_pool_indices=torch.ones(request_count)),
            target,
            draft,
        )
    assert (result is requests) == expected


def test_post_pool_preparation_passes_only_frozen_exact_signatures():
    from sglang.srt.kernel_plugins.nvidia import binding

    obj = indexer()
    model = torch.nn.Module()
    model.add_module("indexer", obj)
    runner = NS(
        token_to_kv_pool=NS(qsa_compressed_page_size=16),
        model_config=NS(context_len=32769),
    )
    bound = binding.MegakernelBinding(model=model)
    bound._numeric_spec = obj._numeric_capability_spec
    with (
        flags(),
        patch.object(bound, "_prepare_hybrid", return_value=None),
        patch.object(bound, "assert_live"),
    ):
        prepared = bound.preparation_view(runner)
        assert prepared.hybrid is None and len(prepared.numeric_signatures) == 2
        assert all(s.context_bound == 8208 for s in prepared.numeric_signatures)
        assert bound.preparation_view(runner) is prepared
        assert all(
            isinstance(getattr(s, field), int)
            for s in prepared.numeric_signatures
            for field in s.__dataclass_fields__
        )


def test_hybrid_pool_preparation_uses_actual_leaf_page_size():
    from sglang.srt.kernel_plugins.nvidia import binding

    obj = indexer()
    model = torch.nn.Module()
    model.add_module("indexer", obj)
    runner = NS(
        token_to_kv_pool=NS(full_kv_pool=NS(qsa_compressed_page_size=32)),
        model_config=NS(context_len=32768),
    )
    bound = binding.MegakernelBinding(model=model)
    bound._numeric_spec = obj._numeric_capability_spec
    with flags(), patch.object(bound, "_prepare_hybrid", return_value=None):
        assert bound.preparation_view(runner).numeric_signatures == ()
        assert obj._numeric_score_capacity == 0


def test_public_select_stock_fallback_and_selected_expansion():
    obj = indexer()
    args = list(tensors(8192))
    q, cache, table, lengths, capacity, requests = args
    positions = torch.arange(4)
    scores = torch.zeros((4, capacity))
    native = torch.zeros((4, 512), dtype=torch.int32)
    fake_module = NS(fast_topk=lambda *a, **kw: native)
    import sys

    with (
        patch.dict(
            sys.modules, {"sglang.kernels.ops.attention.fast_topk": fake_module}
        ),
        patch.object(impl, "qsa_mqa_decode", return_value=scores) as score,
        patch.object(impl, "is_batch_invariant_mode_enabled", return_value=False),
        patch.object(obj, "_numeric_score_select", return_value=None),
    ):
        assert (
            obj.select_decode_tokens(
                q, cache, table, lengths, capacity, positions, lengths, True, requests
            )
            is native
        )
        score.assert_called_once_with(
            q, cache, table, lengths, capacity, min_scored_len=512
        )
    selected = torch.ones((4, 512), dtype=torch.int32)
    expanded = torch.ones((4, 2051), dtype=torch.int32)
    with (
        patch.object(obj, "_numeric_score_select", return_value=selected),
        patch.object(impl, "qsa_mqa_decode") as native_score,
        patch.object(impl, "expand_qsa_block_indices", return_value=expanded) as expand,
    ):
        assert (
            obj.select_decode_tokens(
                q, cache, table, lengths, capacity, positions, lengths, False, requests
            )
            is expanded
        )
        native_score.assert_not_called()
        expand.assert_called_once_with(
            selected, positions, lengths, compress_ratio=4, token_topk=2048
        )


def prefill_flags(tp=1, fold="1"):
    from contextlib import ExitStack

    from sglang.srt.kernel_plugins.nvidia import binding, pool_attention

    stack = ExitStack()
    for field, value in [
        (envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION, True),
        (envs.SGLANG_ENABLE_QSA_PREFILL_FAST_SELECT, True),
        (envs.SGLANG_OPT_QSA_PREFILL_SCALE_FOLD, fold),
        (envs.SGLANG_QSA_SPLIT_TOPK, False),
    ]:
        stack.enter_context(field.override(value))
    for target, name, value in [
        (binding, "megakernel_enabled", True),
        (impl, "get_parallel", NS(tp_size=tp)),
        (impl, "is_sm100_supported", True),
        (impl, "is_batch_invariant_mode_enabled", False),
        (torch.cuda, "is_current_stream_capturing", False),
        (pool_attention, "pool_device_sms", 148),
    ]:
        stack.enter_context(patch.object(target, name, return_value=value))
    stack.enter_context(patch.object(impl, "HAS_TILELANG", True))
    return stack


def prefill_args(rows, width=8193):
    q = torch.zeros((rows, 4, 128), dtype=torch.bfloat16).as_subclass(DeviceMetadata)
    positions = torch.arange(rows)
    return [
        q,
        torch.zeros((2 * width, 1, 128), dtype=torch.bfloat16),
        torch.zeros(rows, dtype=torch.int64),
        torch.full((rows,), width, dtype=torch.int64),
        positions,
        positions + 1,
        width,
    ]


@pytest.mark.parametrize(
    "tp,heads,fast,declared",
    [
        (1, 4, True, True),
        (4, 4, True, True),
        (1, 8, True, False),
        (1, 4, False, False),
    ],
)
def test_prefill_select_declares_one_row_independent_signature(
    tp, heads, fast, declared
):
    from sglang.kernels.plugin_kernels.nvidia.numeric import (
        NumericSignature,
        ScalarType,
    )

    obj = indexer()
    obj.index_n_heads = heads
    with (
        prefill_flags(tp=tp),
        envs.SGLANG_ENABLE_QSA_PREFILL_FAST_SELECT.override(fast),
    ):
        found = [
            s
            for s in obj.numeric_preparation_signatures(8193 * 4, 16)
            if s.operation == NumericOperation.PREFILL_SELECT
        ]
    expected = NumericSignature(
        NumericOperation.PREFILL_SELECT,
        0,
        4,
        1,
        128,
        512,
        4,
        ScalarType.BF16,
        context_bound=8208,
    )
    assert found == ([expected] if declared else [])


@pytest.mark.parametrize("rows", [1, 33, 64])
def test_prefill_select_fills_output_through_provider(rows):
    obj = indexer()
    args = prefill_args(rows)
    with prefill_flags():
        declared = obj.numeric_preparation_signatures(8193 * 4, 16)
        seen = []

        def choose(signature, *, prepared):
            assert prepared and signature in declared
            return seen.append

        with (
            patch.object(numeric_binding, "numeric_operator", side_effect=choose),
            patch.object(impl, "qsa_mqa_prefill_relative") as stock,
        ):
            output = obj.select_prefill_tokens(*args)
    stock.assert_not_called()
    (io,) = seen
    padded = (rows + 31) // 32 * 32
    assert io.selected_indices_out is output and output.shape == (rows, 2051)
    assert io.query.data_ptr() == args[0].data_ptr()
    assert (io.chunk_rows, io.scorer_sms) == (rows, 148)
    assert io.logits_out.shape == (padded, 8193)
    assert io.query_scratch_out.shape == (padded, 4, 128)
    assert io.block_indices_out.shape == (rows, 512)
    assert io.split_tag.shape == (4096,) and io.zero_offsets.shape == (rows,)
    assert all(
        t.dtype == torch.int32
        for t in (io.row_starts, io.row_ends, io.query_positions, io.sequence_lengths)
    )
    assert torch.equal(io.row_lengths, torch.full((rows,), 8193, dtype=torch.int32))


def test_prefill_select_chunk_plan_matches_the_source():
    # Literal source plan_chunk_rows with its 128 MiB budget and 2 * SMs top-k slots.
    def source(rows, width, slots, budget=128 * 1024 * 1024):
        max_rows = max(32, budget // (width * 4))
        max_rows = max(32, max_rows // 32 * 32)
        if rows <= max_rows:
            return rows
        unit = slots // 32 * 32
        if unit >= 32 and max_rows >= unit:
            return max_rows // unit * unit
        return max_rows

    for rows in (1, 31, 33, 4064, 4065, 20000, 131072):
        for width in (1, 512, 4097, 8193, 32768, 131072):
            for sms in (8, 148, 160):
                assert impl._fast_select_chunk_rows(rows, width, sms) == source(
                    rows, width, 2 * sms
                )


@pytest.mark.parametrize(
    "case", ["fold_off", "wider_than_declared", "capturing", "batch_invariant", "off"]
)
def test_prefill_select_outside_source_policy_is_stock(case):
    obj = indexer()
    args = prefill_args(33, 8209 if case == "wider_than_declared" else 8193)
    with prefill_flags(fold="0" if case == "fold_off" else "1"):
        obj.numeric_preparation_signatures(8193 * 4, 16)
        logits = torch.zeros((33, args[-1]))
        with (
            patch.object(
                numeric_binding,
                "numeric_operator",
                side_effect=AssertionError("provider lookup"),
            ),
            patch.object(
                torch.cuda,
                "is_current_stream_capturing",
                return_value=case == "capturing",
            ),
            patch.object(
                impl,
                "is_batch_invariant_mode_enabled",
                return_value=case == "batch_invariant",
            ),
            envs.SGLANG_ENABLE_QSA_PREFILL_FAST_SELECT.override(case != "off"),
            patch.object(
                impl, "qsa_mqa_prefill_relative", return_value=logits
            ) as stock,
            patch.object(impl, "qsa_fast_topk", return_value=torch.zeros(33, 512)),
            patch.object(
                impl,
                "expand_qsa_block_indices",
                return_value=torch.zeros((33, 2051), dtype=torch.int32),
            ),
        ):
            obj.select_prefill_tokens(*args)
    stock.assert_called_once()


@pytest.mark.parametrize(
    "context,page",
    [
        (None, 16),
        ("32768", 16),
        (True, 16),
        (-1, 16),
        (32768, None),
        (32768, "16"),
        (32768, 0),
        (524289, 16),
    ],
)
def test_invalid_resolved_metadata_is_stock(context, page):
    obj = indexer()
    with flags():
        assert obj.numeric_preparation_signatures(context, page) == ()
        assert obj._numeric_score_capacity == 0


@pytest.mark.parametrize(
    "runner", [NS(), NS(token_to_kv_pool=NS()), NS(model_config=NS(context_len=32768))]
)
def test_missing_resolved_objects_are_stock(runner):
    from sglang.srt.kernel_plugins.nvidia import binding

    obj = indexer()
    model = torch.nn.Module()
    model.add_module("indexer", obj)
    bound = binding.MegakernelBinding(model=model)
    with flags(), patch.object(bound, "_prepare_hybrid", return_value=None):
        assert bound.preparation_view(runner).numeric_signatures == ()
        assert obj._numeric_score_capacity == 0


@pytest.fixture(autouse=True)
def capability_provider_enabled(monkeypatch):
    from sglang.srt.kernel_plugins.nvidia import binding

    monkeypatch.setattr(binding, "megakernel_enabled", lambda: True)
    monkeypatch.setenv("SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION", "1")


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
