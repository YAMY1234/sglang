"""Actual TP1 host dispatch; CUDA/provider are explicit CPU doubles."""

from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
from test_sparse_score_producer import DeviceMetadata, indexer

from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia import numeric_binding
from sglang.srt.layers.attention.qsa import qsa_indexer as impl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


def flags():
    stack = ExitStack()
    for field, value in [
        (envs.SGLANG_QSA_NUMERIC_SCORE, True),
        (envs.SGLANG_QSA_MK_SCORE, True),
        (envs.SGLANG_QSA_GROUPED_SCORE, True),
        (envs.SGLANG_QSA_SPLIT_TOPK, True),
    ]:
        stack.enter_context(field.override(value))
    stack.enter_context(patch.object(impl, "get_parallel", return_value=NS(tp_size=1)))
    stack.enter_context(patch.object(impl, "is_sm100_supported", return_value=True))
    from sglang.srt.kernel_plugins.nvidia import binding

    stack.enter_context(patch.object(binding, "megakernel_enabled", return_value=True))
    stack.enter_context(envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True))
    stack.enter_context(
        patch.object(impl, "is_batch_invariant_mode_enabled", return_value=False)
    )
    return stack


def inputs(rows, heads, length=8193):
    capacity = ((length + 15) // 16) * 16
    q = torch.zeros((rows, heads, 128), dtype=torch.bfloat16).as_subclass(
        DeviceMetadata
    )
    return [
        q,
        torch.zeros((1, 16, 1, 128), dtype=torch.bfloat16),
        torch.zeros((rows, capacity // 16), dtype=torch.int64),
        torch.full((rows,), length, dtype=torch.int64),
        capacity,
        None,
    ]


@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize(
    "rows,group,variant",
    [
        (4, 4, 4),
        (8, 4, 4),
        (12, 4, 4),
        (32, 4, 4),
        (64, 4, 4),
        (1, 1, 5),
        (8, 1, 5),
        (32, 1, 5),
        (64, 1, 5),
    ],
)
def test_declared_pair_and_real_public_dispatch(rows, group, variant, heads):
    obj = indexer()
    obj.index_n_heads = heads
    obj.numeric_signatures((rows,), 4)
    args = inputs(rows, heads)
    with flags():
        declarations = obj.numeric_preparation_signatures(8193 * 4, 16)
        calls = []
        outs = []
        expected = (
            torch.arange(512, dtype=torch.int32).roll(11).expand(rows, -1).clone()
        )

        def choose(signature, *, prepared):
            assert prepared and signature in declarations

            def run(io):
                calls.append(signature)
                outs.append(io)
                if signature.operation == NumericOperation.SPARSE_SCORE:
                    assert (
                        signature.variant == variant and signature.query_heads == heads
                    )
                    assert (
                        io.query.shape == (rows, heads, 128) and io.row_requests is None
                    )
                    assert io.query.data_ptr() == args[0].data_ptr()
                    io.scores_out.zero_()
                else:
                    assert signature.variant == 1 and io.scores is outs[0].scores_out
                    assert io.counts_out is None
                    io.selected_indices_out.copy_(expected)

            return run

        with (
            patch.object(numeric_binding, "numeric_operator", side_effect=choose),
            patch.object(impl, "qsa_mqa_decode") as native,
        ):
            actual = obj.select_decode_tokens(
                *args[:5], torch.zeros(rows), args[3], True, None, group
            )
        assert torch.equal(actual, expected) and len(calls) == 2
        native.assert_not_called()


@pytest.mark.parametrize(
    "missing", [NumericOperation.SPARSE_SCORE, NumericOperation.SPARSE_SELECT]
)
def test_pair_omission_does_not_execute_either(missing):
    obj = indexer()
    obj.index_n_heads = 16
    obj.numeric_signatures((32,), 4)
    args = inputs(32, 16)
    with flags():
        obj.numeric_preparation_signatures(8193 * 4, 16)
        runs = []

        def lookup(sig, *, prepared):
            return None if sig.operation == missing else lambda io: runs.append(io)

        with patch.object(numeric_binding, "numeric_operator", side_effect=lookup):
            assert obj._numeric_score_select(*args, 4) is None
        assert not runs


@pytest.mark.parametrize(
    "field,value,group",
    [
        ("SGLANG_QSA_SCORER_V2", False, 4),
        ("SGLANG_QSA_SCORER_V2_VER", 6, 4),
        ("SGLANG_QSA_SCORER_V2_C", 256, 4),
        ("SGLANG_QSA_SCORER_V2_TARGET", 2048, 4),
        ("SGLANG_QSA_SCORER_V2_NT", 2, 4),
        ("SGLANG_QSA_SCORER_V2_GUARD", True, 4),
        ("SGLANG_QSA_SCORER_V2_PEEL", True, 4),
        ("SGLANG_QSA_SCORER_V2_ROWS", True, 1),
        ("SGLANG_QSA_TOPK_THREADS", 512, 4),
        ("SGLANG_QSA_TOPK_CLUSTER", 8, 4),
    ],
)
def test_nondefault_unoffered_policy_is_stock_before_lookup(field, value, group):
    obj = indexer()
    obj.index_n_heads = 16
    obj.numeric_signatures((32,), 4)
    args = inputs(32, 16)
    with flags(), getattr(envs, field).override(value):
        obj.numeric_preparation_signatures(8193 * 4, 16)
        with patch.object(numeric_binding, "numeric_operator") as lookup:
            assert obj._numeric_score_select(*args, group) is None
            lookup.assert_not_called()


def test_mk_with_source_split_precedence_uses_split_not_mk_selector():
    obj = indexer()
    args = inputs(4, 8)
    args[-1] = torch.ones(4, dtype=torch.int64)
    with flags():
        declarations = obj.numeric_preparation_signatures(8193 * 4, 16)
        selected = []

        def choose(sig, *, prepared):
            assert sig in declarations
            selected.append(sig)
            return lambda io: None

        with patch.object(numeric_binding, "numeric_operator", side_effect=choose):
            assert obj._numeric_score_select(*args, 1) is not None
        assert [s.variant for s in selected] == [0, 1]


def test_q4_without_mk_does_not_fall_into_native_score():
    obj = indexer()
    args = inputs(4, 8)
    args[-1] = torch.ones(4, dtype=torch.int64)
    with (
        flags(),
        envs.SGLANG_QSA_MK_SCORE.override(False),
        envs.SGLANG_QSA_Q4_SCORE.override(True),
    ):
        obj.numeric_preparation_signatures(8193 * 4, 16)
        with patch.object(numeric_binding, "numeric_operator") as lookup:
            assert obj._numeric_score_select(*args, 1) is None
            lookup.assert_not_called()


def test_grouped_verify_never_admits_q4_and_both_runtime_heads_are_exact():
    with flags(), envs.SGLANG_QSA_Q4_SCORE.override(True):
        for actual_heads in (4, 16):
            obj = indexer()
            obj.index_n_heads = actual_heads
            heads = ((actual_heads + 7) // 8) * 8
            args = inputs(32, heads)
            obj.numeric_preparation_signatures(8193 * 4, 16)
            requests = torch.ones(32, dtype=torch.int64)
            assert (
                obj._score_requests(
                    args[0],
                    args[1],
                    NS(is_cuda_graph=True, req_pool_indices=requests),
                    NS(req_pool_indices=torch.ones(8)),
                    True,
                    False,
                )
                is None
            )
            with patch.object(numeric_binding, "numeric_operator") as lookup:
                wrong = inputs(32, 16 if heads == 8 else 8)
                assert obj._numeric_score_select(*wrong, 4) is None
                lookup.assert_not_called()


@pytest.mark.parametrize("rows", [65, 68, 96, 128])
def test_outside_source_split_row_domain_is_stock(rows):
    obj = indexer()
    obj.index_n_heads = 16
    args = inputs(rows, 16)
    with flags():
        obj.numeric_preparation_signatures(8193 * 4, 16)
        with patch.object(numeric_binding, "numeric_operator") as lookup:
            assert obj._numeric_score_select(*args, 4 if rows % 4 == 0 else 1) is None
            lookup.assert_not_called()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
