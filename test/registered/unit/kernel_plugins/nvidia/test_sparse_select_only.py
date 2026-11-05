"""Native scorer -> prepared SELECT1; CPU tensors and explicit CUDA metadata."""

from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
from test_sparse_score_producer import DeviceMetadata, indexer

from sglang.kernels.ops.attention import fast_topk as stock
from sglang.kernels.plugin_kernels.nvidia import numeric as api
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia import binding, numeric_binding
from sglang.srt.layers.attention.qsa import qsa_indexer as impl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


def flags():
    stack = ExitStack()
    for field, value in [
        (envs.SGLANG_QSA_MK_SCORE, False),
        (envs.SGLANG_QSA_NUMERIC_SCORE, False),
        (envs.SGLANG_QSA_SPLIT_TOPK, True),
        (envs.SGLANG_QSA_TOPK_THREADS, None),
        (envs.SGLANG_QSA_TOPK_CLUSTER, None),
        (envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION, True),
    ]:
        stack.enter_context(field.override(value))
        if value is None:
            field.clear()  # Raw provider readers must see the original unset env.
    stack.enter_context(patch.object(impl, "get_parallel", return_value=NS(tp_size=1)))
    stack.enter_context(patch.object(impl, "is_sm100_supported", return_value=True))
    stack.enter_context(
        patch.object(impl, "is_batch_invariant_mode_enabled", return_value=False)
    )
    stack.enter_context(patch.object(binding, "megakernel_enabled", return_value=True))
    return stack


def prepare(obj, widths, capacity=8192):
    assert numeric_binding.collect_numeric_signatures((obj,), widths, 4) == ()
    signatures = obj.numeric_preparation_signatures(capacity * 4, 16)
    producer = NS(numeric_signatures=lambda widths, tokens: signatures)
    assert (
        numeric_binding.collect_numeric_signatures((producer,), widths, 4) == signatures
    )
    return signatures


@pytest.mark.parametrize("heads", [4, 8, 16])
def test_startup_select_only_is_finite_without_score_or_forward_registration(heads):
    obj = indexer()
    obj.index_n_heads = heads
    with flags():
        signatures = prepare(obj, (True, 0, 4, 8, 12, 16, 65), 65536)
    assert {s.rows for s in signatures} == {4, 8, 12, 16}
    assert all(
        s.operation == api.NumericOperation.SPARSE_SELECT
        and s.variant == 1
        and s.scalar_type == api.ScalarType.FP32
        and s.context_bound == 65536
        and s.query_heads == ((heads + 7) // 8) * 8
        for s in signatures
    )


@pytest.mark.parametrize("rows", [4, 8, 12, 16])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
def test_native_scorer_uses_real_select_facade_and_preserves_order(
    rows, dtype, length_dtype
):
    obj = indexer()
    capacity = 8192
    q = torch.zeros((rows, 8, 128), dtype=dtype).as_subclass(DeviceMetadata)
    cache = torch.zeros((1, 16, 1, 128), dtype=dtype)
    scores = torch.zeros((rows, capacity), dtype=torch.float32).as_subclass(
        DeviceMetadata
    )
    lengths = torch.full((rows,), 513, dtype=length_dtype)
    expected = torch.arange(512, dtype=torch.int32).roll(17).expand(rows, -1).clone()
    calls, views = [], []
    registry = api.NumericRegistry()

    def select(io):
        calls.append(api.NumericOperation.SPARSE_SELECT)
        views.append(io)
        assert io.scores is scores and io.lengths.dtype == torch.int32
        assert io.counts_out is None and io.scratch_out.numel() == 0
        assert io.selected_indices_out.shape == (rows, 512)
        if length_dtype == torch.int32:
            assert io.lengths.data_ptr() == lengths.data_ptr()
        io.selected_indices_out.copy_(expected)

    with flags():
        signatures = prepare(obj, (rows,))
        for sig in signatures:
            registry.register(api.NumericOffer(sig, select))
        with (
            patch.object(numeric_binding, "numeric_registry", registry),
            patch.object(impl, "qsa_mqa_decode", return_value=scores) as native,
            patch.object(
                stock, "fast_topk", side_effect=AssertionError("stock repeated")
            ),
        ):
            actual = obj.select_decode_tokens(
                q,
                cache,
                torch.zeros((rows, capacity // 16), dtype=torch.int32),
                lengths,
                capacity,
                torch.zeros(rows),
                lengths,
                True,
            )
    assert torch.equal(actual, expected) and calls == [
        api.NumericOperation.SPARSE_SELECT
    ]
    assert actual is views[0].selected_indices_out
    native.assert_called_once()
    assert native.call_args.kwargs["min_scored_len"] == 512


@pytest.mark.parametrize(
    "case",
    [
        "TP4",
        "arch",
        "sourceoff",
        "threads",
        "cluster",
        "undeclared",
        "capacity",
        "too_wide",
        "zero_width",
        "scores_dtype",
        "scores_device",
        "scores_stride",
        "lengths_dtype",
        "lengths_stride",
        "lengths_shape",
    ],
)
def test_source_and_layout_refusal_precedes_registry(case):
    obj = indexer()
    scores = torch.zeros((4, 8192), dtype=torch.float32).as_subclass(DeviceMetadata)
    lengths = torch.ones(4, dtype=torch.int32)
    with flags(), ExitStack() as change:
        prepare(obj, (4,))
        if case == "TP4":
            change.enter_context(
                patch.object(impl, "get_parallel", return_value=NS(tp_size=4))
            )
        elif case == "arch":
            change.enter_context(
                patch.object(impl, "is_sm100_supported", return_value=False)
            )
        elif case == "sourceoff":
            change.enter_context(envs.SGLANG_QSA_SPLIT_TOPK.override(False))
        elif case == "threads":
            change.enter_context(envs.SGLANG_QSA_TOPK_THREADS.override(512))
        elif case == "cluster":
            change.enter_context(envs.SGLANG_QSA_TOPK_CLUSTER.override(8))
        elif case == "undeclared":
            obj._numeric_split_select_widths = ()
        elif case == "capacity":
            obj._numeric_score_capacity = 4096
        elif case == "too_wide":
            scores = torch.empty((4, 131088)).as_subclass(DeviceMetadata)
        elif case == "zero_width":
            scores = torch.empty((4, 0)).as_subclass(DeviceMetadata)
        elif case == "scores_dtype":
            scores = scores.to(torch.bfloat16)
        elif case == "scores_device":
            scores = torch.zeros((4, 8192))
        elif case == "scores_stride":
            scores = torch.zeros((4, 16384)).as_subclass(DeviceMetadata)[:, ::2]
        elif case == "lengths_dtype":
            lengths = lengths.float()
        elif case == "lengths_stride":
            lengths = torch.ones(8, dtype=torch.int32)[::2]
        elif case == "lengths_shape":
            lengths = lengths[:3]
        with patch.object(numeric_binding, "numeric_operator") as lookup:
            assert obj._numeric_split_select(scores, lengths) is None
        lookup.assert_not_called()


@pytest.mark.parametrize(
    "change", ["heads", "kv_heads", "dimension", "topk", "ratio", "page", "capacity"]
)
def test_startup_rejects_other_domains(change):
    obj = indexer()
    context, page = 32768, 16
    if change == "heads":
        obj.index_n_heads = 17
    elif change == "kv_heads":
        obj.index_kv_heads = 2
    elif change == "dimension":
        obj.index_head_dim = 64
    elif change == "topk":
        obj.block_topk = 256
    elif change == "ratio":
        obj.compress_ratio = 2
    elif change == "page":
        page = 32
    elif change == "capacity":
        context = 131088 * 4
    with flags():
        obj.numeric_signatures((4,), 4)
        assert obj.numeric_preparation_signatures(context, page) == ()


@pytest.mark.parametrize("threads", [None, 1024])
def test_default_and_explicit_default_threads_share_original_policy(threads):
    with flags(), envs.SGLANG_QSA_TOPK_THREADS.override(threads):
        assert prepare(indexer(), (4,))


@pytest.mark.parametrize("mode", ["missing", "masteroff", "selectedfailure"])
def test_stock_once_before_selection_and_never_after_partial_write(mode):
    obj = indexer()
    q = torch.zeros((4, 8, 128), dtype=torch.float8_e4m3fn).as_subclass(DeviceMetadata)
    scores = torch.zeros((4, 8192)).as_subclass(DeviceMetadata)
    lengths = torch.full((4,), 513, dtype=torch.int32)
    registry = api.NumericRegistry()
    views = []

    def failing(io):
        views.append(io)
        io.selected_indices_out.fill_(7)
        raise RuntimeError("selected write failed")

    with flags(), ExitStack() as controls:
        signatures = prepare(obj, (4,))
        if mode != "missing":
            registry.register(api.NumericOffer(signatures[0], failing))
        if mode == "masteroff":
            controls.enter_context(
                envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(False)
            )
        with (
            patch.object(numeric_binding, "numeric_registry", registry),
            patch.object(impl, "qsa_mqa_decode", return_value=scores) as native,
            patch.object(stock, "fast_topk", return_value="stock") as fallback,
        ):
            call = lambda: obj.select_decode_tokens(
                q,
                q,
                torch.zeros((4, 512)),
                lengths,
                8192,
                torch.zeros(4),
                lengths,
                True,
            )
            if mode == "selectedfailure":
                with pytest.raises(RuntimeError, match="selected write failed"):
                    call()
                fallback.assert_not_called()
                assert torch.all(views[0].selected_indices_out == 7)
            else:
                assert call() == "stock"
                fallback.assert_called_once()
                assert not views
            native.assert_called_once()


def test_pair_selected_still_returns_before_native_or_select_only():
    obj = indexer()
    with (
        flags(),
        patch.object(obj, "_numeric_score_select", return_value="paired"),
        patch.object(
            obj, "_numeric_split_select", side_effect=AssertionError("late select")
        ),
        patch.object(
            impl, "qsa_mqa_decode", side_effect=AssertionError("native scorer")
        ),
        envs.SGLANG_QSA_NUMERIC_SCORE.override(True),
    ):
        q = torch.zeros((4, 8, 128), dtype=torch.bfloat16)
        assert obj.select_decode_tokens(q, q, q, q, 8192, q, q, True) == "paired"


@pytest.mark.parametrize("reason", ["CPU", "topk", "invariant"])
def test_original_generic_fast_topk_parent_gates_remain_unchanged(reason):
    obj = indexer()
    q = torch.zeros((4, 8, 128), dtype=torch.float8_e4m3fn)
    if reason != "CPU":
        q = q.as_subclass(DeviceMetadata)
    if reason == "topk":
        obj.block_topk = 256
    lengths = torch.ones(4, dtype=torch.int32)
    with (
        flags(),
        patch.object(
            impl, "is_batch_invariant_mode_enabled", return_value=reason == "invariant"
        ),
        patch.object(
            impl, "qsa_mqa_decode", return_value=torch.zeros((4, 8192))
        ) as score,
        patch.object(impl, "qsa_fast_topk", return_value="original") as generic,
        patch.object(
            obj,
            "_numeric_split_select",
            side_effect=AssertionError("split parent false"),
        ),
    ):
        assert (
            obj.select_decode_tokens(q, q, q, lengths, 8192, q, lengths, True)
            == "original"
        )
    assert score.call_args.kwargs["min_scored_len"] == 0
    generic.assert_called_once()


def test_real_post_pool_preparation_freezes_only_declared_select_widths():
    obj = indexer()
    model = torch.nn.Module()
    model.add_module("indexer", obj)
    runner = NS(
        token_to_kv_pool=NS(qsa_compressed_page_size=16),
        model_config=NS(context_len=32769),
    )
    bound = binding.MegakernelBinding(model=model)
    bound._numeric_spec = obj._numeric_capability_spec
    with flags(), patch.object(bound, "_prepare_hybrid", return_value=None):
        numeric_binding.collect_numeric_signatures((obj,), (4, 8, 12, 16), 4)
        view = bound.preparation_view(runner)
    assert {s.rows for s in view.numeric_signatures} == {4, 8, 12, 16}
    assert all(
        s.context_bound == 8208
        and s.operation == api.NumericOperation.SPARSE_SELECT
        and s.variant == 1
        for s in view.numeric_signatures
    )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
