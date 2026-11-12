"""Source branch boundaries before any mutable GDN provider is called."""

import pytest
import torch
from source_policy_reference import gdn_front_source_family

from sglang.srt.kernel_plugins.nvidia.gdn import gdn_front_metadata_supported
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


def front(**overrides):
    args = dict(
        mk_enabled=True,
        front_enabled=False,
        wide_enabled=False,
        glue_eligible=False,
        target_verify=True,
        rows=4,
        hidden_size=2560,
        input_dtype=torch.bfloat16,
        input_rank=2,
        tp_size=4,
        batch_size=1,
        packed_weight_shape=(4120, 2560),
        packed_weight_dtype=torch.bfloat16,
        activation="silu",
    )
    args.update(overrides)
    return gdn_front_source_family(**args)


@pytest.mark.parametrize("rows", [4, 8, 12])
@pytest.mark.parametrize("tp", [1, 4])
def test_mk_source_rows(rows, tp):
    assert front(rows=rows, tp_size=tp) == "mk"


@pytest.mark.parametrize("rows", [0, 1, 2, 3, 5, 16, 20, 32, 64])
def test_outside_original_mk_rows(rows):
    assert front(rows=rows) is None


def test_wide_is_an_independent_source_gate():
    assert front(rows=16, wide_enabled=True, batch_size=4) == "mk"
    assert front(rows=16, wide_enabled=True, batch_size=1) is None
    assert front(rows=16, wide_enabled=True, batch_size=4, tp_size=1) is None
    assert front(rows=32, wide_enabled=True, batch_size=8) is None


@pytest.mark.parametrize(
    "change",
    [
        dict(mk_enabled=False),
        dict(glue_eligible=True),
        dict(target_verify=False),
        dict(input_dtype=torch.float16),
        dict(hidden_size=2048),
        dict(tp_size=2),
        dict(input_rank=1),
        dict(input_rank=3),
        dict(packed_weight_shape=None),
        dict(packed_weight_dtype=torch.float32),
        dict(activation="relu"),
    ],
)
def test_rejected_before_any_provider_mutation(change):
    assert front(**change) is None


def test_legacy_remains_separate_from_mk_and_exact_tp1_m4():
    args = dict(
        mk_enabled=False,
        front_enabled=True,
        tp_size=1,
        packed_weight_shape=(16480, 2560),
    )
    assert front(**args) == "legacy"
    assert front(**(args | dict(rows=8))) is None
    assert front(**(args | dict(tp_size=4))) is None
    assert front(**(args | dict(packed_weight_shape=(4120, 2560)))) is None
    assert front(**(args | dict(front_enabled=False))) is None


def metadata(**overrides):
    args = dict(
        stable_rows=False,
        has_mis_metadata=False,
        tokens_per_request=4,
        query_start_count=2,
        batch_size=1,
        mk_enabled=True,
        wide_enabled=False,
        has_next_token=False,
        has_next_sibling=False,
        has_parent_token=False,
    )
    args.update(overrides)
    return gdn_front_metadata_supported(**args)


@pytest.mark.parametrize("count", [2, 3, 4])
def test_linear_t4_metadata(count):
    assert metadata(query_start_count=count, batch_size=count - 1)


def test_metadata_wide_source_guard():
    assert not metadata(query_start_count=5)
    assert metadata(query_start_count=5, wide_enabled=True, batch_size=4)
    assert not metadata(query_start_count=5, wide_enabled=True, batch_size=1)
    assert metadata(mk_enabled=False)
    assert not metadata(mk_enabled=False, query_start_count=3)


@pytest.mark.parametrize(
    "change",
    [
        dict(stable_rows=True),
        dict(has_mis_metadata=True),
        dict(tokens_per_request=0),
        dict(query_start_count=1),
        dict(query_start_count=6),
        dict(has_next_token=True),
        dict(has_next_sibling=True),
        dict(has_parent_token=True),
    ],
)
def test_metadata_refusal_before_conv_or_window_write(change):
    assert not metadata(**change)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
