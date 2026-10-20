"""SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS: a pre-planned verify replay skips
re-copying input_ids / positions only when they are the very tensors the plan-time
load copied; any other source, even with equal values, is copied."""

from types import SimpleNamespace

import torch

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _runner():
    r = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    r.buffers = SimpleNamespace(
        input_ids=torch.full((8,), -1, dtype=torch.int64),
        positions=torch.full((8,), -1, dtype=torch.int64),
    )
    r.raw_num_token = 3
    r._prefilled_inputs = None
    r.num_prefilled_input_skips = 0
    return r


def _batch(base):
    return SimpleNamespace(
        input_ids=torch.arange(base, base + 3), positions=torch.arange(3) + 100
    )


def test_copies_without_a_plan_time_fill():
    r, fb = _runner(), _batch(10)
    r._copy_replay_inputs(fb)
    assert r.buffers.input_ids[:3].tolist() == [10, 11, 12]
    assert r.buffers.positions[:3].tolist() == [100, 101, 102]
    assert r.num_prefilled_input_skips == 0


def test_skips_only_the_plan_time_tensors_and_only_once():
    r, fb = _runner(), _batch(10)
    r._prefilled_inputs = (fb.input_ids, fb.positions)
    r._copy_replay_inputs(fb)
    # Untouched: the plan-time load owns these rows.
    assert r.buffers.input_ids[:3].tolist() == [-1, -1, -1]
    assert r.buffers.positions[:3].tolist() == [-1, -1, -1]
    assert r.num_prefilled_input_skips == 1 and r._prefilled_inputs is None
    # The record is consumed: the next replay copies again.
    r._copy_replay_inputs(fb)
    assert r.buffers.input_ids[:3].tolist() == [10, 11, 12]


def test_equal_values_from_other_tensors_are_copied():
    r, fb = _runner(), _batch(10)
    other = _batch(10)
    r._prefilled_inputs = (other.input_ids, other.positions)
    r._copy_replay_inputs(fb)
    assert r.buffers.input_ids[:3].tolist() == [10, 11, 12]
    assert r.buffers.positions[:3].tolist() == [100, 101, 102]
