"""SGLANG_OPT_DSV41_DSPARK_DRAFT_PREP_FORK: the draft graph's in-graph metadata prep
runs on a side stream and the first draft KV store joins it, before anything reads
the metadata it builds."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.runner_utils import forked_metadata_prep
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.fixture
def waits(monkeypatch):
    """CPU stand-ins for the CUDA stream calls; returns the events waited on."""
    waited = []
    current = SimpleNamespace(wait_event=waited.append)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(forked_metadata_prep, "_pending_join", None)
    return waited


def _stream():
    return SimpleNamespace(wait_stream=lambda s: None)


def test_fork_is_joined_exactly_once(waits):
    """An unjoined fork cannot be hidden by a second fork, and only the first join
    waits; a forward that never joins is what the runner's capture check catches."""
    done = SimpleNamespace(record=lambda: None)
    ran = []
    forked_metadata_prep.fork_in_graph_metadata_prep(
        stream=_stream(), done=done, prep=lambda: ran.append(1)
    )
    assert ran == [1] and not waits
    with pytest.raises(AssertionError):
        forked_metadata_prep.fork_in_graph_metadata_prep(
            stream=_stream(), done=done, prep=lambda: None
        )
    assert forked_metadata_prep.join_in_graph_metadata_prep()
    assert waits == [done]
    assert not forked_metadata_prep.join_in_graph_metadata_prep()


def test_kv_store_location_joins_before_the_deferred_commit_hook(monkeypatch):
    """The first draft KV store reads its location through get_swa_out_cache_loc,
    which must join the forked prep before reading the metadata it builds."""
    from sglang.srt.layers.attention import deepseek_v4_backend as dsv4

    order = []
    monkeypatch.setattr(
        dsv4, "join_in_graph_metadata_prep", lambda: order.append("prep") or True
    )

    class _Stop(Exception):
        pass

    def deferred_commit_join():
        order.append("commit")
        raise _Stop

    backend = dsv4.DeepseekV4AttnBackend.__new__(dsv4.DeepseekV4AttnBackend)
    backend.pre_kv_store_hook = deferred_commit_join
    with pytest.raises(_Stop):
        backend.get_swa_out_cache_loc(SimpleNamespace())
    assert order == ["prep", "commit"]
