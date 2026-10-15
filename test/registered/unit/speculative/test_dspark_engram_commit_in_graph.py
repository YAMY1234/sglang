"""SGLANG_OPT_DSV41_POST_FORWARD_CUDA_GRAPH: the static verify epilogue records the
engram history commit, gated per row, and reports it only when it was captured."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.dspark_components import dspark_verify
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _epilogue(resolve):
    epi = dspark_verify.DsparkVerifyEpilogue.__new__(dspark_verify.DsparkVerifyEpilogue)
    epi.commit_ctx = SimpleNamespace(resolve_engram_hasher=resolve)
    epi.inject_gate_buf = torch.ones(1, dtype=torch.int32)
    epi.folds_engram_commit = False
    epi.static_io = {}
    epi.stride = 6
    return epi


def test_commit_passes_rows_and_gate(monkeypatch):
    calls = []
    monkeypatch.setattr(
        dspark_verify, "engram_commit_history", lambda *a, **k: calls.append(k)
    )
    epi = _epilogue(lambda: SimpleNamespace(history=torch.zeros(4, 3)))
    lens = torch.tensor([6, 0])
    gate = epi.inject_gate_buf
    assert epi._commit_engram_history(
        torch.zeros(2, 6), torch.arange(2), torch.ones(2), lens, gate
    )
    assert len(calls) == 1
    assert calls[0]["live_rows"] is lens and calls[0]["gate"] is gate


@pytest.mark.parametrize("capturing", [False, True])
def test_commit_recorded_only_under_capture(monkeypatch, capturing):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    epi = _epilogue(None)
    epi.run_static = lambda io, accept, gate: True
    fb = SimpleNamespace(
        batch_size=2,
        input_ids=torch.zeros(12, dtype=torch.int64),
        seq_lens=torch.ones(2),
        req_pool_indices=torch.arange(2),
        out_cache_loc=torch.zeros(12),
        positions=torch.zeros(12),
    )
    out = SimpleNamespace(next_token_logits=torch.zeros(12, 5), hidden_states=None)
    epi._static_epilogue(out, fb)
    # Only a captured commit lets the worker skip its eager one.
    assert epi.folds_engram_commit == capturing
    assert (2 in epi.static_io) == capturing


@pytest.mark.parametrize(
    "resolve", [None, lambda: None, lambda: SimpleNamespace(history=None)]
)
def test_no_hasher_records_nothing(monkeypatch, resolve):
    calls = []
    monkeypatch.setattr(
        dspark_verify, "engram_commit_history", lambda *a, **k: calls.append(k)
    )
    epi = _epilogue(resolve)
    assert not epi._commit_engram_history(None, None, None, None, None)
    assert not calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gated_kernel_skips_dead_rows_and_closed_gate():
    from sglang.kernels.ops.embeddings.engram_hash import engram_commit_history

    dev = "cuda"
    history = torch.arange(4 * 3, dtype=torch.int32, device=dev).view(4, 3)
    tokens = torch.arange(100, 112, dtype=torch.int64, device=dev).view(2, 6)
    slots = torch.tensor([1, 3], device=dev)
    commit = torch.tensor([2, 4], dtype=torch.int32, device=dev)
    ungated = history.clone()
    engram_commit_history(ungated, tokens, slots, commit)

    gated = history.clone()
    live = torch.tensor([6, 0], device=dev)  # row 1 is a padded capture row
    gate = torch.ones(1, dtype=torch.int32, device=dev)
    engram_commit_history(gated, tokens, slots, commit, live_rows=live, gate=gate)
    assert torch.equal(gated[1], ungated[1])
    assert torch.equal(gated[3], history[3])

    closed = history.clone()
    engram_commit_history(
        closed, tokens, slots, commit, live_rows=torch.full_like(live, 6), gate=gate * 0
    )
    assert torch.equal(closed, history)
