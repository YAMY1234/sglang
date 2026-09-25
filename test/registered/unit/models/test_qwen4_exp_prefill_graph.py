"""SGLANG_QWEN4_PREFILL_GRAPH eager breaks read the live batch and keep the
captured bucket shape; padded rows stay inert."""

from types import SimpleNamespace

import torch

import sglang.srt.models.qwen4_exp as qwen4_exp
from sglang.srt.models import qwen4_exp_prefill_graph as graph
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _live(monkeypatch, real, **fields):
    batch = SimpleNamespace(global_num_token_non_padded_cpu=real, **fields)
    monkeypatch.setattr(
        graph, "get_tc_piecewise_forward_context", lambda: SimpleNamespace(forward_batch=batch)
    )
    return batch


def test_qsa_topk_uses_live_rows_and_pads_with_minus_one(monkeypatch):
    batch = _live(monkeypatch, 3)
    seen = {}

    class Layer:
        def qsa_topk_for_prefill_graph(self, hidden, positions, forward_batch):
            seen.update(rows=hidden.shape[0], positions=positions.tolist(), batch=forward_batch)
            return torch.arange(3 * 5, dtype=torch.int32).reshape(3, 5)

    layer = Layer()
    key = graph.register(layer)
    out = graph.qsa_topk(key, torch.zeros(8, 4), torch.arange(8), 5)
    assert out.shape == (8, 5) and out.dtype == torch.int32
    assert torch.equal(out[:3], torch.arange(15, dtype=torch.int32).reshape(3, 5))
    assert bool((out[3:] == -1).all())
    assert seen == {"rows": 3, "positions": [0, 1, 2], "batch": batch}


def test_qsa_topk_narrows_mrope_positions(monkeypatch):
    _live(monkeypatch, 2)

    class Layer:
        def qsa_topk_for_prefill_graph(self, hidden, positions, forward_batch):
            assert positions.shape == (3, 2)
            return torch.zeros(2, 1, dtype=torch.int32)

    layer = Layer()
    graph.qsa_topk(graph.register(layer), torch.zeros(4, 2), torch.zeros(3, 4, dtype=torch.long), 1)


def test_ple_prepares_runs_and_commits_in_order(monkeypatch):
    batch = _live(monkeypatch, 2, input_ids=torch.tensor([5, 6, 0, 0]))
    calls = []

    def prepare(input_ids, forward_batch, *, ngram_size, ngram_eos_token_id):
        calls.append(("prepare", input_ids.tolist(), ngram_size, ngram_eos_token_id))
        return "ple-batch"

    def commit(ple_batch, forward_batch):
        calls.append(("commit", ple_batch))

    monkeypatch.setattr(qwen4_exp, "_prepare_ple_batch", prepare)
    monkeypatch.setattr(qwen4_exp, "_commit_ple_batch", commit)

    def run(query, forward_batch, ple_batch):
        assert forward_batch is batch and ple_batch == "ple-batch"
        calls.append(("run",))
        return query * 2

    class Model:  # nn.Module owners are weak-referenceable; SimpleNamespace is not
        ple_ngram_size = 3
        ple_ngram_eos_token_id = 9
        layers = [SimpleNamespace(ple=None), SimpleNamespace(ple=run)]

    model = Model()
    query = torch.arange(8.0).reshape(4, 2)
    out = graph.ple(graph.register(model), 1, query)
    assert torch.equal(out, query * 2)
    assert calls == [("prepare", [5, 6, 0, 0], 3, 9), ("run",), ("commit", "ple-batch")]


def test_hc_pair_is_published_only_for_pairs(monkeypatch):
    monkeypatch.setattr(graph.envs.SGLANG_QWEN4_PREFILL_GRAPH, "get", lambda: True)
    model = SimpleNamespace(last_hc_hidden_states=None)
    hidden, streams = torch.ones(2, 3), torch.zeros(2, 12)
    assert graph.unpack_hc_pair(model, (hidden, streams)) is hidden
    assert model.last_hc_hidden_states is streams
    assert graph.unpack_hc_pair(model, hidden) is hidden


def test_breaks_are_inactive_outside_breakable_graphs():
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True))
    assert not graph.active(batch)
