# SPDX-License-Identifier: Apache-2.0
"""CPU differential test of actual call-site methods vs frozen rebased parent 4253.

AST isolation avoids initializing a distributed/CUDA backend just to exercise IO.
Both versions run the same tensor-writing entry and the same instrumented readers.
The oracle is the unmodified rebased parent source, not a second implementation of the path.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
from typing import Any, Callable, Optional

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang/srt/speculative/eagle_megakernel.py"
ORACLE = Path(__file__).parent / "fixtures/eagle_megakernel_4253.py"


class Dense:
    use_sliding_window_kv_pool = False


class Linear:
    pass


class Hybrid:
    def __init__(self, full):
        self.full_attn_backend, self.linear_attn_backend = full, Linear()


class View:
    def __init__(self, backend):
        self.metadata = backend.forward_metadata
        self.kv_pool = PoolView(backend.token_to_kv_pool)


class PoolView:
    def __init__(self, pool):
        self.pool = pool


@pytest.fixture(autouse=True)
def backend_types(monkeypatch):
    for path, name, value in [
        ("hybrid_linear_attn_backend", "HybridLinearAttnBackend", Hybrid),
        ("linear.gdn_backend", "GDNAttnBackend", Linear),
        ("trtllm_mha_backend", "TRTLLMHAAttnBackend", Dense),
    ]:
        module = ModuleType("sglang.srt.layers.attention." + path)
        setattr(module, name, value)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    module = ModuleType("sglang.srt.layers.logits_processor")
    module.LogitsProcessorOutput = NS
    monkeypatch.setitem(sys.modules, module.__name__, module)


def load(path, trace, current, alias=False):
    tree = ast.parse(path.read_text())
    # Definitions only, plus postponed annotations: no model imports or mock kernels.
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body += [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    def split(backend):
        return (backend.full_attn_backend, backend.linear_attn_backend) if isinstance(backend, Hybrid) else (backend, None)
    def metadata(backend):
        trace.append("attn_step_metadata")
        return split(backend)[0].metadata
    def kv(**kwargs):
        trace.append("kv_pool_buffers")
        return kwargs["pool"]
    def active():
        trace.append("get_attn_backend")
        return current[0]
    env = dict(Any=Any, Callable=Callable, Optional=Optional, SimpleNamespace=NS,
               torch=torch, split_backend=split, attn_step_metadata=metadata,
               kv_pool_buffers=kv, get_attn_backend=active, ForwardView=View,
               KvPoolView=PoolView, EagleDraftLoopIO=NS, EagleDraftExtendIO=NS,
               envs=NS(SGLANG_PERFLL_STEP5A=NS(get=lambda: alias)),
               logger=NS(info=lambda *args: trace.append("alias_log")))
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), env)
    return env


def exercise(path, *, generic=False, mrope=False, alias=False):
    trace, current = [], [None]
    env = load(path, trace, current, alias=alias)
    pool = object()
    if generic:
        backend = NS(forward_metadata=None, token_to_kv_pool=pool, forward=lambda: None)
        # Deliberately no dense fields, no data_type and no scalar-scale reader.
    else:
        backend = Dense()
        backend.metadata = NS(page_table=torch.arange(4), cache_seqlens=torch.tensor([8]))
    current[0] = backend
    cfg = NS(hidden_size=8, dtype=torch.bfloat16, model_is_mrope=mrope)
    model = NS(input_embeds_hidden_size=32 if generic else 8)
    obj = env["EagleMegakernel"].__new__(env["EagleMegakernel"])
    obj._draft_config, obj._draft_runner = cfg, NS(model=model, attn_backend=backend, token_to_kv_pool=pool)
    obj._draft_kv = None
    obj._draft_worker = NS(_topk1_parents_prealloc=torch.arange(4), _topk1_score_indices_prealloc=torch.arange(4)+7)
    obj._draft_worker.cuda_graph_runner = NS(buffers=NS(
        hidden_states=torch.zeros((2, model.input_embeds_hidden_size), dtype=torch.bfloat16),
        topk_index=torch.zeros((2, 1), dtype=torch.int64),
        topk_p=torch.ones((2, 1), dtype=torch.float32)))
    obj._extend_graphs, obj._draft_graphs = {}, {}
    obj._draft_extend = None
    captured = []
    def factory(**kwargs):
        trace.append("draft_extend_factory")
        captured.append(kwargs)
        def step(io):
            trace.append("extend_step")
            if generic:
                assert isinstance(io.attn, View) and io.attn.metadata is None
                assert io.kv.pool is pool
                assert io.extend_seq_lens is lengths
                assert io.num_front_tokens == 1
            # Same real BF16 tensor arithmetic in both versions.
            io.hidden_out.copy_(io.hidden_states.index_select(0, io.select_index))
            io.topk_index_out.copy_(io.input_ids.index_select(0, io.select_index).view(-1, 1))
            io.topk_p_out.fill_(1.)
        return lambda bs: step
    obj._draft_extend = factory
    env["megakernel_agree"] = lambda value, **kw: trace.append("agree")
    runner = NS(max_bs=2, device="cpu", captured_req_width=4, capture_bs=[1, 2], speculative_num_steps=3,
                draft_attn_backend=NS(attn_backends=(backend, backend)))
    obj.bind_extend_graphs(runner)
    if alias and not generic:
        aliased = obj._extend_graphs[id(runner)].out
        buffers = obj._draft_worker.cuda_graph_runner.buffers
        assert aliased.hidden.data_ptr() == buffers.hidden_states.data_ptr()
        assert aliased.topk_index.data_ptr() == buffers.topk_index.data_ptr()
        assert aliased.topk_p.data_ptr() == buffers.topk_p.data_ptr()
    width = model.input_embeds_hidden_size
    lengths = torch.tensor([4, 4], dtype=torch.int32)
    spec = NS(hidden_states=torch.arange(8*width, dtype=torch.float32).view(8, width).bfloat16(),
              select_index=torch.tensor([2, 7]), num_accept_tokens=torch.tensor([3, 4]),
              topk_index=torch.tensor([[10], [20]]), extend_seq_lens_tensor=lengths, num_front_tokens=1)
    batch = NS(batch_size=2, spec_info=spec, positions=torch.arange(8), mrope_positions=torch.arange(24).view(3, 8),
               input_ids=torch.arange(8)+40, out_cache_loc=torch.arange(8)+100,
               req_pool_indices=torch.tensor([1, 2]), seq_lens=torch.tensor([8, 8]), extend_seq_lens=lengths)
    out = obj.run_draft_extend(runner=runner, forward_batch=batch)
    def loop(io):
        trace.append("loop_step")
        if generic:
            assert all(isinstance(x, View) for x in io.steps)
            assert io.kv.pool is pool
        io.draft_tokens_out.copy_(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    obj._draft_graphs[id(runner)] = NS(steps={2: loop}, draft_tokens=torch.empty((2, 3), dtype=torch.int64))
    loop_out = obj.run_draft_loop(runner=runner, forward_batch=batch)
    result = [out.hidden_states.clone(), *[x.clone() for x in loop_out[:3]],
              obj._extend_graphs[id(runner)].out.topk_index.clone(), obj._extend_graphs[id(runner)].out.topk_p.clone()]
    return trace, result, env, obj, pool


@pytest.mark.parametrize("mrope", [False, True])
@pytest.mark.parametrize("alias", [False, True])
def test_dense_sequence_and_bits_unchanged(mrope, alias):
    old, expected, *_ = exercise(ORACLE, mrope=mrope, alias=alias)
    new, actual, *_ = exercise(SOURCE, mrope=mrope, alias=alias)
    assert old == new
    for a, b in zip(expected, actual):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def test_generic_no_dense_metadata_and_wide_hidden():
    _, actual, env, obj, pool = exercise(SOURCE, generic=True)
    assert actual[0].shape == (2, 32)
    assert env["_backend_reasons"](target=Hybrid(NS(forward_metadata=None, token_to_kv_pool=pool, forward=lambda: None)),
                                     draft=(NS(forward_metadata=None, token_to_kv_pool=pool, forward=lambda: None),)) == []
    obj._draft_runner.token_to_kv_pool = object()
    with pytest.raises(RuntimeError, match="reallocated"):
        obj._kv()


def test_generic_lengths_fallback_preserves_tensor_identity():
    trace, current = [], [None]
    env = load(SOURCE, trace, current)
    backend = NS(forward_metadata=object(), token_to_kv_pool=object(), forward=lambda: None)
    lengths = torch.tensor([4])
    view = env["_draft_extend_view"](backend, NS(spec_info=NS(num_front_tokens=0), extend_seq_lens=lengths))
    assert view["extend_seq_lens"] is lengths and view["num_front_tokens"] == 0


def test_non_view_backend_still_refused():
    env = load(SOURCE, [], [None])
    assert env["_backend_reasons"](target=Hybrid(Dense()), draft=(object(),))


def test_entry_gating_and_unrelated_dispatch_unchanged():
    # Models offering no draft entries (including the existing generic dense
    # consumer) still bypass this adapter; do not simulate them as an Eagle model.
    def method(path, name):
        return ast.dump(next(n for n in ast.parse(path.read_text()).body
                             if isinstance(n, ast.FunctionDef) and n.name == name))
    for name in ["_offers_eagle_entries", "init_eagle_megakernel", "_unservable_reasons"]:
        assert method(SOURCE, name) == method(ORACLE, name)
    import hashlib
    binding = ROOT / "python/sglang/srt/kernel_plugins/nvidia/binding.py"
    assert hashlib.sha256(binding.read_bytes()).hexdigest() == "0a5791b8fad9ab53db6c0db3aff1b12afab3c691bf485775eacb64705fce453d"
