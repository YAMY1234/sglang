"""Breakable prefill CUDA graph support for Qwen4-Exp (SGLANG_QWEN4_PREFILL_GRAPH).

Captured segments hold the projections, MoE, hyper-connections and collectives.
The code whose work depends on the request layout runs as eager breaks against
the live batch: the QSA indexer (pending ring, compression plan, ragged top-k)
and the PLE layer (n-gram history, short-conv state). Their outputs keep the
captured bucket's row count; rows past the live token count are inert
(top-k -1, PLE zero), and the attention break ignores them.
"""

import os
import weakref
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (
    get_tc_piecewise_forward_context,
)

# The PLE layer prepares, runs and commits its batch inside one eager break.
PLE_IN_BREAK = object()

_OWNERS = {}


def register(module) -> int:
    """Key an eager body's owner by identity: draft and target layers can share
    a layer_id, and TwinStar emitters can own private indexers."""
    key = id(module)
    _OWNERS[key] = weakref.ref(module)
    return key


def active(forward_batch) -> bool:
    return (
        envs.SGLANG_QWEN4_PREFILL_GRAPH.get()
        and is_in_breakable_cuda_graph()
        and forward_batch.forward_mode.is_extend()
        and get_tc_piecewise_forward_context() is not None
    )


def _live_batch():
    forward_batch = get_tc_piecewise_forward_context().forward_batch
    real = forward_batch.global_num_token_non_padded_cpu
    if real is None:
        raise RuntimeError("Qwen4 prefill graph needs the live non-padded token count")
    return forward_batch, int(real)


def _owner(key):
    owner = _OWNERS[key]()
    if owner is None:
        raise RuntimeError("Qwen4 prefill graph break outlived its module")
    return owner


def _qsa_topk(hidden_states, positions, output, key) -> None:
    forward_batch, real = _live_batch()
    topk = _owner(key).qsa_topk_for_prefill_graph(
        hidden_states[:real], positions[..., :real], forward_batch
    )
    rows = topk.shape[0]
    output[:rows].copy_(topk)
    output[rows:].fill_(-1)


def _qsa_topk_stub(hidden_states, positions, output, key) -> None:
    output.fill_(-1)


qsa_topk_break = eager_on_graph(True, capture_stub=_qsa_topk_stub)(_qsa_topk)


def qsa_topk(owner_key: int, hidden_states, positions, width: int) -> torch.Tensor:
    output = torch.empty(
        (hidden_states.shape[0], width), dtype=torch.int32, device=hidden_states.device
    )
    qsa_topk_break(hidden_states, positions, output, owner_key)
    return output


# #886 diagnostic (SGLANG_P886_BORROW_PROOF=<path prefix>): may the PLE eager break borrow the shared graph pool?
# At every PLE break during capture, record the pool blocks that are live at that point (what the rest of the
# replay still needs); at the first post-capture replay, take the free runs a borrow would use
# (runner_utils.pool.find_free_graph_pool_runs) and report every overlap, plus the measured peak temporary bytes of
# one real PLE call. Recording only; serving behaviour is unchanged.
_P886 = os.environ.get("SGLANG_P886_BORROW_PROOF")
_P886_LIVE = []
_P886_DONE = [False]
_P886_MAX_TOKENS = [0]


def _p886_pool_blocks(state):
    from sglang.srt.model_executor.runner_utils.pool import get_global_graph_memory_pool

    pool = get_global_graph_memory_pool()
    if pool is None or not torch.cuda.is_available():
        return None, []
    blocks = []
    for segment in torch.cuda.memory_snapshot(pool, include_traces=False):
        for block in segment["blocks"]:
            if block["state"] == state:
                blocks.append((int(block["address"]), int(block["size"])))
    return pool, blocks


def _p886_record(ple_query, output, layer_index) -> None:
    _, live = _p886_pool_blocks("active_allocated")
    _P886_LIVE.append(dict(tokens=int(ple_query.shape[0]), layer_index=int(layer_index), live=live,
                           query=(ple_query.data_ptr(), ple_query.numel() * ple_query.element_size()),
                           output=(output.data_ptr(), output.numel() * output.element_size())))


def _p886_report(peak_bytes, live_tokens) -> None:
    import json

    from sglang.srt.model_executor.runner_utils.pool import find_free_graph_pool_runs, get_global_graph_memory_pool

    _P886_MAX_TOKENS[0] = live_tokens
    _P886_DONE[0] = live_tokens >= 32768  # keep measuring until a full 32K-row chunk has been seen
    runs = find_free_graph_pool_runs(get_global_graph_memory_pool())
    overlaps = []
    for rec in _P886_LIVE:
        spans = rec["live"] + [rec["query"], rec["output"]]
        hit = 0
        for a0, n0 in runs:
            for a1, n1 in spans:
                lo, hi = max(a0, a1), min(a0 + n0, a1 + n1)
                if hi > lo:
                    hit += hi - lo
        if hit:
            overlaps.append(dict(tokens=rec["tokens"], layer_index=rec["layer_index"], overlap_bytes=hit))
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    out = dict(breaks_recorded=len(_P886_LIVE), largest_bucket=max((r["tokens"] for r in _P886_LIVE), default=0),
               borrow_runs=len(runs), borrow_bytes=sum(n for _, n in runs),
               breaks_with_overlap=len(overlaps), overlap_bytes_max=max((o["overlap_bytes"] for o in overlaps), default=0),
               overlaps=overlaps[:20], proof_pass=not overlaps,
               ple_peak_temporary_bytes=peak_bytes, ple_live_tokens=live_tokens)
    with open(f"{_P886}.rank{rank}.json", "w") as f:
        json.dump(out, f, indent=1)


def _ple(ple_query, output, key, layer_index) -> None:
    from sglang.srt.models.qwen4_exp import _commit_ple_batch, _prepare_ple_batch

    forward_batch, real = _live_batch()
    model = _owner(key)
    # Only a real replay counts: capture_one's two eager warm-up passes also call this body (no capture or replay
    # context set), and measuring there reported before any capture-time live set was recorded (j896044).
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
        _current_capture_var,
        _current_stream_var,
    )

    in_replay = _current_stream_var.get(None) is not None and _current_capture_var.get(None) is None
    measure = (_P886 and in_replay and _P886_LIVE and not _P886_DONE[0]
               and int(real) > _P886_MAX_TOKENS[0])
    if measure:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    batch = _prepare_ple_batch(
        forward_batch.input_ids,
        forward_batch,
        ngram_size=model.ple_ngram_size,
        ngram_eos_token_id=model.ple_ngram_eos_token_id,
    )
    result = model.layers[layer_index].ple(ple_query, forward_batch, batch)
    _commit_ple_batch(batch, forward_batch)
    output.copy_(result)
    if measure:
        torch.cuda.synchronize()
        _p886_report(int(torch.cuda.max_memory_allocated() - base), int(real))


def _ple_stub(ple_query, output, key, layer_index) -> None:
    if _P886:
        _p886_record(ple_query, output, layer_index)
    output.zero_()


ple_break = eager_on_graph(True, capture_stub=_ple_stub)(_ple)


def ple(model_key: int, layer_index: int, ple_query: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(ple_query)
    ple_break(ple_query, output, model_key, layer_index)
    return output


def returns_hc_pair(forward_batch) -> bool:
    """Captured bodies return (hidden, hyper-connection streams): replay skips
    the Python side effect that would otherwise publish the streams."""
    return active(forward_batch)


def unpack_hc_pair(model, output) -> Optional[torch.Tensor]:
    if isinstance(output, tuple) and len(output) == 2 and envs.SGLANG_QWEN4_PREFILL_GRAPH.get():
        model.last_hc_hidden_states = output[1]
        return output[0]
    return output
