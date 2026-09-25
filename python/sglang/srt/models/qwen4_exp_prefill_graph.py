"""Breakable prefill CUDA graph support for Qwen4-Exp (SGLANG_QWEN4_PREFILL_GRAPH).

Captured segments hold the projections, MoE, hyper-connections and collectives.
The code whose work depends on the request layout runs as eager breaks against
the live batch: the QSA indexer (pending ring, compression plan, ragged top-k)
and the PLE layer (n-gram history, short-conv state). Their outputs keep the
captured bucket's row count; rows past the live token count are inert
(top-k -1, PLE zero), and the attention break ignores them.
"""

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


def _ple(ple_query, output, key, layer_index) -> None:
    from sglang.srt.models.qwen4_exp import _commit_ple_batch, _prepare_ple_batch

    forward_batch, _ = _live_batch()
    model = _owner(key)
    batch = _prepare_ple_batch(
        forward_batch.input_ids,
        forward_batch,
        ngram_size=model.ple_ngram_size,
        ngram_eos_token_id=model.ple_ngram_eos_token_id,
    )
    result = model.layers[layer_index].ple(ple_query, forward_batch, batch)
    _commit_ple_batch(batch, forward_batch)
    output.copy_(result)


def _ple_stub(ple_query, output, key, layer_index) -> None:
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
