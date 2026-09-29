"""Breakable-prefill bridge for request-dependent QSA indexer work."""

import torch
from sglang.srt.layers.attention.qsa.glue import resolve_qsa_sparse_backend
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)


def _qsa_indexer_prefill_with_output(
    indexer,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    layer_id: int,
) -> None:
    # Capture arguments hold dummy metadata. Resolve the current replay batch and
    # metadata inside the eager break, as the DSA prefill bridges do.
    context = get_tc_piecewise_forward_context()
    if context is None or context.forward_batch is None:
        raise RuntimeError("QSA breakable prefill requires a forward context")
    forward_batch = context.forward_batch
    num_tokens = context.raw_num_tokens
    if num_tokens is None:
        num_tokens = forward_batch.extend_num_tokens
    if num_tokens is None or not 0 <= num_tokens <= hidden_states.shape[0]:
        raise ValueError(f"Invalid QSA prefill token count: {num_tokens}")
    if positions.shape[-1] < num_tokens:
        raise ValueError("QSA prefill positions are shorter than the live batch")

    backend = get_attn_backend()
    metadata = backend.get_indexer_metadata(layer_id, forward_batch)
    result = indexer._forward_impl(
        hidden_states[:num_tokens],
        positions[..., :num_tokens],
        forward_batch,
        metadata,
    )
    if (
        result.ndim != 2
        or result.shape[0] > output.shape[0]
        or result.shape[1] != output.shape[1]
    ):
        raise ValueError(
            "QSA prefill returned an unexpected top-k shape: "
            f"got {tuple(result.shape)}, capacity {tuple(output.shape)}"
        )

    # Draft-runner prefill seeds MTP sharing from the selected rows; keep its
    # request-dependent row packing in this same eager break as the indexer.
    sparse_backend = resolve_qsa_sparse_backend(backend)
    should_capture = getattr(sparse_backend, "should_capture_mtp_sparse_indices", None)
    if should_capture is not None and should_capture(forward_batch):
        sparse_backend.capture_mtp_sparse_indices(
            result, forward_batch, layer_id, metadata=metadata
        )

    # The next captured attention segment reads this stable bucket-sized buffer;
    # -1 matches DSA's sentinel for padded, deliberately unconsumed rows.
    output[: result.shape[0]].copy_(result)
    output[result.shape[0] :].fill_(-1)


bcg_qsa_indexer_prefill_with_output = eager_on_graph(True)(
    _qsa_indexer_prefill_with_output
)
