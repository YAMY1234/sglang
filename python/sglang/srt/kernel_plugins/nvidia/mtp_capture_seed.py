"""Source-qualified BS1 sparse-index publication into the existing seed pool."""

import os
from dataclasses import fields

import torch

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    MtpCaptureSeedIO,
    MtpCaptureSeedSignature,
    NumericOperation,
    ScalarType,
)


def prepare(indexer, model_runner):
    """Describe the real graph allocation; the provider declares supported shapes."""
    if os.environ.get("SGLANG_Q38_GLUE_MERGE", "0") != "1":
        return ()
    from sglang.srt.kernel_plugins.nvidia.numeric_binding import (
        signature_capability_supported,
    )
    from sglang.srt.layers.attention.qsa.glue import resolve_qsa_sparse_backend
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.utils import is_sm100_supported

    tokens = getattr(indexer, "_numeric_mtp_tokens_per_request", 0)
    if tokens not in getattr(indexer, "_numeric_mtp_capture_widths", ()):
        return ()
    backend = resolve_qsa_sparse_backend(getattr(model_runner, "attn_backend", None))
    state = getattr(backend, "_mtp_shared_sparse_indices", None)
    if state is None or indexer.layer_id not in state.layer_slots:
        return ()
    slot = state.layer_slots[indexer.layer_id]
    seed, captured = state.indices[slot], state.captured_len[slot]
    if not (
        type(state.trash_row) is int
        and state.trash_row >= 0
        and seed.ndim == 2
        and seed.shape[0] == state.trash_row + 1
        and seed.shape[1] > state.tail_width
        and seed.dtype == torch.int32
        and seed.stride(1) == 1
        and seed.stride(0) >= seed.shape[1]
        and seed.is_cuda
        and captured.shape == (seed.shape[0],)
        and captured.dtype == torch.int32
        and captured.is_contiguous()
        and captured.device == seed.device
    ):
        return ()
    # The graph expander owns contiguous top-k rows. Front=0 is its original
    # whole-request publication protocol; accepted dimensions belong to the spec.
    columns = seed.shape[1] - state.tail_width
    selected = MtpCaptureSeedSignature(
        NumericOperation.MTP_CAPTURE_SEED,
        tokens,
        columns,
        columns,
        seed.shape[0],
        seed.shape[1],
        seed.stride(0),
        0,
        state.trash_row,
        ScalarType.INT32,
        ScalarType.INT32,
        ScalarType.INT64,
        ScalarType.INT32,
        int(os.environ.get("SGLANG_Q38_PDL_SMALL", "0") == "1"),
    )
    if not signature_capability_supported(
        selected,
        spec=getattr(indexer, "_numeric_capability_spec", None),
        tp_size=get_parallel().tp_size,
        sm100_supported=int(is_sm100_supported()),
        is_draft_worker=int(getattr(model_runner, "is_draft_worker", False)),
        indexer_topk=indexer.token_topk,
        compress_ratio=indexer.compress_ratio,
        tail_width=state.tail_width,
        tokens_per_request=tokens,
    ):
        return ()
    backend._numeric_mtp_capture_signatures = (selected,)
    return (selected,)


def signature(io, *, front, trash, pdl):
    """Read actual borrowed views before any seed row can be written."""
    topk, seed, captured = io.topk_indices, io.seed_out, io.captured_lengths_out
    if not (
        type(front) is int
        and front >= 0
        and type(trash) is int
        and trash >= 0
        and type(pdl) is int
        and pdl in (0, 1)
        and topk.ndim == 2
        and topk.shape[0] > 0
        and topk.shape[1] > 0
        and topk.dtype == torch.int32
        and topk.stride(1) == 1
        and topk.stride(0) >= topk.shape[1]
        and seed.ndim == 2
        and seed.shape[0] == trash + 1
        and seed.dtype == torch.int32
        and seed.stride(1) == 1
        and seed.stride(0) >= seed.shape[1]
        and captured.shape == (trash + 1,)
        and captured.dtype == torch.int32
        and captured.is_contiguous()
        and io.expanded_lengths.ndim == 1
        and io.expanded_lengths.numel() >= topk.shape[0]
        and io.expanded_lengths.is_contiguous()
    ):
        return None
    integer_types = {torch.int32: ScalarType.INT32, torch.int64: ScalarType.INT64}
    scalars = (io.extend_lengths, io.accept_counts, io.requests)
    if any(t.ndim != 1 or t.numel() != 1 for t in scalars):
        return None
    types = tuple(integer_types.get(t.dtype) for t in (*scalars, io.expanded_lengths))
    if None in types or any(
        not t.is_cuda or t.device != topk.device
        for field in fields(io)
        if (t := getattr(io, field.name)) is not None
    ):
        return None
    return MtpCaptureSeedSignature(
        NumericOperation.MTP_CAPTURE_SEED,
        topk.shape[0],
        topk.shape[1],
        topk.stride(0),
        seed.shape[0],
        seed.shape[1],
        seed.stride(0),
        front,
        trash,
        *types,
        pdl,
    )


def capture_seed(backend, topk, batch, metadata, layer_id):
    """A miss preserves the original capture; a selected error never retries it."""
    if not (
        os.environ.get("SGLANG_Q38_GLUE_MERGE", "0") == "1"
        and batch.forward_mode.is_draft_extend_v2()
        and batch.batch_size == 1
        and batch._original_forward_mode is None
    ):
        return False
    candidates = getattr(backend, "_numeric_mtp_capture_signatures", ())
    state = backend._mtp_shared_sparse_indices
    if not candidates or state is None or layer_id not in state.layer_slots:
        return False
    spec = batch.spec_info
    accepted = getattr(spec, "num_accept_tokens", None)
    if accepted is None or accepted.numel() < 1 or batch.req_pool_indices.numel() < 1:
        return False
    from sglang.srt.kernel_plugins.nvidia.numeric_binding import numeric_operator

    (selected,) = candidates
    run = numeric_operator(selected, prepared=True)
    if run is None:
        return False
    slot = state.layer_slots[layer_id]
    io = MtpCaptureSeedIO(
        topk,
        backend._capture_extend_seq_lens(batch)[:1],
        accepted[:1],
        batch.req_pool_indices[:1],
        metadata.get_seqlens_expanded(),
        state.indices[slot],
        state.captured_len[slot],
    )
    actual = signature(
        io,
        front=spec.num_front_tokens,
        trash=state.trash_row,
        pdl=int(os.environ.get("SGLANG_Q38_PDL_SMALL", "0") == "1"),
    )
    if actual != selected:
        return False
    run(io)
    return True
