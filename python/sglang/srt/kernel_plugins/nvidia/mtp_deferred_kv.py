"""Original TP4 BS1 deferred lookup, consumed only by a complete KV capability."""

import os
from dataclasses import dataclass, fields

import torch

from sglang.kernels.plugin_kernels.nvidia.numeric import (
    MtpDeferredKvIO,
    MtpDeferredKvSignature,
    NumericOperation,
    ScalarType,
)


@dataclass(frozen=True)
class DeferredMtpSparseIndices:
    """Borrowed selection and dynamic tail, resolved by the native KV packer."""

    indices: torch.Tensor
    captured_len: torch.Tensor
    requests: torch.Tensor
    positions: torch.Tensor
    tail_offsets: torch.Tensor

    @property
    def shape(self):
        return (self.requests.numel(), self.indices.shape[1])

    @property
    def tail_width(self):
        return self.tail_offsets.numel()

    def record_stream(self, stream):
        for tensor in (
            self.indices,
            self.captured_len,
            self.requests,
            self.positions,
            self.tail_offsets,
        ):
            tensor.record_stream(stream)

    def materialize(self):
        rows = self.requests.to(torch.long)
        out = self.indices[rows]
        base = self.captured_len[rows].to(torch.int64)
        tail = base.unsqueeze(1) + self.tail_offsets.unsqueeze(0)
        valid = tail <= self.positions.to(torch.int64).unsqueeze(1)
        out[:, -self.tail_width :] = torch.where(valid, tail, -1).to(out.dtype)
        return out


def _enabled():
    return (
        os.environ.get("SGLANG_Q38_LOOKUP_NATIVE", "0") == "1"
        and os.environ.get("SGLANG_Q38_MK_MTP", "0") == "1"
    )


def prepare(indexer, runner):
    """Describe allocated state and source facts without selecting a model shape."""
    widths = getattr(indexer, "_numeric_mtp_deferred_widths", ())
    if not _enabled() or not widths:
        return ()
    from sglang.srt.kernel_plugins.nvidia.numeric_binding import (
        signature_capability_supported,
    )
    from sglang.srt.runtime_context import get_parallel, get_spec
    from sglang.srt.utils import is_sm100_supported

    config = getattr(runner.model_config, "hf_text_config", None)
    if config is None:
        config = getattr(runner.model_config, "hf_config", None)
    dimensions = tuple(
        getattr(config, name, None)
        for name in ("hidden_size", "hc_count", "num_attention_heads")
    )
    if any(type(value) is not int for value in dimensions):
        return ()
    hidden, streams, heads = dimensions
    parallel, spec = get_parallel(), get_spec()
    if parallel.tp_size <= 0 or heads % parallel.tp_size:
        return ()
    signatures = []
    backends = getattr(getattr(runner, "draft_attn_backend", None), "attn_backends", ())
    for backend in backends:
        state = backend._mtp_shared_sparse_indices
        if (
            not backend._fused_kv_pool_eligible
            or state is None
            or indexer.layer_id not in state.layer_slots
        ):
            continue
        slot = state.layer_slots[indexer.layer_id]
        indices, captured = state.indices[slot], state.captured_len[slot]
        cache = backend.token_to_kv_pool.layer_router.read(indexer.layer_id)
        kc, vc = cache.key(), cache.value()
        table = backend.req_to_token_pool.req_to_token
        page_size = backend.sparse_decode_page_size
        if not (
            kc.ndim == 3
            and all(size > 0 for size in kc.shape)
            and vc.shape == kc.shape
            and kc.dtype == vc.dtype == torch.bfloat16
            and kc.is_contiguous()
            and vc.is_contiguous()
            and indices.ndim == 2
            and indices.shape[1] > 0
            and indices.dtype == captured.dtype == torch.int32
            and indices.is_contiguous()
            and captured.shape == (indices.shape[0],)
            and captured.is_contiguous()
            and table.ndim == 2
            and table.shape[0] > 0
            and table.shape[1] > 0
            and table.stride(1) == 1
            and table.stride(0) > 0
            and type(page_size) is int
            and page_size > 0
            and table.dtype in (torch.int32, torch.int64)
            and all(
                t.is_cuda and t.device == kc.device
                for t in (kc, vc, indices, captured, table)
            )
        ):
            continue
        packed_columns = (indices.shape[1] + page_size - 1) // page_size * page_size
        # Fused QK's dense head allocation and backend reshape give the same
        # inner-row pitch; actual borrowed IO must equal the declaration later.
        row_stride = kc.shape[1] * kc.shape[2]
        for rows in widths:
            selected = MtpDeferredKvSignature(
                NumericOperation.MTP_DEFERRED_KV_PREPARE,
                rows,
                kc.shape[1],
                kc.shape[2],
                ScalarType.BF16,
                indices.shape[1],
                packed_columns,
                state.tail_width,
                indices.shape[0],
                kc.shape[0],
                table.shape[0],
                table.shape[1],
                table.stride(0),
                row_stride,
                row_stride,
                17 | (8 if table.dtype == torch.int64 else 0),
            )
            if not signature_capability_supported(
                selected,
                spec=getattr(indexer, "_numeric_capability_spec", None),
                tp_size=parallel.tp_size,
                sm100_supported=int(is_sm100_supported()),
                is_draft_worker=int(runner.is_draft_worker),
                hidden_size=hidden,
                hc_count=streams,
                speculative_num_steps=spec.speculative_num_steps,
                speculative_eagle_topk=spec.speculative_eagle_topk,
                indexer_topk=indexer.token_topk,
                compress_ratio=indexer.compress_ratio,
                query_heads=heads // parallel.tp_size,
                tokens_per_request=indexer._numeric_mtp_tokens_per_request,
            ):
                continue
            declarations = getattr(backend, "_numeric_mtp_deferred_signatures", {})
            declarations[indexer.layer_id] = selected
            backend._numeric_mtp_deferred_signatures = declarations
            signatures.append(selected)
    return tuple(dict.fromkeys(signatures))


def lookup(backend, batch, positions, layer_id):
    """Leave ordinary lookup intact unless the exact downstream capability exists."""
    selected = getattr(backend, "_numeric_mtp_deferred_signatures", {}).get(layer_id)
    if not (
        _enabled()
        and selected is not None
        and batch.forward_mode.is_decode()
        and batch.batch_size == selected.rows
        and batch.req_pool_indices.numel() == positions.numel() == selected.rows
    ):
        return None
    from sglang.srt.kernel_plugins.nvidia.numeric_binding import numeric_operator

    if numeric_operator(selected, prepared=True) is None:
        return None
    state = backend._mtp_shared_sparse_indices
    slot = state.layer_slots[layer_id]
    return DeferredMtpSparseIndices(
        state.indices[slot],
        state.captured_len[slot],
        batch.req_pool_indices,
        positions,
        state._tail_offsets,
    )


def signature(io, *, tail_width):
    """Exact borrowed metadata; no device values are read and no view is copied."""
    if any(
        not isinstance(t, torch.Tensor)
        or not t.is_cuda
        or t.device != io.k_cache.device
        for f in fields(io)
        for t in (getattr(io, f.name),)
    ):
        return None
    kc, vc, nk, nv = io.k_cache, io.v_cache, io.new_k, io.new_v
    indices, captured, table = io.indices_pool, io.captured_lengths, io.req_to_token
    if not (
        kc.ndim == 3
        and all(size > 0 for size in kc.shape)
        and vc.shape == kc.shape
        and kc.is_contiguous()
        and vc.is_contiguous()
        and nk.ndim == 3
        and nk.shape == nv.shape
        and nk.shape[1:] == kc.shape[1:]
        and nk.stride() == nv.stride()
        and nk.stride(1) == nk.shape[2]
        and nk.stride(2) == 1
        and indices.ndim == 2
        and indices.shape[1] > 0
        and indices.dtype == torch.int32
        and indices.is_contiguous()
        and captured.shape == (indices.shape[0],)
        and captured.dtype == torch.int32
        and captured.is_contiguous()
        and table.ndim == 2
        and table.stride(1) == 1
        and table.shape[0] > 0
        and table.shape[1] > 0
        and table.stride(0) > 0
        and io.counts_out.shape == (nk.shape[0],)
        and io.counts_out.dtype == torch.int32
        and io.counts_out.is_contiguous()
        and all(
            t.dtype == torch.bfloat16
            for t in (kc, vc, nk, nv, io.packed_k_out, io.packed_v_out)
        )
        and io.packed_v_out.shape == io.packed_k_out.shape
        and all(
            t.ndim == 3 and t.shape[1:] == kc.shape[1:] and t.is_contiguous()
            for t in (io.packed_k_out, io.packed_v_out)
        )
    ):
        return None
    metadata = (
        io.locations,
        io.row_requests,
        io.sequence_lengths,
        table,
        io.lookup_requests,
        io.lookup_positions,
    )
    if any(t.dtype not in (torch.int32, torch.int64) for t in metadata):
        return None
    if any(
        t.shape != (nk.shape[0],) or t.stride(0) <= 0
        for t in metadata
        if t is not table
    ):
        return None
    mask = sum((int(t.dtype == torch.int64) << bit) for bit, t in enumerate(metadata))
    return MtpDeferredKvSignature(
        NumericOperation.MTP_DEFERRED_KV_PREPARE,
        nk.shape[0],
        kc.shape[1],
        kc.shape[2],
        ScalarType.BF16,
        indices.shape[1],
        io.packed_k_out.shape[0],
        tail_width,
        indices.shape[0],
        kc.shape[0],
        table.shape[0],
        table.shape[1],
        table.stride(0),
        nk.stride(0),
        nv.stride(0),
        mask,
    )


def bind_prepare(backend, q, k, v, layer, batch, indices):
    """Prepare the complete packer before its first KV write, or leave stock whole."""
    selected = getattr(backend, "_numeric_mtp_deferred_signatures", {}).get(
        layer.layer_id
    )
    if not (
        _enabled()
        and selected is not None
        and batch.forward_mode.is_decode()
        and batch.batch_size == selected.rows
        and indices.shape == (selected.rows, selected.index_columns)
        and indices.tail_width == selected.tail_width
        and q.is_cuda
        and q.dtype == torch.bfloat16
        and q.shape[0] == selected.rows
        and q.numel() == selected.rows * layer.tp_q_head_num * selected.head_dim
        and q.is_contiguous()
        and layer.head_dim == selected.head_dim
        and all(
            t is not None
            and t.shape[0] == selected.rows
            and t.numel() == selected.rows * selected.kv_heads * selected.head_dim
            and t.stride(-1) == 1
            and t.dtype == torch.bfloat16
            for t in (k, v)
        )
    ):
        return None
    from sglang.srt.kernel_plugins.nvidia.numeric_binding import numeric_operator

    run = numeric_operator(selected, prepared=True)
    decode = backend.sparse_decode_kernel()
    if run is None or decode is None:
        return None
    metadata = backend._resolve_metadata(batch)
    counts = (
        metadata.fa2_valid_counts
        if metadata.is_cuda_graph
        else torch.empty(
            selected.rows,
            dtype=torch.int32,
            device=q.device,
        )
    )
    if counts is None:
        return None
    cache = backend.token_to_kv_pool.layer_router.read(layer.layer_id)
    kc, vc = cache.key(), cache.value()
    if q.device != kc.device:
        return None
    capacity = (
        max(selected.rows, backend._cuda_graph_max_tokens)
        if metadata.is_cuda_graph
        else selected.rows
    )
    packed_k, packed_v = backend._get_fa2_scratch(
        capacity * selected.packed_columns,
        selected.kv_heads,
        selected.head_dim,
        kc.dtype,
        kc.device,
    )
    requests = metadata.row_req_pool_indices
    if requests is None:
        requests = batch.req_pool_indices
    io = MtpDeferredKvIO(
        kc,
        vc,
        k.reshape(selected.rows, selected.kv_heads, selected.head_dim),
        v.reshape(selected.rows, selected.kv_heads, selected.head_dim),
        batch.out_cache_loc,
        backend.req_to_token_pool.req_to_token,
        requests,
        metadata.sequence_lengths,
        indices.indices,
        indices.requests,
        indices.captured_len,
        indices.positions,
        counts,
        packed_k[: selected.rows * selected.packed_columns],
        packed_v[: selected.rows * selected.packed_columns],
    )
    if signature(io, tail_width=indices.tail_width) != selected:
        return None
    return run, io, metadata, decode
