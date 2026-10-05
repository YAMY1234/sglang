"""Torch byte-exact reference for descriptor-based KV staging.

Descriptors and streams are owned by the runtime. These operators only copy
specified byte intervals; neither selects a device nor creates runtime state.
"""

import torch


def _rows(tensor, descriptor):
    if tensor.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Staging requires FP16/BF16 storage")
    if str(tensor.dtype).removeprefix("torch.") != descriptor.dtype:
        raise ValueError("Staging tensor dtype differs from descriptor")
    if not tensor.is_contiguous() or tensor.ndim != 3:
        raise ValueError("Staging requires contiguous NHD storage")
    result = tensor.view(torch.uint8).reshape(tensor.shape[0], -1)
    if result.shape[1] != descriptor.row_stride_bytes:
        raise ValueError("Staging tensor stride differs from descriptor")
    return result


def _validate_indices(indices, row_count):
    if indices.dtype not in (torch.int32, torch.int64) or indices.ndim != 1:
        raise ValueError("Staging rows require a one-dimensional integer table")
    if indices.numel() and (
        indices.min().item() < 0 or indices.max().item() >= row_count
    ):
        raise ValueError("Staging row is outside registered storage")


def gather_staging(buffers, source_rows, staging, region):
    """Pack one writer region, including zeroed alignment padding."""
    if staging.dtype != torch.uint8 or staging.numel() < region.length:
        raise ValueError("Source staging buffer is too small or has wrong dtype")
    count = source_rows.numel()
    staging[: region.length].zero_()
    for copy in region.entries:
        if copy.length != count * copy.width:
            raise ValueError("Source rows differ from the staging token count")
        rows = _rows(buffers[copy.source.index], copy.source)
        _validate_indices(source_rows, rows.shape[0])
        payload = rows[
            source_rows.long(), copy.src_offset : copy.src_offset + copy.width
        ]
        staging[copy.offset : copy.offset + copy.length].copy_(payload.reshape(-1))


def scatter_staging(staging, buffers, destination_rows, plan):
    """Scatter valid token rows through the final request table, leaving holes intact."""
    if staging.dtype != torch.uint8 or staging.numel() < plan.total_bytes:
        raise ValueError("Destination staging buffer is too small or has wrong dtype")
    if destination_rows.numel() != plan.valid_tokens:
        raise ValueError("Destination rows differ from the staging token count")
    for region in plan.regions:
        for copy in region.entries:
            rows = _rows(buffers[copy.destination.index], copy.destination)
            _validate_indices(destination_rows, rows.shape[0])
            start = region.offset + copy.offset
            payload = staging[start : start + copy.length].view(
                plan.valid_tokens, copy.width
            )
            rows[
                destination_rows.long(), copy.dst_offset : copy.dst_offset + copy.width
            ] = payload
