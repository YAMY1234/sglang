"""Exact prefill-owned ring gathers; no recurrence or factorization changes."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gather_ring(RING, IDS, OUT, TOTAL: tl.constexpr, ROW: tl.constexpr,
                 BATCH: tl.constexpr, RINGS: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < TOTAL
    layer = offset // (BATCH * ROW)
    batch = (offset // ROW) % BATCH
    element = offset % ROW
    slot = tl.load(IDS + batch, mask=valid, other=0).to(tl.int64)
    source = (layer.to(tl.int64) * RINGS + slot) * ROW + element
    value = tl.load(RING + source, mask=valid, other=0)
    tl.store(OUT + offset, value, mask=valid)


def gather_owned_ring_layers(ring, indices, *, out=None, block=4096):
    """Copy owned, valid ring indices to an independent contiguous L/B/H/V/K stage.

    The caller already proved every requested slot owns a valid ring entry.
    Layout/dtype fallbacks keep torch.index_select semantics. Allocation and
    all arithmetic in the stored states are unchanged.
    """
    if ring.ndim != 5 or indices.ndim != 1:
        raise ValueError('expected L/R/H/V/K ring and one-dimensional ring indices')
    if block not in (1024, 4096):
        raise ValueError('unsupported gather block')
    shape = (ring.shape[0], indices.numel(), *ring.shape[2:])
    if out is None:
        out = torch.empty(shape, dtype=ring.dtype, device=ring.device)
    if tuple(out.shape) != shape or out.dtype != ring.dtype or out.device != ring.device or not out.is_contiguous():
        raise ValueError('output must be an independent contiguous matching stage')
    if ring.untyped_storage().data_ptr() == out.untyped_storage().data_ptr():
        raise ValueError('ring gather output may not alias the persistent ring')
    if not ring.is_contiguous() or ring.dtype != torch.float32 or not indices.is_contiguous():
        return torch.index_select(ring, 1, indices, out=out)
    if out.numel():
        row = ring.shape[2] * ring.shape[3] * ring.shape[4]
        _gather_ring[(triton.cdiv(out.numel(), block),)](
            ring, indices, out, out.numel(), row, indices.numel(), ring.shape[1],
            block, num_warps=4 if block == 1024 else 8)
    return out
