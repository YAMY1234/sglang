"""Eager GDN prefill state copies in one launch per family.

* ``gdn_gather_state_rows`` / ``gdn_scatter_state_rows``: the chunked kernel's
  gather and write-back of its pool rows, with ``state_pool_io``'s padding rule
  (a slot <= 0 reads zeros and writes a zero row to the sink slot 0).

Each moves the same bytes as the torch indexing it replaces.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# 16 KB of fp16 state per program; a 48 x 128 x 128 row is 96 programs.
_BLOCK_ROW = 8192
_logged: set = set()


def _log_once(key: str, msg: str, *args) -> None:
    if key not in _logged:
        _logged.add(key)
        logger.info(msg, *args)


@triton.jit
def _gdn_pool_rows_kernel(
    src,
    dst,
    slots,  # [rows] pool slots; <= 0 is padding
    row_elems,
    stride_src,
    stride_dst,
    GATHER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < row_elems
    slot = tl.load(slots + r).to(tl.int64)
    # A padded slot reads nothing (zeros) and writes to the sink slot 0.
    live = offs < tl.where(slot > 0, row_elems, 0)
    slot = tl.maximum(slot, 0)
    if GATHER:
        val = tl.load(src + slot * stride_src + offs, mask=live, other=0)
        tl.store(dst + r.to(tl.int64) * stride_dst + offs, val, mask=mask)
    else:
        val = tl.load(src + r.to(tl.int64) * stride_src + offs, mask=live, other=0)
        tl.store(
            dst + slot * stride_dst + offs, val.to(dst.dtype.element_ty), mask=mask
        )


def _row_major_tail(t: torch.Tensor) -> bool:
    """Every dim after the first is dense (one row = one contiguous span)."""
    expected = 1
    for size, stride in zip(reversed(t.shape[1:]), reversed(t.stride()[1:])):
        if size != 1 and stride != expected:
            return False
        expected *= size
    return True


def _pool_rows(src, dst, slots, gather: bool) -> None:
    rows, row_elems = slots.shape[0], (src if gather else dst)[0].numel()
    if rows:
        _gdn_pool_rows_kernel[(rows, triton.cdiv(row_elems, _BLOCK_ROW))](
            src,
            dst,
            slots,
            row_elems,
            src.stride(0),
            dst.stride(0),
            GATHER=gather,
            BLOCK=_BLOCK_ROW,
        )


def gdn_state_rows_supported(pool: torch.Tensor, slots: torch.Tensor) -> bool:
    """Whether the one-launch gather / write-back serves this pool and slot list."""
    if _row_major_tail(pool) and slots.dim() == 1 and slots.is_contiguous():
        return True
    _log_once(
        "rows-miss",
        "SGLANG_OPT_GDN_PREFILL_FUSED_STATE_COPY: stock copies (pool rows not dense "
        "or slots not a contiguous vector)",
    )
    return False


def gdn_gather_state_rows(pool: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """``gather_pool_states(pool, slots)[2]`` (zeros for slots <= 0) in one launch."""
    out = torch.empty(
        (slots.shape[0], *pool.shape[1:]), dtype=pool.dtype, device=pool.device
    )
    _pool_rows(pool, out, slots, gather=True)
    _log_once(
        "gather",
        "SGLANG_OPT_GDN_PREFILL_FUSED_STATE_COPY: chunked-GDN state rows gathered and "
        "written back by one launch each (first: %d rows of %d elements)",
        slots.shape[0],
        pool[0].numel(),
    )
    return out


def gdn_scatter_state_rows(
    pool: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor
) -> None:
    """``scatter_pool_states`` in one launch: ``pool[slot] = rows[i]`` in the pool's
    dtype, a zero row to slot 0 for slots <= 0 (rows: contiguous)."""
    _pool_rows(rows, pool, slots, gather=False)
