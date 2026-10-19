"""Draft-extend padding rows (SGLANG_Q35MF_E2X_DRAFT_FUSE, E2xDraftFuse.EXTEND_PADDING).

A draft extend after verify runs a fixed number of rows per request; rows at or past the request's
accept length (bonus included) are padding whose outputs are never read. Their router logits become
the request's last real row's, so they route to no expert outside the real rows' set. Real rows keep
their own logits, routing and MoE outputs; the grouped GEMM's total M (its tactic bucket) is unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# (num_accept_tokens [bs], rows per request) while a fixed-width draft extend forward runs.
_active: Optional[Tuple[torch.Tensor, int]] = None


@contextmanager
def padding_rows(num_accept_tokens: torch.Tensor, rows_per_req: int):
    global _active
    prev, _active = _active, (num_accept_tokens, rows_per_req)
    try:
        yield
    finally:
        _active = prev


@triton.jit
def _remap_padding_rows(
    src,
    dst,
    num_accept,
    T: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    req = row // T
    accept = tl.maximum(tl.load(num_accept + req), 1)
    source = req * T + tl.minimum(row % T, accept - 1)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(
        dst + row * N + offs, tl.load(src + source * N + offs, mask=mask), mask=mask
    )


def remap_router_logits(router_logits: torch.Tensor) -> torch.Tensor:
    """router_logits with padding rows replaced, or unchanged outside a fixed-width draft extend."""
    if _active is None:
        return router_logits
    num_accept, rows_per_req = _active
    rows, experts = router_logits.shape
    if rows != num_accept.numel() * rows_per_req or not router_logits.is_contiguous():
        return router_logits
    out = torch.empty_like(router_logits)
    _remap_padding_rows[(rows,)](
        router_logits,
        out,
        num_accept,
        rows_per_req,
        experts,
        triton.next_power_of_2(experts),
        num_warps=4,
    )
    return out
