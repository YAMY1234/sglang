from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# fc output rows per CTA held in shared memory (embedding half, 8 KB each at H = 4096).
MAX_ROWS = 27


@cache_once
def _module(hidden: int, world: int, max_rows: int) -> Module:
    args = make_cpp_args(hidden, world, max_rows, is_arch_support_pdl())
    return load_jit(
        "mkd_draft_fc",
        *args,
        cuda_files=["speculative/mkd_draft_fc.cuh"],
        cuda_wrappers=[
            ("sum", f"MkdFc<{args}>::sum"),
            ("memo", f"MkdFc<{args}>::memo"),
        ],
    )


def mkd_fc_sum(
    emb: torch.Tensor,
    memo: torch.Tensor | None,
    hidden_partials: torch.Tensor,
    fc_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    world: int,
) -> torch.Tensor:
    """fc output [1, H] of one token: rank-order sum of the embedding-half partials
    (computed here from ``emb`` [1, H], or the memo row [1, H] fp32) and the received
    hidden-half partials [world / 2, H] fp32; bitwise equal to the row-split fc."""
    hidden = out.shape[-1]
    if memo is None:
        memo = hidden_partials.new_empty(0)
    _module(hidden, world, MAX_ROWS).sum(emb, memo, hidden_partials, fc_weight, norm_weight, out, eps)
    return out


def mkd_fc_memo(
    emb: torch.Tensor,
    fc_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    world: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Memo rows [N, H] fp32 for embedding rows [N, H]: the rank-order sum of the
    embedding-half partials the row-split fc would compute for each row."""
    _module(emb.shape[-1], world, MAX_ROWS).memo(emb, fc_weight, norm_weight, out, eps)
    return out
