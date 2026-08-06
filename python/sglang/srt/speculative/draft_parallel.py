"""Parallelism scoping for draft-model work.

Kept free of heavy imports so memory-pool and attention code can use it without
pulling in the speculative worker stack.
"""

from __future__ import annotations

from contextlib import contextmanager

from sglang.srt.runtime_context import get_parallel


@contextmanager
def draft_dcp_context():
    # Drafts are TP-sharded, but the DCP topology is process-global and shared
    # with the target, so unscoped reads give draft code the target's layout.
    # NOTE(kpham-sgl): attn_dcp_* is attention layout intent -- neutralized here.
    # dcp_size / dcp_rank / dcp_enabled describe the real group and stay intact:
    # a replicated draft pool spans the shared allocator's virtual loc space of
    # max_total * dcp_size, so it needs the true size (see loc_space_scale).
    parallel = get_parallel()
    if parallel.attn_dcp_size == 1:
        yield
        return
    with parallel.override(attn_dcp_size=1, attn_dcp_rank=0):
        yield
