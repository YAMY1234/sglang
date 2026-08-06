"""Parallelism scoping for draft-model work.

Kept free of heavy imports so memory-pool and attention code can use it without
pulling in the speculative worker stack.
"""

from __future__ import annotations

from contextlib import contextmanager

from sglang.srt.runtime_context import get_parallel


@contextmanager
def draft_dcp_context():
    # Drafts are TP-sharded over the whole sequence and never split the token
    # dimension, but the DCP topology is process-global and shared with the
    # target. Read unscoped, it makes draft code derive a K/V layout the draft
    # model never builds. Only attn_dcp_* is neutralized; the raw dcp_size /
    # dcp_rank describe the real group and target-side DCP code reads them.
    parallel = get_parallel()
    if parallel.attn_dcp_size == 1:
        yield
        return
    with parallel.override(dcp_enabled=False, attn_dcp_size=1, attn_dcp_rank=0):
        yield


def assert_draft_dcp_neutralized(where: str) -> None:
    # draft_dcp_context is scoped, so a missed call site degrades silently back
    # to the target's DCP size -- fail closed instead.
    attn_dcp_size = get_parallel().attn_dcp_size
    if attn_dcp_size != 1:
        raise AssertionError(
            f"{where}: draft work running with attn_dcp_size={attn_dcp_size}; "
            "drafts never participate in DCP. This path is missing a "
            "draft_dcp_context() scope."
        )
