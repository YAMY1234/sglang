"""This runner's own K/V layout, resolved once and threaded to consumers."""

from __future__ import annotations

from sglang.srt.runtime_context import get_parallel


def resolve_attn_dcp_size(*, is_draft_worker: bool) -> int:
    """DCP size for this runner's K/V layout.

    DCP shards the token dimension, so every rank in a group must project K/V
    identically -- hence a DCP-aware model replicates K/V within its group
    instead of sharding across the full attention TP width. A draft runs
    TP-sharded over the whole sequence and never splits the token dimension, so
    DCP does not apply to it.

    Consumers read the resolved value rather than ``get_parallel()``, whose DCP
    group is process-global and shared with the target. Note this is layout
    intent only: loc-space sizing (a replicated draft pool spans the shared
    allocator's ``max_total * dcp_size``) needs the real group size and must
    keep reading it.
    """
    if is_draft_worker:
        return 1
    return get_parallel().attn_dcp_size
