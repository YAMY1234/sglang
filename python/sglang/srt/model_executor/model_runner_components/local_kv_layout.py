"""This runner's own K/V layout, resolved once and read by its consumers."""

from __future__ import annotations

import msgspec

from sglang.srt.runtime_context import get_parallel


class LocalKVLayout(msgspec.Struct, frozen=True):
    """The K/V layout this runner's model actually builds.

    Consumers read this instead of deriving from ``get_parallel()``, whose DCP
    group is process-global and shared with the target -- a draft deriving from
    it gets the target's layout.
    """

    # DCP size for the K/V layout: 1 on a draft (see resolve_local_kv_layout).
    attn_dcp_size: int


def resolve_local_kv_layout(*, is_draft_worker: bool) -> LocalKVLayout:
    """DCP shards the token dimension, so every rank in a group must project K/V
    identically -- hence a DCP-aware model replicates K/V within its group
    instead of sharding across the full attention TP width. A draft runs
    TP-sharded over the whole sequence and never splits the token dimension, so
    DCP does not apply to it.

    Layout intent only: loc-space sizing (a replicated draft pool spans the
    shared allocator's ``max_total * dcp_size``) needs the real group size and
    must keep reading it.
    """
    attn_dcp_size = 1 if is_draft_worker else get_parallel().attn_dcp_size
    return LocalKVLayout(attn_dcp_size=attn_dcp_size)
