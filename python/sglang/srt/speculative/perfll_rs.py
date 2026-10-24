"""PERF-LL remove-stages D2 (SGLANG_PERFLL_RS_D2): a draft-side graph's attention metadata on a branch.

The draft extend and draft graphs open with the attention metadata update (page table, sequence lengths),
and everything else in the graph waited for it. With the switch, the update runs on a side stream forked at
the start of the captured body, beside the embedding and projections that do not read it, and the capture
stream joins that branch where the metadata is first read: a field read of the AttnStepMetadata the
megakernel binding hands out (eagle_binding.attn_step_metadata), the TRT-LLM MHA backend's forward, or else
the end of the body. Same kernels and values; only the order between them changes.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, TypeVar

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

T = TypeVar("T")

# the graphs whose metadata takes a branch: "extend" (draft extend), "draft"
_GRAPHS = frozenset(envs.SGLANG_PERFLL_RS_D2.get())
_side: Optional[torch.cuda.Stream] = None
# (stream that joins, what the graph is) while a branch is open
_pending: Optional[tuple] = None
# what each graph's branch joined at, logged once per graph
_joined_at: dict[str, str] = {}


def enabled(graph: str) -> bool:
    return graph in _GRAPHS


def run_with_metadata_branch(
    graph: str, *, metadata: Callable[[], None], body: Callable[[], T], what: str
) -> T:
    """``metadata()`` then ``body()``; for a graph in SGLANG_PERFLL_RS_D2, the metadata on a branch the
    body joins at its first metadata read (join_metadata) or at its end."""
    global _side, _pending
    if graph not in _GRAPHS:
        metadata()
        return body()
    main = torch.cuda.current_stream()
    if _side is None:
        _side = torch.cuda.Stream()
    _side.wait_stream(main)
    with torch.cuda.stream(_side):
        metadata()
    _pending = (main, what)
    try:
        return body()
    finally:
        join_metadata("the end of the graph body")


def join_metadata(reader: str) -> None:
    """Make the stream that forked the open branch, if any, wait for it; ``reader`` names the site."""
    global _pending
    if _pending is None:
        return
    main, what = _pending
    _pending = None
    main.wait_stream(_side)
    if _joined_at.get(what) != reader:
        _joined_at[what] = reader
        logger.info(
            "PERF-LL rs-d2: the %s joins its metadata branch at %s", what, reader
        )


def branch_open() -> bool:
    return _pending is not None
