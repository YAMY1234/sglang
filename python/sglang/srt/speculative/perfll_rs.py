"""PERF-LL remove-stages D2 (SGLANG_PERFLL_RS_D2): a draft-side graph's attention metadata on a branch.

The draft extend and draft graphs open with the attention metadata update (page table, sequence lengths),
and everything else in the graph waited for it. With the switch, the update runs on a side stream forked at
the start of the captured body, beside the embedding and projections that do not read it, and the capture
stream joins that branch where the metadata is first read: a field read of the AttnStepMetadata the
megakernel binding hands out (eagle_binding.attn_step_metadata), the TRT-LLM MHA backend's forward, or else
the end of the body. Same kernels and values; only the order between them changes.

V3 (SGLANG_PERFLL_RS_V3), T (SGLANG_PERFLL_RS_T) and the same-service A/B of the remove-stages
features: a label of SGLANG_TRTL17_PAD_AB holding "<feature>off" ("+"-joined, e.g. "v3off+toff")
turns those features off for its rounds; a graph runner whose graphs take a feature the labels
turn off also captures each of them with it off ("perfll-<feature>off").
"""

from __future__ import annotations

import contextlib
import logging
from typing import Callable, Iterator, Optional, TypeVar

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

V3 = envs.SGLANG_PERFLL_RS_V3.get()
T = envs.SGLANG_PERFLL_RS_T.get()
# the features each A/B label turns off
_AB_OFF = {
    label: frozenset(p[: -len("off")] for p in label.split("+") if p.endswith("off"))
    for label in envs.SGLANG_TRTL17_PAD_AB.get()
}
_AB_LABELS = frozenset(p for off in _AB_OFF.values() for p in (f"{f}off" for f in off))
# the feature the target verify graphs take and the A/B turns off: they are also captured without it
AB_VERIFY_FEATURE: Optional[str] = "v3" if V3 and "v3off" in _AB_LABELS else None
# the features the graph being captured leaves off; None outside capture_variant
_capturing_off: Optional[frozenset] = None


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


def ab_off(feature: str) -> bool:
    """Whether this round's same-service A/B label turns ``feature`` off."""
    if f"{feature}off" not in _AB_LABELS:
        return False
    from sglang.srt.model_executor.runner_backend import ab_variants

    return feature in _AB_OFF.get(ab_variants.current(), frozenset())


def off_label(feature: str) -> str:
    """The graph variant label of the graphs captured with ``feature`` off."""
    return f"perfll-{feature}off"


@contextlib.contextmanager
def capture_variant(off: frozenset) -> Iterator[None]:
    """Capture (warmups included) with the features in ``off`` left off."""
    global _capturing_off
    prev, _capturing_off = _capturing_off, off
    try:
        yield
    finally:
        _capturing_off = prev


def feature_on(feature: str) -> bool:
    """Whether a forward runs ``feature`` (once its switch is on): as the variant being
    captured has it, else as this round's A/B label says."""
    if _capturing_off is not None:
        return feature not in _capturing_off
    return not ab_off(feature)
