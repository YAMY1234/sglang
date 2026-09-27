"""Prefill-graph hooks imported by the twinstar_sgl Qwen4Exp overlay (emitter QSA-indexer and P-trunk paths).

The prefill CUDA graph itself lives on the prefill-graph line (#287, pfgraph forks) and is not part of this branch;
here it is never active, so every overlay call site takes its non-graph path -- the same path the overlay takes on a
pfgraph fork when the recipe resolves ``prefill.backend='disabled'`` (the docs/105 / docs/120 serving recipe).
"""

PLE_IN_BREAK = None


def active(forward_batch) -> bool:
    return False


def register(owner):
    raise RuntimeError("prefill CUDA graph is not available on this branch")


def _live_batch():
    raise RuntimeError("prefill CUDA graph is not available on this branch")


def _owner(key):
    raise RuntimeError("prefill CUDA graph is not available on this branch")
