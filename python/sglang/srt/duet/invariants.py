"""Minimal common hooks for the future emitter_runner (docs/162 A14/norm).

Adapters merge their fused residual at the cut, then pass h_cut and a callable
embedding_lookup, never the mutable embedding tensor used by shallow layers.
The hook performs a fresh lookup after the cut and decodes the actual packed
record with its stored token IDs. The codec owns subtraction of h0, side-input
addition and quantization-aware rounding; no bypass tensor accompanies it.

Adapters choose where to apply reference_rms_norm and the input dtype: base
Lightning norms receive BF16 after the rounded residual addition; its emitters
receive FP32. The common invariant is cast-before-weight, without reassociation.
These are ordinary eager operations, not deterministic-inference controls.
"""

from __future__ import annotations

import torch


def reference_rms_norm(x, weight, eps):
    """FP32 statistics -> cast to x.dtype -> multiply weight, in this order."""
    normalized = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + eps
    )
    return weight * normalized.to(x.dtype)


def reconstruct_boundary(code, hidden, token_ids, *, embedding_lookup):
    """Return (stored record, decoded h_cut); lookup must return fresh embeddings.

    ``code`` implements encode(hidden, embeddings, ids), decode(record,
    embeddings); ``record.token_ids`` is the only decode-side ID source.
    Metadata assertions detect shape/alias errors without extra GPU reductions.
    """
    embeddings = embedding_lookup(token_ids)
    if embeddings.shape != hidden.shape or embeddings.device != hidden.device:
        raise ValueError("fresh side embedding must match boundary shape/device")
    if (
        hidden.numel()
        and embeddings.untyped_storage().data_ptr()
        == hidden.untyped_storage().data_ptr()
    ):
        raise ValueError("side embedding aliases the mutable boundary residual")
    record = code.encode(hidden, embeddings, token_ids)
    restored_embeddings = embedding_lookup(record.token_ids.long())
    if (
        restored_embeddings.shape != hidden.shape
        or restored_embeddings.device != hidden.device
    ):
        raise ValueError("stored token IDs must reconstruct the same boundary layout")
    reconstructed = code.decode(record, restored_embeddings)
    if reconstructed.shape != hidden.shape or reconstructed.dtype != hidden.dtype:
        raise ValueError("decoded boundary must preserve residual shape/dtype")
    return record, reconstructed
