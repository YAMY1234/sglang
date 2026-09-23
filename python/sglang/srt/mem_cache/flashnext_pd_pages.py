"""CPU page plans for complete-KV PD handoff from the Flash-Next arena.

Rows identify logical transfer entries, not backing pointers: K/V buffers alias
one arena, but every layer owns different physical pages. No latent goes to D.
"""
import numpy as np


def request_layer_pages(shared, deep, private_pages, shared_pages, start_token, page_size=64):
    pages = np.asarray(shared_pages, dtype=np.int32)
    if pages.ndim != 1 or start_token < 0 or start_token % page_size:
        raise ValueError("PD source range must begin at a logical page boundary")
    begin = start_token // page_size
    owned = private_pages[begin:begin + len(pages)]
    if len(owned) != len(pages):
        raise ValueError("PD range exceeds live private reservation")
    if not len(pages):
        return np.empty((12, 0), dtype=np.int32)
    # Missing owners are errors, never silently mapped to reserved page zero.
    rows = [shared[int(s)][:7] + deep[int(d)] for s, d in zip(pages, owned)]
    out = np.asarray(rows, dtype=np.int32).T.copy()
    if out.shape != (12, len(pages)) or np.any(out <= 0):
        raise ValueError("PD layer pages must be live 7+5 QSA owners")
    return out


def entry_transfer_blocks(src_ptrs, dst_ptrs, item_lens, pages_by_entry, dst_pages, pairs):
    """Return one coalesced byte-copy plan per matched logical entry."""
    pages = np.asarray(pages_by_entry, dtype=np.int32)
    dst = np.asarray(dst_pages, dtype=np.int32)
    if (pages.ndim != 2 or dst.ndim != 1
            or pages.shape != (len(src_ptrs), len(dst))
            or len(item_lens) != len(src_ptrs)):
        raise ValueError("PD entry/page dimensions do not match registered buffers")
    if np.any(pages <= 0) or np.any(dst <= 0):
        raise ValueError("PD must not transfer reserved page zero")
    plans = []
    for i, j in pairs:
        src = pages[i]
        if not len(src):
            plans.append([])
            continue
        breaks = np.flatnonzero((np.diff(src) != 1) | (np.diff(dst) != 1)) + 1
        starts = np.r_[0, breaks]
        ends = np.r_[breaks, len(src)]
        size = int(item_lens[i])
        plans.append([(int(src_ptrs[i]) + int(src[a]) * size,
                       int(dst_ptrs[j]) + int(dst[a]) * size,
                       int(b - a) * size) for a, b in zip(starts, ends)])
    return plans
