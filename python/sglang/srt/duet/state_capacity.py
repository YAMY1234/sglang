"""Explicit capacity guard for experimental fixed-rank period comparisons.

The production ceiling stays 32. RMAX64 is opt-in for r8/r16 W32 only;
allocation and kernels still use the next power of two of r + W.
"""
import os


def validate_factored_capacity(rank: int, period: int) -> int:
    extended = os.environ.get("SGLANG_GDN_EXPERIMENTAL_RMAX64", "0") == "1"
    if rank < 1 or period < 1:
        raise ValueError("factored recurrence requires positive rank and period")
    if rank + period > 32 and not (extended and rank in (8, 16) and period == 32):
        raise ValueError(
            "factored recurrence supports r + W <= 32; experimental r8/r16 W32 "
            "requires SGLANG_GDN_EXPERIMENTAL_RMAX64=1"
        )
    return max(16, 1 << (rank + period - 1).bit_length())
