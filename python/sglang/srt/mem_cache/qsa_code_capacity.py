"""Byte accounting for the fixed x256 layout (per tensor-parallel rank)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QSACodeCapacity:
    layers: int
    heads: int
    exact_fraction: float = 0.25
    page_size: int = 64

    def __post_init__(self):
        if self.layers <= 0 or self.heads not in (1, 2) or self.page_size != 64:
            raise ValueError(
                "x256 capacity requires local QSA layers, TP1/2 and page64"
            )
        if not 0 < self.exact_fraction < 1:
            raise ValueError("Exact physical fraction must lie in (0,1)")

    def allocation(self, tokens, request_slots):
        if tokens < 0 or tokens % self.page_size or request_slots < 1:
            raise ValueError("Page-aligned tokens and positive request slots required")
        pages = tokens // self.page_size
        exact_pages = int(pages * self.exact_fraction)
        code_pages = pages - exact_pages
        return {
            "virtual_token_capacity": tokens,
            "exact_page_capacity": exact_pages,
            "code_page_capacity": code_pages,
            "exact": (exact_pages + 1)
            * self.page_size
            * self.layers
            * self.heads
            * 1024,
            "code": (code_pages + 1) * self.page_size * self.layers * self.heads * 640,
            "page_tables": (pages + 1) * 12,
            "weights": self.layers * self.heads * 280320,
            "prefix_length_table": request_slots * 4,
            "indexer_compressed": (tokens + self.page_size) * self.layers * 64,
            "indexer_pending": request_slots * 4 * (self.layers * 256 + 24),
        }

    def total_bytes(self, tokens, request_slots):
        return sum(
            v
            for k, v in self.allocation(tokens, request_slots).items()
            if not k.endswith("capacity")
        )

    def from_budget(self, budget, request_slots, *, reserved_bytes=0):
        # Search exact integer page counts, including the fraction's floor and
        # both physical sentinels; do not hide them in an average B/token label.
        lo, hi = (
            0,
            max(0, budget // (self.page_size * self.layers * (640 * self.heads + 64)))
            + 1,
        )
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if (
                self.total_bytes(mid * self.page_size, request_slots) + reserved_bytes
                <= budget
            ):
                lo = mid
            else:
                hi = mid
        return lo * self.page_size


def qsa_read_workspace_bytes(
    queries, topk, query_heads, kv_heads, *, stored_residuals=False
):
    splits = min(16, (topk + 127) // 128)
    corrections = 0 if stored_residuals else topk * kv_heads * 64
    return (
        queries
        * 4
        * (
            query_heads * 96
            + corrections
            + 2 * query_heads * topk
            + query_heads * splits * (256 + 64 + 1)
        )
    )
