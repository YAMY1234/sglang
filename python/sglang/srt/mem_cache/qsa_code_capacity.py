"""Byte accounting for the fixed x256 layout (per tensor-parallel rank)."""

import os
from dataclasses import dataclass


def qsa_exact_reserve_tokens(
    *, slots, context, chunk, draft_tokens=0, prefill_parallelism=1
):
    """Fixed x256 working set; unfinished prompts retain all exact history.

    Two extra pages per slot cover full-page ageing and allocation/tail views.
    With one unfinished chunked request and no scheduler overlap, context plus
    one chunk covers its resident history and a newly admitted prefill batch.
    This is a capacity reserve, never permission to encode a prompt early.
    """
    if (
        slots < 1
        or context < 1
        or chunk < 1
        or draft_tokens < 0
        or prefill_parallelism < 1
    ):
        raise ValueError("Positive slots/context/chunk/prefill parallelism required")

    def rounded(n):
        return (n + 63) // 64 * 64

    tail = slots * (rounded(256 + draft_tokens) + 128)
    prefill = prefill_parallelism * (rounded(context) + rounded(chunk))
    return tail + prefill


def resolve_qsa_exact_tokens(requested, *, slots, context, chunk, draft_tokens=0):
    """None preserves legacy fractions; zero selects the demand formula."""
    if requested is None:
        return None
    if requested < 0 or requested % 64:
        raise ValueError(
            "QSA exact token reserve must be zero or a positive multiple of 64"
        )
    if requested:
        return requested
    return qsa_exact_reserve_tokens(
        slots=slots,
        context=context,
        chunk=chunk if chunk and chunk > 0 else context,
        draft_tokens=draft_tokens,
    )


@dataclass(frozen=True)
class QSACodeCapacity:
    layers: int
    heads: int
    exact_fraction: float = 0.25
    page_size: int = 64
    draft_layers: int = 0
    exact_tokens: int | None = None

    def __post_init__(self):
        if self.layers <= 0 or self.heads not in (1, 2) or self.page_size != 64:
            raise ValueError(
                "x256 capacity requires local QSA layers, TP1/2 and page64"
            )
        if not 0 < self.exact_fraction < 1:
            raise ValueError("Exact physical fraction must lie in (0,1)")
        if self.draft_layers < 0:
            raise ValueError("Draft layer count must be nonnegative")
        if self.exact_tokens is not None and (
            self.exact_tokens < 64 or self.exact_tokens % 64
        ):
            raise ValueError(
                "Fixed exact reserve must be a positive page-aligned token count"
            )

    def allocation(self, tokens, request_slots):
        if tokens < 0 or tokens % self.page_size or request_slots < 1:
            raise ValueError("Page-aligned tokens and positive request slots required")
        pages = tokens // self.page_size
        exact_pages = (
            int(pages * self.exact_fraction)
            if self.exact_tokens is None
            else min(pages, self.exact_tokens // self.page_size)
        )
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
            "page_tables": (pages + 1) * 16,
            "host_control": (pages + 1) * 4,
            "weights": self.layers * self.heads * 280320,
            "prefix_length_table": request_slots * 4,
            "indexer_compressed": (tokens + self.page_size) * self.layers * 64,
            "indexer_pending": request_slots * 4 * (self.layers * 256 + 24),
            # Each MTP runner retains a dense QSA layer and its own pending
            # position table. Its token IDs span the entire virtual target pool.
            "draft_exact": (tokens + self.page_size)
            * self.draft_layers
            * self.heads
            * 1024,
            "draft_indexer_compressed": (tokens + self.page_size)
            * self.draft_layers
            * 64,
            "draft_indexer_pending": self.draft_layers * request_slots * 4 * (256 + 24),
        }

    def total_bytes(self, tokens, request_slots):
        return sum(
            v
            for k, v in self.allocation(tokens, request_slots).items()
            if not k.endswith("capacity") and k != "host_control"
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
        tokens = lo * self.page_size
        if self.exact_tokens is not None and tokens <= self.exact_tokens:
            return 0
        return tokens


def qsa_fused_read_enabled():
    """Opt-in service candidate; allocator and reader must agree on scratch."""
    value = os.environ.get("SGLANG_QSA_CODE_FUSED_READ", "0")
    if value not in ("0", "1"):
        raise ValueError("SGLANG_QSA_CODE_FUSED_READ must be 0 or 1")
    return value == "1"


def qsa_read_workspace_bytes(
    queries, topk, query_heads, kv_heads, *, stored_residuals=False, fused=False
):
    if fused:
        # Q projection and unnormalized sufficient statistics, shared by layers.
        return queries * query_heads * 4 * (97 + ((topk + 127) // 128) * 323)
    splits = min(16, (topk + 127) // 128)
    corrections = 0 if stored_residuals else topk * kv_heads * 64
    return (
        queries
        * 4
        * (
            query_heads * 97
            + corrections
            + 2 * query_heads * topk
            + query_heads * splits * (256 + 64 + 1)
        )
    )
