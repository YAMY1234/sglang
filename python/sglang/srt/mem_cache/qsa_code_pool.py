"""Separate physical prefix-code and exact pages behind stable virtual pages.

This is the storage/ownership primitive; scheduler and radix adapters must name
their owners explicitly. CPU ownership operations run outside CUDA graphs. A
prefix conversion publishes only after every layer has been encoded, and never
invalidates an active exact reader. The graph-facing page tables keep addresses.
"""

from dataclasses import dataclass, field

import torch
from sglang.srt.layers.attention.qsa.code import (
    PrefixCode,
    SparseCode,
    encode_prefix,
)


@dataclass
class _Page:
    exact: int
    code: int = 0
    valid_tokens: int = 0
    owners: dict[str, str] = field(default_factory=dict)
    written: dict[int, int] = field(default_factory=dict)


class QSAPrefixPageStore:
    def __init__(
        self, *, code_pages, exact_pages, page_size, layout, weights, device="cpu"
    ):
        if min(code_pages, exact_pages, page_size) <= 0 or page_size % 4:
            raise ValueError(
                "Positive pools and a page size divisible by 4 are required"
            )
        if not weights:
            raise ValueError("Every stored QSA layer needs code weights")
        self.layout = layout
        self.weights = dict(weights)  # global layer -> (key, value) CodeWeights
        self.page_size = page_size
        self.device = torch.device(device)
        self.virtual_capacity = code_pages + exact_pages
        self._free_virtual = list(range(self.virtual_capacity, 0, -1))
        self._free_exact = list(range(exact_pages, 0, -1))
        self._free_code = list(range(code_pages, 0, -1))
        self.pages: dict[int, _Page] = {}
        self.exact_page = torch.zeros(
            self.virtual_capacity + 1, dtype=torch.int32, device=device
        )
        self.code_page = torch.zeros_like(self.exact_page)
        self.valid_tokens = torch.zeros_like(self.exact_page)
        self.exact = {}
        self.codes = {}
        head_count = next(iter(weights.values()))[0].mean.shape[0]
        self.head_count = head_count
        for layer, (kw, vw) in weights.items():
            if kw.mean.shape != (
                head_count,
                layout.head_dim - layout.rotary_dim,
            ) or vw.mean.shape != (head_count, layout.head_dim):
                raise ValueError("Layer weights do not agree with pool head dimensions")
            if kw.encoder.device != self.device or vw.encoder.device != self.device:
                raise ValueError("Pool and code weights must share a device")

            def buffer(pages, width, dtype=torch.bfloat16):
                result = torch.empty(
                    (pages + 1, page_size, head_count, width),
                    dtype=dtype,
                    device=device,
                )
                result[0].zero_()
                return result

            self.exact[layer] = (
                buffer(exact_pages, layout.head_dim),
                buffer(exact_pages, layout.head_dim),
            )
            self.codes[layer] = PrefixCode(
                buffer(code_pages, layout.rotary_dim),
                SparseCode(
                    buffer(code_pages, layout.key_rank),
                    buffer(code_pages, layout.key_sparse, torch.uint8),
                    buffer(code_pages, layout.key_sparse),
                ),
                SparseCode(
                    buffer(code_pages, layout.value_rank),
                    buffer(code_pages, layout.value_sparse, torch.uint8),
                    buffer(code_pages, layout.value_sparse),
                ),
            )

    @property
    def available_exact_pages(self):
        return min(len(self._free_exact), len(self._free_virtual))

    @property
    def available_code_pages(self):
        return len(self._free_code)

    def _control_only(self):
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Page ownership changes must happen outside CUDA graphs")

    def allocate(self, count, *, owner):
        """Reserve exact pages atomically; no partial allocation on exhaustion."""
        self._control_only()
        if count < 0 or not owner:
            raise ValueError("Page count must be nonnegative and owner must be named")
        if count > self.available_exact_pages:
            raise MemoryError("Insufficient exact or virtual pages")
        result = []
        for _ in range(count):
            virtual, exact = self._free_virtual.pop(), self._free_exact.pop()
            self.pages[virtual] = _Page(exact=exact, owners={owner: "exact"})
            self.exact_page[virtual] = exact
            result.append(virtual)
        return result

    def write_page(self, virtual, layer, k, v, *, owner, start=0):
        """Control-path page writer for encoding/ownership validation.

        Serving decode must use a graph-safe token writer against exact_page;
        it must obey the same no-write-after-publish and owner rules.
        """
        self._control_only()
        page = self.pages[virtual]
        if page.owners.get(owner) != "exact" or not page.exact:
            raise ValueError("Writer does not own the exact representation")
        if page.code or len(page.owners) != 1:
            raise ValueError(
                "Shared or encoded pages are immutable; copy before writing"
            )
        if k.shape != v.shape or k.shape[1:] != (self.head_count, self.layout.head_dim):
            raise ValueError("Invalid page K/V shape")
        if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
            raise ValueError("Original page K/V must be bf16")
        if start < 0 or start + len(k) > self.page_size:
            raise ValueError("Write crosses page boundary")
        if start > page.written.get(layer, 0) or start < page.valid_tokens:
            raise ValueError("Write leaves a hole or overwrites committed tokens")
        kb, vb = self.exact[layer]
        kb[page.exact, start : start + len(k)].copy_(k)
        vb[page.exact, start : start + len(v)].copy_(v)
        page.written[layer] = max(page.written.get(layer, 0), start + len(k))

    def mark_complete(self, virtual, valid_tokens, *, owner):
        """Caller commits a common written length after ALL layers finish."""
        self._control_only()
        page = self.pages[virtual]
        if page.owners.get(owner) != "exact" or page.code:
            raise ValueError("Only an unpublished exact owner may commit a page")
        if not page.valid_tokens <= valid_tokens <= self.page_size:
            raise ValueError("Committed page length cannot decrease or overflow")
        if any(page.written.get(layer, 0) < valid_tokens for layer in self.weights):
            raise ValueError("All layers must finish before publishing a common length")
        page.valid_tokens = valid_tokens
        self.valid_tokens[virtual] = valid_tokens

    def acquire_exact(self, virtual_pages, *, owner):
        self._control_only()
        virtual_pages = list(dict.fromkeys(virtual_pages))
        if not owner:
            raise ValueError("Owner must be named")
        for virtual in virtual_pages:
            page = self.pages[virtual]
            if not page.exact or owner in page.owners:
                raise ValueError("Exact view unavailable or owner already present")
        for virtual in virtual_pages:
            self.pages[virtual].owners[owner] = "exact"

    @staticmethod
    def _copy_code(destination, source, row):
        destination.rotary[row].copy_(source.rotary)
        for name in ("key", "value"):
            dst, src = getattr(destination, name), getattr(source, name)
            for field_name in ("latent", "indices", "originals"):
                getattr(dst, field_name)[row].copy_(getattr(src, field_name))

    def acquire_prefix(self, virtual_pages, *, owner):
        """Prepare/publish immutable code views; keep other owners' exact pages.

        Existing code is reused without re-encoding. If any encoding fails, the
        transaction rolls back physical reservations and publishes no mappings.
        Reads/writes must be ordered on the caller's CUDA stream.
        """
        self._control_only()
        virtual_pages = list(dict.fromkeys(virtual_pages))
        if not owner:
            raise ValueError("Owner must be named")
        missing = []
        for virtual in virtual_pages:
            page = self.pages[virtual]
            if owner in page.owners:
                raise ValueError("Prefix owner already present")
            if page.valid_tokens != self.page_size:
                raise ValueError("Partial pages retain their exact representation")
            if not page.code:
                if not page.exact:
                    raise ValueError("No source representation")
                missing.append(virtual)
        if len(missing) > len(self._free_code):
            raise MemoryError("Insufficient code pages; exact views are unchanged")
        prepared = []
        try:
            for virtual in missing:
                physical = self._free_code.pop()
                prepared.append((virtual, physical))
                page = self.pages[virtual]
                for layer, (kw, vw) in self.weights.items():
                    kb, vb = self.exact[layer]
                    encoded = encode_prefix(
                        kb[page.exact], vb[page.exact], kw, vw, self.layout
                    )
                    self._copy_code(self.codes[layer], encoded, physical)
        except Exception:
            self._free_code.extend(physical for _, physical in reversed(prepared))
            raise
        for virtual, physical in prepared:
            self.pages[virtual].code = physical
            self.code_page[virtual] = physical
        for virtual in virtual_pages:
            self.pages[virtual].owners[owner] = "code"

    def release(self, virtual_pages, *, owner):
        """Release one explicit reader/tree owner, then reclaim unused views."""
        self._control_only()
        virtual_pages = list(dict.fromkeys(virtual_pages))
        if any(owner not in self.pages[v].owners for v in virtual_pages):
            raise ValueError("Cannot release an owner that is not present")
        for virtual in virtual_pages:
            page = self.pages[virtual]
            del page.owners[owner]
            roles = set(page.owners.values())
            if page.exact and "exact" not in roles:
                self._free_exact.append(page.exact)
                page.exact = 0
                self.exact_page[virtual] = 0
            if page.code and "code" not in roles:
                self._free_code.append(page.code)
                page.code = 0
                self.code_page[virtual] = 0
            if not page.owners:
                del self.pages[virtual]
                self.valid_tokens[virtual] = 0
                self._free_virtual.append(virtual)

    def allocation_bytes(self):
        """Physical tensors, including sentinel pages; indexer lives separately."""
        return {
            "exact": sum(t.nbytes for pair in self.exact.values() for t in pair),
            "code": sum(code.nbytes for code in self.codes.values()),
            "page_tables": self.exact_page.nbytes
            + self.code_page.nbytes
            + self.valid_tokens.nbytes,
            "weights": sum(
                t.nbytes
                for pair in self.weights.values()
                for w in pair
                for t in w.tensors()
            ),
        }
