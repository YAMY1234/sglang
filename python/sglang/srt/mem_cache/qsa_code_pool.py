"""Separate physical prefix-code and exact pages behind stable virtual pages.

This is the storage/ownership primitive; scheduler and radix adapters must name
their owners explicitly. CPU ownership operations run outside CUDA graphs. A
prefix conversion publishes only after every layer has been encoded, and never
invalidates an active exact reader. The graph-facing page tables keep addresses.
"""

import heapq
from collections import Counter, defaultdict
from contextlib import contextmanager
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
    code_valid_tokens: int = 0
    owners: dict[str, str] = field(default_factory=dict)
    written: dict[int, int] = field(default_factory=dict)
    generated_ends: dict[str, int] = field(default_factory=dict)
    generated_starts: dict[str, int] = field(default_factory=dict)


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
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.virtual_capacity = code_pages + exact_pages
        # Keep LIFO allocation while allowing O(1) reservation of externally
        # chosen virtual IDs, without scanning the whole free address space.
        self._free_virtual = dict.fromkeys(range(self.virtual_capacity, 0, -1))
        self._free_exact = list(range(exact_pages, 0, -1))
        self._free_code = list(range(code_pages, 0, -1))
        self.reserved_code_pages = 0
        self.pages: dict[int, _Page] = {}
        self._next_query: dict[str, int] = {}
        self._age_heaps = {}
        self._age_ready = {}
        self._age_sequence = 0
        self.delay_histogram = Counter()
        self.conversion_deferred = Counter()
        self._page_table_batch_depth = 0
        self._pending_page_tables = defaultdict(dict)
        self.exact_page = torch.zeros(
            self.virtual_capacity + 1, dtype=torch.int32, device=device
        )
        self.code_page = torch.zeros_like(self.exact_page)
        self.valid_tokens = torch.zeros_like(self.exact_page)
        self.code_valid_tokens = torch.zeros_like(self.exact_page)
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
                    layout.stored_residuals,
                ),
                SparseCode(
                    buffer(code_pages, layout.value_rank),
                    buffer(code_pages, layout.value_sparse, torch.uint8),
                    buffer(code_pages, layout.value_sparse),
                    layout.stored_residuals,
                    layout.value_bitmap,
                ),
            )

    @property
    def available_exact_pages(self):
        return min(len(self._free_exact), len(self._free_virtual))

    @property
    def available_code_pages(self):
        return max(0, len(self._free_code) - self.reserved_code_pages)

    def _control_only(self):
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Page ownership changes must happen outside CUDA graphs")

    @contextmanager
    def batch_page_tables(self):
        """Publish metadata once before the next attention call, on this stream.

        Nested ownership operations may update an address repeatedly; its last
        value wins. Flush on errors too, preserving earlier successful changes.
        No attention call may execute inside this control-only context.
        """
        self._control_only()
        self._page_table_batch_depth += 1
        try:
            yield
        finally:
            self._page_table_batch_depth -= 1
            if self._page_table_batch_depth == 0:
                pending = self._pending_page_tables
                self._pending_page_tables = defaultdict(dict)
                for name, entries in pending.items():
                    if not entries:
                        continue
                    table = getattr(self, name)
                    ids = torch.tensor(
                        list(entries), dtype=torch.int64, device=self.device
                    )
                    values = torch.tensor(
                        list(entries.values()), dtype=table.dtype, device=self.device
                    )
                    table.index_copy_(0, ids, values)

    def _set_page_table(self, name, virtual, value):
        if self._page_table_batch_depth:
            self._pending_page_tables[name][virtual] = value
        else:
            getattr(self, name)[virtual] = value

    def allocate(self, count, *, owner):
        """Reserve exact pages atomically; no partial allocation on exhaustion."""
        self._control_only()
        if count < 0 or not owner:
            raise ValueError("Page count must be nonnegative and owner must be named")
        if count > self.available_exact_pages:
            raise MemoryError("Insufficient exact or virtual pages")
        result = []
        for _ in range(count):
            virtual, exact = self._free_virtual.popitem()[0], self._free_exact.pop()
            self.pages[virtual] = _Page(exact=exact, owners={owner: "exact"})
            self._set_page_table("exact_page", virtual, exact)
            result.append(virtual)
        return result

    def write_page(self, virtual, layer, k, v, *, owner, start=0):
        """Control-path page writer for encoding/ownership validation.

        Serving decode must use a graph-safe token writer against exact_page;
        it must obey the same no-write-after-publish and owner rules.
        """
        self._control_only()
        page = self.pages[virtual]
        if page.owners.get(owner) not in ("exact", "both") or not page.exact:
            raise ValueError("Writer does not own the exact representation")
        if start < page.code_valid_tokens or len(page.owners) != 1:
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
        if page.owners.get(owner) not in ("exact", "both"):
            raise ValueError("Only an unpublished exact owner may commit a page")
        if not page.valid_tokens <= valid_tokens <= self.page_size:
            raise ValueError("Committed page length cannot decrease or overflow")
        if any(page.written.get(layer, 0) < valid_tokens for layer in self.weights):
            raise ValueError("All layers must finish before publishing a common length")
        page.valid_tokens = valid_tokens
        self._set_page_table("valid_tokens", virtual, valid_tokens)

    def commit_serving_writes(self, virtual_pages, valid_lengths, *, owner):
        """Commit a successfully completed ALL-layer model forward, outside graphs.

        A failed/cancelled forward must never call this hook. Graph-safe writers
        cannot mutate Python per-layer counters; their caller owns this fence.
        """
        self._control_only()
        pairs = list(zip(virtual_pages, valid_lengths, strict=True))
        for virtual, length in pairs:
            page = self.pages[virtual]
            if page.owners.get(owner) not in ("exact", "both"):
                raise ValueError("Only writable exact ownership can commit a forward")
            if not page.valid_tokens <= length <= self.page_size:
                raise ValueError("Serving commit cannot rewind or overflow")
        for virtual, length in pairs:
            page = self.pages[virtual]
            if page.valid_tokens == length:
                continue
            page.written = {layer: length for layer in self.weights}
            page.valid_tokens = length
            self._set_page_table("valid_tokens", virtual, length)

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
    def _copy_code(destination, source, rows):
        def copy(dst, src):
            dst.index_copy_(
                0, rows, src.reshape(len(rows), dst.shape[1], *src.shape[1:])
            )

        copy(destination.rotary, source.rotary)
        for name in ("key", "value"):
            dst, src = getattr(destination, name), getattr(source, name)
            for field_name in ("latent", "indices", "originals"):
                copy(getattr(dst, field_name), getattr(src, field_name))

    def acquire_prefix(self, virtual_pages, *, owner, valid_lengths=None):
        """Prepare/publish immutable code views; keep other owners' exact pages.

        Existing code is reused without re-encoding. If any encoding fails, the
        transaction rolls back physical reservations and publishes no mappings.
        Reads/writes must be ordered on the caller's CUDA stream.
        """
        self._control_only()
        virtual_pages = list(dict.fromkeys(virtual_pages))
        lengths = (
            [self.page_size] * len(virtual_pages)
            if valid_lengths is None
            else list(valid_lengths)
        )
        requested = list(zip(virtual_pages, lengths, strict=True))
        if not owner:
            raise ValueError("Owner must be named")
        missing = []
        extensions = []
        for virtual, length in requested:
            page = self.pages[virtual]
            if owner in page.owners:
                raise ValueError("Prefix owner already present")
            if not 0 < length <= page.valid_tokens:
                raise ValueError("Code length exceeds committed page contents")
            if not page.code:
                if not page.exact:
                    raise ValueError("No source representation")
                missing.append(virtual)
            elif length > page.code_valid_tokens:
                if not page.exact:
                    raise ValueError("Missing exact source for code suffix")
                extensions.append((virtual, length))
        if len(missing) > self.available_code_pages:
            raise MemoryError("Insufficient code pages; exact views are unchanged")
        prepared = []
        try:
            for virtual in missing:
                physical = self._free_code.pop()
                prepared.append((virtual, physical))
            if prepared:
                exact_ids = torch.tensor(
                    [self.pages[v].exact for v in missing], device=self.device
                )
                code_ids = torch.tensor([p for _, p in prepared], device=self.device)
                # Bound conversion scratch independently of a 32K prefix or a
                # large batch's aged pages. Publish only after ALL tiles/layers.
                tile_pages = max(1, 1024 // self.page_size)
                for layer, (kw, vw) in self.weights.items():
                    kb, vb = self.exact[layer]
                    for start in range(0, len(prepared), tile_pages):
                        source = exact_ids[start : start + tile_pages]
                        target = code_ids[start : start + tile_pages]
                        encoded = encode_prefix(
                            kb.index_select(0, source).flatten(0, 1),
                            vb.index_select(0, source).flatten(0, 1),
                            kw,
                            vw,
                            self.layout,
                        )
                        self._copy_code(self.codes[layer], encoded, target)
            # Existing code coordinates are immutable. Append only the newly
            # committed suffix, retaining the old published length until every
            # layer succeeds. Unpublished suffix writes may safely be retried.
            for virtual, length in extensions:
                page = self.pages[virtual]
                start = page.code_valid_tokens
                for layer, (kw, vw) in self.weights.items():
                    kb, vb = self.exact[layer]
                    encoded = encode_prefix(
                        kb[page.exact, start:length],
                        vb[page.exact, start:length],
                        kw,
                        vw,
                        self.layout,
                    )
                    dst = self.codes[layer]
                    dst.rotary[page.code, start:length].copy_(encoded.rotary)
                    for side in ("key", "value"):
                        for field_name in ("latent", "indices", "originals"):
                            getattr(getattr(dst, side), field_name)[
                                page.code, start:length
                            ].copy_(getattr(getattr(encoded, side), field_name))
        except Exception:
            self._free_code.extend(physical for _, physical in reversed(prepared))
            raise
        with self.batch_page_tables():
            for virtual, physical in prepared:
                self.pages[virtual].code = physical
                self._set_page_table("code_page", virtual, physical)
            for virtual, length in requested:
                page = self.pages[virtual]
                page.code_valid_tokens = max(page.code_valid_tokens, length)
                self._set_page_table(
                    "code_valid_tokens", virtual, page.code_valid_tokens
                )
                self.pages[virtual].owners[owner] = "code"

    def track_generated_pages(
        self, virtual_pages, end_positions, *, owner, start_offsets=None
    ):
        """Register committed full generated pages and their logical end indices.

        The scheduler calls this only after target acceptance and ALL layer
        writes. Shared exact readers each register their own logical positions.
        Missing metadata is conservative: it blocks a global conversion.
        """
        self._control_only()
        pairs = list(zip(virtual_pages, end_positions, strict=True))
        starts = [0] * len(pairs) if start_offsets is None else list(start_offsets)
        for (virtual, end), start in zip(pairs, starts, strict=True):
            page = self.pages[virtual]
            if (
                page.owners.get(owner) not in ("exact", "both")
                or page.valid_tokens != self.page_size
            ):
                raise ValueError("Only committed full exact pages can enter ageing")
            if end < self.page_size - 1:
                raise ValueError("Generated page logical end is invalid")
            if not 0 <= start < self.page_size:
                raise ValueError("Generated suffix must lie within this page")
            if owner in page.generated_ends and page.generated_ends[owner] != end:
                raise ValueError("A shared page cannot change its logical position")
        for (virtual, end), start in zip(pairs, starts, strict=True):
            page = self.pages[virtual]
            new_owner = owner not in page.generated_ends
            page.generated_ends[owner] = int(end)
            page.generated_starts[owner] = int(start)
            if new_owner:
                due = int(end) + 256
                if self._next_query.get(owner, -1) >= due:
                    self._age_ready[virtual] = page
                else:
                    self._age_sequence += 1
                    heapq.heappush(
                        self._age_heaps.setdefault(owner, []),
                        (due, self._age_sequence, virtual, page),
                    )

    def advance_generation(self, *, owner, next_query_position, committed_position):
        """Advance from accepted tokens, never from uncommitted draft positions."""
        self._control_only()
        if (
            not owner
            or next_query_position < 0
            or next_query_position > committed_position + 1
        ):
            raise ValueError("Next query must follow the accepted commit watermark")
        if next_query_position < self._next_query.get(owner, 0):
            raise ValueError("Generation age cannot rewind; use a new request owner")
        self._next_query[owner] = int(next_query_position)
        pending = self._age_heaps.get(owner, [])
        while pending and pending[0][0] <= next_query_position:
            due, _, virtual, page = heapq.heappop(pending)
            # A virtual ID can be freed and reused before its old deadline.
            if (
                self.pages.get(virtual) is page
                and page.generated_ends.get(owner) == due - 256
                and page.owners.get(owner) in ("exact", "both")
            ):
                self._age_ready[virtual] = page

    def stop_generation(self, owner):
        """A finished request no longer advances its pending age events."""
        self._next_query.pop(owner, None)
        self._age_heaps.pop(owner, None)

    def convert_aged_pages(self, *, delay=256):
        """Atomically publish code for every reader once the slowest is old enough.

        This scheduler operation is outside graphs. It preserves virtual IDs,
        indexer slots and the stable page-table addresses consumed by graphs.
        Code OOM defers conversion and is reported; an encoding error rolls back.
        """
        self._control_only()
        if delay != 256:
            raise ValueError("The published x256 policy has a fixed delay of 256")
        candidates, ages = [], {}
        reserve = self.available_code_pages
        for virtual, page in list(self._age_ready.items()):
            if self.pages.get(virtual) is not page:
                self._age_ready.pop(virtual, None)
                continue
            exact_owners = [
                o for o, role in page.owners.items() if role in ("exact", "both")
            ]
            if not exact_owners or page.valid_tokens != self.page_size:
                self._age_ready.pop(virtual, None)
                continue
            if any(
                o not in page.generated_ends or o not in self._next_query
                for o in exact_owners
            ):
                self.conversion_deferred["uncommitted_or_untracked_reader"] += 1
                self._age_ready.pop(virtual, None)
                continue
            owner_ages = [
                self._next_query[o] - page.generated_ends[o] for o in exact_owners
            ]
            if min(owner_ages) < delay:
                if max(owner_ages) >= delay:
                    self.conversion_deferred["younger_shared_reader"] += 1
                self._age_ready.pop(virtual, None)
                continue
            if not page.code:
                if reserve == 0:
                    self.conversion_deferred["code_pool_capacity"] += 1
                    continue
                reserve -= 1
            candidates.append(virtual)
            ages[virtual] = [
                (age, page.generated_starts.get(owner, 0))
                for owner, age in zip(exact_owners, owner_ages, strict=True)
            ]
        if not candidates:
            return []
        temporary_owner = object()
        self.acquire_prefix(candidates, owner=temporary_owner)
        for virtual in candidates:
            page = self.pages[virtual]
            for owner in page.owners:
                page.owners[owner] = "code"
            for age, start in ages[virtual]:
                # Logical-reader/token ages: shared readers are counted separately.
                self.delay_histogram.update(range(age, age + self.page_size - start))
            page.generated_ends.clear()
            page.generated_starts.clear()
            self._age_ready.pop(virtual, None)
        self.release(candidates, owner=temporary_owner)
        return candidates

    def forget_generation(self, owner):
        """Drop a completed/aborted request watermark after its views are released."""
        if any(owner in p.owners for p in self.pages.values()):
            raise ValueError("Release request page ownership before forgetting it")
        self.stop_generation(owner)

    def release(self, virtual_pages, *, owner):
        """Release one explicit reader/tree owner, then reclaim unused views."""
        self._control_only()
        virtual_pages = list(dict.fromkeys(virtual_pages))
        if any(owner not in self.pages[v].owners for v in virtual_pages):
            raise ValueError("Cannot release an owner that is not present")
        with self.batch_page_tables():
            for virtual in virtual_pages:
                page = self.pages[virtual]
                del page.owners[owner]
                page.generated_ends.pop(owner, None)
                page.generated_starts.pop(owner, None)
                roles = set(page.owners.values())
                if page.exact and not roles.intersection(("exact", "both")):
                    self._free_exact.append(page.exact)
                    page.exact = 0
                    self._set_page_table("exact_page", virtual, 0)
                if page.code and not roles.intersection(("code", "both")):
                    self._free_code.append(page.code)
                    page.code = 0
                    self._set_page_table("code_page", virtual, 0)
                    page.code_valid_tokens = 0
                    self._set_page_table("code_valid_tokens", virtual, 0)
                if page.exact and page.generated_ends:
                    # Removing a slow reader can unblock an already due page.
                    self._age_ready[virtual] = page
                else:
                    self._age_ready.pop(virtual, None)
                if not page.owners:
                    del self.pages[virtual]
                    self._set_page_table("valid_tokens", virtual, 0)
                    self._free_virtual[virtual] = None

    def allocation_bytes(self):
        """Physical tensors, including sentinel pages; indexer lives separately."""
        return {
            "exact": sum(t.nbytes for pair in self.exact.values() for t in pair),
            "code": sum(code.nbytes for code in self.codes.values()),
            "page_tables": self.exact_page.nbytes
            + self.code_page.nbytes
            + self.valid_tokens.nbytes
            + self.code_valid_tokens.nbytes,
            "weights": sum(
                t.nbytes
                for pair in self.weights.values()
                for w in pair
                for t in w.tensors()
            ),
        }
