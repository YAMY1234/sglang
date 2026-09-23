"""One physical free list for shared prefix pages and private deep QSA pages.

A unit holds one QSA layer's K/V and compressed indexer for 64 tokens.
Seven units hold shallow QSA and two hold the exact Scheme-C wire bytes.
Five units hold a private deep page. Only shared owners may enter radix.
"""
from sglang.srt.mem_cache.flashnext_latent_layout import PrivatePageOwners


class UnifiedPageOwners:
    shared_units = 9
    deep_units = 5

    def __init__(self, units):
        if units <= 0:
            raise ValueError("physical capacity must be positive")
        self.capacity = units
        self.free = list(range(units, 0, -1))
        self.shared = {}
        self.deep = {}
        self.pending_shared = {}
        self.pending_deep = {}

    def _reserve(self, kind, pages):
        table = getattr(self, kind)
        width = getattr(self, kind + "_units")
        pages = list(pages)
        if (len(set(pages)) != len(pages) or any(p <= 0 or p in table for p in pages)):
            raise ValueError("virtual pages must be positive, distinct and unowned")
        if len(pages) * width > len(self.free):
            raise MemoryError("shared physical free list exhausted")
        pending = getattr(self, "pending_" + kind)
        for page in pages:
            table[page] = [self.free.pop() for _ in range(width)]
            pending[page] = table[page]

    def reserve_shared(self, pages):
        self._reserve("shared", pages)

    def reserve_deep(self, pages):
        self._reserve("deep", pages)

    def _release(self, kind, pages):
        table = getattr(self, kind)
        pages = list(pages)
        if len(set(pages)) != len(pages) or any(p not in table for p in pages):
            raise ValueError("release requires distinct live virtual pages")
        for page in pages:
            self.free.extend(table.pop(page))
            getattr(self, "pending_" + kind).pop(page, None)

    def release_shared(self, pages):
        self._release("shared", pages)

    def release_deep(self, pages):
        self._release("deep", pages)

    def audit(self, *, full=True):
        live_count = len(self.shared)*self.shared_units + len(self.deep)*self.deep_units
        if live_count + len(self.free) != self.capacity:
            raise AssertionError("physical ownership/free-list partition does not close")
        if full:
            live = [u for table in (self.shared, self.deep) for group in table.values() for u in group]
            all_units = live + self.free
            if len(all_units) != self.capacity or set(all_units) != set(range(1, self.capacity+1)):
                raise AssertionError("physical ownership/free-list partition does not close")
        return dict(physical_units=self.capacity, free_units=len(self.free),
                    shared_units=len(self.shared)*self.shared_units,
                    private_units=len(self.deep)*self.deep_units)


class UnifiedPrivatePageOwners(PrivatePageOwners):
    """Virtual private IDs; backing comes from the SAME physical prefix arena."""
    def __init__(self, tokens, page_size, arena):
        super().__init__(tokens, page_size)
        self.arena = arena

    def can_admit(self, key, tokens, pending=()):
        seen = {key}
        needed = self.pages_needed(key, tokens)
        for candidate, bound in pending:
            if candidate not in seen:
                needed += self.pages_needed(candidate, bound)
                seen.add(candidate)
        return needed <= len(self.free) and needed*self.arena.deep_units <= len(self.arena.free)

    def bind(self, key, tokens):
        if key in self.owners:
            return self.owners[key]
        needed = self.pages_needed(key, tokens)
        if needed > len(self.free) or needed*self.arena.deep_units > len(self.arena.free):
            raise MemoryError("private reservation was not admitted against the physical arena")
        pages = super().bind(key, tokens)
        self.arena.reserve_deep(pages)
        return pages

    def release(self, key):
        self.arena.release_deep(self.owners.get(key, ()))
        super().release(key)
