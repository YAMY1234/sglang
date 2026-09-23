"""Versioned v3 storage geometry and request-private page ownership (no CUDA)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class FlashNextLatentLayout:
    tp_size: int
    rank: int = 4096
    sparse: int = 512
    width: int = 10240
    scheme_c: bool = False

    def __post_init__(self):
        if self.tp_size not in (1, 2):
            raise ValueError("v3 latent transport supports TP1 and TP2")

    @property
    def local_rank(self):
        return self.rank // self.tp_size

    @property
    def local_sparse(self):
        return self.sparse // self.tp_size

    @property
    def local_gap_bytes(self):
        capacity = self.sparse + 2 * min(self.sparse, (self.width - self.sparse) // 255)
        return (capacity + self.tp_size - 1) // self.tp_size

    @property
    def token_bytes(self):
        if self.scheme_c:
            # Packed z + block scales + values + worst-case gap bytes; both
            # fp32 scales, token id and uint16 length are replicated per TP.
            return (self.local_rank // 2 + self.local_rank // 16
                    + self.local_sparse * 2 + self.local_gap_bytes + 14)
        # z + fp32 scale/rms + int16 coordinates + fp32 correction + token id.
        return self.local_rank + 8 + self.local_sparse * 6 + 4


class PrivatePageOwners:
    """Reserve the complete prompt/output bound before accepting a request.

    Pages have no radix owner. A second request with an identical prompt must
    reserve different pages. Admission is read-only; binding commits ownership.
    """
    def __init__(self, tokens, page_size):
        if tokens <= 0 or tokens % page_size:
            raise ValueError("private capacity must be a positive page multiple")
        self.page_size = page_size
        self.capacity = tokens
        self.free = list(range(tokens // page_size, 0, -1))
        self.owners = {}

    def pages_needed(self, key, tokens):
        count = (tokens + self.page_size - 1) // self.page_size
        old = self.owners.get(key)
        if old is not None:
            if count > len(old):
                raise ValueError("request outgrew its private reservation")
            return 0
        if count > self.capacity // self.page_size:
            raise ValueError("request exceeds the entire private deep pool")
        return count

    def can_admit(self, key, tokens, pending=()):
        needed = self.pages_needed(key, tokens)
        seen = {key}
        for candidate, bound in pending:
            if candidate not in seen:
                needed += self.pages_needed(candidate, bound)
                seen.add(candidate)
        return needed <= len(self.free)

    def bind(self, key, tokens):
        needed = self.pages_needed(key, tokens)
        if key in self.owners:
            return self.owners[key]
        if needed > len(self.free):
            raise MemoryError("private deep pages were not reserved by admission")
        pages = [self.free.pop() for _ in range(needed)]
        self.owners[key] = pages
        return pages

    def release(self, key):
        self.free.extend(self.owners.pop(key, ()))

    def clear(self):
        if self.owners:
            raise RuntimeError("cannot clear the private pool with active requests")
        self.free = list(range(self.capacity // self.page_size, 0, -1))
