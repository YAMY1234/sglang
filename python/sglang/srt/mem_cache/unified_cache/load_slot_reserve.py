"""Landing slots for external-linker loads whose page ids are known on the CPU.

SGLANG_LINKER_LOAD_RESERVE_PAGES. The paged allocator's indices are produced by
GPU ops queued with the batch, so a load thread that starts before they run reads
stale ids (Hecate D4b: misaligned reads, one request restored into the wrong
slots). Pages held here belong to no request and no tree node, so nothing on the
GPU reads or writes them; a load can land in them as soon as it starts.
"""

import logging
from collections import Counter, deque
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


class LoadSlotReserve:
    """``alloc(num_tokens)`` returns page-aligned device indices or None. Pages
    taken are replaced from it; a replacement joins the free list only once its
    ids, copied after an event recorded when it was allocated, are on the CPU."""

    def __init__(
        self,
        target_pages: int,
        page_size: int,
        device: torch.device,
        alloc: Callable[[int], Optional[torch.Tensor]],
        event_factory: Callable[[], object],
        pin_memory: bool,
    ):
        self.target, self.page_size, self.device = target_pages, page_size, device
        self.alloc, self.event_factory, self.pin_memory = alloc, event_factory, pin_memory
        self.free: deque[int] = deque()
        # (event, CPU tensor of page ids filled by a non-blocking copy)
        self.pending: deque[tuple[object, torch.Tensor]] = deque()
        self.counts: Counter = Counter()
        indices = alloc(target_pages * page_size)
        if indices is None:
            raise RuntimeError(
                f"Linker load reserve: cannot allocate {target_pages} pages"
            )
        # One synchronous copy, at the first load.
        self.free.extend(self._page_ids(indices).tolist())
        logger.info(
            "Linker load reserve: %d pages (%d tokens) held as CPU-known load slots",
            target_pages,
            target_pages * page_size,
        )

    def _page_ids(self, indices: torch.Tensor) -> torch.Tensor:
        return indices.view(-1, self.page_size)[:, 0] // self.page_size

    def held_pages(self) -> int:
        return len(self.free) + sum(len(ids) for _, ids in self.pending)

    def held_tokens(self) -> int:
        return self.held_pages() * self.page_size

    def poll(self) -> None:
        """Move replacements whose copy has landed to the free list (no wait)."""
        while self.pending and self.pending[0][0].query():
            self.free.extend(self.pending.popleft()[1].tolist())

    def _request(self, num_pages: int) -> None:
        indices = self.alloc(num_pages * self.page_size)
        if indices is None:
            self.counts["refill_short"] += 1
            return
        ids = torch.empty(num_pages, dtype=torch.int64, pin_memory=self.pin_memory)
        ids.copy_(self._page_ids(indices), non_blocking=True)
        event = self.event_factory()
        event.record()
        self.pending.append((event, ids))

    def take(self, num_pages: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """(CPU slot indices, device copy) of ``num_pages`` reserved pages, or None
        when too few ids are on the CPU (the load then waits as without a reserve)."""
        self.poll()
        deficit = self.target - self.held_pages()
        if deficit > 0:
            self._request(deficit)
        if len(self.free) < num_pages:
            self.counts["fallback"] += 1
            return None
        pages = torch.tensor([self.free.popleft() for _ in range(num_pages)], dtype=torch.int64)
        cpu = (pages[:, None] * self.page_size + torch.arange(self.page_size)).reshape(-1)
        if self.pin_memory:
            cpu = cpu.pin_memory()
        device = cpu.to(self.device, non_blocking=True)
        self.counts["taken"] += 1
        self.counts["taken_pages"] += num_pages
        self._request(num_pages)
        return cpu, device
