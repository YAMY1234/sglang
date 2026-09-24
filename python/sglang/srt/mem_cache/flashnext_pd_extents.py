"""PD-only continuous deep layer runs in the existing unified physical arena.

The shared radix representation is unchanged. Shared and private allocations
still consume one free set; fragmentation falls back to several exact runs,
never to aliasing live pages or changing cache contents.
"""
from bisect import bisect_left
import heapq

from sglang.srt.mem_cache.flashnext_unified_layout import UnifiedPageOwners


class FreeExtents:
    def __init__(self, capacity):
        self.capacity = capacity
        self.starts = [1]
        self.ends = {1:capacity+1}
        self.heap = [(-capacity, 1, capacity+1)]
        self.available = capacity

    def __len__(self):
        return self.available

    def __iter__(self):
        for start in self.starts:
            yield from range(start, self.ends[start])

    def _erase(self, start):
        end = self.ends.pop(start)
        self.starts.pop(bisect_left(self.starts, start))
        return end

    def _insert(self, start, end):
        self.starts.insert(bisect_left(self.starts, start), start)
        self.ends[start] = end
        heapq.heappush(self.heap, (start-end, start, end))
        if len(self.heap) > 4*len(self.starts)+1024:
            self.heap = [(s-e,s,e) for s,e in self.ends.items()]
            heapq.heapify(self.heap)

    def take(self, count):
        if count < 0 or count > self.available:
            raise MemoryError('unified free extents exhausted')
        runs = []
        left = count
        while left:
            while self.heap:
                _, start, end = self.heap[0]
                if self.ends.get(start) == end:
                    break
                heapq.heappop(self.heap)
            if not self.heap:
                raise AssertionError('free extent inventory diverged')
            self._erase(start)
            stop = min(end, start+left)
            runs.append((start, stop))
            if stop < end:
                self._insert(stop, end)
            left -= stop-start
        self.available -= count
        return runs

    def extend(self, units):
        units = sorted(units)
        if not units:
            return
        if units[0] <= 0 or units[-1] > self.capacity or len(set(units)) != len(units):
            raise ValueError('invalid released physical units')
        runs = []
        begin = previous = units[0]
        for value in units[1:]:
            if value != previous+1:
                runs.append((begin, previous+1))
                begin = value
            previous = value
        runs.append((begin, previous+1))
        # Check every run before any mutation, including double-free overlap.
        for begin, end in runs:
            index = bisect_left(self.starts, begin)
            if ((index and self.ends[self.starts[index-1]] > begin)
                    or (index < len(self.starts) and self.starts[index] < end)):
                raise ValueError('physical unit already free')
        for begin, end in runs:
            index = bisect_left(self.starts, begin)
            if index and self.ends[self.starts[index-1]] == begin:
                begin = self.starts[index-1]
                self._erase(begin)
            index = bisect_left(self.starts, end)
            if index < len(self.starts) and self.starts[index] == end:
                end = self._erase(end)
            self._insert(begin, end)
        self.available += len(units)


class ContiguousDeepOwners(UnifiedPageOwners):
    def __init__(self, units):
        super().__init__(units)
        self.free = FreeExtents(units)
        self.pd_extent_stats = dict(deep_reservations=0, deep_layer_runs=0,
                                    fragmented_deep_layers=0)

    def _reserve(self, kind, pages):
        table = getattr(self, kind)
        width = getattr(self, kind+'_units')
        pages = list(pages)
        if len(set(pages)) != len(pages) or any(p <= 0 or p in table for p in pages):
            raise ValueError('virtual pages must be positive, distinct and unowned')
        if len(pages)*width > len(self.free):
            raise MemoryError('shared physical free set exhausted')
        if not pages:
            return
        pending = getattr(self, 'pending_'+kind)
        if kind == 'deep':
            columns = []
            for layer in range(width):
                runs = self.free.take(len(pages))
                columns.append([u for start,end in runs for u in range(start,end)])
                self.pd_extent_stats['deep_layer_runs'] += len(runs)
                self.pd_extent_stats['fragmented_deep_layers'] += len(runs) > 1
            self.pd_extent_stats['deep_reservations'] += 1
            values = zip(*columns)
        else:
            # Shared pages retain their original seven-QSA/two-latent layout.
            units = [u for start,end in self.free.take(len(pages)*width) for u in range(start,end)]
            values = (units[i:i+width] for i in range(0,len(units),width))
        for page, value in zip(pages, values):
            table[page] = list(value)
            pending[page] = table[page]

    def _release(self, kind, pages):
        table = getattr(self, kind)
        pages = list(pages)
        if len(set(pages)) != len(pages) or any(p not in table for p in pages):
            raise ValueError('release requires distinct live virtual pages')
        self.free.extend(u for page in pages for u in table[page])
        for page in pages:
            del table[page]
            getattr(self, 'pending_'+kind).pop(page, None)

    def audit(self, *, full=True):
        result = super().audit(full=False)
        if full:
            units = [u for table in (self.shared,self.deep) for row in table.values() for u in row]
            units += list(self.free)
            if len(units) != self.capacity or set(units) != set(range(1,self.capacity+1)):
                raise AssertionError('physical ownership/free extent partition does not close')
        return dict(**result, **self.pd_extent_stats, free_extents=len(self.free.starts))
