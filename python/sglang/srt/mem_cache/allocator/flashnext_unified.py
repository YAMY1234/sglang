"""Paged logical prefix IDs and a common physical arena with private requests."""
import math
import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.utils import get_num_new_pages


class FlashNextUnifiedAllocator(PagedTokenToKVPoolAllocator):
    def __init__(self, *args, kvcache, **kwargs):
        self.unified_pool = kvcache
        super().__init__(*args, kvcache=kvcache, **kwargs)

    def available_size(self):
        return min(super().available_size(), len(self.unified_pool.arena.free)//9*self.page_size)

    def _flush_deferred(self):
        if self.free_group is not None and (self.free_group or self.free_page_ids_group):
            self.free_group_end()
            self.free_group_begin()

    def prepare_private_admission(self, req, pending, tree_cache):
        """Caller holds the candidate's radix lock while donor pages are evicted."""
        needed = self.unified_pool.admission_units(req, pending)
        self._flush_deferred()
        while needed > len(self.unified_pool.arena.free):
            deficit = needed - len(self.unified_pool.arena.free)
            evicted = tree_cache.evict(EvictParams(num_tokens=math.ceil(deficit/9)*self.page_size))
            self._flush_deferred()
            if not evicted.num_tokens_evicted:
                break
        return self.unified_pool.can_admit(req, pending)

    def _reserve(self, count):
        if count > len(self.free_pages) or 9*count > len(self.unified_pool.arena.free):
            return False
        if count:
            self.unified_pool.arena.reserve_shared(self.free_pages[:count].tolist())
        return True

    def alloc(self, need_size):
        if need_size//self.page_size > len(self.free_pages):
            self.merge_and_sort_free()
        if not self._reserve(need_size//self.page_size):
            return None
        return super().alloc(need_size)

    def alloc_extend(self, prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu,
                     last_loc, extend_num_tokens, num_new_pages=None):
        # Match the parent's merge condition BEFORE binding physical ownership;
        # sorting later could give the request different logical page IDs.
        if extend_num_tokens//self.page_size + len(prefix_lens) + 1 > len(self.free_pages):
            self.merge_and_sort_free()
        count = num_new_pages
        if count is None:
            count = get_num_new_pages(seq_lens=seq_lens_cpu, page_size=self.page_size,
                                      prefix_lens=prefix_lens_cpu)
        if not self._reserve(count):
            return None
        return super().alloc_extend(prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu,
                                    last_loc, extend_num_tokens, count)

    def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
        if len(seq_lens) > len(self.free_pages):
            self.merge_and_sort_free()
        count = get_num_new_pages(seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True)
        if not self._reserve(count):
            return None
        return super().alloc_decode(seq_lens, seq_lens_cpu, last_loc)

    def _release_page_ids(self, *page_ids):
        self.unified_pool.arena.release_shared(torch.cat(page_ids).tolist())
        super()._release_page_ids(*page_ids)

    def clear(self):
        arena = self.unified_pool.arena
        if arena.deep:
            raise RuntimeError('cannot reset prefix allocator with live private requests')
        arena.release_shared(list(arena.shared))
        super().clear()

    def verify_byte_accounting(self):
        try:
            # Allocation/release preserve exclusivity by construction. Avoid an
            # O(pool capacity) Python set rebuild on every idle scheduler tick;
            # debug mode and the ownership stress tests also census every unit.
            self.unified_pool.arena.audit(full=self.debug_mode)
        except AssertionError as exc:
            return [str(exc)]
        if super().available_size()//self.page_size + len(self.unified_pool.arena.shared) != self.num_pages:
            return ['shared virtual ownership does not close']
        return []
