"""Virtual paged allocator with separately bounded exact physical backing."""

import torch
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.utils import get_num_new_pages


class QSACodePageAllocator(PagedTokenToKVPoolAllocator):
    def __init__(self, *args, kvcache, **kwargs):
        self.code_pool = kvcache.full_kv_pool
        super().__init__(*args, kvcache=kvcache, **kwargs)

    def available_exact_size(self):
        return min(
            super().available_size(),
            self.code_pool.store.available_exact_pages * self.page_size,
        )

    def available_for_prefill(self, tree_cache, req=None):
        evictable = tree_cache.full_evictable_size()
        store = self.code_pool.store
        credit = self.code_pool.prefix_code_credit(req) if req is not None else 0
        code = (
            len(store._free_code) - store.reserved_code_pages + credit
        ) * self.page_size
        return max(
            0,
            min(
                super().available_size() + evictable,
                len(store._free_exact) * self.page_size,
                code + evictable,
            ),
        )

    def check_decode_capacity(self, *, num_tokens, tree_cache):
        if self.available_exact_size() >= num_tokens:
            return True
        # Evict code pages as well as virtual IDs. Freed code space can allow
        # an accepted old page to convert and release its exact backing.
        while tree_cache is not None:
            if (
                tree_cache.evict_full(
                    max(self.page_size, num_tokens - self.available_exact_size())
                )
                == 0
            ):
                break
            self.code_pool.store.convert_aged_pages()
            if self.available_exact_size() >= num_tokens:
                return True
        return self.available_exact_size() >= num_tokens

    def _reserve(self, count):
        # The base kernels may merge for a batch-size upper bound. Merge now
        # so that their selected virtual prefix matches the backing we reserve.
        self.merge_and_sort_free()
        if count > self.code_pool.store.available_exact_pages:
            self.code_pool.store.convert_aged_pages()
        if (
            count > len(self.free_pages)
            or count > self.code_pool.store.available_exact_pages
        ):
            return False
        self.code_pool.reserve_pages(self.free_pages[:count])
        return True

    def alloc(self, need_size):
        if not self._reserve(need_size // self.page_size):
            return None
        return super().alloc(need_size)

    def alloc_extend(
        self,
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
        num_new_pages=None,
    ):
        count = num_new_pages
        if count is None:
            count = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
        if not self._reserve(count):
            return None
        return super().alloc_extend(
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            count,
        )

    def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
        count = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True
        )
        if not self._reserve(count):
            return None
        return super().alloc_decode(seq_lens, seq_lens_cpu, last_loc)

    def _release_page_ids(self, *page_ids):
        self.code_pool.free_pages(torch.cat(page_ids))
        super()._release_page_ids(*page_ids)

    def clear(self):
        self.code_pool.clear()
        super().clear()

    def verify_byte_accounting(self):
        store = self.code_pool.store
        issues = []
        virtual_free = super().available_size() // self.page_size
        if virtual_free + len(store.pages) != self.num_pages:
            issues.append("QSA virtual-page ownership count mismatch")
        exact = [p.exact for p in store.pages.values() if p.exact]
        code = [p.code for p in store.pages.values() if p.code]
        if len(set(exact)) != len(exact) or len(set(code)) != len(code):
            issues.append("QSA duplicate physical page mapping")
        if set(exact).intersection(store._free_exact) or set(code).intersection(
            store._free_code
        ):
            issues.append("QSA mapped physical page also appears on free list")
        return issues
