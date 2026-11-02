"""need_sort allocators stage released pages and merge only under allocation pressure."""

import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

PAGE_SIZE = 2
NUM_PAGES = 16


def _make_allocator(*, need_sort: bool) -> PagedTokenToKVPoolAllocator:
    return PagedTokenToKVPoolAllocator(
        size=NUM_PAGES * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.float16,
        device="cpu",
        kvcache=None,
        need_sort=need_sort,
    )


def _page_ids(indices: torch.Tensor) -> torch.Tensor:
    return indices[::PAGE_SIZE] // PAGE_SIZE


class TestPagedAllocatorLazyRelease(CustomTestCase):
    def test_release_stages_until_free_pages_run_dry(self):
        allocator = _make_allocator(need_sort=True)
        allocated = allocator.alloc(24)
        free_pages_before = allocator.free_pages

        with patch.object(torch, "cat", wraps=torch.cat) as cat_mock:
            allocator.free(allocated[:2])
            allocator.free(allocated[4:6])
            self.assertIs(allocator.free_pages, free_pages_before)
            self.assertEqual(allocator.num_staged_pages, 2)
            self.assertEqual(allocator.available_size(), 12)
            cat_mock.assert_not_called()

            # free_pages still covers this: no merge.
            primary = allocator.alloc(8)
            self.assertTrue(torch.equal(_page_ids(primary), torch.arange(13, 17)))
            cat_mock.assert_not_called()

            # Pressure: one merge, staged pages come back sorted.
            reused = allocator.alloc(4)
            self.assertTrue(torch.equal(_page_ids(reused), torch.tensor([1, 3])))
            self.assertEqual(cat_mock.call_count, 1)

        self.assertEqual(allocator.num_staged_pages, 0)
        self.assertEqual(allocator.staged_pages, [])

    def test_clear_drops_staged_pages(self):
        allocator = _make_allocator(need_sort=True)
        allocator.free(allocator.alloc(4))

        allocator.clear()
        self.assertEqual(allocator.available_size(), allocator.size)
        self.assertEqual(allocator.staged_pages, [])

    def test_reuse_guard_waits_only_when_staged_pages_are_needed(self):
        allocator = _make_allocator(need_sort=True)
        allocated = allocator.alloc(24)
        event = Mock()
        allocator.set_reuse_guard_event(event)
        allocator.free(allocated[:2])
        # Fresh pages are safe to hand to an asynchronous PD transfer.
        allocator.alloc(8)
        event.synchronize.assert_not_called()

        reused = allocator.alloc(2)
        self.assertTrue(torch.equal(_page_ids(reused), torch.tensor([1])))
        event.synchronize.assert_called_once_with()
        allocator.merge_and_sort_free()
        event.synchronize.assert_called_once_with()

    def test_later_unguarded_free_keeps_pending_guard(self):
        allocator = _make_allocator(need_sort=True)
        allocated = allocator.alloc(allocator.size)
        event = Mock()
        allocator.set_reuse_guard_event(event)
        allocator.free(allocated[:2])
        allocator.set_reuse_guard_event(None)
        allocator.free(allocated[2:4])
        self.assertIsNotNone(allocator.alloc(4))
        event.synchronize.assert_called_once_with()

    def test_grouped_frees_wait_for_latest_forward(self):
        allocator = _make_allocator(need_sort=True)
        allocated = allocator.alloc(allocator.size)
        first, latest = Mock(), Mock()
        allocator.set_reuse_guard_event(first)
        allocator.free(allocated[:2])
        allocator.set_reuse_guard_event(latest)
        allocator.free_group_begin()
        allocator.free_page_ids(torch.tensor([2]))
        allocator.free_group_end()
        allocator.alloc(4)
        first.synchronize.assert_not_called()
        latest.synchronize.assert_called_once_with()

    def test_clear_drops_guard_with_staged_pages(self):
        allocator = _make_allocator(need_sort=True)
        event = Mock()
        allocator.set_reuse_guard_event(event)
        allocator.free(allocator.alloc(4))
        allocator.clear()
        allocator.alloc(allocator.size)
        event.synchronize.assert_not_called()
        self.assertIsNone(allocator._reuse_guard_event)
        self.assertIsNone(allocator._staged_guard_event)


if __name__ == "__main__":
    unittest.main()
