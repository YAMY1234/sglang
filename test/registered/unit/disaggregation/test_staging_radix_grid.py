"""Unit tests for staging-buffer × RadixCache grid alignment and lifecycle.

The staging control plane identifies chunks positionally
(chunk_idx = start_page // full_chunk_pages against a uniform prefetched
ring-allocation grid), which requires every prefill send to land exactly on
grid boundaries. These tests cover:

- staging_grid_tokens / compute_grid_segments: the sender-side floor+split
  math that keeps radix-prefix-inflated (or batch-budget-truncated) sends on
  the grid, for both prefill-side prefixes (base=0) and decode-side prefixes
  (base=decode_prefix_len).
- DecodeStagingHandler completion protocol: scatter is arrival-driven for
  every chunk (including the last); all-ranks Success alone must NOT finish
  a room while an allocation still waits for its CHUNK_READY.
- DecodeStagingHandler.release_room / unregister_decode_req: outstanding
  ring allocations are freed on failure/abort removal instead of pinning the
  watermark forever, and per-room registries are pruned.
- register_decode_req rejects a non-page-aligned decode prefix (the scatter
  offset shift would be inexact).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from sglang.srt.disaggregation.common import staging_handler as sh
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGridMath(CustomTestCase):
    def test_grid_tokens_basic(self):
        # cps=8192, page=64 -> one slot = 128 pages = 8192 tokens
        self.assertEqual(sh.staging_grid_tokens(8192, 64), 8192)
        # None falls back to 8192 (mirrors prefetch_staging_reqs)
        self.assertEqual(sh.staging_grid_tokens(None, 64), 8192)
        # cps smaller than a page floors to one page (max(1, ...) guard)
        self.assertEqual(sh.staging_grid_tokens(16, 64), 64)
        # non-multiple cps floors to whole pages (rejected at startup, but
        # the math must stay consistent with prefetch if it ever runs)
        self.assertEqual(sh.staging_grid_tokens(8200, 64), 8192)

    def test_single_slot_send_is_untouched(self):
        # A normal cps-sized chunk send: exactly one segment, unchanged.
        self.assertEqual(sh.compute_grid_segments(0, 8192, 0, 8192), [(0, 8192)])
        self.assertEqual(
            sh.compute_grid_segments(8192, 16384, 0, 8192), [(8192, 16384)]
        )

    def test_jumbo_prefix_first_send_splits_per_slot(self):
        # Prefill radix hit P=6144 bundled with first chunk C=8192:
        # send [0, 14336) must split at the 8192 boundary.
        self.assertEqual(
            sh.compute_grid_segments(0, 14336, 0, 8192),
            [(0, 8192), (8192, 14336)],
        )

    def test_early_send_long_prefix_multi_slot(self):
        # Early-send of a 3.5-slot prefix: 3 full slots + remainder.
        self.assertEqual(
            sh.compute_grid_segments(0, 28672, 0, 8192),
            [(0, 8192), (8192, 16384), (16384, 24576), (24576, 28672)],
        )

    def test_decode_prefix_base_shifts_grid(self):
        # decode_prefix_len=6144: grid boundaries at 6144 + k*8192.
        # First send after a short first chunk [6144, 8192) stays in slot 0.
        self.assertEqual(
            sh.compute_grid_segments(6144, 8192, 6144, 8192),
            [(6144, 8192)],
        )
        # A send crossing the shifted boundary splits at 6144+8192=14336,
        # NOT at 8192 or 16384.
        self.assertEqual(
            sh.compute_grid_segments(6144, 22528, 6144, 8192),
            [(6144, 14336), (14336, 22528)],
        )

    def test_start_on_boundary(self):
        self.assertEqual(
            sh.compute_grid_segments(14336, 22528, 6144, 8192),
            [(14336, 22528)],
        )

    def test_empty_range_yields_one_empty_segment(self):
        # Fully decode-cached prompt: the metadata-only last chunk still
        # needs exactly one (empty) send to deliver aux/state.
        self.assertEqual(sh.compute_grid_segments(4096, 4096, 0, 8192), [(4096, 4096)])

    def test_floor_matches_segments(self):
        # The floor applied in send_kv_chunk (non-last chunks) must produce
        # an end that compute_grid_segments splits into full slots only.
        base, g = 6144, 8192
        for end in (6145, 14336, 14400, 30719):
            floored = base + ((end - base) // g) * g
            for s, e in sh.compute_grid_segments(base, floored, base, g):
                self.assertEqual((e - s) % g, 0)
                self.assertEqual((s - base) % g, 0)


def _make_handler():
    kv_manager = MagicMock()
    kv_manager._staging_ctx.room_receivers = {}
    kv_manager._staging_ctx.room_bootstrap = {}
    return sh.DecodeStagingHandler(
        kv_manager=kv_manager,
        staging_allocator=MagicMock(),
        kv_buffer_info={"page_size": 64},
        decode_tp=2,
        total_kv_heads=8,
        tp_rank=0,
        scheduler=MagicMock(),
    )


def _make_decode_req(room=7, prefix_len=0, chunk_infos=None):
    receiver = SimpleNamespace(
        chunk_staging_infos=chunk_infos if chunk_infos is not None else [],
        prefill_info=SimpleNamespace(attn_tp_size=4),
    )
    req = SimpleNamespace(bootstrap_room=room, cache_protected_len=prefix_len)
    return SimpleNamespace(req=req, kv_receiver=receiver)


class TestRegisterDecodeReq(CustomTestCase):
    def test_page_aligned_prefix_ok(self):
        handler = _make_handler()
        dr = _make_decode_req(prefix_len=6144)  # 96 pages * 64
        handler.register_decode_req(7, dr)
        self.assertFalse(dr._staging_all_success)
        self.assertEqual(dr._chunk_events, [])
        self.assertTrue(handler.is_staging_room(7))

    def test_misaligned_prefix_raises(self):
        handler = _make_handler()
        dr = _make_decode_req(prefix_len=100)  # not a multiple of 64
        with self.assertRaises(RuntimeError):
            handler.register_decode_req(7, dr)


class _FakeEvent:
    def __init__(self, done=True):
        self.done = done
        self.synchronized = False

    def query(self):
        return self.done

    def synchronize(self):
        self.synchronized = True


class TestArrivalDrivenCompletion(CustomTestCase):
    """all-ranks Success must not finish a room while a staging allocation
    is still waiting for its CHUNK_READY (previously the last chunk was
    blind-scattered from chunk_infos[-1] at Success, corrupting the tail
    whenever actual sends did not line up with the prefetch grid)."""

    def test_success_alone_does_not_finish_with_outstanding_alloc(self):
        handler = _make_handler()
        # chunk 0 scattered (zeroed), chunk 1 still waiting for arrival
        infos = [(-1, -1, 0, -1, 0), (5, 4096, 0, 8192, 128)]
        dr = _make_decode_req(chunk_infos=infos)
        handler.register_decode_req(7, dr)
        handler._free_and_send_watermark = MagicMock()

        self.assertTrue(handler.submit_last_scatter_async(7))
        self.assertTrue(dr._staging_all_success)
        handler.advance_scatter(dr)
        self.assertFalse(handler.is_done(dr))

        # Arrival consumed the allocation (slot zeroed by scatter) and its
        # event fired -> now the room completes.
        infos[1] = (-1, -1, 0, -1, 0)
        dr._chunk_events.append((_FakeEvent(done=True), 5))
        handler.advance_scatter(dr)
        self.assertTrue(handler.is_done(dr))
        handler._free_and_send_watermark.assert_called_once_with(5, dr)

    def test_pending_event_blocks_completion(self):
        handler = _make_handler()
        dr = _make_decode_req(chunk_infos=[(-1, -1, 0, -1, 0)])
        handler.register_decode_req(7, dr)
        handler._free_and_send_watermark = MagicMock()
        handler.submit_last_scatter_async(7)

        ev = _FakeEvent(done=False)
        dr._chunk_events.append((ev, 3))
        handler.advance_scatter(dr)
        self.assertFalse(handler.is_done(dr))

        ev.done = True
        handler.advance_scatter(dr)
        self.assertTrue(handler.is_done(dr))

    def test_nothing_staged_finishes_on_success(self):
        # e.g. fully decode-cached prompt: no chunk was ever staged.
        handler = _make_handler()
        dr = _make_decode_req(chunk_infos=[])
        handler.register_decode_req(7, dr)
        handler.submit_last_scatter_async(7)
        handler.advance_scatter(dr)
        self.assertTrue(handler.is_done(dr))

    def test_success_for_unregistered_room_is_safe(self):
        handler = _make_handler()
        self.assertFalse(handler.submit_last_scatter_async(99))

    def test_completion_timeout_marks_failed(self):
        handler = _make_handler()
        infos = [(5, 4096, 0, 8192, 128)]  # allocation never scattered
        dr = _make_decode_req(chunk_infos=infos)
        handler.register_decode_req(7, dr)
        handler.submit_last_scatter_async(7)
        handler.advance_scatter(dr)
        self.assertFalse(handler.is_failed(dr))
        # Simulate the CHUNK_READY never arriving for longer than the
        # completion timeout.
        dr._staging_success_ts -= handler.completion_timeout + 1
        handler.advance_scatter(dr)
        self.assertTrue(handler.is_failed(dr))
        self.assertFalse(handler.is_done(dr))

    def test_reregistration_preserves_inflight_state(self):
        """pop_preallocated registers before send_metadata and
        DecodeTransferQueue.extend registers again; the second registration
        must not wipe scatter events / the Success flag recorded by the
        decode_thread in between."""
        handler = _make_handler()
        dr = _make_decode_req(chunk_infos=[(-1, -1, 0, -1, 0)])
        handler.register_decode_req(7, dr)
        # decode_thread activity in the window between the two registrations
        dr._chunk_events.append((_FakeEvent(done=False), 5))
        handler.submit_last_scatter_async(7)
        # Second registration of the SAME request must be a no-op.
        handler.register_decode_req(7, dr)
        self.assertEqual(len(dr._chunk_events), 1)
        self.assertTrue(dr._staging_all_success)


class TestFanInArrivals(CustomTestCase):
    """Writer fan-in (prefill_tp > decode_tp): the chunk scatters only after
    all writers arrived, and partial counts survive an interleaved Success
    (they are purged only at room teardown)."""

    def _arrive(self, handler, room, chunk_idx, writer):
        return handler.handle_chunk_arrived(room, chunk_idx, 0, 128, writer)

    def test_two_writer_fan_in(self):
        handler = _make_handler()  # decode_tp=2, prefill attn_tp=4 -> 2 writers
        dr = _make_decode_req(chunk_infos=[(5, 0, 0, 8192, 128)])
        handler.register_decode_req(7, dr)
        with patch.object(handler, "_scatter_region", return_value=True):
            with patch.object(sh.torch.cuda, "Event", MagicMock()):
                self.assertFalse(self._arrive(handler, 7, 0, "writerA"))
                # Success lands between the two writers' arrivals — counts
                # must survive (the old per-backend pop wiped them here).
                handler.submit_last_scatter_async(7)
                handler.advance_scatter(dr)
                self.assertFalse(handler.is_done(dr))
                self.assertTrue(self._arrive(handler, 7, 0, "writerB"))
        # Slot consumed, event pending; drain it -> room completes.
        self.assertEqual(dr.kv_receiver.chunk_staging_infos[0][0], -1)
        self.assertEqual(len(dr._chunk_events), 1)
        dr._chunk_events[0] = (_FakeEvent(done=True), 5)
        handler._free_and_send_watermark = MagicMock()
        handler.advance_scatter(dr)
        self.assertTrue(handler.is_done(dr))

    def test_unregistered_room_arrival_not_counted(self):
        handler = _make_handler()
        self.assertFalse(handler.handle_chunk_arrived(42, 0, 0, 128, "w"))
        self.assertNotIn(42, handler._writer_counts)


class TestReleaseRoom(CustomTestCase):
    def test_unregister_frees_outstanding_allocations(self):
        handler = _make_handler()
        infos = [(3, 0, 0, 4096, 128), (-1, -1, 0, -1, 0), (9, 8192, 0, 12288, 64)]
        dr = _make_decode_req(chunk_infos=infos)
        handler.register_decode_req(7, dr)
        handler.kv_manager._staging_ctx.room_receivers[7] = object()
        handler.kv_manager._staging_ctx.room_bootstrap[7] = ["bi"]
        handler._free_and_send_watermark = MagicMock()

        handler.unregister_decode_req(7)

        handler._free_and_send_watermark.assert_has_calls([call(3, dr), call(9, dr)])
        self.assertEqual(handler._free_and_send_watermark.call_count, 2)
        # Slots zeroed so a late double-release cannot double-free.
        self.assertTrue(all(info[0] == -1 for info in infos))
        self.assertFalse(handler.is_staging_room(7))
        self.assertNotIn(7, handler.kv_manager._staging_ctx.room_receivers)
        self.assertNotIn(7, handler.kv_manager._staging_ctx.room_bootstrap)

    def test_release_waits_for_inflight_scatter_events(self):
        handler = _make_handler()
        dr = _make_decode_req(chunk_infos=[])
        handler.register_decode_req(7, dr)
        handler._free_and_send_watermark = MagicMock()
        ev = _FakeEvent(done=False)
        dr._chunk_events.append((ev, 11))

        handler.unregister_decode_req(7)

        self.assertTrue(ev.synchronized)
        handler._free_and_send_watermark.assert_called_once_with(11, dr)
        self.assertEqual(dr._chunk_events, [])

    def test_success_path_release_is_noop(self):
        handler = _make_handler()
        dr = _make_decode_req(chunk_infos=[(-1, -1, 0, -1, 0)])
        handler.register_decode_req(7, dr)
        handler._free_and_send_watermark = MagicMock()
        handler.unregister_decode_req(7)
        handler._free_and_send_watermark.assert_not_called()

    def test_unregister_twice_is_safe(self):
        handler = _make_handler()
        dr = _make_decode_req(chunk_infos=[])
        handler.register_decode_req(7, dr)
        handler.unregister_decode_req(7)
        handler.unregister_decode_req(7)  # must not raise


if __name__ == "__main__":
    unittest.main()
