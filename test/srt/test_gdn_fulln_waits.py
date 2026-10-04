"""Publication-scoped wait accounting, with no tensors/GPU needed."""
from collections import deque
import copy
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest

from test_gdn_fulln_overlap import ov, Runtime, request, batch, prepared
import test_gdn_fulln_overlap as previous


def typed(kind, req=None):
    sb = batch(req)
    sb.forward_mode = NS(is_mixed=lambda: kind == 'mixed', is_decode=lambda: kind == 'decode')
    return sb


class PublicationWaitTest(unittest.TestCase):
    def test_no_pending_checkpoint_does_not_wait_previous_forward(self):
        rt = Runtime(); c = ov.FullNOverlap('cpu', rt)
        with patch.object(rt, 'fence_before_plan', side_effect=AssertionError('whole-forward fence')):
            prepared(c, batch())
        self.assertEqual(rt.events, [])
        self.assertEqual(c.stats['plan_events'], 0)

    def test_512_decode_iterations_have_zero_waits_and_zero_events(self):
        rt = Runtime(); c = ov.FullNOverlap('cpu', rt)
        scheduler = NS(result_queue=deque())
        for _ in range(512):
            self.assertFalse(c.drain_before_planning(scheduler, lambda: self.fail('pop')))
            c.note_iteration(typed('decode'))
        self.assertEqual(rt.events, [])
        for point in c.POINTS:
            row = c.wait_stats[point, 'decode']
            self.assertEqual(row['checks'], 512)
            self.assertEqual(row['wait_count'], 0)
            self.assertEqual(row['gpu_wait_us'], 0)
            self.assertEqual(row['result_host_us'], 0)
            self.assertEqual(row['pending_sum'], 0)

    def test_request_slot_keys_do_not_read_device_index(self):
        req = request(17)
        class DeviceIndex:
            def item(self): raise AssertionError('D2H')
            def tolist(self): raise AssertionError('D2H')
        req.kv.mamba_pool_idx = DeviceIndex()
        self.assertEqual(ov.request_slots(batch(req)), {17})
        self.assertEqual(ov.request_slots(None), set())

    def test_two_pending_only_intersection_gets_wait_event(self):
        for kind in ('decode', 'mixed', 'prefill'):
            for slot in (1, 2, 99):
                with self.subTest(kind=kind, slot=slot):
                    rt = Runtime(); c = ov.FullNOverlap('cpu', rt)
                    first = prepared(c, batch(request(1))); c.publish(first)
                    second = prepared(c, batch(request(2))); c.publish(second)
                    rt.events.clear()
                    records, _ = c.wait_for_slots(c.POINTS[1], {slot})
                    c.note_iteration(typed(kind, request(slot)))
                    self.assertEqual(rt.events.count('schedule-wait-event'), int(slot != 99))
                    self.assertEqual([r.serial for r in records], [slot] if slot != 99 else [])
                    row = c.wait_stats[c.POINTS[1], ov.iteration_kind(typed(kind))]
                    self.assertEqual(row['pending_sum'], 2)
                    self.assertEqual(row['wait_count'], int(slot != 99))

    def test_disjoint_result_is_not_drained_early(self):
        rt = Runtime(); c = ov.FullNOverlap('cpu', rt)
        sb = batch(request(1)); r = prepared(c, sb); c.publish(r)
        q = deque([(copy.copy(sb), None)])
        scheduler = NS(result_queue=q, running_batch=batch(request(2)),
                       last_batch=typed('decode', request(3)), chunked_req=None)
        self.assertFalse(c.drain_before_planning(scheduler, lambda: self.fail('unrelated pop')))
        self.assertEqual(rt.events.count('schedule-wait-event'), 0)
        # Ordinary FIFO result handling later in the same iteration consumes it.
        saved, _ = q.popleft(); rt.copy_sync(); r.before_result(); c.result_consumed(saved)
        self.assertTrue(r.consumed); self.assertIsNone(c.pending)

    def test_intersecting_owner_or_chunk_must_drain_before_plan(self):
        for source in ('running', 'last', 'chunk'):
            with self.subTest(source=source):
                rt = Runtime(); c = ov.FullNOverlap('cpu', rt)
                sb = batch(request(8)); r = prepared(c, sb); c.publish(r)
                q = deque([(copy.copy(sb), None)])
                scheduler = NS(result_queue=q, running_batch=batch(request(2)),
                               last_batch=batch(request(3)), chunked_req=None)
                if source == 'running': scheduler.running_batch = sb
                elif source == 'last': scheduler.last_batch = sb
                else: scheduler.chunked_req = sb.reqs[0]
                def pop():
                    saved, _ = q.popleft(); rt.copy_sync(); r.before_result(); c.result_consumed(saved)
                self.assertTrue(c.drain_before_planning(scheduler, pop))
                self.assertEqual(rt.events.count('schedule-wait-event'), 1)
                self.assertLess(rt.events.index('schedule-wait-event'), rt.events.index('existing-copy-sync'))
                c.note_iteration(typed('decode'))
                self.assertEqual(c.wait_stats[c.POINTS[1], 'decode']['wait_count'], 1)

    def test_checkpoint_intersection_waits_then_rejects_unconsumed_cpu_owner(self):
        rt = Runtime(); c = ov.FullNOverlap('cpu', rt)
        r = prepared(c, batch(request(1))); c.publish(r)
        with self.assertRaisesRegex(RuntimeError, 'prior result drain'):
            prepared(c, batch(request(1)))
        self.assertEqual(rt.events.count('schedule-wait-event'), 1)
        # A different live slot does not wait on unrelated publication.
        n = len(rt.events); prepared(c, batch(request(2)))
        self.assertEqual(len(rt.events), n)

    def test_async_gpu_timing_reports_pending_not_fake_zero_completion(self):
        rt = Runtime(); c = ov.FullNOverlap('cpu', rt)
        r = prepared(c, batch()); c.publish(r)
        c.wait_for_slots(c.POINTS[1], {1}); c.note_iteration(typed('decode'))
        row = c.wait_stats[c.POINTS[1], 'decode']
        self.assertEqual((row['gpu_samples'], row['gpu_pending']), (0, 1))
        rt.copy_sync(); c.log_waits()
        self.assertEqual((row['gpu_samples'], row['gpu_pending'], row['gpu_wait_us']), (1, 0, 7.0))

    def test_first_and_every_100_iterations_log_each_point_and_kind(self):
        c = ov.FullNOverlap('cpu', Runtime())
        with patch.object(c, 'log_waits') as emit:
            for i in range(201): c.note_iteration(typed('decode' if i % 2 else 'mixed'))
            self.assertEqual(emit.call_count, 3)
        for point in c.POINTS:
            self.assertEqual(c.wait_stats[point, 'decode']['checks'], 100)
            self.assertEqual(c.wait_stats[point, 'mixed']['checks'], 101)

    def test_result_completion_is_per_publication_and_does_not_release_another(self):
        rt = Runtime(); c = ov.FullNOverlap('cpu', rt)
        a = batch(request(1)); b = batch(request(2))
        ra = prepared(c, a); c.publish(ra); rb = prepared(c, b); c.publish(rb)
        rt.copy_sync(); ra.before_result(); c.result_consumed(copy.copy(a))
        self.assertIs(c.pending, rb); self.assertIsNotNone(rb.plan)
        rb.before_result(); c.result_consumed(copy.copy(b)); self.assertIsNone(c.pending)

    def test_existing_overlap_on_and_off_sequence_wait_budget(self):
        # Execute the actual Scheduler event_loop_overlap body, not a model loop.
        runner = previous.OverlapEventTest()
        for enabled in (False, True):
            _, c = runner.run_sequence(('decode',) * 8, enabled=enabled)
            self.assertEqual(c.stats['result_waits'], 0)
            self.assertEqual(c.stats['plan_events'], 0)
        _, c = runner.run_sequence(('prefill', 'decode', 'decode', 'prefill', 'decode'))
        self.assertEqual(c.stats['publication_events'], 2)
        self.assertEqual(c.stats['result_waits'], 2)
        self.assertEqual(c.stats['plan_events'], 0)


if __name__ == '__main__': unittest.main()
