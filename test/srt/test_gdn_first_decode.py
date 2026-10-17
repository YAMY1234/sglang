"""docs/223 §13.A: host state machine of SGLANG_GDN_FIRST_DECODE_NO_WAIT (CPU, no model)."""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "gdn_first_decode",
    Path(__file__).resolve().parents[2] / "python/sglang/srt/mem_cache/gdn_first_decode.py")
fd = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = fd
_SPEC.loader.exec_module(fd)


class FirstDecodeNoWaitTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.m = fd.FirstDecodeNoWait(max_batch=1, catchup_fn=lambda slots, saved: self.calls.append((slots, saved)))

    def test_switch_default_off(self):
        saved = os.environ.pop(fd.SWITCH, None)
        try:
            self.assertIsNone(fd.FirstDecodeNoWait.from_env())
            os.environ[fd.SWITCH] = "1"
            self.assertIsNotNone(fd.FirstDecodeNoWait.from_env())
        finally:
            os.environ.pop(fd.SWITCH, None)
            if saved is not None:
                os.environ[fd.SWITCH] = saved

    def test_first_decode_skips_join_then_next_reader_catches_up(self):
        self.m.on_submit([7], [True])
        d = self.m.decide(is_decode=True, batch_slots=[7], publication_pending=True)
        self.assertFalse(d.join)
        self.assertEqual((d.rows, d.sources, d.slots), ((0,), (0,), (7,)))
        with self.m.scope(d, saved="x1"):
            self.m.on_join()  # an unforeseen join inside the forward: inputs not there yet -> no catch-up
            self.assertEqual(self.calls, [])
        self.assertTrue(self.m.needs_join())
        # step 2: no candidate left -> today's join, which then replays the first step on the factored path
        d2 = self.m.decide(is_decode=True, batch_slots=[7], publication_pending=True)
        self.assertEqual(d2, fd.JOIN)
        self.m.on_join()
        self.assertEqual(self.calls, [((7,), "x1")])
        self.assertFalse(self.m.needs_join())

    def test_non_final_rows_are_not_candidates(self):
        self.m.on_submit([3, 4], [False, True])
        self.assertEqual(self.m.candidates, {4: 1})

    def test_batch_above_bucket_joins_and_drops(self):
        self.m.on_submit([7], [True])
        d = self.m.decide(is_decode=True, batch_slots=[7, 9], publication_pending=True)
        self.assertTrue(d.join)
        self.assertEqual(self.m.candidates, {})
        self.assertEqual(self.m.stats["join_batch"], 1)

    def test_larger_bucket_allows_mixed_batch(self):
        m = fd.FirstDecodeNoWait(max_batch=4, catchup_fn=lambda slots, saved: None)
        m.on_submit([7], [True])
        d = m.decide(is_decode=True, batch_slots=[9, 7], publication_pending=True)
        self.assertFalse(d.join)
        self.assertEqual((d.rows, d.slots), ((1,), (7,)))

    def test_pool_operation_on_fresh_slot_joins(self):
        self.m.on_submit([7], [True])
        d = self.m.decide(is_decode=True, batch_slots=[7], pool_op_slots=[7], publication_pending=True)
        self.assertTrue(d.join)
        self.assertEqual(self.m.stats["join_pool_op"], 1)

    def test_extend_or_forward_publication_voids_candidates(self):
        self.m.on_submit([7], [True])
        d = self.m.decide(is_decode=False, batch_slots=[8], publication_pending=True)
        self.assertTrue(d.join)
        self.assertEqual(self.m.candidates, {})
        self.m.on_submit([7], [True])
        self.m.on_forward_stream_publication()
        self.assertEqual(self.m.decide(is_decode=True, batch_slots=[7], publication_pending=False), fd.JOIN)

    def test_batch_without_fresh_row_keeps_candidates(self):
        self.m.on_submit([7], [True])
        d = self.m.decide(is_decode=True, batch_slots=[9], publication_pending=True)
        self.assertTrue(d.join)
        self.assertEqual(self.m.candidates, {7: 0})

    def test_inert_until_backend_wires_catchup(self):
        m = fd.FirstDecodeNoWait(max_batch=1)
        m.on_submit([7], [True])
        self.assertEqual(m.decide(is_decode=True, batch_slots=[7], publication_pending=True), fd.JOIN)

    def test_nothing_pending_is_inert(self):
        self.assertEqual(self.m.decide(is_decode=True, batch_slots=[7], publication_pending=True), fd.JOIN)
        self.m.on_join()
        self.assertEqual(self.calls, [])


class FirstDecodeABTest(unittest.TestCase):
    def test_abba_labels_alternate_join_and_dense(self):
        saved = os.environ.get(fd.SWITCH)
        os.environ[fd.SWITCH] = "ab"
        try:
            m = fd.FirstDecodeNoWait(max_batch=1, catchup_fn=lambda slots, saved: None)
            joins = []
            for i in range(4):
                m.on_submit([7], [True])
                d = m.decide(is_decode=True, batch_slots=[7], publication_pending=True)
                joins.append(d.join)
            self.assertEqual(joins, [True, False, False, True])
        finally:
            if saved is None:
                os.environ.pop(fd.SWITCH, None)
            else:
                os.environ[fd.SWITCH] = saved


class _FM:
    def __init__(self, decode):
        self.decode = decode

    def is_decode(self):
        return self.decode


class _FB:
    def __init__(self, n, decode=True):
        self.batch_size, self.forward_mode = n, _FM(decode)


class FirstDecodeVariantsTest(unittest.TestCase):
    def setUp(self):
        self.joins = []
        self.m = fd.FirstDecodeNoWait(max_batch=2, catchup_fn=lambda slots, saved: None)
        self.m.join_fn = lambda: self.joins.append(1)
        self.v = fd.FirstDecodeVariants(self.m, [1, 2, 4, 8])

    def test_labels_only_for_small_buckets(self):
        self.assertEqual(self.v.labels_for(2), (fd.BASE_LABEL, fd.FDNW_LABEL))
        self.assertEqual(self.v.labels_for(4), (fd.BASE_LABEL,))

    def test_base_without_decision(self):
        self.assertEqual(self.v.select(_FB(1)), fd.BASE_LABEL)

    def test_dense_decision_without_graph_cancels_and_joins(self):
        self.m.on_submit([7], [True])
        d = self.m.decide(is_decode=True, batch_slots=[7], publication_pending=True)
        with self.m.scope(d):
            self.assertEqual(self.v.select(_FB(1)), fd.BASE_LABEL)  # graph state not captured
            self.assertEqual(self.joins, [1])
        self.assertFalse(self.m.needs_join())  # no catch-up after a cancelled dense step

    def test_dense_decision_selects_fdnw_and_prepares(self):
        class G:
            def __init__(self):
                self.calls = []

            def prepare(self, d, bucket):
                self.calls.append((d.rows, d.sources, bucket))

        self.m.graph = G()
        self.m.on_submit([5, 7], [True, True])
        d = self.m.decide(is_decode=True, batch_slots=[9, 7], publication_pending=True)
        with self.m.scope(d):
            self.assertEqual(self.v.select(_FB(2)), fd.FDNW_LABEL)
        self.assertEqual(self.m.graph.calls, [((1,), (1,), 2)])
        self.assertEqual(self.m.pending, [((7,), (1,))])


if __name__ == "__main__":
    unittest.main()
