import unittest

from sglang.srt.disaggregation.prefill_lending import (
    BorrowAction,
    BorrowerLedger,
    BorrowState,
    LendCancel,
    LendCancelAck,
    LendCandidate,
    LendDone,
    LendingConfig,
    LendRequest,
    OwnerAction,
    OwnerLedger,
    RankStatus,
    full_page_split,
    page_runs,
    plan_lends,
    region_copies,
    split_surplus,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CFG = LendingConfig(enabled=True, max_inflight_per_borrower=2, min_uncached_tokens=1000, margin_tokens=500)


def status(rank, *, surplus=0, spare=20000, inflight=0, stamp=10.0):
    return RankStatus(rank=rank, seq=1, surplus_tokens=surplus, spare_tokens=spare, borrowed_inflight=inflight, stamp=stamp)


def cand(rid, uncached, on_device=True):
    return LendCandidate(rid=rid, uncached_tokens=uncached, prefix_on_device=on_device)


class TestSplitSurplus(unittest.TestCase):
    def test_next_batch_keeps_whole_requests(self):
        queue = [("a", 3000, True), ("b", 3000, True), ("c", 2000, True), ("d", 4000, True)]
        out, surplus = split_surplus(queue, next_batch_budget=5000)
        # a and b fill the next batch (b is started even though it overruns the budget).
        self.assertEqual([c.rid for c in out], ["c", "d"])
        self.assertEqual(surplus, 6000)

    def test_short_queue_has_no_surplus(self):
        out, surplus = split_surplus([("a", 100, True)], next_batch_budget=32768)
        self.assertEqual((out, surplus), ([], 0))


class TestPlanLends(unittest.TestCase):
    def test_lends_go_to_the_peer_with_most_spare(self):
        peers = {1: status(1, spare=8000), 2: status(2, spare=15000)}
        plan = plan_lends([cand("a", 4000)], own_rank=0, peers=peers, cfg=CFG, now=10.0)
        self.assertEqual(plan, [("a", 2)])

    def test_spare_must_hold_request_plus_margin(self):
        peers = {1: status(1, spare=4400)}
        self.assertEqual(plan_lends([cand("a", 4000)], own_rank=0, peers=peers, cfg=CFG, now=10.0), [])
        peers = {1: status(1, spare=4500)}
        self.assertEqual(plan_lends([cand("a", 4000)], own_rank=0, peers=peers, cfg=CFG, now=10.0), [("a", 1)])

    def test_donors_stale_peers_and_self_never_borrow(self):
        peers = {
            0: status(0, spare=30000),
            1: status(1, surplus=5000, spare=30000),
            2: status(2, spare=30000, stamp=8.5),
        }
        self.assertEqual(plan_lends([cand("a", 4000)], own_rank=0, peers=peers, cfg=CFG, now=10.0), [])

    def test_inflight_slots_cap_lends(self):
        peers = {1: status(1, spare=100000, inflight=1)}
        cands = [cand(f"r{i}", 2000) for i in range(4)]
        plan = plan_lends(cands, own_rank=0, peers=peers, cfg=CFG, now=10.0)
        self.assertEqual(plan, [("r0", 1)])
        plan = plan_lends(cands, own_rank=0, peers={1: status(1, spare=100000)}, cfg=CFG, now=10.0, lent_to={1: 2})
        self.assertEqual(plan, [])

    def test_spare_is_consumed_across_the_plan(self):
        peers = {1: status(1, spare=9000)}
        plan = plan_lends([cand("a", 4000), cand("b", 4000)], own_rank=0, peers=peers, cfg=CFG, now=10.0)
        # 4000 + 500 leaves 4500, which still takes b.
        self.assertEqual(plan, [("a", 1), ("b", 1)])

    def test_a_request_that_fits_nowhere_stays_and_later_ones_may_go(self):
        peers = {1: status(1, spare=6000)}
        plan = plan_lends([cand("big", 30000), cand("small", 3000)], own_rank=0, peers=peers, cfg=CFG, now=10.0)
        self.assertEqual(plan, [("small", 1)])

    def test_host_resident_prefix_and_tiny_requests_stay(self):
        peers = {1: status(1, spare=50000)}
        cands = [cand("host", 4000, on_device=False), cand("tiny", 999), cand("ok", 1000)]
        self.assertEqual(plan_lends(cands, own_rank=0, peers=peers, cfg=CFG, now=10.0), [("ok", 1)])

    def test_force_ignores_spare_and_surplus_but_keeps_slots(self):
        cfg = LendingConfig(enabled=True, max_inflight_per_borrower=1, force=True)
        peers = {1: status(1, surplus=9000, spare=0)}
        plan = plan_lends([cand("a", 10), cand("b", 10)], own_rank=0, peers=peers, cfg=cfg, now=10.0)
        self.assertEqual(plan, [("a", 1)])


class TestPagePlanning(unittest.TestCase):
    def test_runs_coalesce_only_when_both_sides_are_consecutive(self):
        self.assertEqual(page_runs([4, 5, 6, 9], [10, 11, 20, 21]), [(4, 10, 2), (6, 20, 1), (9, 21, 1)])
        with self.assertRaises(ValueError):
            page_runs([1, 2], [3])

    def test_region_copies_scale_by_page_bytes(self):
        srcs, dsts, lens = region_copies([1000, 5000], [9000, 13000], [10, 4], [(2, 7, 3)])
        self.assertEqual(srcs, [1020, 5008])
        self.assertEqual(dsts, [9070, 13028])
        self.assertEqual(lens, [30, 12])

    def test_owner_keeps_the_last_token_page(self):
        # 3 prefix pages; input of 7 pages exactly: the borrower returns pages 3..5, the owner
        # recomputes page 6 (it holds the last token, which it samples).
        self.assertEqual(full_page_split(3 * 256, 7 * 256, 256), (3, 3))
        self.assertEqual(full_page_split(3 * 256, 3 * 256 + 10, 256), (3, 0))
        with self.assertRaises(ValueError):
            full_page_split(100, 1000, 256)


class TestOwnerLedger(unittest.TestCase):
    def setUp(self):
        self.ledger = OwnerLedger(timeout_s=5.0)

    def test_done_inserts_and_requeues(self):
        lid = self.ledger.new_lend("r", borrower=1, now=0.0)
        self.assertTrue(self.ledger.is_lent("r"))
        action, entry = self.ledger.on_done(LendDone(lend_id=lid, borrower=1, ok=True))
        self.assertEqual(action, OwnerAction.INSERT_AND_REQUEUE)
        self.assertEqual(entry.rid, "r")
        self.assertFalse(self.ledger.is_lent("r"))
        self.assertEqual(self.ledger.entries, {})

    def test_failed_done_runs_at_home(self):
        lid = self.ledger.new_lend("r", borrower=1, now=0.0)
        action, _ = self.ledger.on_done(LendDone(lend_id=lid, borrower=1, ok=False, reason="oom"))
        self.assertEqual(action, OwnerAction.REQUEUE_AND_FREE)

    def test_reply_from_another_rank_or_unknown_lend_is_ignored(self):
        lid = self.ledger.new_lend("r", borrower=1, now=0.0)
        self.assertEqual(self.ledger.on_done(LendDone(lend_id=lid, borrower=2, ok=True))[0], OwnerAction.IGNORE)
        self.assertEqual(self.ledger.on_done(LendDone(lend_id=lid + 9, borrower=1, ok=True))[0], OwnerAction.IGNORE)
        self.assertTrue(self.ledger.is_lent("r"))

    def test_a_request_is_lent_once(self):
        self.ledger.new_lend("r", borrower=1, now=0.0)
        with self.assertRaises(ValueError):
            self.ledger.new_lend("r", borrower=2, now=0.0)

    def test_timeout_requeues_and_late_done_only_frees_quarantined_pages(self):
        lid = self.ledger.new_lend("r", borrower=1, now=0.0)
        self.assertEqual(self.ledger.expire(now=4.9), [])
        expired = self.ledger.expire(now=5.1)
        self.assertEqual([e.lend_id for e in expired], [lid])
        # The request runs at home now; it may even be lent again under a new id.
        self.assertFalse(self.ledger.is_lent("r"))
        lid2 = self.ledger.new_lend("r", borrower=2, now=6.0)
        # A late success must not insert pages the home run no longer expects.
        self.assertEqual(self.ledger.on_done(LendDone(lend_id=lid, borrower=1, ok=True))[0], OwnerAction.FREE_QUARANTINE)
        self.assertTrue(self.ledger.is_lent("r"))
        self.assertEqual(self.ledger.on_done(LendDone(lend_id=lid2, borrower=2, ok=True))[0], OwnerAction.INSERT_AND_REQUEUE)

    def test_timeout_then_cancel_ack_frees_quarantine(self):
        lid = self.ledger.new_lend("r", borrower=1, now=0.0)
        self.ledger.expire(now=6.0)
        action, _ = self.ledger.on_cancel_ack(LendCancelAck(lend_id=lid, borrower=1))
        self.assertEqual(action, OwnerAction.FREE_QUARANTINE)
        self.assertEqual(self.ledger.entries, {})

    def test_abort_waits_for_the_final_reply_before_freeing(self):
        lid = self.ledger.new_lend("r", borrower=1, now=0.0)
        entry = self.ledger.on_abort("r")
        self.assertEqual(entry.lend_id, lid)
        self.assertIn(lid, self.ledger.entries)
        self.assertIsNone(self.ledger.on_abort("r"))
        action, _ = self.ledger.on_done(LendDone(lend_id=lid, borrower=1, ok=True))
        self.assertEqual(action, OwnerAction.RETIRE_AND_FREE)
        self.assertEqual(self.ledger.entries, {})

    def test_inflight_counts_include_quarantined_lends(self):
        self.ledger.new_lend("a", borrower=1, now=0.0)
        self.ledger.new_lend("b", borrower=1, now=0.0)
        self.ledger.new_lend("c", borrower=2, now=10.0)
        self.ledger.expire(now=6.0)
        self.assertEqual(self.ledger.inflight_by_borrower(), {1: 2, 2: 1})


def lend(lend_id=1, owner=0):
    return LendRequest(
        lend_id=lend_id, owner=owner, rid="r", input_ids=[1, 2, 3], prefix_tokens=0,
        owner_prefix_pages=[], owner_dst_pages=[],
    )


class TestBorrowerLedger(unittest.TestCase):
    def setUp(self):
        self.ledger = BorrowerLedger(rank=1)

    def test_states_only_move_forward(self):
        self.ledger.add(lend(), now=0.0)
        self.ledger.advance(0, 1, BorrowState.SEEDING)
        self.ledger.advance(0, 1, BorrowState.QUEUED)
        with self.assertRaises(ValueError):
            self.ledger.advance(0, 1, BorrowState.SEEDING)
        self.ledger.advance(0, 1, BorrowState.RETURNING)
        self.assertIsNotNone(self.ledger.finish(0, 1))
        self.assertEqual(self.ledger.inflight(), 0)

    def test_duplicate_lend_is_refused(self):
        self.ledger.add(lend(), now=0.0)
        with self.assertRaises(ValueError):
            self.ledger.add(lend(), now=0.0)
        # Same id from another owner is a different lend.
        self.ledger.add(lend(owner=2), now=0.0)

    def test_cancel_while_writing_waits_for_the_copy(self):
        self.ledger.add(lend(), now=0.0)
        self.ledger.advance(0, 1, BorrowState.RETURNING)
        action, entry = self.ledger.on_cancel(LendCancel(lend_id=1, owner=0))
        self.assertEqual(action, BorrowAction.ACK_AFTER_COPY)
        self.assertEqual(entry.state, BorrowState.CANCELLED)
        with self.assertRaises(ValueError):
            self.ledger.advance(0, 1, BorrowState.RETURNING)

    def test_cancel_before_writing_drops_the_local_request(self):
        self.ledger.add(lend(), now=0.0)
        self.ledger.advance(0, 1, BorrowState.QUEUED)
        self.assertEqual(self.ledger.on_cancel(LendCancel(lend_id=1, owner=0))[0], BorrowAction.DROP_REQUEST)
        self.ledger.add(lend(lend_id=2), now=0.0)
        self.assertEqual(self.ledger.on_cancel(LendCancel(lend_id=2, owner=0))[0], BorrowAction.ACK_NOW)

    def test_cancel_after_finish_is_ignored(self):
        self.ledger.add(lend(), now=0.0)
        self.ledger.finish(0, 1)
        self.assertEqual(self.ledger.on_cancel(LendCancel(lend_id=1, owner=0))[0], BorrowAction.IGNORE)


if __name__ == "__main__":
    unittest.main()
