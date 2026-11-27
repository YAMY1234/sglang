import os
import unittest
from array import array
from unittest.mock import patch

import torch

from sglang.srt.managers import schedule_policy
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import CacheAwarePolicy, SchedulePolicy
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_req(rid, origin_input_text, origin_input_ids, sampling_params=None, **kwargs):
    if sampling_params is None:
        sampling_params = SamplingParams()
    return Req(
        rid,
        origin_input_text,
        array("q", origin_input_ids),
        sampling_params,
        **kwargs,
    )


class TestSchedulePolicyHRRN(CustomTestCase):
    def test_calc_priority_hrrn(self):
        """HRRN sorts by response ratio (waited_tokens / uncached_tokens).

        With three fresh reqs (waited_tokens = 0 for all), the ratio is 0 / uncached for each,
        and stable sort keeps original order among equal keys -- effectively pure SUF via the
        rid tie-breaker is avoided, so we set a non-zero arrival snapshot on some reqs to exercise
        the aging half of the formula.
        """
        tree_cache = RadixCache.create_simulated()

        # r_short: small uncached, just arrived (waited=0)
        # r_long:  large uncached, just arrived (waited=0)
        # r_aged:  medium uncached, arrived long ago (waited>>0)
        r_short = _make_req("short", "a", [1])
        r_long = _make_req("long", "a" * 10, list(range(10)))
        r_aged = _make_req("aged", "a" * 3, [1, 2, 3])

        # Fresh reqs arrived with the counter still at 0.
        r_short.arrival_processed_tokens = 0
        r_long.arrival_processed_tokens = 0
        r_aged.arrival_processed_tokens = 0

        waiting_queue = [r_long, r_aged, r_short]

        policy = SchedulePolicy(
            policy="hrrn",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        # processed_tokens=1000 -> waited_tokens is 1000 for every req here.
        # Ratios: short = 1000/1 = 1000; aged = 1000/3 ~= 333; long = 1000/10 = 100.
        # Highest ratio first -> short, aged, long.
        policy.calc_priority(waiting_queue, processed_tokens=1000)

        self.assertEqual(waiting_queue[0].rid, "short")
        self.assertEqual(waiting_queue[1].rid, "aged")
        self.assertEqual(waiting_queue[2].rid, "long")

    def test_calc_priority_hrrn_aging_overtakes_short(self):
        """A long request that has waited enough should overtake a
        just-arrived short request."""
        tree_cache = RadixCache.create_simulated()

        r_long_old = _make_req("long_old", "a" * 100, list(range(100)))
        r_short_new = _make_req("short_new", "a", [1])

        # long_old arrived at counter=0 and has been waiting; short_new
        # just arrived (its snapshot equals the current counter).
        r_long_old.arrival_processed_tokens = 0
        r_short_new.arrival_processed_tokens = 100000

        waiting_queue = [r_short_new, r_long_old]

        policy = SchedulePolicy(
            policy="hrrn",
            tree_cache=tree_cache,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        # processed_tokens=100000 -> long_old.waited = 100000, ratio = 100000/100 = 1000.
        #                            short_new.waited = 0, ratio = 0.
        # long_old should now be first.
        policy.calc_priority(waiting_queue, processed_tokens=100000)

        self.assertEqual(waiting_queue[0].rid, "long_old")
        self.assertEqual(waiting_queue[1].rid, "short_new")

    def test_calc_priority_hrrn_cached_length_affects_order(self):
        """Cached prefix length shortens uncached, so reqs with the same input
        length can sort differently by HRRN. Also verifies rid tie-break for
        equal-ratio reqs.

        Uses _sort_by_hrrn directly to bypass the prefix-match pass inside
        calc_priority, which would overwrite num_matched_prefix_tokens.
        """
        r_more_cached = _make_req("a", "x" * 100, list(range(100)))
        r_less_cached = _make_req("b", "x" * 100, list(range(100)))
        r_tie = _make_req("c", "x" * 100, list(range(100)))

        # r_more_cached: 90 cached -> uncached = 10 -> ratio = 1000 / 10 = 100.
        # r_less_cached:  0 cached -> uncached = 100 -> ratio = 1000 / 100 = 10.
        # r_tie:          0 cached -> uncached = 100 -> ratio = 1000 / 100 = 10 (ties with r_less_cached; rid "b" < "c").
        r_more_cached.num_matched_prefix_tokens = 90
        r_less_cached.num_matched_prefix_tokens = 0
        r_tie.num_matched_prefix_tokens = 0
        r_more_cached.arrival_processed_tokens = 0
        r_less_cached.arrival_processed_tokens = 0
        r_tie.arrival_processed_tokens = 0

        waiting_queue = [r_tie, r_less_cached, r_more_cached]
        SchedulePolicy._sort_by_hrrn(waiting_queue, set(), processed_tokens=1000)

        self.assertEqual(waiting_queue[0].rid, "a")
        self.assertEqual(waiting_queue[1].rid, "b")
        self.assertEqual(waiting_queue[2].rid, "c")


class TestShortestPrefillFirst(CustomTestCase):
    def setUp(self):
        self.policy = SchedulePolicy(
            policy="shortest-prefill-first",
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )

    def make_req(self, rid, uncached, *, cached=0, arrived=0):
        req = _make_req(rid, "", list(range(uncached + cached)))
        req.full_untruncated_fill_ids = req.origin_input_ids[:]
        req.num_matched_prefix_tokens = cached
        req.prefix_len = cached
        req.time_stats.wait_queue_entry_time = arrived
        return req

    def test_calc_priority_uses_uncached_work(self):
        cached = self.make_req("cached", 16, cached=4096)
        short = self.make_req("short", 32)
        long = self.make_req("long", 1024)
        queue = [long, short, cached]
        with patch.object(self.policy, "_compute_prefix_matches", return_value=set()):
            self.policy.calc_priority(queue)
        self.assertEqual([req.rid for req in queue], ["cached", "short", "long"])

    def test_equal_work_uses_arrival_time(self):
        older = self.make_req("z", 32, arrived=1)
        newer = self.make_req("a", 32, arrived=2)
        queue = [newer, older]
        self.policy._sort_by_shortest_prefill(queue, set())
        self.assertEqual(queue, [older, newer])

    def test_duplicate_prefix_is_deprioritized(self):
        duplicate = self.make_req("duplicate", 1)
        other = self.make_req("other", 1024)
        queue = [duplicate, other]
        with patch.object(
            self.policy, "_compute_prefix_matches", return_value={duplicate.rid}
        ):
            self.policy.calc_priority(queue)
        self.assertEqual(queue, [other, duplicate])

    def test_shared_prefix_holds_the_longer_request_not_the_shorter(self):
        # long and short share 32 tokens, the dedupe threshold.
        shared = list(range(32))
        long = _make_req("long", "", shared + list(range(1000, 5000)))
        other = _make_req("other", "", list(range(20000, 26000)))
        short = _make_req("short", "", shared + list(range(9000, 9393)))
        for arrived, req in enumerate([long, other, short]):
            req.full_untruncated_fill_ids = req.origin_input_ids[:]
            req.time_stats.wait_queue_entry_time = arrived
        queue = [long, other, short]
        self.policy.calc_priority(queue)
        self.assertEqual([req.rid for req in queue], ["short", "other", "long"])

    def test_retracted_output_is_part_of_uncached_work(self):
        replay = self.make_req("replay", 16, cached=1024)
        replay.output_ids.extend([0] * 64)
        short = self.make_req("short", 32)
        queue = [replay, short]
        self.policy._sort_by_shortest_prefill(queue, set())
        self.assertEqual(queue, [short, replay])

    def test_chunk_limit_reserves_complete_short_prefills(self):
        continuation = self.make_req("continuation", 16384)
        waiting = [self.make_req("a", 512), self.make_req("b", 1024)]
        self.assertEqual(
            self.policy.shortest_prefill_chunk_limit(continuation, waiting, 4096, 256),
            2560,
        )

    def test_reservation_rounds_to_pages_and_keeps_continuation_progress(self):
        continuation = self.make_req("continuation", 16384)
        self.assertEqual(
            self.policy.shortest_prefill_chunk_limit(
                continuation, [self.make_req("short", 257)], 4096, 256
            ),
            3584,
        )
        self.assertEqual(
            self.policy.shortest_prefill_chunk_limit(
                continuation, [self.make_req("short", 3840)], 4096, 256
            ),
            256,
        )

    def test_no_reservation_when_request_cannot_fit_or_is_not_shorter(self):
        continuation = self.make_req("continuation", 8192)
        for waiting, budget in [
            ([], 4096),
            ([self.make_req("same", 8192)], 4096),
            ([self.make_req("too-large", 4096)], 4096),
            ([self.make_req("short", 1)], 256),
        ]:
            with self.subTest(budget=budget, waiting=[req.rid for req in waiting]):
                self.assertIsNone(
                    self.policy.shortest_prefill_chunk_limit(
                        continuation, waiting, budget, 256
                    )
                )

    def _aged_queue(self, processed):
        # 'long' (400k uncached) arrived first; short ones keep the queue non-empty.
        long = self.make_req("long", 400_000, arrived=0)
        long.arrival_processed_tokens = 0
        shorts = [self.make_req(f"s{i}", 2048, arrived=i + 1) for i in range(3)]
        for r in shorts:
            r.arrival_processed_tokens = processed
        return long, [shorts[0], long, shorts[1], shorts[2]]

    def test_aged_request_goes_to_head_without_chunk_in_flight(self):
        long, queue = self._aged_queue(processed=5_000_000)
        with patch.dict(os.environ, {"SGLANG_SPF_AGE_TOKENS": "4000000"}), patch.object(
            self.policy, "_compute_prefix_matches", return_value=set()
        ):
            self.policy.calc_priority(queue, processed_tokens=5_000_000)
        self.assertEqual(queue[0].rid, "long")
        self.assertEqual([r.rid for r in queue[1:]], ["s0", "s1", "s2"])

    def test_aged_request_waits_while_a_chunk_is_in_flight(self):
        long, queue = self._aged_queue(processed=5_000_000)
        with patch.dict(os.environ, {"SGLANG_SPF_AGE_TOKENS": "4000000"}), patch.object(
            self.policy, "_compute_prefix_matches", return_value=set()
        ):
            self.policy.calc_priority(
                queue,
                processed_tokens=5_000_000,
                chunked_req=self.make_req("chunk", 8192),
            )
        self.assertEqual(queue[-1].rid, "long")

    def test_young_or_disabled_aging_keeps_shortest_first(self):
        for env, processed in [({"SGLANG_SPF_AGE_TOKENS": "4000000"}, 3_000_000), ({}, 50_000_000)]:
            with self.subTest(env=env, processed=processed):
                long, queue = self._aged_queue(processed=processed)
                with patch.dict(os.environ, env), patch.object(
                    self.policy, "_compute_prefix_matches", return_value=set()
                ):
                    self.policy.calc_priority(queue, processed_tokens=processed)
                self.assertEqual(queue[-1].rid, "long")

    def test_aged_chunk_keeps_its_share_of_the_batch(self):
        continuation = self.make_req("continuation", 400_000)
        continuation.arrival_processed_tokens = 0
        waiting = [self.make_req(f"s{i}", 2048) for i in range(8)]
        with patch.dict(os.environ, {"SGLANG_SPF_AGE_TOKENS": "4000000"}):
            self.policy._spf_processed_tokens = 1_000_000
            self.assertEqual(
                self.policy.shortest_prefill_chunk_limit(continuation, waiting, 16384, 64),
                2048,
            )
            self.policy._spf_processed_tokens = 5_000_000
            self.assertEqual(
                self.policy.shortest_prefill_chunk_limit(continuation, waiting, 16384, 64),
                8192,
            )

    def test_other_policy_keeps_normal_chunk_limit(self):
        self.policy.policy = CacheAwarePolicy.HRRN
        self.assertIsNone(
            self.policy.shortest_prefill_chunk_limit(
                self.make_req("continuation", 8192),
                [self.make_req("short", 512)],
                4096,
                256,
            )
        )


class TestShortestPrefillMatchRefresh(CustomTestCase):
    """SGLANG_SPF_MATCH_REFRESH_PASSES bounds the prefix matches per pass
    without changing the shortest-prefill-first order."""

    # (cached, total) tokens; uncached work 64, 1984, 76, 300, 552, 3488.
    SHAPES = [(4096, 4160), (64, 2048), (1024, 1100), (0, 300), (2048, 2600), (512, 4000)]
    SPF_ORDER = ["r0", "r2", "r3", "r4", "r1", "r5"]

    def setUp(self):
        self.tree = RadixCache.create_simulated()
        self.reqs = [self._req(i, cached, total) for i, (cached, total) in enumerate(self.SHAPES)]

    def _req(self, i, cached, total):
        ids = [i * 100000 + t for t in range(total)]
        self._cache(ids[:cached])
        req = _make_req(f"r{i}", "", ids)
        req.time_stats.wait_queue_entry_time = i
        return req

    def _cache(self, ids):
        if ids:
            self.tree.insert(
                InsertParams(key=RadixKey(array("q", ids)), value=torch.arange(len(ids)))
            )

    def _policy(self):
        return SchedulePolicy(
            policy="shortest-prefill-first",
            tree_cache=self.tree,
            enable_hierarchical_cache=True,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )

    def _run(self, policy, queue, refresh):
        env = {} if refresh is None else {"SGLANG_SPF_MATCH_REFRESH_PASSES": refresh}
        with patch.dict(os.environ, env), patch.object(
            schedule_policy, "match_kv_cache", wraps=schedule_policy.match_kv_cache
        ) as match:
            policy.calc_priority(queue)
        return [r.rid for r in queue], {c.args[0].rid for c in match.call_args_list}

    def test_refresh_keeps_spf_order_on_a_static_tree(self):
        base, fixed = self._policy(), self._policy()
        base_queue, fixed_queue = list(self.reqs), list(self.reqs)
        for _ in range(6):
            base_order, _ = self._run(base, base_queue, None)
            fixed_order, _ = self._run(fixed, fixed_queue, "3")
            self.assertEqual(base_order, self.SPF_ORDER)
            self.assertEqual(fixed_order, base_order)

    def test_refresh_matches_new_plus_oldest_share_each_pass(self):
        policy, queue = self._policy(), list(self.reqs)
        _, first = self._run(policy, queue, "3")
        self.assertEqual(first, {r.rid for r in self.reqs})
        seen = set()
        for _ in range(3):
            _, matched = self._run(policy, queue, "3")
            self.assertEqual(len(matched), 2)  # ceil(6 / 3)
            seen |= matched
        self.assertEqual(seen, {r.rid for r in self.reqs})

    def test_new_request_is_matched_and_ordered_on_arrival(self):
        policy, queue = self._policy(), list(self.reqs)
        self._run(policy, queue, "3")
        arrival = self._req(6, 4000, 4010)
        queue.append(arrival)
        order, matched = self._run(policy, queue, "3")
        self.assertIn("r6", matched)
        self.assertEqual(len(matched), 1 + 3)  # the arrival + ceil(7 / 3)
        self.assertEqual(order[0], "r6")

    def test_tree_change_reaches_the_order_within_refresh_passes(self):
        policy, queue = self._policy(), list(self.reqs)
        self._run(policy, queue, "3")
        # r5's prefix becomes cached: 8 uncached tokens, the shortest work.
        self._cache([5 * 100000 + t for t in range(3992)])
        for _ in range(3):
            order, _ = self._run(policy, queue, "3")
        self.assertEqual(order[0], "r5")


if __name__ == "__main__":
    unittest.main()
