import asyncio
import unittest
from types import SimpleNamespace

from sglang.srt.managers.explicit_prefill_cohort_dispatcher import (
    ExplicitPrefillCohortDispatcher,
)
from sglang.srt.managers.io_struct import BlockReqType


class TestExplicitPrefillCohortDispatcher(unittest.IsolatedAsyncioTestCase):
    def make_dispatcher(
        self, *, enabled=True, dp_size=None, assign_balanced_dp_ranks=False
    ):
        self.events = []
        return ExplicitPrefillCohortDispatcher(
            enabled=enabled,
            dispatch_one=self.events.append,
            dispatch_control=self.events.append,
            dp_size=dp_size,
            assign_balanced_dp_ranks=assign_balanced_dp_ranks,
        )

    @staticmethod
    def register(dispatcher, rid, cohort_id, size, index):
        return dispatcher.register(
            rid=rid,
            cohort_id=cohort_id,
            cohort_size=size,
            cohort_index=index,
        )

    def assert_cohort(self, *requests):
        self.assertEqual(len(self.events), len(requests) + 2)
        self.assertEqual(self.events[0].req_type, BlockReqType.BLOCK)
        self.assertEqual(self.events[1:-1], list(requests))
        self.assertEqual(self.events[-1].req_type, BlockReqType.UNBLOCK)

    async def test_unmarked_and_complete_cohort_dispatch(self):
        dispatcher = self.make_dispatcher(enabled=False)
        plain = SimpleNamespace(routed_dp_rank=0)
        self.assertFalse(self.register(dispatcher, "plain", None, None, None))
        await dispatcher.dispatch("plain", plain)
        self.assertEqual(self.events, [plain])

        dispatcher = self.make_dispatcher()
        requests = [SimpleNamespace(routed_dp_rank=index) for index in range(4)]
        for index in (2, 0, 3, 1):
            self.register(dispatcher, f"r{index}", "c0", 4, index)
        await asyncio.gather(
            *(
                dispatcher.dispatch(f"r{index}", requests[index])
                for index in reversed(range(4))
            )
        )
        self.assert_cohort(*requests)

    async def test_rejects_invalid_contracts(self):
        dispatcher = self.make_dispatcher()
        invalid = (("c0", None, None), ("c0", 0, 0), ("c0", 2, 2), ("", 1, 0))
        for spec in invalid:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                self.register(dispatcher, "r0", *spec)

        disabled = self.make_dispatcher(enabled=False)
        with self.assertRaisesRegex(ValueError, "SGLANG_ENABLE_COLOCATED_BATCH_GEN"):
            self.register(disabled, "r0", "c0", 1, 0)

    async def test_balances_cohort_and_rejects_bad_rank_coverage(self):
        dispatcher = self.make_dispatcher(dp_size=4, assign_balanced_dp_ranks=True)
        requests = [
            SimpleNamespace(routed_dp_rank=0, input_ids=[0] * length)
            for length in (16, 15, 12, 11, 8, 7, 4, 3)
        ]
        for index in range(8):
            self.register(dispatcher, f"r{index}", "balanced", 8, index)
        await asyncio.gather(
            *(
                dispatcher.dispatch(f"r{index}", request)
                for index, request in enumerate(requests)
            )
        )
        ranks = [request.routed_dp_rank for request in requests]
        self.assertEqual([ranks.count(rank) for rank in range(4)], [2, 2, 2, 2])
        self.assert_cohort(*requests)

        dispatcher = self.make_dispatcher(dp_size=4)
        bad = [SimpleNamespace(routed_dp_rank=rank) for rank in (0, 1, 1, 3)]
        for index in range(4):
            self.register(dispatcher, f"bad-{index}", "bad", 4, index)
        results = await asyncio.gather(
            *(
                dispatcher.dispatch(f"bad-{index}", request)
                for index, request in enumerate(bad)
            ),
            return_exceptions=True,
        )
        self.assertTrue(all(isinstance(result, ValueError) for result in results))
        self.assertEqual(self.events, [])

    async def test_abort_wakes_ready_member_and_closes_cohort(self):
        dispatcher = self.make_dispatcher()
        self.register(dispatcher, "ready", "c0", 3, 0)
        self.register(dispatcher, "failed", "c0", 3, 1)
        task = asyncio.create_task(
            dispatcher.dispatch("ready", SimpleNamespace(routed_dp_rank=0))
        )
        await asyncio.sleep(0)

        dispatcher.abort("failed", ValueError("tokenization failed"))
        with self.assertRaisesRegex(RuntimeError, "tokenization failed"):
            await task
        with self.assertRaisesRegex(ValueError, "already closed"):
            self.register(dispatcher, "late", "c0", 3, 2)
        self.assertEqual(self.events, [])


if __name__ == "__main__":
    unittest.main()
