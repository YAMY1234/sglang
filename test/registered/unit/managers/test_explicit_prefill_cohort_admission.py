import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler_components.explicit_prefill_cohort_admission import (
    ExplicitPrefillCohortAdmissionCoordinator,
)


def make_req(cohort_id="c0", rank=0, size=4, ordinal=0):
    return SimpleNamespace(
        rid=f"r{rank}-{ordinal}",
        routed_dp_rank=rank,
        prefill_cohort_id=cohort_id,
        prefill_cohort_size=size,
        prefill_cohort_index=ordinal * 4 + rank,
    )


class TestExplicitPrefillCohortAdmission(unittest.TestCase):
    def test_holds_until_every_rank_is_ready(self):
        snapshots = [[1, 0, 1, 0], [1, 1, 1, 1]]

        def gather(cohort_hash, _local_status):
            return [(cohort_hash, status) for status in snapshots.pop(0)]

        coordinator = ExplicitPrefillCohortAdmissionCoordinator(
            enabled=True, dp_size=4, gather_status=gather
        )
        req = make_req()
        coordinator.register(req)
        self.assertEqual(coordinator.stage_and_release([req], []), [])
        self.assertEqual(coordinator.stage_and_release([], []), [req])

    def test_unmarked_and_partial_dp_requests_pass_through(self):
        coordinator = ExplicitPrefillCohortAdmissionCoordinator(
            enabled=True,
            dp_size=4,
            gather_status=lambda *_: self.fail("unexpected collective"),
        )
        requests = [SimpleNamespace(rid="plain"), make_req(size=2)]
        for req in requests:
            coordinator.register(req)
        self.assertEqual(coordinator.stage_and_release(requests, []), requests)

    def test_releases_all_local_members_together(self):
        local_statuses = []
        snapshots = [[1, 1, 1, 1], [2, 2, 2, 2]]

        def gather(cohort_hash, local_status):
            local_statuses.append(local_status)
            return [(cohort_hash, status) for status in snapshots.pop(0)]

        coordinator = ExplicitPrefillCohortAdmissionCoordinator(
            enabled=True, dp_size=4, gather_status=gather
        )
        requests = [make_req(size=8, ordinal=ordinal) for ordinal in range(2)]
        for req in requests:
            coordinator.register(req)
        self.assertEqual(coordinator.stage_and_release(requests[:1], []), [])
        self.assertEqual(coordinator.stage_and_release(requests[1:], []), requests)
        self.assertEqual(local_statuses, [1, 2])

    def test_mismatch_fails_and_bootstrap_failure_releases(self):
        mismatch = ExplicitPrefillCohortAdmissionCoordinator(
            enabled=True,
            dp_size=4,
            gather_status=lambda cohort_hash, _: [
                (cohort_hash, 1),
                (cohort_hash + 1, 1),
            ],
        )
        req = make_req()
        mismatch.register(req)
        with self.assertRaisesRegex(RuntimeError, "order diverged"):
            mismatch.stage_and_release([req], [])

        failed = ExplicitPrefillCohortAdmissionCoordinator(
            enabled=True,
            dp_size=4,
            gather_status=lambda cohort_hash, _: [(cohort_hash, -1), (cohort_hash, 0)],
        )
        req = make_req()
        failed.register(req)
        failed.mark_failed(req)
        self.assertEqual(failed.stage_and_release([req], []), [req])


if __name__ == "__main__":
    unittest.main()
