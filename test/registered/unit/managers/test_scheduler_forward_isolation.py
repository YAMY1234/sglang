"""Field-descriptor reuse must preserve transactional snapshots and pinning."""

import dataclasses
import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


@dataclasses.dataclass
class ExtendedBatch(ScheduleBatch):
    extra_input: object = None


class TestForwardIsolation(CustomTestCase):
    def test_current_values_and_subclass_fields_survive_exception(self):
        # Reusing descriptors must not reuse a previous forward's values, omit
        # subclass fields, or drop the pinned references on the exception path.
        for enabled in (False, True):
            for batch_type in (ScheduleBatch, ExtendedBatch):
                with (
                    self.subTest(enabled=enabled, batch_type=batch_type),
                    envs.SGLANG_ENABLE_EAGLE_PREPARE_REUSE.override(enabled),
                ):
                    scheduler = Scheduler.__new__(Scheduler)
                    scheduler.spec_algorithm = SpeculativeAlgorithm.EAGLE
                    scheduler.init_eagle_prepare_reuse()
                    scheduler.batch_record_ct = 0
                    scheduler.batch_record_buf = [None, None]
                    batch = batch_type(reqs=[])
                    batch.spec_algorithm = SpeculativeAlgorithm.EAGLE
                    for value in (3, 7):
                        original = torch.tensor([value])
                        batch.seq_lens = original
                        if batch_type is ExtendedBatch:
                            batch.extra_input = original
                        with self.assertRaisesRegex(RuntimeError, "forward failed"):
                            with scheduler._forward_isolation(batch, overlap=True):
                                batch.seq_lens = torch.tensor([99])
                                if batch_type is ExtendedBatch:
                                    batch.extra_input = None
                                raise RuntimeError("forward failed")
                        self.assertIs(batch.seq_lens, original)
                        if batch_type is ExtendedBatch:
                            self.assertIs(batch.extra_input, original)
                        pinned = scheduler.batch_record_buf[scheduler.batch_record_ct][
                            1
                        ]
                        self.assertTrue(any(item is original for item in pinned))


if __name__ == "__main__":
    unittest.main()
