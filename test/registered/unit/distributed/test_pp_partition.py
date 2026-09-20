import os
import unittest
from unittest.mock import patch

from sglang.srt.arg_groups.speculative_hook import pp_draft_virtual_layers
from sglang.srt.distributed.utils import _auto_pp_partition, get_pp_indices
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _sizes(num_layers, pp_size, virtual_last=0):
    return [
        e - s
        for s, e in (
            _auto_pp_partition(num_layers, r, pp_size, virtual_last)
            for r in range(pp_size)
        )
    ]


class TestPPPartition(CustomTestCase):
    def test_legacy_even_split_gives_remainder_to_last_stages(self):
        self.assertEqual(_sizes(43, 4), [10, 11, 11, 11])
        self.assertEqual(_sizes(40, 4), [10, 10, 10, 10])
        self.assertEqual(_sizes(45, 4), [11, 11, 11, 12])

    def test_virtual_last_moves_remainder_forward_and_lightens_last_stage(self):
        # DeepSeek-V4-Flash: 43 layers, 1 MTP layer + head on the last stage.
        self.assertEqual(_sizes(43, 4, virtual_last=1), [11, 11, 11, 10])
        self.assertEqual(
            _sizes(43, 4, virtual_last=2), [11, 11, 11, 10]
        )  # 12,11,11,9 would make stage 0 the new bottleneck
        # GLM-5.3-Flash: 45 layers.
        self.assertEqual(_sizes(45, 4, virtual_last=1), [12, 11, 11, 11])
        self.assertEqual(
            _sizes(60, 4, virtual_last=2), [16, 15, 15, 14]
        )  # ties keep the most balanced real split
        # Even count still balances.
        self.assertEqual(
            _sizes(40, 4, virtual_last=1), [10, 10, 10, 10]
        )  # already balanced within one layer
        self.assertEqual(
            _sizes(40, 4, virtual_last=3), [11, 11, 10, 8]
        )  # last+3 == 11 == max head

    def test_partition_covers_every_layer_once(self):
        for layers in (37, 40, 43, 45, 60, 93):
            for pp in (2, 3, 4, 6, 8):
                for v in (0, 1, 2, 3):
                    sizes = _sizes(layers, pp, v)
                    self.assertEqual(sum(sizes), layers, (layers, pp, v, sizes))
                    self.assertTrue(
                        all(sz >= 1 for sz in sizes), (layers, pp, v, sizes)
                    )
                    bounds = [_auto_pp_partition(layers, r, pp, v) for r in range(pp)]
                    self.assertEqual(bounds[0][0], 0)
                    for (s0, e0), (s1, _) in zip(bounds, bounds[1:]):
                        self.assertEqual(e0, s1)

    def test_get_pp_indices_honors_env_knob(self):
        with patch.dict(os.environ, {"SGLANG_PP_LAST_STAGE_VIRTUAL_LAYERS": "1"}):
            os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)
            self.assertEqual(
                [get_pp_indices(43, r, 4) for r in range(4)],
                [(0, 11), (11, 22), (22, 33), (33, 43)],
            )

    def test_virtual_layer_count_from_draft_config(self):
        self.assertEqual(pp_draft_virtual_layers(None), 2)
        self.assertEqual(pp_draft_virtual_layers(1), 2)
        self.assertEqual(pp_draft_virtual_layers(3), 4)

    def test_explicit_partition_list_wins(self):
        with patch.dict(
            os.environ,
            {
                "SGLANG_PP_LAYER_PARTITION": "8,6,6,20",
                "SGLANG_PP_LAST_STAGE_VIRTUAL_LAYERS": "3",
            },
        ):
            self.assertEqual(get_pp_indices(40, 3, 4), (20, 40))


if __name__ == "__main__":
    unittest.main()
