import unittest
from unittest.mock import patch

from sglang.srt.layers.moe.token_dispatcher.flashinfer_utils import (
    TorchDistributedCommBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class _FakeProcessGroup:
    def rank(self):
        return 1

    def size(self):
        return 4


class TestTorchDistributedCommBackend(unittest.TestCase):
    def test_bcast_translates_group_local_root_to_global_rank(self):
        group = _FakeProcessGroup()
        backend = TorchDistributedCommBackend(group)

        def fake_broadcast_object_list(obj_list, *, src, group):
            self.assertEqual(src, 8)
            self.assertIs(group, backend._group)
            obj_list[0] = "from-group-root"

        with (
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer_utils.dist.get_global_rank",
                return_value=8,
            ) as get_global_rank,
            patch(
                "sglang.srt.layers.moe.token_dispatcher.flashinfer_utils.dist.broadcast_object_list",
                side_effect=fake_broadcast_object_list,
            ) as broadcast_object_list,
        ):
            result = backend.bcast("local-value", root=0)

        get_global_rank.assert_called_once_with(group, 0)
        broadcast_object_list.assert_called_once()
        self.assertEqual(result, "from-group-root")


if __name__ == "__main__":
    unittest.main()
