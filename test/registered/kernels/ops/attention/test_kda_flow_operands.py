import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestKdaFlowOperands(CustomTestCase):
    """KdaFlow hands layer parameters (A_log, dt_bias, the norm weight: nn.Parameter,
    requires grad, even under no_grad) to CuTe through DLPack, which refuses tensors
    that require grad; the operands must leave detached."""

    def test_parameters_export(self):
        try:
            from sglang.kernels.ops.attention.kda_flow import cute_views
        except ImportError as exc:  # no CuTe DSL in this environment
            raise unittest.SkipTest(str(exc))
        a_log = torch.nn.Parameter(torch.randn(24))
        dt_bias = torch.nn.Parameter(torch.randn(24 * 128))
        state = torch.zeros(3, 24, 128, 128)
        with torch.no_grad():
            views = cute_views((a_log, dt_bias, state, -5.0))
        self.assertEqual(len(views), 4)
        self.assertTrue(a_log.requires_grad)


if __name__ == "__main__":
    unittest.main()
