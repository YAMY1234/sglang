"""T64: the AIR top-p dispatch under the mid-curve decode layout (TP4, no DP attention).

The EAGLE target verify of a TP4 decode takes the deterministic renorm (tp_size > 1), so with
SGLANG_OPT_AIR_TOP_P_PARALLEL the parallel kernel receives bs x 6 rows; without the flag (the
default) the flashinfer call is unchanged; a single-rank worker never reaches either.
CUDA calls are mocked (CPU test); dispatch, policy, env registry and input checks are real.
"""

import os
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.sampling import air_top_p_parallel as air
from sglang.srt.environ import envs
from sglang.srt.layers import sampling_renorm
from sglang.srt.runtime_context import get_context, get_parallel, reset_context
from sglang.srt.speculative import eagle_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_air_top_p_dispatch import _CudaProbs  # noqa: E402

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_TP4 = dict(tp_size=4)
DRAFT = 6


class TestAirTopPDispatchTP4(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.fi = mock.patch.object(
            sampling_renorm._flashinfer_sampling, "top_p_renorm_probs", return_value="fi"
        ).start()
        self.par = mock.patch.object(
            sampling_renorm, "air_top_p_renorm_parallel", return_value="parallel"
        ).start()
        self.fast = mock.patch.object(
            sampling_renorm._platform, "top_p_renorm_prob", return_value="fast"
        ).start()
        self.addCleanup(mock.patch.stopall)
        self.enterContext(mock.patch.dict(air._bypass_totals, clear=True))

    def test_default_is_off(self):
        # Default-off contract of the lever: neither key set means the stock kernel.
        with mock.patch.dict(os.environ, clear=False):
            os.environ.pop("SGLANG_OPT_AIR_TOP_P_PARALLEL", None)
            os.environ.pop("SGLANG_AIR_TOP_P_REQUIRED_TARGET", None)
            self.assertFalse(envs.SGLANG_OPT_AIR_TOP_P_PARALLEL.get())
            self.assertEqual(envs.SGLANG_AIR_TOP_P_REQUIRED_TARGET.get(), "")
            with get_context().override_server_args(**_TP4):
                probs, top_p = _CudaProbs(rows=24), torch.full((24,), 0.95)
                self.assertEqual(sampling_renorm._top_p_renorm_prob(probs, top_p, parallel=True), "fi")
        self.par.assert_not_called()

    def test_tp4_verify_rows_take_parallel(self):
        # bs 1..4 running requests x 6 draft tokens, top_ps repeated per draft token as the verify does.
        for bs in (1, 2, 3, 4):
            self.par.reset_mock()
            info = SimpleNamespace(
                need_top_p_sampling=True, need_top_k_sampling=False, top_ps=torch.full((bs,), 0.95)
            )
            probs = _CudaProbs(rows=bs * DRAFT)
            with mock.patch.object(eagle_utils, "top_p_renorm_prob", sampling_renorm._top_p_renorm_prob):
                with envs.SGLANG_OPT_AIR_TOP_P_PARALLEL.override(True):
                    with get_context().override_server_args(**_TP4):
                        self.assertEqual(get_parallel().tp_size, 4)
                        self.assertEqual(get_parallel().attn_tp_size, 4)
                        self.assertTrue(sampling_renorm.renorm_deterministic())
                        out = eagle_utils._apply_joint_top_k_top_p(
                            probs, info, DRAFT, None, eagle_utils._verify_top_p_renorm_prob
                        )
            self.assertEqual(out, "parallel")
            (got_probs, got_top_p), _ = self.par.call_args
            self.assertIs(got_probs, probs)
            self.assertEqual(tuple(got_top_p.shape), (bs * DRAFT,))
        self.fi.assert_not_called()
        self.assertEqual(air._bypass_totals, {})

    def test_tp4_flag_off_keeps_flashinfer(self):
        probs, top_p = _CudaProbs(rows=24), torch.full((24,), 0.95)
        with envs.SGLANG_OPT_AIR_TOP_P_PARALLEL.override(False):
            with get_context().override_server_args(**_TP4):
                self.assertEqual(eagle_utils_dispatch(probs, top_p), "fi")
        self.fi.assert_called_once_with(probs, top_p, is_deterministic=True)
        self.par.assert_not_called()

    def test_single_rank_never_reaches_air(self):
        # tp 1: nobody to agree with, the fast kernel runs and the lever is inert.
        probs, top_p = _CudaProbs(rows=24), torch.full((24,), 0.95)
        with envs.SGLANG_OPT_AIR_TOP_P_PARALLEL.override(True):
            with get_context().override_server_args(tp_size=1):
                self.assertFalse(sampling_renorm.renorm_deterministic())
                self.assertEqual(eagle_utils_dispatch(probs, top_p), "fast")
        self.par.assert_not_called()
        self.fi.assert_not_called()


def eagle_utils_dispatch(probs, top_p):
    return sampling_renorm._top_p_renorm_prob(probs, top_p, parallel=True)


if __name__ == "__main__":
    unittest.main()
