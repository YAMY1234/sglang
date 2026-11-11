"""CPU tests for the causal_conv1d forward launch configuration."""

import importlib
import unittest

import torch

from sglang.kernels.ops.mamba import causal_conv1d_triton as conv
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def reload_with(value, master=True):
    with (
        envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(master),
        envs.SGLANG_CAUSAL_CONV1D_FWD_CONFIG.override(value),
    ):
        importlib.reload(conv)


class TestCausalConv1dLaunchConfig(CustomTestCase):
    def tearDown(self):
        importlib.reload(conv)

    def test_unset_or_master_off_keeps_the_default_launch(self):
        for value, master in (("", True), ("0", True), ("16,256,1,3", False)):
            with self.subTest(value=value, master=master):
                reload_with(value, master)
                self.assertEqual(
                    conv._fwd_launch_config(None, 4), conv._STOCK_FWD_LAUNCH
                )
        self.assertEqual(
            conv._STOCK_FWD_LAUNCH,
            {
                "BLOCK_M": conv.CAUSAL_CONV1D_FWD_BLOCK_M,
                "BLOCK_N": 256,
                "num_stages": 2,
            },
        )

    def test_config_applies_only_to_the_rectangular_launch(self):
        reload_with("16,256,1,3")
        tile = {"BLOCK_M": 16, "BLOCK_N": 256, "num_warps": 1, "num_stages": 3}
        for width in (2, 3, 4):
            self.assertEqual(conv._fwd_launch_config(None, width), tile)
        # A block table counts chunks of CAUSAL_CONV1D_FWD_BLOCK_M tokens, and the
        # tap chain covers widths 2-4 only: both keep the default tile.
        block_table = torch.zeros(4, 2, dtype=torch.int32)
        self.assertEqual(
            conv._fwd_launch_config(block_table, 4), conv._STOCK_FWD_LAUNCH
        )
        self.assertEqual(conv._fwd_launch_config(None, 5), conv._STOCK_FWD_LAUNCH)

    def test_out_of_range_tiles_fail_at_import(self):
        # BLOCK_M below 8 would let a chunk's prior-token reads cross into the
        # previous sequence; BLOCK_N must cover at least one element per thread.
        for value in (
            "4,256,2,2",
            "8,32,2,2",
            "8,384,2,2",
            "8,256,3,2",
            "8,256,2",
            "a,b,c,d",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                reload_with(value)


class TestPrefillConvStrip(CustomTestCase):
    def tearDown(self):
        importlib.reload(conv)

    def _reload(self, value, enabled=True, master=True):
        with (
            envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(master),
            envs.SGLANG_OPT_GDN_PREFILL_CONV_STRIP.override(enabled),
            envs.SGLANG_GDN_PREFILL_CONV_RETILE.override(value),
        ):
            importlib.reload(conv)

    def test_switch_and_value_select_the_strip_tile(self):
        default = conv._PrefillConvStrip(32, 128, 4, 2048)
        for value, enabled, master, expected in (
            ("", True, True, default),
            ("1", True, True, default),
            ("0", True, True, None),
            ("64,256,8", True, True, conv._PrefillConvStrip(64, 256, 8, 2048)),
            ("16,64,2,512", True, True, conv._PrefillConvStrip(16, 64, 2, 512)),
            ("1", False, True, None),
            ("1", True, False, None),
        ):
            with self.subTest(value=value, enabled=enabled, master=master):
                self._reload(value, enabled, master)
                self.assertEqual(conv._PREFILL_CONV_STRIP, expected)
        for value in ("12,128,4", "32,96,4", "32,128,3", "8,128,16", "x"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self._reload(value)

    def test_short_block_table_or_unsupported_calls_keep_the_triton_launch(self):
        self._reload("1")
        x = torch.zeros(1280, 4096, dtype=torch.bfloat16)
        args = dict(
            x=x,
            weight=torch.zeros(1280, 4, dtype=torch.bfloat16),
            bias=None,
            conv_states=torch.zeros(2, 1280, 3, dtype=torch.bfloat16),
            query_start_loc=torch.tensor([0, 4096], dtype=torch.int32),
            seq_lens_cpu=[4096],
            q_dim=256,
            k_dim=256,
        )
        short = dict(args, x=x[:, :2047])
        self.assertIsNone(conv._prefill_conv_strip(**short, kwargs={}))
        table = {"block_table": torch.zeros(512, 2, dtype=torch.int32)}
        self.assertIsNone(conv._prefill_conv_strip(**args, kwargs=table))
        # A host tensor is not one the strip kernel serves.
        self.assertIsNone(conv._prefill_conv_strip(**args, kwargs={}))


if __name__ == "__main__":
    unittest.main()
