"""CPU coverage of explicit KV+Mamba host budgets without allocating pools."""

import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt import runtime_context as rc
from sglang.srt.arg_groups.hicache_hook import (
    handle_hicache,
    handle_hicache_ratio_default,
    validate_hicache_mamba_fraction,
)
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.pool_host import base, mamba
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _StopBeforeAllocation(Exception):
    pass


class TestHicacheMambaFraction(unittest.TestCase):
    def test_default_explicit_budget_rounding_and_resolution(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        cli = parser.parse_args(
            ["--model-path", "local-hybrid", "--hicache-mamba-fraction", "0.26"]
        )
        self.assertEqual(cli.hicache_mamba_fraction, 0.26)
        self.assertIsNone(ServerArgs(model_path="dummy").hicache_mamba_fraction)

        args = ServerArgs(
            model_path="local-hybrid",
            enable_hierarchical_cache=True,
            hicache_size=374,
            hicache_mamba_fraction=0.26,
        )
        args._resolved_overrides = [("fixture-model", {"uses_mamba_radix_cache": True})]
        args._model_config = SimpleNamespace(is_hybrid_swa=False)
        validate_hicache_mamba_fraction(args)
        for name, value in (
            ("enable_hierarchical_cache", False),
            ("hicache_size", 0),
            ("disable_radix_cache", True),
            ("radix_cache_backend", "custom"),
            ("enable_lmcache", True),
            ("enable_unified_memory", True),
            ("enable_page_major_kv_layout", True),
            ("disaggregation_mode", "decode"),
            ("hicache_mamba_fraction", 0.0),
            ("hicache_mamba_fraction", 1.0),
            ("hicache_mamba_fraction", float("nan")),
            ("hicache_mamba_fraction", float("inf")),
        ):
            with self.subTest(rejected_field=name, value=value):
                old = getattr(args, name)
                try:
                    setattr(args, name, value)
                    with self.assertRaisesRegex(ValueError, "hicache-mamba-fraction"):
                        handle_hicache_ratio_default(args)
                finally:
                    setattr(args, name, old)
        args._resolved_overrides = []
        with self.assertRaisesRegex(ValueError, "hybrid-Mamba"):
            handle_hicache(args)
        args._resolved_overrides = [("fixture-model", {"uses_mamba_radix_cache": True})]
        args._model_config.is_hybrid_swa = True
        with self.assertRaisesRegex(ValueError, "KV.only and KV.SWA"):
            handle_hicache(args)
        args.hicache_mamba_fraction = None
        validate_hicache_mamba_fraction(args)  # No new restrictions when unset.
        with self.assertRaisesRegex(ValueError, "built-in KV.Mamba"):
            ServerArgs(
                model_path="dummy",
                enable_hierarchical_cache=True,
                hicache_size=374,
                hicache_mamba_fraction=0.26,
            ).resolve_once()

        kv = SimpleNamespace(
            host_capacity_bytes=(60 * 10**9, 40 * 10**9),
            get_kv_size_bytes=lambda: 999,  # host_capacity_bytes wins.
            store_dtype=torch.bfloat16,
            size=10,
            start_layer=0,
            end_layer=1,
        )
        state = SimpleNamespace(
            get_kv_size_bytes=lambda: 60 * 10**9,
            num_mamba_layers=36,
            size=10,
            mamba_cache=SimpleNamespace(
                conv=[SimpleNamespace(shape=(36, 1, 3, 10240), dtype=torch.bfloat16)],
                temporal=SimpleNamespace(
                    shape=(36, 1, 48, 128, 128), dtype=torch.bfloat16
                ),
            ),
        )
        old_shares = assembler._split_hicache_size(374, (kv, state))
        self.assertEqual(old_shares, (233.75, 140.25))
        for fraction, expected in ((None, old_shares), (0.26, (276.76, 97.24))):
            with self.subTest(fraction=fraction):
                shares = assembler._split_mamba_hicache_size(374, kv, state, fraction)
                for actual, reference in zip(shares, expected):
                    self.assertAlmostEqual(actual, reference, places=12)
                self.assertAlmostEqual(sum(shares) * 10**9, 374 * 10**9, places=4)

                # Exercise the production caller, not only the arithmetic helper.
                params = SimpleNamespace(
                    req_to_token_pool=SimpleNamespace(mamba_allocator=object()),
                    mtp_draft_device_pools=(),
                    page_size=64,
                )
                with rc.get_context().override_server_args(
                    hicache_size=374, hicache_mamba_fraction=fraction
                ), patch.object(
                    assembler, "build_kv_host_pool", side_effect=_StopBeforeAllocation
                ) as build:
                    with self.assertRaises(_StopBeforeAllocation):
                        assembler.build_hybrid_mamba_stack(
                            params=params, kv_pool=kv, mamba_pool=state,
                            full_layer_mapping={0: 0}, mamba_layer_mapping={1: 0},
                            load_cache_event=None, storage_backend=None, use_mla=False,
                        )
                    self.assertEqual(build.call_args.kwargs["host_size"], shares[0])

                # Execute actual pool constructors through their sizing and page
                # rounding. Intercept the budget check BEFORE any buffer allocation.
                sizes = []
                for module, share, bpt, page in (
                    (base, shares[0], 13312, 64),
                    (mamba, shares[1], 58834944, 1),
                ):
                    with patch.object(
                        module, "host_memory_budget_bytes",
                        side_effect=_StopBeforeAllocation,
                    ) as budget, patch.object(
                        module, "sync_fixed_hicache_size",
                        side_effect=lambda size, _: size,
                    ), patch.object(module, "get_allocator_from_storage", return_value=None):
                        with self.assertRaises(_StopBeforeAllocation):
                            if module is base:
                                probe = SimpleNamespace(get_size_per_token=lambda: bpt)
                                base.HostKVCache.__init__(
                                    probe, kv, 2.0, share, page, "page_first", False, "cpu"
                                )
                            else:
                                probe = object.__new__(mamba.MambaPoolHost)
                                mamba.MambaPoolHost.__init__(
                                    probe, state, 2.0, share,
                                    pin_memory=False, layout="page_first",
                                )
                        expected_rows = (int(share * 1e9 // bpt) // page + 1) * page
                        self.assertEqual(probe.size_per_token, bpt)
                        self.assertEqual(probe.size, expected_rows)
                        self.assertEqual(probe.size % page, 0)
                        actual_bytes = expected_rows * bpt
                        budget.assert_called_once_with(actual_bytes)
                        self.assertGreater(actual_bytes, share * 1e9)
                        self.assertLessEqual(actual_bytes - share * 1e9, page * bpt)
                        sizes.append(actual_bytes)
                self.assertGreater(sum(sizes), 374 * 10**9)
                self.assertLessEqual(
                    sum(sizes) - 374 * 10**9, 64 * 13312 + 58834944
                )


if __name__ == "__main__":
    unittest.main()
