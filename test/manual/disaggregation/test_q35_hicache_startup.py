"""Four-process CPU/Gloo regression for the experimental startup budget."""

import datetime
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.multiprocessing as mp
from sglang.srt import runtime_context
from sglang.srt.mem_cache.pool_host import base


def _rank(rank, rendezvous, directory, remaining, enabled):
    torch.distributed.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=45),
    )
    parallel = SimpleNamespace(
        nnodes=1,
        launch_world_size=4,
        pp_group=SimpleNamespace(world_size=4, cpu_group=torch.distributed.group.WORLD),
    )
    args = SimpleNamespace(disaggregation_mode="prefill")

    def available():
        with remaining.get_lock():
            return remaining.value

    def allocate(requested):
        budget = base.host_memory_budget_bytes(requested)
        if requested > budget:
            raise ValueError("insufficient budget")
        with remaining.get_lock():
            remaining.value -= requested
        return budget

    result = {"rank": rank, "enabled": enabled, "allocated": [], "error": None}
    try:
        with (
            patch.dict(
                os.environ,
                {
                    "SGLANG_Q35_PREFILL_HOST_BUDGET": str(int(enabled)),
                    "SGLANG_Q35_PREFILL_HOST_REQUIRED_BYTES": "573000000000",
                },
            ),
            patch.object(base, "get_parallel", return_value=parallel),
            patch.object(
                runtime_context,
                "get_memory",
                return_value=SimpleNamespace(hicache_host_memory_fraction=None),
            ),
            patch.object(base, "available_host_memory_bytes", side_effect=available),
        ):
            # Deliberately let early ranks allocate first in the old path.
            time.sleep(rank * 0.2)
            with base.q35_prefill_host_budget(args):
                result["snapshot"] = base.host_memory_budget_bytes()
                for requested in (
                    109_350_000_000,
                    27_027_000_000 if rank == 3 else 36_036_000_000,
                ):
                    result["allocated"].append(allocate(requested))
                result["remaining_budget"] = base.host_memory_budget_bytes()
    except ValueError as error:
        result["error"] = str(error)
    finally:
        result["scope_reset"] = base._host_memory_budget.get() is None
        Path(directory, f"{rank}.json").write_text(json.dumps(result))
        torch.distributed.destroy_process_group()


class TestQ35HiCacheStartup(unittest.TestCase):
    def run_ranks(self, total, enabled):
        with tempfile.TemporaryDirectory(prefix="q35-host-budget-") as directory:
            remaining = mp.get_context("spawn").Value("q", total)
            mp.spawn(
                _rank,
                args=(
                    "file://" + directory + "/rendezvous",
                    directory,
                    remaining,
                    enabled,
                ),
                nprocs=4,
            )
            rows = [
                json.loads(Path(directory, f"{rank}.json").read_text())
                for rank in range(4)
            ]
            self.assertTrue(all(r["scope_reset"] for r in rows))
            return rows, remaining.value

    def test_staggered_rank_allocations(self):
        old, _ = self.run_ranks(660_000_000_000, False)
        self.assertTrue(any(r["error"] for r in old))
        new, remaining = self.run_ranks(660_000_000_000, True)
        self.assertTrue(all(r["error"] is None for r in new), new)
        budget = (660_000_000_000 - base.HICACHE_HOST_MEMORY_RESERVE_BYTES) // 4
        self.assertEqual([r["snapshot"] for r in new], [budget] * 4)
        self.assertEqual(
            [r["remaining_budget"] for r in new],
            [budget - 145_386_000_000] * 3 + [budget - 136_377_000_000],
        )
        self.assertEqual(remaining, 87_465_000_000)

    def test_insufficient_budget_still_fails(self):
        rows, remaining = self.run_ranks(400_000_000_000, True)
        self.assertTrue(all("below demand +10%" in (r["error"] or "") for r in rows))
        self.assertEqual(remaining, 400_000_000_000)

    def test_disabled_path_never_reads_parallel(self):
        with (
            patch.dict(os.environ, {"SGLANG_Q35_PREFILL_HOST_BUDGET": "0"}),
            patch.object(
                base, "get_parallel", side_effect=AssertionError("must not run")
            ),
            base.q35_prefill_host_budget(None),
        ):
            pass

    def test_other_topologies_fail_closed(self):
        with (
            patch.dict(os.environ, {"SGLANG_Q35_PREFILL_HOST_BUDGET": "1"}),
            patch.object(base, "get_parallel", return_value=SimpleNamespace(nnodes=2)),
        ):
            with self.assertRaisesRegex(ValueError, "single-host PP4"):
                with base.q35_prefill_host_budget(
                    SimpleNamespace(disaggregation_mode="prefill")
                ):
                    self.fail("unsupported topology entered allocation scope")


if __name__ == "__main__":
    unittest.main()
