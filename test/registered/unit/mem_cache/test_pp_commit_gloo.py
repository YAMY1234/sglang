"""Full bridge + v3.2 chain + real CPU/Gloo, with and without TP."""

import multiprocessing
import tempfile
import time
import types
import unittest
from datetime import timedelta

import torch.distributed as dist
from test_pp_commit_integration import Cluster, bridge_module, method


def worker(global_rank, tp, path, output):
    world = 3 * tp
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        rank=global_rank,
        world_size=world,
        timeout=timedelta(seconds=30),
    )
    own_pp = own_tp = control = None
    for lane in range(tp):
        ranks = list(range(lane, world, tp))
        pp = dist.new_group(ranks, backend="gloo")
        reverse = dist.new_group(ranks, backend="gloo")
        if global_rank in ranks:
            own_pp, control = pp, reverse
    for stage in range(3):
        ranks = list(range(stage * tp, (stage + 1) * tp))
        group = dist.new_group(ranks, backend="gloo")
        if global_rank in ranks:
            own_tp = group
    stage = global_rank // tp
    local = Cluster(enabled=False)
    c = local.caches[stage]
    c.pp_group, c.tp_group, c.tp_world_size = own_pp, own_tp, tp
    c.work_list = []
    c._pp_sync_stats = {
        "calls": 0,
        "sent": 0,
        "recv": 0,
        "pending": 0,
        "pending_peak": 0,
        "warned": False,
    }
    for name in ("_pp_sync", "_drain_async_work"):
        setattr(
            c,
            name,
            types.MethodType(method("mem_cache/unified_radix_cache.py", name), c),
        )
    c._all_reduce_attn_groups = lambda tensor, op: dist.all_reduce(
        tensor, op=op, group=own_tp
    )
    c._pp_commit = bridge_module.PPCommitBridge(c, control)
    c.cache_controller.pp_commit_bridge = c._pp_commit
    c.storage_existence_cache.defer_mutation = c._pp_commit.defer_belief
    applied_rounds = []
    original_unlock = c.dec_host_lock_ref

    def unlock(node, lock):
        applied_rounds.append(c._pp_commit.state.round)
        original_unlock(node, lock)

    c.dec_host_lock_ref = unlock
    try:
        for step in range(35):
            if step == (6 if global_rank == world - 1 else global_rank % tp):
                local.backup(stage)
                local.drain(stage)
            c._drain_async_work()
            c._pp_commit.tick()
            time.sleep(0.01)
        c._drain_async_work()
        output.put(
            (
                global_rank,
                applied_rounds,
                c._pp_commit.state.snapshot(),
                c._pp_sync_stats["calls"],
                len(c.work_list),
            )
        )
        c._pp_commit.reports.close(2)
        dist.barrier()
        if not c._pp_commit.reports.close(2):
            raise AssertionError("READY workers failed to close")
    finally:
        dist.destroy_process_group()


class FullGlooTest(unittest.TestCase):
    def run_topology(self, tp):
        context = multiprocessing.get_context("spawn")
        output = context.Queue()
        with tempfile.TemporaryDirectory() as directory:
            processes = [
                context.Process(
                    target=worker, args=(rank, tp, directory + "/init", output)
                )
                for rank in range(3 * tp)
            ]
            for process in processes:
                process.start()
            try:
                rows = [output.get(timeout=45) for _ in processes]
                for process in processes:
                    process.join(10)
                    self.assertEqual(process.exitcode, 0)
                self.assertTrue(all(row[2]["applied"] == 1 for row in rows))
                self.assertTrue(all(row[2]["commit_stall"] == 0 for row in rows))
                self.assertTrue(all(row[3:] == (70, 0) for row in rows))
                self.assertEqual(len({tuple(row[1]) for row in rows}), 1)
                self.assertGreaterEqual(rows[0][1][0], 8)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(5)

    def test_pp3_tp1_real_forward_and_reverse(self):
        self.run_topology(1)

    def test_pp3_tp2_late_tp_ack_cannot_commit_early(self):
        self.run_topology(2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
