"""Regression for holey Mamba endpoints and private asynchronous reservations."""

import queue
import threading
import types
import unittest
from unittest.mock import patch
import torch
from test_pp_commit_integration import load, method

boundary = load(
    "sglang.srt.mem_cache.prefetch_boundaries", "mem_cache/prefetch_boundaries.py"
)


class MinGroup:
    def __init__(self, n):
        self.frames = [None] * n
        self.barrier = threading.Barrier(n, action=self.finish, timeout=5)
        self.leave = threading.Barrier(n, timeout=5)

    def finish(self):
        result = torch.stack(self.frames).amin(0)
        for frame in self.frames:
            frame.copy_(result)

    def reduce(self, rank):
        def run(frame):
            self.frames[rank] = frame
            self.barrier.wait()
            self.leave.wait()

        return run


class Host:
    def __init__(self, pages):
        self.free_pages, self.freed = pages, []

    def alloc(self, tokens):
        if tokens > self.available_size():
            return None
        self.free_pages -= tokens // 64
        return torch.arange(tokens)

    def available_size(self):
        return self.free_pages * 64

    def free(self, indices):
        self.freed.append(len(indices))
        self.free_pages += len(indices) // 64


def operation(rid="same", valid=(3, 7)):
    return types.SimpleNamespace(
        request_id=rid,
        restorable_prefix_pages=list(valid),
        storage_hit_count=max(valid, default=0) * 64,
        hash_value=list(range(max(valid, default=0))),
        host_indices=None,
        is_terminated=lambda: False,
        pool_transfers_done=False,
    )


class BoundariesTest(unittest.TestCase):
    def intersection(self, sets):
        group, answers, errors = MinGroup(len(sets)), {}, []

        def run(rank, valid):
            try:
                answers[rank] = boundary.intersect_boundaries(
                    operation(), valid, 7, group.reduce(rank)
                )
            except Exception as error:
                errors.append(error)

        threads = [
            threading.Thread(target=run, args=(i, s)) for i, s in enumerate(sets)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        self.assertFalse(errors)
        self.assertFalse(any(t.is_alive() for t in threads))
        return list(answers.values())

    def test_rank_holes_intersection_not_minimum_of_maxima(self):
        self.assertEqual(self.intersection([[3, 7], [3, 5], [1, 3, 7]]), [[3]] * 3)
        self.assertNotIn(min(7, 5, 7), [3, 7])

    def test_one_empty_rank_has_no_common_endpoint(self):
        self.assertEqual(self.intersection([[3, 7], []]), [[], []])

    def caches(self, capacities):
        group, stop, peers, ops = MinGroup(len(capacities)), threading.Event(), [], []
        for rank, capacity in enumerate(capacities):
            state = boundary.PrefetchAllocConsensus(group.reduce(rank), stop, 64)
            cc = types.SimpleNamespace(
                prefetch_alloc_consensus=state,
                mem_pool_host=Host(capacity),
                prefetch_buffer=queue.Queue(),
            )
            info = types.SimpleNamespace(
                _replace=lambda **kw: types.SimpleNamespace(**kw)
            )
            c = types.SimpleNamespace(
                cache_controller=cc,
                page_size=64,
                prefetch_threshold=64,
                ongoing_prefetch={"same": info},
                _storage_prefetch_missed_rids=set(),
                _invalidate_absent_from_hit_query=lambda op: None,
                _record_storage_prefetch_hit=lambda *a: None,
                _account_prefetch_outcome=lambda *a, **kw: None,
                discard_storage_prefetch_accounting=lambda *a: None,
                evict_host=lambda size: None,
                _resolve_storage_prefetch_tokens=lambda *a, **kw: None,
                _finish_storage_prefetch=lambda *a, **kw: None,
            )
            c.revoke_pending_prefetch = lambda rid, c=c: c.ongoing_prefetch.pop(
                rid, None
            )
            peers.append(c)
            ops.append(operation())

        def cleanup():
            stop.set()
            for c in peers:
                t = c.cache_controller.prefetch_alloc_consensus.thread
                t.join(6)
                self.assertFalse(t.is_alive())

        self.addCleanup(cleanup)
        return peers, ops

    def reserve(self, c, op):
        method("mem_cache/unified_radix_cache.py", "_reserve_prefetch_boundary")(c, op)

    def complete(self, c):
        cc = c.cache_controller
        op, pages = cc.prefetch_alloc_consensus.ready.get(timeout=5)
        method("mem_cache/unified_radix_cache.py", "_finish_prefetch_boundary")(
            c, op, pages
        )
        cc.prefetch_alloc_consensus.assert_idle()
        return pages

    def test_host_cut_common_legal_grant_and_private_tail_rollback(self):
        peers, ops = self.caches([5, 7, 6])
        for c, op in zip(peers, ops):
            self.reserve(c, op)
        self.assertEqual([self.complete(c) for c in peers], [3, 3, 3])
        for c, op, cap in zip(peers, ops, [5, 7, 6]):
            self.assertEqual(len(op.host_indices), 192)
            self.assertEqual(len(op.hash_value), 3)
            self.assertEqual(c.cache_controller.mem_pool_host.free_pages, cap - 3)
            self.assertIs(c.cache_controller.prefetch_buffer.get_nowait(), op)
        self.assertEqual(peers[1].cache_controller.mem_pool_host.freed, [256])

    def test_no_capacity_rolls_back_all_reservations(self):
        peers, ops = self.caches([7, 2, 5])
        for c, op in zip(peers, ops):
            self.reserve(c, op)
        self.assertEqual([self.complete(c) for c in peers], [0, 0, 0])
        self.assertEqual(
            [c.cache_controller.mem_pool_host.free_pages for c in peers], [7, 2, 5]
        )
        for c in peers:
            self.assertFalse(c.ongoing_prefetch)
            self.assertTrue(c.cache_controller.prefetch_buffer.empty())

    def test_local_abort_still_sends_ticket_and_zero_grant(self):
        peers, ops = self.caches([7, 7, 7])
        peers[1].ongoing_prefetch.clear()
        for c, op in zip(peers, ops):
            self.reserve(c, op)
        self.assertEqual([self.complete(c) for c in peers], [0, 0, 0])
        self.assertEqual(
            [c.cache_controller.mem_pool_host.free_pages for c in peers], [7, 7, 7]
        )

    def test_late_abort_keeps_io_ack_sequence(self):
        peers, ops = self.caches([7, 7, 7])
        for c, op in zip(peers, ops):
            self.reserve(c, op)
        peers[1].ongoing_prefetch.clear()
        ops[1].is_terminated = lambda: True
        ops[1].pool_transfers_done = True
        self.assertEqual([self.complete(c) for c in peers], [7, 7, 7])
        for c, op in zip(peers, ops):
            self.assertIs(c.cache_controller.prefetch_buffer.get_nowait(), op)
        self.assertTrue(ops[1].pool_transfers_done)

    def test_request_mismatch_fails_explicitly(self):
        peers, ops = self.caches([7, 7, 7])
        ops[1].request_id = "wrong"
        for c, op in zip(peers, ops):
            c.cache_controller.prefetch_alloc_consensus.submit(op, None)
        for c in peers:
            state = c.cache_controller.prefetch_alloc_consensus
            state.thread.join(6)
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                state.check()
            self.assertIn("sequence/request mismatch", str(state.error))

    def test_reset_cannot_drop_private_ownership_and_timeout_unchanged(self):
        peers, ops = self.caches([7, 7, 7])
        for c, op in zip(peers, ops):
            self.reserve(c, op)
        state = peers[0].cache_controller.prefetch_alloc_consensus
        with self.assertRaisesRegex(RuntimeError, "Cannot reset"):
            state.assert_idle()
        with patch.object(boundary.time, "monotonic", return_value=1e15):
            with self.assertRaisesRegex(RuntimeError, "120 seconds"):
                state.check()
        for c in peers:
            self.complete(c)


if __name__ == "__main__":
    unittest.main()


# Real Gloo verifies that all collective IO lives in the background worker.
def gloo_worker(rank, path, output):
    import torch.distributed as dist
    from datetime import timedelta

    dist.init_process_group(
        "gloo",
        init_method="file://" + path,
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=20),
    )
    reduce = lambda data: dist.all_reduce(data, op=dist.ReduceOp.MIN)
    valid = boundary.intersect_boundaries(
        operation(), ([3, 7], [3, 5], [3, 7])[rank], 7, reduce
    )
    stop = threading.Event()
    state = boundary.PrefetchAllocConsensus(reduce, stop, 64)
    try:
        for i in range(3):
            op = operation(str(i), valid=valid)
            # Last request misses on rank 2, every rank still sends a ticket.
            state.submit(op, None if i == 2 and rank == 2 else torch.arange(192))
        grants = []
        for i in range(3):
            op, pages = state.ready.get(timeout=15)
            state.take(op)
            grants.append(pages)
        state.assert_idle()
        output.put((rank, valid, grants))
    finally:
        stop.set()
        state.thread.join(5)
        assert not state.thread.is_alive()
        dist.destroy_process_group()


class BoundaryGlooTest(unittest.TestCase):
    def test_three_rank_background_grants(self):
        import multiprocessing, tempfile

        ctx = multiprocessing.get_context("spawn")
        output = ctx.Queue()
        with tempfile.TemporaryDirectory() as directory:
            workers = [
                ctx.Process(target=gloo_worker, args=(r, directory + "/init", output))
                for r in range(3)
            ]
            for p in workers:
                p.start()
            try:
                rows = [output.get(timeout=35) for p in workers]
                for p in workers:
                    p.join(5)
                    self.assertEqual(p.exitcode, 0)
                self.assertEqual(sorted(rows), [(r, [3], [3, 3, 0]) for r in range(3)])
            finally:
                for p in workers:
                    if p.is_alive():
                        p.terminate()
                        p.join()


class BoundaryProofTest(unittest.TestCase):
    def test_proof_requires_root_mamba_get_and_is_bounded(self):
        import gzip, json, os, tempfile

        with (
            tempfile.TemporaryDirectory() as path,
            patch.dict(os.environ, {"Q35_HIGHX_LOG_DIR": path}),
        ):
            cc = types.SimpleNamespace(pp_rank=1, tp_rank=0)
            cache = types.SimpleNamespace(cache_controller=cc)
            op = operation()
            op.storage_start = 0
            op.completed_tokens = 192
            op.diagnostic_input_ids = list(range(192))
            op.pool_transfers = [types.SimpleNamespace(name="mamba")]
            from pathlib import Path

            boundary.record_boundary_proof(cache, op)
            boundary.record_boundary_proof(cache, op)
            files = list(Path(path).glob("*.json.gz"))
            self.assertEqual(len(files), 1)
            data = json.loads(gzip.decompress(files[0].read_bytes()))
            self.assertEqual(data["prefix_tokens"], list(range(192)))
            self.assertEqual(data["verified_by"], "successful_root_hybrid_get")
            op.storage_start = 64
            op.diagnostic_input_ids = list(range(1, 193))
            boundary.record_boundary_proof(cache, op)
            self.assertEqual(len(list(Path(path).glob("*.json.gz"))), 1)
            op.storage_start = 0
            op.pool_transfers = []
            boundary.record_boundary_proof(cache, op)
            self.assertEqual(len(list(Path(path).glob("*.json.gz"))), 1)
