"""Bounded payload pipeline: production window, wire codec, and Gloo FIFO."""

import hashlib
import multiprocessing
import tempfile
import threading
import time
import unittest
from collections import defaultdict, deque
from datetime import timedelta
from unittest.mock import patch

import torch.distributed as dist
from test_pp_commit_integration import Cluster, core, proposals_module, transport


def proposal(rank, serial):
    return {
        "epoch": 0,
        "origin": rank,
        "serial": serial,
        "action": "delete",
        "pool": "kv",
        "origin_site": 4,
        "hashes": [
            hashlib.sha256(f"{rank}-{serial}-{i}".encode()).hexdigest()
            for i in range(8)
        ],
    }


def payload_worker(rank, path, output):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=30),
    )
    control = dist.new_group([0, 1, 2], backend="gloo")
    box = transport.PreviousRoundReports(control)
    try:
        deadline = time.monotonic() + 25
        seen = {1: 0, 2: 0}
        if rank == 0:
            # Force many metadata overwrites and receiver FIFO pressure.
            time.sleep(0.5)
            while min(seen.values()) < 2400:
                assert time.monotonic() < deadline
                for peer, report in box.poll().items():
                    items = report["belief_proposals"]
                    for item in items:
                        assert item == proposal(peer, seen[peer] + 1)
                        seen[peer] += 1
                    box.acknowledge(peer, 0, seen[peer])
                time.sleep(0.001)
        else:
            serial = 1
            while serial <= 2400:
                assert time.monotonic() < deadline
                items = [
                    proposal(rank, i) for i in range(serial, min(serial + 64, 2401))
                ]
                report = {
                    "epoch": 0,
                    "round": serial,
                    "confirmed": 0,
                    "belief_proposals": items,
                }
                if box.publish(report):
                    serial += len(items)
                box.publish({"epoch": 0, "round": serial, "confirmed": 0})
                time.sleep(0.001)
        output.put((rank, seen, box.stats))
        box.close(1)
        dist.barrier()
        assert box.close(2)
    finally:
        dist.destroy_process_group()


class PipelineTest(unittest.TestCase):
    def test_window_sends_new_prefix_without_waiting_for_old_commit(self):
        c = Cluster().caches[1]
        p = c._pp_commit.belief_proposals
        c.storage_existence_cache.add("kv", [f"{i:064x}" for i in range(8 * 1600)])
        for offset in range(0, 1536, 64):
            items = p.batch()
            self.assertEqual(
                [x["serial"] for x in items], list(range(offset + 1, offset + 65))
            )
            self.assertEqual(p.batch(), items)  # rejected transport can retry intact
            p.mark_sent(items)
        self.assertEqual(p.batch(), [])
        self.assertEqual(len(p.pending), 1600)
        first = [p.pending[i][0] for i in range(1, 65)]
        for item in first:
            p.mark_assigned(item)
            p.complete(item)
        self.assertEqual([x["serial"] for x in p.batch()], list(range(1537, 1601)))
        self.assertEqual(p.snapshot()["window_limit"], 1536)

    def test_pending_bound_and_120_second_guard_unchanged(self):
        c = Cluster().caches[1]
        p = c._pp_commit.belief_proposals
        with self.assertRaisesRegex(RuntimeError, "proposal bound"):
            c.storage_existence_cache.add("kv", [f"{i:064x}" for i in range(8 * 8193)])
        self.assertEqual(len(p.pending), 8192)
        self.assertEqual(p.timeout, 120)

    def test_maximum_wire_and_manifest_remain_in_existing_byte_caps(self):
        cls = transport.PreviousRoundReports
        report = {
            "wire_seq": 99999999,
            "report": {
                "epoch": 0,
                "round": 99999999,
                "confirmed": 99999999,
                "digest": "f" * 64,
                "applied": 99999999,
                "applied_digest": "f" * 64,
                "belief_proposals": [proposal(1, i + 99999999) for i in range(64)],
            },
        }
        encoded = cls.encode(report)
        self.assertEqual(cls.decode(encoded), report)
        self.assertEqual(len(encoded), 32768)
        cluster = Cluster()
        leader = cluster.caches[0]._pp_commit
        leader.state.size = 4
        cluster.caches[0].storage_existence_cache.add(
            "kv", [f"{i:064x}" for i in range(512)]
        )
        leader._assign_belief_proposals(
            {
                rank: {"belief_proposals": [proposal(rank, i) for i in range(1, 65)]}
                for rank in range(1, 4)
            }
        )
        frame = leader.state.leader_frame({}, limit=288)
        frame["belief_entries"] = [
            [wire, leader.belief_effects[core.OperationId(*wire)]]
            for _, wire, _ in frame["entries"]
        ]
        self.assertEqual(len(frame["entries"]), 256)
        # Physical entries have larger content keys than belief digests. Reserve
        # realistic full manifests, not just a synthetic 64-character identity.
        for i in range(32):
            frame["entries"].append([257 + i, [0, "backup", "a" * 256, i], "b" * 64])
        frame.update(proposal_admitted={str(i): 99999999 for i in range(4)}, admit=True)
        self.assertLessEqual(len(core.canonical(frame)) + 4096, 262144)

    def test_production_window_sustained_peak_for_one_hour_and_double(self):
        # Real BeliefProposals code, bounded credit and common-completion FIFO.
        # Three mock stages confirm at 10/3,20/3,10s; only all-stage ACK commits.
        for scale, limit in ((1, 60), (2, 120)):
            with self.subTest(scale=scale):
                now = [0.0]
                p = proposals_module.BeliefProposals(1)
                belief = type("Belief", (), {"peek_present": lambda *a: True})()
                wire = deque()
                serial = 0
                with patch.object(
                    proposals_module.time, "monotonic", new=lambda now=now: now[0]
                ):
                    for step in range(7500):
                        now[0] = step * 0.5
                        # Repeat measured1793/30s in the strongest paired-edge
                        # placement: two bins arrive together every60s.
                        if step < 7200 and step % 120 == 60:
                            n = 1793 * scale * 2
                            for _ in range(n):
                                serial += 1
                                p.propose(
                                    "delete",
                                    "kv",
                                    [f"{serial:064x}"],
                                    belief,
                                    "_invalidate_absent_from_hit_query",
                                )
                        while wire and wire[0][0] <= now[0]:
                            _, item = wire.popleft()
                            p.mark_assigned(item)
                            p.complete(item)
                        items = p.batch()
                        p.mark_sent(items)
                        wire.extend((now[0] + 10.5, item) for item in items)
                        p.check_age()
                        self.assertLessEqual(p.sent - p.last_committed, 1536)
                        if step >= 7200 and not p.pending:
                            break
                self.assertFalse(p.pending)
                self.assertEqual(p.completed, 1793 * scale * 120)
                self.assertLess(p.age_peak, limit)
                self.assertLessEqual(p.peak, 8192)

    def test_close_receives_pending_metadata_then_peer_closed_frame(self):
        cls = transport.PreviousRoundReports
        box = cls.__new__(cls)
        box.group = None
        box._condition = threading.Condition()
        box._incoming = defaultdict(deque)
        box._latest = {}
        box._closing = True
        box.stats = {"received": 0}
        frames = [
            cls.encode({"wire_seq": 1, "report": {"round": 5}}),
            cls.encode({"wire_seq": 2, "closed": True}),
        ]

        def receive(tensor, **kwargs):
            tensor.copy_(frames.pop(0))

        with patch.object(transport.dist, "recv", new=receive):
            box._receiver(1)
        self.assertFalse(frames)
        self.assertEqual(box._latest[1]["round"], 5)

    def test_real_three_rank_payload_fifo_survives_ready_coalescing(self):
        context = multiprocessing.get_context("spawn")
        output = context.Queue()
        with tempfile.TemporaryDirectory() as directory:
            workers = [
                context.Process(
                    target=payload_worker, args=(r, directory + "/init", output)
                )
                for r in range(3)
            ]
            for worker in workers:
                worker.start()
            try:
                rows = [output.get(timeout=40) for _ in workers]
                for worker in workers:
                    worker.join(10)
                    self.assertEqual(worker.exitcode, 0)
                self.assertEqual(
                    next(row[1] for row in rows if row[0] == 0), {1: 2400, 2: 2400}
                )
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
