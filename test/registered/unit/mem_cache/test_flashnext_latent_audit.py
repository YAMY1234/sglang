"""#905: the opt-in shared-arena latent audit (CPU).

The stamp must leave the 1,972 decoded wire bytes untouched; each check must name the mechanism it was built for
(unflushed forward mapping, stale GPU page map at store, row provenance at load).
"""
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache import flashnext_latent_audit as audit
from sglang.srt.mem_cache.flashnext_scheme_c import _pack_gap8_torch


def _pool(pages=4):
    shared = {p: list(range(10 * p, 10 * p + 9)) for p in range(1, pages)}
    table = torch.zeros((pages + 1, 9), dtype=torch.int32)
    for p, units in shared.items():
        table[p] = torch.tensor(units, dtype=torch.int32)
    arena = SimpleNamespace(shared=shared, pending_shared={}, pending_deep={})
    return SimpleNamespace(page_size=64, physical_page_map=table, arena=arena, tp_rank=0)


class LatentAuditTest(unittest.TestCase):
    def setUp(self):
        audit.counts.clear()

    def test_stamp_leaves_wire_bytes(self):
        payload = torch.randint(0, 256, (3, 2048), dtype=torch.uint8)
        payload[:, 1972:] = 0
        before = payload.clone()
        audit.stamp(payload, torch.tensor([64, 65, 130]), torch.tensor([0, 1, 7]), 5)
        self.assertTrue(torch.equal(payload[:, :1972], before[:, :1972]))
        self.assertTrue(torch.equal(payload[:, 1996:], before[:, 1996:]))
        self.assertEqual(audit.read_stamp(payload)[:, [0, 1, 2, 4]].tolist(),
                         [[audit.MAGIC, 0, 64, 5], [audit.MAGIC, 1, 65, 5], [audit.MAGIC, 7, 130, 5]])

    def test_classify(self):
        payload = torch.zeros((4, 2048), dtype=torch.uint8)
        audit.stamp(payload[0:1], torch.tensor([64]), torch.tensor([3]), 1)       # ok
        audit.stamp(payload[2:3], torch.tensor([99]), torch.tensor([5]), 1)       # another token's latent
        payload[3, 1976:1996] = 7                                                 # overwritten by a non-latent write
        cls = audit.classify(audit.read_stamp(payload), torch.tensor([64, 65, 66, 67]), torch.tensor([3, 4, 5, 6]))
        self.assertEqual(cls.tolist(), [0, 1, 2, 3])

    def test_store_map_mismatch_and_unflushed_forward(self):
        pool = _pool()
        with self.assertNoLogs(audit.logger, level="ERROR"):
            audit.check_store_map(pool, torch.tensor([64, 65, 128]))
        pool.physical_page_map[2] = 0          # page 2 reserved on CPU, never published to the GPU map
        pool.arena.pending_shared[2] = pool.arena.shared[2]
        with self.assertLogs(audit.logger, level="ERROR") as log:
            audit.check_store_map(pool, torch.tensor([64, 128]))
            audit.forward_unflushed(pool, None)
        self.assertIn('"kind": "store_map_mismatch"', log.output[0])
        self.assertIn('"pending": true', log.output[0])
        self.assertIn('"kind": "forward_unflushed"', log.output[1])

    def test_load_reports_zero_row(self):
        pool = _pool()
        idx = torch.stack([torch.randperm(10240)[:512] for _ in range(2)])
        stream, lengths, _ = _pack_gap8_torch(idx, width=10240)
        stream = torch.cat([stream, torch.zeros_like(stream[:1])])          # third row never written
        lengths = torch.cat([lengths, torch.zeros_like(lengths[:1])])
        payload = torch.zeros((3, 2048), dtype=torch.uint8)
        locs, pos = torch.tensor([64, 65, 66]), torch.tensor([0, 1, 2])
        audit.stamp(payload[:2], locs[:2], pos[:2], 1)
        with self.assertLogs(audit.logger, level="ERROR") as log:
            audit.check_load(pool, locs, pos, payload, stream, lengths, torch.tensor([[11], [12], [0]]))
        self.assertIn('"never_written": 1', log.output[0])
        self.assertIn('"invalid_gap8": 1', log.output[0])

    def test_forward_write_ownership(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        pool = _pool()
        deep_owned = {1: [100, 101, 102, 103, 104]}
        pool.arena.deep = deep_owned
        pool.deep = SimpleNamespace(physical_page_map=torch.zeros((4, 5), dtype=torch.int32))
        pool.deep.physical_page_map[1] = torch.tensor(deep_owned[1], dtype=torch.int32)
        pool.deep_req_to_token = torch.zeros((4, 256), dtype=torch.int64)
        pool.deep_req_to_token[2, :64] = torch.arange(64, 128)          # slot 2: deep page 1
        fb = SimpleNamespace(forward_mode=ForwardMode.DECODE, out_cache_loc=torch.tensor([130]),
                             req_pool_indices=torch.tensor([2]), req_pool_indices_cpu=torch.tensor([2]),
                             seq_lens=torch.tensor([10]), seq_lens_cpu=torch.tensor([10]), batch_size=1)
        with self.assertNoLogs(audit.logger, level="ERROR"):
            audit.check_forward_writes(pool, fb)
        del pool.arena.shared[2]                                         # shallow page 2 released, GPU row stale
        pool.deep_req_to_token[2, 9] = 200                               # a deep location on unowned page 3
        with self.assertLogs(audit.logger, level="ERROR") as log:
            audit.check_forward_writes(pool, fb)
        self.assertIn('"kind": "forward_write_violation"', log.output[0])
        self.assertIn('"page": 2, "owned": false', log.output[0])
        self.assertIn('"page": 3, "owned": false', log.output[0])


if __name__ == "__main__":
    unittest.main()
