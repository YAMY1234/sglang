"""CPU tests of real factor payloads and local P/D lifecycle (no CUDA imports)."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


factors = load("_pd_test_factor_pool", "mem_cache/gdn_factored_pool.py")
handoff = load("_pd_test_handoff", "disaggregation/state_handoff.py")


class TestFactoredPDHandoff(unittest.TestCase):
    def setUp(self):
        self.pool = factors.FactoredGDNPool(
            size=8, cache_params=SimpleNamespace(shape=SimpleNamespace(temporal=(2, 4, 4))),
            mamba_layer_ids=[0, 2], device="cpu",
            cfg=factors.FactoredGDNConfig(r=2, m=2, ring=3),
        )
        self.req = SimpleNamespace(kv=SimpleNamespace(
            mamba_pool_idx=torch.tensor(2),
            mamba_ping_pong_track_buffer=torch.tensor([3, 4]),
            mamba_last_track_idx=1, mamba_last_track_seqlen=128,
            mamba_cow_src_index=torch.tensor([7]), mamba_needs_clear=True,
        ))
        self.handler = handoff.FactorStateHandoff(self.pool)

    def payload(self):
        return [x.clone() for x in (self.pool.a, self.pool.U, self.pool.W, self.pool.count)]

    def test_receive_overwrites_authority_without_touching_payload(self):
        self.pool.a.normal_(); self.pool.U.normal_(); self.pool.W.normal_()
        self.pool.ring_owner[:] = [2, 3, 7]
        self.pool.stale.zero_(); self.pool.dense_of.fill_(1)
        before = self.payload()
        self.handler.prepare_receive(self.req)
        # Emulate RDMA into destination, then a successful metadata gate.
        self.pool.a[:, 2].add_(10)
        expected = self.payload()
        self.handler.commit_receive(self.req)
        self.handler.commit_receive(self.req)  # duplicate success/retry
        for actual, want in zip(self.payload(), expected):
            self.assertTrue(torch.equal(actual, want))
        self.assertFalse(torch.equal(before[0], expected[0]))
        self.assertEqual(self.pool.ring_owner, [-1, -1, 7])
        self.assertEqual(self.pool.stale[2:5].tolist(), [1, 1, 1])
        self.assertEqual(self.pool.dense_of[2:5].tolist(), [-1, -1, -1])
        self.assertEqual(self.pool.stale[7].item(), 0)
        self.assertIsNone(self.req.kv.mamba_cow_src_index)
        self.assertFalse(self.req.kv.mamba_needs_clear)

    def test_cancel_then_reuse_does_not_reuse_dense_ring(self):
        self.handler.prepare_receive(self.req)
        self.pool.a[:, 2].fill_(123)  # cancelled transfer payload, never committed
        self.pool.ring_owner[0] = 2
        self.pool.dense_of[2] = 0
        self.pool.stale[2] = 0
        self.handler.prepare_receive(self.req)  # only after old writer drained
        self.pool.a[:, 2].fill_(456)
        self.handler.commit_receive(self.req)
        self.assertEqual(self.pool.ring_owner[0], -1)
        self.assertTrue(bool((self.pool.a[:, 2] == 456).all()))
        self.assertEqual(self.pool.stale[2].item(), 1)

    def test_send_requires_final_truncation(self):
        self.handler.before_send(self.req)
        self.pool.count[1, 2, 0] += 1
        with self.assertRaisesRegex(RuntimeError, "final prefill"):
            self.handler.before_send(self.req)

    def test_wire_entries_exclude_local_metadata(self):
        entries = list(self.pool.iter_transfer_state_entries())
        self.assertEqual(len(entries), 8)
        self.assertEqual({x[0] for x in entries}, {
            "gdn_factored_a", "gdn_factored_u", "gdn_factored_w", "gdn_factored_count"})
        for _, tensor, axis, lid in entries:
            self.assertEqual(tensor.shape[0], 9)
            self.assertEqual(axis, 0)
            self.assertIn(lid, [0, 2])

    def test_flag_off_and_explicit_extension_registration(self):
        pool = SimpleNamespace()
        before = vars(self.req.kv).copy()
        handoff.dispatch_handoff(pool, "commit_receive", self.req)
        self.assertEqual(vars(self.req.kv), before)
        handoff.register_handoff(pool, handoff.HandoffKind.STATE_FACTOR, self.handler)
        with self.assertRaises(ValueError):
            handoff.register_handoff(pool, handoff.HandoffKind.STATE_FACTOR, self.handler)
        with self.assertRaises(TypeError):
            handoff.register_handoff(pool, handoff.HandoffKind.LATENT, object())
        with self.assertRaises(ValueError):
            handoff.dispatch_handoff(pool, "invented", self.req)

    def test_invalid_slot_cannot_mutate_pool(self):
        before = self.payload()
        for indices in (torch.tensor([0]), torch.tensor([9]), torch.tensor([-1])):
            with self.assertRaises(ValueError):
                self.pool.mark_transferred_slots(indices)
        for actual, want in zip(self.payload(), before):
            self.assertTrue(torch.equal(actual, want))


if __name__ == "__main__":
    unittest.main()
