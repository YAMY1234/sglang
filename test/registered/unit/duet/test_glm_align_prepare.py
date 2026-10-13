"""GLM-private P1 N/S CPU fixtures; no CUDA initialization or shared solver change."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_glm_duet import _Pool, glm_state_policy, state_factor


def make_policy(enabled=True, slots=6):
    gen = torch.Generator().manual_seed(617)
    state = torch.randn(3, slots, 2, 16, 16, generator=gen)
    directions = torch.randn(3, 2, 16, generator=gen)
    mapping = torch.tensor([0, 1, 2, 3, 4, 5, -1, slots, -1, -1])
    pool = _Pool(state, mapping)
    policy = glm_state_policy.GlmStatePolicy(
        pool, directions, {"state_rank": 4, "state_every": 3}, [0, 1, 2],
        omega_heads=(2, 8), align_prepare=enabled,
    )
    return policy, pool


def batch(requests, cpu=True):
    rpi = torch.tensor(requests, dtype=torch.long)
    return SimpleNamespace(req_pool_indices=rpi, req_pool_indices_cpu=rpi if cpu else None,
                           batch_size=len(requests))


class ProjectionIdentity(unittest.TestCase):
    def test_glm64_probe_tp_slices_cold_warm_and_cache(self):
        gen = torch.Generator().manual_seed(18)
        states = torch.randn(2, 64, 128, 128, generator=gen)
        direction = torch.randn(2, 64, 128, generator=gen)
        for tp in (1, 2, 4, 8):
            for offset in range(0, 64, 64 // tp):
                with self.subTest(tp=tp, offset=offset):
                    heads = slice(offset, offset + 64 // tp)
                    sf, d = states[:, heads].contiguous(), direction[:, heads].contiguous()
                    ws = state_factor.ProjectionWorkspace()
                    prev = [None, None]
                    for _ in range(3):
                        old, old_basis = state_factor.project_state_batched(sf, d, 32, prev, omega_heads=(offset, 64))
                        new, new_basis = state_factor.project_state_batched(sf, d, 32, prev, omega_heads=(offset, 64), workspace=ws)
                        self.assertTrue(torch.equal(old, new))
                        for a, b in zip(old_basis, new_basis):
                            self.assertTrue(torch.equal(a, b))
                        prev = old_basis
                    self.assertEqual(len(ws.probes), 2)
                    for (_, total, n, columns, seed), value in ws.probes.items():
                        reference = torch.randn(1, total, n, columns, generator=torch.Generator().manual_seed(seed))
                        self.assertTrue(torch.equal(value, reference))

    def test_mixed_rows_d1_multiple_d_and_nonfull_chunk(self):
        gen = torch.Generator().manual_seed(77)
        for rows in (1, 3, 259):
            with self.subTest(rows=rows):
                sf = torch.randn(rows, 2, 16, 16, generator=gen)
                direction = torch.randn(rows, 2, 16, generator=gen)
                _, warm = state_factor.project_state_batched(sf, direction, 4, [None] * rows)
                for prev in ([None] * rows, warm, [v if i % 2 else None for i, v in enumerate(warm)]):
                    old, ob = state_factor.project_state_batched(sf, direction, 4, prev, omega_heads=(2, 8))
                    new, nb = state_factor.project_state_batched(sf, direction, 4, prev, omega_heads=(2, 8), workspace=state_factor.ProjectionWorkspace())
                    self.assertTrue(torch.equal(old, new))
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(ob, nb)))

    def test_zero_rank_deficient_flat_and_near_boundary_spectra(self):
        gen = torch.Generator().manual_seed(32)
        q = torch.linalg.qr(torch.randn(16, 16, generator=gen)).Q
        d = torch.randn(2, 16, generator=gen)
        for spectrum in (torch.zeros(16), torch.tensor([1.] * 3 + [0.] * 13),
                         torch.ones(16), torch.tensor([1.] * 4 + [1. - 1e-5] * 12)):
            sf = (q @ torch.diag(spectrum) @ q.T).expand(2, 2, -1, -1).contiguous()
            old, ob = state_factor.project_state_batched(sf, d, 4, [None] * 2)
            new, nb = state_factor.project_state_batched(sf, d, 4, [None] * 2, workspace=state_factor.ProjectionWorkspace())
            self.assertTrue(torch.equal(old, new))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(ob, nb)))


class Lifecycle(unittest.TestCase):
    def test_default_off_and_independent_switch(self):
        with patch.dict(os.environ, {}, clear=True):
            p, _ = make_policy(None)
            self.assertFalse(p.align_prepare)
        with patch.dict(os.environ, {"SGLANG_GLM_ALIGN_PREPARE": "1", "SGLANG_GLM_SAFE_PACKED_DECODE": "0"}):
            p, _ = make_policy(None)
            self.assertTrue(p.align_prepare)

    def test_reset_copy_spill_restore_mixed_phases(self):
        p, pool = make_policy()
        p.mark_prompt_end([0, 1, 2], [0, 1, 2])
        p.before_graph(batch([0, 1]))
        p.after_graph(batch([0]))
        p.after_graph(batch([0]))
        self.assertEqual(p.h_count[0], 2)
        self.assertIn(2, p.h_pending)
        state = p.states[:, [0, 1, 2]].clone()
        saved = p.get_cpu_slots([0, 1, 2])
        warm = p.warm_slots[0].clone()
        p.copy_slots([0, 2], [1, 3])
        self.assertNotIn(1, p.h_pending)
        self.assertIn(3, p.h_pending)
        self.assertEqual(p.h_count[1], 2)
        self.assertNotIn(1, p.slot_of)  # forked checkpoint has no copied request owner
        self.assertNotEqual(p.warm_slots[0].data_ptr(), p.warm_slots[1].data_ptr())
        p.reset_slots([0, 1, 2, 3])
        p.states.zero_()
        p.states[:, [3, 4, 5]] = state
        p.load_cpu_slots(saved, [3, 4, 5])
        pool.slot_of_request[:3] = torch.tensor([3, 4, 5])
        self.assertEqual(p.slot_of, {0: 3, 1: 4, 2: 5})
        self.assertEqual(p.h_count[3], 2)
        self.assertIn(5, p.h_pending)
        self.assertTrue(torch.equal(p.warm_slots[3], warm))
        self.assertTrue(p.warm_slots[3].is_contiguous())
        self.assertEqual(p.warm_slots[3].untyped_storage().nbytes(), p.warm_slots[3].numel() * 4)
        old_generation = p.generation[3]
        p.after_graph(batch([0]))  # W-1 -> cut -> 0
        self.assertEqual(p.h_count[3], 0)
        self.assertEqual(p.generation[3], old_generation)
        p.after_graph(batch([0]))
        self.assertEqual(p.h_count[3], 1)
        p.before_graph(batch([2]))
        self.assertNotIn(5, p.h_pending)
        p.after_graph(batch([2]))
        self.assertEqual(p.h_count[5], 1)  # prompt final -> first decode counts once

    def test_invalid_padding_duplicates_and_missing_metadata(self):
        p, _ = make_policy()
        p.mark_prompt_end([0, 6, 7], [0, -1, 6])
        p.before_graph(batch([0, 0, 6, 7]))
        p.after_graph(batch([0, 0, 6, 7]))
        self.assertEqual(p.h_count, {0: 1})
        p.after_graph(batch([0], cpu=False))
        self.assertEqual(p.h_count[0], 2)
        p.slot_of[0] = 1
        p.req_generation[0] = -1  # stale mapping resolves current pool slot 0
        self.assertEqual(p._batch_slots(batch([0])), [0])
        last = p.states[:, -1].clone()
        p.reset_slots([-1, 6, 6])
        self.assertTrue(torch.equal(last, p.states[:, -1]))
        for slots in ([-1], [6], [1, 1]):
            with self.assertRaises(ValueError):
                p.get_cpu_slots(slots)
        with self.assertRaises(ValueError):
            p.copy_slots([0, 1], [2, 2])

    def test_legacy_checkpoint_missing_host_metadata(self):
        old, _ = make_policy(False)
        old.count[:, 0] = 2
        old.pending_prefix[:, 1] = True
        saved = old.get_cpu_slots([0, 1])
        new, _ = make_policy()
        new.mark_prompt_end([2, 3], [2, 3])
        new.load_cpu_slots(saved, [2, 3])
        self.assertEqual(new.h_count[2], 2)
        self.assertIn(3, new.h_pending)
        self.assertEqual(new.slot_of, {})

    def test_stale_spill_cannot_steal_readmitted_request(self):
        p, _ = make_policy()
        p.mark_prompt_end([0], [0])
        saved = p.get_cpu_slots([0])
        p.reset_slots([0])
        p.mark_prompt_end([0], [1])
        p.load_cpu_slots(saved, [2])
        self.assertEqual(p.slot_of[0], 1)
        # Even if a later reset removes that current mapping, the saved request
        # epoch cannot be rebound by an old restore.
        p.reset_slots([1])
        p.load_cpu_slots(saved, [3])
        self.assertNotIn(0, p.slot_of)

    def test_copy_snapshot_swap_and_self_copy(self):
        p, _ = make_policy()
        p.h_count.update({0: 1, 1: 2})
        p.h_pending.add(1)
        p.copy_slots([0, 1], [1, 0])
        self.assertEqual(p.h_count, {0: 2, 1: 1})
        self.assertEqual(p.h_pending, {0})
        p.mark_prompt_end([0], [0])
        p.copy_slots([0], [0])
        self.assertEqual(p.slot_of[0], 0)

    def test_invalid_restore_and_copy_leave_transaction_unchanged(self):
        p, _ = make_policy()
        p.mark_prompt_end([0], [0])
        p.before_graph(batch([0]))
        saved = p.get_cpu_slots([0])
        p.mark_prompt_end([1], [1])
        generation = dict(p.generation)
        owners = dict(p.slot_of)
        malformed = {**saved, "ordinary": (saved["ordinary"][0][:, :0], saved["ordinary"][1])}
        with self.assertRaises(ValueError):
            p.load_cpu_slots(malformed, [1])
        malformed = {**saved, "slots": [{**saved["slots"][0], "warm": torch.zeros(1)}]}
        with self.assertRaises(ValueError):
            p.load_cpu_slots(malformed, [1])
        with self.assertRaises(ValueError):
            p.copy_slots([0, 1], [2])
        self.assertEqual(p.generation, generation)
        self.assertEqual(p.slot_of, owners)
        self.assertIn(1, p.h_pending)

    def test_changing_input_replay_hooks_match_legacy(self):
        old, _ = make_policy(False)
        new, _ = make_policy(True)
        gen = torch.Generator().manual_seed(86)
        for p in (old, new):
            p.mark_prompt_end([0, 1], [0, 1])
        # Fixed tensor addresses with changing contents emulate the external
        # graph hooks; actual CUDA capture/replay is a separate GPU gate.
        b = batch([0, 1])
        ptr = b.req_pool_indices.data_ptr()
        for tick in range(12):
            b.req_pool_indices.copy_(torch.tensor([1, 0] if tick % 2 else [0, 1]))
            write = torch.randn(3, 2, 2, 16, 16, generator=gen)
            for p in (old, new):
                p.before_graph(b)
                p.states[:, :2] += write
                p.after_graph(b)
            self.assertEqual(b.req_pool_indices.data_ptr(), ptr)
            self.assertTrue(torch.equal(old.states, new.states))
            self.assertEqual(old.h_count, new.h_count)
            for s in (0, 1):
                for l in range(3):
                    self.assertTrue(torch.equal(old.warm[(l, s)], new.warm_slots[s][l:l + 1]))


if __name__ == "__main__":
    unittest.main()
