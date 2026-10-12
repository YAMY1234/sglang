# ruff: noqa: I001
# Bootstrap lightweight serving modules before importing the GLM adapter.
"""CPU routing, batching, TP probe and warm-storage tests for GLM W32 cuts."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_kimi_duet_fastpath import load_shared_mgs
from test_glm_duet import _Pool, glm_state_policy
from glm_duet import _fast_state, _state_factor


class GlmFastStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        load_shared_mgs()

    def policy(self):
        gen = torch.Generator().manual_seed(194)
        pool = _Pool(torch.randn(3, 5, 2, 48, 48, generator=gen), torch.arange(5))
        return glm_state_policy.GlmStatePolicy(
            pool,
            torch.randn(3, 2, 48, generator=gen),
            {"state_rank": 32, "state_every": 32, "state_sink": "explicit"},
            [0, 1, 2],
            omega_heads=(2, 8),
        )

    def test_prompt_unchanged_periodic_one_dispatch_and_probe_cache(self):
        p = self.policy()
        before = p.states.clone()
        with patch.object(
            _fast_state.fast_state, "project", wraps=_fast_state.fast_state.project
        ) as project:
            p._cut([1, 3], fresh=True)
            self.assertEqual(project.call_count, 0)
            p._cut([1, 3], fresh=False)
            self.assertEqual(project.call_count, 1)
            self.assertEqual(project.call_args.args[0].shape, (6, 2, 48, 48))
            omega = p._fast_omega
            p._cut([1], fresh=False)
            self.assertIs(p._fast_omega, omega)
        self.assertTrue(torch.equal(p.states[:, [0, 2, 4]], before[:, [0, 2, 4]]))
        self.assertEqual(set(p.warm), {(l, s) for l in range(3) for s in [1, 3]})

    def test_mixed_warm_rows_chunking_and_owned_storage(self):
        p = self.policy()
        p._cut([1], fresh=True)
        with (
            patch.object(_state_factor, "ROW_CHUNK", 2),
            patch.object(
                _fast_state.fast_state, "project", wraps=_fast_state.fast_state.project
            ) as project,
        ):
            p._cut([1, 3], fresh=False)
            self.assertEqual(project.call_count, 3)
            masks = [call.args[4].tolist() for call in project.call_args_list]
            self.assertEqual(masks, [[True, False]] * 3)
        values = list(p.warm.values())
        self.assertEqual(len({v.untyped_storage().data_ptr() for v in values}), 6)
        for value in values:
            self.assertEqual(
                value.untyped_storage().nbytes(), value.numel() * value.element_size()
            )

    def test_full_model_probe_draw_then_head_slice(self):
        state = torch.zeros(3, 8, 48, 48)
        full = _fast_state.probes(state, 32, None)
        part = _fast_state.probes(state[:, 2:4], 32, (2, 8))
        for got, want in zip(part, full):
            self.assertTrue(torch.equal(got, want[:, 2:4]))
        gen = torch.Generator().manual_seed(194)
        state.normal_(generator=gen)
        direction = torch.randn(3, 8, 48, generator=gen)
        whole, _bases = _fast_state.project_state_batched(
            state, direction, 32, [None] * 3, full
        )
        sliced, _ = _fast_state.project_state_batched(
            state[:, 2:4], direction[:, 2:4], 32, [None] * 3, part
        )
        torch.testing.assert_close(sliced, whole[:, 2:4], rtol=1e-6, atol=1e-6)

    def test_reset_copy_and_restore_warm_lifecycle(self):
        p = self.policy()
        p._cut([1], fresh=False)
        p.h_count[1] = 17
        saved = {key: value.clone() for key, value in p.warm.items()}
        p.copy_slots(torch.tensor([1]), torch.tensor([3]))
        self.assertEqual(p.h_count[3], 17)
        for layer in range(3):
            self.assertTrue(torch.equal(p.warm[layer, 3], saved[layer, 1]))
            self.assertNotEqual(
                p.warm[layer, 3].data_ptr(), p.warm[layer, 1].data_ptr()
            )
        exported = p.get_cpu_slots(torch.tensor([3]))
        p.reset_slots(torch.tensor([1, 3]))
        self.assertFalse(p.warm)
        p.load_cpu_slots(exported, torch.tensor([4]))
        for layer in range(3):
            self.assertTrue(torch.equal(p.warm[layer, 4], saved[layer, 1]))

    def test_host_schedule_projects_after_exactly_32_steps(self):
        p = self.policy()
        p.mark_prompt_end([1, 3], [1, 3])
        indices = torch.tensor([1, 3])
        batch = SimpleNamespace(
            req_pool_indices=indices, req_pool_indices_cpu=indices, batch_size=2
        )
        p.before_graph(batch)
        with patch.object(
            _fast_state.fast_state, "project", wraps=_fast_state.fast_state.project
        ) as project:
            for step in range(1, 65):
                p.after_graph(batch)
                self.assertEqual(project.call_count, step // 32)
                self.assertEqual(p.h_count[1], step % 32)
                self.assertEqual(p.steps[3], step)
        self.assertEqual(p.prefix_cuts, 6)
        self.assertEqual(p.decode_cuts, 12)


if __name__ == "__main__":
    unittest.main()
