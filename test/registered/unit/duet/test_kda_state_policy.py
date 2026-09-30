"""CPU lifecycle checks for the shared KDA policy and legacy sink owner."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[4]
path = ROOT / "python/sglang/srt/layers/attention/linear/kda_state_prune.py"
spec = importlib.util.spec_from_file_location("kda_policy_under_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Pool:
    def __init__(self):
        self.registered = []
        self.mamba_pool = SimpleNamespace(
            mamba_cache=SimpleNamespace(temporal=torch.zeros(1, 4, 2, 8, 8)),
            register_slot_state=self.registered.append,
        )

    def mamba2_layer_index(self, layer):
        assert layer == 3
        return 0


class StatePolicyTests(unittest.TestCase):
    def test_common_policy_allocates_no_auxiliary_sink_or_solver(self):
        pool = Pool()
        policy = module.KDAStatePolicy(pool, {3: torch.ones(2, 8)}, 2, every=5)
        self.assertEqual(pool.registered, [policy])
        self.assertFalse(hasattr(policy, "sinks"))
        self.assertFalse(hasattr(policy, "solver"))
        self.assertEqual(policy.count.shape, (1, 4))
        self.assertEqual(policy.vbar.shape, (1, 2, 8))

    def test_count_and_prefix_survive_copy_offload_restore_then_reset(self):
        policy = module.KDAStatePolicy(Pool(), {3: torch.ones(2, 8)}, 2)
        policy.count[0, 1], policy.pending_prefix[0, 1] = 5, True
        policy.copy_slots(torch.tensor([1]), torch.tensor([2]))
        saved = policy.get_cpu_slots(torch.tensor([2]))
        self.assertEqual(len(saved), 2)
        policy.reset_slots(torch.tensor([2]))
        self.assertEqual(policy.count[0, 2], 0)
        self.assertFalse(policy.pending_prefix[0, 2])
        policy.load_cpu_slots(saved, torch.tensor([3]))
        self.assertEqual(policy.count[0, 3], 5)
        self.assertTrue(policy.pending_prefix[0, 3])

    def test_legacy_policy_keeps_sink_lifecycle_and_snapshot_format(self):
        policy = module.KDAStatePruner(Pool(), {3: torch.ones(2, 8)}, 2, solver="eigh32")
        self.assertEqual(policy.solver, "eigh32")
        self.assertEqual(policy.sinks.shape, (1, 4, 2, 16, 8))
        policy.sinks[:, 1] = 3
        policy.count[:, 1] = 4
        policy.pending_prefix[:, 1] = True
        policy.copy_slots(torch.tensor([1]), torch.tensor([2]))
        saved = policy.get_cpu_slots(torch.tensor([2]))
        self.assertEqual(len(saved), 3)
        policy.reset_slots(torch.tensor([2]))
        self.assertEqual(policy.sinks[:, 2].count_nonzero(), 0)
        policy.load_cpu_slots(saved, torch.tensor([3]))
        self.assertTrue(torch.equal(policy.sinks[:, 3], policy.sinks[:, 1]))
        self.assertEqual(policy.count[0, 3], 4)
        self.assertTrue(policy.pending_prefix[0, 3])

    def test_legacy_projection_retains_sink_and_rank_two_content(self):
        policy = module.KDAStatePruner(Pool(), {3: torch.eye(8)[0].expand(2, -1)}, 2)
        # Orthogonal diagonal content: exact best-rank-2 result is unambiguous.
        state = torch.diag(torch.tensor([3., 8., 6., 4., 2., 1., .5, .25]))
        policy.states[0, 1] = state
        policy.sinks[0, 1, :, 0, 0] = 3
        policy.pending_prefix[0, 1] = True
        policy.flush(torch.tensor([1]), prefix=True)
        expected = torch.diag(torch.tensor([3., 8., 6., 0., 0., 0., 0., 0.])).expand(2, -1, -1)
        torch.testing.assert_close(policy.states[0, 1], expected, rtol=0, atol=0)
        self.assertEqual(policy.prefix_cuts, 1)
        self.assertFalse(policy.pending_prefix[0, 1])


if __name__ == "__main__":
    unittest.main()
