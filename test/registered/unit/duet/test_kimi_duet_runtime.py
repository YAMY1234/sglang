"""CPU request-slot lifecycle test; imports only the actual policy classes."""
import ast
import importlib.util
from pathlib import Path
import unittest

import torch
import sys
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/sglang/srt/models"))
from lightning_duet._common import load
project_state = load("state_factor").project_state


def policy_class():
    fork = ROOT
    path = fork / "python/sglang/srt/layers/attention/linear/kda_state_prune.py"
    spec = importlib.util.spec_from_file_location("kda_policy_under_test", path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    source = ROOT / "python/sglang/srt/models/kimi_linear_duet/runtime.py"
    cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "DuetStatePruner")
    ns = {"torch": torch, "KDAStatePolicy": base.KDAStatePolicy, "project_state": project_state}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), ns)
    return ns["DuetStatePruner"], base.KDAStatePruner


class Pool:
    def __init__(self):
        from types import SimpleNamespace
        self.mamba_pool = SimpleNamespace(mamba_cache=SimpleNamespace(temporal=torch.zeros(2, 4, 2, 32, 32)),
                                          register_slot_state=lambda x: None)
    def mamba2_layer_index(self, layer):
        return {0: 0, 2: 1}[layer]


class RuntimeTests(unittest.TestCase):
    def test_graph_counter_ignores_padding_and_preserves_request_cadence(self):
        from types import SimpleNamespace
        cls, _ = policy_class()
        policy = cls(Pool(), torch.zeros(3, 2, 32),
                     {"state_rank": 8, "state_every": 16}, [0, 2], graph_safe=True)
        for slots in (torch.tensor([1, -1, -1]), torch.tensor([2, 1, -1])):
            policy.decode(None, SimpleNamespace(layer_id=0), None, None, None, slots, None)
        self.assertEqual(policy.count[0].tolist(), [0, 2, 1, 0])
        self.assertEqual(policy.count[1].tolist(), [0, 0, 0, 0])
        policy.copy_slots(torch.tensor([1]), torch.tensor([3]))
        policy.reset_slots(torch.tensor([1]))
        self.assertEqual(policy.count[0].tolist(), [0, 0, 1, 2])

    def test_warm_basis_reset_copy_and_restore(self):
        torch.set_num_threads(1)
        cls, _ = policy_class()
        torch.manual_seed(41)
        policy = cls(Pool(), torch.randn(3, 2, 32), {"state_rank": 8, "state_every": 16}, [0, 2])
        self.assertFalse(hasattr(policy, "sinks"))
        self.assertFalse(hasattr(policy, "solver"))
        policy.states[0, 1].normal_()
        policy.pending_prefix[0, 1] = True
        policy.flush(torch.tensor([1]), prefix=True, layer_id=0)
        self.assertEqual(policy.prefix_cuts, 1)
        self.assertIn((0, 1), policy.warm)
        policy.copy_slots(torch.tensor([1]), torch.tensor([2]))
        self.assertTrue(torch.equal(policy.warm[0, 1], policy.warm[0, 2]))
        self.assertNotEqual(policy.warm[0, 1].data_ptr(), policy.warm[0, 2].data_ptr())
        saved = policy.get_cpu_slots(torch.tensor([2]))
        self.assertEqual(len(saved[0]), 2)  # count + pending_prefix; no legacy sink
        policy.reset_slots(torch.tensor([2]))
        self.assertNotIn((0, 2), policy.warm)
        policy.load_cpu_slots(saved, torch.tensor([3]))
        self.assertTrue(torch.equal(policy.warm[0, 1], policy.warm[0, 3]))

    def test_boundary_counts_as_first_decode_and_cut_is_at_sixteen(self):
        torch.set_num_threads(1)
        cls, _ = policy_class()
        from types import SimpleNamespace
        policy = cls(Pool(), torch.zeros(3, 2, 32), {"state_rank": 8, "state_every": 16}, [0, 2])
        slots = torch.tensor([1])
        for step in range(1, 17):
            policy.decode(None, SimpleNamespace(layer_id=0), None, None, None, slots, None)
            policy.flush(slots, prefix=False, layer_id=0)
            self.assertEqual(policy.decode_cuts, int(step == 16))
        self.assertEqual(policy.count[0, 1].item(), 0)

    def test_factory_hook_and_flagoff(self):
        _, base = policy_class()
        from types import SimpleNamespace
        from unittest.mock import patch
        with patch.dict("os.environ", {"SGLANG_KDA_STATE_PRUNE_RANK": "0"}):
            self.assertIsNone(base.from_runner(SimpleNamespace(model=SimpleNamespace())))
            runner = SimpleNamespace(model=SimpleNamespace(kda_state_pruner_factory=lambda r: "checkpoint-policy"))
            self.assertEqual(base.from_runner(runner), "checkpoint-policy")

    def test_zero_cadence_keeps_prefix_cut_and_never_decode_cuts(self):
        torch.set_num_threads(1)
        cls, _ = policy_class()
        from types import SimpleNamespace
        policy = cls(Pool(), torch.zeros(3, 2, 32),
                     {"state_rank": 7, "state_every": 0, "state_sink": "implicit"}, [0, 2])
        slots = torch.tensor([1])
        policy.states[0, 1].normal_()
        policy.pending_prefix[0, 1] = True
        policy.flush(slots, prefix=True, layer_id=0)
        self.assertEqual(policy.prefix_cuts, 1)
        saved = policy.states.clone()
        for _ in range(33):
            policy.decode(None, SimpleNamespace(layer_id=0), None, None, None, slots, None)
            policy.flush(slots, prefix=False, layer_id=0)
        self.assertEqual(policy.decode_cuts, 0)
        self.assertTrue(torch.equal(saved, policy.states))

    def test_full_rank_has_no_warm_basis_to_copy_or_offload(self):
        torch.set_num_threads(1)
        cls, _ = policy_class()
        policy = cls(Pool(), torch.zeros(3, 2, 32),
                     {"state_rank": 32, "state_every": 5, "state_sink": "implicit"}, [0, 2])
        policy.states[0, 1].normal_()
        saved = policy.states.clone()
        policy.pending_prefix[0, 1] = True
        policy.flush(torch.tensor([1]), prefix=True, layer_id=0)
        self.assertTrue(torch.equal(saved, policy.states))
        self.assertEqual(policy.warm, {})
        policy.copy_slots(torch.tensor([1]), torch.tensor([2]))
        policy.load_cpu_slots(policy.get_cpu_slots(torch.tensor([2])), torch.tensor([3]))
        self.assertEqual(policy.warm, {})


if __name__ == "__main__":
    unittest.main()
