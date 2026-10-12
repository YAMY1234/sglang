"""CPU checks of sglang.srt.duet.dense_state_policy.DenseStatePolicy, the cut on SGLang's dense Mamba pool that both hooked
families serve with: (1) the Mamba-2 (left sink) dense pool reproduces the Lightning functional factor pool
(state_pool.LeftSinkStatePool: cold cut at the prompt end, exact rank-1 updates, warm cut every W steps) over several
windows; (2) the KDA layout (right sink, pool stored transposed) cuts every due (layer, slot) as the per-slot reference does;
(3) slot reset / copy / offload keep the warm bases with their slots.

Run from this directory:  python -m unittest test_duet_dense_state_policy
"""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[4]   # test/registered/unit/duet -> repo root
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/sglang/srt/models"))

# the policy base class by path, registered under its import name so the module under test finds it without sglang/__init__
_KDA = "sglang.srt.layers.attention.linear.kda_state_prune"
_spec = importlib.util.spec_from_file_location(_KDA, ROOT / "python/sglang/srt/layers/attention/linear/kda_state_prune.py")
sys.modules[_KDA] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sys.modules[_KDA])

from lightning_duet._common import load as _load  # noqa: E402

from glm_duet import _dense_state_policy as dense_state_policy
from glm_duet import _state_factor as state_factor
state_pool = _load("state_pool")


class Pool:
    """The slice of HybridReqToTokenPool the policy reads: the dense temporal pool (layers, slots, H, P, N), the layer map,
    slot ids (physical = virtual here) and the sibling registration."""

    def __init__(self, layer_ids, slots, heads, p, n):
        self.layer_ids = list(layer_ids)
        self.mamba_pool = SimpleNamespace(
            mamba_cache=SimpleNamespace(temporal=torch.zeros(len(self.layer_ids), slots, heads, p, n)),
            register_slot_state=lambda state: None,
        )

    def mamba2_layer_index(self, layer):
        return self.layer_ids.index(layer)

    def get_mamba_indices(self, req_pool_indices):
        return req_pool_indices


def batch_of(slots):
    return SimpleNamespace(req_pool_indices=torch.tensor(slots), batch_size=len(slots))


def rel(a, b):
    return float((a - b).norm() / (b.norm() + 1e-12))


class DenseStatePolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_mamba_dense_pool_reproduces_the_functional_factor_pool(self):
        layers, slots, heads, p, n, rank, window = 3, 4, 2, 8, 16, 3, 4
        directions = torch.randn(layers, heads, p)
        spec = {"state_rank": rank, "state_every": window, "state_sink": "explicit"}
        pool = Pool(range(layers), slots, heads, p, n)
        policy = dense_state_policy.DenseStatePolicy(pool, directions, spec, list(range(layers)), side="left",
                                                     pool_transposed=False, tag="test")
        functional = state_pool.LeftSinkStatePool.for_test(slots, directions, n, rank, window)
        active = [0, 2, 3]
        dense = pool.mamba_pool.mamba_cache.temporal
        for slot in active:     # prompt-final states
            for layer in range(layers):
                state = torch.randn(heads, p, n) + 2.0 * directions[layer][:, :, None] * torch.randn(heads, 1, n)
                dense[layer, slot] = state
                functional.initialize(layer, slot, state, torch.zeros(4, 3))
        policy.note_prompt_end(torch.tensor(active))
        for step in range(3 * window + 1):
            for slot in active:
                for layer in range(layers):
                    decay = torch.rand(heads) * 0.5 + 0.5
                    scaled_x, b = torch.randn(heads, p), torch.randn(heads, n)
                    dense[layer, slot] = dense[layer, slot] * decay[:, None, None] + scaled_x[:, :, None] * b[:, None, :]
                    functional.step(layer, slot, decay, scaled_x, b, torch.zeros(4, 3))
            policy.after_graph(batch_of(active))
            for slot in active:
                for layer in range(layers):
                    self.assertLess(rel(dense[layer, slot], functional.materialize(layer, slot)), 1e-4,
                                    f"step {step} layer {layer} slot {slot}")
        self.assertEqual(policy.prefix_cuts, layers * len(active))
        self.assertEqual(policy.decode_cuts, 3 * layers * len(active))

    def test_kda_layout_cuts_like_the_per_slot_reference(self):
        layers, slots, heads, v, k, rank = 2, 3, 2, 16, 12, 4
        directions = torch.randn(layers, heads, v)
        spec = {"state_rank": rank, "state_every": 2, "state_sink": "explicit"}
        pool = Pool(range(layers), slots, heads, v, k)
        policy = dense_state_policy.DenseStatePolicy(pool, directions, spec, list(range(layers)), side="right",
                                                     pool_transposed=True, tag="test")
        dense = pool.mamba_pool.mamba_cache.temporal
        dense.normal_()
        before = dense.clone()
        policy.note_prompt_end(torch.tensor([0, 2]))
        for slot in (0, 2):
            for layer in range(layers):
                ref, _ = state_factor.project_state(before[layer, slot].transpose(-1, -2)[None], directions[layer], rank,
                                                    side="right")
                self.assertLess(rel(dense[layer, slot], ref[0].transpose(-1, -2)), 1e-5)
        self.assertTrue(torch.equal(dense[:, 1], before[:, 1]))      # slot 1 had no prompt end: untouched

    def test_warm_bases_follow_their_slots(self):
        layers, slots, heads, p, n = 1, 4, 2, 8, 16
        spec = {"state_rank": 3, "state_every": 1, "state_sink": "explicit"}
        pool = Pool(range(layers), slots, heads, p, n)
        policy = dense_state_policy.DenseStatePolicy(pool, torch.randn(layers, heads, p), spec, [0], side="left",
                                                     pool_transposed=False, tag="test")
        pool.mamba_pool.mamba_cache.temporal.normal_()
        policy.note_prompt_end(torch.tensor([1]))
        policy.after_graph(batch_of([1]))
        self.assertIn((0, 1), policy.warm)
        policy.copy_slots(torch.tensor([1]), torch.tensor([3]))
        self.assertTrue(torch.equal(policy.warm[(0, 3)], policy.warm[(0, 1)]))
        saved = policy.get_cpu_slots(torch.tensor([3]))
        policy.reset_slots(torch.tensor([3]))
        self.assertNotIn((0, 3), policy.warm)
        policy.load_cpu_slots(saved, torch.tensor([2]))
        self.assertTrue(torch.equal(policy.warm[(0, 2)], policy.warm[(0, 1)]))


if __name__ == "__main__":
    unittest.main()
