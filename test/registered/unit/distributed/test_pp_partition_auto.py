"""Boot-time PP split from per-GPU speed (SGLANG_PP_PARTITION_AUTO): the search, its memory floor and the agreement."""

import unittest

from sglang.srt.distributed import pp_partition_auto as ppa
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

# Kimi-K3.1-Pro: 93 layers, full-attention (DSA) layers at 1-indexed 4, 8, ..., 92 and 93.
DSA = {i - 1 for i in list(range(4, 93, 4)) + [93]}
IS_DSA = [i in DSA for i in range(93)]
REFERENCE = (12, 12, 12, 12, 12, 12, 11, 10)
BUDGETS = [286524 * 2**20 * 0.85] * 8
# Per-GPU factors fitted on three rig node pairs with the stages decoupled and P3 on (PP0..PP7 placement).
PAIRS = {
    "A": [0.956, 0.991, 0.990, 0.996, 1.001, 1.043, 1.005, 0.979],
    "B": [0.989, 0.992, 0.957, 0.959, 1.041, 0.980, 1.010, 0.998],
    "C": [1.003, 0.989, 1.065, 1.069, 1.002, 1.006, 0.997, 0.983],
}


def slowest(part, speed):
    return max(
        ppa.stage_cost(ppa.KIMI_K31_PREFILL, IS_DSA, s, sum(part[:s]), part[s], 8) * speed[s]
        for s in range(8)
    )


class TestBestPartition(CustomTestCase):
    def test_reproduces_the_measured_pairs_best_splits(self):
        """Without the memory floor the search finds the split the offline fit found for each pair."""
        expected = {
            "A": (12, 12, 12, 12, 11, 11, 11, 12),
            "B": (11, 12, 12, 12, 11, 12, 11, 12),
            "C": (11, 12, 11, 11, 12, 12, 12, 12),
        }
        for name, speed in PAIRS.items():
            part, value = ppa.best_partition(IS_DSA, speed, ppa.KIMI_K31_PREFILL)
            self.assertEqual(part, expected[name], name)
            self.assertAlmostEqual(value, slowest(part, speed), places=4)
            self.assertLess(value, slowest((11, 12, 12, 12, 12, 12, 12, 10), speed))

    def test_ties_break_on_running_max_then_prefix(self):
        """Uniform layers and GPUs: many splits reach 12; the one with the smallest running max, then prefix, wins."""
        uniform = ppa.StageCostModel(dsa_ms=1.0, kda_ms=1.0)
        part, value = ppa.best_partition([True] * 93, [1.0] * 8, uniform)
        self.assertEqual((part, value), ((11, 11, 11, 12, 12, 12, 12, 12), 12.0))

    def test_every_stage_keeps_the_reference_kv_capacity(self):
        """KV token capacity is the minimum over stages: no stage may fall below the reference split's minimum."""
        m = ppa.KIMI_K31_PREFILL
        floor = ppa.min_capacity(m, IS_DSA, REFERENCE, BUDGETS)
        for speed in PAIRS.values():
            part, _ = ppa.best_partition(IS_DSA, speed, m, BUDGETS, floor)
            self.assertGreaterEqual(ppa.min_capacity(m, IS_DSA, part, BUDGETS), floor)
            # The unconstrained split puts 12 layers on PP7, which holds the draft and the LM head.
            self.assertLessEqual(part[7], 10)
        with self.assertRaises(RuntimeError):
            ppa.best_partition(IS_DSA, PAIRS["A"], m, BUDGETS, floor * 2)


    def test_no_stage_is_left_without_a_full_attention_layer(self):
        """A stage of KDA layers only has no KV pool and the hybrid HiCache split divides by zero at boot; a slow
        first GPU must not shed layer 1 (its only full-attention layer) and keep KDA layer 0 alone."""
        tiny = [i % 2 == 1 for i in range(16)]  # 16 layers, KDA/DSA alternating, two per stage by default
        uniform = ppa.StageCostModel(dsa_ms=1.0, kda_ms=1.0)
        part, _ = ppa.best_partition(tiny, [1.6] + [1.0] * 7, uniform)
        starts = [sum(part[:s]) for s in range(8)]
        self.assertGreaterEqual(part[0], 2)
        self.assertTrue(all(any(tiny[st : st + n]) for st, n in zip(starts, part)), part)

    def test_no_stage_is_left_without_a_linear_attention_layer(self):
        """In a hybrid model a stage of full-attention layers only has an empty Mamba pool, and the HiCache host
        split rejects a 0 GB Mamba share at boot (a slow PP3 once got the split 2,2,3,1,2,2,2,2 and PP3 kept one
        full-attention layer alone)."""
        tiny = [i % 2 == 1 for i in range(16)]
        part, _ = ppa.best_partition(tiny, [1.0] * 3 + [1.4] + [1.0] * 4, ppa.KIMI_K31_PREFILL)
        starts = [sum(part[:s]) for s in range(8)]
        self.assertTrue(all(not all(tiny[st : st + n]) for st, n in zip(starts, part)), part)


class TestResolve(CustomTestCase):
    def run_ranks(self, speeds_by_rank, pp_of_rank, chosen_by_rank0=None):
        rows = [(pp, speeds_by_rank[r], BUDGETS[0]) for r, pp in enumerate(pp_of_rank)]
        out = {}
        for r, pp in enumerate(pp_of_rank):
            out[r] = ppa.resolve(
                is_dsa=IS_DSA,
                pp_rank=pp,
                pp_size=8,
                world_rank=r,
                all_gather=lambda _obj: rows,
                broadcast_from_0=lambda obj: chosen_by_rank0 if chosen_by_rank0 is not None else obj,
                measure=lambda r=r: speeds_by_rank[r],
                budget=BUDGETS[0],
                m=ppa.KIMI_K31_PREFILL,
                reference=REFERENCE,
            )
        return out

    def test_a_stage_runs_at_its_slowest_rank_and_all_ranks_agree(self):
        """With attention TP 2 a stage's two GPUs both run its layers, so the slower one sets the stage's speed."""
        pp_of_rank = [r // 2 for r in range(16)]
        speeds = [1.0] * 16
        speeds[5] = 1.06  # stage 2's second GPU is slow
        out = self.run_ranks(speeds, pp_of_rank)
        parts = {part for part, _ in out.values()}
        self.assertEqual(len(parts), 1)
        report = out[0][1]
        self.assertAlmostEqual(report["factors"][2], max(report["factors"]))
        self.assertEqual(report["source"], "burn")

    def test_a_rank_that_disagrees_with_rank_0_stops(self):
        with self.assertRaises(AssertionError):
            self.run_ranks([1.0] * 8, list(range(8)), chosen_by_rank0=(93, 0, 0, 0, 0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
