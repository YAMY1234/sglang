"""KDA ReplaySSM spec window is request-scoped scratch, not per mamba slot.

The window must stay one row per running request (like intermediate_conv_window)
and one record per draft token, with no chunked d/k rings; the GDN window keeps
its per-slot keying. Regression guard for the pool sizing that decides how much
of the mamba budget is left for state slots.
"""

import unittest

import torch

from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

DTYPE = Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32)
LAYERS = [0, 1]
NUM_SLOTS = 12
SPEC_STATE_SIZE = 3
DRAFT_TOKENS = 4
CACHE_LEN = 16


def _kda_params():
    shape = KimiLinearStateShape.create(
        tp_world_size=1, num_heads=4, head_dim=8, num_k_heads=4, head_k_dim=8
    )
    return KimiLinearCacheParams(shape=shape, dtype=DTYPE, layers=LAYERS)


def _gdn_params():
    shape = Mamba2StateShape(
        conv=[(4 * 8 * 3, 3)],
        temporal=(4, 8, 8),
        intermediate_size=0,
        conv_dim=0,
        ssm_state_size=0,
        num_heads=0,
        head_dim=0,
        state_size=0,
        conv_kernel=4,
        num_k_heads_per_tp=4,
    )
    return Mamba2CacheParams(shape=shape, dtype=DTYPE, layers=LAYERS)


def _make_pool(cache_params):
    return MambaPool(
        size=NUM_SLOTS,
        spec_state_size=SPEC_STATE_SIZE,
        cache_params=cache_params,
        mamba_layer_ids=LAYERS,
        device="cuda",
        speculative_num_draft_tokens=DRAFT_TOKENS,
        linear_replayssm_cache_len=CACHE_LEN,
        enable_linear_replayssm_spec=True,
    )


class TestKDAReplaySSMWindowRows(CustomTestCase):
    def test_kda_window_is_request_scoped(self):
        pool = _make_pool(_kda_params())
        cache = pool.mamba_cache
        hv, v_dim, k_dim = _kda_params().shape.temporal
        rows = SPEC_STATE_SIZE + 1
        self.assertEqual(
            tuple(cache.replayssm_rawv.shape),
            (len(LAYERS), rows, hv, DRAFT_TOKENS, v_dim),
        )
        self.assertEqual(
            tuple(cache.replayssm_rawk.shape),
            (len(LAYERS), rows, hv, DRAFT_TOKENS, k_dim),
        )
        self.assertEqual(
            tuple(cache.replayssm_g.shape), (len(LAYERS), rows, hv, DRAFT_TOKENS, k_dim)
        )
        self.assertEqual(
            tuple(cache.replayssm_beta.shape), (len(LAYERS), rows, hv, DRAFT_TOKENS)
        )
        self.assertIsNone(cache.replayssm_d)
        self.assertIsNone(cache.replayssm_k)
        self.assertIsNone(cache.intermediate_ssm)
        # Conv scratch and the window share the per-request row space.
        self.assertEqual(cache.intermediate_conv_window[0].shape[1], rows)
        self.assertTrue(pool.replayssm_is_kda)
        self.assertTrue(pool.replayssm_spec_fold)

    def test_gdn_window_stays_slot_keyed(self):
        pool = _make_pool(_gdn_params())
        cache = pool.mamba_cache
        hv, v_dim, _ = _gdn_params().shape.temporal
        self.assertEqual(
            tuple(cache.replayssm_rawv.shape),
            (len(LAYERS), NUM_SLOTS + 1, hv, DRAFT_TOKENS, v_dim),
        )
        self.assertEqual(
            tuple(cache.replayssm_g.shape),
            (len(LAYERS), NUM_SLOTS + 1, hv, DRAFT_TOKENS),
        )
        self.assertIsNone(cache.replayssm_d)
        self.assertFalse(pool.replayssm_is_kda)


if __name__ == "__main__":
    unittest.main()
