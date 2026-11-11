import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import deepseek_v4_hook
from sglang.srt.arg_groups.engram_config import EngramPlacement
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# DeepSeek-V4.1-Flash (deepseek-ai/DeepSeek-V4.1-Flash config.json, rev dba1be0).
V41_FLASH = SimpleNamespace(
    model_type="deepseek_v41",
    num_hidden_layers=40,
    compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
    kv_source_layer_ids=[2, 8, 14, 20],
    index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
    candidate_source_layer_id=20,
)


def _view(**overrides):
    values = dict(
        pp_size=2,
        tp_size=1,
        attn_cp_size=1,
        attn_dp_size=1,
        dcp_size=1,
        enable_prefill_cp=False,
        ep_join_mode=None,
        moe_a2a_backend="none",
        speculative_algorithm=None,
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="mooncake",
        enable_encoder_swa_bounded_replay=False,
        enable_decoder_swa_bounded_replay=True,
        enable_two_batch_overlap=False,
        enable_hisparse=False,
        dsv4_attn_backend="flashmla_mega",
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend=Backend.DISABLED)
        ),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _validate(view, partition=None):
    env = {} if partition is None else {"SGLANG_PP_LAYER_PARTITION": partition}
    with (
        patch.dict(os.environ, env),
        patch.object(deepseek_v4_hook, "resolving_view", return_value=view),
        patch.object(
            deepseek_v4_hook,
            "model_config_of",
            return_value=SimpleNamespace(hf_config=V41_FLASH, hf_text_config=V41_FLASH),
        ),
        patch(
            "sglang.srt.layers.attention.dsv4.unified_kv_gate.is_unified_kv_triton",
            return_value=False,
        ),
    ):
        if partition is None:
            os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)
        deepseek_v4_hook.validate_deepseek_v41_features(object())


class TestDeepSeekV41PPGate(unittest.TestCase):
    def test_a_stage_starts_only_where_no_source_span_is_cut(self):
        """Layers 21-39 read layer 20's compressed KV, index keys and candidates,
        and ratio-2 layers read the latest of layers 2, 8 and 14: a stage may
        start only at a layer from which every source it reads is its own."""
        boundaries = [
            start
            for start in range(1, 40)
            if deepseek_v4_hook.dsv41_pp_foreign_source_layer(V41_FLASH, start, 40)
            is None
        ]
        self.assertEqual(boundaries, [1, 2, 8, 14, 20])

    def test_partitions_on_a_pd_prefill(self):
        for pp_size, partition in (
            (2, None),
            (2, "20,20"),
            (2, "8,32"),
            (4, "8,6,6,20"),
        ):
            with self.subTest(pp_size=pp_size, partition=partition):
                _validate(_view(pp_size=pp_size), partition)
        # The even PP4 split starts stage 1 at layer 10, which reads layer 8.
        for pp_size, partition in ((4, None), (2, "21,19")):
            with self.subTest(pp_size=pp_size, partition=partition):
                with self.assertRaisesRegex(ValueError, r"\[1, 2, 8, 14, 20\]"):
                    _validate(_view(pp_size=pp_size), partition)

    def test_pp_runs_only_on_a_mooncake_pd_prefill(self):
        for overrides in (
            dict(disaggregation_mode="null"),
            dict(disaggregation_mode="decode"),
            dict(disaggregation_transfer_backend="nixl"),
        ):
            with self.subTest(**overrides):
                with self.assertRaisesRegex(ValueError, "pipeline parallelism"):
                    _validate(_view(**overrides))

    def test_pp_keeps_engram_tables_in_host_memory_by_default(self):
        self.assertEqual(
            deepseek_v4_hook._engram_config_from_env(_view()).placement,
            EngramPlacement.CPU,
        )
        self.assertEqual(
            deepseek_v4_hook._engram_config_from_env(_view(pp_size=1)).placement,
            EngramPlacement.GPU,
        )


if __name__ == "__main__":
    unittest.main()
