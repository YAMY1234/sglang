import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.arg_groups.deepseek_v4_hook import (
    _validate_deepseek_v41_pp_layout,
)
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    build_prefill_registry,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.runner_utils.buffers import PrefillInputBuffers
from sglang.srt.models.deepseek_v4 import DeepseekV4Model
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from torch import nn

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _dsv41_sparse_config():
    return SimpleNamespace(
        num_hidden_layers=40,
        num_nextn_predict_layers=3,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
    )


class _FakeForwardMode:
    @staticmethod
    def is_extend():
        return False


class _FakeEmbedding(nn.Module):
    def forward(self, input_ids):
        return torch.stack(
            (input_ids.float(), input_ids.float() + 1, input_ids.float() + 2),
            dim=-1,
        )


class _FakePreMixLayer(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.engram = None

    def forward_hc_pre_from_prev(self, *, hidden_states, prev_pre, **_kwargs):
        if prev_pre is None:
            prev_pre = hidden_states.new_zeros(hidden_states.shape[:2])
        hidden_states = hidden_states + prev_pre.unsqueeze(-1) + self.layer_id + 1
        next_pre = hidden_states.float().mean(dim=-1) / (self.layer_id + 2)
        return hidden_states, next_pre


def _make_pre_mix_model(*, start, end, is_first_rank):
    model = DeepseekV4Model.__new__(DeepseekV4Model)
    nn.Module.__init__(model)
    model.pp_group = SimpleNamespace(
        is_first_rank=is_first_rank,
        is_last_rank=False,
        world_size=2,
    )
    model.embed_tokens = _FakeEmbedding()
    model.layers = nn.ModuleList([_FakePreMixLayer(i) for i in range(4)])
    model.start_layer = start
    model.end_layer = end
    model.hc_mult = 2
    model.hidden_size = 3
    model.hc_pre_from_prev_sublayer = True
    model.dspark_layers_to_capture = None
    model.engram_hasher = None
    model.engram_prefetch_stream = None
    model.late_layer_start = None
    model.config = SimpleNamespace(model_type="deepseek_v41", vision_n_layers=0)
    model._can_run_tbo = lambda _forward_batch: False
    return model


class TestDeepseekV41PipelineParallel(CustomTestCase):
    def test_standalone_language_only_mode_accepts_deepseek_v4(self):
        self.assertIn(
            "DeepseekV4ForCausalLM", ServerArgs.LANGUAGE_MODEL_ONLY_ARCHITECTURES
        )

    def test_pp2_partition_covers_40_layers_once_and_keeps_sparse_owners_local(self):
        with patch.dict(
            os.environ, {"SGLANG_PP_LAYER_PARTITION": "20,20"}, clear=False
        ):
            partitions, owners = _validate_deepseek_v41_pp_layout(
                _dsv41_sparse_config(), 2
            )

        self.assertEqual(partitions, ((0, 20), (20, 40)))
        covered = [layer for start, end in partitions for layer in range(start, end)]
        self.assertEqual(covered, list(range(40)))
        self.assertEqual(owners[2:8], (2,) * 6)
        self.assertEqual(owners[8:14], (8,) * 6)
        self.assertEqual(owners[14:20], (14,) * 6)
        self.assertEqual(owners[20:40], (20,) * 20)

    def test_partition_crossing_ratio2_owner_remains_fail_closed(self):
        with (
            patch.dict(os.environ, {"SGLANG_PP_LAYER_PARTITION": "10,30"}, clear=False),
            self.assertRaisesRegex(ValueError, "compressed-KV owner boundary"),
        ):
            _validate_deepseek_v41_pp_layout(_dsv41_sparse_config(), 2)

    def test_pre_mix_boundary_relay_matches_unpartitioned_layer_sequence(self):
        input_ids = torch.tensor([3, 7], dtype=torch.long)
        positions = torch.arange(2)
        forward_batch = SimpleNamespace(forward_mode=_FakeForwardMode())
        parallel = SimpleNamespace(attn_dp_size=1)

        with (
            patch("sglang.srt.models.deepseek_v4.is_cp_v2_active", return_value=False),
            patch(
                "sglang.srt.models.deepseek_v4.dsa_use_prefill_cp", return_value=False
            ),
            patch("sglang.srt.models.deepseek_v4.get_parallel", return_value=parallel),
            patch(
                "sglang.srt.models.deepseek_v4.check_cuda_graph_backend",
                return_value=True,
            ),
        ):
            full = _make_pre_mix_model(start=0, end=4, is_first_rank=True)(
                input_ids, positions, forward_batch, None
            )
            first_stage = _make_pre_mix_model(start=0, end=2, is_first_rank=True)(
                input_ids, positions, forward_batch, None
            )
            second_stage = _make_pre_mix_model(start=2, end=4, is_first_rank=False)(
                input_ids,
                positions,
                forward_batch,
                None,
                pp_proxy_tensors=first_stage,
            )

        self.assertIsInstance(first_stage, PPProxyTensors)
        self.assertIn("hc_prev_pre", first_stage.tensors)
        torch.testing.assert_close(second_stage["hidden_states"], full["hidden_states"])
        torch.testing.assert_close(second_stage["hc_prev_pre"], full["hc_prev_pre"])

    def test_prefill_cuda_graph_registers_float32_pre_mix_buffer(self):
        buffers = PrefillInputBuffers.create(
            device=torch.device("cpu"),
            max_bs=2,
            max_num_tokens=16,
            cache_loc_dtype=torch.int64,
            is_multimodal=False,
            hidden_size=3,
            dtype=torch.bfloat16,
            enable_mamba_track=False,
            pp_size=2,
            is_first_pp_rank=False,
            hc_hidden_size=6,
            hc_prev_pre_dim=2,
        )
        registry = build_prefill_registry(
            device=torch.device("cpu"),
            max_bs=2,
            max_num_token=16,
            cache_loc_dtype=torch.int64,
            source=buffers,
        )

        prev_pre = buffers.pp_proxy_tensors["hc_prev_pre"]
        self.assertEqual(tuple(prev_pre.shape), (16, 2))
        self.assertEqual(prev_pre.dtype, torch.float32)
        self.assertTrue(registry.has_slot("pp_proxy_tensors.hc_prev_pre"))
        self.assertEqual(
            registry.get_slot("pp_proxy_tensors.hc_prev_pre").buffer.data_ptr(),
            prev_pre.data_ptr(),
        )


if __name__ == "__main__":
    unittest.main()
