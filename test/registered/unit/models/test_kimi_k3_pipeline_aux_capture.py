import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models import kimi_k3
from sglang.srt.models.kimi_k3 import (
    KimiK3ForConditionalGeneration,
    KimiK3LinearForCausalLM,
    KimiK3LinearModel,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Layer(nn.Module):
    _sp_moe = False

    def __init__(self, layer_id: int):
        super().__init__()
        self.layer_id = layer_id

    def forward(self, *, hidden_states, **kwargs):
        return hidden_states + self.layer_id + 1, None, False


class TestKimiK3PipelineAuxCapture(CustomTestCase):
    def setUp(self):
        super().setUp()
        recorder = SimpleNamespace(with_current_layer=lambda _: nullcontext())
        patcher = patch.object(
            kimi_k3,
            "get_global_expert_distribution_recorder",
            return_value=recorder,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _make_stage(start: int, end: int):
        pp_group = SimpleNamespace(
            is_first_rank=start == 0,
            is_last_rank=end == 6,
            world_size=3,
        )
        model = KimiK3LinearModel.__new__(KimiK3LinearModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            num_hidden_layers=6,
            hidden_size=3,
            attn_res_block_size=None,
        )
        model.pp_group = pp_group
        model.start_layer, model.end_layer = start, end
        model.layers = nn.ModuleList(
            _Layer(i) if start <= i < end else nn.Identity() for i in range(6)
        )
        model.dspark_layers_to_capture = None
        model._trim_padded_attn = False
        model.norm = nn.Identity()

        wrapper = KimiK3LinearForCausalLM.__new__(KimiK3LinearForCausalLM)
        nn.Module.__init__(wrapper)
        wrapper.config = model.config
        wrapper.model = model
        wrapper.pp_group = pp_group
        wrapper.capture_aux_hidden_states = False
        wrapper.lm_head = nn.Identity()
        wrapper.logits_processor = Mock(side_effect=lambda *args, **kwargs: args[4])
        return wrapper

    @staticmethod
    def _forward(stage, embeds, proxy=None):
        with patch.object(kimi_k3, "get_pp_group", return_value=stage.pp_group):
            return stage(
                torch.zeros(embeds.shape[0], dtype=torch.long),
                torch.arange(embeds.shape[0]),
                SimpleNamespace(),
                input_embeds=embeds,
                pp_proxy_tensors=proxy,
            )

    def test_cross_stage_capture_matches_single_stage_and_requested_order(self):
        embeds = torch.arange(15, dtype=torch.float32).view(5, 3)
        for layer_ids in ([5, 0, 2, 1, 4, 3], [0], [2], [5], [4, 0]):
            with self.subTest(layers=layer_ids):
                single = self._make_stage(0, 6)
                single.set_dspark_layers_to_capture(layer_ids)
                expected_aux = self._forward(single, embeds)

                proxy = None
                for start, end in ((0, 2), (2, 4), (4, 6)):
                    stage = self._make_stage(start, end)
                    stage.set_dspark_layers_to_capture(layer_ids)
                    output = self._forward(stage, embeds, proxy)
                    if end == 6:
                        actual_aux = output
                        break
                    expected_keys = {"hidden_states", "residual"} | {
                        f"dspark_aux_hidden_states_{i}" for i in layer_ids if i < end
                    }
                    self.assertEqual(set(output.tensors), expected_keys)
                    proxy = PPProxyTensors(
                        {
                            key: value.clone() if value is not None else None
                            for key, value in output.tensors.items()
                        }
                    )

                torch.testing.assert_close(actual_aux, expected_aux)
                torch.testing.assert_close(
                    actual_aux,
                    [embeds + (i + 1) * (i + 2) / 2 for i in layer_ids],
                )

    def test_configuration_sizes_only_incoming_capture_buffers(self):
        layer_ids = [4, 0, 3]
        for start, end in ((0, 2), (2, 4), (4, 6)):
            stage = self._make_stage(start, end)
            stage.set_dspark_layers_to_capture(layer_ids)
            self.assertTrue(stage.capture_aux_hidden_states)
            self.assertEqual(stage.model.dspark_layers_to_capture, layer_ids)
            self.assertEqual(
                stage.pp_proxy_aux_hidden_state_keys,
                tuple(f"dspark_aux_hidden_states_{i}" for i in layer_ids if i < start),
            )

    def test_multimodal_wrapper_exposes_graph_buffer_keys(self):
        language_model = self._make_stage(4, 6)
        wrapper = KimiK3ForConditionalGeneration.__new__(KimiK3ForConditionalGeneration)
        nn.Module.__init__(wrapper)
        wrapper.language_model = language_model
        wrapper.set_dspark_layers_to_capture([4, 0, 3])
        self.assertEqual(
            wrapper.pp_proxy_aux_hidden_state_keys,
            ("dspark_aux_hidden_states_0", "dspark_aux_hidden_states_3"),
        )

    def test_invalid_layers_fail_on_every_stage(self):
        for start, end in ((0, 2), (2, 4), (4, 6)):
            for layer_ids in (None, [], [0, 0], [-1], [6]):
                with self.subTest(start=start, layers=layer_ids):
                    stage = self._make_stage(start, end)
                    with self.assertRaises(ValueError):
                        stage.set_dspark_layers_to_capture(layer_ids)
                    self.assertFalse(stage.capture_aux_hidden_states)


if __name__ == "__main__":
    unittest.main()
