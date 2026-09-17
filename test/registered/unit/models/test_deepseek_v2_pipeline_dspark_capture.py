import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models import deepseek_v2
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV2Model
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _Layer(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        residual,
        zero_allocator,
        gemm_output_zero_allocator=None,
        llama_4_scaling=None,
        captured_last_layer_outputs=None,
        **kwargs,
    ):
        if captured_last_layer_outputs is not None:
            captured_last_layer_outputs.append(hidden_states)
        return hidden_states + self.layer_id + 1, None, None


class TestDeepseekV2PipelineDSparkCapture(CustomTestCase):
    def setUp(self):
        super().setUp()
        for name, kwargs in (
            ("check_cuda_graph_backend", {"return_value": True}),
            (
                "get_attn_tp_context",
                {
                    "return_value": SimpleNamespace(
                        maybe_input_scattered=lambda _: nullcontext()
                    )
                },
            ),
        ):
            patcher = patch.object(deepseek_v2, name, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def _make_stage(start, end):
        model = DeepseekV2Model.__new__(DeepseekV2Model)
        nn.Module.__init__(model)
        model.start_layer, model.end_layer = start, end
        model.pp_group = SimpleNamespace(
            is_first_rank=start == 0,
            is_last_rank=end == 6,
            world_size=3,
        )
        model.config = SimpleNamespace(num_hidden_layers=6)
        model.layers = nn.ModuleList(
            _Layer(i) if start <= i < end else nn.Identity() for i in range(6)
        )
        model.use_dsa = False
        model.first_k_dense_replace = 0
        model.next_full_attention_layer_id = dict(
            zip(range(start, end), range(start + 1, end))
        )
        model.layers_to_capture = []
        model.dspark_layers_to_capture = None
        model.llama_4_scaling_config = None
        model.norm = nn.Identity()

        wrapper = DeepseekV2ForCausalLM.__new__(DeepseekV2ForCausalLM)
        nn.Module.__init__(wrapper)
        wrapper.model = model
        wrapper.pp_group = model.pp_group
        wrapper.config = model.config
        wrapper.capture_aux_hidden_states = False
        wrapper.lm_head = nn.Identity()
        wrapper.logits_processor = Mock(side_effect=lambda *args, **kwargs: args[4])
        return wrapper

    @staticmethod
    def _forward(stage, embeds, proxy=None):
        num_tokens = embeds.shape[0]
        forward_mode = SimpleNamespace(is_idle=lambda: False)
        return stage.model.forward(
            input_ids=torch.zeros(num_tokens, dtype=torch.long),
            positions=torch.arange(num_tokens),
            forward_batch=SimpleNamespace(can_run_tbo=False, forward_mode=forward_mode),
            input_embeds=embeds,
            pp_proxy_tensors=proxy,
        )

    def test_cross_stage_capture_matches_single_stage_and_requested_order(self):
        for num_tokens in (0, 1, 5):
            for layer_ids in ([4, 0, 2], [0], [2], [4]):
                with self.subTest(tokens=num_tokens, layers=layer_ids):
                    embeds = torch.arange(num_tokens * 3, dtype=torch.float32).view(
                        num_tokens, 3
                    )
                    single = self._make_stage(0, 6)
                    single.pp_group = single.model.pp_group = SimpleNamespace(
                        is_first_rank=True, is_last_rank=True, world_size=1
                    )
                    single.set_dspark_layers_to_capture(layer_ids)
                    expected_output, expected_aux = self._forward(single, embeds)

                    proxy = None
                    for start, end in ((0, 2), (2, 4), (4, 6)):
                        stage = self._make_stage(start, end)
                        stage.set_dspark_layers_to_capture(layer_ids)
                        output = self._forward(stage, embeds, proxy)
                        if end == 6:
                            actual_output, actual_aux = output
                            break
                        adjusted_ids = [i + 1 for i in layer_ids]
                        expected_keys = {"hidden_states", "residual"} | {
                            f"dspark_aux_hidden_states_{i}"
                            for i in adjusted_ids
                            if i < end
                        }
                        self.assertEqual(set(output.tensors), expected_keys)
                        proxy = PPProxyTensors(
                            {
                                key: value.clone() if value is not None else None
                                for key, value in output.tensors.items()
                            }
                        )

                    torch.testing.assert_close(actual_output, expected_output)
                    torch.testing.assert_close(actual_aux, expected_aux)
                    torch.testing.assert_close(
                        pack_aux_hidden_states(actual_aux),
                        torch.cat(
                            [embeds + (i + 1) * (i + 2) / 2 for i in layer_ids],
                            dim=-1,
                        ),
                    )

    def test_configuration_enables_all_stages_and_sizes_incoming_captures(self):
        layer_ids = [4, 0, 2]
        adjusted_ids = [5, 1, 3]
        for start, end in ((0, 2), (2, 4), (4, 6)):
            stage = self._make_stage(start, end)
            stage.set_dspark_layers_to_capture(layer_ids)
            self.assertTrue(stage.capture_aux_hidden_states)
            self.assertEqual(stage.model.dspark_layers_to_capture, adjusted_ids)
            self.assertEqual(
                stage.pp_proxy_aux_hidden_state_keys,
                tuple(
                    f"dspark_aux_hidden_states_{i}" for i in adjusted_ids if i < start
                ),
            )

    def test_invalid_layers_fail_on_every_stage(self):
        for start, end in ((0, 2), (2, 4), (4, 6)):
            for layer_ids in (None, [], [0, 0], [-1], [5], [6]):
                with self.subTest(start=start, layers=layer_ids):
                    stage = self._make_stage(start, end)
                    with self.assertRaises(ValueError):
                        stage.set_dspark_layers_to_capture(layer_ids)
                    self.assertFalse(stage.capture_aux_hidden_states)

    def test_missing_upstream_capture_fails(self):
        stage = self._make_stage(4, 6)
        stage.set_dspark_layers_to_capture([0, 4])
        with self.assertRaisesRegex(KeyError, "dspark_aux_hidden_states_1"):
            self._forward(
                stage,
                torch.zeros(2, 3),
                PPProxyTensors(
                    {
                        "hidden_states": torch.zeros(2, 3),
                        "residual": None,
                    }
                ),
            )

    def test_wrapper_only_unpacks_captures_on_last_stage(self):
        embeds = torch.zeros(2, 3)
        proxy = None
        for start, end in ((0, 2), (2, 4), (4, 6)):
            stage = self._make_stage(start, end)
            stage.set_dspark_layers_to_capture([2, 0])
            output = stage.forward(
                torch.zeros(2, dtype=torch.long),
                torch.arange(2),
                SimpleNamespace(
                    can_run_tbo=False,
                    forward_mode=SimpleNamespace(is_idle=lambda: False),
                ),
                input_embeds=embeds,
                pp_proxy_tensors=proxy,
            )
            if end < 6:
                self.assertIsInstance(output, PPProxyTensors)
                stage.logits_processor.assert_not_called()
                proxy = output
            else:
                stage.logits_processor.assert_called_once()
                torch.testing.assert_close(
                    output,
                    torch.cat([embeds + 6, embeds + 1], dim=-1),
                )


if __name__ == "__main__":
    unittest.main()
