import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.attention.attention_registry import (
    attn_backend_wrapper,
    maybe_auto_select_flashinfer_gdn_backends,
    should_auto_select_flashinfer_gdn_decode,
    should_auto_select_flashinfer_gdn_prefill,
)
from sglang.srt.layers.attention.linear.utils import (
    get_linear_attn_decode_backend,
    get_linear_attn_prefill_backend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_runner(**overrides):
    args = SimpleNamespace(
        linear_attn_backend="triton",
        linear_attn_decode_backend=None,
        linear_attn_prefill_backend=None,
        mamba_ssm_dtype="bfloat16",
        speculative_algorithm=None,
        speculative_eagle_topk=None,
    )
    config = SimpleNamespace(
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    runner = SimpleNamespace(server_args=args, hybrid_gdn_config=config)
    for name, value in overrides.items():
        if name.startswith("config_"):
            setattr(config, name.removeprefix("config_"), value)
        elif name == "hybrid_gdn_config":
            runner.hybrid_gdn_config = value
        else:
            setattr(args, name, value)
    return runner


class TestFlashInferGDNPrefillBackendPolicy(unittest.TestCase):
    def should_select(self, runner, **hardware):
        defaults = dict(
            device_capability=(10, 0),
            cuda_version="13.0",
            flashinfer_gdn_available=True,
        )
        defaults.update(hardware)
        return should_auto_select_flashinfer_gdn_prefill(runner, **defaults)

    def test_selects_supported_sm100_gdn(self):
        self.assertTrue(self.should_select(make_runner()))

    def test_decode_selects_supported_sm100_gdn(self):
        self.assertTrue(
            should_auto_select_flashinfer_gdn_decode(
                make_runner(),
                device_capability=(10, 0),
                flashinfer_gdn_available=True,
            )
        )

    def test_preserves_explicit_prefill_override(self):
        for backend in ("triton", "flashinfer", "cutedsl"):
            with self.subTest(backend=backend):
                self.assertFalse(
                    self.should_select(make_runner(linear_attn_prefill_backend=backend))
                )

    def test_preserves_explicit_decode_override(self):
        for backend in ("triton", "flashinfer", "cutedsl"):
            with self.subTest(backend=backend):
                self.assertFalse(
                    should_auto_select_flashinfer_gdn_decode(
                        make_runner(linear_attn_decode_backend=backend),
                        device_capability=(10, 0),
                        flashinfer_gdn_available=True,
                    )
                )

    def test_preserves_nondefault_base_backend(self):
        for backend in ("flashinfer", "cutedsl"):
            with self.subTest(backend=backend):
                self.assertFalse(
                    self.should_select(make_runner(linear_attn_backend=backend))
                )

    def test_rejects_non_gdn_model(self):
        self.assertFalse(self.should_select(make_runner(hybrid_gdn_config=None)))

    def test_rejects_hopper_and_other_blackwell_generations(self):
        for capability in ((9, 0), (12, 0)):
            with self.subTest(capability=capability):
                self.assertFalse(
                    self.should_select(make_runner(), device_capability=capability)
                )

    def test_rejects_cuda_12(self):
        self.assertFalse(self.should_select(make_runner(), cuda_version="12.9"))

    def test_rejects_non_bf16_state(self):
        for dtype in (None, "float16", "float32"):
            with self.subTest(dtype=dtype):
                self.assertFalse(self.should_select(make_runner(mamba_ssm_dtype=dtype)))

    def test_allocated_state_dtype_takes_precedence(self):
        import torch

        runner = make_runner(mamba_ssm_dtype="bfloat16")
        runner.req_to_token_pool = SimpleNamespace(
            mamba_pool=SimpleNamespace(
                mamba_cache=SimpleNamespace(
                    temporal=torch.empty(0, dtype=torch.float32)
                )
            )
        )
        self.assertFalse(self.should_select(runner))

    def test_rejects_unsupported_head_dimensions(self):
        for key_dim, value_dim in ((64, 128), (128, 64), (64, 64)):
            with self.subTest(key_dim=key_dim, value_dim=value_dim):
                self.assertFalse(
                    self.should_select(
                        make_runner(
                            config_linear_key_head_dim=key_dim,
                            config_linear_value_head_dim=value_dim,
                        )
                    )
                )

    def test_rejects_missing_flashinfer(self):
        self.assertFalse(
            self.should_select(make_runner(), flashinfer_gdn_available=False)
        )

    def test_tree_verification_does_not_disable_prefill(self):
        self.assertTrue(
            self.should_select(
                make_runner(speculative_algorithm="EAGLE", speculative_eagle_topk=2)
            )
        )

    def test_mutation_sets_decode_and_prefill_overrides(self):
        runner = make_runner()
        with (
            patch(
                "sglang.srt.layers.attention.attention_registry.should_auto_select_flashinfer_gdn_decode",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.attention.attention_registry.should_auto_select_flashinfer_gdn_prefill",
                return_value=True,
            ),
        ):
            self.assertTrue(maybe_auto_select_flashinfer_gdn_backends(runner))
        self.assertEqual(runner.server_args.linear_attn_decode_backend, "flashinfer")
        self.assertEqual(runner.server_args.linear_attn_prefill_backend, "flashinfer")

    def test_wrapper_applies_policy_before_dispatcher_construction(self):
        runner = make_runner()
        runner.use_mla_backend = False
        runner.mambaish_config = SimpleNamespace(full_attention_layer_ids=[1])
        runner.is_draft_worker = False
        runner.server_args.attention_backend = "triton"
        full_backend = object()
        linear_backend = object()

        def select_backends(selected_runner):
            selected_runner.server_args.linear_attn_decode_backend = "flashinfer"
            selected_runner.server_args.linear_attn_prefill_backend = "flashinfer"
            return True

        def construct_gdn(selected_runner):
            self.assertIs(selected_runner, runner)
            self.assertTrue(get_linear_attn_decode_backend().is_flashinfer())
            self.assertTrue(get_linear_attn_prefill_backend().is_flashinfer())
            return linear_backend

        with (
            patch(
                "sglang.srt.layers.attention.attention_registry.maybe_auto_select_flashinfer_gdn_backends",
                side_effect=select_backends,
            ) as select,
            patch("sglang.srt.layers.attention.fla.utils.check_environments"),
            patch(
                "sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend",
                side_effect=construct_gdn,
            ),
            patch(
                "sglang.srt.layers.attention.hybrid_linear_attn_backend.HybridLinearAttnBackend",
                side_effect=lambda full, linear, layers: (full, linear, layers),
            ),
            patch("sglang.srt.utils.is_blackwell", return_value=False),
            patch("sglang.srt.utils.is_npu", return_value=False),
        ):
            actual = attn_backend_wrapper(runner, full_backend)

        select.assert_called_once_with(runner)
        self.assertEqual(actual, (full_backend, linear_backend, [1]))

    def test_wrapper_does_not_apply_gdn_policy_to_kda(self):
        runner = make_runner(hybrid_gdn_config=None)
        runner.use_mla_backend = False
        runner.mambaish_config = SimpleNamespace(full_attention_layer_ids=[2])
        runner.mamba2_config = None
        runner.kimi_linear_config = object()
        runner.hybrid_lightning_config = None
        runner.is_draft_worker = False
        runner.server_args.attention_backend = "triton"
        full_backend = object()
        linear_backend = object()

        def construct_kda(selected_runner):
            self.assertIs(selected_runner, runner)
            self.assertTrue(get_linear_attn_decode_backend().is_triton())
            self.assertTrue(get_linear_attn_prefill_backend().is_triton())
            return linear_backend

        with (
            patch(
                "sglang.srt.layers.attention.attention_registry.maybe_auto_select_flashinfer_gdn_backends"
            ) as select,
            patch("sglang.srt.layers.attention.fla.utils.check_environments"),
            patch(
                "sglang.srt.layers.attention.linear.kda_backend.KDAAttnBackend",
                side_effect=construct_kda,
            ),
            patch(
                "sglang.srt.layers.attention.hybrid_linear_attn_backend.HybridLinearAttnBackend",
                side_effect=lambda full, linear, layers: (full, linear, layers),
            ),
            patch("sglang.srt.utils.is_npu", return_value=False),
        ):
            actual = attn_backend_wrapper(runner, full_backend)

        select.assert_not_called()
        self.assertEqual(actual, (full_backend, linear_backend, [2]))


if __name__ == "__main__":
    unittest.main()
