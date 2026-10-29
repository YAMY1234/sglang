"""CPU checks of --enable-nvidia-megakernel's startup resolution."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.linear import (
    gdn_replayssm_commit_cuda,
    gdn_replayssm_spec_decode,
)
from sglang.kernels.plugin_kernels.nvidia import spec as registry
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia import binding
from sglang.srt.kernel_plugins.nvidia.binding import (
    megakernel_disable_reasons,
)
from sglang.srt.speculative import spec_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fakes import ARCH, ISA, server_args  # noqa: E402

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestMegakernelGuards(CustomTestCase):
    def setUp(self):
        registry_patch = patch.dict(registry._REGISTRY, clear=True)
        registry_patch.start()
        self.addCleanup(registry_patch.stop)

    def test_a_disabled_flag_looks_nothing_up_and_keeps_the_forward(self):
        with patch.object(binding, "plugin_reasons") as plugin:
            binding.handle_nvidia_megakernel(server_args())
        plugin.assert_not_called()
        model = SimpleNamespace()  # any attribute read would raise
        with patch.object(binding, "megakernel_enabled", return_value=False):
            binding.install_megakernel(
                model=model, hf_config=None, capture_tail_hooks=[], is_draft=False
            )
        self.assertEqual(vars(model), {})

    def test_an_unmet_tp_size_is_refused_and_an_empty_field_accepts_any(self):
        hf_config = SimpleNamespace(architectures=[ARCH], model_type="fake")

        def tp_reasons(path):
            registry._REGISTRY.clear()
            registry.register(architecture=ARCH, path=path)
            _, reasons = megakernel_disable_reasons(
                cfg=server_args(tp_size=8), hf_config=hf_config, isa=ISA
            )
            return [r for r in reasons if "tp_size" in r]

        self.assertTrue(tp_reasons("_fakes:SPEC"))
        self.assertEqual(tp_reasons("_fakes:ANY_TP"), [])

    def test_gdn_cuda_commit_uses_only_the_runtime_master_switch(self):
        with patch.object(
            gdn_replayssm_commit_cuda, "try_commit", return_value=True
        ) as commit:
            with (
                envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(False),
                patch.object(binding, "megakernel_enabled", return_value=True),
            ):
                self.assertFalse(
                    gdn_replayssm_spec_decode._try_runtime_cuda_commit(probe=True)
                )
            commit.assert_not_called()

            # Plugin presence and the old raw slot are deliberately irrelevant.
            with (
                envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
                patch.object(binding, "megakernel_enabled", return_value=False),
            ):
                self.assertTrue(
                    gdn_replayssm_spec_decode._try_runtime_cuda_commit(probe=True)
                )
            commit.assert_called_once_with(probe=True)

    def test_gdn_cuda_commit_unsupported_isa_and_layout_fall_back(self):
        gdn_replayssm_commit_cuda._device_supported.cache_clear()
        self.addCleanup(gdn_replayssm_commit_cuda._device_supported.cache_clear)
        with patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)):
            self.assertFalse(gdn_replayssm_commit_cuda._device_supported(0))

        with (
            patch.object(gdn_replayssm_commit_cuda, "_supported", return_value=False),
            patch.object(gdn_replayssm_commit_cuda, "_commit") as commit,
        ):
            self.assertFalse(gdn_replayssm_commit_cuda.try_commit(probe=True))
        commit.assert_not_called()

    def _isolate_gdn_cuda_commit_key(self, key):
        for registry in (
            gdn_replayssm_commit_cuda._LOADED,
            gdn_replayssm_commit_cuda._BUILD_FAILURES,
        ):
            registry.discard(key)
            self.addCleanup(registry.discard, key)

    @staticmethod
    def _gdn_pool_views():
        return dict(
            checkpoint_state=SimpleNamespace(shape=(2, 4, 9, 128, 128)),
            d_cache=None,
            k_cache=SimpleNamespace(shape=(2, 4, 3, 16, 128)),
            g_cache=None,
            d_residual_cache=None,
            k_residual_cache=None,
        )

    def test_gdn_cuda_commit_build_failure_is_cached_stock_fallback(self):
        key = (3, 9)
        self._isolate_gdn_cuda_commit_key(key)
        inputs = dict(
            checkpoint_state=SimpleNamespace(shape=(2, 4, 9, 128, 128)),
            k_cache=SimpleNamespace(shape=(2, 4, 3, 16, 128)),
        )
        with (
            patch.object(gdn_replayssm_commit_cuda, "_supported", return_value=True),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(
                gdn_replayssm_commit_cuda,
                "_module",
                side_effect=RuntimeError("nvcc unavailable"),
            ) as build,
            patch.object(gdn_replayssm_commit_cuda, "_commit") as commit,
        ):
            self.assertFalse(gdn_replayssm_commit_cuda.try_commit(**inputs))
            self.assertFalse(gdn_replayssm_commit_cuda.try_commit(**inputs))

        build.assert_called_once_with(*key)
        commit.assert_not_called()

    def test_gdn_cuda_commit_prebuild_builds_both_tracks_once_before_capture(self):
        key = (3, 9)
        self._isolate_gdn_cuda_commit_key(key)
        views = self._gdn_pool_views()
        with (
            patch.object(
                gdn_replayssm_commit_cuda,
                "_static_layout_supported",
                return_value=True,
            ),
            patch.object(gdn_replayssm_commit_cuda, "_module") as build,
        ):
            self.assertTrue(gdn_replayssm_commit_cuda.prebuild(**views))
            self.assertTrue(gdn_replayssm_commit_cuda.prebuild(**views))
        build.assert_called_once_with(*key)

        # A prebuilt module serves rounds under capture, and nothing compiles there.
        with (
            patch.object(gdn_replayssm_commit_cuda, "_supported", return_value=True),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
            patch.object(gdn_replayssm_commit_cuda, "_module") as build,
            patch.object(gdn_replayssm_commit_cuda, "_commit") as commit,
        ):
            self.assertTrue(
                gdn_replayssm_commit_cuda.try_commit(
                    checkpoint_state=views["checkpoint_state"],
                    k_cache=views["k_cache"],
                )
            )
        build.assert_not_called()
        commit.assert_called_once()

    def test_gdn_cuda_commit_prebuild_rejects_layout_and_remembers_build_failure(self):
        key = (3, 9)
        self._isolate_gdn_cuda_commit_key(key)
        views = self._gdn_pool_views()
        with (
            patch.object(
                gdn_replayssm_commit_cuda,
                "_static_layout_supported",
                return_value=False,
            ),
            patch.object(gdn_replayssm_commit_cuda, "_module") as build,
        ):
            self.assertFalse(gdn_replayssm_commit_cuda.prebuild(**views))
        build.assert_not_called()

        with (
            patch.object(
                gdn_replayssm_commit_cuda,
                "_static_layout_supported",
                return_value=True,
            ),
            patch.object(
                gdn_replayssm_commit_cuda,
                "_module",
                side_effect=RuntimeError("nvcc unavailable"),
            ) as build,
        ):
            self.assertFalse(gdn_replayssm_commit_cuda.prebuild(**views))
            self.assertFalse(gdn_replayssm_commit_cuda.prebuild(**views))
        build.assert_called_once_with(*key)

        with (
            patch.object(gdn_replayssm_commit_cuda, "_supported", return_value=True),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(gdn_replayssm_commit_cuda, "_module") as build,
            patch.object(gdn_replayssm_commit_cuda, "_commit") as commit,
        ):
            self.assertFalse(
                gdn_replayssm_commit_cuda.try_commit(
                    checkpoint_state=views["checkpoint_state"],
                    k_cache=views["k_cache"],
                )
            )
        build.assert_not_called()
        commit.assert_not_called()

    def test_gdn_cuda_commit_prebuild_uses_only_the_runtime_master_switch(self):
        with patch.object(
            gdn_replayssm_commit_cuda, "prebuild", return_value=True
        ) as prebuild:
            with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(False):
                self.assertFalse(
                    gdn_replayssm_spec_decode.prebuild_gdn_replayssm_circular_commit(
                        probe=True
                    )
                )
            prebuild.assert_not_called()
            with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True):
                self.assertTrue(
                    gdn_replayssm_spec_decode.prebuild_gdn_replayssm_circular_commit(
                        probe=True
                    )
                )
            prebuild.assert_called_once_with(probe=True)

    def test_verify_commit_prebuild_targets_only_the_circular_commit_pool(self):
        spec_state = SimpleNamespace(
            temporal="temporal",
            replayssm_d="d",
            replayssm_k="k",
            replayssm_g="g",
            replayssm_rawv="rawv",
            replayssm_rawk="rawk",
        )

        def runner(**pool):
            pool = SimpleNamespace(
                replayssm_cache_base=pool.get("cache_base", "base"),
                replayssm_is_kda=pool.get("is_kda", False),
                get_speculative_mamba2_params_all_layers=lambda: spec_state,
            )
            router = SimpleNamespace(part_view=lambda part: pool)
            return SimpleNamespace(
                model_config=SimpleNamespace(),
                token_to_kv_pool=SimpleNamespace(layer_router=router),
            )

        with (
            patch.object(spec_utils, "mambaish_config", return_value=object()),
            patch.object(
                gdn_replayssm_spec_decode,
                "prebuild_gdn_replayssm_circular_commit",
                return_value=True,
            ) as prebuild,
        ):
            # Fold-every-commit pools have no circular cursors; KDA never commits here.
            spec_utils.prebuild_mamba_commit_after_verify(runner(cache_base=None))
            spec_utils.prebuild_mamba_commit_after_verify(runner(is_kda=True))
            prebuild.assert_not_called()
            spec_utils.prebuild_mamba_commit_after_verify(runner())
        prebuild.assert_called_once_with(
            checkpoint_state="temporal",
            d_cache="d",
            k_cache="k",
            g_cache="g",
            d_residual_cache="rawv",
            k_residual_cache="rawk",
        )


if __name__ == "__main__":
    unittest.main()
