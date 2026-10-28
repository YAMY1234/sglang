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
from sglang.srt.kernel_plugins.nvidia.policy import megakernel_slot_requested
from sglang.srt.layers.attention.q35_cp_lite import (
    CpLite,
    cp_lite_requested,
    maybe_prebuild_cp_lite,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fakes import ARCH, ISA, server_args

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

    def test_cp_lite_requires_a_cp_slot(self):
        with patch.object(binding, "megakernel_enabled", return_value=True):
            with envs.SGLANG_NVIDIA_MEGAKERNEL_SLOTS.override(""):
                self.assertFalse(cp_lite_requested())
            with envs.SGLANG_NVIDIA_MEGAKERNEL_SLOTS.override("mka"):
                self.assertFalse(cp_lite_requested())
            for slot in ("skv2_cp", "draft_q1_cp", "draft_skv2_cp"):
                with envs.SGLANG_NVIDIA_MEGAKERNEL_SLOTS.override(slot):
                    self.assertTrue(cp_lite_requested())

    def test_cp_lite_accepts_the_startup_flag_before_config_is_published(self):
        with (
            patch.object(binding, "megakernel_enabled", return_value=False),
            envs.SGLANG_NVIDIA_MEGAKERNEL_SLOTS.override("skv2_cp"),
        ):
            self.assertFalse(cp_lite_requested())
            self.assertFalse(cp_lite_requested(enabled=False))
            self.assertTrue(cp_lite_requested(enabled=True))

    def test_cp_lite_is_built_before_the_plugin_validates_its_variant(self):
        """A requested CP graph variant must exist before attention backends build."""
        config = object()
        built = object()
        with (
            envs.SGLANG_NVIDIA_MEGAKERNEL_SLOTS.override("skv2_cp"),
            patch(
                "sglang.srt.runtime_context.get_platform",
                return_value=SimpleNamespace(is_sm90=False, is_sm120=False),
            ),
            patch.object(CpLite, "maybe_create", return_value=built) as create,
        ):
            got = maybe_prebuild_cp_lite(
                enabled=True,
                config=config,
                device="cuda",
                kv_cache_dtype=torch.float8_e4m3fn,
                page_size=64,
            )
        self.assertIs(got, built)
        create.assert_called_once_with(config=config, device="cuda")

    def test_cp_lite_request_does_not_restore_a_length_gate(self):
        """A requested multi-request CP path must remain active at short KV lengths."""
        import torch.distributed._symmetric_memory as symm_mem

        original_empty = torch.empty
        handle = SimpleNamespace(rank=0, buffer_ptrs=[0, 0])

        def empty_cpu(shape, *, dtype, device):
            del device
            return original_empty(shape, dtype=dtype)

        with (
            patch.object(torch.cuda, "get_device_properties") as properties,
            patch.object(torch.cuda, "synchronize"),
            patch.object(torch.cuda, "Stream", return_value=object()),
            patch.object(torch.distributed, "barrier"),
            patch.object(symm_mem, "get_backend", return_value=object()),
            patch.object(symm_mem, "empty", side_effect=empty_cpu),
            patch.object(symm_mem, "rendezvous", return_value=handle),
            patch("sglang.srt.layers.attention.q35_cp_fused.q35_cp_fused_prepare"),
        ):
            properties.return_value = SimpleNamespace(multi_processor_count=148)
            cp_lite = CpLite(
                group=object(),
                tp_rank=0,
                tp_size=8,
                cp=2,
                q_heads=8,
                device="cpu",
            )

        self.assertIsNone(cp_lite._gate_thr(2))
        self.assertIsNone(cp_lite._gate_thr(4))
        self.assertIsNone(cp_lite._gate_thr(16))

    def test_cp_lite_gate_epilogue_offer_is_consumed_once(self):
        cp_lite = CpLite.__new__(CpLite)
        cp_lite._gate_epilogue = None
        cp_lite._gate_epilogue_done = False
        gate, scale, out = object(), object(), object()

        cp_lite.offer_gate_epilogue(gate, scale, out)
        self.assertEqual(cp_lite._gate_epilogue, (gate, scale, out))
        self.assertFalse(cp_lite.take_gate_epilogue())
        self.assertIsNone(cp_lite._gate_epilogue)

        cp_lite.offer_gate_epilogue(gate, scale, out)
        cp_lite._gate_epilogue_done = True
        self.assertTrue(cp_lite.take_gate_epilogue())
        self.assertFalse(cp_lite.take_gate_epilogue())

    def test_slots_are_inert_without_the_megakernel(self):
        with (
            patch.object(binding, "megakernel_enabled", return_value=False),
            envs.SGLANG_NVIDIA_MEGAKERNEL_SLOTS.override("draft_fin"),
        ):
            self.assertFalse(megakernel_slot_requested("draft_fin"))

    def test_gdn_cuda_commit_uses_only_the_runtime_master_switch(self):
        with patch.object(
            gdn_replayssm_commit_cuda, "try_commit", return_value=True
        ) as commit:
            with (
                envs.ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(False),
                patch.object(binding, "megakernel_enabled", return_value=True),
                envs.SGLANG_NVIDIA_MEGAKERNEL_SLOTS.override("unrelated_slot"),
            ):
                self.assertFalse(
                    gdn_replayssm_spec_decode._try_runtime_cuda_commit(probe=True)
                )
            commit.assert_not_called()

            # Plugin presence and the old raw slot are deliberately irrelevant.
            with (
                envs.ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
                patch.object(binding, "megakernel_enabled", return_value=False),
                envs.SGLANG_NVIDIA_MEGAKERNEL_SLOTS.override(""),
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

    def test_gdn_cuda_commit_prebuild_failure_is_cached_stock_fallback(self):
        key = (77, 3, 9, 16, False)
        gdn_replayssm_commit_cuda._COMPILE_FAILURES.discard(key)
        self.addCleanup(gdn_replayssm_commit_cuda._COMPILE_FAILURES.discard, key)
        checkpoint = SimpleNamespace(
            device=SimpleNamespace(index=77), shape=(2, 4, 9, 128, 128)
        )
        d_cache = SimpleNamespace(shape=(2, 4, 9, 16, 128))
        k_cache = SimpleNamespace(shape=(2, 4, 3, 16, 128))
        inputs = dict(
            checkpoint_state=checkpoint,
            d_cache=d_cache,
            k_cache=k_cache,
        )
        with (
            patch.object(gdn_replayssm_commit_cuda, "_supported", return_value=True),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(
                gdn_replayssm_commit_cuda,
                "_kernel",
                side_effect=RuntimeError("NVRTC unavailable"),
            ) as prebuild,
            patch.object(gdn_replayssm_commit_cuda, "_commit") as commit,
        ):
            self.assertFalse(gdn_replayssm_commit_cuda.try_commit(**inputs))
            self.assertFalse(gdn_replayssm_commit_cuda.try_commit(**inputs))

        prebuild.assert_called_once_with(*key)
        commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
