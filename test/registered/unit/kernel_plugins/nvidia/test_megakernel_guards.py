"""CPU checks of --enable-nvidia-megakernel's startup resolution."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.kernels.plugin_kernels.nvidia import spec as registry
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia import binding
from sglang.srt.kernel_plugins.nvidia.binding import (
    megakernel_disable_reasons,
)
from sglang.srt.layers.attention.q35_cp_lite import cp_lite_requested
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
        with envs.SGLANG_NVIDIA_MEGAKERNEL_SLOTS.override(()):
            self.assertFalse(cp_lite_requested())
        with envs.SGLANG_NVIDIA_MEGAKERNEL_SLOTS.override(("mka",)):
            self.assertFalse(cp_lite_requested())
        for slot in ("skv2_cp", "draft_q1_cp", "draft_skv2_cp"):
            with envs.SGLANG_NVIDIA_MEGAKERNEL_SLOTS.override((slot,)):
                self.assertTrue(cp_lite_requested())


if __name__ == "__main__":
    unittest.main()
