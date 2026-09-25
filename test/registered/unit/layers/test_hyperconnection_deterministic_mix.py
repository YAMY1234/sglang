"""Deterministic inference runs the eager hyper-connection mix / combine bodies:
the torch.compile'd ones are not row-count invariant (#287 root cause 2)."""
import types
import unittest
from unittest import mock

import torch

from sglang.srt.layers import hyperconnection
from sglang.srt.layers.hyperconnection import GatedResidual


def _fake(calls):
    def tagged(tag):
        def fn(*args):
            calls.append(tag)
            return torch.zeros(args[0].shape[0], 4)

        return fn

    weight = types.SimpleNamespace(weight=torch.zeros(2, 8))
    return types.SimpleNamespace(
        hc_count=2, hidden_size=4, params_dtype=torch.float32,
        config=types.SimpleNamespace(hc_per_branch_norm=True),
        hc_norm=lambda x: x, _jit_mix_ok=False, _jit_combine_ok=False,
        input_mix_weight_down=weight, input_mix_weight_up=weight, block_inject_weight=weight,
        _mix_compute=tagged("compiled"), _mix_compute_eager=tagged("eager"),
        _combine_compute=tagged("compiled"), _combine_compute_eager=tagged("eager"),
    )


class DeterministicMixTest(unittest.TestCase):
    def _run(self, deterministic):
        calls = []
        fake = _fake(calls)
        x = torch.randn(3, 8)
        with mock.patch.object(hyperconnection, "_deterministic_inference", return_value=deterministic):
            _, residuals = GatedResidual.mix(fake, x)
            GatedResidual.combine(fake, torch.randn(3, 4), residuals)
        return calls

    def test_deterministic_uses_eager_bodies(self):
        self.assertEqual(self._run(True), ["eager", "eager"])

    def test_default_keeps_compiled_bodies(self):
        self.assertEqual(self._run(False), ["compiled", "compiled"])


if __name__ == "__main__":
    unittest.main()
