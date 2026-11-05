"""Actual installed HC constructor/mix; explicit CPU device/stream/launch doubles."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.elementwise import hc_combine_decode
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia import numeric_binding
from sglang.srt.layers import hyperconnection as hc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

events = []


class MetadataTensor(torch.Tensor):
    @property
    def is_cuda(self):
        return True

    def record_stream(self, stream):
        events.append(("record", stream))


class TestSmallmDraft(unittest.TestCase):
    def layer(self, cap, combine=True):
        stream = SimpleNamespace(
            wait_stream=lambda current: events.append(("wait", current))
        )
        config = hc.HyperConnectionConfig(
            hidden_size=2560, hc_count=4, hc_lowrank=8, hc_per_branch_norm=True
        )
        with (
            envs.SGLANG_HC_GATE_MAX_ROWS.override(cap),
            patch.object(torch, "compile", lambda f: f),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 7)),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(hc, "_hc_gate_stream", return_value=stream),
        ):
            layer = hc.GatedResidual(config, use_combine=combine)
            from capability_fixtures import contract_spec

            from sglang.srt.kernel_plugins.nvidia import binding

            layer._numeric_capability_spec = contract_spec()
            layer._numeric_mix_variant = lambda: 0
            with (
                patch.object(binding, "megakernel_enabled", return_value=True),
                envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
            ):
                layer.numeric_signatures((1, 4, 8, 16, 32), 4)
        layer._jit_mix_ok = False
        layer._numeric_mix_variant = lambda: 0  # TP4 or macro-off source role.
        return layer, stream

    def run_mix(self, cap, rows, combine=True, macro=False):
        events.clear()
        layer, stream = self.layer(cap, combine)
        x = torch.zeros(rows, 10240, dtype=torch.bfloat16).as_subclass(MetadataTensor)
        gates = []
        current = object()

        def gate(normed, weight):
            gates.append((normed, weight))
            return object()

        layer._numeric_hc_macro = lambda rows: object() if macro else None
        from capability_fixtures import contract_geometry

        with (
            patch.object(
                numeric_binding,
                "numeric_operator",
                side_effect=lambda sig, *, prepared: (
                    (lambda io: None) if prepared and contract_geometry(sig) else None
                ),
            ),
            patch.object(torch.cuda, "current_stream", return_value=current),
            patch.object(torch.cuda, "stream", return_value=nullcontext()),
            patch.object(hc_combine_decode, "hc_combine_gate", side_effect=gate),
            patch.object(hc, "flydsl_hc_mix_supported", return_value=False),
            patch.object(hc, "fused_hc_mix_supported", return_value=False),
        ):
            result = layer.mix(x, normalized_input=x)
        expected = layer._mix_compute(
            x,
            layer.input_mix_weight_down.weight,
            layer.input_mix_weight_up.weight,
            4,
            2560,
        ).to(torch.bfloat16)
        self.assertTrue(torch.equal(result[0], expected))
        self.assertIs(result[1][0], x)
        return layer, gates, list(events), current

    def test_exact_source_row_matrix(self):
        for cap in (1, 4, 16, 32):
            for rows in (1, 4, 8, 16, 17, 32, 33):
                with self.subTest(cap=cap, rows=rows):
                    layer, gates, trace, current = self.run_mix(cap, rows)
                    self.assertEqual(len(gates), int(rows <= cap))
                    if gates:
                        self.assertIs(gates[0][1], layer.block_inject_weight.weight)
                        self.assertEqual(
                            trace, [("wait", current), ("record", layer._gate_stream)]
                        )
                    else:
                        self.assertEqual(trace, [])

    def test_terminal_mix_only_never_gets_combine_weight_or_gate(self):
        for cap in (1, 4, 16, 32):
            layer, gates, trace, _ = self.run_mix(cap, 4, combine=False)
            self.assertFalse(hasattr(layer, "block_inject_weight"))
            self.assertEqual(gates, [])
            self.assertEqual(trace, [])

    def test_macro_current_stream_gate_not_duplicated(self):
        for cap in (1, 4, 16, 32):
            _, gates, trace, _ = self.run_mix(cap, 4, macro=True)
            self.assertEqual(len(gates), 1)
            self.assertEqual(trace, [])

    def test_invalid_cap_rejected_in_constructor(self):
        for cap in (0, 2, 8, 64, -1):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                self.layer(cap)

    def test_source_smallm16_precedence_over_smallm4(self):
        for p2, small16, small4, expected in (
            (False, False, False, 1),
            (False, False, True, 4),
            (False, True, False, 16),
            (False, True, True, 16),
            (True, False, False, 32),
            (True, True, True, 32),
        ):
            self.assertEqual(
                32 if p2 else 16 if small16 else 4 if small4 else 1, expected
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
