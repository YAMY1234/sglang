"""Actual HC constructor/mix; CPU tensors with CUDA-metadata and launch doubles.

This reproduces a Python ownership/admission failure, not CUDA arithmetic.
The installed production GatedResidual constructor and complete mix body run.
CUDA/compiler/provider boundaries are CPU doubles; no historical source lookup.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.elementwise import hc_combine_decode
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia import numeric_binding
from sglang.srt.layers import hyperconnection as hc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class CudaMetadataTensor(torch.Tensor):
    @property
    def is_cuda(self):
        return True


class TestActualHcAdmission(unittest.TestCase):
    def layer(self, module, combine=False, tp=1):
        config = module.HyperConnectionConfig(
            hidden_size=2560,
            hc_count=4,
            hc_lowrank=8,
            hc_per_branch_norm=True,
            params_dtype=torch.bfloat16,
        )
        # Actual constructor/real nn.Linear registration. Only compiler/device
        # boundary is a CPU double. No fake block_inject_weight is added.
        with (
            patch.object(torch, "compile", lambda f: f),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            layer = module.GatedResidual(config, use_combine=combine)
        layer._numeric_mix_variant = lambda: int(tp == 1 and envs.SGLANG_HC_MACRO.get())
        return layer

    def execute_mix(self, module, layer, rows=4, offer=True, macro=True):
        x = torch.randn(rows, 10240, dtype=torch.bfloat16).as_subclass(
            CudaMetadataTensor
        )
        gates, selections = [], []

        def choose(signature, *, prepared):
            from capability_fixtures import contract_geometry

            prepared = prepared and contract_geometry(signature)
            selections.append((signature, prepared))
            return (lambda io: None) if offer and prepared else None

        def gate(normed, weight):
            gates.append((normed, weight))
            return torch.empty((rows, 8, 4), dtype=torch.float32)

        with (
            patch.object(numeric_binding, "numeric_operator", side_effect=choose),
            patch.object(hc_combine_decode, "hc_combine_gate", side_effect=gate),
            patch.object(module, "flydsl_hc_mix_supported", return_value=False),
            patch.object(module, "fused_hc_mix_supported", return_value=False),
            envs.SGLANG_HC_MACRO.override(macro),
            envs.SGLANG_HC_MACRO_MIN_ROWS.override(1),
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
        self.assertIs(result[1][1], x)
        return result, gates, selections

    def test_fixed_terminal_no_checkpoint_parameter_no_gate_stock_mix_unchanged(self):
        layer = self.layer(hc)
        result, gates, selections = self.execute_mix(hc, layer)
        self.assertFalse(hasattr(layer, "block_inject_weight"))
        self.assertFalse(any("block_inject_weight" in k for k in layer.state_dict()))
        self.assertEqual(gates, [])
        self.assertTrue(all(not prepared for _, prepared in selections))
        self.assertIsNone(result[1][2])

    def test_combine_tp1_keeps_selected_gate_original_weight_identity(self):
        layer = self.layer(hc, combine=True)
        result, gates, selections = self.execute_mix(hc, layer)
        self.assertEqual(len(gates), 1)
        self.assertIs(gates[0][1], layer.block_inject_weight.weight)
        self.assertTrue(selections[0][1])
        self.assertEqual(tuple(result[1][2].shape), (4, 8, 4))

    def test_fixed_tp4_macro_rejected_regular_mix_retained(self):
        layer = self.layer(hc, combine=True, tp=4)
        result, gates, selections = self.execute_mix(hc, layer)
        self.assertEqual(gates, [])
        self.assertTrue(all(not p for _, p in selections))
        self.assertIsNone(result[1][2])

    def test_macro_off_stays_native_for_both_roles(self):
        for combine in (False, True):
            with self.subTest(combine=combine):
                _, gates, selections = self.execute_mix(
                    hc, self.layer(hc, combine), macro=False
                )
                self.assertEqual(gates, [])
                self.assertTrue(all(not p for _, p in selections))

    def test_no_offer_stays_native_for_both_roles(self):
        for combine in (False, True):
            with self.subTest(combine=combine):
                _, gates, _ = self.execute_mix(hc, self.layer(hc, combine), offer=False)
                self.assertEqual(gates, [])

    def test_real_terminal_warmup_row_matrix(self):
        for rows in (1, 4, 8, 32, 33):
            with self.subTest(rows=rows):
                result, gates, _ = self.execute_mix(hc, self.layer(hc), rows=rows)
                self.assertEqual(tuple(result[0].shape), (rows, 2560))
                self.assertEqual(gates, [])

    def test_row_bounds_kept_with_combine_weight(self):
        for rows, expected in ((1, True), (32, True), (33, False)):
            with self.subTest(rows=rows):
                _, gates, selections = self.execute_mix(
                    hc, self.layer(hc, True), rows=rows
                )
                self.assertEqual(bool(gates), expected)
                self.assertEqual(bool(selections[0][1]), expected)

    def test_actual_variant_policy_matches_source_tp_and_macro(self):
        from types import SimpleNamespace

        from sglang.srt import runtime_context

        for tp in (1, 4):
            for enabled in (False, True):
                with (
                    self.subTest(tp=tp, enabled=enabled),
                    envs.SGLANG_HC_MACRO.override(enabled),
                    patch.object(
                        runtime_context,
                        "get_parallel",
                        return_value=SimpleNamespace(tp_size=tp),
                    ),
                ):
                    self.assertEqual(
                        hc.GatedResidual._numeric_mix_variant(),
                        int(tp == 1 and enabled),
                    )

    def test_startup_signatures_use_the_same_tuned_variant_policy(self):
        from types import SimpleNamespace

        from sglang.kernels.plugin_kernels.nvidia.numeric import NumericOperation
        from sglang.srt import runtime_context

        layer = self.layer(hc, combine=True)
        layer._numeric_mix_variant = hc.GatedResidual._numeric_mix_variant
        for tp in (1, 4):
            for enabled in (False, True):
                with (
                    self.subTest(tp=tp, enabled=enabled),
                    envs.SGLANG_HC_MACRO.override(enabled),
                    patch.object(
                        runtime_context,
                        "get_parallel",
                        return_value=SimpleNamespace(tp_size=tp),
                    ),
                ):
                    from capability_fixtures import contract_spec

                    from sglang.srt.kernel_plugins.nvidia import binding

                    layer._numeric_capability_spec = contract_spec()
                    with (
                        patch.object(binding, "megakernel_enabled", return_value=True),
                        envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
                    ):
                        signatures = layer.numeric_signatures((4, 32), 4)
                    mix = [
                        s for s in signatures if s.operation == NumericOperation.HC_MIX
                    ]
                    self.assertEqual([s.rows for s in mix], [4, 32])
                    self.assertTrue(
                        all(s.variant == layer._numeric_mix_variant() for s in mix)
                    )

    def test_source_splitk_row_cap_and_variant(self):
        from sglang.kernels.ops.gemm import hc_mix_splitk

        for tp, macro, rows, expected in (
            (1, True, 32, True),
            (1, False, 32, False),
            (4, True, 32, False),
            (4, True, 24, True),
        ):
            with self.subTest(tp=tp, macro=macro, rows=rows):
                layer = self.layer(hc, tp=tp)
                layer._jit_mix_ok = True
                layer._mix_up_weight_padded = torch.empty(
                    (10240, 128), dtype=torch.bfloat16
                )
                calls = []

                def mix(normed, down, up, hc, hs, *, numeric_variant):
                    calls.append(numeric_variant)
                    return layer._mix_compute(
                        normed, down, layer.input_mix_weight_up.weight, hc, hs
                    )

                with patch.object(hc_mix_splitk, "hc_mix", side_effect=mix):
                    self.execute_mix(hc, layer, rows=rows, macro=macro)
                self.assertEqual(bool(calls), expected)
                if calls:
                    self.assertEqual(calls, [int(tp == 1 and macro)])

    def test_wide_mix_cap_follows_the_master_and_the_kernel_limit(self):
        for master, cap, expected in (
            (False, 512, 0),
            (True, 0, 0),
            (True, 320, 320),
            (True, 4096, 512),
        ):
            with (
                self.subTest(master=master, cap=cap),
                envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(master),
                envs.SGLANG_HC_MIX_SPLITK_MAX_ROWS.override(cap),
            ):
                self.assertEqual(hc._hc_mix_wide_max_rows(), expected)

    def test_wide_mix_serves_tp1_rows_above_32_up_to_the_cap(self):
        from sglang.kernels.ops.gemm import hc_mix_splitk

        for tp, macro, cap, rows, expected in (
            (1, True, 320, 33, True),
            (1, True, 320, 320, True),
            (1, True, 320, 321, False),
            (1, True, 320, 32, False),
            (1, False, 320, 64, False),
            (4, True, 320, 64, False),
            (1, True, 0, 64, False),
        ):
            with self.subTest(tp=tp, macro=macro, cap=cap, rows=rows):
                layer = self.layer(hc, tp=tp)
                layer._jit_mix_ok = True
                layer._mix_wide_max_rows = cap
                calls = []

                def wide(normed, down, up, hc_count, hs):
                    calls.append((normed.shape[0], tuple(up.shape), hc_count, hs))
                    return layer._mix_compute(
                        normed, down, layer.input_mix_weight_up.weight, hc_count, hs
                    )

                def narrow(normed, down, up, hc_count, hs, *, numeric_variant):
                    return layer._mix_compute(
                        normed, down, layer.input_mix_weight_up.weight, hc_count, hs
                    )

                with (
                    patch.object(hc_mix_splitk, "hc_mix_wide", side_effect=wide),
                    patch.object(hc_mix_splitk, "hc_mix", side_effect=narrow),
                ):
                    # The provider's HC_MIX serves at most 32 rows; none is offered here.
                    self.execute_mix(hc, layer, rows=rows, offer=False, macro=macro)
                self.assertEqual(bool(calls), expected)
                if calls:
                    self.assertEqual(calls, [(rows, (10240, 128), 4, 2560)])

    def test_current_stream_partials_do_not_wait_none_and_side_stream_still_joins(self):
        from types import SimpleNamespace

        for name in ("combine", "combine_and_normalize"):
            for macro in (False, True):
                for side_stream in (None, object()):
                    with self.subTest(
                        method=name, macro=macro, side=side_stream is not None
                    ):
                        events = []
                        current = SimpleNamespace(
                            wait_stream=lambda stream: events.append(("wait", stream))
                        )
                        partials = SimpleNamespace(
                            record_stream=lambda stream: events.append(
                                ("record", stream)
                            )
                        )
                        block = torch.empty((4, 2560), dtype=torch.bfloat16)
                        residual = torch.empty((4, 10240), dtype=torch.bfloat16)
                        norm = SimpleNamespace(
                            group_size=2560,
                            weight=torch.empty((10240,), dtype=torch.bfloat16),
                            variance_epsilon=1e-6,
                        )
                        layer = SimpleNamespace(
                            _gate_stream=side_stream,
                            hidden_size=2560,
                            _numeric_hc_macro=lambda rows: object() if macro else None,
                        )
                        args = (block, (residual, residual, partials)) + (
                            (norm,) if name == "combine_and_normalize" else ()
                        )
                        with (
                            patch.object(
                                torch.cuda, "current_stream", return_value=current
                            ),
                            patch.object(
                                hc_combine_decode,
                                "hc_combine_apply",
                                return_value=residual,
                            ),
                            patch.object(
                                hc_combine_decode,
                                "hc_combine_apply_norm",
                                return_value=(residual, residual),
                            ),
                        ):
                            getattr(hc.GatedResidual, name)(layer, *args)
                        expected = (
                            [("wait", side_stream), ("record", current)]
                            if side_stream is not None and not macro
                            else []
                        )
                        self.assertEqual(events, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
