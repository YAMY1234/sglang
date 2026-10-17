"""SGLANG_NV_MEGA_INTREE_REF: off never reaches the plugin; on, SGLang's own layer
forward runs the plugin slots at their routed widths with run_layer's arguments."""

import sys
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import deepseek_v4
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.models import deepseek_v4_nv_mega_ref as ref_mod
from sglang.srt.monokernels.spec import ShapeKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TOKENS = 6
# rows -> the plugin plan's slots: hop A at <= 8 rows, fast gate at <= 8 and 48 rows
PLAN = {
    6: SimpleNamespace(hop_a=True, fast_gate=True, routed=True),
    12: SimpleNamespace(hop_a=False, fast_gate=False, routed=True),
    48: SimpleNamespace(hop_a=False, fast_gate=True, routed=True),
}
SERVED = frozenset(
    ShapeKey(forward_mode="target_verify", batch_size=w // TOKENS, tokens_per_request=6)
    for w in PLAN
)


def _batch(rows, mode=ForwardMode.TARGET_VERIFY):
    return SimpleNamespace(
        forward_mode=mode,
        batch_size=rows // TOKENS,
        input_ids=torch.zeros(rows, dtype=torch.int64),
        moe_num_token_non_padded=lambda: "num_valid",
    )


class _Layer:
    """The attributes DeepseekV4CudaDecoderLayer.forward reads, as mocks."""

    def __init__(self, ref, *, do_shared_local=False):
        self.log = Mock()
        self._nv_mega_ref = ref
        self.attn_boundary = self.log.attn_boundary
        self.attn_boundary.prepare.return_value = "attn_in"
        self.attn_boundary.finish.side_effect = lambda out, fb: ("attn_done", out)
        self.self_attn = self.log.self_attn
        self.self_attn.maybe_use_decode_attn_tp.return_value = nullcontext()
        self.self_attn.forward_before_wo_b.return_value = "o"
        self.self_attn._project_wo_b.return_value = "projected"
        self._run_boundary_attention = self.log.run_boundary_attention
        self._run_boundary_attention.return_value = "sglang_attn"
        self.ffn_boundary = self.log.ffn_boundary
        self.ffn_boundary.prepare.return_value = "ffn_in"
        self.transport = SimpleNamespace(
            input_ids="ids",
            input_ids_global="ids_global",
            do_shared_local=do_shared_local,
            return_moe_output=True,
        )
        self.ffn_boundary.transport_context.return_value = self.transport
        self.output = self.log.output
        self.output.finish.side_effect = lambda h: ("layer_out", h)
        self.ffn_boundary.exit.return_value = nullcontext(self.output)
        self.mlp = self.log.mlp
        self.mlp.gate.e_score_correction_bias_vl = None
        self.mlp.return_value = "sglang_moe"
        self.mlp.forward_normal_dual_stream.return_value = "slot_moe"

    def forward(self, batch):
        with (
            patch.object(
                deepseek_v4.residual_batch,
                "written_residual",
                return_value=SimpleNamespace(quantized="xq"),
            ),
            patch.object(deepseek_v4.hc_boundary, "LayerCall", return_value="call"),
        ):
            return deepseek_v4.DeepseekV4CudaDecoderLayer.forward(
                self, "pos", "hidden", "ids", batch, "ids_global"
            )


def _ref(runner, moe, *, is_draft=False):
    return ref_mod.LayerRef(
        steps=ref_mod._step_refs(
            served=SERVED, slots=PLAN, runner=runner, moe=moe, is_draft=is_draft
        )
    )


def _slots():
    runner = Mock()
    runner.publish_wo_b.return_value = "partial"
    moe = Mock(gate=True)
    return runner, moe


class TestSwitchOff(unittest.TestCase):
    def _precompile(self):
        model = SimpleNamespace(modules=lambda: iter(()))
        platform = SimpleNamespace(is_blackwell=False)
        with (
            patch.object(deepseek_v4, "get_platform", return_value=platform),
            patch.object(ref_mod, "install") as install,
        ):
            deepseek_v4.DeepseekV4ForCausalLM.precompile_kernels_after_loading(model)
        return model, install

    def test_off_never_installs_or_imports_the_plugin(self):
        self.assertFalse(envs.SGLANG_NV_MEGA_INTREE_REF.get())
        imported = "sgl_nvidia_mega" in sys.modules
        _, install = self._precompile()
        install.assert_not_called()
        self.assertEqual("sgl_nvidia_mega" in sys.modules, imported)
        self.assertIsNone(deepseek_v4.DeepseekV4CudaDecoderLayer._nv_mega_ref)

    def test_on_installs_after_load(self):
        with envs.SGLANG_NV_MEGA_INTREE_REF.override(True):
            model, install = self._precompile()
        install.assert_called_once_with(model)

    def test_layer_without_ref_runs_sglang_path(self):
        layer = _Layer(None)
        batch = _batch(6)
        out = layer.forward(batch)
        self.assertEqual(out, ("layer_out", "sglang_moe"))
        layer._run_boundary_attention.assert_called_once_with(
            "attn_in", "pos", batch, "xq"
        )
        layer.self_attn.forward_before_wo_b.assert_not_called()
        layer.mlp.forward_normal_dual_stream.assert_not_called()


class TestSlots(unittest.TestCase):
    def _dual_stream_kwargs(self, layer):
        layer.mlp.assert_not_called()
        (args, kwargs) = layer.mlp.forward_normal_dual_stream.call_args
        self.assertEqual(args, ("ffn_in", None, "ids"))
        return kwargs

    def test_hop_a_width_publishes_and_fuses_the_post(self):
        runner, moe = _slots()
        layer = _Layer(_ref(runner, moe))
        events = []
        layer.attn_boundary.finish.side_effect = lambda out, fb: (
            events.append("finish") or ("attn_done", out)
        )
        layer.ffn_boundary.prepare.side_effect = lambda *a, **k: (
            events.append("prepare") or "ffn_in"
        )
        runner.check_consumed.side_effect = lambda: events.append("check_consumed")
        batch = _batch(6)
        out = layer.forward(batch)
        self.assertEqual(out, ("layer_out", "slot_moe"))
        layer.self_attn.forward_before_wo_b.assert_called_once_with(
            x="attn_in", positions="pos", forward_batch=batch, x_quant="xq"
        )
        runner.publish_wo_b.assert_called_once_with(layer.self_attn.wo_b, "o")
        layer._run_boundary_attention.assert_not_called()
        finished = layer.attn_boundary.finish.call_args.args[0]
        self.assertIsInstance(finished, mhc.AttnOutput)
        self.assertEqual(finished.partial, "partial")
        self.assertIs(finished.fused_post, runner.hop_a)
        # run_layer's order: the fused post, the FFN prepare, then the consumed check
        self.assertEqual(events, ["finish", "prepare", "check_consumed"])
        self.assertEqual(
            self._dual_stream_kwargs(layer),
            dict(
                input_ids_global="ids_global",
                num_token_non_padded="num_valid",
                use_vision_topk=False,
                return_moe_output=True,
                leave_output_sum=True,
                select_experts=moe.select,
                run_routed_experts=moe.run,
            ),
        )

    def test_hop_a_width_without_a_publish_projects_wo_b(self):
        runner, moe = _slots()
        runner.publish_wo_b.return_value = None
        layer = _Layer(_ref(runner, moe))
        layer.forward(_batch(6))
        layer.self_attn._project_wo_b.assert_called_once_with(
            "o", defer_all_reduce=True
        )
        self.assertEqual(layer.attn_boundary.finish.call_args.args[0], "projected")
        runner.check_consumed.assert_called_once_with()

    def test_wider_widths_keep_sglang_attention(self):
        for rows, gate in ((12, False), (48, True)):
            with self.subTest(rows=rows):
                runner, moe = _slots()
                layer = _Layer(_ref(runner, moe))
                layer.forward(_batch(rows))
                layer.self_attn.forward_before_wo_b.assert_not_called()
                runner.publish_wo_b.assert_not_called()
                runner.check_consumed.assert_not_called()
                layer._run_boundary_attention.assert_called_once()
                self.assertEqual(
                    layer.attn_boundary.finish.call_args.args[0], "sglang_attn"
                )
                kwargs = self._dual_stream_kwargs(layer)
                self.assertIs(kwargs["select_experts"], moe.select if gate else None)
                self.assertIs(kwargs["run_routed_experts"], moe.run)

    def test_fast_gate_needs_the_slot_gate(self):
        runner, moe = _slots()
        moe.gate = False
        layer = _Layer(_ref(runner, moe))
        layer.forward(_batch(48))
        self.assertIsNone(self._dual_stream_kwargs(layer)["select_experts"])

    def test_unrouted_steps_and_shared_local_run_sglang(self):
        cases = (
            (_batch(96), False),  # not a routed width
            (_batch(6, ForwardMode.EXTEND), False),
            (_batch(6), True),  # the transport keeps the shared expert local
        )
        for batch, shared_local in cases:
            with self.subTest(rows=batch.input_ids.shape[0], shared=shared_local):
                runner, moe = _slots()
                layer = _Layer(_ref(runner, moe), do_shared_local=shared_local)
                out = layer.forward(batch)
                self.assertEqual(out, ("layer_out", "sglang_moe"))
                layer.mlp.forward_normal_dual_stream.assert_not_called()
                layer.mlp.assert_called_once_with(
                    "ffn_in",
                    batch,
                    input_ids="ids",
                    input_ids_global="ids_global",
                    skip_shared_experts=shared_local,
                    return_moe_output=True,
                    leave_output_sum=True,
                )

    def test_layer_without_moe_slot_runs_sglang_moe(self):
        runner, _ = _slots()
        layer = _Layer(_ref(runner, None))
        out = layer.forward(_batch(6))
        runner.publish_wo_b.assert_called_once()
        self.assertEqual(out, ("layer_out", "sglang_moe"))
        layer.mlp.forward_normal_dual_stream.assert_not_called()

    def test_draft_stages_take_only_the_moe_slot(self):
        runner, moe = _slots()
        layer = _Layer(_ref(runner, moe, is_draft=True))
        layer.forward(_batch(6))
        runner.publish_wo_b.assert_not_called()
        runner.check_consumed.assert_not_called()
        kwargs = self._dual_stream_kwargs(layer)
        self.assertIsNone(kwargs["select_experts"])
        self.assertIs(kwargs["run_routed_experts"], moe.run)


if __name__ == "__main__":
    unittest.main()
