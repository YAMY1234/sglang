"""mHC consumer ownership, replicated shared add, and source row contracts."""

import unittest
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
import torch

from sglang.srt.layers.layer_boundary.layout import Layout, SumGroup
from sglang.srt.layers.layer_boundary.row_transport import TransportRows
from sglang.srt.models import deepseek_v4_boundary as boundary
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.models.deepseek_common import moe_transport
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHcConsumer(unittest.TestCase):
    def test_dspark_tap_hands_materialized_stream_to_next_layer(self):
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models import deepseek_v4

        class Layer(torch.nn.Module):
            engram = None
            boundary_mhc_fusions = False
            boundary_pre_from_prev = True
            hc_cfg = mhc.HcConfig(2, 4, 1e-5, 1e-6, 4, True, False)

            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, hidden_states, forward_batch, **kwargs):
                stream = residual_batch.stream_of(forward_batch)
                stream.check(hidden_states)
                update = boundary.HcUpdate(self, False)
                update.post = lambda value, state, **kw: mhc.HcState(
                    state.residual + value, torch.ones(3, 2)
                )
                return stream.record(torch.full((3, 2, 4), self.value), update)

        model = deepseek_v4.DeepseekV4Model.__new__(deepseek_v4.DeepseekV4Model)
        torch.nn.Module.__init__(model)
        model.layers = torch.nn.ModuleList([Layer(1.0), Layer(2.0)])
        model.start_layer, model.end_layer = 0, 2
        model.pp_group = SimpleNamespace(world_size=1)
        model.engram_hasher = model.late_layer_start = None
        model.dspark_layers_to_capture = [1]
        positions = torch.arange(3)
        batch = SimpleNamespace(
            positions=positions,
            input_ids=positions,
            forward_mode=ForwardMode.EXTEND,
            residual_stream=None,
        )
        source = torch.arange(24.0).reshape(3, 2, 4)
        aux = []
        recorder = SimpleNamespace(with_current_layer=lambda _: nullcontext())
        with (
            patch.object(deepseek_v4, "_is_cuda", True),
            patch.object(deepseek_v4, "_is_hip", False),
            patch.object(deepseek_v4, "is_cp_active", return_value=False),
            patch.object(deepseek_v4, "check_cuda_graph_backend", return_value=False),
            patch.object(
                deepseek_v4,
                "get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
        ):
            output, pre, tail = model._forward_layers_hc_pre_from_prev(
                positions, source, batch, positions, positions, True, aux
            )
        torch.testing.assert_close(output, source + 3)
        torch.testing.assert_close(aux[0], (source + 1).mean(1))
        torch.testing.assert_close(pre, torch.ones(3, 2))
        self.assertIsNone(tail)
        self.assertIsNone(batch.residual_stream)

    def test_replicated_shared_is_added_after_the_sum(self):
        routed = torch.tensor([[1.0, 1.0]], dtype=torch.bfloat16)
        other = torch.tensor([[-1.0, -0.5]], dtype=torch.bfloat16)
        shared = torch.tensor([[0.01, 0.003]], dtype=torch.bfloat16)
        pieces = SimpleNamespace(
            get_merged=lambda: routed, shared=shared, shared_is_replicated=True
        )
        with (
            patch(
                "sglang.srt.layers.moe.post_experts_all_reduce",
                side_effect=lambda x: x + other,
            ) as reduce,
            patch(
                "sglang.srt.layers.moe.utils.should_add_replicated_moe_output",
                return_value=True,
            ),
        ):
            output = boundary.MoeFinalize(pieces).complete()
        reduce.assert_called_once_with(routed)
        torch.testing.assert_close(output, (routed + other) + shared, atol=0, rtol=0)
        self.assertFalse(torch.equal(output, (routed + shared) + other))


class TestAttentionCompletion(unittest.TestCase):
    def test_projection_and_deferred_sum_preserve_full_precision_policy(self):
        from sglang.srt.layers import linear
        from sglang.srt.layers.moe import utils as moe_utils
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models import deepseek_v4
        from sglang.srt.runtime_context import get_forward

        group = SimpleNamespace(all_reduce=Mock(side_effect=lambda value: value * 2))
        parallel = SimpleNamespace(tp_size=4, tp_group=group, attn_tp_group=group)

        class Projection:
            tp_group = group
            input_is_parallel = True
            bias = None
            skip_bias_add = False
            reduce_results = False
            use_decode_attn_tp = False
            use_dp_attention_reduce = False
            parallel_group = "attn_tp"
            quant_method = SimpleNamespace(apply=lambda layer, value, bias: value + 1)

            def __call__(self, value, **kwargs):
                return linear.RowParallelLinear.forward(self, value, **kwargs)

        attention = deepseek_v4.MQALayer.__new__(deepseek_v4.MQALayer)
        torch.nn.Module.__init__(attention)
        attention.attn_tp_size = 2
        attention.wo_b = Projection()
        layer = SimpleNamespace(boundary_mhc_fusions=False)
        value = torch.arange(12.0).view(3, 4)
        rows = TransportRows(
            Layout(frozenset()),
            Layout(frozenset()),
            Layout(frozenset()),
            SumGroup.ATTN_TP,
        )
        for quant in (False, True):
            for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
                batch = SimpleNamespace(forward_mode=mode)
                with (
                    self.subTest(quant=quant, mode=mode),
                    get_forward().scoped(sp_active=False),
                    patch.object(linear, "get_group_rank_size", return_value=(0, 2)),
                    patch.object(linear, "get_parallel", return_value=parallel),
                    patch.object(
                        linear,
                        "get_exec",
                        return_value=SimpleNamespace(
                            comm=SimpleNamespace(enable_quant_communications=quant)
                        ),
                    ),
                    patch.object(
                        linear,
                        "use_symmetric_memory",
                        side_effect=lambda *a, **k: nullcontext(),
                    ),
                    patch.object(linear, "is_allocation_symmetric", return_value=False),
                    patch.object(
                        linear, "should_skip_mlp_all_reduce", return_value=False
                    ),
                    patch.object(
                        linear,
                        "tensor_model_parallel_all_reduce",
                        side_effect=group.all_reduce,
                    ),
                    patch.object(
                        linear,
                        "tensor_model_parallel_quant_all_reduce",
                        side_effect=AssertionError("unexpected quantized sum"),
                    ),
                    patch.object(
                        moe_utils, "should_skip_mlp_all_reduce", return_value=False
                    ),
                    patch.object(deepseek_v4, "get_parallel", return_value=parallel),
                    patch.object(boundary, "_sum_group", return_value=group),
                    patch.object(
                        deepseek_v4, "attn_tp_all_reduce", side_effect=group.all_reduce
                    ),
                    patch.object(
                        boundary,
                        "get_cp_decode_attn_tp_ctx",
                        return_value=SimpleNamespace(use_decode_attn_tp=False),
                    ),
                ):
                    # The production projection never passes forward_batch to
                    # RowParallelLinear, so this flag does not quantize its sum.
                    with patch.object(deepseek_v4, "_is_cuda", False):
                        immediate = attention._project_wo_b(
                            value, defer_all_reduce=True
                        )
                    with patch.object(deepseek_v4, "_is_cuda", True):
                        deferred = attention._project_wo_b(value, defer_all_reduce=True)
                    transport = boundary.AttentionTransport(
                        layer, value, batch, None, rows
                    )
                    completed = transport.prepare_output(deferred).complete()
                torch.testing.assert_close(completed, immediate, atol=0, rtol=0)
                torch.testing.assert_close(completed, (value + 1) * 2, atol=0, rtol=0)


class TestAttentionWoBSplit(unittest.TestCase):
    def test_forward_is_wo_b_of_forward_before_wo_b(self):
        """A caller that runs wo_b itself must get the attention output forward projects."""
        from sglang.srt.models import deepseek_v4

        attention = deepseek_v4.MQALayer.__new__(deepseek_v4.MQALayer)
        torch.nn.Module.__init__(attention)
        before = torch.randn(3, 4)
        attention.forward_before_wo_b = Mock(return_value=before)
        attention._project_wo_b = Mock(side_effect=lambda o, defer: (o, defer))
        context = SimpleNamespace(input_scattered=False)
        with patch.object(deepseek_v4, "get_attn_tp_context", return_value=context):
            out = deepseek_v4.MQALayer.forward(
                attention,
                torch.ones(3, 8),
                None,
                None,
                x_quant="q",
                defer_all_reduce=True,
            )
            self.assertEqual(out, (before, True))
            self.assertEqual(
                attention.forward_before_wo_b.call_args.kwargs, {"x_quant": "q"}
            )

            empty = torch.ones(0, 8)
            self.assertIs(
                deepseek_v4.MQALayer.forward(attention, empty, None, None), empty
            )
        attention.forward_before_wo_b.assert_called_once()
        attention._project_wo_b.assert_called_once()


class TestAttentionFusedPost(unittest.TestCase):
    def test_fused_post_crosses_the_transport_and_replaces_the_collective(self):
        """A fused post handed in with the attention partial must run in place of
        the collective kernel; losing it on the way silently runs the stock sum."""
        rows = TransportRows(
            Layout(frozenset()),
            Layout(frozenset()),
            Layout(frozenset()),
            SumGroup.ATTN_TP,
        )
        partial, residual = torch.ones(3, 4), torch.zeros(3, 2, 4)
        fused_post = Mock(return_value=(residual + 1, partial + 2))
        group = object()
        with (
            patch.object(boundary, "_sum_group", return_value=group),
            patch.object(
                boundary,
                "get_cp_decode_attn_tp_ctx",
                return_value=SimpleNamespace(use_decode_attn_tp=False),
            ),
        ):
            transport = boundary.AttentionTransport(
                SimpleNamespace(boundary_mhc_fusions=False),
                partial,
                SimpleNamespace(),
                None,
                rows,
            )
            owed = transport.prepare_output(mhc.AttnOutput(partial, fused_post))
        self.assertIs(owed.group, group)
        self.assertIs(owed.fused_post, fused_post)

        norm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
        triplet = (torch.ones(3, 2), torch.ones(3, 2), torch.ones(3, 2, 2))
        state = mhc.run_attn_post(
            None,
            mhc.AttnOutput(owed.partial, owed.fused_post),
            residual,
            stats_stream=None,
            next=SimpleNamespace(norm=norm),
            world_size=4,
            precomputed=triplet,
        )
        pre, post_mix, comb = triplet
        args = fused_post.call_args.args
        self.assertEqual(len(args), 7)
        for got, want in zip(
            args, (partial, residual, post_mix, comb, pre, norm.weight)
        ):
            self.assertIs(got, want)
        self.assertEqual(args[6], norm.variance_epsilon)
        torch.testing.assert_close(state.streams, residual + 1)


class TestMoeFusedPost(unittest.TestCase):
    @staticmethod
    def _moe_output(rows, fused_post):
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            FlashInferTrtllmDeferredFinalizeOutput,
        )
        from sglang.srt.models.deepseek_v2 import MoEOutput

        routed = FlashInferTrtllmDeferredFinalizeOutput(
            torch.ones(rows * 6, 4),
            torch.ones(rows, 6),
            torch.zeros(rows, 6, dtype=torch.int32),
            6,
            fused_post=fused_post,
        )
        return MoEOutput(routed, torch.ones(rows, 4), None, 1.0, False)

    @staticmethod
    def _triplet(rows):
        return (torch.ones(rows, 2), torch.ones(rows, 2), torch.ones(rows, 2, 2))

    def test_fused_post_runs_at_any_width_and_builds_the_next_input(self):
        """A fused post on the deferred output must replace the collective kernel at
        widths the push plane cannot carry, with or without a next boundary."""
        rows = 512
        residual = torch.zeros(rows, 2, 4)
        next_input = object()
        fused_post = Mock(return_value=(residual + 1, next_input))
        out = self._moe_output(rows, fused_post)
        triplet = self._triplet(rows)
        stock = SimpleNamespace(moe_finalize_all_reduce_mhc_post=Mock())
        for nxt in (None, SimpleNamespace(accepts_mxfp8=True, norm_fusable=True)):
            fused_post.reset_mock()
            with patch.dict(
                "sys.modules",
                {"sglang.kernels.ops.communication.all_reduce_mhc": stock},
            ):
                state = mhc.run_moe_post(
                    None,
                    out,
                    residual,
                    stats_stream=None,
                    next=nxt,
                    world_size=4,
                    precomputed=triplet,
                )
            r = out.routed
            pre, post_mix, comb = triplet
            want = (r.gemm2_out, r.expanded_idx_to_permuted_idx, r.expert_weights)
            want += (r.top_k, out.shared, residual, post_mix, comb, pre, nxt)
            args = fused_post.call_args.args
            self.assertEqual(len(args), len(want))
            for got, expected in zip(args, want):
                self.assertIs(got, expected)
            torch.testing.assert_close(state.streams, residual + 1)
            self.assertIs(state.pre, pre)
            self.assertIs(state.input, next_input)
        stock.moe_finalize_all_reduce_mhc_post.assert_not_called()

    def test_unset_fused_post_runs_the_collective_kernel(self):
        from sglang.srt.layers.quantization import mxfp4_flashinfer_trtllm_moe

        rows = 3
        residual = torch.zeros(rows, 2, 4)
        out = self._moe_output(rows, None)
        triplet = self._triplet(rows)
        stock = SimpleNamespace(
            moe_finalize_all_reduce_mhc_post=Mock(return_value=(None, residual + 2))
        )
        with (
            patch.dict(
                "sys.modules",
                {"sglang.kernels.ops.communication.all_reduce_mhc": stock},
            ),
            patch.object(
                mxfp4_flashinfer_trtllm_moe, "can_fuse_all_reduce", return_value=True
            ),
        ):
            state = mhc.run_moe_post(
                SimpleNamespace(cfg=SimpleNamespace(hidden=4)),
                out,
                residual,
                stats_stream=None,
                next=None,
                world_size=4,
                precomputed=triplet,
            )
        r = out.routed
        pre, post_mix, comb = triplet
        want = (r.gemm2_out, r.expanded_idx_to_permuted_idx, r.expert_weights)
        want += (r.top_k, out.shared, residual, post_mix, comb)
        call = stock.moe_finalize_all_reduce_mhc_post.call_args
        self.assertEqual(len(call.args), len(want))
        for got, expected in zip(call.args, want):
            self.assertIs(got, expected)
        self.assertEqual(call.kwargs, {"world_size": 4})
        torch.testing.assert_close(state.streams, residual + 2)
        self.assertIsNone(state.input)


class TestSumDeferral(unittest.TestCase):
    @staticmethod
    def _defers(call, num_rows):
        layer = SimpleNamespace(
            boundary_mhc_fusions=True,
            boundary_pre_from_prev=True,
            _can_fuse_attn_mhc=True,
            _can_fuse_ffn_mhc=True,
            hc_cfg=SimpleNamespace(hidden=4),
            mlp=SimpleNamespace(),
            config=SimpleNamespace(model_type="deepseek_v41"),
        )
        plain = (Layout(frozenset()),) * 3
        hidden = torch.ones(num_rows, 4)
        backend = SimpleNamespace(is_none=lambda: True)
        with (
            patch.object(moe_transport, "get_moe_a2a_backend", return_value=backend),
            patch.object(boundary, "get_moe_a2a_backend", return_value=backend),
        ):
            attn = boundary.AttentionTransport(
                layer,
                hidden,
                SimpleNamespace(),
                call,
                TransportRows(*plain, SumGroup.ATTN_TP),
            )
            moe = boundary.MoeTransport(
                layer,
                hidden,
                SimpleNamespace(dp_padding_mode=None),
                call,
                TransportRows(*plain, SumGroup.MOE_OUTPUT),
            )
        return attn.defers_sum, moe.defers_sum

    def test_call_predicate_decides_both_sublayers_alone(self):
        """A call's predicate must decide the deferral of both sums in both
        directions; the push plane's width check must not veto or force it."""
        predicate = Mock(side_effect=lambda rows, hidden: rows == 300)
        call = boundary.LayerCall(can_fuse_sum=predicate)
        with patch.object(boundary, "can_fuse_all_reduce", return_value=False) as stock:
            self.assertEqual(self._defers(call, 300), (True, True))
            self.assertEqual(predicate.call_args_list[-2:], [((300, 4),)] * 2)
            stock.return_value = True
            self.assertEqual(self._defers(call, 40), (False, False))
        stock.assert_not_called()

    def test_unset_predicate_keeps_the_push_plane_check(self):
        for fits in (True, False):
            with patch.object(
                boundary, "can_fuse_all_reduce", return_value=fits
            ) as stock:
                self.assertEqual(self._defers(boundary.LayerCall(), 300), (fits, fits))
            self.assertEqual(stock.call_args_list, [((300, 4),)] * 2)


class TestMoeSourceRows(unittest.TestCase):
    def test_cp_operand_is_released_before_return_and_v41_uses_contiguous_rows(self):
        from sglang.srt.layers.layer_boundary.layout import TokenAxis

        layer = SimpleNamespace(boundary_mhc_fusions=False, mlp=SimpleNamespace())
        rows = TransportRows(
            Layout.sharded_over(TokenAxis.ATTN_CP, axis_sizes={TokenAxis.ATTN_CP: 2}),
            Layout.sharded_over(axis_sizes={TokenAxis.ATTN_CP: 2}),
            Layout.sharded_over(TokenAxis.ATTN_CP, axis_sizes={TokenAxis.ATTN_CP: 2}),
            SumGroup.MOE_OUTPUT,
        )
        backend = SimpleNamespace(is_none=lambda: True)
        group = SimpleNamespace(all_reduce=lambda value: value * 2)
        parallel = SimpleNamespace(attn_cp_size=2, attn_cp_rank=1, tp_group=group)
        batch = SimpleNamespace(dp_padding_mode=None)
        with (
            patch.object(moe_transport, "get_moe_a2a_backend", return_value=backend),
            patch.object(boundary, "get_moe_a2a_backend", return_value=backend),
            patch.object(moe_transport, "get_parallel", return_value=parallel),
            patch.object(
                moe_transport,
                "attn_cp_interleave_gather",
                side_effect=lambda value: torch.cat((value, value)),
            ),
            patch.object(
                moe_transport,
                "attn_cp_interleave_reduce_scatter",
                side_effect=lambda value: value[1::2] * 2,
            ) as interleave,
        ):
            for kind in ("deepseek_v4", "deepseek_v41"):
                layer.config = SimpleNamespace(model_type=kind)
                call = boundary.MoeTransport(
                    layer, torch.ones(2, 3), batch, boundary.LayerCall(), rows
                )
                reference = weakref.ref(call.input)
                value = torch.arange(12.0).reshape(4, 3)
                output = call.prepare_output(value)
                self.assertIsNone(reference())
                result = call.return_output(output)
                expected = value[1::2] if kind == "deepseek_v4" else value[2:]
                torch.testing.assert_close(result, expected * 2)
                call.release()
        interleave.assert_called_once()


class TestLayerCallMhcOverrides(unittest.TestCase):
    """A call's ``mhc_fusions`` / ``accepts_mxfp8`` replace the layer's own at
    each read; unset, every read sees the layer's values as before."""

    def _layer(self):
        attn = SimpleNamespace(accepts_mxfp8_swizzled_input=Mock(return_value=False))
        return SimpleNamespace(
            boundary_mhc_fusions=False,
            boundary_pre_from_prev=True,
            _can_fuse_attn_mhc=True,
            _can_fuse_ffn_mhc=True,
            hc_cfg=SimpleNamespace(hidden=4),
            engram=None,
            attn_hc="attn_hc",
            input_layernorm="input_norm",
            self_attn=attn,
            mlp=SimpleNamespace(),
            config=SimpleNamespace(model_type="deepseek_v41"),
        )

    def test_transports_defer_their_sum_by_the_call(self):
        rows = TransportRows(
            Layout(frozenset()),
            Layout(frozenset()),
            Layout(frozenset()),
            SumGroup.ATTN_TP,
        )
        moe_rows = msgspec.structs.replace(rows, group=SumGroup.MOE_OUTPUT)
        backend = SimpleNamespace(is_none=lambda: True)
        layer = self._layer()
        cases = (
            (None, False),
            (boundary.LayerCall(), False),
            (boundary.LayerCall(mhc_fusions=True), True),
        )
        with (
            patch.object(boundary, "can_fuse_all_reduce", return_value=True),
            patch.object(boundary, "get_moe_a2a_backend", return_value=backend),
            patch.object(moe_transport, "get_moe_a2a_backend", return_value=backend),
        ):
            for call, defers in cases:
                attn = boundary.AttentionTransport(
                    layer, torch.ones(2, 4), SimpleNamespace(), call, rows
                )
                moe = boundary.MoeTransport(
                    layer,
                    torch.ones(2, 4),
                    SimpleNamespace(dp_padding_mode=None),
                    call or boundary.LayerCall(),
                    moe_rows,
                )
                self.assertIs(attn.defers_sum, defers)
                self.assertIs(moe.defers_sum, defers)
            layer.boundary_mhc_fusions = True
            call = boundary.LayerCall(mhc_fusions=False)
            attn = boundary.AttentionTransport(
                layer, torch.ones(2, 4), SimpleNamespace(), call, rows
            )
            self.assertFalse(attn.defers_sum)

    def test_read_takes_the_call_fusions_and_mxfp8_input(self):
        layer = self._layer()
        spec = boundary.HcRead(layer, True)
        batch = SimpleNamespace()

        unset = spec.bind(batch, boundary.LayerCall())
        self.assertIsNone(unset.next_boundary)

        call = boundary.LayerCall(mhc_fusions=True, accepts_mxfp8=True)
        with patch.object(mhc, "make_boundary", return_value="next") as make:
            self.assertEqual(spec.bind(batch, call).next_boundary, "next")
        make.assert_called_once_with("input_norm", accepts_mxfp8=True, hc="attn_hc")
        layer.self_attn.accepts_mxfp8_swizzled_input.assert_not_called()

        def combine(hc, residual, quantized):
            if quantized is not None:
                quantized.append("x_quant")
            return "x"

        with (
            patch.object(mhc, "use_stats_stream", return_value=False),
            patch.object(mhc, "combine", side_effect=combine),
        ):
            for read_call, quantized in (
                (boundary.LayerCall(), None),
                (boundary.LayerCall(accepts_mxfp8=True), "x_quant"),
            ):
                residual = mhc.HcState(torch.ones(2, 2, 4))
                x, state = spec.bind(batch, read_call).read(residual, None)
                self.assertEqual(x, "x")
                self.assertEqual(state.quantized, quantized)
        layer.self_attn.accepts_mxfp8_swizzled_input.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
