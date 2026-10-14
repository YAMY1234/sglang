"""CPU tests of the DSV4.1 wo_b publish glue (SGLANG_OPT_DSV41_WO_B_PUBLISH): when the
attention's wo_b publishes, and that a published output takes only the consumer."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.models.deepseek_v4 as deepseek_v4
import sglang.srt.models.deepseek_v4_mhc as mhc
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_CFG = mhc.HcConfig(
    mult=4,
    sinkhorn_iters=20,
    eps=1e-6,
    rms_eps=1e-6,
    hidden=8,
    pre_from_prev=True,
    cp_prefill=False,
)


class _Publisher:
    def __init__(self, supported=True, prepared=("a", "scale", (2, 8))):
        self.supported = supported
        self.prepared = prepared
        self.calls = []

    def supports(self, linear, num_tokens):
        return self.supported and 0 < num_tokens <= 8

    def prepare(self, linear, x):
        return self.prepared

    def __call__(self, linear, prepared):
        self.calls.append(prepared)
        return torch.full((2, 8), 3.0)


def _attn(publisher=None):
    return SimpleNamespace(wo_b=SimpleNamespace(), wo_b_publisher=publisher)


class TestPublishingWoB(CustomTestCase):
    def _enter(self, attn, num_tokens=6, enabled=True, publisher=None):
        with patch.object(
            mhc, "_maybe_init_wo_b_publisher", return_value=publisher
        ) as init:
            with mhc.publishing_wo_b(attn, num_tokens, _CFG, top_k=6, enabled=enabled):
                inside = attn.wo_b_publisher
        return inside, init

    def test_flag_off_is_inert(self):
        attn = _attn()
        with envs.SGLANG_OPT_DSV41_WO_B_PUBLISH.override(False):
            inside, init = self._enter(attn, publisher=_Publisher())
        self.assertIsNone(inside)
        init.assert_not_called()

    def test_binds_only_around_a_fused_boundary(self):
        publisher = _Publisher()
        with envs.SGLANG_OPT_DSV41_WO_B_PUBLISH.override(True):
            attn = _attn()
            inside, _ = self._enter(attn, publisher=publisher)
            self.assertIs(inside, publisher)
            self.assertIsNone(attn.wo_b_publisher)
            for kwargs in (
                dict(enabled=False),
                dict(num_tokens=9),
                dict(num_tokens=0),
            ):
                inside, _ = self._enter(_attn(), publisher=publisher, **kwargs)
                self.assertIsNone(inside, kwargs)
            inside, _ = self._enter(_attn(), publisher=None)
            self.assertIsNone(inside)

    def test_unbinds_after_exception(self):
        attn = _attn()
        with (
            envs.SGLANG_OPT_DSV41_WO_B_PUBLISH.override(True),
            patch.object(mhc, "_maybe_init_wo_b_publisher", return_value=_Publisher()),
        ):
            with self.assertRaises(RuntimeError):
                with mhc.publishing_wo_b(attn, 6, _CFG, top_k=6, enabled=True):
                    raise RuntimeError
        self.assertIsNone(attn.wo_b_publisher)

    def test_no_init_during_capture(self):
        with (
            patch.object(mhc, "_wo_b_publisher_tried", False),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
        ):
            self.assertIsNone(mhc._maybe_init_wo_b_publisher(None, _CFG, 6))
            self.assertFalse(mhc._wo_b_publisher_tried)


class TestRunAttnPost(CustomTestCase):
    def _run(self, published):
        rows = 2
        residual = torch.randn(rows, 4, 8)
        triplet = (torch.rand(rows, 4), torch.rand(rows, 4), torch.rand(rows, 4, 4))
        norm = SimpleNamespace(
            weight=torch.nn.Parameter(torch.ones(8)), variance_epsilon=1e-6
        )
        boundary = SimpleNamespace(norm=norm)
        consumer = MagicMock(return_value=(torch.ones(rows, 4, 8), torch.ones(rows, 8)))
        jit = MagicMock(
            return_value=(None, torch.zeros(rows, 4, 8), torch.zeros(rows, 8))
        )
        with (
            patch.object(mhc, "_wo_b_all_reduce", consumer),
            patch(
                "sglang.kernels.ops.communication.all_reduce_mhc."
                "all_reduce_mhc_post_combine_norm",
                jit,
            ),
        ):
            state = mhc.run_attn_post(
                SimpleNamespace(cfg=_CFG),
                mhc.AttnOutput(torch.zeros(rows, 8), published=published),
                residual,
                stats_stream=None,
                next=boundary,
                world_size=4,
                precomputed=triplet,
            )
        return state, consumer, jit, triplet, residual

    def test_published_launches_only_the_consumer(self):
        state, consumer, jit, (pre, post, comb), residual = self._run(True)
        jit.assert_not_called()
        args = consumer.call_args.args
        self.assertIs(args[1], residual)
        self.assertIs(args[2], post)
        self.assertIs(args[3], comb)
        self.assertIs(args[4], pre)
        self.assertFalse(args[5].requires_grad)
        self.assertIs(state.residual, consumer.return_value[0])
        self.assertIs(state.input.rows, consumer.return_value[1])

    def test_unpublished_keeps_the_jit_kernel(self):
        state, consumer, jit, _, _ = self._run(False)
        consumer.assert_not_called()
        jit.assert_called_once()
        self.assertIs(state.residual, jit.return_value[1])


class TestProjectWoB(CustomTestCase):
    def _layer(self, publisher):
        layer = object.__new__(deepseek_v4.MQALayer)
        torch.nn.Module.__init__(layer)
        layer.attn_tp_size = 4
        layer.wo_b = MagicMock(return_value=(torch.ones(2, 8), None))
        layer.wo_b.reduce_results = True
        layer.wo_b.use_decode_attn_tp = False
        layer.wo_b_publisher = publisher
        return layer

    def _project(self, layer, defer=True):
        with (
            patch.object(
                deepseek_v4, "get_parallel", return_value=SimpleNamespace(tp_size=4)
            ),
            patch(
                "sglang.srt.layers.moe.utils.should_skip_mlp_all_reduce",
                return_value=False,
            ),
        ):
            return layer._project_wo_b(torch.zeros(2, 2, 4), defer)

    def test_published(self):
        publisher = _Publisher()
        layer = self._layer(publisher)
        out = self._project(layer)
        self.assertTrue(out.published)
        self.assertEqual(len(publisher.calls), 1)
        layer.wo_b.assert_not_called()

    def test_prepare_declines(self):
        layer = self._layer(_Publisher(prepared=None))
        out = self._project(layer)
        self.assertFalse(out.published)
        layer.wo_b.assert_called_once()
        self.assertTrue(layer.wo_b.call_args.kwargs["skip_all_reduce"])

    def test_not_deferred_never_publishes(self):
        publisher = _Publisher()
        layer = self._layer(publisher)
        out = self._project(layer, defer=False)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(publisher.calls, [])


if __name__ == "__main__":
    unittest.main()
