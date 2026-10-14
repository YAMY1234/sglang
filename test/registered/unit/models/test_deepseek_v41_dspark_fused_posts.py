"""DSpark draft stages on the target's fused all-reduce + mHC posts: dispatch only."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.models import deepseek_v4_dspark as dspark
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.test.test_utils import CustomTestCase

ROWS, HIDDEN = 6, 5120


def _stage(*, attn_gate=True, ffn_gate=True):
    stage = dspark.DSparkV4Stage.__new__(dspark.DSparkV4Stage)
    torch.nn.Module.__init__(stage)
    vars(stage).update(
        hc_cfg=SimpleNamespace(hidden=HIDDEN),
        attn_hc="attn_hc",
        ffn_hc="ffn_hc",
        hc_stats_stream=None,
        local_boundary="ffn_boundary",
        _can_fuse_attn_mhc=attn_gate,
        _can_fuse_ffn_mhc=ffn_gate,
        self_attn=MagicMock(attn_tp_size=4, return_value="attn_out"),
        mlp=SimpleNamespace(tp_size=4),
        _run_moe_ffn_dp_sync=MagicMock(return_value="moe_out"),
        hc_post=MagicMock(return_value=torch.zeros(ROWS, 4, HIDDEN)),
        dim=HIDDEN,
    )
    stage._init_boundaries = lambda: None
    return stage


class TestDraftStageDispatch(CustomTestCase):
    def _run(self, stage, *, flag, rows_fit=True):
        residual = torch.zeros(ROWS, 4, HIDDEN)
        attn_state = mhc.HcState(torch.ones(ROWS, 4, HIDDEN), "attn_pre")
        moe_state = mhc.HcState(torch.full((ROWS, 4, HIDDEN), 2.0), "ffn_pre")
        with (
            envs.SGLANG_OPT_DSV41_DRAFT_FUSED_MHC_ALL_REDUCE.override(flag),
            patch.object(mhc, "use_stats_stream", return_value=False),
            patch.object(mhc, "combine", return_value=torch.zeros(ROWS, HIDDEN)),
            patch.object(mhc, "run_attn_post", return_value=attn_state) as attn_post,
            patch.object(mhc, "run_moe_post", return_value=moe_state) as moe_post,
            patch.object(mhc, "mix_stats", return_value=("pre", "post", "comb")),
            patch.object(dspark, "can_fuse_all_reduce", return_value=rows_fit),
        ):
            out = stage._forward_hc_pre_from_prev(
                torch.zeros(ROWS), residual, MagicMock(), "prev_pre"
            )
        return out, attn_post, moe_post, attn_state, moe_state

    def test_fused_posts_defer_both_reductions(self):
        stage = _stage()
        out, attn_post, moe_post, attn_state, moe_state = self._run(stage, flag=True)
        self.assertTrue(stage.self_attn.call_args.kwargs["defer_all_reduce"])
        self.assertTrue(stage._run_moe_ffn_dp_sync.call_args.kwargs["return_moe_output"])
        kw = attn_post.call_args.kwargs
        self.assertEqual((kw["next"], kw["world_size"]), ("ffn_boundary", 4))
        self.assertEqual(attn_post.call_args.args[1], "attn_out")
        # The MoE post rebuilds the attention post's residual; no seam across stages.
        self.assertIs(moe_post.call_args.args[2], attn_state.residual)
        self.assertIsNone(moe_post.call_args.kwargs["next"])
        self.assertIs(out[0], moe_state.residual)
        self.assertEqual(out[1], "ffn_pre")
        stage.hc_post.assert_not_called()

    def test_closed_gates_reduce_in_the_sublayers(self):
        for gates, rows_fit in (((False, False), True), ((True, True), False)):
            stage = _stage(attn_gate=gates[0], ffn_gate=gates[1])
            _, attn_post, moe_post, _, _ = self._run(stage, flag=True, rows_fit=rows_fit)
            self.assertFalse(stage.self_attn.call_args.kwargs["defer_all_reduce"])
            self.assertFalse(
                stage._run_moe_ffn_dp_sync.call_args.kwargs["return_moe_output"]
            )
            attn_post.assert_called_once()
            moe_post.assert_called_once()

    def test_flag_off_keeps_the_plain_steps(self):
        stage = _stage()
        stage._run_ffn = MagicMock(return_value=torch.zeros(ROWS, HIDDEN))
        _, attn_post, moe_post, _, _ = self._run(stage, flag=False)
        attn_post.assert_not_called()
        moe_post.assert_not_called()
        self.assertNotIn("defer_all_reduce", stage.self_attn.call_args.kwargs)
        self.assertEqual(stage.hc_post.call_count, 2)


class TestDraftWoB(CustomTestCase):
    def _attn(self, *, decode_attn_tp=False):
        attn = dspark.DSparkAttention.__new__(dspark.DSparkAttention)
        torch.nn.Module.__init__(attn)
        wo_b = MagicMock(use_decode_attn_tp=decode_attn_tp)
        wo_b.side_effect = lambda o, **kw: (o + 1, None)
        attn.wo_b = wo_b
        return attn

    def test_deferred_wo_b_hands_the_partial_to_the_post(self):
        attn, o = self._attn(), torch.zeros(ROWS, 8)
        with patch.object(dspark, "should_skip_mlp_all_reduce", return_value=False):
            out = attn._project_wo_b(o, True)
        self.assertIsInstance(out, mhc.AttnOutput)
        self.assertEqual(attn.wo_b.call_args.kwargs, {"skip_all_reduce": True})
        torch.testing.assert_close(out.partial, o + 1)

    def test_wo_b_reduces_when_it_cannot_defer(self):
        for decode_attn_tp, skip_mlp, defer in (
            (True, False, True),
            (False, True, True),
            (False, False, False),
        ):
            attn = self._attn(decode_attn_tp=decode_attn_tp)
            with patch.object(
                dspark, "should_skip_mlp_all_reduce", return_value=skip_mlp
            ):
                out = attn._project_wo_b(torch.zeros(ROWS, 8), defer)
            self.assertIsInstance(out, torch.Tensor)
            self.assertEqual(attn.wo_b.call_args.kwargs, {})


if __name__ == "__main__":
    unittest.main()
